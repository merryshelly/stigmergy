"""Tests for stigmergy.cli's `daemon run --rig <name>` subcommand and its
collaborator-wiring helpers (bead .27 build spec §2, §2.1; frozen case list
§4 cases 13-21).

`_run_daemon` (which loops forever) is ALWAYS monkeypatched to a recording
stub — the real one is never called in a test. Rig fixtures go through the
REAL create_rig (real local git clone) so `repo/prompts/{code01,critic01}`
exist for `_build_daemon`'s `Critic.from_prompt_file` and `derive_steering`.
No real podman / network / provider calls anywhere.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

import stigmergy.cli as cli
from stigmergy.cli import (
    _DEFAULT_PROTECTED_PATHS,
    _build_daemon,
    _make_steering_of,
    main,
)
from stigmergy.container import PodmanContainerReaper
from stigmergy.critic import Critic, CriticInfraError
from stigmergy.daemon import Daemon
from stigmergy.rangereport import RangeCritic
from stigmergy.records import RecordPlane
from stigmergy.rig import create_rig, resolve_rig
from stigmergy.steering import derive_steering

FIXTURES = Path(__file__).parent / "fixtures"
VALID_CHARTER_PATH = FIXTURES / "charter_valid.toml"
MODELS_REGISTRY_PATH = FIXTURES / "models.toml"
BASE_CHARTER_TOML = VALID_CHARTER_PATH.read_text()


# ==========================================================================
# fixtures (real git clone; matches test_rig.py's own helper style)
# ==========================================================================


def make_local_repo_with_prompts(tmp_path: Path, name: str = "source_repo") -> Path:
    repo_dir = tmp_path / name
    (repo_dir / "prompts").mkdir(parents=True)
    (repo_dir / "prompts" / "code01").write_text("code01 template: $goal\n")
    (repo_dir / "prompts" / "critic01").write_text("critic01 template\n")
    (repo_dir / "README.md").write_text("hello from the fixture repo\n")
    env_cfg = ["-c", "user.email=test@example.com", "-c", "user.name=Test User"]
    subprocess.run(["git", "init", "-q", str(repo_dir)], check=True)
    subprocess.run(["git", *env_cfg, "-C", str(repo_dir), "add", "-A"], check=True)
    subprocess.run(
        ["git", *env_cfg, "-C", str(repo_dir), "commit", "-q", "-m", "initial commit"],
        check=True,
    )
    return repo_dir


def make_charter(tmp_path: Path, repo: Path | str) -> Path:
    charter_dir = tmp_path / "charter_src"
    charter_dir.mkdir(exist_ok=True)
    text = BASE_CHARTER_TOML.replace('repo = "path-or-url"', f'repo = "{repo}"')
    charter_path = charter_dir / "charter.toml"
    charter_path.write_text(text)
    shutil.copy(MODELS_REGISTRY_PATH, charter_dir / "models.toml")
    return charter_path


def scaffold_rig(tmp_path: Path, rigs_root: Path | None = None) -> Path:
    """Scaffold a real rig named 'shipyard' and return its rigs_root."""
    repo = make_local_repo_with_prompts(tmp_path)
    charter_path = make_charter(tmp_path, repo)
    if rigs_root is None:
        rigs_root = tmp_path / "rigs"
    create_rig(charter_path, base_dir=rigs_root)
    return rigs_root


# --- case 13: `daemon run` builds and hands off a real Daemon ---------------


def test_main_daemon_run_builds_real_daemon(tmp_path: Path, monkeypatch) -> None:
    rigs_root = scaffold_rig(tmp_path)
    recorded: list[Daemon] = []
    monkeypatch.setattr(cli, "_run_daemon", recorded.append)

    rc = main(["daemon", "run", "--rig", "shipyard", "--rigs-root", str(rigs_root)])

    assert rc == 0
    assert len(recorded) == 1
    daemon = recorded[0]
    try:
        assert isinstance(daemon, Daemon)
        # Real, non-placeholder collaborators where it matters:
        assert isinstance(daemon._container_reaper, PodmanContainerReaper)
        # The critic is wired with the REAL provider-calling client (bead .36),
        # NOT a silently-wrong stub. It's a make_critic_client closure; its own
        # request/response + fail-closed behavior is covered in
        # test_critic_client.py. Daemon construction must NOT shell out to
        # 1Password (make_op_key_provider is lazy) -- reaching this assertion
        # here without any op/network access is itself that regression guard.
        assert callable(daemon._weaver.critic._client)
        assert daemon._weaver.critic.model == "opus"  # charter [roles.critic].model
    finally:
        daemon._store.close()


# --- case 14: unknown rig -> exit 1, stderr, loop NEVER entered -------------


def test_main_daemon_run_unknown_rig_returns_1(tmp_path: Path, monkeypatch, capsys) -> None:
    called: list[Daemon] = []
    monkeypatch.setattr(cli, "_run_daemon", called.append)

    rc = main(["daemon", "run", "--rig", "ghost", "--rigs-root", str(tmp_path)])

    assert rc == 1
    assert called == []  # regression guard: construction failed, never looped
    err = capsys.readouterr().err
    assert "ghost" in err


# --- case 15: --rigs-root is threaded through -------------------------------


def test_main_daemon_run_rigs_root_override(tmp_path: Path, monkeypatch) -> None:
    custom_root = tmp_path / "custom_rigs_location"
    scaffold_rig(tmp_path, rigs_root=custom_root)
    recorded: list[Daemon] = []
    monkeypatch.setattr(cli, "_run_daemon", recorded.append)

    # The rig exists ONLY under the non-default custom root.
    rc = main(["daemon", "run", "--rig", "shipyard", "--rigs-root", str(custom_root)])

    assert rc == 0
    assert len(recorded) == 1
    recorded[0]._store.close()


# --- case 16: _build_daemon wiring points -----------------------------------


def test_build_daemon_wiring(tmp_path: Path) -> None:
    rigs_root = scaffold_rig(tmp_path)
    resolved = resolve_rig("shipyard", rigs_root=rigs_root)
    # _build_daemon takes ownership of resolved.store via the Daemon.
    daemon = _build_daemon(resolved)
    try:
        # checker_image == charter [rig].image (v0: one image for worker+checker)
        assert daemon._checker_image == "stigmergy-worker:py312"

        # spend leash budgets: charter values + reserve_usd v0 default of 0.0
        budgets = daemon._spend_leash._budgets
        assert budgets.reserve_usd == 0.0
        assert budgets.dispatches == 50
        assert budgets.usd == 25.0
        assert budgets.gate_calls == 30

        # weaver protected paths (fixed tuple, incl. prompts/) + journal path
        assert daemon._weaver.protected_paths == list(_DEFAULT_PROTECTED_PATHS)
        assert "prompts/" in _DEFAULT_PROTECTED_PATHS
        assert (
            daemon._weaver.journal_path
            == resolved.rig_paths["records_dir"] / "weave_journal.jsonl"
        )

        # notifier topic from charter [notify].ntfy_topic
        assert daemon._notifier.topic == "stigmergy"
    finally:
        daemon._store.close()


# --- case 17: _make_steering_of == derive_steering (no silent divergence) ---


def test_make_steering_of_matches_derive_steering(tmp_path: Path) -> None:
    rigs_root = scaffold_rig(tmp_path)
    resolved = resolve_rig("shipyard", rigs_root=rigs_root)
    try:
        resolved.store.add_ticket(
            id="t-1",
            title="Do the thing",
            goal="ship it",
            required_reading=[],
            target_scope=["src/foo.py"],
            acceptance_criteria=["works"],
            tier1_checks=[],
        )
        prompts_dir = resolved.rig_paths["prompts_dir"]
        steering_of = _make_steering_of(resolved.store, resolved.charter, prompts_dir)

        via_closure = steering_of("t-1")
        direct = derive_steering(
            resolved.store.get_ticket("t-1"), resolved.charter, prompts_dir
        )
        assert via_closure == direct
    finally:
        resolved.store.close()


# --- case 18: any raising critic client surfaces as CriticInfraError ---------


def test_raising_critic_client_surfaces_as_infra_error() -> None:
    """A critic client that raises must surface as a critic-INFRA trip
    (non-crashing), never a raw exception and never a silent gate verdict —
    proved against the REAL Critic.judge(). (bead .36 removed the old
    _unwired_critic_client placeholder; the real provider-calling client's own
    fail-closed paths — transport error, malformed/missing verdict — are
    covered in test_critic_client.py. This preserves the invariant that ANY
    client exception is infra, independent of which client is wired.)"""

    def raising_client(prompt: str, *, model: str, **decoding_params: object) -> object:
        raise RuntimeError("provider unreachable")

    critic = Critic(
        client=raising_client,
        model="opus",
        decoding_params={"temperature": 0.0},
        template="judge the artifact",
    )
    with pytest.raises(CriticInfraError):
        critic.judge("some artifact", ["some rubric item"])


# --- case 20: bare invocation still prints usage + returns 0 ----------------


def test_main_bare_invocation_prints_usage(capsys) -> None:
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "usage" in out.lower()


# --- case 21: the new subcommand is discoverable in usage -------------------


def test_usage_mentions_daemon_run(capsys) -> None:
    main([])
    out = capsys.readouterr().out
    assert "daemon run --rig" in out


# ==========================================================================
# Bead .42 — triage CLI (approve/unapprove/reject/promote) + folded .33
# monitoring CLI (status/tickets/ticket/range-report) + REPORT event.
# ==========================================================================

RIG = "shipyard"


def _seed_ticket(rigs_root: Path, ticket_id: str = "t-1", **fields) -> None:
    resolved = resolve_rig(RIG, rigs_root=rigs_root)
    try:
        base = {
            "title": "Do the thing",
            "goal": "ship it",
            "required_reading": [],
            "target_scope": ["src/foo.py"],
            "acceptance_criteria": ["works"],
            "tier1_checks": {"pytest": "pytest -q"},
            "functional_summary": "Operator-facing: the thing works now.",
            "lane_hint": None,
        }
        base.update(fields)
        resolved.store.add_ticket(id=ticket_id, **base)
    finally:
        resolved.store.close()


def _seed_filed(rigs_root: Path, filed_id: str = "filed-dispatch-1-1") -> None:
    resolved = resolve_rig(RIG, rigs_root=rigs_root)
    try:
        resolved.store.add_filed_ticket(
            id=filed_id,
            title="proposal",
            description="a discovered proposal",
            origin_role="worker",
            origin_worker="worker-1",
            origin_dispatch_id="dispatch-1",
            origin_parent_ticket="workspace-e2uh.8",
            discovered_from="dispatch-1@workspace-e2uh.8",
            proposal_hash="proposalhash",
        )
    finally:
        resolved.store.close()


def _records_events(rigs_root: Path, event_type: str) -> list[dict]:
    resolved = resolve_rig(RIG, rigs_root=rigs_root)
    try:
        plane = RecordPlane(resolved.rig_paths["records_dir"])
        return [e for e in plane.read_events() if e.get("event_type") == event_type]
    finally:
        resolved.store.close()


def _ticket(rigs_root: Path, ticket_id: str) -> dict:
    resolved = resolve_rig(RIG, rigs_root=rigs_root)
    try:
        return resolved.store.get_ticket(ticket_id)
    finally:
        resolved.store.close()


def _make_staging(rigs_root: Path) -> None:
    """Create a `staging` branch in the rig's repo clone so range-report has a
    ref to compute against."""
    repo = rigs_root / RIG / "repo"
    subprocess.run(["git", "-C", str(repo), "branch", "staging", "HEAD"], check=True)


_ATTRIB = ["--agent", "merry", "--operator-session", "sess-42"]


# --- C1: approve -----------------------------------------------------------


def test_C1_approve_sets_hash_and_logs_attribution(tmp_path: Path, capsys) -> None:
    rigs_root = scaffold_rig(tmp_path)
    _seed_ticket(rigs_root, "t-1")

    rc = main(["approve", "t-1", "--rig", RIG, "--rigs-root", str(rigs_root), *_ATTRIB])
    assert rc == 0
    out = capsys.readouterr().out

    ticket = _ticket(rigs_root, "t-1")
    assert ticket["approved"] == 1
    assert ticket["approval_hash"]
    assert ticket["approval_hash"] in out  # spec §C: approve prints the resulting hash

    events = _records_events(rigs_root, "approval")
    assert len(events) == 1
    assert events[0]["subject_id"] == "t-1"
    assert events[0]["acting_agent"] == "merry"
    assert events[0]["operator_session"] == "sess-42"
    assert events[0]["approval_hash"] == ticket["approval_hash"]


def test_C1_approve_json_prints_steering_dict(tmp_path: Path, capsys) -> None:
    rigs_root = scaffold_rig(tmp_path)
    _seed_ticket(rigs_root, "t-1")
    rc = main(["approve", "t-1", "--rig", RIG, "--rigs-root", str(rigs_root), *_ATTRIB, "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    # a parseable JSON object containing the steering set is somewhere in stdout
    payload = None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("{"):
            payload = json.loads(line)
            break
    assert payload is not None
    assert "functional_summary" in payload


# --- C2: approve missing ticket -------------------------------------------


def test_C2_approve_missing_ticket_returns_1(tmp_path: Path, capsys) -> None:
    rigs_root = scaffold_rig(tmp_path)
    rc = main(["approve", "ghost", "--rig", RIG, "--rigs-root", str(rigs_root), *_ATTRIB])
    assert rc == 1
    assert _records_events(rigs_root, "approval") == []


# --- C3: approve missing attribution --------------------------------------


def test_C3_approve_missing_attribution_is_argparse_error(tmp_path: Path) -> None:
    rigs_root = scaffold_rig(tmp_path)
    _seed_ticket(rigs_root, "t-1")
    with pytest.raises(SystemExit) as exc:
        main(["approve", "t-1", "--rig", RIG, "--rigs-root", str(rigs_root)])
    assert exc.value.code == 2
    assert _ticket(rigs_root, "t-1")["approved"] == 0
    assert _records_events(rigs_root, "approval") == []


# --- C4: unapprove ---------------------------------------------------------


def test_C4_unapprove_clears_approval_and_logs(tmp_path: Path) -> None:
    rigs_root = scaffold_rig(tmp_path)
    _seed_ticket(rigs_root, "t-1")
    main(["approve", "t-1", "--rig", RIG, "--rigs-root", str(rigs_root), *_ATTRIB])

    rc = main(["unapprove", "t-1", "--rig", RIG, "--rigs-root", str(rigs_root), *_ATTRIB])
    assert rc == 0
    ticket = _ticket(rigs_root, "t-1")
    assert ticket["approved"] == 0
    assert ticket["approval_hash"] is None
    assert len(_records_events(rigs_root, "unapproval")) == 1


# --- C5: reject ------------------------------------------------------------


def test_C5_reject_tombstones_filed_and_logs_reason(tmp_path: Path) -> None:
    rigs_root = scaffold_rig(tmp_path)
    _seed_filed(rigs_root, "filed-dispatch-1-1")

    rc = main(
        ["reject", "filed-dispatch-1-1", "--rig", RIG, "--rigs-root", str(rigs_root),
         *_ATTRIB, "--reason", "duplicate of t-12"]
    )
    assert rc == 0

    resolved = resolve_rig(RIG, rigs_root=rigs_root)
    try:
        filed = resolved.store.list_filed_tickets(triaged=True)
        assert len(filed) == 1
        assert filed[0]["triage_outcome"] == "rejected"
    finally:
        resolved.store.close()

    events = _records_events(rigs_root, "triage-rejected")
    assert len(events) == 1
    assert events[0]["reason"] == "duplicate of t-12"


# --- C6: promote from stdin JSON ------------------------------------------


def test_C6_promote_from_stdin(tmp_path: Path, monkeypatch, capsys) -> None:
    rigs_root = scaffold_rig(tmp_path)
    _seed_filed(rigs_root, "filed-dispatch-1-1")

    spec = {
        "id": "t-promoted",
        "title": "Promoted ticket",
        "functional_summary": "Operator-facing: fixes the export bug.",
        "acceptance_criteria": ["export is complete"],
        "tier1_checks": {"pytest": "pytest -q"},
        "target_scope": ["src/export.py"],
    }
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(spec)))
    rc = main(
        ["promote", "filed-dispatch-1-1", "--rig", RIG, "--rigs-root", str(rigs_root),
         "--spec", "-"]
    )
    assert rc == 0

    ticket = _ticket(rigs_root, "t-promoted")
    assert ticket is not None
    assert ticket["approved"] == 0
    resolved = resolve_rig(RIG, rigs_root=rigs_root)
    try:
        assert resolved.store.count_untriaged_filings() == 0
    finally:
        resolved.store.close()


# --- C7: promote with malformed spec input (robustness) --------------------


@pytest.mark.parametrize(
    "spec_arg, stdin_text",
    [
        ("/no/such/spec/file.json", None),  # unreadable path -> OSError
        ("-", "{not valid json"),  # malformed JSON -> JSONDecodeError
        ("-", "[]"),  # valid JSON, not an object -> non-dict
    ],
)
def test_C7_promote_bad_spec_input_returns_1(
    tmp_path: Path, monkeypatch, capsys, spec_arg: str, stdin_text: str | None
) -> None:
    rigs_root = scaffold_rig(tmp_path)
    _seed_filed(rigs_root, "filed-dispatch-1-1")
    if stdin_text is not None:
        import io

        monkeypatch.setattr("sys.stdin", io.StringIO(stdin_text))

    rc = main(
        ["promote", "filed-dispatch-1-1", "--rig", RIG, "--rigs-root", str(rigs_root),
         "--spec", spec_arg]
    )
    assert rc == 1
    assert "stigmergy promote:" in capsys.readouterr().err
    # nothing landed: the filed row is still untriaged
    resolved = resolve_rig(RIG, rigs_root=rigs_root)
    try:
        assert resolved.store.count_untriaged_filings() == 1
    finally:
        resolved.store.close()


# --- D1c: status -----------------------------------------------------------


def test_D1c_status(tmp_path: Path, capsys) -> None:
    rigs_root = scaffold_rig(tmp_path)
    _seed_ticket(rigs_root, "t-1")
    rc = main(["status", "--rig", RIG, "--rigs-root", str(rigs_root)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ticket states" in out
    assert "untriaged_filings" in out


# --- D2c: tickets / ticket -------------------------------------------------


def test_D2c_tickets_and_ticket_detail(tmp_path: Path, capsys) -> None:
    rigs_root = scaffold_rig(tmp_path)
    _seed_ticket(rigs_root, "t-1")

    rc = main(["tickets", "--rig", RIG, "--rigs-root", str(rigs_root)])
    assert rc == 0
    assert "t-1" in capsys.readouterr().out

    rc = main(["ticket", "t-1", "--rig", RIG, "--rigs-root", str(rigs_root)])
    assert rc == 0
    assert "functional_summary" in capsys.readouterr().out

    rc = main(["ticket", "ghost", "--rig", RIG, "--rigs-root", str(rigs_root)])
    assert rc == 0
    assert "no such ticket" in capsys.readouterr().out


# --- D3c: range-report (no critic) -----------------------------------------


def test_D3c_range_report_no_critic(tmp_path: Path, capsys) -> None:
    rigs_root = scaffold_rig(tmp_path)
    _make_staging(rigs_root)
    rc = main(["range-report", "--rig", RIG, "--rigs-root", str(rigs_root)])
    assert rc == 0
    assert "staging:" in capsys.readouterr().out
    assert _records_events(rigs_root, "report") == []  # no event without --critic


# --- D4c: range-report --critic emits a REPORT event -----------------------


def _stub_range_critic(model: str = "opus", usage: dict | None = None) -> RangeCritic:
    usage = usage if usage is not None else {"in": 1000, "cached": 0, "out": 200, "reasoning": 0}

    def client(prompt: str, *, model: str, **params: object) -> dict:
        return {"text": "advisory findings: looks fine.", "usage": usage}

    return RangeCritic(
        client=client,
        model=model,
        decoding_params={"temperature": 0.0},
        template="rangecrit template\n",
    )


def test_D4c_range_report_critic_emits_report_event(tmp_path: Path, monkeypatch, capsys) -> None:
    rigs_root = scaffold_rig(tmp_path)
    _make_staging(rigs_root)
    monkeypatch.setattr(cli, "_build_range_critic", lambda resolved: _stub_range_critic("opus"))

    rc = main(["range-report", "--rig", RIG, "--rigs-root", str(rigs_root), "--critic"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "advisory findings" in out

    events = _records_events(rigs_root, "report")
    assert len(events) == 1
    ev = events[0]
    assert ev["attempt_kind"] == "report"
    assert ev["prompt_artifact_hash"]
    assert ev["model"] == "opus"
    assert ev["tokens"] == {"in": 1000, "cached": 0, "out": 200, "reasoning": 0}
    assert isinstance(ev["computed_usd"], float)  # metered "opus" -> a real number


# --- D5c: range-report --critic with an unbudgetable model -----------------


def test_D5c_range_report_critic_unbudgetable_model(tmp_path: Path, monkeypatch, capsys) -> None:
    rigs_root = scaffold_rig(tmp_path)
    _make_staging(rigs_root)
    monkeypatch.setattr(
        cli, "_build_range_critic", lambda resolved: _stub_range_critic("no-such-model")
    )
    rc = main(["range-report", "--rig", RIG, "--rigs-root", str(rigs_root), "--critic"])
    assert rc == 0
    events = _records_events(rigs_root, "report")
    assert len(events) == 1
    assert events[0]["computed_usd"] == "unbudgetable"  # fail-closed, never $0


# --- D6c: usage discoverability --------------------------------------------


@pytest.mark.parametrize(
    "token",
    ["approve", "unapprove", "reject", "promote", "status", "tickets", "range-report"],
)
def test_D6c_usage_mentions_new_subcommands(token: str, capsys) -> None:
    main([])
    assert token in capsys.readouterr().out


# --- D7c: read-only command closes its store -------------------------------


def test_D7c_status_closes_store(tmp_path: Path, monkeypatch) -> None:
    rigs_root = scaffold_rig(tmp_path)
    _seed_ticket(rigs_root, "t-1")
    closed: list[bool] = []
    real_resolve = cli.resolve_rig

    def spy(name, *, rigs_root=None):
        resolved = real_resolve(name, rigs_root=rigs_root)
        orig_close = resolved.store.close

        def tracked() -> None:
            closed.append(True)
            orig_close()

        monkeypatch.setattr(resolved.store, "close", tracked)
        return resolved

    monkeypatch.setattr(cli, "resolve_rig", spy)
    rc = main(["status", "--rig", RIG, "--rigs-root", str(rigs_root)])
    assert rc == 0
    assert closed == [True]
