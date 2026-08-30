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
from stigmergy import approval
from stigmergy.approval import steering_hash
from stigmergy.cli import (
    _DEFAULT_PROTECTED_PATHS,
    _build_daemon,
    _make_steering_of,
    main,
)
from stigmergy.container import PodmanContainerReaper
from stigmergy.critic import Critic, CriticInfraError
from stigmergy.daemon import Daemon
from stigmergy.intake import eligible
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
    # bead .140: production staging-gate critic reads critic03 (moved-file
    # trusted-evidence bump, rollover of the bead .39 critic02 filing bump).
    (repo_dir / "prompts" / "critic03").write_text("critic03 template\n")
    # beads .41: production `range-report --critic` reads rangecrit02.
    (repo_dir / "prompts" / "rangecrit02").write_text("rangecrit02 template\n")
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
    # bead .39: inject SENTINEL filing caps on the resolved charter so the
    # weaver-wiring assertion below proves the values FLOW from
    # charter.raw[loop][dispatch_limits] (an anti-.51 hardcode would not track
    # these sentinels — the defaults are 5 / 16384).
    resolved.charter.raw["loop"]["dispatch_limits"]["filed_tickets"] = 3
    resolved.charter.raw["loop"]["dispatch_limits"]["filed_ticket_bytes"] = 9999
    # _build_daemon takes ownership of resolved.store via the Daemon.
    daemon = _build_daemon(resolved)
    try:
        # bead .39: D14 filing caps flow charter -> weaver.
        assert daemon._weaver.filing_max_filings == 3
        assert daemon._weaver.filing_max_bytes == 9999
        # bead .140: _build_daemon loads critic03 (moved-file bump), NOT critic01/02.
        assert daemon._weaver.critic.template == (
            resolved.rig_paths["prompts_dir"] / "critic03"
        ).read_text(encoding="utf-8")
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

        # bead .25: egress + relay wired for real (were .22-era placeholders).
        assert daemon._egress_setup_fn is cli.egress.setup_dispatch_egress
        assert daemon._relay_setup_fn is not None
        assert daemon._relay_teardown_fn is not None
    finally:
        daemon._store.close()


# --- case 16b: _build_daemon seeds spend leash from prior events -----------


def test_build_daemon_seeds_spend_leash_from_events(tmp_path: Path) -> None:
    """_build_daemon passes record_plane.read_events() to SpendLeash,
    seeding its initial state from any prior events already on disk."""
    from stigmergy.records import EventType, make_event

    rigs_root = scaffold_rig(tmp_path)
    resolved = resolve_rig("shipyard", rigs_root=rigs_root)
    try:
        # Seed the record plane with some prior dispatch/gate events
        record_plane = RecordPlane(resolved.rig_paths["records_dir"])

        event1 = make_event(
            EventType.DISPATCH,
            rig="shipyard",
            ticket="t-1",
            dispatch_id="d-1",
            attempt=0,
            attempt_kind="initial",
            rung=1,
            worker="worker-1",
            charter_hash="charter-hash",
            approval_hash="approval-hash",
            image_digest="image-digest",
            model="haiku",
            model_version=None,
            price_table_version="v1",
            tokens={"in": 1_000_000, "cached": 0, "out": 0, "reasoning": 0},
            computed_usd=0.8,
            prompt_artifact_hash="code01-hash-1",
            wall_time_seconds=1.0,
        )
        record_plane.append(event1)

        event2 = make_event(
            EventType.DISPATCH,
            rig="shipyard",
            ticket="t-1",
            dispatch_id="d-2",
            attempt=0,
            attempt_kind="initial",
            rung=1,
            worker="worker-1",
            charter_hash="charter-hash",
            approval_hash="approval-hash",
            image_digest="image-digest",
            model="haiku",
            model_version=None,
            price_table_version="v1",
            tokens={"in": 1_000_000, "cached": 0, "out": 0, "reasoning": 0},
            computed_usd=0.8,
            prompt_artifact_hash="code01-hash-2",
            wall_time_seconds=1.0,
        )
        record_plane.append(event2)

        event3 = make_event(
            EventType.GATE,
            rig="shipyard",
            ticket=None,
            dispatch_id=None,
            attempt=0,
            attempt_kind="initial",
            rung=None,
            worker=None,
            charter_hash="charter-hash",
            approval_hash=None,
            image_digest=None,
            model="haiku",
            model_version=None,
            price_table_version="v1",
            tokens={"in": 500_000, "cached": 0, "out": 0, "reasoning": 0},
            computed_usd=0.4,
            prompt_artifact_hash="critic01-hash",
            decoding_params={},
            wall_time_seconds=1.0,
        )
        record_plane.append(event3)

        # Now build the daemon — it should seed from these events
        daemon = _build_daemon(resolved)
        try:
            report = daemon._spend_leash.run_report()
            # Should reflect seeded state: 2 dispatches, 1 gate, $2.0 total
            assert report["dispatches_used"] == 2
            assert report["gate_calls_used"] == 1
            assert report["metered_spent"] == pytest.approx(2.0)
        finally:
            daemon._store.close()
    finally:
        # resolved.store may already be closed by _build_daemon, but try anyway
        try:
            resolved.store.close()
        except Exception:
            pass


# --- bead .25: _build_daemon credential-relay wiring ------------------------


def test_build_daemon_relay_wiring(tmp_path: Path, monkeypatch) -> None:
    """bead .25: the relay-setup closure constructs a CredentialRelay that
    SHARES the daemon's CapabilityStore, pins anthropic-version, widens the
    header allowlist to include anthropic-beta (SB option A), and start_relay
    gets the make_urllib_forwarder result. make_op_key_provider is lazy, so
    no real op/network call happens here."""
    rigs_root = scaffold_rig(tmp_path)
    resolved = resolve_rig("shipyard", rigs_root=rigs_root)

    kp_refs: list[str] = []
    monkeypatch.setattr(
        cli, "make_op_key_provider",
        lambda ref: (kp_refs.append(ref), (lambda: "sk-fake-not-real"))[1],
    )
    fwd_urls: list[str] = []

    def fake_make_forwarder(*, base_url, **kw):
        fwd_urls.append(base_url)
        return "FAKE_FORWARDER"

    monkeypatch.setattr(cli, "make_urllib_forwarder", fake_make_forwarder)

    captured: dict = {}

    def fake_start_relay(provisional_id, runtime_dir, relay, *, forwarder, log_path):
        captured["relay"] = relay
        captured["forwarder"] = forwarder
        captured["provisional_id"] = provisional_id
        captured["log_path"] = log_path
        return object()  # stand-in RelayHandle

    monkeypatch.setattr(cli, "start_relay", fake_start_relay)

    daemon = _build_daemon(resolved)
    try:
        # E19/E20: the relay key ref + forwarder base URL are wired.
        assert cli._RELAY_KEY_REF in kp_refs
        assert "https://api.anthropic.com" in fwd_urls

        # invoke the per-dispatch relay closure
        handle = daemon._relay_setup_fn("relay-xyz", tmp_path)
        assert handle is not None
        relay = captured["relay"]
        # shared store (load-bearing — E18)
        assert relay._store is daemon._capability_store
        # pinned version + widened allowlist (SB option A)
        assert relay._upstream_headers_pinned == {"anthropic-version": "2023-06-01"}
        assert relay._upstream_header_allowlist == frozenset(
            {"content-type", "accept", "anthropic-beta"}
        )
        # start_relay got the forwarder from make_urllib_forwarder
        assert captured["forwarder"] == "FAKE_FORWARDER"
        assert captured["provisional_id"] == "relay-xyz"

        # bead .25 audit F-2: the transcript backstop is armed against the REAL
        # key (not just the capability token).
        from stigmergy.relay import Capability

        cap = Capability(token="cap-tok-xyz", dispatch_id="d1",
                         max_output_tokens=1, max_calls=1)
        secret_set = daemon._secrets_for_capability(cap)
        assert "cap-tok-xyz" in secret_set
        assert "sk-fake-not-real" in secret_set  # the (monkeypatched) real key
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
    """Ensure a `staging` branch exists in the rig's repo clone so range-report
    has a ref to compute against. Scaffold now creates the charter's
    dispatch_base branch (bead .90), so this is idempotent — a no-op when
    staging already exists, still creating it for a rig whose dispatch_base is
    not `staging`."""
    repo = rigs_root / RIG / "repo"
    exists = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--verify", "--quiet", "refs/heads/staging"],
        capture_output=True,
        text=True,
        check=False,
    )
    if exists.returncode != 0:
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


# --- C8: resume ---------------------------------------------------------------


def test_C8_resume_escalated_ticket_success(tmp_path: Path) -> None:
    """Resume an escalated ticket: state->pool, leases cleared, counters zeroed
    (except lifetime counters), approval preserved."""
    rigs_root = scaffold_rig(tmp_path)

    # Create a ticket in ESCALATED with approval and non-trivial lease/counters
    steering = {
        "ticket_text": "Test ticket",
        "checks": {},
        "rubric": [],
        "lane": "cheap",
        "prompt_bytes": "code01-prompt-v1",
        "context_set": [],
    }
    approval_hash_value = steering_hash(steering)

    _seed_ticket(
        rigs_root,
        "t-esc",
        state="escalated",
        approved=1,
        approval_hash=approval_hash_value,
        lease_owner="worker-1",
        lease_dispatch_id="dispatch-1",
        lease_expires_at=1000.0,
        lease_heartbeat_at=999.0,
        attempts_used=3,
        current_rung="exquisite",
        integration_failures=1,
        target_scope=["src/test.py"],
        functional_summary="Test the thing",
    )

    rc = main(["resume", "t-esc", "--rig", RIG, "--rigs-root", str(rigs_root), *_ATTRIB])
    assert rc == 0

    # Check ticket state after resume
    ticket = _ticket(rigs_root, "t-esc")
    assert ticket["state"] == "pool"
    # Leases cleared
    assert ticket["lease_owner"] is None
    assert ticket["lease_dispatch_id"] is None
    assert ticket["lease_expires_at"] is None
    assert ticket["lease_heartbeat_at"] is None
    # Counters zeroed
    assert ticket["attempts_used"] == 0
    assert ticket["current_rung"] is None
    # Lifetime counters preserved
    assert ticket["integration_failures"] == 1
    # Approval preserved
    assert ticket["approved"] == 1
    assert ticket["approval_hash"] == approval_hash_value
    # Steering preserved
    assert ticket["target_scope"] == ["src/test.py"]
    assert ticket["functional_summary"] == "Test the thing"

    # Check resume event was logged
    events = _records_events(rigs_root, "resume")
    assert len(events) == 1
    assert events[0]["subject_id"] == "t-esc"
    assert events[0]["outcome"] == "resumed"
    assert events[0]["acting_agent"] == "merry"
    assert events[0]["operator_session"] == "sess-42"
    assert events[0]["approval_hash"] == approval_hash_value


def test_C8b_resume_non_escalated_ticket_returns_error(tmp_path: Path, capsys) -> None:
    """Resume of a non-escalated ticket returns error without mutation."""
    rigs_root = scaffold_rig(tmp_path)
    _seed_ticket(rigs_root, "t-pool", state="pool")

    rc = main(["resume", "t-pool", "--rig", RIG, "--rigs-root", str(rigs_root), *_ATTRIB])
    assert rc == 1

    # Error message names the actual state
    err = capsys.readouterr().err
    assert "not escalated" in err
    assert "state='pool'" in err

    # Ticket untouched
    ticket = _ticket(rigs_root, "t-pool")
    assert ticket["state"] == "pool"

    # No event logged
    assert _records_events(rigs_root, "resume") == []


def test_C8c_resume_missing_ticket_returns_error(tmp_path: Path, capsys) -> None:
    """Resume of a non-existent ticket returns error without mutation."""
    rigs_root = scaffold_rig(tmp_path)

    rc = main(["resume", "ghost", "--rig", RIG, "--rigs-root", str(rigs_root), *_ATTRIB])
    assert rc == 1

    # Error message names the missing ticket
    err = capsys.readouterr().err
    assert "no such ticket" in err

    # No event logged
    assert _records_events(rigs_root, "resume") == []


def test_C8d_resume_missing_attribution_is_argparse_error(tmp_path: Path) -> None:
    """Resume without attribution flags is an argparse error."""
    rigs_root = scaffold_rig(tmp_path)
    _seed_ticket(rigs_root, "t-esc", state="escalated")

    # Missing --agent should cause SystemExit(2)
    with pytest.raises(SystemExit) as exc_info:
        main(["resume", "t-esc", "--rig", RIG, "--rigs-root", str(rigs_root)])
    assert exc_info.value.code == 2

    # Ticket untouched
    ticket = _ticket(rigs_root, "t-esc")
    assert ticket["state"] == "escalated"

    # No event logged
    assert _records_events(rigs_root, "resume") == []


def test_C8e_resumed_escalated_ticket_becomes_claimable(tmp_path: Path) -> None:
    """After resume, an approved+hash-valid ticket is eligible for claiming."""
    rigs_root = scaffold_rig(tmp_path)

    # Create a ticket in ESCALATED with approval
    steering = {
        "ticket_text": "Test ticket for claiming",
        "checks": {},
        "rubric": [],
        "lane": "cheap",
        "prompt_bytes": "code01-prompt-v1",
        "context_set": [],
    }
    approval_hash_value = steering_hash(steering)

    _seed_ticket(
        rigs_root,
        "t-claim",
        state="escalated",
        approved=1,
        approval_hash=approval_hash_value,
    )

    # Resume the ticket
    rc = main(["resume", "t-claim", "--rig", RIG, "--rigs-root", str(rigs_root), *_ATTRIB])
    assert rc == 0

    # Check that the ticket is now eligible
    resolved = resolve_rig(RIG, rigs_root=rigs_root)
    try:
        # Build a steering_of function that returns the matching steering
        def steering_of(ticket_id: str) -> dict:
            return steering if ticket_id == "t-claim" else {}

        eligible_ids = eligible(resolved.store, now=1000.0, steering_of=steering_of)
        assert "t-claim" in eligible_ids
    finally:
        resolved.store.close()


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


def _stub_range_critic(
    model: str = "opus",
    usage: dict | None = None,
    filed_tickets: list | None = None,
) -> RangeCritic:
    usage = usage if usage is not None else {"in": 1000, "cached": 0, "out": 200, "reasoning": 0}

    def client(prompt: str, *, model: str, **params: object) -> dict:
        resp = {"text": "advisory findings: looks fine.", "usage": usage}
        if filed_tickets is not None:
            resp["filed_tickets"] = filed_tickets
        return resp

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
    [
        "approve",
        "unapprove",
        "resume",
        "reject",
        "promote",
        "status",
        "monitor",
        "tickets",
        "range-report",
        "--operator-session",
    ],
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


# --- D8c: monitor command ---------------------------------------------------


def test_D8c_monitor(tmp_path: Path, capsys) -> None:
    rigs_root = scaffold_rig(tmp_path)
    _seed_ticket(rigs_root, "t-1")
    rc = main(["monitor", "--rig", RIG, "--rigs-root", str(rigs_root)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ticket states" in out
    assert "recent events" in out


def test_D8d_monitor_closes_store(tmp_path: Path, monkeypatch) -> None:
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
    rc = main(["monitor", "--rig", RIG, "--rigs-root", str(rigs_root)])
    assert rc == 0
    assert closed == [True]


# ==========================================================================
# beads .51 + .41 — combined-schema range-critic: production wiring fix +
# filing of range-critic findings + unbudgetable-never-$0.
# AUTHORED BY THE ORCHESTRATOR (Merry), not the implementor.
# ==========================================================================


def _records_all(rigs_root: Path) -> list[dict]:
    resolved = resolve_rig(RIG, rigs_root=rigs_root)
    try:
        return list(RecordPlane(resolved.rig_paths["records_dir"]).read_events())
    finally:
        resolved.store.close()


def _filed(rigs_root: Path) -> list[dict]:
    resolved = resolve_rig(RIG, rigs_root=rigs_root)
    try:
        return resolved.store.list_filed_tickets()
    finally:
        resolved.store.close()


# --- .51 wiring regression: the REAL _build_range_critic ------------------
# The .51 bug was a WIRING bug: _build_range_critic used the verdict client
# (make_critic_client) + rangecrit01. The fix wires make_range_critic_client
# + rangecrit02. This exercises the REAL builder (no _build_range_critic
# monkeypatch), proving both, without shelling out to 1Password.


def test_51_build_range_critic_uses_range_client_and_rangecrit02(tmp_path, monkeypatch):
    import hashlib

    rigs_root = scaffold_rig(tmp_path)
    resolved = resolve_rig(RIG, rigs_root=rigs_root)
    try:
        made = {"range_client": False}

        def fake_range_client(**kwargs):
            made["range_client"] = True

            def client(prompt, *, model, **params):
                return {"text": "F", "usage": {}, "filed_tickets": []}

            return client

        def boom_verdict_client(**kwargs):  # the .51 bug: MUST NOT be used here
            raise AssertionError("range-report must NOT use the verdict client")

        # make_op_key_provider is lazy in production; stub it so no `op` call.
        monkeypatch.setattr(cli, "make_op_key_provider", lambda ref: (lambda: "dummy-key"))
        monkeypatch.setattr(cli, "make_range_critic_client", fake_range_client)
        monkeypatch.setattr(cli, "make_critic_client", boom_verdict_client)

        critic = cli._build_range_critic(resolved)

        assert made["range_client"] is True  # range client, not verdict client
        # ...and the rangecrit02 template (the .41 prompt bump), not rangecrit01.
        rc02 = resolved.rig_paths["prompts_dir"] / "rangecrit02"
        assert critic.prompt_artifact_hash == hashlib.sha256(rc02.read_bytes()).hexdigest()
    finally:
        resolved.store.close()


# --- .41 filing: range-critic proposals land unapproved + provenanced -----

_TICKETS = [
    {
        "title": "Dedup range-base resolution",
        "description": "weaver+rangereport diverge",
        "evidence": "rangereport.py:210",
    },
    {"title": "Add cross-ticket test", "description": "interaction untested"},
]


def test_41_range_report_critic_files_proposals_unapproved(tmp_path, monkeypatch, capsys):
    rigs_root = scaffold_rig(tmp_path)
    _make_staging(rigs_root)
    monkeypatch.setattr(
        cli, "_build_range_critic",
        lambda resolved: _stub_range_critic("opus", filed_tickets=_TICKETS),
    )
    rc = main(["range-report", "--rig", RIG, "--rigs-root", str(rigs_root), "--critic"])
    assert rc == 0

    filed = _filed(rigs_root)
    assert len(filed) == 2
    for row in filed:
        assert row["origin_role"] == "range-critic"
        assert row["triaged"] == 0  # UNAPPROVED — sits until human triage
        # loop-stamped discovered-from provenance (report dispatch id).
        assert row["discovered_from"].startswith("report-")
        assert row["origin_dispatch_id"].startswith("report-")
    titles = {r["title"] for r in filed}
    assert titles == {"Dedup range-base resolution", "Add cross-ticket test"}

    # ticket-filed events emitted (honest zero cost — no double count).
    tf = _records_events(rigs_root, "ticket-filed")
    assert len(tf) == 2
    for ev in tf:
        assert ev["origin"]["role"] == "range-critic"
        assert ev["computed_usd"] == 0.0
        assert ev["outcome"] == "accepted"

    # findings AND the filed proposal ids are surfaced to the operator.
    out = capsys.readouterr().out
    assert "advisory findings" in out
    for row in filed:
        assert row["id"] in out


def test_41_range_report_critic_files_nothing_when_no_proposals(tmp_path, monkeypatch, capsys):
    rigs_root = scaffold_rig(tmp_path)
    _make_staging(rigs_root)
    monkeypatch.setattr(
        cli, "_build_range_critic", lambda resolved: _stub_range_critic("opus", filed_tickets=[])
    )
    rc = main(["range-report", "--rig", RIG, "--rigs-root", str(rigs_root), "--critic"])
    assert rc == 0
    assert _filed(rigs_root) == []
    assert _records_events(rigs_root, "ticket-filed") == []
    # the REPORT event is still emitted (the paid call happened).
    assert len(_records_events(rigs_root, "report")) == 1


def test_41_report_event_emitted_before_ticket_filed_events(tmp_path, monkeypatch, capsys):
    # Ordering (advisor pt4): the paid REPORT event is recorded BEFORE filing,
    # so the accounting survives even if filing degrades.
    rigs_root = scaffold_rig(tmp_path)
    _make_staging(rigs_root)
    monkeypatch.setattr(
        cli, "_build_range_critic",
        lambda resolved: _stub_range_critic("opus", filed_tickets=_TICKETS),
    )
    main(["range-report", "--rig", RIG, "--rigs-root", str(rigs_root), "--critic"])
    kinds = [e["event_type"] for e in _records_all(rigs_root)]
    assert "report" in kinds
    assert "ticket-filed" in kinds
    assert kinds.index("report") < kinds.index("ticket-filed")


def test_41_bad_shape_proposals_rejected_without_corrupting_pool(tmp_path, monkeypatch, capsys):
    # file_proposals is the single validation authority: a malformed item is
    # rejected (bad-shape event), the well-formed one lands, and the REPORT
    # event is still emitted. review() passes items verbatim; the CLI files.
    rigs_root = scaffold_rig(tmp_path)
    _make_staging(rigs_root)
    mixed = [{"title": "good", "description": "d"}, {"missing": "title"}]
    monkeypatch.setattr(
        cli, "_build_range_critic",
        lambda resolved: _stub_range_critic("opus", filed_tickets=mixed),
    )
    rc = main(["range-report", "--rig", RIG, "--rigs-root", str(rigs_root), "--critic"])
    assert rc == 0
    filed = _filed(rigs_root)
    assert [r["title"] for r in filed] == ["good"]
    outcomes = {e["outcome"] for e in _records_events(rigs_root, "ticket-filed")}
    assert outcomes == {"accepted", "rejected"}
    assert len(_records_events(rigs_root, "report")) == 1


# --- finding #5 fold-in: absent metered usage -> unbudgetable, never $0 ----


def test_51_metered_model_absent_usage_is_unbudgetable_not_zero(tmp_path, monkeypatch, capsys):
    # SB decision: a paid (metered) call whose usage cannot be parsed records
    # computed_usd="unbudgetable", NEVER $0 — AND still delivers the findings
    # and files the proposals (never lose the report).
    rigs_root = scaffold_rig(tmp_path)
    _make_staging(rigs_root)
    monkeypatch.setattr(
        cli, "_build_range_critic",
        lambda resolved: _stub_range_critic("opus", usage={}, filed_tickets=_TICKETS),
    )
    rc = main(["range-report", "--rig", RIG, "--rigs-root", str(rigs_root), "--critic"])
    assert rc == 0

    events = _records_events(rigs_root, "report")
    assert len(events) == 1
    assert events[0]["computed_usd"] == "unbudgetable"  # never a silent 0.0
    # findings delivered AND proposals filed despite the unbudgetable usage.
    assert "advisory findings" in capsys.readouterr().out
    assert len(_filed(rigs_root)) == 2


def test_51_subscription_model_absent_usage_stays_declared_zero(tmp_path, monkeypatch, capsys):
    # Only METERED absent-usage is unbudgetable. A subscription model's marginal
    # cost is a declared $0 (registry value), not a fallback — usage absence
    # does not make it unbudgetable.
    rigs_root = scaffold_rig(tmp_path)
    _make_staging(rigs_root)
    monkeypatch.setattr(
        cli, "_build_range_critic",
        lambda resolved: _stub_range_critic("claude-max-sub", usage={}),
    )
    rc = main(["range-report", "--rig", RIG, "--rigs-root", str(rigs_root), "--critic"])
    assert rc == 0
    events = _records_events(rigs_root, "report")
    assert len(events) == 1
    assert events[0]["computed_usd"] == 0.0  # declared subscription $0, not unbudgetable


# --- filed: list untriaged filed proposals ---


def test_filed_with_populated_untriaged_pool(tmp_path: Path, capsys) -> None:
    rigs_root = scaffold_rig(tmp_path)
    _seed_filed(rigs_root, "filed-1")
    _seed_filed(rigs_root, "filed-2")

    rc = main(["filed", "--rig", RIG, "--rigs-root", str(rigs_root)])
    assert rc == 0
    out = capsys.readouterr().out

    # Check that both filed proposals appear
    assert "filed-1" in out
    assert "filed-2" in out
    assert "proposal" in out  # title field

    # Check that provenance fields are rendered
    assert "origin_role=worker" in out
    assert "origin_worker=worker-1" in out
    assert "origin_dispatch_id=dispatch-1" in out
    assert "discovered_from=dispatch-1@workspace-e2uh.8" in out


def test_filed_with_empty_pool(tmp_path: Path, capsys) -> None:
    rigs_root = scaffold_rig(tmp_path)

    rc = main(["filed", "--rig", RIG, "--rigs-root", str(rigs_root)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "(no filed proposals)" in out


# --- ticket new subcommand tests -------------------------------------------------


def test_ticket_new_from_file(tmp_path: Path, capsys) -> None:
    """Test ticket new creates an unapproved ticket from a file spec."""
    rigs_root = scaffold_rig(tmp_path)

    spec = {
        "id": "t-new-1",
        "title": "New ticket from file",
        "functional_summary": "Operator-facing: new ticket created via ticket new.",
        "acceptance_criteria": ["works perfectly"],
        "target_scope": ["src/new.py"],
    }
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(json.dumps(spec))

    rc = main(
        ["ticket", "new", "--rig", RIG, "--rigs-root", str(rigs_root), "--spec", str(spec_file)]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "t-new-1" in out

    ticket = _ticket(rigs_root, "t-new-1")
    assert ticket is not None
    assert ticket["title"] == "New ticket from file"
    assert ticket["approved"] == 0  # unapproved
    assert ticket["functional_summary"] == "Operator-facing: new ticket created via ticket new."

    # Verify it can be approved without error
    resolved = resolve_rig(RIG, rigs_root=rigs_root)
    try:
        steering = derive_steering(ticket, resolved.charter, resolved.rig_paths["prompts_dir"])
        assert steering is not None
        approval.approve(resolved.store, "t-new-1", steering=steering)
        approved_ticket = resolved.store.get_ticket("t-new-1")
        assert approved_ticket is not None
        assert approved_ticket["approved"] == 1
    finally:
        resolved.store.close()


def test_ticket_new_from_stdin(tmp_path: Path, monkeypatch, capsys) -> None:
    """Test ticket new reads from stdin when spec is '-'."""
    rigs_root = scaffold_rig(tmp_path)

    spec = {
        "id": "t-new-stdin",
        "title": "New ticket from stdin",
        "functional_summary": "Operator-facing: created from stdin.",
        "acceptance_criteria": ["done"],
        "target_scope": ["src/"],
    }
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(spec)))
    rc = main(["ticket", "new", "--rig", RIG, "--rigs-root", str(rigs_root), "--spec", "-"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "t-new-stdin" in out

    ticket = _ticket(rigs_root, "t-new-stdin")
    assert ticket is not None
    assert ticket["approved"] == 0


def test_ticket_new_with_optional_fields(tmp_path: Path, capsys) -> None:
    """Test ticket new forwards optional fields correctly."""
    rigs_root = scaffold_rig(tmp_path)

    spec = {
        "id": "t-new-optional",
        "title": "New ticket with optional fields",
        "functional_summary": "Operator-facing: optional fields forwarded.",
        "acceptance_criteria": ["done"],
        "target_scope": ["src/"],
        "goal": "Ship the feature",
        "required_reading": ["docs/design.md"],
        "difficulty": "medium",
        "lane_hint": "normal",
        "rubric_only": False,
        "work_product": "pull request",
    }
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(json.dumps(spec))

    rc = main(
        ["ticket", "new", "--rig", RIG, "--rigs-root", str(rigs_root), "--spec", str(spec_file)]
    )
    assert rc == 0

    ticket = _ticket(rigs_root, "t-new-optional")
    assert ticket is not None
    assert ticket["goal"] == "Ship the feature"
    assert ticket["difficulty"] == "medium"


def test_ticket_new_missing_required_field(tmp_path: Path, capsys) -> None:
    """Test ticket new rejects incomplete specs."""
    rigs_root = scaffold_rig(tmp_path)

    # Missing 'functional_summary'
    spec = {
        "id": "t-new-incomplete",
        "title": "Incomplete ticket",
        "acceptance_criteria": ["done"],
        "target_scope": ["src/"],
    }
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(json.dumps(spec))

    rc = main(
        ["ticket", "new", "--rig", RIG, "--rigs-root", str(rigs_root), "--spec", str(spec_file)]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "stigmergy ticket new:" in err
    assert "missing required key" in err

    # Verify nothing was created
    assert _ticket(rigs_root, "t-new-incomplete") is None


def test_ticket_new_empty_functional_summary(tmp_path: Path, capsys) -> None:
    """Test ticket new rejects empty functional_summary."""
    rigs_root = scaffold_rig(tmp_path)

    spec = {
        "id": "t-new-empty-summary",
        "title": "Empty summary",
        "functional_summary": "   ",  # whitespace-only
        "acceptance_criteria": ["done"],
        "target_scope": ["src/"],
    }
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(json.dumps(spec))

    rc = main(
        ["ticket", "new", "--rig", RIG, "--rigs-root", str(rigs_root), "--spec", str(spec_file)]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "stigmergy ticket new:" in err
    assert "must be a non-empty string" in err

    assert _ticket(rigs_root, "t-new-empty-summary") is None


def test_ticket_new_duplicate_id(tmp_path: Path, capsys) -> None:
    """Test ticket new rejects duplicate ticket ids."""
    rigs_root = scaffold_rig(tmp_path)
    _seed_ticket(rigs_root, "t-existing")

    spec = {
        "id": "t-existing",
        "title": "Duplicate id",
        "functional_summary": "Operator-facing: duplicate.",
        "acceptance_criteria": ["done"],
        "target_scope": ["src/"],
    }
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(json.dumps(spec))

    rc = main(
        ["ticket", "new", "--rig", RIG, "--rigs-root", str(rigs_root), "--spec", str(spec_file)]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "stigmergy ticket new:" in err
    assert "ticket id already exists" in err


def test_ticket_new_blocks_key_rejected(tmp_path: Path, capsys) -> None:
    """Test ticket new rejects specs with blocks key."""
    rigs_root = scaffold_rig(tmp_path)

    spec = {
        "id": "t-new-with-blocks",
        "title": "Has blocks key",
        "functional_summary": "Operator-facing: has blocks key.",
        "acceptance_criteria": ["done"],
        "target_scope": ["src/"],
        "blocks": ["t-predecessor"],
    }
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(json.dumps(spec))

    rc = main(
        ["ticket", "new", "--rig", RIG, "--rigs-root", str(rigs_root), "--spec", str(spec_file)]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "stigmergy ticket new:" in err
    assert "blocks" in err and "not allowed" in err


# --- intake subcommand tests ---------------------------------------------------


def test_intake_creates_multiple_tickets(tmp_path: Path, capsys) -> None:
    """Test intake creates multiple tickets from a manifest."""
    rigs_root = scaffold_rig(tmp_path)

    manifest = [
        {
            "id": "t-intake-1",
            "title": "Intake ticket 1",
            "functional_summary": "Operator-facing: first intake ticket.",
            "acceptance_criteria": ["done"],
            "target_scope": ["src/"],
        },
        {
            "id": "t-intake-2",
            "title": "Intake ticket 2",
            "functional_summary": "Operator-facing: second intake ticket.",
            "acceptance_criteria": ["works"],
            "target_scope": ["src/"],
        },
    ]
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest))

    rc = main(
        ["intake", "--rig", RIG, "--rigs-root", str(rigs_root), "--manifest", str(manifest_file)]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "t-intake-1" in out
    assert "t-intake-2" in out

    # Verify both tickets were created
    t1 = _ticket(rigs_root, "t-intake-1")
    t2 = _ticket(rigs_root, "t-intake-2")
    assert t1 is not None
    assert t2 is not None
    assert t1["approved"] == 0
    assert t2["approved"] == 0


def test_intake_with_blocks_dependency(tmp_path: Path, capsys) -> None:
    """Test intake wires blocks dependencies correctly."""
    rigs_root = scaffold_rig(tmp_path)

    manifest = [
        {
            "id": "t-predecessor",
            "title": "Predecessor",
            "functional_summary": "Operator-facing: predecessor ticket.",
            "acceptance_criteria": ["done"],
            "target_scope": ["src/"],
        },
        {
            "id": "t-dependent",
            "title": "Dependent",
            "functional_summary": "Operator-facing: dependent ticket.",
            "acceptance_criteria": ["done"],
            "target_scope": ["src/"],
            "blocks": ["t-predecessor"],
        },
    ]
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest))

    rc = main(
        ["intake", "--rig", RIG, "--rigs-root", str(rigs_root), "--manifest", str(manifest_file)]
    )
    assert rc == 0

    # Verify the dependency was wired
    resolved = resolve_rig(RIG, rigs_root=rigs_root)
    try:
        deps = resolved.store.deps_of("t-dependent")
        assert "t-predecessor" in deps
    finally:
        resolved.store.close()


def test_intake_with_predecessor_defined_later_in_manifest(tmp_path: Path, capsys) -> None:
    """Test intake allows blocks reference to entries defined later in manifest."""
    rigs_root = scaffold_rig(tmp_path)

    manifest = [
        {
            "id": "t-later",
            "title": "Later entry",
            "functional_summary": "Defined later but referenced earlier.",
            "acceptance_criteria": ["done"],
            "target_scope": ["src/"],
            "blocks": ["t-earlier"],
        },
        {
            "id": "t-earlier",
            "title": "Earlier entry",
            "functional_summary": "Defined later in manifest.",
            "acceptance_criteria": ["done"],
            "target_scope": ["src/"],
        },
    ]
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest))

    rc = main(
        ["intake", "--rig", RIG, "--rigs-root", str(rigs_root), "--manifest", str(manifest_file)]
    )
    assert rc == 0

    # Verify both tickets exist and dependency is wired
    t1 = _ticket(rigs_root, "t-later")
    t2 = _ticket(rigs_root, "t-earlier")
    assert t1 is not None
    assert t2 is not None

    resolved = resolve_rig(RIG, rigs_root=rigs_root)
    try:
        deps = resolved.store.deps_of("t-later")
        assert "t-earlier" in deps
    finally:
        resolved.store.close()


def test_intake_one_invalid_entry_fails_whole_manifest(tmp_path: Path, capsys) -> None:
    """Test intake validates all entries before inserting anything."""
    rigs_root = scaffold_rig(tmp_path)

    manifest = [
        {
            "id": "t-valid-1",
            "title": "Valid ticket 1",
            "functional_summary": "Operator-facing: first valid.",
            "acceptance_criteria": ["done"],
            "target_scope": ["src/"],
        },
        {
            "id": "t-invalid",
            "title": "Invalid ticket",
            # Missing 'functional_summary'
            "acceptance_criteria": ["done"],
            "target_scope": ["src/"],
        },
        {
            "id": "t-valid-2",
            "title": "Valid ticket 2",
            "functional_summary": "Operator-facing: second valid.",
            "acceptance_criteria": ["done"],
            "target_scope": ["src/"],
        },
    ]
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest))

    rc = main(
        ["intake", "--rig", RIG, "--rigs-root", str(rigs_root), "--manifest", str(manifest_file)]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "stigmergy intake:" in err
    assert "missing required key" in err

    # Verify NONE of the tickets were created
    assert _ticket(rigs_root, "t-valid-1") is None
    assert _ticket(rigs_root, "t-invalid") is None
    assert _ticket(rigs_root, "t-valid-2") is None


def test_intake_duplicate_id_in_manifest(tmp_path: Path, capsys) -> None:
    """Test intake rejects duplicate ids within the manifest."""
    rigs_root = scaffold_rig(tmp_path)

    manifest = [
        {
            "id": "t-duplicate",
            "title": "First",
            "functional_summary": "Operator-facing: first.",
            "acceptance_criteria": ["done"],
            "target_scope": ["src/"],
        },
        {
            "id": "t-duplicate",  # duplicate id
            "title": "Second",
            "functional_summary": "Operator-facing: second.",
            "acceptance_criteria": ["done"],
            "target_scope": ["src/"],
        },
    ]
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest))

    rc = main(
        ["intake", "--rig", RIG, "--rigs-root", str(rigs_root), "--manifest", str(manifest_file)]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "duplicate id" in err

    # Verify nothing was created
    assert _ticket(rigs_root, "t-duplicate") is None


def test_intake_unresolved_blocks_predecessor_rejected(tmp_path: Path, capsys) -> None:
    """Test intake rejects manifest with unresolved blocks reference.

    Regression test: lock in the existing unresolved-blocks rejection behavior
    in the intake manifest handler so a future refactor cannot silently drop it.
    """
    rigs_root = scaffold_rig(tmp_path)

    manifest = [
        {
            "id": "t-dependent",
            "title": "Dependent ticket",
            "functional_summary": "Operator-facing: depends on unknown ticket.",
            "acceptance_criteria": ["done"],
            "target_scope": ["src/"],
            "blocks": ["t-nonexistent"],  # references a ticket that does not exist
        },
    ]
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest))

    rc = main(
        ["intake", "--rig", RIG, "--rigs-root", str(rigs_root), "--manifest", str(manifest_file)]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "unresolved predecessor" in err
    assert "t-nonexistent" in err

    # Verify no tickets were created (even the dependent one)
    assert _ticket(rigs_root, "t-dependent") is None
    assert _ticket(rigs_root, "t-nonexistent") is None


def test_ticket_detail_still_works_unchanged(tmp_path: Path, capsys) -> None:
    """Regression test: ticket <id> show-detail subcommand still works."""
    rigs_root = scaffold_rig(tmp_path)
    _seed_ticket(rigs_root, "t-1")

    rc = main(["ticket", "t-1", "--rig", RIG, "--rigs-root", str(rigs_root)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "t-1" in out
    assert "functional_summary" in out


# ==========================================================================
# bead .143 — critic wiring migration to the OA provider layer
# ==========================================================================


def _set_charter_critic_section(rigs_root: Path, *, model: str | None = None, max_tokens=None):
    """Rewrite the scaffolded rig's charter `[roles.critic]` section
    (set `model` and/or `max_tokens` when given)."""
    import re as _re

    charter_path = rigs_root / RIG / "charter.toml"
    text = charter_path.read_text()

    def _replacement(_m: _re.Match[str]) -> str:
        # Preserve keys the caller did NOT override: parse the matched
        # section's existing key=value lines, overlay the requested
        # changes, re-emit. (Rebuilding from scratch dropped an unset
        # `model` and produced charter-invalidating sections.)
        existing: dict[str, str] = {}
        for line in _m.group(0).splitlines()[1:]:
            parts = line.split("=", 1)
            if len(parts) == 2:
                existing[parts[0].strip()] = parts[1].strip()
        if model is not None:
            existing["model"] = f'"{model}"'
        if max_tokens is not None:
            existing["max_tokens"] = str(max_tokens)
        lines = ["[roles.critic]"]
        for key, value in existing.items():
            lines.append(f"{key} = {value}")
        return "\n".join(lines)

    # Match ONLY the header + its key=value lines (lines not starting
    # with `[`), leaving the blank line + next `[table]` header intact.
    # Function-replacement (not a template string) so no re.sub escaping.
    new_text = _re.sub(r"\[roles\.critic\](?:\n[^\n\[]+)*", _replacement, text)
    assert new_text != text
    assert "[roles.critic]" in new_text
    # sanity: the next table after [roles.critic] is still a valid header
    assert "\n[models]" in new_text
    charter_path.write_text(new_text)


def test_143_daemon_builds_critic_via_oa_factory(tmp_path: Path, monkeypatch) -> None:
    """`_build_daemon` constructs the critic through
    `make_oa_critic_client` (NOT the deleted `make_critic_client`),
    forwarding the charter's `roles.critic.max_tokens` to the factory."""
    rigs_root = scaffold_rig(tmp_path)
    _set_charter_critic_section(rigs_root, max_tokens=8192)
    recorded: list[dict] = []

    def fake_oa_client(**kwargs):
        recorded.append(kwargs)

        def client(prompt, *, model, **params):
            return {}

        return client

    monkeypatch.setattr(cli, "make_oa_critic_client", fake_oa_client)
    monkeypatch.setattr(cli, "make_op_key_provider", lambda ref: (lambda: "dummy-key"))

    from stigmergy.rig import resolve_rig

    resolved = resolve_rig(RIG, rigs_root=rigs_root)
    try:
        daemon = _build_daemon(resolved)
        try:
            assert len(recorded) == 1
            assert recorded[0]["max_tokens"] == 8192  # charter value forwarded
            # the key provider is the one built from the rig's critic op ref:
            assert recorded[0]["key_provider"]() == "dummy-key"
            assert isinstance(daemon._weaver.critic, Critic)
        finally:
            daemon._store.close()
    finally:
        resolved.store.close()


def test_143_daemon_critic_key_ref_defaults_for_anthropic(
    tmp_path: Path, monkeypatch
) -> None:
    """When the resolved critic entry is Anthropic-routed (the default),
    the key provider is built from `_CRITIC_KEY_REF` (unchanged) — the
    `STIGMERGY_CRITIC_OA_KEY_REF` env is NOT consulted (and its absence is not an
    error)."""
    rigs_root = scaffold_rig(tmp_path)
    refs: list[str] = []
    monkeypatch.delenv("STIGMERGY_CRITIC_OA_KEY_REF", raising=False)
    monkeypatch.setattr(
        cli, "make_op_key_provider", lambda ref: (refs.append(ref), lambda: "k")[1]
    )
    monkeypatch.setattr(
        cli,
        "make_oa_critic_client",
        lambda **kw: (lambda prompt, **p: {}),
    )

    from stigmergy.rig import resolve_rig

    resolved = resolve_rig(RIG, rigs_root=rigs_root)
    try:
        daemon = _build_daemon(resolved)
        try:
            daemon._weaver.critic._client("x")  # force the key-provider build
        finally:
            daemon._store.close()
    finally:
        resolved.store.close()
    # _build_daemon builds providers for BOTH the critic and the relay —
    # and critic + relay SHARE the same op item today
    # (_CRITIC_KEY_REF == _RELAY_KEY_REF, the .64/.25 rig-00 item), so the
    # ref appears twice. Assert: the critic ref rode the default path, and
    # the env var was NOT consulted for an Anthropic-routed entry.
    assert refs.count(cli._CRITIC_KEY_REF) == 2
    assert "op://shelly/Synthetic key/credential" not in refs


def test_143_non_anthropic_critic_entry_requires_key_ref_env(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """A non-Anthropic-routed critic entry (oa_provider_key !=
    "anthropic") resolves its op ref via `STIGMERGY_CRITIC_OA_KEY_REF`; the env
    missing -> a LOUD rig-launch failure (the `daemon run` path surfaces
    it as exit 1 with stderr naming the env var), NOT a first-gate infra
    trip (Decision 3)."""
    rigs_root = scaffold_rig(tmp_path)
    # registry: point the critic at a synthetic entry with explicit OA wiring
    models_path = rigs_root / RIG / "models.toml"
    models_path.write_text(
        models_path.read_text()
        + """
[kimi3]
provider = "synthetic"
family = "kimi"
version = "hf:moonshotai/Kimi-K3"
pricing = "subscription"
marginal_usd = 0.0
quota = "synthetic-2500req-5h"
oa_provider_key = "synthetic"
oa_base_url = "https://api.synthetic.new/openai/v1"
oa_type = "openai"
"""
    )
    _set_charter_critic_section(rigs_root, model="kimi3")
    monkeypatch.delenv("STIGMERGY_CRITIC_OA_KEY_REF", raising=False)
    monkeypatch.setattr(cli, "make_op_key_provider", lambda ref: (lambda: "k"))
    monkeypatch.setattr(
        cli,
        "make_oa_critic_client",
        lambda **kw: (_ for _ in ()).throw(AssertionError("factory must not be reached")),
    )

    rc = main(["daemon", "run", "--rig", RIG, "--rigs-root", str(rigs_root)])
    # construction fails BEFORE a daemon exists, so `daemon run` returns 1:
    assert rc == 1
    err = capsys.readouterr().err
    assert "STIGMERGY_CRITIC_OA_KEY_REF" in err


def test_143_non_anthropic_critic_key_ref_env_is_used(
    tmp_path: Path, monkeypatch
) -> None:
    """WITH `STIGMERGY_CRITIC_OA_KEY_REF` set, the non-Anthropic entry's key
    provider is built from THAT ref (Decision 3's hook)."""
    rigs_root = scaffold_rig(tmp_path)
    models_path = rigs_root / RIG / "models.toml"
    models_path.write_text(
        models_path.read_text()
        + """
[kimi3]
provider = "synthetic"
family = "kimi"
version = "hf:moonshotai/Kimi-K3"
pricing = "subscription"
marginal_usd = 0.0
quota = "synthetic-2500req-5h"
oa_provider_key = "synthetic"
oa_base_url = "https://api.synthetic.new/openai/v1"
oa_type = "openai"
"""
    )
    _set_charter_critic_section(rigs_root, model="kimi3")
    monkeypatch.setenv("STIGMERGY_CRITIC_OA_KEY_REF", "op://shelly/Synthetic key/credential")
    refs: list[str] = []
    monkeypatch.setattr(
        cli, "make_op_key_provider", lambda ref: (refs.append(ref), lambda: "k")[1]
    )
    recorded: list[dict] = []

    def fake_oa_client(**kwargs):
        recorded.append(kwargs)
        return (lambda prompt, **p: {})

    monkeypatch.setattr(cli, "make_oa_critic_client", fake_oa_client)
    recorded_daemons: list = []
    monkeypatch.setattr(cli, "_run_daemon", recorded_daemons.append)

    rc = main(["daemon", "run", "--rig", RIG, "--rigs-root", str(rigs_root)])
    assert rc == 0
    # The relay's _RELAY_KEY_REF also rides make_op_key_provider at
    # _build_daemon; assert the critic's env-resolved ref specifically.
    assert refs.count("op://shelly/Synthetic key/credential") == 1
    assert len(recorded) == 1
    if recorded_daemons:
        recorded_daemons[0]._store.close()
