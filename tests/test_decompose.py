"""Tests for the decompose station driver (bead workspace-e2uh.152,
Decision 17 — decomposer band automation, HITL at the edges only).

Two deliverables under test:

1. `intake.ingest_manifest` — the manifest-ingestion core extracted out of
   `cli._cmd_intake` (bead workspace-e2uh.152, deliverable 1). The CLI
   stays byte-identical: its existing tests in `tests/test_cli.py` pass
   UNMODIFIED (that is the proof of extraction fidelity), and the tests
   here pin the direct-function contract — (inserted_ids, errors) return
   shape, read-only-until-first-insertion, and the exact message strings
   (minus the `stigmergy intake:` prefix the CLI adds).

2. `stigmergy.decompose` — the deterministic host-side driver that turns an
   operator spec into a seeded, machine-validated, critic-cleared ticket DAG
   in ONE command (render task -> `openalph exec` decomposer -> classify
   manifest/phase_plan/escape/failure -> deterministic validator ->
   validation critic STATION (an ephemeral agent exec, Decision 18) ->
   bounded edit-in-place repair loop -> per-phase fan-out -> intake +
   auto-approve -> DECOMPOSE provenance events).

ALL stub-driven: no network, no real `openalph exec`, no real critic
sessions. The seams are the `decompose` module's `_run_exec` (decomposer
exec) and `_run_critic_exec` (critic station exec) subprocess wrappers;
rigs are real tmp scaffolds (real git clone, real SQLite store) so
intake/approval/records run on production code paths.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

import stigmergy.cli as cli
from stigmergy import decompose
from stigmergy.intake import ingest_manifest
from stigmergy.rig import RigStore, create_rig, resolve_rig

FIXTURES = Path(__file__).parent / "fixtures"
VALID_CHARTER_PATH = FIXTURES / "charter_valid.toml"
MODELS_REGISTRY_PATH = FIXTURES / "models.toml"
BASE_CHARTER_TOML = VALID_CHARTER_PATH.read_text()
RIG = "shipyard"

_GIT_IDENTITY = ["-c", "user.email=test@example.com", "-c", "user.name=Test User"]


# ==========================================================================
# fixtures (real git clone + real create_rig; mirrors tests/test_cli.py's
# own helper style so the rig scaffolds are the production shape)
# ==========================================================================


def make_local_repo_with_prompts(tmp_path: Path, name: str = "source_repo") -> Path:
    """A local git repo carrying the prompt artifacts the rig resolves
    from its clone (`prompts/` at the repo root per the fixture charter's
    `[prompts].dir`) plus a few source files the validator's R8 scope
    checks can resolve against (the decompose manifests target these)."""
    repo_dir = tmp_path / name
    (repo_dir / "prompts").mkdir(parents=True)
    (repo_dir / "prompts" / "code01").write_text("code01 template\n")
    (repo_dir / "prompts" / "critic01").write_text("critic01 template\n")
    (repo_dir / "prompts" / "decomposer01").write_text("decomposer01 template\n")
    (repo_dir / "prompts" / "decomposecritic01").write_text("decomposecritic01 template\n")
    (repo_dir / "src").mkdir()
    (repo_dir / "src" / "app.py").write_text("def app():\n    return 1\n")
    (repo_dir / "src" / "util.py").write_text("def util():\n    return 2\n")
    (repo_dir / "tests").mkdir()
    (repo_dir / "tests" / "test_app.py").write_text("def test_app():\n    assert True\n")
    (repo_dir / "README.md").write_text("hello from the fixture repo\n")
    subprocess.run(["git", "init", "-q", str(repo_dir)], check=True)
    subprocess.run(["git", *_GIT_IDENTITY, "-C", str(repo_dir), "add", "-A"], check=True)
    subprocess.run(
        ["git", *_GIT_IDENTITY, "-C", str(repo_dir), "commit", "-q", "-m", "initial commit"],
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


def _ticket(rigs_root: Path, ticket_id: str) -> dict | None:
    resolved = resolve_rig(RIG, rigs_root=rigs_root)
    try:
        return resolved.store.get_ticket(ticket_id)
    finally:
        resolved.store.close()


def _entry(tid: str, **overrides: Any) -> dict[str, Any]:
    """A minimal valid intake entry (the intake key vocabulary, not the
    stricter decomposer contract — intake does not require
    tier1_checks)."""
    entry = {
        "id": tid,
        "title": f"Ticket {tid}",
        "functional_summary": f"Operator-facing summary for {tid}.",
        "acceptance_criteria": ["it works"],
        "target_scope": ["src/"],
    }
    entry.update(overrides)
    return entry


def _entry_missing(tid: str, *keys: str) -> dict[str, Any]:
    """A valid intake entry with ``keys`` DELETED — the missing-required-key
    error fixtures (overrides alone cannot remove a helper-filled key)."""
    entry = _entry(tid)
    for key in keys:
        del entry[key]
    return entry


# ==========================================================================
# deliverable 1 — `intake.ingest_manifest` extraction
# ==========================================================================


@pytest.fixture
def store(tmp_path: Path) -> RigStore:
    s = RigStore.create(tmp_path / "tickets.db")
    yield s
    s.close()


def test_ingest_manifest_inserts_and_wires_deps(store: RigStore) -> None:
    manifest = [
        _entry("t-a"),
        _entry("t-b", blocks=["t-a"]),
    ]
    inserted, errors = ingest_manifest(store, manifest)
    assert errors == []
    assert inserted == ["t-a", "t-b"]
    assert store.get_ticket("t-a") is not None
    assert store.get_ticket("t-b") is not None
    assert store.deps_of("t-b") == ["t-a"]


def test_ingest_manifest_allows_predecessor_defined_later(store: RigStore) -> None:
    manifest = [
        _entry("t-later", blocks=["t-earlier"]),
        _entry("t-earlier"),
    ]
    inserted, errors = ingest_manifest(store, manifest)
    assert errors == []
    assert inserted == ["t-later", "t-earlier"]
    assert store.deps_of("t-later") == ["t-earlier"]


def test_ingest_manifest_allows_store_predecessor(store: RigStore) -> None:
    store.add_ticket(id="t-existing", title="Existing", functional_summary="x")
    manifest = [_entry("t-new", blocks=["t-existing"])]
    inserted, errors = ingest_manifest(store, manifest)
    assert errors == []
    assert inserted == ["t-new"]
    assert store.deps_of("t-new") == ["t-existing"]


def test_ingest_manifest_read_only_until_first_insertion_on_error(
    store: RigStore,
) -> None:
    """Entry-level validation is a full first pass: a later bad entry means
    NOTHING was inserted (the earlier good entries are not half-seeded)."""
    manifest = [
        _entry("t-good"),
        _entry_missing("t-bad", "acceptance_criteria", "target_scope"),
    ]
    inserted, errors = ingest_manifest(store, manifest)
    assert inserted == []
    assert len(errors) == 1
    assert (
        errors[0]
        == "manifest entry 1 (t-bad) missing required key(s): acceptance_criteria, target_scope"
    )
    assert store.get_ticket("t-good") is None


def test_ingest_manifest_error_messages_are_exact(store: RigStore) -> None:
    """The returned error strings are the CLI's message bodies verbatim
    (the CLI prefixes `stigmergy intake: ` and prints to stderr) — pin
    every branch's exact wording so the extraction cannot drift."""
    # non-object entry
    _, errs = ingest_manifest(store, ["not-a-dict"])
    assert errs == ["manifest entry 0 must be a JSON object (got str)"]

    # missing keys (sorted, joined)
    _, errs = ingest_manifest(store, [_entry_missing("t-m", "acceptance_criteria", "target_scope")])
    assert errs == [
        "manifest entry 0 (t-m) missing required key(s): acceptance_criteria, target_scope"
    ]

    # missing id in the entry -> the <index N> label
    no_id = _entry("t-noid")
    del no_id["id"]
    _, errs = ingest_manifest(store, [no_id])
    assert errs == ["manifest entry 0 (<index 0>) missing required key(s): id"]

    # empty functional_summary
    _, errs = ingest_manifest(store, [_entry("t-es", functional_summary="   ")])
    assert errs == [
        "manifest entry 0 (t-es) 'functional_summary' must be a non-empty string "
        "(got '   ')"
    ]

    # duplicate id within the manifest
    _, errs = ingest_manifest(store, [_entry("t-dup"), _entry("t-dup")])
    assert errs == ["manifest entry 0 (t-dup) has duplicate id"]

    # collision with an existing store ticket
    store.add_ticket(id="t-store", title="Store", functional_summary="x")
    _, errs = ingest_manifest(store, [_entry("t-store")])
    assert errs == ["manifest entry 0 (t-store) ticket id already exists"]

    # unresolved blocks reference
    _, errs = ingest_manifest(store, [_entry("t-u", blocks=["t-ghost"])])
    assert errs == ["manifest entry 0 (t-u) blocks unresolved predecessor: t-ghost"]

    # Read-only on error: `ingest_manifest` itself inserted nothing (the only
    # row in the store is `t-store`, which THIS test seeded directly).
    ids = {t["id"] for t in store.list_tickets()}
    assert ids == {"t-store"}


def test_cli_intake_delegates_and_stays_byte_identical(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The `intake` CLI's observable behavior (stdout/stderr/exit code) is
    byte-identical before and after the extraction to
    `intake.ingest_manifest` — the existing tests/test_cli.py intake tests
    passing UNMODIFIED are the broader proof; this pins one success and one
    failure byte-for-byte."""
    rigs_root = scaffold_rig(tmp_path)

    manifest = [_entry("t-cli-a"), _entry("t-cli-b", blocks=["t-cli-a"])]
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest))

    rc = cli.main(
        ["intake", "--rig", RIG, "--rigs-root", str(rigs_root), "--manifest", str(manifest_file)]
    )
    out, err = capsys.readouterr()
    assert rc == 0
    assert out == "t-cli-a\nt-cli-b\n"
    assert err == ""
    assert _ticket(rigs_root, "t-cli-a") is not None
    assert _ticket(rigs_root, "t-cli-b") is not None

    # failure case: exact stderr byte-for-byte, rc 1, nothing seeded
    capsys.readouterr()
    bad = [_entry("t-ok"), _entry_missing("t-bad2", "acceptance_criteria", "target_scope")]
    bad_file = tmp_path / "bad.json"
    bad_file.write_text(json.dumps(bad))
    rc = cli.main(
        ["intake", "--rig", RIG, "--rigs-root", str(rigs_root), "--manifest", str(bad_file)]
    )
    out, err = capsys.readouterr()
    assert rc == 1
    assert out == ""
    assert err == (
        "stigmergy intake: manifest entry 1 (t-bad2) "
        "missing required key(s): acceptance_criteria, target_scope\n"
    )
    assert _ticket(rigs_root, "t-ok") is None


# ==========================================================================
# deliverable 2 — the decompose driver
# ==========================================================================

@pytest.fixture(autouse=True)
def _hermetic_station(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Decision-18 hermeticity for EVERY test in this module: the agent-TOML
    preflight is pinned at an existing dummy file and the critic-exec seam
    carries a DEFAULT accept response — so no test depends on the host's OA
    config dir or the real `openalph` binary. Tests that need specific
    critic behavior override the seam via `_critic_exec` (test-level setattr
    runs after this fixture and wins)."""
    agent_toml = tmp_path / "hermetic-agent.toml"
    agent_toml.write_text("[agent]\nname = 'stigmergy-decomposer'\n", encoding="utf-8")
    monkeypatch.setattr(decompose, "_DECOMP_AGENT_TOML", agent_toml)

    def _default_critic_exec(
        argv: list[str], env: dict[str, str], timeout: float
    ) -> subprocess.CompletedProcess:
        line = _critic_exec_result()
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(line) + "\n", stderr="")

    monkeypatch.setattr(decompose, "_run_critic_exec", _default_critic_exec)


SPEC_TEXT = (
    "Build a CLI decompose station for the shipyard rig. Deliverables: a "
    "rendering module and a runner module."
)


def _ticket_entry(
    tid: str,
    *,
    scope: list[str] | None = None,
    blocks: list[str] | None = None,
    reading: list[str] | None = None,
) -> dict[str, Any]:
    """A validator-clean ticket entry (the decomposer contract:
    tier1_checks verbatim from the fixture charter's [checks.*], existing
    scope paths, existing required_reading)."""
    entry = {
        "id": tid,
        "title": f"Ticket {tid}",
        "functional_summary": f"Operator-facing summary for {tid}.",
        "acceptance_criteria": [
            f"src/app.py contains a function named {tid.replace('-', '_')}",
            f"tests/test_app.py asserts the {tid} behavior",
        ],
        "tier1_checks": {
            "pytest": "pytest -x -q",
            "lint": "ruff check .",
        },
        "target_scope": scope if scope is not None else ["src/app.py", "tests/test_app.py"],
    }
    if blocks is not None:
        entry["blocks"] = blocks
    if reading is not None:
        entry["required_reading"] = reading
    return entry


def _exec_result(
    *,
    status: str = "done",
    deny_reason: str | None = None,
    ceiling_trip: str | None = None,
    usage: dict | None = None,
    detail: str | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "content": "decomposer says done",
        "usage": usage if usage is not None else {"in": 10, "cached": 2, "out": 30, "reasoning": 0},
        "stop_reason": "end_turn",
        "ceiling_trip": ceiling_trip,
        "deny_reason": deny_reason,
        "tool_trace": [],
        "detail": detail or "",
    }


def _fake_exec(
    monkeypatch: pytest.MonkeyPatch,
    script: list[Any],
    *,
    env_probe: list[dict[str, str]] | None = None,
    preseed_log: list[tuple[str | None, str | None]] | None = None,
) -> list[list[str]]:
    """Monkeypatch the decompose module's `_run_exec` seam (the DECOMPOSER's
    exec; the critic STATION's exec is the `_run_critic_exec` seam, faked
    by `_critic_exec` in the same style) with a scripted fake. Each script
    item is one of:

    - a callable ``(argv, env) -> CompletedProcess``;
    - an Exception instance to raise;
    - a dict of workspace writes: ``{"manifest": <list|str>, "notes": str}``
      plus an optional ``"result"`` key (defaults to a clean done result);
      the fake writes the manifest/notes into the ``--task-file``'s sibling
      dir (the scratch dir the task names) and returns the result line.

    Records every argv (for the count/shape assertions) and every child env
    (for the proxy-removal + STIG_DECOMP_KEY assertions). The LAST script
    item repeats if more attempts occur than items (a stable final state).
    """
    calls: list[list[str]] = []
    cursor = 0

    def _next() -> Any:
        nonlocal cursor
        item = script[min(cursor, len(script) - 1)]
        cursor += 1
        return item

    def fake(argv: list[str], env: dict[str, str], timeout: float) -> subprocess.CompletedProcess:
        calls.append(list(argv))
        if env_probe is not None:
            env_probe.append(dict(env))
        item = _next()
        if callable(item):
            return item(argv, env)
        if isinstance(item, BaseException):
            raise item
        result_obj = item.get("result")
        if result_obj is None:
            result_obj = _exec_result()
        task_idx = argv.index("--task-file") + 1
        scratch = Path(argv[task_idx]).parent
        # Snapshot the scratch state at EXEC ENTRY (before this round's
        # writes) — the only observable record of what the driver
        # pre-seeded (Decision 18 edit-in-place pre-seeding).
        if preseed_log is not None:
            m = scratch / "manifest.json"
            n = scratch / "notes.md"
            preseed_log.append(
                (
                    m.read_text(encoding="utf-8") if m.exists() else None,
                    n.read_text(encoding="utf-8") if n.exists() else None,
                )
            )
        if item.get("manifest") is not None:
            text = item["manifest"]
            (scratch / "manifest.json").write_text(
                text if isinstance(text, str) else json.dumps(text), encoding="utf-8"
            )
        if item.get("notes") is not None:
            (scratch / "notes.md").write_text(item["notes"], encoding="utf-8")
        return subprocess.CompletedProcess(
            argv, 0, stdout=json.dumps(result_obj) + "\n", stderr=""
        )

    monkeypatch.setattr(decompose, "_run_exec", fake)
    monkeypatch.setattr(decompose, "_decomp_key_provider", lambda: "stub-decomp-key")
    return calls


def _critic_exec_result(
    *,
    verdict: str = "accept",
    findings: list[dict] | None = None,
    evidence_log: list[dict] | None = None,
    summary: str = "manifest is faithful to the spec",
    usage: dict | None = None,
) -> dict[str, Any]:
    """One critic STATION exec stdout JSON line (Decision 18): a `done`
    status whose top-level `result` field IS the submit_validation payload
    (the OA terminal-tool mechanism)."""
    return {
        "status": "done",
        "content": "critic says done",
        "usage": usage if usage is not None else {"in": 5, "cached": 1, "out": 9, "reasoning": 0},
        "stop_reason": "end_turn",
        "ceiling_trip": None,
        "deny_reason": None,
        "tool_trace": [],
        "detail": "",
        "result": {
            "verdict": verdict,
            "summary": summary,
            "findings": findings if findings is not None else [],
            "evidence_log": evidence_log if evidence_log is not None else [],
        },
    }




def _resolve(rigs_root: Path) -> Any:
    return resolve_rig(RIG, rigs_root=rigs_root)


def _write_spec(tmp_path: Path, text: str = SPEC_TEXT) -> Path:
    spec = tmp_path / "spec.md"
    spec.write_text(text, encoding="utf-8")
    return spec


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    script: list[Any],
    critic_script: list[Any] | None = None,
    env_probe: list[dict[str, str]] | None = None,
    **kwargs: Any,
) -> tuple[int, list[list[str]], list[list[str]], Path]:
    """Scaffold a rig + spec, stub the exec + critic seams, and call
    `decompose.run_decompose` (NO critic_factory — Decision 18: the
    driver owns the critic exec invocation). Returns
    `(rc, exec_calls, critic_argvs, run_dir)` where `run_dir` is the
    single subdir under the decompose root."""
    rigs_root = scaffold_rig(tmp_path)
    spec = _write_spec(tmp_path, kwargs.pop("spec_text", SPEC_TEXT))
    decompose_root = tmp_path / "decompose-root"
    # Hermetic preflight: pin the agent-TOML path at an existing dummy file
    # so the suite never depends on the host's OA config dir (the dedicated
    # missing-TOML test overrides this with a nonexistent path).
    agent_toml = tmp_path / "agent.toml"
    agent_toml.write_text("[agent]\nname = 'stigmergy-decomposer'\n", encoding="utf-8")
    monkeypatch.setattr(decompose, "_DECOMP_AGENT_TOML", agent_toml)
    critic_script = critic_script if critic_script is not None else [_critic_exec_result()]
    exec_calls = _fake_exec(monkeypatch, script, env_probe=env_probe)
    critic_argvs, _critic_scratches = _critic_exec(monkeypatch, critic_script, env_probe=env_probe)
    rc = decompose.run_decompose(
        rig_name=RIG,
        spec_path=spec,
        decompose_root=decompose_root,
        rigs_root=rigs_root,
        **kwargs,
    )
    run_dirs = list((decompose_root / "decompose").iterdir())
    return rc, exec_calls, critic_argvs, run_dirs[0]


def _decompose_events(rigs_root: Path) -> list[dict[str, Any]]:
    resolved = _resolve(rigs_root)
    try:
        from stigmergy.records import RecordPlane

        events = RecordPlane(resolved.rig_paths["records_dir"]).read_events()
    finally:
        resolved.store.close()
    return [e for e in events if e.get("event_type") == "decompose"]


# --- render_task -----------------------------------------------------------


def test_render_task_initial_has_required_sections() -> None:
    text = decompose.render_task(
        mode="initial",
        spec=SPEC_TEXT,
        repo_root="/tmp/rigs/demo/repo",
        charter_path="/tmp/rigs/demo/charter.toml",
        manifest_path="/tmp/stig-decomposer/ws/abc/manifest.json",
        notes_path="/tmp/stig-decomposer/ws/abc/notes.md",
    )
    assert "<spec>" in text and SPEC_TEXT in text and "</spec>" in text
    assert "/tmp/rigs/demo/repo" in text
    assert "/tmp/rigs/demo/charter.toml" in text
    assert "/tmp/stig-decomposer/ws/abc/manifest.json" in text
    assert "/tmp/stig-decomposer/ws/abc/notes.md" in text


def test_render_task_repair_carries_edit_in_place_rules_verbatim() -> None:
    """The repair task carries the load-bearing edit-in-place rules pinned
    VERBATIM (Decision 18) and points at the on-disk previous files (paths
    — their full text is NOT embedded; the round's scratch dir is
    pre-seeded by the driver)."""
    text = decompose.render_task(
        mode="repair",
        spec=SPEC_TEXT,
        repo_root="/repo",
        charter_path="/charter.toml",
        manifest_path="/scratch/manifest.json",
        notes_path="/scratch/notes.md",
        previous_manifest_path="/scratch/manifest.json",
        previous_notes_path="/scratch/notes.md",
        findings=["fidelity: ticket a misses the CLI deliverable"],
        round_no=1,
    )
    # The load-bearing edit-in-place repair rules, pinned VERBATIM
    # (Decision 18).
    assert (
        "The files already exist in your workspace — edit them in place "
        "with file_edit/file_patch. Change only what the findings require; "
        "leave every other byte identical. Keep every ticket id stable "
        "unless a finding demands a rename. Address every finding; do not "
        "weaken criteria to make findings disappear. After editing, "
        "file_read your output and verify it. Manifest lines are "
        "independent JSON objects (JSONL) — a damaged line is fixed "
        "line-wise."
    ) in text
    # The on-disk previous files are named by PATH; the previous text is
    # NOT embedded (edit-in-place, not re-emit-both-files).
    assert "/scratch/manifest.json" in text
    assert "/scratch/notes.md" in text
    assert "[{" not in text
    # The findings still ride along.
    assert "fidelity: ticket a misses the CLI deliverable" in text



def test_render_task_phase_carries_prior_manifests_and_real_ids() -> None:
    prior = [
        {
            "id": "phase-1",
            "manifest_text": '[{"id": "real-ticket-a"}]',
            "ticket_ids": ["real-ticket-a"],
        }
    ]
    text = decompose.render_task(
        mode="phase",
        spec="ignored for phase mode",
        repo_root="/repo",
        charter_path="/charter.toml",
        manifest_path="/scratch/manifest.json",
        notes_path="/scratch/notes.md",
        phase={
            "id": "phase-2",
            "title": "The runner",
            "goal": "Implement the runner module and its tests.",
            "depends_on": ["phase-1"],
            "done_condition": "runner tests green",
        },
        prior_phases=prior,
    )
    # The phase's goal brief is THIS session's spec.
    assert "<spec>" in text
    assert "Implement the runner module and its tests." in text
    assert "phase-2" in text
    # Prior phase's manifest verbatim + its now-real ticket ids.
    assert '[{"id": "real-ticket-a"}]' in text
    assert "real-ticket-a" in text
    # The cross-phase blocks instruction.
    assert "Cross-phase `blocks` edges MUST reference these real ticket ids" in text


def test_render_task_modes_are_distinct() -> None:
    common = dict(
        spec=SPEC_TEXT,
        repo_root="/repo",
        charter_path="/charter.toml",
        manifest_path="/scratch/manifest.json",
        notes_path="/scratch/notes.md",
    )
    initial = decompose.render_task(mode="initial", **common)
    assert "## Findings" not in initial
    assert "Prior phases" not in initial
    phase = decompose.render_task(
        mode="phase",
        phase={"id": "p1", "title": "t", "goal": "g", "depends_on": [], "done_condition": "d"},
        **common,
    )
    assert "## Phase: p1" in phase


# --- detect_kind ------------------------------------------------------------


def test_detect_kind_manifest() -> None:
    assert decompose.detect_kind([_ticket_entry("a"), _ticket_entry("b")]) == "manifest"


def test_detect_kind_phase_plan() -> None:
    phases = [
        {"id": "p1", "goal": "g", "done_condition": "d", "depends_on": []},
        {"id": "p2", "goal": "g", "done_condition": "d", "depends_on": ["p1"]},
    ]
    assert decompose.detect_kind(phases) == "phase_plan"


def test_detect_kind_mixed_raises() -> None:
    with pytest.raises(ValueError, match="mixed or unrecognized"):
        decompose.detect_kind([
            _ticket_entry("a"),
            {"id": "p1", "goal": "g", "done_condition": "d", "depends_on": []},
        ])


def test_detect_kind_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        decompose.detect_kind([])


def test_detect_kind_neither_raises() -> None:
    with pytest.raises(ValueError, match="mixed or unrecognized"):
        decompose.detect_kind([{"id": "a", "title": "x"}])


# --- run_decomposer classification ------------------------------------------


def _rd_common(tmp_path: Path) -> dict[str, Any]:
    return dict(
        model="stub-model",
        effort="xhigh",
        prompts_dir=tmp_path / "prompts",
        scratch_dir=tmp_path / "ws",
    )


def test_run_decomposer_manifest_classification(tmp_path: Path, monkeypatch) -> None:
    calls = _fake_exec(
        monkeypatch, [{"manifest": [_ticket_entry("a")], "notes": "notes"}]
    )
    out = decompose.run_decomposer("task", **_rd_common(tmp_path))
    assert out.kind == "manifest"
    assert out.manifest == [_ticket_entry("a")]
    assert out.notes_text == "notes"
    assert out.usage == {"in": 10, "cached": 2, "out": 30, "reasoning": 0}
    assert len(calls) == 1


def test_run_decomposer_phase_plan_classification(tmp_path: Path, monkeypatch) -> None:
    phases = [
        {"id": "p1", "goal": "g", "done_condition": "d", "depends_on": []},
        {"id": "p2", "goal": "g", "done_condition": "d", "depends_on": ["p1"]},
    ]
    calls = _fake_exec(monkeypatch, [{"manifest": phases}])
    out = decompose.run_decomposer("task", **_rd_common(tmp_path))
    assert out.kind == "phase_plan"
    assert out.manifest == phases
    assert len(calls) == 1


def test_run_decomposer_escape_not_retried(tmp_path: Path, monkeypatch) -> None:
    """The no-manifest escape (notes with a diagnosis) is legitimate output
    — NEVER retried: exactly ONE exec invocation."""
    calls = _fake_exec(monkeypatch, [{"notes": "The spec is contradictory: A and B."}])
    out = decompose.run_decomposer("task", **_rd_common(tmp_path))
    assert out.kind == "escape"
    assert out.manifest is None
    assert "contradictory" in out.notes_text
    assert out.detail["reason"] == "no-manifest-escape"
    assert len(calls) == 1  # exactly one exec invocation — no retry


def test_run_decomposer_failure_retried_once(tmp_path: Path, monkeypatch) -> None:
    """A failure (no output at all) gets ONE retry: exactly 2 exec
    invocations, then a failure result."""
    calls = _fake_exec(monkeypatch, [{}, {}])  # two no-output attempts
    out = decompose.run_decomposer("task", **_rd_common(tmp_path))
    assert out.kind == "failure"
    assert out.detail["reason"] == "no-output"
    assert len(calls) == 2


def test_run_decomposer_failure_then_success(tmp_path: Path, monkeypatch) -> None:
    """A first-attempt failure followed by a clean manifest: 2 invocations,
    final kind manifest (the retry worked)."""
    calls = _fake_exec(monkeypatch, [{}, {"manifest": [_ticket_entry("a")], "notes": "n"}])
    out = decompose.run_decomposer("task", **_rd_common(tmp_path))
    assert out.kind == "manifest"
    assert len(calls) == 2


def test_run_decomposer_deny_reason_is_failure(tmp_path: Path, monkeypatch) -> None:
    """A `done` status carrying a deny_reason is an EXEC FAILURE (never an
    escape) — the kdsn.305 distinction."""
    calls = _fake_exec(
        monkeypatch,
        [
            {"result": _exec_result(deny_reason="quota-tokens")},
            {"result": _exec_result(deny_reason="quota-tokens")},
        ],
    )
    out = decompose.run_decomposer("task", **_rd_common(tmp_path))
    assert out.kind == "failure"
    assert out.detail["reason"] == "exec-deny-or-ceiling"
    assert out.detail["deny_reason"] == "quota-tokens"
    assert len(calls) == 2


def test_run_decomposer_ceiling_trip_is_failure(tmp_path: Path, monkeypatch) -> None:
    calls = _fake_exec(
        monkeypatch,
        [
            {"result": _exec_result(ceiling_trip="driver_turns")},
            {"result": _exec_result(ceiling_trip="driver_turns")},
        ],
    )
    out = decompose.run_decomposer("task", **_rd_common(tmp_path))
    assert out.kind == "failure"
    assert out.detail["reason"] == "exec-deny-or-ceiling"
    assert out.detail["ceiling_trip"] == "driver_turns"
    assert len(calls) == 2


def test_run_decomposer_nonzero_exit_is_failure(tmp_path: Path, monkeypatch) -> None:
    def argv_proc(argv: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(argv, 3, stdout="", stderr="boom")

    calls = _fake_exec(monkeypatch, [argv_proc, argv_proc])
    out = decompose.run_decomposer("task", **_rd_common(tmp_path))
    assert out.kind == "failure"
    assert out.detail["reason"] == "exec-nonzero-exit"
    assert out.detail["exit"] == 3
    assert len(calls) == 2


def test_run_decomposer_unparseable_stdout_is_failure(tmp_path: Path, monkeypatch) -> None:
    def argv_proc(argv: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(argv, 0, stdout="not json at all\n", stderr="")

    calls = _fake_exec(monkeypatch, [argv_proc, argv_proc])
    out = decompose.run_decomposer("task", **_rd_common(tmp_path))
    assert out.kind == "failure"
    assert out.detail["reason"] == "exec-unparseable-stdout"
    assert len(calls) == 2


def test_run_decomposer_not_done_status_is_failure(tmp_path: Path, monkeypatch) -> None:
    calls = _fake_exec(
        monkeypatch,
        [
            {"result": _exec_result(status="infra", detail="upstream 500")},
            {"result": _exec_result(status="infra", detail="upstream 500")},
        ],
    )
    out = decompose.run_decomposer("task", **_rd_common(tmp_path))
    assert out.kind == "failure"
    assert out.detail["reason"] == "exec-not-done"
    assert len(calls) == 2


def test_run_decomposer_garbage_manifest_is_failure(tmp_path: Path, monkeypatch) -> None:
    """A manifest file present but empty JSON array / not a list -> failure
    (retried once), even when notes exist (NOT an escape — a manifest was
    attempted)."""
    calls = _fake_exec(
        monkeypatch,
        [
            {"manifest": "not json", "notes": "n"},
            {"manifest": "not json", "notes": "n"},
        ],
    )
    out = decompose.run_decomposer("task", **_rd_common(tmp_path))
    assert out.kind == "failure"
    assert out.detail["reason"] == "manifest-unparseable-json"
    assert len(calls) == 2


def test_run_decomposer_exec_argv_contract(tmp_path: Path, monkeypatch) -> None:
    """The argv is the exact `openalph exec` contract: agent, task-file,
    system-prompt-file (prompts_dir/decomposer01), model, effort, tools."""
    calls = _fake_exec(monkeypatch, [{"manifest": [_ticket_entry("a")]}])
    decompose.run_decomposer("task", **_rd_common(tmp_path))
    argv = calls[0]
    assert "exec" in argv
    assert argv[argv.index("--agent") + 1] == "stigmergy-decomposer"
    assert argv[argv.index("--task-file") + 1].endswith("task.md")
    assert argv[argv.index("--system-prompt-file") + 1].endswith(
        str(Path("prompts") / "decomposer01")
    )
    assert argv[argv.index("--model") + 1] == "stub-model"
    assert argv[argv.index("--effort") + 1] == "xhigh"
    assert argv[argv.index("--tools") + 1] == "file_read,glob,grep,file_write"


def test_run_decomposer_child_env_proxy_unset_and_key_set(
    tmp_path: Path, monkeypatch
) -> None:
    """The child env: the four proxy vars UNSET, STIG_DECOMP_KEY present
    (fetched from the provider seam, never logged)."""
    monkeypatch.setenv("HTTP_PROXY", "http://proxy:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy:1")
    monkeypatch.setenv("http_proxy", "http://proxy:1")
    monkeypatch.setenv("https_proxy", "http://proxy:1")
    envs: list[dict[str, str]] = []
    _fake_exec(monkeypatch, [{"manifest": [_ticket_entry("a")]}], env_probe=envs)
    decompose.run_decomposer("task", **_rd_common(tmp_path))
    assert len(envs) == 1
    env = envs[0]
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        assert var not in env
    assert env["STIG_DECOMP_KEY"] == "stub-decomp-key"



# --- evidence bundle ---------------------------------------------------------


def test_build_evidence_bundle_sections_and_exact_strings(tmp_path: Path) -> None:
    from stigmergy.charter import load_charter

    charter = load_charter(VALID_CHARTER_PATH, env={})
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "app.py").write_text("x")
    bundle = decompose.build_evidence_bundle(
        charter=charter,
        repo_root=repo,
        context_root=tmp_path / "ctx",
        manifest=[
            _ticket_entry("a", scope=["src/app.py", "src/new_mod.py"]),
        ],
        validator_report=[],
    )
    # Charter named checks with VERBATIM cmds.
    assert "pytest: pytest -x -q" in bundle
    assert "lint: ruff check ." in bundle
    # Gates + lanes + budgets.
    assert "attempt: ['lint', 'pytest']" in bundle
    assert "staging: ['lint', 'pytest']" in bundle
    assert "cheap: model=haiku" in bundle
    assert "budgets.dispatches: 50" in bundle
    assert "concurrency.workers: 1" in bundle
    # The EXACT target_scope resolution strings.
    assert "a: src/app.py -> exists" in bundle
    assert "a: src/new_mod.py -> new-file (parent exists)" in bundle
    # Bounded tree with the target marked.
    assert "*src/app.py" in bundle
    # Validator report line.
    assert "clean (0 defects)" in bundle
    # Deterministic: same inputs -> same bytes.
    again = decompose.build_evidence_bundle(
        charter=charter,
        repo_root=repo,
        context_root=tmp_path / "ctx",
        manifest=[_ticket_entry("a", scope=["src/app.py", "src/new_mod.py"])],
        validator_report=[],
    )
    assert bundle == again


# --- run_decompose: exit codes, intake, approval, provenance ----------------


def test_run_decompose_happy_path_seeds_and_approves(tmp_path: Path, monkeypatch) -> None:
    """End-to-end happy path on a tmp rig with stubbed exec+critic: spec ->
    2-ticket DAG seeded + approved + DECOMPOSE events appended + summary.md
    written, rc 0."""
    manifest = [
        _ticket_entry("t-base", scope=["src/app.py", "tests/test_app.py"]),
        _ticket_entry(
            "t-cli",
            scope=["src/util.py"],
            blocks=["t-base"],
        ),
    ]
    rigs_root_holder: dict[str, Any] = {}

    def with_rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        rigs_root = scaffold_rig(tmp_path)
        rigs_root_holder["rr"] = rigs_root
        spec = _write_spec(tmp_path)
        decompose_root = tmp_path / "decompose-root"
        critic_script = [_critic_exec_result(verdict="accept")]
        exec_calls = _fake_exec(monkeypatch, [{"manifest": manifest, "notes": "n"}])
        critic_argvs, _critic_scratches = _critic_exec(monkeypatch, critic_script)
        rc = decompose.run_decompose(
            rig_name=RIG,
            spec_path=spec,
            decompose_root=decompose_root,
            rigs_root=rigs_root,
        )
        return rc, exec_calls, critic_argvs, decompose_root, rigs_root

    rc, exec_calls, critic_argvs, decompose_root, rigs_root = with_rig(tmp_path, monkeypatch)
    assert rc == 0
    assert len(exec_calls) == 1  # one decomposer session, no repairs
    assert len(critic_argvs) == 1  # one critic station exec

    # The 2-ticket DAG is seeded with the dependency wired.
    t_base = _ticket(rigs_root, "t-base")
    t_cli = _ticket(rigs_root, "t-cli")
    assert t_base is not None and t_cli is not None
    resolved = _resolve(rigs_root)
    try:
        assert resolved.store.deps_of("t-cli") == ["t-base"]
    finally:
        resolved.store.close()

    # Both tickets APPROVED with the decompose-band attribution.
    assert t_base["approved"] == 1
    assert t_cli["approved"] == 1
    assert t_base["approval_hash"]

    # Approval events carry agent=merry + operator_session=decompose-<run_id>.
    resolved = _resolve(rigs_root)
    try:
        from stigmergy.records import RecordPlane

        events = RecordPlane(resolved.rig_paths["records_dir"]).read_events()
    finally:
        resolved.store.close()
    approvals = [e for e in events if e.get("event_type") == "approval"]
    assert len(approvals) == 2
    for ev in approvals:
        assert ev["acting_agent"] == "merry"
        assert ev["operator_session"].startswith("decompose-")
        assert ev["outcome"] == "approved"
        assert ev["approval_hash"]

    # DECOMPOSE provenance: exactly one (the critic) + one decomposer event.
    d_events = [e for e in events if e.get("event_type") == "decompose"]
    assert len(d_events) == 2  # one decomposer + one critic
    stations = {e["station"] for e in d_events}
    assert stations == {"decomposer", "decompose-critic"}

    # summary.md written.
    run_dir = list((decompose_root / "decompose").iterdir())[0]
    assert (run_dir / "summary.md").exists()
    summary = (run_dir / "summary.md").read_text()
    assert "t-base" in summary and "t-cli" in summary


def test_run_decompose_no_approve_leaves_unapproved(tmp_path: Path, monkeypatch) -> None:
    manifest = [_ticket_entry("t-only")]
    rigs_root = scaffold_rig(tmp_path)
    spec = _write_spec(tmp_path)
    agent_toml = tmp_path / "agent.toml"
    agent_toml.write_text("[agent]\nname = 'stigmergy-decomposer'\n", encoding="utf-8")
    monkeypatch.setattr(decompose, "_DECOMP_AGENT_TOML", agent_toml)
    _fake_exec(monkeypatch, [{"manifest": manifest, "notes": "n"}])
    _critic_exec(monkeypatch, [_critic_exec_result()])
    rc = decompose.run_decompose(
        rig_name=RIG,
        spec_path=spec,
        decompose_root=tmp_path / "d",
        rigs_root=rigs_root,
        no_approve=True,
    )
    assert rc == 0
    t = _ticket(rigs_root, "t-only")
    assert t is not None
    assert t["approved"] == 0  # stays unapproved/pool


def test_run_decompose_dry_run_intakes_nothing(tmp_path: Path, monkeypatch, capsys) -> None:
    rigs_root = scaffold_rig(tmp_path)
    spec = _write_spec(tmp_path)
    _fake_exec(monkeypatch, [{"manifest": [_ticket_entry("t-dry")], "notes": "n"}])
    rc = decompose.run_decompose(
        rig_name=RIG,
        spec_path=spec,
        decompose_root=tmp_path / "d",
        rigs_root=rigs_root,
        dry_run=True,
    )
    assert rc == 0
    assert _ticket(rigs_root, "t-dry") is None  # nothing intaked
    out = capsys.readouterr().out
    assert "would seed t-dry" in out


def test_run_decompose_escape_is_exit_1(tmp_path: Path, monkeypatch, capsys) -> None:
    """A no-manifest escape -> rc 1, stderr names the run dir, summary
    records the diagnosis, and NO ticket is seeded."""
    rigs_root = scaffold_rig(tmp_path)
    spec = _write_spec(tmp_path)
    critic_script = [_critic_exec_result()]
    exec_calls = _fake_exec(
        monkeypatch, [{"notes": "The spec is contradictory between A and B."}]
    )
    critic_argvs, _critic_scratches = _critic_exec(monkeypatch, critic_script)
    rc = decompose.run_decompose(
        rig_name=RIG,
        spec_path=spec,
        decompose_root=tmp_path / "d",
        rigs_root=rigs_root,
    )
    assert rc == 1
    assert len(exec_calls) == 1  # escape never retried
    err = capsys.readouterr().err
    assert "decompose:" in err
    # No critic call (no manifest to judge).
    assert len(critic_argvs) == 0
    # No tickets seeded.
    assert _resolve(rigs_root).store.list_tickets() == [] or True
    resolved = _resolve(rigs_root)
    try:
        assert resolved.store.list_tickets() == []
    finally:
        resolved.store.close()


# --- run_decompose: repair loop ---------------------------------------------


def _defective_manifest() -> list[dict[str, Any]]:
    """A manifest with ONE R6 defect (a non-kebab id) — validator-defective
    but otherwise shaped like a ticket manifest."""
    m = [_ticket_entry("t-good"), _ticket_entry("Bad_ID")]
    return m


def test_run_decompose_validator_defect_repairs_without_critic(
    tmp_path: Path, monkeypatch
) -> None:
    """Validator defects -> a repair round WITHOUT a critic call (mechanics
    first). The repaired (clean) manifest is then judged once by the
    critic."""
    good = [_ticket_entry("t-good")]
    rigs_root = scaffold_rig(tmp_path)
    spec = _write_spec(tmp_path)
    critic_script = [_critic_exec_result()]
    _fake_exec(
        monkeypatch,
        [
            {"manifest": _defective_manifest(), "notes": "n"},  # round 0: defective
            {"manifest": good, "notes": "n2"},  # repair round: clean
        ],
    )
    critic_argvs, _critic_scratches = _critic_exec(monkeypatch, critic_script)
    rc = decompose.run_decompose(
        rig_name=RIG,
        spec_path=spec,
        decompose_root=tmp_path / "d",
        rigs_root=rigs_root,
    )
    assert rc == 0
    assert _ticket(rigs_root, "t-good") is not None
    # The critic was called exactly ONCE — only after the repair produced a
    # validator-clean manifest (never over the schema-broken one).
    assert len(critic_argvs) == 1


def test_run_decompose_clean_manifest_critic_once(tmp_path: Path, monkeypatch) -> None:
    manifest = [_ticket_entry("t-a")]
    rigs_root = scaffold_rig(tmp_path)
    spec = _write_spec(tmp_path)
    agent_toml = tmp_path / "agent.toml"
    agent_toml.write_text("[agent]\nname = 'stigmergy-decomposer'\n", encoding="utf-8")
    monkeypatch.setattr(decompose, "_DECOMP_AGENT_TOML", agent_toml)
    exec_calls = _fake_exec(monkeypatch, [{"manifest": manifest, "notes": "n"}])
    critic_argvs, _critic_scratches = _critic_exec(
        monkeypatch, [_critic_exec_result()]
    )
    rc = decompose.run_decompose(
        rig_name=RIG,
        spec_path=spec,
        decompose_root=tmp_path / "d",
        rigs_root=rigs_root,
    )
    assert rc == 0
    assert len(critic_argvs) == 1  # exactly one critic call
    assert len(exec_calls) == 1


def test_run_decompose_accept_with_minor_proceeds(tmp_path: Path, monkeypatch) -> None:
    """verdict=accept with only MINOR findings -> no repair round (minors
    ride along recorded)."""
    manifest = [_ticket_entry("t-a")]
    minor = [
        {"aspect": "notes", "severity": "minor", "tickets": ["t-a"],
         "evidence": "thin", "direction": "add a note"}
    ]
    rigs_root = scaffold_rig(tmp_path)
    spec = _write_spec(tmp_path)
    critic_script = [
        _critic_exec_result(verdict="accept", findings=minor)
    ]
    exec_calls = _fake_exec(monkeypatch, [{"manifest": manifest, "notes": "n"}])
    critic_argvs, _critic_scratches = _critic_exec(monkeypatch, critic_script)
    rc = decompose.run_decompose(
        rig_name=RIG,
        spec_path=spec,
        decompose_root=tmp_path / "d",
        rigs_root=rigs_root,
    )
    assert rc == 0
    assert len(exec_calls) == 1  # no repair
    assert len(critic_argvs) == 1  # one critic call
    assert _ticket(rigs_root, "t-a") is not None


def test_run_decompose_verdict_repair_triggers_second_round(
    tmp_path: Path, monkeypatch
) -> None:
    """A `repair` verdict (with a major finding) triggers a repair round;
    the repaired manifest is re-judged and accepted."""
    bad = [
        _ticket_entry("t-a"),
    ]
    good = [_ticket_entry("t-a")]
    major = [
        {"aspect": "fidelity", "severity": "major", "tickets": ["t-a"],
         "evidence": "misses the CLI", "direction": "add it"}
    ]
    rigs_root = scaffold_rig(tmp_path)
    spec = _write_spec(tmp_path)
    critic_script = [
        _critic_exec_result(verdict="repair", findings=major),  # round 0
        _critic_exec_result(verdict="accept"),  # after repair
    ]
    exec_calls = _fake_exec(
        monkeypatch,
        [
            {"manifest": bad, "notes": "n"},
            {"manifest": good, "notes": "n2"},
        ],
    )
    critic_argvs, _critic_scratches = _critic_exec(monkeypatch, critic_script)
    rc = decompose.run_decompose(
        rig_name=RIG,
        spec_path=spec,
        decompose_root=tmp_path / "d",
        rigs_root=rigs_root,
    )
    assert rc == 0
    assert len(exec_calls) == 2  # initial + one repair
    assert len(critic_argvs) == 2  # two critic calls (round 0 + after repair)
    assert _ticket(rigs_root, "t-a") is not None


def test_run_decompose_non_convergence_fails_loud(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Problem count flat after a repair round -> fail loud immediately
    (exit 1), WITHOUT spending the next repair."""
    manifest = [_ticket_entry("t-a")]
    major = [
        {"aspect": "fidelity", "severity": "major", "tickets": ["t-a"],
         "evidence": "misses the CLI", "direction": "add it"}
    ]
    rigs_root = scaffold_rig(tmp_path)
    spec = _write_spec(tmp_path)
    # Round 0: repair with 1 major. After repair: SAME 1 major (flat).
    critic_script = [
        _critic_exec_result(verdict="repair", findings=major),
        _critic_exec_result(verdict="repair", findings=major),  # flat -> non-converge
    ]
    exec_calls = _fake_exec(
        monkeypatch,
        [
            {"manifest": manifest, "notes": "n"},
            {"manifest": manifest, "notes": "n2"},
        ],
    )
    critic_argvs, _critic_scratches = _critic_exec(monkeypatch, critic_script)
    rc = decompose.run_decompose(
        rig_name=RIG,
        spec_path=spec,
        decompose_root=tmp_path / "d",
        rigs_root=rigs_root,
        max_repairs=2,
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "non-convergence" in err
    # Only ONE repair was spent (the flat one) — not the second budget slot.
    assert len(exec_calls) == 2  # initial + one repair


def test_run_decompose_budget_exhaustion_carries_findings(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Budget (max_repairs=1) exhausted with findings still standing ->
    exit 1, and the error carries the last findings."""
    manifest = [_ticket_entry("t-a")]
    two_major = [
        {"aspect": "sizing", "severity": "major", "tickets": ["t-a"],
         "evidence": "too big", "direction": "split"},
        {"aspect": "fidelity", "severity": "major", "tickets": ["t-a"],
         "evidence": "misses CLI", "direction": "add it"},
    ]
    one_major = [
        {"aspect": "sizing", "severity": "major", "tickets": ["t-a"],
         "evidence": "still big", "direction": "split more"}
    ]
    rigs_root = scaffold_rig(tmp_path)
    spec = _write_spec(tmp_path)
    # round 0: 2 major, then round 1: 1 major (strictly decreased, budget gone)
    critic_script = [
        _critic_exec_result(verdict="repair", findings=two_major),
        _critic_exec_result(verdict="repair", findings=one_major),
    ]
    exec_calls = _fake_exec(
        monkeypatch,
        [
            {"manifest": manifest, "notes": "n"},
            {"manifest": manifest, "notes": "n2"},
            {"manifest": manifest, "notes": "n3"},
        ],
    )
    critic_argvs, _critic_scratches = _critic_exec(monkeypatch, critic_script)
    rc = decompose.run_decompose(
        rig_name=RIG,
        spec_path=spec,
        decompose_root=tmp_path / "d",
        rigs_root=rigs_root,
        max_repairs=1,
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "budget exhausted" in err
    # initial + exactly max_repairs(1) repair = 2 execs.
    assert len(exec_calls) == 2


def test_run_decompose_exec_failure_after_retry_is_exit_2(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """An exec that fails both attempts (denied) -> exit 2, distinct from
    the escape's exit 1."""
    rigs_root = scaffold_rig(tmp_path)
    spec = _write_spec(tmp_path)
    critic_script = [_critic_exec_result()]
    exec_calls = _fake_exec(
        monkeypatch,
        [
            {"result": _exec_result(deny_reason="quota-tokens")},
            {"result": _exec_result(deny_reason="quota-tokens")},
        ],
    )
    critic_argvs, _critic_scratches = _critic_exec(monkeypatch, critic_script)
    rc = decompose.run_decompose(
        rig_name=RIG,
        spec_path=spec,
        decompose_root=tmp_path / "d",
        rigs_root=rigs_root,
    )
    assert rc == 2
    assert len(exec_calls) == 2
    err = capsys.readouterr().err
    assert "exec failed" in err
    assert len(critic_argvs) == 0  # no critic call


# --- run_decompose: provenance ----------------------------------------------


def test_run_decompose_provenance_event_shape(tmp_path: Path, monkeypatch) -> None:
    """ONE DECOMPOSE event per SUCCESSFUL LLM invocation, with the exact
    field shape: station, attempt_kind='decompose', 4-key tokens,
    computed_usd 0.0 for a subscription model, the prompt hashes, and the
    right dispatch_id/attempt/rig fields."""
    manifest = [_ticket_entry("t-a")]
    decomp_usage = {"in": 11, "cached": 2, "out": 33, "reasoning": 0}
    rigs_root = scaffold_rig(tmp_path)
    spec = _write_spec(tmp_path)

    critic_usage = {"in": 4, "cached": 0, "out": 6, "reasoning": 0}
    critic_script = [
        _critic_exec_result(usage=critic_usage),
    ]
    _fake_exec(
        monkeypatch,
        [{"manifest": manifest, "notes": "n",
          "result": _exec_result(usage=decomp_usage)}],
    )
    agent_toml = tmp_path / "agent.toml"
    agent_toml.write_text("[agent]\nname = 'stigmergy-decomposer'\n", encoding="utf-8")
    monkeypatch.setattr(decompose, "_DECOMP_AGENT_TOML", agent_toml)
    _critic_exec(monkeypatch, critic_script)
    rc = decompose.run_decompose(
        rig_name=RIG,
        spec_path=spec,
        decompose_root=tmp_path / "d",
        rigs_root=rigs_root,
        decomposer_model="claude-max-sub",  # also subscription -> 0.0
    )
    assert rc == 0
    events = _decompose_events(rigs_root)
    assert len(events) == 2
    by_station = {e["station"]: e for e in events}
    assert set(by_station) == {"decomposer", "decompose-critic"}
    for ev in events:
        # The exact DECOMPOSE field shape.
        assert ev["attempt_kind"] == "decompose"
        assert ev["attempt"] == 0
        assert ev["ticket"] is None
        assert ev["rung"] is None
        assert ev["worker"] is None
        assert ev["model_version"] is None
        assert ev["approval_hash"] is None
        assert ev["image_digest"] is None
        assert ev["charter_hash"]
        assert ev["price_table_version"]
        assert ev["wall_time_seconds"] >= 0.0
        # 4-key tokens, matching the invocation's usage verbatim.
        assert set(ev["tokens"]) == {"in", "cached", "out", "reasoning"}
        # Hash-bearing prompt artifact (SPEC §4).
        assert isinstance(ev["prompt_artifact_hash"], str)
        assert len(ev["prompt_artifact_hash"]) == 64
        # dispatch_id carries the run id (UTC compact + 8 hex of spec sha).
        assert ev["dispatch_id"].startswith("decompose-")
    decomp = by_station["decomposer"]
    assert decomp["tokens"] == decomp_usage
    assert decomp["model"] == "claude-max-sub"
    # Subscription pricing -> computed_usd exactly 0.0 (never
    # "unbudgetable", never a fabricated positive).
    assert decomp["computed_usd"] == 0.0
    critic = by_station["decompose-critic"]
    assert critic["tokens"] == critic_usage
    # The critic event's model is the CHARTER's critic model (the
    # registry-priced entry, here the metered `opus`), priced via
    # spend.cost_usd for its usage.
    assert critic["model"] == "opus"
    resolved = _resolve(rigs_root)
    try:
        import hashlib

        from stigmergy import spend as spend_mod

        expected_decomp_hash = hashlib.sha256(
            (resolved.rig_paths["prompts_dir"] / "decomposer01").read_bytes()
        ).hexdigest()
        expected_critic_usd = spend_mod.cost_usd(
            resolved.registry.resolve("opus"), critic_usage
        )
    finally:
        resolved.store.close()
    assert decomp["prompt_artifact_hash"] == expected_decomp_hash
    assert critic["computed_usd"] == expected_critic_usd


def test_run_decompose_failed_exec_emits_no_event(tmp_path: Path, monkeypatch) -> None:
    """A failed exec (both attempts) emits NO decomposer DECOMPOSE event
    (nothing invoked successfully) — and no critic event (no manifest)."""
    rigs_root = scaffold_rig(tmp_path)
    spec = _write_spec(tmp_path)
    critic_script = [_critic_exec_result()]
    _fake_exec(
        monkeypatch,
        [
            {"result": _exec_result(deny_reason="quota-tokens")},
            {"result": _exec_result(deny_reason="quota-tokens")},
        ],
    )
    critic_argvs, _critic_scratches = _critic_exec(monkeypatch, critic_script)
    rc = decompose.run_decompose(
        rig_name=RIG,
        spec_path=spec,
        decompose_root=tmp_path / "d",
        rigs_root=rigs_root,
    )
    assert rc == 2
    assert _decompose_events(rigs_root) == []
    assert len(critic_argvs) == 0


def test_run_decompose_dispatch_id_phase_and_round_suffixes(
    tmp_path: Path, monkeypatch
) -> None:
    """Phase fan-out + a repair round -> dispatch_id suffixes:
    `-phase-<id>` for phase sessions and `-r<round>` for repair rounds."""
    plan = [
        {"id": "p1", "title": "one", "goal": "g1", "depends_on": [],
         "done_condition": "d"},
        {"id": "p2", "title": "two", "goal": "g2", "depends_on": ["p1"],
         "done_condition": "d"},
    ]
    p1_manifest = [_ticket_entry("t-p1", scope=["src/app.py", "tests/test_app.py"])]
    p2_manifest = [_ticket_entry("t-p2", scope=["src/util.py"], blocks=["t-p1"])]
    rigs_root = scaffold_rig(tmp_path)
    spec = _write_spec(tmp_path)
    critic_script = [
        _critic_exec_result(),  # p1 critic
        _critic_exec_result(),  # p2 critic
    ]
    _fake_exec(
        monkeypatch,
        [
            {"manifest": plan},            # initial (root) -> phase plan
            {"manifest": p1_manifest, "notes": "n1"},  # phase p1
            {"manifest": p2_manifest, "notes": "n2"},  # phase p2
        ],
    )
    # Decision 18: the critic is ALSO an exec station — stub its seam +
    # pin the agent-TOML preflight hermetically.
    agent_toml = tmp_path / "agent.toml"
    agent_toml.write_text("[agent]\nname = 'stigmergy-decomposer'\n", encoding="utf-8")
    monkeypatch.setattr(decompose, "_DECOMP_AGENT_TOML", agent_toml)
    _critic_exec(monkeypatch, critic_script)
    rc = decompose.run_decompose(
        rig_name=RIG,
        spec_path=spec,
        decompose_root=tmp_path / "d",
        rigs_root=rigs_root,
    )
    assert rc == 0
    events = _decompose_events(rigs_root)
    ids = [e["dispatch_id"] for e in events]
    # Concretely: the two phase sessions carry -phase-p1 / -phase-p2, and
    # the root initial session's id has no phase/round suffix.
    assert any("-phase-p1" in i for i in ids)
    assert any("-phase-p2" in i for i in ids)
    root_ids = [i for i in ids if "-phase-" not in i and "-r" not in i]
    assert len(root_ids) == 1  # the root initial session only
    assert root_ids[0].startswith("decompose-")
    # One decomposer event per session (3: root + 2 phases) + 2 critics.
    decomp = [e for e in events if e["station"] == "decomposer"]
    assert len(decomp) == 3
    assert len([e for e in events if e["station"] == "decompose-critic"]) == 2


# --- phase fan-out ------------------------------------------------------------


def _two_phase_plan() -> list[dict[str, Any]]:
    return [
        {"id": "p2", "title": "two", "goal": "Build the runner module.",
         "depends_on": ["p1"], "done_condition": "runner tests green"},
        {"id": "p1", "title": "one", "goal": "Build the base module.",
         "depends_on": [], "done_condition": "base tests green"},
    ]


def test_phase_fanout_topo_order_and_real_ids(tmp_path: Path, monkeypatch) -> None:
    """Phase order is topological (p2 depends on p1, so p1's task runs
    first) EVEN when the plan lists p2 first; and the p2 task carries p1's
    NOW-REAL ticket ids (intake happened between the two sessions)."""
    p1_manifest = [_ticket_entry("t-p1", scope=["src/app.py", "tests/test_app.py"])]
    p2_manifest = [_ticket_entry("t-p2", scope=["src/util.py"], blocks=["t-p1"])]
    rigs_root = scaffold_rig(tmp_path)
    spec = _write_spec(tmp_path)
    critic_script = [
        _critic_exec_result(),  # p1 critic
        _critic_exec_result(),  # p2 critic
    ]
    _fake_exec(
        monkeypatch,
        [
            {"manifest": _two_phase_plan()},
            {"manifest": p1_manifest, "notes": "n1"},  # p1 (first, topo)
            {"manifest": p2_manifest, "notes": "n2"},  # p2 (second)
        ],
    )
    agent_toml = tmp_path / "agent.toml"
    agent_toml.write_text("[agent]\nname = 'stigmergy-decomposer'\n", encoding="utf-8")
    monkeypatch.setattr(decompose, "_DECOMP_AGENT_TOML", agent_toml)
    _critic_exec(monkeypatch, critic_script)
    rc = decompose.run_decompose(
        rig_name=RIG,
        spec_path=spec,
        decompose_root=tmp_path / "d",
        rigs_root=rigs_root,
    )
    assert rc == 0

    # Both phases' tickets seeded, cross-phase edge wired by the phase-2
    # session (validated by the validator's store-aware R7).
    t_p1 = _ticket(rigs_root, "t-p1")
    t_p2 = _ticket(rigs_root, "t-p2")
    assert t_p1 is not None and t_p2 is not None
    resolved = _resolve(rigs_root)
    try:
        assert resolved.store.deps_of("t-p2") == ["t-p1"]
    finally:
        resolved.store.close()

    # Per-phase intake happened BEFORE the next phase decomposed: the p2
    # task (the 3rd exec's task file) cites p1's real ticket id. The task
    # artifacts are in the run dir; the p2 task is the later one.
    run_dir = list((tmp_path / "d" / "decompose").iterdir())[0]
    tasks = sorted(run_dir.glob("task-*.md"), key=lambda p: int(p.stem.split("-")[1]))
    p2_task_text = max(tasks, key=lambda p: int(p.stem.split("-")[1]))
    text = p2_task_text.read_text()
    assert "real-ticket" not in text
    assert "t-p1" in text  # p1's REAL ticket id rides the p2 task
    assert "Cross-phase `blocks` edges MUST reference these real ticket ids" in text
    # And the p2 task is a phase-mode task (p1's goal brief would be the
    # p1 session, not p2).
    assert "p2" in text


def test_phase_fanout_cycle_is_error(tmp_path: Path, monkeypatch, capsys) -> None:
    plan = [
        {"id": "p1", "title": "a", "goal": "g", "depends_on": ["p2"],
         "done_condition": "d"},
        {"id": "p2", "title": "b", "goal": "g", "depends_on": ["p1"],
         "done_condition": "d"},
    ]
    rigs_root = scaffold_rig(tmp_path)
    spec = _write_spec(tmp_path)
    critic_script = [_critic_exec_result()]
    _fake_exec(monkeypatch, [{"manifest": plan}])
    critic_argvs, _critic_scratches = _critic_exec(monkeypatch, critic_script)
    rc = decompose.run_decompose(
        rig_name=RIG,
        spec_path=spec,
        decompose_root=tmp_path / "d",
        rigs_root=rigs_root,
    )
    assert rc == 1
    assert "cycle" in capsys.readouterr().err
    assert len(critic_argvs) == 0  # no critic: the plan defect fails before any phase


def test_phase_fanout_unknown_dep_is_error(tmp_path: Path, monkeypatch, capsys) -> None:
    plan = [
        {"id": "p1", "title": "a", "goal": "g", "depends_on": ["p-ghost"],
         "done_condition": "d"},
    ]
    rigs_root = scaffold_rig(tmp_path)
    spec = _write_spec(tmp_path)
    critic_script = [_critic_exec_result()]
    _fake_exec(monkeypatch, [{"manifest": plan}])
    critic_argvs, _critic_scratches = _critic_exec(monkeypatch, critic_script)
    rc = decompose.run_decompose(
        rig_name=RIG,
        spec_path=spec,
        decompose_root=tmp_path / "d",
        rigs_root=rigs_root,
    )
    assert rc == 1
    assert "unknown phase" in capsys.readouterr().err
    assert len(critic_argvs) == 0


# --- CLI wiring ---------------------------------------------------------------


def test_cli_decompose_verb_wired(monkeypatch) -> None:
    """The `decompose` subparser exists with all the documented flags and
    dispatches to `run_decompose` (monkeypatched) with the parsed args
    threaded through (rig name, spec path, every flag). The rig resolution
    + critic build are stubbed (this test pins the dispatch, not the
    resolver)."""
    seen: dict[str, Any] = {}

    def fake_run_decompose(*args: Any, **kwargs: Any) -> int:
        seen.update(kwargs)
        return 0

    # Stub the shared rig resolver (the command goes through
    # _resolve_rig_or_none -> cli.resolve_rig) and the critic builder. The
    # stub resolved rig carries a closable store (the command's `finally`
    # closes it, mirroring every other C+D verb).
    class _StubResolved:
        class _Store:
            def close(self) -> None:
                pass

        store = _Store()

    monkeypatch.setattr(
        cli, "resolve_rig", lambda name, rigs_root=None: _StubResolved()
    )
    monkeypatch.setattr(cli, "run_decompose", fake_run_decompose)
    rc = cli.main([
        "decompose",
        "--rig", "shipyard",
        "--spec", "/tmp/spec.md",
        "--no-approve",
        "--max-repairs", "3",
        "--decomposer-model", "some/model",
        "--decomposer-effort", "medium",
        "--dry-run",
    ])
    assert rc == 0
    assert seen["rig_name"] == "shipyard"
    assert str(seen["spec_path"]) == "/tmp/spec.md"
    assert seen["no_approve"] is True
    assert seen["max_repairs"] == 3
    assert seen["decomposer_model"] == "some/model"
    assert seen["decomposer_effort"] == "medium"
    assert seen["dry_run"] is True


def test_cli_decompose_default_flags() -> None:
    """The decompose subparser's defaults (max-repairs 2, the default
    decomposer model, effort xhigh, no --no-approve/--dry-run)."""
    parser = cli._build_parser()
    args = parser.parse_args(["decompose", "--rig", "r", "--spec", "s.md"])
    assert args.no_approve is False
    assert args.max_repairs == 2
    assert args.decomposer_model == "synthetic/hf:moonshotai/Kimi-K3"
    assert args.decomposer_effort == "xhigh"
    assert args.dry_run is False


def test_cli_decompose_effort_choices_rejected() -> None:
    """--decomposer-effort is a closed vocabulary (none/low/medium/xhigh) —
    argparse rejects anything else (the choices are pinned by the valid
    values accepted in test_cli_decompose_verb_wired)."""
    parser = cli._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["decompose", "--rig", "r", "--spec", "s.md", "--decomposer-effort", "huge"]
        )
    for ok in ("none", "low", "medium", "xhigh"):
        args = parser.parse_args(
            ["decompose", "--rig", "r", "--spec", "s.md", "--decomposer-effort", ok]
        )
        assert args.decomposer_effort == ok


def test_cli_decompose_usage_lists_the_verb() -> None:
    assert "decompose" in cli._USAGE


def test_preflight_agent_toml_missing_fails_loud(tmp_path: Path, monkeypatch, capsys) -> None:
    """A missing decomposer agent TOML fails BEFORE any exec (bead .152:
    the station is templated at src/stigmergy/agents/ and installed into
    OA's CONFIG_DIR; a hand-placed-only artifact is the archaeology
    failure). The stderr line carries the install command with the exact
    source + destination paths, and NO LLM call is burned on 'agent not
    found'."""
    rigs_root = scaffold_rig(tmp_path)
    spec = _write_spec(tmp_path)
    missing = tmp_path / "nowhere" / "stigmergy-decomposer.toml"
    monkeypatch.setattr(decompose, "_DECOMP_AGENT_TOML", missing)
    rc = decompose.run_decompose(
        rig_name=RIG,
        spec_path=spec,
        decompose_root=tmp_path / "decompose-root",
        rigs_root=rigs_root,
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "stigmergy decompose: decomposer agent TOML missing" in err
    assert str(missing) in err
    assert "src/stigmergy/agents/stigmergy-decomposer.toml" in err


def test_cli_decompose_exit_codes_surfaced(monkeypatch) -> None:
    """The CLI surfaces run_decompose's return code verbatim (0/1/2)."""
    # Stub rig resolution + the critic builder (this test pins the
    # return-code passthrough, not the resolver or the critic) so the
    # command reaches the monkeypatched run_decompose — same stub shape as
    # test_cli_decompose_verb_wired. The stub resolved rig carries a
    # closable store (the command's `finally` closes it).
    class _StubResolved:
        class _Store:
            def close(self) -> None:
                pass

        store = _Store()

    monkeypatch.setattr(
        cli, "resolve_rig", lambda name, rigs_root=None: _StubResolved()
    )
    # The stub's return value lives in a stable dict (identity constant,
    # contents mutated per iteration) so the closure is B023-safe: the call
    # is synchronous within each iteration, after the mutation.
    box: dict[str, int] = {"code": 0}

    def _fake_rc(*_a: Any, **_k: Any) -> int:
        return box["code"]

    monkeypatch.setattr(cli, "run_decompose", _fake_rc)
    for code in (0, 1, 2):
        box["code"] = code
        rc = cli.main(["decompose", "--rig", RIG, "--spec", "/tmp/spec.md"])
        assert rc == code


# ==========================================================================
# Decision 18 — the Station Contract (three reworks of the driver)
#
# CHANGE 1: the decomposer manifest is JSON Lines (one object per line);
#   the driver's JSON-array parsing is backward compatibility.
# CHANGE 2: repair rounds are edit-in-place (the round's scratch dir is
#   PRE-SEEDED with the previous round's files; the repair exec carries the
#   edit-capable tool set; the task points at the on-disk files).
# CHANGE 3: the validation critic is an EXEC STATION (an ephemeral agent
#   invoked exactly like the decomposer, submitting through the terminal
#   `submit_validation` tool — its payload is the exec's `result` field).
#   The in-process critic path (DecomposeCritic /
#   build_decompose_critic_prompt / critic_factory / cli._build_decompose_critic)
#   is GONE — the tests of that deleted machinery are removed with it.
# ==========================================================================


def _jsonl(entries: list[dict[str, Any]]) -> str:
    """The decomposer01 v0.4 manifest shape: one JSON object per line, no
    array brackets, no comma separators."""
    return "".join(json.dumps(e) + "\n" for e in entries)


# --- _parse_manifest_text (CHANGE 1: JSONL with array backward compat) ------


def test_parse_manifest_text_jsonl_round_trip_matches_array() -> None:
    """Driver-side round-trip: a JSONL manifest parses to the SAME list an
    equivalent JSON array would (the two encodings are interchangeable)."""
    entries = [
        _ticket_entry("t-a", scope=["src/app.py", "tests/test_app.py"]),
        _ticket_entry("t-b", scope=["src/util.py"], blocks=["t-a"]),
    ]
    array = json.loads(json.dumps(entries))  # the equivalent array, round-tripped
    assert decompose._parse_manifest_text(_jsonl(entries)) == array
    assert decompose._parse_manifest_text(json.dumps(entries, indent=2)) == array


def test_parse_manifest_text_array_backward_compat() -> None:
    """A first non-whitespace `[` parses as a JSON array (existing
    artifacts stay valid)."""
    entries = [_ticket_entry("t-a")]
    for prefix in ("", "\n  ", "  \n\t"):
        assert decompose._parse_manifest_text(prefix + json.dumps(entries)) == entries


def test_parse_manifest_text_bad_jsonl_line_is_defect_with_line_number() -> None:
    """A JSONL line that fails to parse is a DEFECT naming the line number
    (it feeds the normal repair loop — the content arrived, it is malformed
    content, never an exec-level failure)."""
    text = _jsonl([_ticket_entry("t-a")]) + "{broken json\n"
    with pytest.raises(ValueError, match="line 2"):
        decompose._parse_manifest_text(text)


def test_parse_manifest_text_jsonl_non_object_line_is_defect() -> None:
    """Every JSONL line must be a complete JSON OBJECT — a valid JSON line
    that is not an object is a defect naming its line number."""
    text = json.dumps(_ticket_entry("t-a")) + "\n" + '"just a string"\n'
    with pytest.raises(ValueError, match="line 2"):
        decompose._parse_manifest_text(text)


def test_parse_manifest_text_jsonl_skips_blank_lines() -> None:
    text = _jsonl([_ticket_entry("t-a")]).replace("}\n", "}\n\n  \n", 1)
    assert decompose._parse_manifest_text(text) == [_ticket_entry("t-a")]


def test_parse_manifest_text_empty_is_defect() -> None:
    with pytest.raises(ValueError, match="empty"):
        decompose._parse_manifest_text("")
    with pytest.raises(ValueError, match="empty"):
        decompose._parse_manifest_text("  \n\t\n")


def test_run_decomposer_jsonl_manifest_classification(tmp_path: Path, monkeypatch) -> None:
    """A JSONL manifest written by the (v0.4) decomposer is classified as a
    manifest (one object per line, detect_kind over the resulting list)."""
    entries = [
        _ticket_entry("t-a", scope=["src/app.py", "tests/test_app.py"]),
        _ticket_entry("t-b", scope=["src/util.py"], blocks=["t-a"]),
    ]
    calls = _fake_exec(
        monkeypatch, [{"manifest": _jsonl(entries), "notes": "jsonl notes"}]
    )
    out = decompose.run_decomposer("task", **_rd_common(tmp_path))
    assert out.kind == "manifest"
    assert out.manifest == entries
    assert out.notes_text == "jsonl notes"
    assert len(calls) == 1


def test_run_decomposer_jsonl_phase_plan_classification(tmp_path: Path, monkeypatch) -> None:
    """Phase plans ride the SAME JSONL parsing (detect_kind is the
    discriminator over the resulting list)."""
    plan = [
        {"id": "p1", "goal": "g", "done_condition": "d", "depends_on": []},
        {"id": "p2", "goal": "g", "done_condition": "d", "depends_on": ["p1"]},
    ]
    calls = _fake_exec(monkeypatch, [{"manifest": _jsonl(plan)}])
    out = decompose.run_decomposer("task", **_rd_common(tmp_path))
    assert out.kind == "phase_plan"
    assert out.manifest == plan
    assert len(calls) == 1


def test_run_decomposer_broken_jsonl_line_is_defect_not_exec_failure(
    tmp_path: Path, monkeypatch
) -> None:
    """A damaged JSONL line is a CLASSIFICATION DEFECT (the line number is
    named) feeding the normal repair loop — NOT an exec-level failure: the
    content arrived, it is malformed content. The driver gives it the
    normal one-retry (fresh exec), exactly like any other failure."""
    bad = _jsonl([_ticket_entry("t-a")]) + "{broken json\n"
    good = [_ticket_entry("t-a")]
    calls = _fake_exec(monkeypatch, [{"manifest": bad, "notes": "n"},
                                     {"manifest": good, "notes": "n2"}])
    out = decompose.run_decomposer("task", **_rd_common(tmp_path))
    assert out.kind == "manifest"
    assert len(calls) == 2  # one retry, then clean

    # The defect path names the damaged line (a fresh single-line failure).
    single = _jsonl([_ticket_entry("t-a")]) + "{broken\n"
    calls2 = _fake_exec(monkeypatch, [{"manifest": single, "notes": "n"}])
    out2 = decompose.run_decomposer("task", **_rd_common(tmp_path))
    assert out2.kind == "failure"
    assert "line 2" in str(out2.detail.get("error", ""))
    assert len(calls2) == 2


# --- render_task repair mode: edit-in-place (CHANGE 2) ----------------------


def test_render_task_repair_edit_in_place_points_at_on_disk_files() -> None:
    """The repair task points at the PREVIOUS FILES ON DISK (paths), not
    their embedded text, and carries the updated pinned edit-in-place
    rules VERBATIM."""
    text = decompose.render_task(
        mode="repair",
        spec=SPEC_TEXT,
        repo_root="/repo",
        charter_path="/charter.toml",
        manifest_path="/scratch/manifest.json",
        notes_path="/scratch/notes.md",
        previous_manifest_path="/scratch/manifest.json",
        previous_notes_path="/scratch/notes.md",
        findings=["fidelity: ticket a misses the CLI deliverable"],
        round_no=1,
    )
    # The on-disk previous files are named by PATH.
    assert "/scratch/manifest.json" in text
    assert "/scratch/notes.md" in text
    # The previous text is NO LONGER embedded (edit-in-place, not
    # re-emit-both-files).
    assert "[{" not in text
    # The updated pinned rules, VERBATIM (Decision 18).
    assert (
        "The files already exist in your workspace \u2014 edit them in place "
        "with file_edit/file_patch. Change only what the findings require; "
        "leave every other byte identical. Keep every ticket id stable "
        "unless a finding demands a rename. Address every finding; do not "
        "weaken criteria to make findings disappear. After editing, "
        "file_read your output and verify it. Manifest lines are "
        "independent JSON objects (JSONL) \u2014 a damaged line is fixed "
        "line-wise."
    ) in text
    # The findings still ride along.
    assert "fidelity: ticket a misses the CLI deliverable" in text


def test_render_task_initial_output_contract_unchanged() -> None:
    """Initial rounds keep the current write-capable output contract (the
    task text is unchanged by Decision 18 — only the repair mode's wording
    moves to edit-in-place)."""
    text = decompose.render_task(
        mode="initial",
        spec=SPEC_TEXT,
        repo_root="/repo",
        charter_path="/charter.toml",
        manifest_path="/scratch/manifest.json",
        notes_path="/scratch/notes.md",
    )
    assert "MANIFEST" in text and "NOTES" in text
    assert "file_edit" not in text  # no edit-in-place language initially


# --- the critic EXEC station (CHANGE 3) -------------------------------------


def _critic_exec(
    monkeypatch: pytest.MonkeyPatch,
    script: list[Any],
    *,
    env_probe: list[dict[str, str]] | None = None,
) -> tuple[list[list[str]], list[Path]]:
    """A SECOND exec seam dedicated to the critic station: monkeypatches
    `decompose._run_critic_exec` (the driver's critic-exec wrapper) with a
    scripted fake in the SAME `_fake_exec` style — each item is a
    dict-of-writes with a `"result"` key (the exec stdout JSON line) or a
    callable. Returns `(argvs, scratch_dirs)`; `scratch_dirs` is the list
    of the critic scratch dirs the driver passed (for the pre-seed
    assertions)."""
    argvs: list[list[str]] = []
    scratch_dirs: list[Path] = []
    cursor = 0

    def _next() -> Any:
        nonlocal cursor
        item = script[min(cursor, len(script) - 1)]
        cursor += 1
        return item

    def fake(argv: list[str], env: dict[str, str], timeout: float) -> subprocess.CompletedProcess:
        argvs.append(list(argv))
        if env_probe is not None:
            env_probe.append(dict(env))
        task_idx = argv.index("--task-file") + 1
        scratch = Path(argv[task_idx]).parent
        scratch_dirs.append(scratch)
        item = _next()
        if callable(item):
            return item(argv, env)
        if isinstance(item, BaseException):
            raise item
        # Each script item IS a full exec stdout JSON line (see
        # `_critic_exec_result`). Tolerate a legacy wrapper style
        # ({"result": <line>}) for items lacking a status key, but never
        # unwrap a real exec line's top-level `result` payload — that
        # field belongs to the TERMINAL TOOL mechanism, not the seam.
        line = item
        if isinstance(line, dict) and "status" not in line and isinstance(line.get("result"), dict):
            line = line["result"]
        if not (isinstance(line, dict) and "status" in line):
            line = _critic_exec_result()
        for name, content in item.get("writes", {}).items():
            (scratch / name).write_text(content, encoding="utf-8")
        return subprocess.CompletedProcess(
            argv, 0, stdout=json.dumps(line) + "\n", stderr=""
        )

    monkeypatch.setattr(decompose, "_run_critic_exec", fake)
    return argvs, scratch_dirs


def _scaffold_and_patch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    script: list[Any],
    critic_script: list[Any] | None = None,
    env_probe: list[dict[str, str]] | None = None,
    **kwargs: Any,
) -> tuple[int, list[list[str]], list[list[str]], list[Path], Path, Path]:
    """Scaffold the rig + spec, pin the agent TOML, stub BOTH exec seams
    (the decomposer's `_run_exec` and the critic's `_run_critic_exec`), and
    call `decompose.run_decompose` with NO critic_factory (Decision 18: the
    driver owns the exec invocation). Returns
    `(rc, exec_calls, critic_argvs, critic_scratch_dirs, run_dir, rigs_root)`.
    """
    rigs_root = scaffold_rig(tmp_path)
    spec = _write_spec(tmp_path, kwargs.pop("spec_text", SPEC_TEXT))
    decompose_root = tmp_path / "decompose-root"
    agent_toml = tmp_path / "agent.toml"
    agent_toml.write_text("[agent]\nname = 'stigmergy-decomposer'\n", encoding="utf-8")
    monkeypatch.setattr(decompose, "_DECOMP_AGENT_TOML", agent_toml)
    exec_calls = _fake_exec(
        monkeypatch, script, env_probe=env_probe, preseed_log=kwargs.pop("preseed_log", None)
    )
    critic_argvs, critic_scratch_dirs = _critic_exec(
        monkeypatch,
        critic_script if critic_script is not None else [_critic_exec_result()],
        env_probe=env_probe,
    )
    rc = decompose.run_decompose(
        rig_name=RIG,
        spec_path=spec,
        decompose_root=decompose_root,
        rigs_root=rigs_root,
        **kwargs,
    )
    run_dirs = list((decompose_root / "decompose").iterdir())
    return rc, exec_calls, critic_argvs, critic_scratch_dirs, run_dirs[0], rigs_root


def test_critic_exec_argv_contract(tmp_path: Path, monkeypatch) -> None:
    """The critic exec's argv is the Station Contract invocation: the SAME
    agent, the critic task file, system-prompt-file decomposecritic01, the
    registry-QUALIFIED critic model, --effort none, read-only tools, and
    --submit-schema run-dir/submit-schema.json."""
    manifest = [_ticket_entry("t-a")]
    rc, _, critic_argvs, _, run_dir, _rr = _scaffold_and_patch(
        tmp_path,
        monkeypatch,
        script=[{"manifest": manifest, "notes": "n"}],
        critic_script=[_critic_exec_result(verdict="accept")],
    )
    assert rc == 0
    argv = critic_argvs[0]
    assert "exec" in argv
    assert argv[argv.index("--agent") + 1] == "stigmergy-decomposer"
    assert argv[argv.index("--task-file") + 1].endswith("task.md")
    assert argv[argv.index("--system-prompt-file") + 1].endswith("decomposecritic01")
    # The charter's [roles.critic].model ("opus") resolved to the
    # registry-qualified provider/version form.
    assert argv[argv.index("--model") + 1] == "anthropic/opus-4-1-20250805"
    assert argv[argv.index("--effort") + 1] == "none"
    assert argv[argv.index("--tools") + 1] == "file_read,glob,grep"
    schema_arg = argv[argv.index("--submit-schema") + 1]
    assert schema_arg == str(run_dir / "submit-schema.json")
    assert Path(schema_arg).is_file()


def test_critic_submit_schema_file_matches_module_data(
    tmp_path: Path, monkeypatch
) -> None:
    """submit-schema.json is written into the run dir and matches
    `decompose._SUBMIT_VALIDATION_SCHEMA` — the driver-owned data file with
    the EXTENDED required keys verdict/summary/findings/evidence_log."""
    manifest = [_ticket_entry("t-a")]
    rc, _, _, _, run_dir, _rr = _scaffold_and_patch(
        tmp_path,
        monkeypatch,
        script=[{"manifest": manifest, "notes": "n"}],
        critic_script=[_critic_exec_result(verdict="accept")],
    )
    assert rc == 0
    on_disk = json.loads((run_dir / "submit-schema.json").read_text(encoding="utf-8"))
    assert on_disk == decompose._SUBMIT_VALIDATION_SCHEMA
    schema = decompose._SUBMIT_VALIDATION_SCHEMA
    assert schema["name"] == "submit_validation"
    input_schema = schema["input_schema"]
    assert input_schema["required"] == ["verdict", "summary", "findings", "evidence_log"]
    evidence_prop = input_schema["properties"]["evidence_log"]
    assert evidence_prop["type"] == "array"
    items = evidence_prop["items"]
    assert items["type"] == "object"
    assert items["required"] == ["claim_checked", "method", "found"]


def test_critic_scratch_pre_seeded_with_materials(
    tmp_path: Path, monkeypatch
) -> None:
    """The driver PRE-SEEDS the critic's scratch dir with the spec (in the
    task file), the manifest, the notes, the evidence bundle, and the
    validator report — the task file names the on-disk paths."""
    manifest = [_ticket_entry("t-a", scope=["src/app.py", "tests/test_app.py"])]
    rc, _, critic_argvs, critic_scratch_dirs, _run_dir, _rr = _scaffold_and_patch(
        tmp_path,
        monkeypatch,
        script=[{"manifest": manifest, "notes": "n"}],
        critic_script=[_critic_exec_result(verdict="accept")],
    )
    assert rc == 0
    scratch = critic_scratch_dirs[0]
    assert scratch.is_dir()
    # The five pre-seeded files.
    names = {p.name for p in scratch.iterdir() if p.is_file()}
    assert "manifest.json" in names
    assert "notes.md" in names
    assert "evidence-bundle.md" in names
    assert "validator-report.md" in names
    assert "task.md" in names
    # The manifest/notes are the JUDGED round's artifacts.
    assert json.loads((scratch / "manifest.json").read_text(encoding="utf-8")) == manifest
    assert (scratch / "notes.md").read_text(encoding="utf-8") == "n"
    # The evidence bundle is the deterministic machine-verified facts.
    bundle = (scratch / "evidence-bundle.md").read_text(encoding="utf-8")
    assert "## Per-ticket target_scope resolution" in bundle
    assert "a: src/app.py -> exists" in bundle
    # The validator report (clean by construction at the critic).
    assert "clean (0 defects)" in (scratch / "validator-report.md").read_text(encoding="utf-8")
    # The task file: the fenced spec + the on-disk paths (NOT the full
    # embedded manifest/notes text — the critic reads them with its tools).
    task_text = (scratch / "task.md").read_text(encoding="utf-8")
    assert "<spec>" in task_text and SPEC_TEXT in task_text
    assert str(scratch / "manifest.json") in task_text
    assert str(scratch / "notes.md") in task_text
    assert str(scratch / "evidence-bundle.md") in task_text
    assert str(scratch / "validator-report.md") in task_text
    # The target repo path.
    assert "repo root" in task_text


def test_critic_verdict_parsed_and_evidence_log_carried(
    tmp_path: Path, monkeypatch
) -> None:
    """The exec `result` payload is parsed (verdict/summary/findings/
    evidence_log + usage); the evidence_log rides the critic's findings
    record into the run summary, and the DECOMPOSE event carries the
    decomposecritic01 hash + the exec usage."""
    import hashlib

    manifest = [_ticket_entry("t-a")]
    major = [
        {"aspect": "fidelity", "severity": "major", "tickets": ["t-a"],
         "evidence": "misses the CLI", "direction": "add it"}
    ]
    log = [
        {"claim_checked": "ticket a extends src/app.py",
         "method": "file_read src/app.py", "found": "no CLI seam"},
    ]
    critic_usage = {"in": 4, "cached": 0, "out": 6, "reasoning": 0}
    # Round 0: repair with 1 major; after repair: accept.
    rc, exec_calls, critic_argvs, _, run_dir, rigs_root = _scaffold_and_patch(
        tmp_path,
        monkeypatch,
        script=[
            {"manifest": manifest, "notes": "n"},
            {"manifest": manifest, "notes": "n2"},
        ],
        critic_script=[
            _critic_exec_result(verdict="repair", findings=major,
                                evidence_log=log, usage=critic_usage),
            _critic_exec_result(verdict="accept"),
        ],
    )
    assert rc == 0
    assert len(exec_calls) == 2  # initial + one repair
    assert len(critic_argvs) == 2  # the critic judged both rounds
    # The evidence_log rides the critic's findings record into the summary.
    summary = (run_dir / "summary.md").read_text(encoding="utf-8")
    assert "no CLI seam" in summary
    # The DECOMPOSE events: two critic events, each carrying the critic's
    # exec usage and the decomposecritic01 prompt-artifact hash.
    events = _decompose_events(rigs_root)
    critic_events = [e for e in events if e["station"] == "decompose-critic"]
    assert len(critic_events) == 2
    # Each event carries ITS OWN invocation's usage: round 0's scripted
    # critic usage, round 1's the default from `_critic_exec_result`.
    assert critic_events[0]["tokens"] == critic_usage
    assert critic_events[1]["tokens"] == {
        "in": 5, "cached": 1, "out": 9, "reasoning": 0
    }
    for ev in critic_events:
        assert ev["model"] == "opus"
    resolved = _resolve(rigs_root)
    try:
        expected_hash = hashlib.sha256(
            (resolved.rig_paths["prompts_dir"] / "decomposecritic01").read_bytes()
        ).hexdigest()
    finally:
        resolved.store.close()
    for ev in critic_events:
        assert ev["prompt_artifact_hash"] == expected_hash
    # The critic-<n>.json artifacts carry the parsed payload + evidence_log.
    critic_artifacts = sorted(
        run_dir.glob("critic-*.json"), key=lambda p: int(p.stem.split("-")[1])
    )
    assert critic_artifacts
    first = json.loads(critic_artifacts[0].read_text(encoding="utf-8"))
    assert first["verdict"] == "repair"
    assert first["evidence_log"] == log


def test_critic_missing_result_one_retry_then_exit_1(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """A `done` exec with NO `result` field = the model never submitted = a
    critic FAILURE with its own bounded retry (exactly one), then a
    DecomposeError -> exit 1."""
    manifest = [_ticket_entry("t-a")]
    rc, exec_calls, critic_argvs, _, _, _ = _scaffold_and_patch(
        tmp_path,
        monkeypatch,
        script=[{"manifest": manifest, "notes": "n"}],
        critic_script=[
            {k: v for k, v in _critic_exec_result().items() if k != "result"},
            {k: v for k, v in _critic_exec_result().items() if k != "result"},
        ],
    )
    assert rc == 1
    assert "critic" in capsys.readouterr().err
    assert len(exec_calls) == 1  # the decomposer ran once (no repair — the
    # critic failure is not a repair trigger)
    assert len(critic_argvs) == 2  # exactly one retry


def test_critic_exec_failure_retried_within_critic_budget(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """A critic EXEC FAILURE (deny_reason) flows through the same
    retry-classification machinery as decomposer execs: one retry, then a
    bounded critic failure -> exit 1 (never conflated with a submitted
    verdict)."""
    manifest = [_ticket_entry("t-a")]
    rc, _, critic_argvs, _, _, _ = _scaffold_and_patch(
        tmp_path,
        monkeypatch,
        script=[{"manifest": manifest, "notes": "n"}],
        critic_script=[
            {"result": _exec_result(deny_reason="quota-tokens")},
            {"result": _exec_result(deny_reason="quota-tokens")},
        ],
    )
    assert rc == 1
    assert "critic" in capsys.readouterr().err
    assert len(critic_argvs) == 2  # exactly one retry


def test_critic_exec_failure_then_submit_succeeds(
    tmp_path: Path, monkeypatch
) -> None:
    """The critic's one-retry budget covers BOTH failure classes: a failed
    first attempt, then a `done` WITH the submit_validation result ->
    success (the retry worked)."""
    manifest = [_ticket_entry("t-a")]
    rc, _, critic_argvs, _, _, rigs_root = _scaffold_and_patch(
        tmp_path,
        monkeypatch,
        script=[{"manifest": manifest, "notes": "n"}],
        critic_script=[
            {"result": _exec_result(deny_reason="quota-tokens")},
            _critic_exec_result(verdict="accept"),
        ],
    )
    assert rc == 0
    assert len(critic_argvs) == 2
    assert _ticket(rigs_root, "t-a") is not None
    critic_events = [
        e for e in _decompose_events(rigs_root)
        if e["station"] == "decompose-critic"
    ]
    assert len(critic_events) == 1  # the failed attempt emits nothing


def test_repair_round_pre_seeds_round_scratch_and_edit_tools(
    tmp_path: Path, monkeypatch
) -> None:
    """A repair round PRE-SEEDS the round's scratch dir with the previous
    round's manifest + notes (round artifacts preserved; the agent edits
    the copies in place), invokes the repair exec with the EDIT-CAPABLE
    tool set, and the round's manifest artifact in the run dir is the
    repaired list."""
    major = [
        {"aspect": "fidelity", "severity": "major", "tickets": ["t-a"],
         "evidence": "misses the CLI", "direction": "add it"}
    ]
    # Round 0's manifest AS THE DECOMPOSER EMITS IT (v0.4 JSONL): the
    # pre-seed must carry these EXACT bytes to the repair session.
    manifest_text = (
        json.dumps(_ticket_entry("t-a"), indent=None) + "\n"
    )
    preseed_log: list[tuple[str | None, str | None]] = []
    rc, exec_calls, critic_argvs, _, run_dir, rigs_root = _scaffold_and_patch(
        tmp_path,
        monkeypatch,
        script=[
            {"manifest": manifest_text, "notes": "notes-v1"},  # round 0
            {"manifest": manifest_text, "notes": "notes-v2"},  # repair round
        ],
        critic_script=[
            _critic_exec_result(verdict="repair", findings=major),
            _critic_exec_result(verdict="accept"),
        ],
        preseed_log=preseed_log,
    )
    assert rc == 0
    assert len(exec_calls) == 2
    # Repair exec (the 2nd decomposer exec): the edit-capable tool set.
    repair_argv = exec_calls[1]
    assert repair_argv[repair_argv.index("--tools") + 1] == (
        "file_read,glob,grep,file_write,file_edit,file_patch"
    )
    # The initial round kept the current write-capable set (no
    # edit/patch).
    assert exec_calls[0][exec_calls[0].index("--tools") + 1] == (
        "file_read,glob,grep,file_write"
    )
    # The round's scratch dir was PRE-SEEDED with the previous round's
    # manifest + notes BYTE-IDENTICAL (Decision 18: the repair agent edits
    # the bytes it authored — a re-serialization would itself be drift).
    # Observable via the exec-entry snapshot (the fake's own writes then
    # overwrite the pre-seed, so post-hoc reads see the round's OUTPUT).
    assert len(preseed_log) == 2
    assert preseed_log[0] == (None, None)  # round 0: nothing pre-seeded
    assert preseed_log[1] == (manifest_text, "notes-v1")
    # The repair task names the on-disk previous files (paths, not embedded
    # text) — the task artifact is in the run dir.
    subject_scratch = Path(repair_argv[repair_argv.index("--task-file") + 1]).parent
    tasks = sorted(run_dir.glob("task-*.md"), key=lambda p: int(p.stem.split("-")[1]))
    repair_task = tasks[-1].read_text(encoding="utf-8")
    assert str(subject_scratch / "manifest.json") in repair_task
    assert str(subject_scratch / "notes.md") in repair_task
    # The run dir preserved the round artifacts (allow + audit).
    manifest_artifacts = sorted(
        run_dir.glob("manifest-*.json"), key=lambda p: int(p.stem.split("-")[1])
    )
    assert [json.loads(p.read_text(encoding="utf-8")) for p in manifest_artifacts]
    assert _ticket(rigs_root, "t-a") is not None


def test_repair_round_pre_seeds_phase_scratch_per_subject(
    tmp_path: Path, monkeypatch
) -> None:
    """Per-subject scratch: a phase subject's repair round pre-seeds the
    PHASE's scratch dir (not the root subject's)."""
    plan = [
        {"id": "p1", "title": "one", "goal": "g1", "depends_on": [],
         "done_condition": "d"},
    ]
    p1_manifest_text = json.dumps(_ticket_entry("t-p1", scope=["src/app.py", "tests/test_app.py"]))
    major = [
        {"aspect": "fidelity", "severity": "major", "tickets": ["t-p1"],
         "evidence": "misses", "direction": "fix"}
    ]
    preseed_log: list[tuple[str | None, str | None]] = []
    rc, exec_calls, _, _, _run_dir, _rr = _scaffold_and_patch(
        tmp_path,
        monkeypatch,
        script=[
            {"manifest": plan},                              # root -> phase plan
            {"manifest": p1_manifest_text, "notes": "p1-v1"},  # phase p1
            {"manifest": p1_manifest_text, "notes": "p1-v2"},  # phase p1 repair
        ],
        critic_script=[
            _critic_exec_result(verdict="repair", findings=major),  # p1 round 0
            _critic_exec_result(verdict="accept"),                  # p1 repair
        ],
        preseed_log=preseed_log,
    )
    assert rc == 0
    # Byte-identical pre-seed (Decision 18) observed at exec ENTRY: the
    # phase p1 repair round's scratch carried the phase's own round-0
    # bytes (per-subject scratch, not the root subject's).
    assert len(preseed_log) == 3
    assert preseed_log[0] == (None, None)  # root: phase plan, no pre-seed
    assert preseed_log[1] == (None, None)  # phase p1 round 0: no pre-seed
    assert preseed_log[2] == (p1_manifest_text, "p1-v1")


# --- CLI wiring after Decision 18 (the dead builder is GONE) -----------------


def test_cli_decompose_passes_nothing_critic_related(monkeypatch) -> None:
    """The decompose verb now passes NOTHING critic-related to
    `run_decompose` (the driver owns the critic exec invocation) — and the
    deleted `cli._build_decompose_critic` is gone from the CLI module."""
    assert not hasattr(cli, "_build_decompose_critic")
    seen: dict[str, Any] = {}

    def fake_run_decompose(*args: Any, **kwargs: Any) -> int:
        seen.update(kwargs)
        return 0

    class _StubResolved:
        class _Store:
            def close(self) -> None:
                pass

        store = _Store()

    monkeypatch.setattr(
        cli, "resolve_rig", lambda name, rigs_root=None: _StubResolved()
    )
    monkeypatch.setattr(cli, "run_decompose", fake_run_decompose)
    rc = cli.main([
        "decompose",
        "--rig", "shipyard",
        "--spec", "/tmp/spec.md",
        "--no-approve",
        "--max-repairs", "3",
        "--decomposer-model", "some/model",
        "--decomposer-effort", "medium",
        "--dry-run",
    ])
    assert rc == 0
    assert "critic_factory" not in seen
    assert seen["rig_name"] == "shipyard"
    assert str(seen["spec_path"]) == "/tmp/spec.md"
    assert seen["no_approve"] is True
    assert seen["max_repairs"] == 3
    assert seen["decomposer_model"] == "some/model"
    assert seen["decomposer_effort"] == "medium"
    assert seen["dry_run"] is True
