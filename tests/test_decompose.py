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
   in-process validation critic -> bounded repair loop -> per-phase fan-out
   -> intake + auto-approve -> DECOMPOSE provenance events).

ALL stub-driven: no network, no real `openalph exec`, no real critic calls.
The seams are the `decompose` module's `_run_exec` subprocess wrapper and
the injected critic client; rigs are real tmp scaffolds (real git clone,
real SQLite store) so intake/approval/records run on production code paths.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import stigmergy.cli as cli
from stigmergy import decompose
from stigmergy.intake import ingest_manifest
from stigmergy.oa_critic import CriticOAUnavailableError
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
) -> list[list[str]]:
    """Monkeypatch the decompose module's `_run_exec` seam with a scripted
    fake. Each script item is one of:

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


def _accept_response(
    *,
    verdict: str = "accept",
    findings: list[dict] | None = None,
    usage: dict | None = None,
) -> dict[str, Any]:
    return {
        "verdict": verdict,
        "summary": "manifest is faithful to the spec",
        "findings": findings or [],
        "usage": usage if usage is not None else {"in": 5, "cached": 1, "out": 9, "reasoning": 0},
    }


def _critic_factory(
    script: list[Any],
    *,
    template: str = "decomposecritic01 template",
    model: str = "stub-critic-model",
) -> tuple[Callable[[Any], decompose.DecomposeCritic], list[str], list[dict[str, Any]]]:
    """A stub critic factory for `run_decompose(critic_factory=...)`.

    Each client call returns the next item of `script` (a response dict, or
    an Exception instance to raise; the LAST item repeats if more calls
    occur than items). Returns `(factory, prompts, calls)` where `prompts`
    is the list of composed prompts (in call order) and `calls` records
    each call's kwargs (model/decoding_params) for the call-discipline
    assertions. The factory ignores its `resolved` argument (the CLI passes
    it; the stub doesn't need it).
    """
    prompts: list[str] = []
    calls: list[dict[str, Any]] = []
    cursor = 0

    def client(p: str, *, model: str, **params: Any) -> Any:
        nonlocal cursor
        prompts.append(p)
        calls.append({"model": model, "params": dict(params)})
        item = script[min(cursor, len(script) - 1)]
        cursor += 1
        if isinstance(item, BaseException):
            raise item
        return item

    def factory(_resolved: Any) -> decompose.DecomposeCritic:
        return decompose.DecomposeCritic(
            client=client,
            model=model,
            template=template,
            decoding_params={},
        )

    return factory, prompts, calls


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
) -> tuple[int, list[list[str]], list[str], list[dict[str, Any]], Path]:
    """Scaffold a rig + spec, stub the exec + critic seams, and call
    `decompose.run_decompose`. Returns
    `(rc, exec_calls, critic_prompts, critic_calls, run_dir)` where
    `run_dir` is the single subdir under the decompose root."""
    rigs_root = scaffold_rig(tmp_path)
    spec = _write_spec(tmp_path, kwargs.pop("spec_text", SPEC_TEXT))
    decompose_root = tmp_path / "decompose-root"
    # Hermetic preflight: pin the agent-TOML path at an existing dummy file
    # so the suite never depends on the host's OA config dir (the dedicated
    # missing-TOML test overrides this with a nonexistent path).
    agent_toml = tmp_path / "agent.toml"
    agent_toml.write_text("[agent]\nname = 'stigmergy-decomposer'\n", encoding="utf-8")
    monkeypatch.setattr(decompose, "_DECOMP_AGENT_TOML", agent_toml)
    critic_factory, prompts, calls = _critic_factory(critic_script or [_accept_response()])
    exec_calls = _fake_exec(monkeypatch, script, env_probe=env_probe)
    rc = decompose.run_decompose(
        rig_name=RIG,
        spec_path=spec,
        decompose_root=decompose_root,
        rigs_root=rigs_root,
        critic_factory=critic_factory,
        **kwargs,
    )
    run_dirs = list((decompose_root / "decompose").iterdir())
    return rc, exec_calls, prompts, calls, run_dirs[0]


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


def test_render_task_repair_carries_load_bearing_rules_verbatim() -> None:
    text = decompose.render_task(
        mode="repair",
        spec=SPEC_TEXT,
        repo_root="/repo",
        charter_path="/charter.toml",
        manifest_path="/scratch/manifest.json",
        notes_path="/scratch/notes.md",
        previous_manifest_text='[{"id": "a"}]',
        previous_notes_text="prior notes",
        findings=["1. fidelity: ticket a misses the CLI deliverable"],
        round_no=1,
    )
    # The three load-bearing repair rules, pinned VERBATIM (bead
    # workspace-e2uh.152).
    assert (
        "Change only what the findings require. Keep every ticket id stable "
        "unless a finding demands a rename. Address every finding; do not "
        "weaken criteria to make findings disappear. Re-emit BOTH complete "
        "files."
    ) in text
    # Previous artifacts + numbered findings ride along.
    assert '[{"id": "a"}]' in text
    assert "prior notes" in text
    assert "fidelity: ticket a misses the CLI deliverable" in text
    # And the re-emit target paths.
    assert "/scratch/manifest.json" in text
    assert "/scratch/notes.md" in text


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


# --- DecomposeCritic --------------------------------------------------------


def test_decompose_critic_prompt_composition_order(tmp_path: Path) -> None:
    """Prompt order (the decomposecritic01 <input_contract>): artifact text
    FIRST, then spec, manifest, notes, evidence bundle, validator report —
    plus the pre-verified-mechanical-checks instruction."""
    factory, prompts, _ = _critic_factory([_accept_response()])
    critic = factory(None)
    critic.review(
        spec="SPEC-CONTENT",
        manifest_text='[{"id": "a"}]',
        notes_text="NOTES-CONTENT",
        evidence_bundle="EVIDENCE-CONTENT",
        validator_report=[],
    )
    assert len(prompts) == 1
    prompt = prompts[0]
    i_template = prompt.find("decomposecritic01 template")
    i_spec = prompt.find("## Spec\nSPEC-CONTENT")
    i_manifest = prompt.find("## Manifest (JSON)\n[")
    i_notes = prompt.find("## Notes\nNOTES-CONTENT")
    i_bundle = prompt.find("## Evidence bundle\nEVIDENCE-CONTENT")
    i_report = prompt.find("## Validator report\n")
    assert -1 < i_template < i_spec < i_manifest < i_notes < i_bundle < i_report
    assert "pre-verified" in prompt
    assert "submit_validation" in prompt


def test_decompose_critic_from_prompt_file_hashes_raw_bytes(tmp_path: Path) -> None:
    import hashlib

    template_path = tmp_path / "decomposecritic01"
    template_path.write_text("critic template bytes\n", encoding="utf-8")
    critic = decompose.DecomposeCritic.from_prompt_file(
        template_path, client=lambda *a, **k: {}, model="m"
    )
    # SPEC §4: sha256 of the RAW template bytes.
    assert critic.prompt_artifact_hash == hashlib.sha256(
        b"critic template bytes\n"
    ).hexdigest()
    assert critic.decoding_params == {}


def test_decompose_critic_returns_verdict_and_usage() -> None:
    factory, _, calls = _critic_factory(
        [_accept_response(verdict="accept", findings=[
            {"aspect": "sizing", "severity": "minor", "tickets": ["a"],
             "evidence": "big", "direction": "split"}
        ], usage={"in": 7, "cached": 1, "out": 3, "reasoning": 0})]
    )
    critic = factory(None)
    result = critic.review(
        spec="s", manifest_text="m", notes_text="n", evidence_bundle="e",
        validator_report=["clean"],
    )
    assert result["verdict"] == "accept"
    assert result["summary"]
    assert result["findings"][0]["aspect"] == "sizing"
    assert result["usage"] == {"in": 7, "cached": 1, "out": 3, "reasoning": 0}
    assert result["prompt_artifact_hash"] == critic.prompt_artifact_hash
    # Call discipline: the model + decoding_params={} (the OA forced-tool
    # path rejects non-empty decoding params).
    assert calls[0]["model"] == "stub-critic-model"
    assert calls[0]["params"] == {}


def test_decompose_critic_malformed_one_retry_then_error() -> None:
    """A malformed response gets exactly ONE retry call, then a
    DecomposeError (never a third call)."""
    factory, _, calls = _critic_factory([
        "not a dict",
        {"verdict": "banana", "summary": "s", "findings": []},  # bad verdict
    ])
    critic = factory(None)
    with pytest.raises(decompose.DecomposeError, match="malformed after retry"):
        critic.review(
            spec="s", manifest_text="m", notes_text="n", evidence_bundle="e",
            validator_report=[],
        )
    assert len(calls) == 2  # exactly one retry


def test_decompose_critic_client_exception_one_retry_then_error() -> None:
    """A client/transport exception is the same one-retry budget: two calls
    total, then DecomposeError."""
    factory, _, calls = _critic_factory([
        RuntimeError("provider down"),
        RuntimeError("provider down"),
    ])
    critic = factory(None)
    with pytest.raises(decompose.DecomposeError, match="malformed after retry"):
        critic.review(
            spec="s", manifest_text="m", notes_text="n", evidence_bundle="e",
            validator_report=[],
        )
    assert len(calls) == 2


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
        critic_factory, prompts, calls = _critic_factory(
            [_accept_response(verdict="accept")]
        )
        exec_calls = _fake_exec(monkeypatch, [{"manifest": manifest, "notes": "n"}])
        rc = decompose.run_decompose(
            rig_name=RIG,
            spec_path=spec,
            decompose_root=decompose_root,
            rigs_root=rigs_root,
            critic_factory=critic_factory,
        )
        return rc, exec_calls, prompts, calls, decompose_root, rigs_root

    rc, exec_calls, prompts, calls, decompose_root, rigs_root = with_rig(tmp_path, monkeypatch)
    assert rc == 0
    assert len(exec_calls) == 1  # one decomposer session, no repairs
    assert len(prompts) == 1  # one critic call

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
    critic_factory, prompts, _ = _critic_factory([_accept_response()])
    _fake_exec(monkeypatch, [{"manifest": manifest, "notes": "n"}])
    rc = decompose.run_decompose(
        rig_name=RIG,
        spec_path=spec,
        decompose_root=tmp_path / "d",
        rigs_root=rigs_root,
        critic_factory=critic_factory,
        no_approve=True,
    )
    assert rc == 0
    t = _ticket(rigs_root, "t-only")
    assert t is not None
    assert t["approved"] == 0  # stays unapproved/pool


def test_run_decompose_dry_run_intakes_nothing(tmp_path: Path, monkeypatch, capsys) -> None:
    manifest = [_ticket_entry("t-dry")]
    rigs_root = scaffold_rig(tmp_path)
    spec = _write_spec(tmp_path)
    critic_factory, prompts, _ = _critic_factory([_accept_response()])
    _fake_exec(monkeypatch, [{"manifest": manifest, "notes": "n"}])
    rc = decompose.run_decompose(
        rig_name=RIG,
        spec_path=spec,
        decompose_root=tmp_path / "d",
        rigs_root=rigs_root,
        critic_factory=critic_factory,
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
    critic_factory, prompts, _ = _critic_factory([_accept_response()])
    exec_calls = _fake_exec(
        monkeypatch, [{"notes": "The spec is contradictory between A and B."}]
    )
    rc = decompose.run_decompose(
        rig_name=RIG,
        spec_path=spec,
        decompose_root=tmp_path / "d",
        rigs_root=rigs_root,
        critic_factory=critic_factory,
    )
    assert rc == 1
    assert len(exec_calls) == 1  # escape never retried
    err = capsys.readouterr().err
    assert "decompose:" in err
    # No critic call (no manifest to judge).
    assert len(prompts) == 0
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
    critic_factory, prompts, calls = _critic_factory([_accept_response()])
    _fake_exec(
        monkeypatch,
        [
            {"manifest": _defective_manifest(), "notes": "n"},  # round 0: defective
            {"manifest": good, "notes": "n2"},  # repair round: clean
        ],
    )
    rc = decompose.run_decompose(
        rig_name=RIG,
        spec_path=spec,
        decompose_root=tmp_path / "d",
        rigs_root=rigs_root,
        critic_factory=critic_factory,
    )
    assert rc == 0
    assert _ticket(rigs_root, "t-good") is not None
    # The critic was called exactly ONCE — only after the repair produced a
    # validator-clean manifest (never over the schema-broken one).
    assert len(prompts) == 1
    assert len(calls) == 1


def test_run_decompose_clean_manifest_critic_once(tmp_path: Path, monkeypatch) -> None:
    manifest = [_ticket_entry("t-a")]
    rigs_root = scaffold_rig(tmp_path)
    spec = _write_spec(tmp_path)
    critic_factory, prompts, calls = _critic_factory([_accept_response()])
    _fake_exec(monkeypatch, [{"manifest": manifest, "notes": "n"}])
    rc = decompose.run_decompose(
        rig_name=RIG,
        spec_path=spec,
        decompose_root=tmp_path / "d",
        rigs_root=rigs_root,
        critic_factory=critic_factory,
    )
    assert rc == 0
    assert len(prompts) == 1  # exactly one critic call
    assert len(calls) == 1


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
    critic_factory, prompts, calls = _critic_factory(
        [_accept_response(verdict="accept", findings=minor)]
    )
    exec_calls = _fake_exec(monkeypatch, [{"manifest": manifest, "notes": "n"}])
    rc = decompose.run_decompose(
        rig_name=RIG,
        spec_path=spec,
        decompose_root=tmp_path / "d",
        rigs_root=rigs_root,
        critic_factory=critic_factory,
    )
    assert rc == 0
    assert len(exec_calls) == 1  # no repair
    assert len(prompts) == 1  # one critic call
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
    critic_factory, prompts, calls = _critic_factory([
        _accept_response(verdict="repair", findings=major),  # round 0
        _accept_response(verdict="accept"),  # after repair
    ])
    exec_calls = _fake_exec(
        monkeypatch,
        [
            {"manifest": bad, "notes": "n"},
            {"manifest": good, "notes": "n2"},
        ],
    )
    rc = decompose.run_decompose(
        rig_name=RIG,
        spec_path=spec,
        decompose_root=tmp_path / "d",
        rigs_root=rigs_root,
        critic_factory=critic_factory,
    )
    assert rc == 0
    assert len(exec_calls) == 2  # initial + one repair
    assert len(prompts) == 2  # two critic calls (round 0 + after repair)
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
    critic_factory, prompts, calls = _critic_factory([
        _accept_response(verdict="repair", findings=major),
        _accept_response(verdict="repair", findings=major),  # flat -> non-converge
    ])
    exec_calls = _fake_exec(
        monkeypatch,
        [
            {"manifest": manifest, "notes": "n"},
            {"manifest": manifest, "notes": "n2"},
        ],
    )
    rc = decompose.run_decompose(
        rig_name=RIG,
        spec_path=spec,
        decompose_root=tmp_path / "d",
        rigs_root=rigs_root,
        critic_factory=critic_factory,
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
    critic_factory, prompts, calls = _critic_factory([
        _accept_response(verdict="repair", findings=two_major),
        _accept_response(verdict="repair", findings=one_major),
    ])
    exec_calls = _fake_exec(
        monkeypatch,
        [
            {"manifest": manifest, "notes": "n"},
            {"manifest": manifest, "notes": "n2"},
            {"manifest": manifest, "notes": "n3"},
        ],
    )
    rc = decompose.run_decompose(
        rig_name=RIG,
        spec_path=spec,
        decompose_root=tmp_path / "d",
        rigs_root=rigs_root,
        critic_factory=critic_factory,
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
    critic_factory, prompts, _ = _critic_factory([_accept_response()])
    exec_calls = _fake_exec(
        monkeypatch,
        [
            {"result": _exec_result(deny_reason="quota-tokens")},
            {"result": _exec_result(deny_reason="quota-tokens")},
        ],
    )
    rc = decompose.run_decompose(
        rig_name=RIG,
        spec_path=spec,
        decompose_root=tmp_path / "d",
        rigs_root=rigs_root,
        critic_factory=critic_factory,
    )
    assert rc == 2
    assert len(exec_calls) == 2
    err = capsys.readouterr().err
    assert "exec failed" in err
    assert len(prompts) == 0  # no critic call


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
    critic_factory, prompts, _ = _critic_factory(
        [_accept_response(usage=critic_usage)],
        model="claude-max-sub",  # a subscription entry -> computed_usd 0.0
    )
    _fake_exec(
        monkeypatch,
        [{"manifest": manifest, "notes": "n",
          "result": _exec_result(usage=decomp_usage)}],
    )
    rc = decompose.run_decompose(
        rig_name=RIG,
        spec_path=spec,
        decompose_root=tmp_path / "d",
        rigs_root=rigs_root,
        critic_factory=critic_factory,
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
    critic_factory, prompts, _ = _critic_factory([_accept_response()])
    _fake_exec(
        monkeypatch,
        [
            {"result": _exec_result(deny_reason="quota-tokens")},
            {"result": _exec_result(deny_reason="quota-tokens")},
        ],
    )
    rc = decompose.run_decompose(
        rig_name=RIG,
        spec_path=spec,
        decompose_root=tmp_path / "d",
        rigs_root=rigs_root,
        critic_factory=critic_factory,
    )
    assert rc == 2
    assert _decompose_events(rigs_root) == []
    assert len(prompts) == 0


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
    critic_factory, prompts, _ = _critic_factory([
        _accept_response(),  # p1 critic
        _accept_response(),  # p2 critic
    ])
    _fake_exec(
        monkeypatch,
        [
            {"manifest": plan},            # initial (root) -> phase plan
            {"manifest": p1_manifest, "notes": "n1"},  # phase p1
            {"manifest": p2_manifest, "notes": "n2"},  # phase p2
        ],
    )
    rc = decompose.run_decompose(
        rig_name=RIG,
        spec_path=spec,
        decompose_root=tmp_path / "d",
        rigs_root=rigs_root,
        critic_factory=critic_factory,
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
    critic_factory, prompts, _ = _critic_factory([
        _accept_response(),  # p1 critic
        _accept_response(),  # p2 critic
    ])
    _fake_exec(
        monkeypatch,
        [
            {"manifest": _two_phase_plan()},
            {"manifest": p1_manifest, "notes": "n1"},  # p1 (first, topo)
            {"manifest": p2_manifest, "notes": "n2"},  # p2 (second)
        ],
    )
    rc = decompose.run_decompose(
        rig_name=RIG,
        spec_path=spec,
        decompose_root=tmp_path / "d",
        rigs_root=rigs_root,
        critic_factory=critic_factory,
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
    tasks = sorted(run_dir.glob("task-*.md"))
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
    critic_factory, prompts, _ = _critic_factory([_accept_response()])
    _fake_exec(monkeypatch, [{"manifest": plan}])
    rc = decompose.run_decompose(
        rig_name=RIG,
        spec_path=spec,
        decompose_root=tmp_path / "d",
        rigs_root=rigs_root,
        critic_factory=critic_factory,
    )
    assert rc == 1
    assert "cycle" in capsys.readouterr().err
    assert len(prompts) == 0  # no critic: the plan defect fails before any phase


def test_phase_fanout_unknown_dep_is_error(tmp_path: Path, monkeypatch, capsys) -> None:
    plan = [
        {"id": "p1", "title": "a", "goal": "g", "depends_on": ["p-ghost"],
         "done_condition": "d"},
    ]
    rigs_root = scaffold_rig(tmp_path)
    spec = _write_spec(tmp_path)
    critic_factory, prompts, _ = _critic_factory([_accept_response()])
    _fake_exec(monkeypatch, [{"manifest": plan}])
    rc = decompose.run_decompose(
        rig_name=RIG,
        spec_path=spec,
        decompose_root=tmp_path / "d",
        rigs_root=rigs_root,
        critic_factory=critic_factory,
    )
    assert rc == 1
    assert "unknown phase" in capsys.readouterr().err
    assert len(prompts) == 0


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
    monkeypatch.setattr(cli, "_build_decompose_critic", lambda resolved: object())
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
        critic_factory=lambda resolved: object(),
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
    monkeypatch.setattr(cli, "_build_decompose_critic", lambda resolved: object())
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


# --- _build_decompose_critic (CLI mold) ---------------------------------------


def test_build_decompose_critic_wiring(tmp_path: Path, monkeypatch) -> None:
    """`cli._build_decompose_critic` (the `_build_range_critic` mold):
    reads [roles.critic].model + max_tokens, wires
    make_op_key_provider(_critic_key_ref_for(...)) into
    make_oa_decompose_critic_client, and reads the decomposecritic01
    prompt artifact (sha256'd for provenance)."""
    import hashlib

    from stigmergy.oa_critic import DEFAULT_MAX_TOKENS

    rigs_root = scaffold_rig(tmp_path)
    resolved = resolve_rig(RIG, rigs_root=rigs_root)
    try:
        seen: dict[str, Any] = {}

        def fake_key_provider(ref: str) -> Any:
            seen["ref"] = ref
            return lambda: "stub-key"

        def fake_client(*, key_provider: Any, registry: Any, max_tokens: int) -> Any:
            seen["key_provider"] = key_provider
            seen["registry"] = registry
            seen["max_tokens"] = max_tokens
            return lambda *a, **k: {}

        monkeypatch.setattr(cli, "make_op_key_provider", fake_key_provider)
        monkeypatch.setattr(cli, "make_oa_decompose_critic_client", fake_client)
        critic = cli._build_decompose_critic(resolved)
        # opus is Anthropic-routed -> the dedicated critic key item.
        assert seen["ref"] == cli._CRITIC_KEY_REF
        assert seen["max_tokens"] == DEFAULT_MAX_TOKENS
        assert seen["key_provider"]() == "stub-key"
        assert critic.model == "opus"
        assert critic.decoding_params == {}
        assert critic.prompt_artifact_hash == hashlib.sha256(
            b"decomposecritic01 template\n"
        ).hexdigest()
    finally:
        resolved.store.close()


def test_build_decompose_critic_charter_max_tokens_override(
    tmp_path: Path, monkeypatch
) -> None:
    """A charter's [roles.critic].max_tokens overrides the default."""
    repo = make_local_repo_with_prompts(tmp_path)
    charter_dir = tmp_path / "charter_src"
    charter_dir.mkdir()
    text = BASE_CHARTER_TOML.replace('repo = "path-or-url"', f'repo = "{repo}"')
    text = text.replace(
        '[roles.critic]\nmodel = "opus"',
        '[roles.critic]\nmodel = "opus"\nmax_tokens = 1234',
    )
    charter_path = charter_dir / "charter.toml"
    charter_path.write_text(text)
    shutil.copy(MODELS_REGISTRY_PATH, charter_dir / "models.toml")
    rigs_root = tmp_path / "rigs"
    create_rig(charter_path, base_dir=rigs_root)
    resolved = resolve_rig(RIG, rigs_root=rigs_root)
    try:
        seen: dict[str, Any] = {}

        def fake_client(*, key_provider: Any, registry: Any, max_tokens: int) -> Any:
            seen["max_tokens"] = max_tokens
            return lambda *a, **k: {}

        monkeypatch.setattr(cli, "make_op_key_provider", lambda ref: (lambda: "k"))
        monkeypatch.setattr(cli, "make_oa_decompose_critic_client", fake_client)
        cli._build_decompose_critic(resolved)
        assert seen["max_tokens"] == 1234
    finally:
        resolved.store.close()


def test_build_decompose_critic_fails_closed_oa_unavailable(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """An OA-less environment (CriticOAUnavailableError at factory build) is
    a loud LAUNCH failure (exit 1, stderr) — exactly like range-report's
    handling — and run_decompose is never reached. The stub message mirrors
    the REAL factory's fail-closed phrasing (names the openalph provider
    layer), so the assertion proves the message reaches stderr verbatim."""
    _oa_msg = "openalph provider layer unavailable — openalph is not importable"

    def _fail(**_kw: Any) -> Any:
        raise CriticOAUnavailableError(_oa_msg)

    monkeypatch.setattr(cli, "make_oa_decompose_critic_client", _fail)

    def _must_not_run(*_a: Any, **_k: Any) -> int:
        raise AssertionError("must not run")

    monkeypatch.setattr(cli, "run_decompose", _must_not_run)
    rigs_root = scaffold_rig(tmp_path)
    spec = _write_spec(tmp_path)
    rc = cli.main([
        "decompose", "--rig", RIG, "--rigs-root", str(rigs_root),
        "--spec", str(spec),
    ])
    assert rc == 1
    err = capsys.readouterr().err
    assert "stigmergy decompose:" in err
    assert "openalph" in err


def test_build_decompose_critic_missing_key_ref_fails_closed(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """A non-Anthropic-routed critic model with no STIGMERGY_CRITIC_OA_KEY_REF
    is a loud rig-launch CharterError (exit 1, stderr)."""
    repo = make_local_repo_with_prompts(tmp_path)
    charter_dir = tmp_path / "charter_src"
    charter_dir.mkdir()
    text = BASE_CHARTER_TOML.replace('repo = "path-or-url"', f'repo = "{repo}"')
    text = text.replace('model = "opus"\n\n[models]', 'model = "local-qwen"\n\n[models]')
    charter_path = charter_dir / "charter.toml"
    charter_path.write_text(text)
    shutil.copy(MODELS_REGISTRY_PATH, charter_dir / "models.toml")
    rigs_root = tmp_path / "rigs"
    create_rig(charter_path, base_dir=rigs_root)
    monkeypatch.delenv("STIGMERGY_CRITIC_OA_KEY_REF", raising=False)

    def _must_not_run2(*_a: Any, **_k: Any) -> int:
        raise AssertionError("must not run")

    monkeypatch.setattr(cli, "run_decompose", _must_not_run2)
    spec = _write_spec(tmp_path)
    rc = cli.main([
        "decompose", "--rig", RIG, "--rigs-root", str(rigs_root),
        "--spec", str(spec),
    ])
    assert rc == 1
    err = capsys.readouterr().err
    assert "STIGMERGY_CRITIC_OA_KEY_REF" in err


def test_cli_decompose_builds_critic_before_dispatch(
    tmp_path: Path, monkeypatch
) -> None:
    """The critic is built at LAUNCH (fail-closed) and the BUILT instance is
    what run_decompose receives as its critic_factory (not rebuilt per
    judgment)."""
    rigs_root = scaffold_rig(tmp_path)
    spec = _write_spec(tmp_path)
    sentinel = object()
    monkeypatch.setattr(cli, "_build_decompose_critic", lambda resolved: sentinel)
    seen: dict[str, Any] = {}

    def fake_run_decompose(*args: Any, **kwargs: Any) -> int:
        seen["critic_factory"] = kwargs.get("critic_factory")
        return 0

    monkeypatch.setattr(cli, "run_decompose", fake_run_decompose)
    rc = cli.main([
        "decompose", "--rig", RIG, "--rigs-root", str(rigs_root),
        "--spec", str(spec),
    ])
    assert rc == 0
    assert callable(seen["critic_factory"])
    assert seen["critic_factory"](None) is sentinel
