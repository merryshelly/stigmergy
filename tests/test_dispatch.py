"""Tests for stigmergy.dispatch (SPEC.md §3 dispatch station, §4 path
safety, §5 `[lanes]`/`[prompts]`, §6 ticket work-order contract; bead .21
build spec §3 — frozen case list, cases 1-20).

Deterministic, offline except for real local git repos and a real SQLite
`RigStore` (mirrors `test_driver.py`'s and `rangereport.py`'s real-temp-repo
test style — git itself is never mocked).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

import stigmergy.dispatch as dispatch_module
from stigmergy.charter import Charter, load_charter
from stigmergy.dispatch import (
    _MAX_TASK_PACK_BYTES,
    _NAME_RETRY_LIMIT,
    _WORDLIST_PATH,
    DispatchError,
    PriorEvidence,
    assemble_task_pack,
    create_work_clone,
    generate_worker_name,
    prepare_dispatch,
    select_lane,
)
from stigmergy.drivers import claude_code
from stigmergy.relay import CapabilityStore
from stigmergy.rig import RigStore

FIXTURES = Path(__file__).parent / "fixtures"
VALID_CHARTER_PATH = FIXTURES / "charter_valid.toml"
MODELS_REGISTRY_PATH = FIXTURES / "models.toml"
BASE_CHARTER_TOML = VALID_CHARTER_PATH.read_text()

PINNED_IMAGE = "localhost/stigmergy-worker@sha256:" + "a" * 64


# --------------------------------------------------------------------------
# shared fixtures / helpers
# --------------------------------------------------------------------------


def _lane(
    *,
    driver: str = "claude-code",
    model: str = "haiku",
    prompt: str = "code01",
    egress: list[str] | None = None,
    selector: dict | None = None,
    entry: bool = True,
) -> dict:
    cfg: dict = {
        "driver": driver,
        "model": model,
        "prompt": prompt,
        "egress": egress or ["inference"],
    }
    if selector is not None:
        cfg["selector"] = selector
    if entry is False:
        cfg["entry"] = False
    return cfg


def _lanes_charter(lanes: dict) -> Charter:
    """A minimal Charter carrying only `[lanes]` — enough for `select_lane`,
    which reads nothing else."""
    return Charter(raw={"lanes": lanes}, resolved_hash="test-hash", warnings=())


def _git(repo: Path, args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=check
    )


def make_repo(tmp_path: Path, name: str = "source_repo", *, branch: str = "staging") -> Path:
    """A real, small local git repo with one commit on ``branch``."""
    repo = tmp_path / name
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, ["init", "--quiet", "-b", branch])
    _git(repo, ["config", "user.email", "test@example.com"])
    _git(repo, ["config", "user.name", "Test User"])
    (repo / "README.md").write_text("hello from the fixture repo\n")
    _git(repo, ["add", "README.md"])
    _git(repo, ["commit", "--quiet", "-m", "initial commit"])
    return repo


def make_full_charter(tmp_path: Path) -> Charter:
    """A fully valid, loaded Charter (via the real `charter_valid.toml`
    fixture + its models registry) for the end-to-end `prepare_dispatch`
    tests (cases 18-20). `tiers.dispatch_base = "staging"`, lanes `cheap`
    (selector `local-ok`), `default` (selector-less fallthrough),
    `exquisite` (entry=false, step-up only)."""
    charter_dir = tmp_path / "charter_src"
    charter_dir.mkdir(exist_ok=True)
    charter_path = charter_dir / "charter.toml"
    charter_path.write_text(BASE_CHARTER_TOML)
    shutil.copy(MODELS_REGISTRY_PATH, charter_dir / "models.toml")
    return load_charter(charter_path, env={})


def _sequence_rng(words: list[str]):
    """An injected `rng` that yields ``words`` in order, one per call."""
    it = iter(words)

    def _rng() -> str:
        return next(it)

    return _rng


def _constant_rng(word: str):
    def _rng() -> str:
        return word

    return _rng


def _success_json() -> str:
    import json

    return json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "total_cost_usd": 0.01,
            "usage": {
                "input_tokens": 10,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 20,
            },
        }
    )


class _CapturingRunOne:
    """Deterministic injected executor (mirrors test_driver.py's
    CapturingRunOne): records every call, returns a scripted
    CompletedProcess. No real podman/claude-code invocation."""

    def __init__(self, stdout: str = "", *, returncode: int = 0):
        self.stdout = stdout
        self.returncode = returncode
        self.calls: list[tuple] = []

    def __call__(self, argv, env, timeout):
        self.calls.append((argv, env, timeout))
        return subprocess.CompletedProcess(
            args=argv, returncode=self.returncode, stdout=self.stdout, stderr=""
        )


# ==========================================================================
# Lane selection. Cases 1-4.
# ==========================================================================


def test_case1_first_match_selector_wins(tmp_path: Path) -> None:
    lanes = {
        "first": _lane(model="model-a", selector={"label": "dup"}),
        "second": _lane(model="model-b", selector={"label": "dup"}),
        "fallthrough": _lane(model="model-c"),
    }
    charter = _lanes_charter(lanes)
    ticket_row = {"lane_hint": "dup"}

    selection = select_lane(charter, ticket_row)

    assert selection.name == "first"
    assert selection.model == "model-a"


def test_case2_entry_false_lane_never_selected_even_if_selector_matches(tmp_path: Path) -> None:
    lanes = {
        "stepup-only": _lane(model="opus", selector={"label": "match-me"}, entry=False),
        "fallthrough": _lane(model="sonnet"),
    }
    charter = _lanes_charter(lanes)
    ticket_row = {"lane_hint": "match-me"}

    selection = select_lane(charter, ticket_row)

    # Falls through past the entry=false lane (whose selector otherwise
    # matches) to the selector-less fallthrough lane — never selects
    # "stepup-only".
    assert selection.name == "fallthrough"
    assert selection.model == "sonnet"


def test_case2_entry_false_lane_skipped_falls_to_another_matching_lane(tmp_path: Path) -> None:
    """A variant of case 2: when another entry-eligible lane's selector
    also matches, selection falls through to THAT lane, not the
    entry=false one."""
    lanes = {
        "stepup-only": _lane(model="opus", selector={"label": "match-me"}, entry=False),
        "also-matches": _lane(model="sonnet", selector={"label": "match-me"}),
        "fallthrough": _lane(model="haiku"),
    }
    charter = _lanes_charter(lanes)
    ticket_row = {"lane_hint": "match-me"}

    selection = select_lane(charter, ticket_row)

    assert selection.name == "also-matches"


def test_case3_unmatched_lane_hint_falls_through(tmp_path: Path) -> None:
    lanes = {
        "cheap": _lane(model="haiku", selector={"label": "local-ok"}),
        "fallthrough": _lane(model="sonnet"),
    }
    charter = _lanes_charter(lanes)
    ticket_row = {"lane_hint": "no-such-label"}

    selection = select_lane(charter, ticket_row)

    assert selection.name == "fallthrough"


def test_case4_lane_hint_none_or_absent_falls_through(tmp_path: Path) -> None:
    lanes = {
        "cheap": _lane(model="haiku", selector={"label": "local-ok"}),
        "fallthrough": _lane(model="sonnet"),
    }
    charter = _lanes_charter(lanes)

    selection_none = select_lane(charter, {"lane_hint": None})
    selection_absent = select_lane(charter, {})

    assert selection_none.name == "fallthrough"
    assert selection_absent.name == "fallthrough"


# ==========================================================================
# Required-reading / task-pack assembly. Cases 5-10.
# ==========================================================================


def test_case5_context_and_repo_prefixes_resolve_correctly(tmp_path: Path) -> None:
    context_root = tmp_path / "context"
    repo_root = tmp_path / "repo"
    (context_root / "docs").mkdir(parents=True)
    (repo_root / "src").mkdir(parents=True)
    (context_root / "docs" / "readme.md").write_text("context content")
    (repo_root / "src" / "foo.py").write_text("repo content")

    ticket_row = {
        "required_reading": ["context:docs/readme.md", "repo:src/foo.py"],
        "goal": "do the thing",
    }
    dest = tmp_path / "task_pack"

    assemble_task_pack(
        ticket_row,
        context_root=context_root,
        repo_root=repo_root,
        prompt_template="Goal: $goal",
        dest=dest,
    )

    assert (dest / "context" / "docs" / "readme.md").read_text() == "context content"
    assert (dest / "context" / "src" / "foo.py").read_text() == "repo content"
    assert (dest / "prompt.md").read_text() == "Goal: do the thing"


def test_case6_context_traversal_rejected_all_or_nothing(tmp_path: Path) -> None:
    context_root = tmp_path / "context"
    context_root.mkdir()
    (context_root / "good.txt").write_text("good content")
    # A sibling file OUTSIDE context_root that the traversal would reach if
    # it were tolerated.
    (tmp_path / "escape.txt").write_text("should never be copied")

    ticket_row = {
        "required_reading": ["context:good.txt", "context:../escape.txt"],
        "goal": "goal text",
    }
    dest = tmp_path / "task_pack_case6"

    with pytest.raises(DispatchError):
        assemble_task_pack(
            ticket_row,
            context_root=context_root,
            repo_root=tmp_path / "repo",
            prompt_template="Goal: $goal",
            dest=dest,
        )

    # All-or-nothing: nothing was written, not even the already-valid entry.
    assert not dest.exists()


def test_case7_unrecognized_prefix_raises(tmp_path: Path) -> None:
    context_root = tmp_path / "context"
    context_root.mkdir()
    ticket_row = {"required_reading": ["other:foo"], "goal": "goal text"}
    dest = tmp_path / "task_pack_case7"

    with pytest.raises(DispatchError):
        assemble_task_pack(
            ticket_row,
            context_root=context_root,
            repo_root=tmp_path / "repo",
            prompt_template="Goal: $goal",
            dest=dest,
        )
    assert not dest.exists()


def test_case8_special_file_rejected(tmp_path: Path) -> None:
    context_root = tmp_path / "context"
    context_root.mkdir()
    fifo_path = context_root / "fifo"
    os.mkfifo(fifo_path)

    ticket_row = {"required_reading": ["context:fifo"], "goal": "goal text"}
    dest = tmp_path / "task_pack_case8"

    with pytest.raises(DispatchError):
        assemble_task_pack(
            ticket_row,
            context_root=context_root,
            repo_root=tmp_path / "repo",
            prompt_template="Goal: $goal",
            dest=dest,
        )
    assert not dest.exists()


def test_case9_total_bytes_over_cap_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dispatch_module, "_MAX_TASK_PACK_BYTES", 10)

    context_root = tmp_path / "context"
    context_root.mkdir()
    (context_root / "big.txt").write_text("x" * 20)

    ticket_row = {"required_reading": ["context:big.txt"], "goal": "goal text"}
    dest = tmp_path / "task_pack_case9"

    with pytest.raises(DispatchError):
        assemble_task_pack(
            ticket_row,
            context_root=context_root,
            repo_root=tmp_path / "repo",
            prompt_template="Goal: $goal",
            dest=dest,
        )
    assert not dest.exists()


def test_case9_real_max_task_pack_bytes_constant_is_8mib() -> None:
    assert _MAX_TASK_PACK_BYTES == 8 * 1024 * 1024


def test_case10_prompt_verbatim_not_reinterpreted(tmp_path: Path) -> None:
    context_root = tmp_path / "context"
    context_root.mkdir()

    tricky_goal = "Use ${weird} and {curly} and %s format specifiers verbatim."
    tricky_criterion = "Criterion with ${nested} braces and {more} and %d specifiers"
    ticket_row = {
        "required_reading": [],
        "goal": tricky_goal,
        "acceptance_criteria": [tricky_criterion],
    }
    dest = tmp_path / "task_pack_case10"

    template = "Goal:\n$goal\n\nAcceptance criteria:\n$acceptance_criteria\n"
    assemble_task_pack(
        ticket_row,
        context_root=context_root,
        repo_root=tmp_path / "repo",
        prompt_template=template,
        dest=dest,
    )

    rendered = (dest / "prompt.md").read_text()
    assert tricky_goal in rendered
    assert tricky_criterion in rendered


# ==========================================================================
# Per-dispatch clone. Cases 11-12.
# ==========================================================================


def test_prior_evidence_renders_into_prompt(tmp_path: Path) -> None:
    # bead .29: a retry dispatch's PriorEvidence renders into the $prior_evidence
    # slot as DATA describing the prior failure (kind + critic reason + note).
    context_root = tmp_path / "context"
    context_root.mkdir()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    dest = tmp_path / "task_pack"
    prior = PriorEvidence(
        attempt_kind="critic-revision",
        critic_reason="rubric item 2 UNMET: missing null check",
        failure_note="gate-unmet",
    )
    assemble_task_pack(
        {"goal": "g"},
        context_root=context_root,
        repo_root=repo_root,
        prompt_template="EVIDENCE:\n$prior_evidence",
        dest=dest,
        prior_evidence=prior,
    )
    text = (dest / "prompt.md").read_text()
    assert "did NOT land" in text
    assert "critic-revision" in text
    assert "rubric item 2 UNMET: missing null check" in text
    assert "gate-unmet" in text


def test_prior_evidence_renders_failing_check_output(tmp_path: Path) -> None:
    # bead .29: a tier1-repair retry surfaces the prior failing check output.
    context_root = tmp_path / "context"
    context_root.mkdir()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    dest = tmp_path / "task_pack"
    prior = PriorEvidence(
        attempt_kind="tier1-repair",
        check_output="tests/test_foo.py::test_bar FAILED\nAssertionError: 1 != 2",
    )
    assemble_task_pack(
        {"goal": "g"},
        context_root=context_root,
        repo_root=repo_root,
        prompt_template="$prior_evidence",
        dest=dest,
        prior_evidence=prior,
    )
    text = (dest / "prompt.md").read_text()
    assert "test_bar FAILED" in text
    assert "AssertionError: 1 != 2" in text


def test_no_prior_evidence_renders_first_attempt(tmp_path: Path) -> None:
    # bead .29: an initial dispatch (no prior evidence) says so plainly.
    context_root = tmp_path / "context"
    context_root.mkdir()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    dest = tmp_path / "task_pack"
    assemble_task_pack(
        {"goal": "g"},
        context_root=context_root,
        repo_root=repo_root,
        prompt_template="EVIDENCE:\n$prior_evidence",
        dest=dest,
    )
    text = (dest / "prompt.md").read_text()
    assert "first attempt" in text


def test_real_code01_has_prior_evidence_slot() -> None:
    # bead .29: the reviewed code01 artifact carries the $prior_evidence slot
    # (guards against the slot being dropped; the D13 prompt change is
    # flagged for operator sign-off before live use).
    code01 = Path(__file__).resolve().parents[1] / "prompts" / "code01"
    assert "$prior_evidence" in code01.read_text(encoding="utf-8")


def test_case11_create_work_clone_shape_and_no_shared_objects(tmp_path: Path) -> None:
    source = make_repo(tmp_path, branch="staging")
    dispatch_base_tip = _git(source, ["rev-parse", "staging"]).stdout.strip()

    dest = tmp_path / "clones" / "disp-1" / "work"
    create_work_clone(source, dest, dispatch_base="staging")

    current_branch = _git(dest, ["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
    assert current_branch == "work"

    work_tip = _git(dest, ["rev-parse", "HEAD"]).stdout.strip()
    assert work_tip == dispatch_base_tip

    # --no-hardlinks: the clone must share no object files with source.
    blob_oid = _git(source, ["rev-parse", "staging:README.md"]).stdout.strip()
    sub, rest = blob_oid[:2], blob_oid[2:]
    source_obj = source / ".git" / "objects" / sub / rest
    dest_obj = dest / ".git" / "objects" / sub / rest
    assert source_obj.is_file()
    assert dest_obj.is_file()
    assert os.stat(source_obj).st_ino != os.stat(dest_obj).st_ino


def test_case12_clone_of_invalid_repo_raises_dispatch_error(tmp_path: Path) -> None:
    nonexistent = tmp_path / "does-not-exist"
    dest = tmp_path / "clones" / "disp-2" / "work"

    with pytest.raises(DispatchError):
        create_work_clone(nonexistent, dest, dispatch_base="staging")


# ==========================================================================
# Worker naming / uniqueness. Cases 13-17.
# ==========================================================================


def test_case13_worker_name_shape(tmp_path: Path) -> None:
    store = RigStore.create(tmp_path / "tickets.db")
    try:
        rng = _sequence_rng(["broom", "casino", "flock"])
        name = generate_worker_name(store, "worker", "haiku", "code01", rng=rng)

        assert name == "worker-haiku-code01-broom-casino-flock"
        parts = name.split("-")
        assert len(parts) == 6
        assert parts[0] == "worker"
        assert parts[1] == "haiku"
        assert parts[2] == "code01"
    finally:
        store.close()


def test_case14_many_calls_never_collide(tmp_path: Path) -> None:
    store = RigStore.create(tmp_path / "tickets.db")
    try:
        names = [generate_worker_name(store, "worker", "haiku", "code01") for _ in range(500)]

        assert len(set(names)) == 500
        for name in names:
            assert store.worker_name_exists(name)
    finally:
        store.close()


def test_case15_collision_forces_retry_and_returns_different_name(tmp_path: Path) -> None:
    store = RigStore.create(tmp_path / "tickets.db")
    try:
        colliding_name = "worker-haiku-code01-alpha-beta-gamma"
        store.insert_worker_name(colliding_name)

        rng = _sequence_rng(["alpha", "beta", "gamma", "delta", "epsilon", "zeta"])
        name = generate_worker_name(store, "worker", "haiku", "code01", rng=rng)

        assert name != colliding_name
        assert name == "worker-haiku-code01-delta-epsilon-zeta"
        assert store.worker_name_exists(name)
    finally:
        store.close()


def test_case16_retry_budget_exhaustion_raises_dispatch_error(tmp_path: Path) -> None:
    store = RigStore.create(tmp_path / "tickets.db")
    try:
        # Every candidate the constant rng can ever produce is the same
        # name; pre-seed it so every attempt collides.
        colliding_name = "worker-haiku-code01-same-same-same"
        store.insert_worker_name(colliding_name)

        rng = _constant_rng("same")
        with pytest.raises(DispatchError):
            generate_worker_name(store, "worker", "haiku", "code01", rng=rng)
    finally:
        store.close()


def test_case16_retry_limit_is_bounded_constant() -> None:
    # Bounds the test itself: the retry cap is a small fixed constant, not
    # an unbounded loop.
    assert 0 < _NAME_RETRY_LIMIT <= 100


def test_case17_vendored_wordlist_has_exactly_2048_unique_lowercase_words() -> None:
    assert _WORDLIST_PATH.is_file()
    words = _WORDLIST_PATH.read_text(encoding="utf-8").splitlines()

    assert len(words) == 2048
    assert len(set(words)) == 2048
    for word in words:
        assert word == word.lower()
        assert word.isalpha()


# ==========================================================================
# Capability minting / end-to-end prepare_dispatch. Cases 18-20.
# ==========================================================================


class _SpyCapabilityStore(CapabilityStore):
    """Counts calls to `mint` (case 20) without changing behavior."""

    def __init__(self) -> None:
        super().__init__()
        self.mint_calls = 0

    def mint(self, *args, **kwargs):  # type: ignore[override]
        self.mint_calls += 1
        return super().mint(*args, **kwargs)


def _prepare_test_dispatch(tmp_path: Path, *, capability_store=None, **prepare_kwargs):
    charter = make_full_charter(tmp_path)
    rig_repo = make_repo(tmp_path, name="rig_repo", branch="staging")
    store = RigStore.create(tmp_path / "tickets.db")

    context_root = tmp_path / "context"
    context_root.mkdir()
    clones_root = tmp_path / "clones"
    prompts_dir = tmp_path / "prompts_dir"
    prompts_dir.mkdir()
    (prompts_dir / "code01").write_text(
        "Goal:\n$goal\n\nAcceptance criteria:\n$acceptance_criteria\n"
        "Tier 1 checks:\n$tier1_checks\n"
    )

    ticket_row = {
        "id": "ticket-1",
        "title": "Do the thing",
        "goal": "Implement the feature end to end.",
        "required_reading": [],
        "target_scope": ["src/foo.py"],
        "acceptance_criteria": ["foo() returns 42"],
        "tier1_checks": {"pytest": "pytest -q"},
        "lane_hint": None,
    }

    cap_store = capability_store if capability_store is not None else CapabilityStore()

    plan = prepare_dispatch(
        charter=charter,
        ticket_row=ticket_row,
        store=store,
        capability_store=cap_store,
        rig_repo=rig_repo,
        context_root=context_root,
        clones_root=clones_root,
        prompts_dir=prompts_dir,
        relay_base_url="http://relay.local:9999",
        image=PINNED_IMAGE,
        **prepare_kwargs,
    )
    return charter, plan, store, cap_store


def test_prepare_dispatch_threads_relay_socket(tmp_path: Path) -> None:
    # bead .25: the daemon's per-dispatch relay socket path flows into the
    # DispatchPlan's ModelConfig so spawn mounts it into the cage.
    sock = tmp_path / "relay.sock"
    _charter, plan, store, _cap = _prepare_test_dispatch(tmp_path, relay_socket=sock)
    try:
        assert plan.model_cfg.relay_socket == sock
    finally:
        store.close()


def test_prepare_dispatch_relay_socket_defaults_none(tmp_path: Path) -> None:
    _charter, plan, store, _cap = _prepare_test_dispatch(tmp_path)
    try:
        assert plan.model_cfg.relay_socket is None
    finally:
        store.close()


def test_case18_prepare_dispatch_quota_invariant(tmp_path: Path) -> None:
    charter, plan, store, _cap_store = _prepare_test_dispatch(tmp_path)
    try:
        dispatch_limits = charter.raw["loop"]["dispatch_limits"]
        assert plan.capability.max_calls == dispatch_limits["driver_turns"]
        assert plan.capability.max_output_tokens == dispatch_limits["output_tokens"]
        assert plan.budgets.driver_turns == dispatch_limits["driver_turns"]
        assert plan.budgets.output_tokens == dispatch_limits["output_tokens"]
    finally:
        store.close()


def test_case18b_prepare_dispatch_sets_prompt_artifact_hash(tmp_path: Path) -> None:
    """bead .22 build spec §2 additive extension: `DispatchPlan.
    prompt_artifact_hash` is `sha256(prompt_template.encode()).hexdigest()`
    — a non-empty, required field, never silently defaulted to `""`."""
    import hashlib

    _charter, plan, store, _cap_store = _prepare_test_dispatch(tmp_path)
    try:
        prompt_template = (
            "Goal:\n$goal\n\nAcceptance criteria:\n$acceptance_criteria\n"
            "Tier 1 checks:\n$tier1_checks\n"
        )
        expected = hashlib.sha256(prompt_template.encode()).hexdigest()
        assert plan.prompt_artifact_hash == expected
        assert isinstance(plan.prompt_artifact_hash, str)
        assert plan.prompt_artifact_hash != ""
    finally:
        store.close()


def test_case19_work_clone_and_task_pack_satisfy_spawn_contract(tmp_path: Path) -> None:
    _charter, plan, store, _cap_store = _prepare_test_dispatch(tmp_path)
    try:
        assert plan.work_clone.is_dir()
        assert (plan.work_clone / ".git").is_dir()
        assert plan.task_pack.is_dir()
        assert (plan.task_pack / "prompt.md").is_file()

        current_branch = _git(plan.work_clone, ["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
        assert current_branch == "work"

        runner = _CapturingRunOne(stdout=_success_json())
        result = claude_code.spawn(
            plan.task_pack,
            plan.work_clone,
            plan.model_cfg,
            plan.capability,
            plan.budgets,
            run_one=runner,
        )
        # No DriverError/ContainerError raised for a structural reason --
        # prepare_dispatch's output genuinely satisfies .13's input contract.
        assert result.status is claude_code.DispatchStatus.DONE
        assert len(runner.calls) == 1
    finally:
        store.close()


def test_case20_prepare_dispatch_mints_capability_exactly_once(tmp_path: Path) -> None:
    spy = _SpyCapabilityStore()
    _charter, _plan, store, cap_store = _prepare_test_dispatch(tmp_path, capability_store=spy)
    try:
        assert cap_store.mint_calls == 1
    finally:
        store.close()


# ==========================================================================
# Explicit rung override (bead .28) — select_lane / prepare_dispatch must be
# able to dispatch a stepped-up (entry=false) rung named by current_rung,
# WITHOUT changing the plain lane_hint-only default path (the frozen .35
# constraint: derive_steering keeps calling select_lane(charter, ticket_row)
# with no rung, and must never see current_rung).
# ==========================================================================


def _ladder_charter() -> Charter:
    return _lanes_charter(
        {
            "cheap": _lane(model="haiku", selector={"label": "local-ok"}),
            "default": _lane(model="sonnet"),
            "exquisite": _lane(model="opus", entry=False),
        }
    )


def test_rung_override_selects_entry_false_lane() -> None:
    """THE .28 fix: an explicit rung reaches a step-up-only (entry=false)
    lane that select_lane's default path can never select."""
    selection = select_lane(_ladder_charter(), {"lane_hint": None}, rung="exquisite")
    assert selection.name == "exquisite"
    assert selection.model == "opus"


def test_rung_override_takes_precedence_over_lane_hint() -> None:
    # lane_hint alone would pick "cheap"; the explicit rung wins.
    selection = select_lane(_ladder_charter(), {"lane_hint": "local-ok"}, rung="exquisite")
    assert selection.name == "exquisite"


def test_rung_override_selects_entry_lane_by_name() -> None:
    # Names an entry lane directly, bypassing selector-label matching.
    selection = select_lane(_ladder_charter(), {}, rung="cheap")
    assert selection.name == "cheap"
    assert selection.model == "haiku"


def test_rung_none_or_absent_preserves_lane_hint_behavior() -> None:
    """Regression / the frozen constraint: rung=None (and absent rung) behave
    EXACTLY as pre-.28 — lane_hint-only, entry=false lanes never reachable."""
    charter = _ladder_charter()
    assert select_lane(charter, {"lane_hint": "local-ok"}, rung=None).name == "cheap"
    assert select_lane(charter, {"lane_hint": "local-ok"}).name == "cheap"
    assert select_lane(charter, {"lane_hint": None}, rung=None).name == "default"
    assert select_lane(charter, {}).name == "default"


def test_rung_override_unknown_lane_raises() -> None:
    with pytest.raises(DispatchError):
        select_lane(_ladder_charter(), {}, rung="ghost-rung")


def _prepare_with_rung(tmp_path: Path, current_rung):
    """Minimal prepare_dispatch setup (mirrors _prepare_test_dispatch) with a
    caller-controlled current_rung."""
    charter = make_full_charter(tmp_path)  # lanes: cheap/default/exquisite(entry=false)
    rig_repo = make_repo(tmp_path, name="rig_repo", branch="staging")
    store = RigStore.create(tmp_path / "tickets.db")
    context_root = tmp_path / "context"
    context_root.mkdir()
    clones_root = tmp_path / "clones"
    prompts_dir = tmp_path / "prompts_dir"
    prompts_dir.mkdir()
    (prompts_dir / "code01").write_text(
        "Goal:\n$goal\n\nAcceptance criteria:\n$acceptance_criteria\n"
        "Tier 1 checks:\n$tier1_checks\n"
    )
    ticket_row = {
        "id": "ticket-1",
        "title": "Do the thing",
        "goal": "Implement the feature end to end.",
        "required_reading": [],
        "target_scope": ["src/foo.py"],
        "acceptance_criteria": ["foo() returns 42"],
        "tier1_checks": {"pytest": "pytest -q"},
        "lane_hint": None,
        "current_rung": current_rung,
    }
    plan = prepare_dispatch(
        charter=charter,
        ticket_row=ticket_row,
        store=store,
        capability_store=CapabilityStore(),
        rig_repo=rig_repo,
        context_root=context_root,
        clones_root=clones_root,
        prompts_dir=prompts_dir,
        relay_base_url="http://relay.local:9999",
        image=PINNED_IMAGE,
    )
    return plan, store


def test_prepare_dispatch_dispatches_on_stepped_up_rung(tmp_path: Path) -> None:
    """End-to-end: a ticket that has stepped up (current_rung='exquisite',
    an entry=false lane) actually dispatches on the exquisite lane — the
    bug this bead fixes was that it silently fell back to the entry lane."""
    plan, store = _prepare_with_rung(tmp_path, "exquisite")
    try:
        assert plan.lane.name == "exquisite"
        assert plan.lane.model == "opus"
    finally:
        store.close()


def test_prepare_dispatch_first_dispatch_uses_entry_lane(tmp_path: Path) -> None:
    """Regression: current_rung=None (first dispatch) still resolves via the
    plain lane_hint path — here, the selector-less entry fallthrough."""
    plan, store = _prepare_with_rung(tmp_path, None)
    try:
        assert plan.lane.name == "default"
    finally:
        store.close()
