"""Tests for stigmergy.weaver (SPEC.md §3 `weave` station, §6 "Protection
asymmetry", §9 "Weave triggers" + "Crash recovery", §10 AC6/AC8/AC9).

Case numbering below matches the bead .18 design doc's exact case list
(design §2, cases 1-16, grouped AC8/AC6/AC9/protection-asymmetry/ordering).
Uses REAL local git repos in `tmp_path` (host-safe: init, commit, bundle,
clone, fetch, update-ref are all real `git` subprocess calls over throwaway
directories) — only `run_checks_fn` (no podman) and the critic `client`
(no live model) are faked, per the design doc's explicit instruction.

**Event-phase vocabulary note** (see `weaver.py`'s `_die`/`_weave_one`
docstrings): the weaver writes a record-plane INTEGRATION event at phase
`"apply"` for EVERY ticket whose bundle applies cleanly (independent of
what happens next at checks/gate), plus additional INTEGRATION events at
`"land"` (successful CAS), `"abort"` (CAS lost the race), and
`"gate-infra"` (critic-call failure, SPEC §1.3 step 7's explicit
"journal an INTEGRATION 'gate-infra' entry" instruction). A plain
checks-red/gate-rejected/conflict rejection therefore still has an
`"apply"` INTEGRATION event (or none at all, for a conflict, since the
bundle never applied) — tests assert on the ABSENCE of a `"land"`-phase
INTEGRATION event specifically, never on the total absence of any
INTEGRATION event, so this is never a spurious failure against the
design doc's "no land/INTEGRATION event" wording for a rejection.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from stigmergy.checks import CheckOutcome, CheckResources, CheckResult
from stigmergy.records import RecordPlane
from stigmergy.rig import RigStore
from stigmergy.statemachine import GATED, LANDED, PARKED, REJECTED, FailureClass
from stigmergy.weaver import Weaver

GIT_ENV_CFG = [
    "-c",
    "user.email=fixture@example.com",
    "-c",
    "user.name=Fixture User",
]


# --------------------------------------------------------------------------
# git fixture helpers (REAL git, host-safe: everything lives under tmp_path)
# --------------------------------------------------------------------------


def run_git(repo: Path | None, args: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    argv = ["git"]
    if repo is not None:
        argv += ["-C", str(repo)]
    argv += GIT_ENV_CFG + args
    result = subprocess.run(argv, capture_output=True, text=True, check=False, **kwargs)
    assert result.returncode == 0, f"git {args} failed: {result.stderr}"
    return result


def rev_parse(repo: Path, ref: str) -> str:
    return run_git(repo, ["rev-parse", ref]).stdout.strip()


def make_staging_repo(tmp_path: Path, name: str = "staging_repo") -> Path:
    """A BARE loop-owned integration repo whose `staging` branch starts at
    one base commit. Bare so there is no checked-out worktree/index for
    `update-ref` to leave stale, and no `receive.denyCurrentBranch`
    wrinkle — matches the "weaver-only-writer" model (SPEC §3)."""
    staging_repo = tmp_path / name
    seed = tmp_path / f"{name}-seed"
    run_git(None, ["init", "--quiet", "--bare", str(staging_repo)])
    run_git(None, ["init", "--quiet", "-b", "staging", str(seed)])
    (seed / "README.md").write_text("base content\n")
    run_git(seed, ["add", "README.md"])
    run_git(seed, ["commit", "--quiet", "-m", "base"])
    run_git(seed, ["remote", "add", "origin", str(staging_repo)])
    run_git(seed, ["push", "--quiet", "origin", "staging"])
    return staging_repo


def make_bundle(
    tmp_path: Path,
    staging_repo: Path,
    *,
    name: str,
    files: dict[str, str],
    base_ref: str = "refs/heads/staging",
) -> Path:
    """Build a real `git bundle` file carrying `refs/heads/work` — a
    worker clone off `staging_repo`'s current tip, with `files` written/
    committed on top. This is the ONLY ref the weaver ever fetches from a
    bundle (SPEC §10 AC6 case 9)."""
    clone_dir = tmp_path / f"{name}-clone"
    run_git(None, ["clone", "--quiet", "--", str(staging_repo), str(clone_dir)])
    run_git(clone_dir, ["fetch", "--quiet", "origin", "staging"])
    run_git(clone_dir, ["checkout", "--quiet", "-b", "work", "origin/staging"])
    for rel, content in files.items():
        path = clone_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        run_git(clone_dir, ["add", rel])
    run_git(clone_dir, ["commit", "--quiet", "-m", f"worker change for {name}"])
    bundle_path = tmp_path / f"{name}.bundle"
    run_git(clone_dir, ["bundle", "create", str(bundle_path), "refs/heads/work"])
    return bundle_path


def make_conflicting_bundle(tmp_path: Path, staging_repo: Path, *, name: str) -> Path:
    """A bundle guaranteed to conflict: an unrelated orphan history that
    touches the same tracked file, so `git merge` refuses (either
    "refusing to merge unrelated histories" or a real content conflict —
    both are non-zero exits `_apply_bundle` treats identically)."""
    orphan_dir = tmp_path / f"{name}-orphan"
    run_git(None, ["init", "--quiet", "-b", "work", str(orphan_dir)])
    (orphan_dir / "README.md").write_text("an entirely unrelated orphan history\n")
    run_git(orphan_dir, ["add", "README.md"])
    run_git(orphan_dir, ["commit", "--quiet", "-m", "orphan"])
    bundle_path = tmp_path / f"{name}.bundle"
    run_git(orphan_dir, ["bundle", "create", str(bundle_path), "refs/heads/work"])
    return bundle_path


# --------------------------------------------------------------------------
# fake collaborators
# --------------------------------------------------------------------------


class FakeRunChecks:
    """A scripted stand-in for `checks.run_checks`: returns a fixed list of
    `CheckResult`s regardless of input, unless `by_tree` is given, in which
    case it inspects the candidate work tree's content to decide (used for
    case 5's seeded semantic-conflict pair, where "red" depends on what
    already landed onto staging)."""

    def __init__(
        self,
        results: list[CheckResult] | None = None,
        *,
        by_tree: Any = None,
    ):
        self._results = results
        self._by_tree = by_tree
        self.calls: list[Any] = []
        self.resources_seen: Any = None

    def __call__(
        self,
        checks: dict[str, str],
        work_tree: Path,
        *,
        image: str,
        flake_reruns: int,
        resources: Any = None,
    ):
        self.calls.append(work_tree)
        self.resources_seen = resources
        if self._by_tree is not None:
            return self._by_tree(work_tree)
        return list(self._results)


def green_result(name: str = "tests") -> CheckResult:
    return CheckResult(
        name=name, outcome=CheckOutcome.PASS, runs=(0,), output="ok", wall_time_seconds=0.1
    )


def result_with_outcome(outcome: CheckOutcome, name: str = "tests") -> CheckResult:
    runs = (1,) if outcome in (CheckOutcome.FAIL, CheckOutcome.ERROR) else (1, 0)
    return CheckResult(
        name=name, outcome=outcome, runs=runs, output="scripted", wall_time_seconds=0.1
    )


class StubCriticClient:
    def __init__(self, *, response: Any = None, raises: Exception | None = None):
        self._response = response
        self._raises = raises
        self.calls: list[dict] = []

    def __call__(self, prompt: str, *, model: str, **kwargs: Any):
        self.calls.append({"prompt": prompt, "model": model, "kwargs": kwargs})
        if self._raises is not None:
            raise self._raises
        return self._response


class SpyCritic:
    """Wraps a real `Critic` (or a scripted one) and records whether
    `.judge` was ever called — used to assert critic.judge is NEVER
    called for conflict/checks-red outcomes (cases 3/4)."""

    def __init__(self, inner: Any):
        self._inner = inner
        self.judge_called = False

    def judge(self, artifact: str, rubric_items: list[str], *, check_evidence: str | None = None):
        self.judge_called = True
        return self._inner.judge(artifact, rubric_items, check_evidence=check_evidence)


def make_critic(
    *,
    outcome: str = "met",
    severity: str = "none",
    reason: str = "looks good",
    filed_tickets: list | None = None,
):
    from stigmergy.critic import Critic

    response = {"outcome": outcome, "tier": 1, "reason": reason, "severity": severity}
    if filed_tickets is not None:
        response["filed_tickets"] = filed_tickets  # bead .39: optional filing channel
    client = StubCriticClient(response=response)
    return Critic(
        client=client,
        model="stub-model",
        decoding_params={"temperature": 0.0},
        template="Judge the artifact.",
    )


def make_infra_critic():
    from stigmergy.critic import Critic

    client = StubCriticClient(raises=RuntimeError("provider 503"))
    return Critic(
        client=client,
        model="stub-model",
        decoding_params={"temperature": 0.0},
        template="Judge the artifact.",
    )


# --------------------------------------------------------------------------
# store / plane / weaver fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> RigStore:
    s = RigStore.create(tmp_path / "tickets.db")
    yield s
    s.close()


@pytest.fixture
def plane(tmp_path: Path) -> RecordPlane:
    return RecordPlane(tmp_path / "records")


def add_parked_ticket(
    store: RigStore,
    ticket_id: str,
    *,
    work_product: str | Path,
    tier1_checks: dict | None = None,
    target_scope: list[str] | None = None,
) -> None:
    store.add_ticket(
        id=ticket_id,
        title=f"Ticket {ticket_id}",
        state=PARKED,
        work_product=str(work_product),
        tier1_checks=tier1_checks or {"tests": "true"},
        acceptance_criteria=["does the thing"],
        target_scope=target_scope,
    )


def stub_ctx_of(ticket_id: str) -> dict[str, Any]:
    return {
        "rig": "test-rig",
        "dispatch_id": f"dispatch-{ticket_id}",
        "attempt": 1,
        "attempt_kind": "initial",
        "rung": "cheap",
        "worker": "worker-1",
        "charter_hash": "charterhash123",
        "approval_hash": "approvalhash456",
        "image_digest": "sha256:deadbeef",
        "model": "haiku",
        "model_version": "haiku-3-5",
        "price_table_version": "modelshash789",
        "tokens": {"in": 0, "cached": 0, "out": 0, "reasoning": 0},
        "computed_usd": 0.0,
        "wall_time_seconds": 0.0,
    }


def make_weaver(
    tmp_path: Path,
    store: RigStore,
    plane: RecordPlane,
    staging_repo: Path,
    *,
    run_checks_fn: Any,
    critic: Any,
    protected_paths: list[str] | None = None,
    journal_name: str = "weave-journal.jsonl",
    filing_max_filings: int = 5,
    filing_max_bytes: int = 16384,
    check_resources_fn: Any = None,
) -> Weaver:
    return Weaver(
        store=store,
        record_plane=plane,
        staging_repo=staging_repo,
        run_checks_fn=run_checks_fn,
        critic=critic,
        checker_image="unused:image",
        flake_reruns=0,
        protected_paths=protected_paths or [],
        journal_path=tmp_path / journal_name,
        ctx_of=stub_ctx_of,
        filing_max_filings=filing_max_filings,
        filing_max_bytes=filing_max_bytes,
        check_resources_fn=check_resources_fn,
    )


def integration_events(
    plane: RecordPlane, ticket_id: str, phase: str | None = None
) -> list[dict]:
    events = [
        e
        for e in plane.read_events()
        if e["event_type"] == "integration" and e["ticket"] == ticket_id
    ]
    if phase is not None:
        events = [e for e in events if e.get("phase") == phase]
    return events


def gate_events(plane: RecordPlane, ticket_id: str) -> list[dict]:
    return [
        e for e in plane.read_events() if e["event_type"] == "gate" and e["ticket"] == ticket_id
    ]


def disposition_events(plane: RecordPlane, ticket_id: str) -> list[dict]:
    return [
        e
        for e in plane.read_events()
        if e["event_type"] == "disposition" and e["ticket"] == ticket_id
    ]


# ==========================================================================
# AC8 -- gate-then-land integrity
# ==========================================================================


def test_met_verdict_lands_on_staging(tmp_path, store, plane):
    """1. test_met_verdict_lands_on_staging"""
    staging_repo = make_staging_repo(tmp_path)
    pinned = rev_parse(staging_repo, "refs/heads/staging")
    bundle = make_bundle(tmp_path, staging_repo, name="t1", files={"feature.txt": "new feature\n"})
    add_parked_ticket(store, "t1", work_product=bundle)

    weaver = make_weaver(
        tmp_path,
        store,
        plane,
        staging_repo,
        run_checks_fn=FakeRunChecks([green_result()]),
        critic=make_critic(outcome="met"),
    )
    results = weaver.weave(now=1000.0)

    assert len(results) == 1
    result = results[0]
    assert result.outcome == "landed"
    assert result.landed_oid is not None
    assert result.failure_class is None

    new_tip = rev_parse(staging_repo, "refs/heads/staging")
    assert new_tip == result.landed_oid
    assert new_tip != pinned

    log = run_git(staging_repo, ["log", "refs/heads/staging", "--oneline"]).stdout
    assert result.landed_oid[:7] in log or new_tip in log  # commit is on staging

    assert store.get_ticket("t1")["state"] == LANDED

    assert len(gate_events(plane, "t1")) == 1
    assert len(integration_events(plane, "t1", phase="land")) == 1
    dispositions = disposition_events(plane, "t1")
    assert len(dispositions) == 1
    assert dispositions[0]["disposition"] == "landed"


def test_check_resources_fn_threads_per_check_bounds(tmp_path, store, plane):
    """bead .91: with a resolver wired, the weaver's staging-gate full-suite
    re-run passes a per-check CheckResources map to run_checks — so charter-
    configured bounds apply at the staging gate, not just the daemon attempt
    gate. A dropped `resources=` on the weaver's run_checks call surfaces here.
    """
    staging_repo = make_staging_repo(tmp_path)
    bundle = make_bundle(tmp_path, staging_repo, name="t1", files={"feature.txt": "x\n"})
    add_parked_ticket(store, "t1", work_product=bundle)
    run_checks = FakeRunChecks([green_result()])
    weaver = make_weaver(
        tmp_path,
        store,
        plane,
        staging_repo,
        run_checks_fn=run_checks,
        critic=make_critic(outcome="met"),
        check_resources_fn=lambda name: CheckResources(timeout_seconds=1234),
    )
    weaver.weave(now=1000.0)

    assert run_checks.resources_seen is not None
    assert all(isinstance(v, CheckResources) for v in run_checks.resources_seen.values())
    assert all(v.timeout_seconds == 1234 for v in run_checks.resources_seen.values())


def test_unmet_verdict_never_lands(tmp_path, store, plane):
    """2. test_unmet_verdict_never_lands"""
    staging_repo = make_staging_repo(tmp_path)
    pinned = rev_parse(staging_repo, "refs/heads/staging")
    bundle = make_bundle(
        tmp_path, staging_repo, name="t2", files={"feature.txt": "rejected feature\n"}
    )
    add_parked_ticket(store, "t2", work_product=bundle)

    weaver = make_weaver(
        tmp_path,
        store,
        plane,
        staging_repo,
        run_checks_fn=FakeRunChecks([green_result()]),
        critic=make_critic(outcome="unmet", reason="missing tests"),
    )
    results = weaver.weave(now=1000.0)

    result = results[0]
    assert result.outcome == "rejected"
    assert result.failure_class is FailureClass.REJECTED
    assert result.landed_oid is None

    new_tip = rev_parse(staging_repo, "refs/heads/staging")
    assert new_tip == pinned  # staging UNCHANGED

    log = run_git(staging_repo, ["log", "refs/heads/staging", "--all", "--oneline"]).stdout
    assert "rejected feature" not in run_git(
        staging_repo, ["log", "refs/heads/staging", "-p"]
    ).stdout  # commit absent from staging's history

    assert store.get_ticket("t2")["state"] == REJECTED

    # The GATE event IS recorded (the verdict happened)...
    gates = gate_events(plane, "t2")
    assert len(gates) == 1
    assert gates[0]["outcome"] == "unmet"
    # ...but there is NO land-phase INTEGRATION event (see module docstring
    # re: the apply/land/abort/gate-infra event-phase vocabulary).
    assert integration_events(plane, "t2", phase="land") == []
    del log


def test_integration_conflict_dies_before_gate(tmp_path, store, plane):
    """3. test_integration_conflict_dies_before_gate"""
    staging_repo = make_staging_repo(tmp_path)
    pinned = rev_parse(staging_repo, "refs/heads/staging")
    bundle = make_conflicting_bundle(tmp_path, staging_repo, name="t3")
    add_parked_ticket(store, "t3", work_product=bundle)

    critic = make_critic(outcome="met")
    spy = SpyCritic(critic)
    run_checks = FakeRunChecks([green_result()])
    weaver = make_weaver(tmp_path, store, plane, staging_repo, run_checks_fn=run_checks, critic=spy)
    results = weaver.weave(now=1000.0)

    result = results[0]
    assert result.outcome == "integration-conflict"
    assert result.failure_class is FailureClass.INTEGRATION_CONFLICT

    assert spy.judge_called is False
    assert run_checks.calls == []  # checks never run

    assert rev_parse(staging_repo, "refs/heads/staging") == pinned
    assert store.get_ticket("t3")["state"] == REJECTED


def test_integration_regression_dies_at_checks(tmp_path, store, plane):
    """4. test_integration_regression_dies_at_checks"""
    staging_repo = make_staging_repo(tmp_path)
    pinned = rev_parse(staging_repo, "refs/heads/staging")
    bundle = make_bundle(tmp_path, staging_repo, name="t4", files={"feature.txt": "x\n"})
    add_parked_ticket(store, "t4", work_product=bundle)

    critic = make_critic(outcome="met")
    spy = SpyCritic(critic)
    run_checks = FakeRunChecks([result_with_outcome(CheckOutcome.FAIL)])
    weaver = make_weaver(tmp_path, store, plane, staging_repo, run_checks_fn=run_checks, critic=spy)
    results = weaver.weave(now=1000.0)

    result = results[0]
    assert result.outcome == "integration-regression"
    assert result.failure_class is FailureClass.INTEGRATION_REGRESSION

    assert spy.judge_called is False
    assert rev_parse(staging_repo, "refs/heads/staging") == pinned
    assert store.get_ticket("t4")["state"] == REJECTED


def test_seeded_semantic_conflict_pair_second_dies_at_gate(tmp_path, store, plane):
    """5. test_seeded_semantic_conflict_pair_second_dies_at_gate"""
    staging_repo = make_staging_repo(tmp_path)
    bundle_a = make_bundle(tmp_path, staging_repo, name="t5a", files={"shared.txt": "from A\n"})
    bundle_b = make_bundle(tmp_path, staging_repo, name="t5b", files={"other.txt": "from B\n"})
    add_parked_ticket(store, "t5a", work_product=bundle_a)
    add_parked_ticket(store, "t5b", work_product=bundle_b)

    # Both are green in isolation; checks go RED for the SECOND ticket only
    # once its candidate tree already contains A's file (i.e. after A has
    # landed onto the pinned base the second candidate is built from).
    def by_tree(work_tree: Path):
        if (Path(work_tree) / "shared.txt").exists() and (Path(work_tree) / "other.txt").exists():
            return [result_with_outcome(CheckOutcome.FAIL)]
        return [green_result()]

    run_checks = FakeRunChecks(by_tree=by_tree)
    weaver = make_weaver(
        tmp_path,
        store,
        plane,
        staging_repo,
        run_checks_fn=run_checks,
        critic=make_critic(outcome="met"),
    )
    results = weaver.weave(now=1000.0)

    assert len(results) == 2
    first, second = results
    assert first.ticket == "t5a"
    assert first.outcome == "landed"
    assert second.ticket == "t5b"
    assert second.outcome == "integration-regression"
    assert second.failure_class is FailureClass.INTEGRATION_REGRESSION

    tip = rev_parse(staging_repo, "refs/heads/staging")
    assert tip == first.landed_oid
    diff = run_git(staging_repo, ["show", tip, "--stat"]).stdout
    assert "shared.txt" in diff
    assert "other.txt" not in diff  # second never landed

    assert store.get_ticket("t5a")["state"] == LANDED
    assert store.get_ticket("t5b")["state"] == REJECTED


def test_cas_abort_on_concurrent_staging_move(tmp_path, store, plane):
    """6. test_cas_abort_on_concurrent_staging_move"""
    staging_repo = make_staging_repo(tmp_path)
    pinned = rev_parse(staging_repo, "refs/heads/staging")
    bundle = make_bundle(tmp_path, staging_repo, name="t6", files={"feature.txt": "x\n"})
    add_parked_ticket(store, "t6", work_product=bundle)

    class RacingWeaver(Weaver):
        """Subclass that simulates a concurrent `staging` move landing
        RIGHT BEFORE the CAS `update-ref` call — i.e. after the weaver's
        own pre-CAS re-read of `staging`'s tip already passed, so the
        `update-ref` old-value argument is the ONLY thing standing between
        this concurrent write and a corrupted land."""

        def _fetch_candidate_into_staging(self, candidate_dir, ticket_id):
            concurrent_dir = tmp_path / "concurrent-writer"
            run_git(None, ["clone", "--quiet", "--", str(staging_repo), str(concurrent_dir)])
            run_git(concurrent_dir, ["checkout", "--quiet", "-b", "staging", "origin/staging"])
            (concurrent_dir / "concurrent.txt").write_text("a concurrent, out-of-band commit\n")
            run_git(concurrent_dir, ["add", "concurrent.txt"])
            run_git(concurrent_dir, ["commit", "--quiet", "-m", "concurrent move"])
            run_git(concurrent_dir, ["push", "--quiet", "origin", "staging"])
            super()._fetch_candidate_into_staging(candidate_dir, ticket_id)

    weaver = RacingWeaver(
        store=store,
        record_plane=plane,
        staging_repo=staging_repo,
        run_checks_fn=FakeRunChecks([green_result()]),
        critic=make_critic(outcome="met"),
        checker_image="unused",
        flake_reruns=0,
        protected_paths=[],
        journal_path=tmp_path / "journal.jsonl",
        ctx_of=stub_ctx_of,
        filing_max_filings=5,
        filing_max_bytes=16384,
    )
    results = weaver.weave(now=1000.0)

    result = results[0]
    assert result.outcome != "landed"
    assert result.landed_oid is None

    concurrent_tip = rev_parse(staging_repo, "refs/heads/staging")
    assert concurrent_tip != pinned  # the concurrent write IS there
    log = run_git(staging_repo, ["log", "refs/heads/staging", "-p"]).stdout
    assert "a concurrent, out-of-band commit" in log
    assert "feature.txt" not in log  # candidate NOT force-landed over it

    journal_lines = weaver._read_journal()
    assert any(line.get("phase") == "abort" for line in journal_lines)

    # Ticket returned to a resting state to be re-gated later (never
    # force-landed, never left stuck in GATED).
    assert store.get_ticket("t6")["state"] in (PARKED, GATED)
    assert store.get_ticket("t6")["state"] != LANDED


def test_flaky_and_error_check_outcomes_block_landing(tmp_path, store, plane):
    """7. test_flaky_and_error_check_outcomes_block_landing"""
    staging_repo = make_staging_repo(tmp_path)
    pinned = rev_parse(staging_repo, "refs/heads/staging")

    for outcome_name, outcome in (("flaky", CheckOutcome.FLAKY), ("error", CheckOutcome.ERROR)):
        ticket_id = f"t7-{outcome_name}"
        bundle = make_bundle(
            tmp_path, staging_repo, name=ticket_id, files={f"{ticket_id}.txt": "x\n"}
        )
        add_parked_ticket(store, ticket_id, work_product=bundle)
        run_checks = FakeRunChecks([result_with_outcome(outcome)])
        weaver = make_weaver(
            tmp_path,
            store,
            plane,
            staging_repo,
            run_checks_fn=run_checks,
            critic=make_critic(outcome="met"),
            journal_name=f"journal-{outcome_name}.jsonl",
        )
        results = weaver.weave(now=1000.0)
        result = results[0]
        assert result.outcome == "integration-regression", outcome_name
        assert result.failure_class is FailureClass.INTEGRATION_REGRESSION
        assert rev_parse(staging_repo, "refs/heads/staging") == pinned
        assert store.get_ticket(ticket_id)["state"] == REJECTED


# ==========================================================================
# AC6 -- git-metadata isolation (SECURITY)
# ==========================================================================


def test_seeded_git_hook_never_fires(tmp_path, store, plane, monkeypatch):
    """8. test_seeded_git_hook_never_fires

    Seeds an executable `post-merge` hook via `GIT_TEMPLATE_DIR` (so every
    `git init` the weaver performs on a fresh candidate clone copies the
    hook into `.git/hooks/`, exactly like a worker-poisoned `.git/hooks/`
    would in a shared-worktree world) that writes a sentinel file if it
    ever runs. A full weave (bundle applies via a real, genuine merge
    commit -- the operation `post-merge` fires on) must never create the
    sentinel, because every loop-side git invocation forces
    `core.hooksPath=/dev/null`.

    A separate CONTROL sub-assertion (a bare `git merge` run WITHOUT the
    `-c core.hooksPath=/dev/null` override, against the identical seeded
    template) proves the seeded hook is genuinely live and would fire
    without the mitigation -- so this test cannot be vacuous.
    """
    template_dir = tmp_path / "hook-template"
    (template_dir / "hooks").mkdir(parents=True)
    sentinel = tmp_path / "SENTINEL"
    control_sentinel = tmp_path / "CONTROL_SENTINEL"
    hook_path = template_dir / "hooks" / "post-merge"
    hook_path.write_text(f"#!/bin/sh\necho FIRED > {sentinel}\n")
    hook_path.chmod(0o755)

    monkeypatch.setenv("GIT_TEMPLATE_DIR", str(template_dir))

    staging_repo = make_staging_repo(tmp_path)
    bundle = make_bundle(tmp_path, staging_repo, name="t8", files={"feature.txt": "x\n"})
    add_parked_ticket(store, "t8", work_product=bundle)

    weaver = make_weaver(
        tmp_path,
        store,
        plane,
        staging_repo,
        run_checks_fn=FakeRunChecks([green_result()]),
        critic=make_critic(outcome="met"),
    )
    results = weaver.weave(now=1000.0)

    assert results[0].outcome == "landed"
    assert not sentinel.exists(), "seeded hook fired despite core.hooksPath=/dev/null"

    # --- CONTROL: prove the hook is genuinely live and WOULD fire --------
    control_hook_path = template_dir / "hooks" / "post-merge"
    control_hook_path.write_text(f"#!/bin/sh\necho FIRED > {control_sentinel}\n")
    control_hook_path.chmod(0o755)

    control_repo = tmp_path / "control-repo"
    run_git(None, ["init", "--quiet", "-b", "main", str(control_repo)])
    (control_repo / "f.txt").write_text("base\n")
    run_git(control_repo, ["add", "f.txt"])
    run_git(control_repo, ["commit", "--quiet", "-m", "base"])
    run_git(control_repo, ["checkout", "--quiet", "-b", "topic"])
    (control_repo / "f.txt").write_text("topic change\n")
    run_git(control_repo, ["commit", "--quiet", "-am", "topic change"])
    run_git(control_repo, ["checkout", "--quiet", "main"])
    run_git(control_repo, ["checkout", "--quiet", "-b", "other-base", "main"])
    (control_repo / "g.txt").write_text("other\n")
    run_git(control_repo, ["add", "g.txt"])
    run_git(control_repo, ["commit", "--quiet", "-m", "other-base commit"])
    # A plain merge with NO hooksPath override -- this is the control case.
    run_git(control_repo, ["merge", "--no-ff", "--no-edit", "--quiet", "topic"])

    assert control_sentinel.exists(), (
        "control case failed: the seeded post-merge hook did not fire even "
        "WITHOUT the hooksPath=/dev/null mitigation -- this would make the "
        "main assertion above vacuous"
    )


def test_worker_bundle_cannot_reach_staging_ref_directly(tmp_path, store, plane):
    """9. test_worker_bundle_cannot_reach_staging_ref_directly"""
    staging_repo = make_staging_repo(tmp_path)
    pinned = rev_parse(staging_repo, "refs/heads/staging")

    # Build a bundle carrying BOTH a legit refs/heads/work AND its own
    # malicious refs/heads/staging pointing at unreviewed content.
    worker_clone = tmp_path / "t9-clone"
    run_git(None, ["clone", "--quiet", "--", str(staging_repo), str(worker_clone)])
    run_git(worker_clone, ["fetch", "--quiet", "origin", "staging"])
    run_git(worker_clone, ["checkout", "--quiet", "-b", "work", "origin/staging"])
    (worker_clone / "feature.txt").write_text("legit change\n")
    run_git(worker_clone, ["add", "feature.txt"])
    run_git(worker_clone, ["commit", "--quiet", "-m", "legit change"])

    run_git(worker_clone, ["checkout", "--quiet", "--orphan", "malicious-staging"])
    run_git(worker_clone, ["rm", "-rf", "--cached", "."])
    for p in worker_clone.iterdir():
        if p.name != ".git":
            import shutil as _shutil

            if p.is_dir():
                _shutil.rmtree(p)
            else:
                p.unlink()
    (worker_clone / "pwned.txt").write_text("unreviewed malicious content\n")
    run_git(worker_clone, ["add", "pwned.txt"])
    run_git(worker_clone, ["commit", "--quiet", "-m", "malicious staging move attempt"])
    # Actually create a LOCAL branch literally named `staging` pointing at
    # the malicious commit, so the bundle carries a genuine
    # `refs/heads/staging` ref (the attack SPEC §10 AC6 case 9 names).
    run_git(worker_clone, ["branch", "-f", "staging", "malicious-staging"])
    malicious_oid = rev_parse(worker_clone, "staging")

    bundle_path = tmp_path / "t9.bundle"
    run_git(
        worker_clone,
        ["bundle", "create", str(bundle_path), "refs/heads/work", "refs/heads/staging"],
    )

    add_parked_ticket(store, "t9", work_product=bundle_path)
    weaver = make_weaver(
        tmp_path,
        store,
        plane,
        staging_repo,
        run_checks_fn=FakeRunChecks([green_result()]),
        critic=make_critic(outcome="met"),
    )
    results = weaver.weave(now=1000.0)

    result = results[0]
    # The legit refs/heads/work content lands normally...
    assert result.outcome == "landed"
    new_tip = rev_parse(staging_repo, "refs/heads/staging")
    assert new_tip != malicious_oid
    assert new_tip != pinned

    # ...and the malicious ref the bundle carried is NEVER adopted anywhere:
    show_ref = run_git(staging_repo, ["show-ref"]).stdout
    assert malicious_oid not in show_ref
    log = run_git(staging_repo, ["log", "refs/heads/staging", "-p"]).stdout
    assert "pwned.txt" not in log
    assert "unreviewed malicious content" not in log


# ==========================================================================
# AC9 -- mid-weave crash / idempotency
# ==========================================================================


def _seed_gated_ticket(store: RigStore, ticket_id: str) -> None:
    store.add_ticket(id=ticket_id, title=f"Ticket {ticket_id}", state=GATED)


def test_resolve_landed_when_cas_committed_before_crash(tmp_path, store, plane):
    """10. test_resolve_landed_when_cas_committed_before_crash"""
    staging_repo = make_staging_repo(tmp_path)
    pinned = rev_parse(staging_repo, "refs/heads/staging")

    # Simulate the land HAVING happened (a prior, un-journaled-to-completion
    # `update-ref`) by directly landing a commit onto staging via the same
    # CAS mechanism the weaver itself would use.
    candidate_dir = tmp_path / "t10-candidate"
    run_git(None, ["clone", "--quiet", "--", str(staging_repo), str(candidate_dir)])
    run_git(candidate_dir, ["checkout", "--quiet", "-b", "weave-candidate", "origin/staging"])
    (candidate_dir / "landed.txt").write_text("already landed before crash\n")
    run_git(candidate_dir, ["add", "landed.txt"])
    run_git(candidate_dir, ["commit", "--quiet", "-m", "landed before crash"])
    candidate_oid = rev_parse(candidate_dir, "weave-candidate")
    run_git(
        staging_repo,
        ["fetch", "--quiet", "--", str(candidate_dir), "weave-candidate:refs/weaver/incoming/t10"],
    )
    run_git(staging_repo, ["update-ref", "refs/heads/staging", candidate_oid, pinned])
    assert rev_parse(staging_repo, "refs/heads/staging") == candidate_oid

    _seed_gated_ticket(store, "t10")
    weaver = make_weaver(
        tmp_path,
        store,
        plane,
        staging_repo,
        run_checks_fn=FakeRunChecks([green_result()]),
        critic=make_critic(),
    )
    weaver._journal_append(
        ticket="t10", dispatch_id="d-t10", phase="begin", pinned_oid=pinned, candidate_oid=None
    )
    weaver._journal_append(
        ticket="t10",
        dispatch_id="d-t10",
        phase="applied",
        pinned_oid=pinned,
        candidate_oid=candidate_oid,
    )
    # Crash here -- no "complete" line was ever written.

    assert weaver.in_progress() is True
    outcome = weaver.resolve()

    assert outcome == "landed"
    assert store.get_ticket("t10")["state"] == LANDED
    assert rev_parse(staging_repo, "refs/heads/staging") == candidate_oid  # unchanged by resolve
    assert weaver.in_progress() is False


def test_resolve_rolled_back_when_land_did_not_commit(tmp_path, store, plane):
    """11. test_resolve_rolled_back_when_land_did_not_commit"""
    staging_repo = make_staging_repo(tmp_path)
    pinned = rev_parse(staging_repo, "refs/heads/staging")

    _seed_gated_ticket(store, "t11")
    weaver = make_weaver(
        tmp_path,
        store,
        plane,
        staging_repo,
        run_checks_fn=FakeRunChecks([green_result()]),
        critic=make_critic(),
    )
    weaver._journal_append(
        ticket="t11", dispatch_id="d-t11", phase="begin", pinned_oid=pinned, candidate_oid=None
    )
    # Crash before (or at) the CAS: no "applied"/"complete" line, staging
    # never moved.

    assert weaver.in_progress() is True
    outcome = weaver.resolve()

    assert outcome == "rolled-back"
    assert rev_parse(staging_repo, "refs/heads/staging") == pinned  # UNTOUCHED
    assert store.get_ticket("t11")["state"] == PARKED
    assert weaver.in_progress() is False


def test_resolve_leaves_staging_untouched_in_both_branches(tmp_path, store, plane):
    """12. test_resolve_leaves_staging_untouched_in_both_branches"""
    staging_repo = make_staging_repo(tmp_path)
    pinned = rev_parse(staging_repo, "refs/heads/staging")

    # -- rolled-back branch --
    _seed_gated_ticket(store, "t12-rb")
    weaver_rb = make_weaver(
        tmp_path,
        store,
        plane,
        staging_repo,
        run_checks_fn=FakeRunChecks([green_result()]),
        critic=make_critic(),
        journal_name="journal-rb.jsonl",
    )
    weaver_rb._journal_append(
        ticket="t12-rb", dispatch_id="d-rb", phase="begin", pinned_oid=pinned, candidate_oid=None
    )
    before = rev_parse(staging_repo, "refs/heads/staging")
    weaver_rb.resolve()
    after = rev_parse(staging_repo, "refs/heads/staging")
    assert before == after == pinned

    # -- landed branch --
    candidate_dir = tmp_path / "t12-candidate"
    run_git(None, ["clone", "--quiet", "--", str(staging_repo), str(candidate_dir)])
    run_git(candidate_dir, ["checkout", "--quiet", "-b", "weave-candidate", "origin/staging"])
    (candidate_dir / "x.txt").write_text("landed content\n")
    run_git(candidate_dir, ["add", "x.txt"])
    run_git(candidate_dir, ["commit", "--quiet", "-m", "land"])
    candidate_oid = rev_parse(candidate_dir, "weave-candidate")
    run_git(
        staging_repo,
        ["fetch", "--quiet", "--", str(candidate_dir), "weave-candidate:refs/weaver/incoming/t12"],
    )
    run_git(staging_repo, ["update-ref", "refs/heads/staging", candidate_oid, pinned])

    _seed_gated_ticket(store, "t12-landed")
    weaver_landed = make_weaver(
        tmp_path,
        store,
        plane,
        staging_repo,
        run_checks_fn=FakeRunChecks([green_result()]),
        critic=make_critic(),
        journal_name="journal-landed.jsonl",
    )
    weaver_landed._journal_append(
        ticket="t12-landed",
        dispatch_id="d-landed",
        phase="begin",
        pinned_oid=pinned,
        candidate_oid=None,
    )
    weaver_landed._journal_append(
        ticket="t12-landed",
        dispatch_id="d-landed",
        phase="applied",
        pinned_oid=pinned,
        candidate_oid=candidate_oid,
    )
    weaver_landed.resolve()
    assert rev_parse(staging_repo, "refs/heads/staging") == candidate_oid


def test_in_progress_false_after_clean_complete(tmp_path, store, plane):
    """13. test_in_progress_false_after_clean_complete"""
    staging_repo = make_staging_repo(tmp_path)
    bundle = make_bundle(tmp_path, staging_repo, name="t13", files={"feature.txt": "x\n"})
    add_parked_ticket(store, "t13", work_product=bundle)

    weaver = make_weaver(
        tmp_path,
        store,
        plane,
        staging_repo,
        run_checks_fn=FakeRunChecks([green_result()]),
        critic=make_critic(),
    )
    results = weaver.weave(now=1000.0)
    assert results[0].outcome == "landed"

    assert weaver.in_progress() is False
    tip_before = rev_parse(staging_repo, "refs/heads/staging")
    journal_lines_before = weaver._read_journal()
    outcome = weaver.resolve()  # safe no-op: journal already sealed complete
    tip_after = rev_parse(staging_repo, "refs/heads/staging")
    assert tip_before == tip_after
    # A repeat resolve() against an already-`complete` journal recomputes
    # the same, already-true fact (t13 did land) rather than mutating
    # anything -- consistent, not a fresh mutation.
    assert outcome == "landed"
    assert weaver._read_journal() == journal_lines_before  # no new journal line appended


def test_weave_idempotent_after_resolve(tmp_path, store, plane):
    """14. test_weave_idempotent_after_resolve"""
    staging_repo = make_staging_repo(tmp_path)
    pinned = rev_parse(staging_repo, "refs/heads/staging")

    _seed_gated_ticket(store, "t14")
    weaver = make_weaver(
        tmp_path,
        store,
        plane,
        staging_repo,
        run_checks_fn=FakeRunChecks([green_result()]),
        critic=make_critic(),
    )
    weaver._journal_append(
        ticket="t14", dispatch_id="d-t14", phase="begin", pinned_oid=pinned, candidate_oid=None
    )
    outcome = weaver.resolve()
    assert outcome == "rolled-back"
    assert store.get_ticket("t14")["state"] == PARKED

    events_after_resolve = plane.read_events()

    bundle = make_bundle(tmp_path, staging_repo, name="t14b", files={"feature.txt": "x\n"})
    add_parked_ticket(store, "t14b", work_product=bundle)
    results = weaver.weave(now=2000.0)

    # t14 (rolled back to PARKED, but with no real bundle) is picked up
    # again by this fresh weave and dies as a conflict (expected: rolling
    # back to PARKED means "eligible for a later weave", not "excluded
    # forever") -- the key assertion is that it is NOT double-landed and
    # its journal is not double-completed. t14b lands normally.
    by_ticket = {r.ticket: r for r in results}
    assert by_ticket["t14b"].outcome == "landed"
    assert by_ticket["t14"].outcome != "landed"

    # No double-land / double-journal for t14 itself: still exactly one
    # "complete" journal line for t14 (the resolve()'s seal), even though
    # this second weave() ran t14 again as a brand-new (separate) cycle.
    t14_complete_lines = [
        line
        for line in weaver._read_journal()
        if line.get("ticket") == "t14" and line.get("phase") == "complete"
    ]
    assert len(t14_complete_lines) == 2  # one from resolve(), one from this weave's own fresh cycle

    events_after_second_weave = plane.read_events()
    assert events_after_second_weave[: len(events_after_resolve)] == events_after_resolve


# ==========================================================================
# Protection asymmetry (SPEC §6)
# ==========================================================================


def test_bundle_touching_protected_path_flags_for_human(tmp_path, store, plane):
    """15. test_bundle_touching_protected_path_flags_for_human"""
    staging_repo = make_staging_repo(tmp_path)

    protected_bundle = make_bundle(
        tmp_path,
        staging_repo,
        name="t15-protected",
        files={"tests/test_acceptance.py": "def test_x(): assert True\n"},
    )
    add_parked_ticket(store, "t15-protected", work_product=protected_bundle)

    ordinary_bundle = make_bundle(
        tmp_path, staging_repo, name="t15-ordinary", files={"src/thing.py": "x = 1\n"}
    )
    add_parked_ticket(store, "t15-ordinary", work_product=ordinary_bundle)

    weaver = make_weaver(
        tmp_path,
        store,
        plane,
        staging_repo,
        run_checks_fn=FakeRunChecks([green_result()]),
        critic=make_critic(outcome="met"),
        protected_paths=["tests/test_acceptance.py", "Containerfile"],
    )
    results = weaver.weave(now=1000.0)

    protected_result = next(r for r in results if r.ticket == "t15-protected")
    ordinary_result = next(r for r in results if r.ticket == "t15-ordinary")

    assert protected_result.flagged_for_human is True
    assert ordinary_result.flagged_for_human is False

    protected_dispositions = disposition_events(plane, "t15-protected")
    assert any(
        d.get("reason") and "protected-path-touched" in d["reason"] for d in protected_dispositions
    )
    ordinary_dispositions = disposition_events(plane, "t15-ordinary")
    assert not any(
        d.get("reason") and "protected-path-touched" in d["reason"] for d in ordinary_dispositions
    )


# ==========================================================================
# Ordering / batch
# ==========================================================================


def test_parked_tickets_processed_in_deterministic_order(tmp_path, store, plane):
    """16. test_parked_tickets_processed_in_deterministic_order"""
    staging_repo = make_staging_repo(tmp_path)

    bundle1 = make_bundle(tmp_path, staging_repo, name="t16-first", files={"a.txt": "a\n"})
    add_parked_ticket(store, "t16-first", work_product=bundle1)
    bundle2 = make_bundle(tmp_path, staging_repo, name="t16-second", files={"b.txt": "b\n"})
    add_parked_ticket(store, "t16-second", work_product=bundle2)

    weaver = make_weaver(
        tmp_path,
        store,
        plane,
        staging_repo,
        run_checks_fn=FakeRunChecks([green_result()]),
        critic=make_critic(outcome="met"),
    )
    results = weaver.weave(now=1000.0)

    assert [r.ticket for r in results] == ["t16-first", "t16-second"]
    assert results[0].outcome == "landed"
    assert results[1].outcome == "landed"
    # The batch stacks: the second's pinned_oid (captured live, mid-batch)
    # equals the first's landed_oid.
    second_journal_begin = next(
        line
        for line in weaver._read_journal()
        if line.get("ticket") == "t16-second" and line.get("phase") == "begin"
    )
    assert second_journal_begin["pinned_oid"] == results[0].landed_oid

    tip = rev_parse(staging_repo, "refs/heads/staging")
    assert tip == results[1].landed_oid
    log = run_git(staging_repo, ["log", "refs/heads/staging", "--oneline"]).stdout
    assert results[0].landed_oid[:7] in log
    assert results[1].landed_oid[:7] in log


# ==========================================================================
# D14: a filings file COMMITTED into the worker bundle is stripped at weave
# and flagged (belt-and-suspenders — the normal path is host-side harvest of
# the UNCOMMITTED file; code01 tells the worker never to commit it).
# bead workspace-e2uh.38, AC14 case 6.
# ==========================================================================


def test_committed_filed_tickets_is_stripped_and_flagged(tmp_path, store, plane):
    staging_repo = make_staging_repo(tmp_path)
    pinned = rev_parse(staging_repo, "refs/heads/staging")
    # a REAL change plus the stray filings file the worker wrongly committed.
    bundle = make_bundle(
        tmp_path,
        staging_repo,
        name="t6",
        files={
            "feature.txt": "a real, legitimate change\n",
            ".stigmergy/filed-tickets.json": '[{"title": "x", "description": "y"}]\n',
        },
    )
    add_parked_ticket(store, "t6", work_product=bundle)

    weaver = make_weaver(
        tmp_path,
        store,
        plane,
        staging_repo,
        run_checks_fn=FakeRunChecks([green_result()]),
        critic=make_critic(outcome="met"),
    )
    results = weaver.weave(now=1000.0)

    assert len(results) == 1
    result = results[0]
    # the legitimate change still lands ...
    assert result.outcome == "landed"
    assert store.get_ticket("t6")["state"] == LANDED
    new_tip = rev_parse(staging_repo, "refs/heads/staging")
    assert new_tip != pinned

    # ... but the stray filings file NEVER reaches the landed commit ...
    tree = run_git(
        staging_repo, ["ls-tree", "-r", "--name-only", "refs/heads/staging"]
    ).stdout
    assert ".stigmergy/filed-tickets.json" not in tree
    assert "feature.txt" in tree

    # ... and the strip is surfaced for a human (flag + disposition marker).
    assert result.flagged_for_human is True
    dispositions = disposition_events(plane, "t6")
    assert len(dispositions) == 1
    assert dispositions[0]["disposition"] == "landed"
    assert "filed-tickets-stripped" in (dispositions[0].get("reason") or "")


# ==========================================================================
# D14 (bead .39): the STAGING CRITIC files out-of-rubric proposals.
# After the GATE event (verdict-first), the weaver files the critic's
# filed_tickets via file_proposals(origin_role="critic") — on BOTH the MET
# and UNMET paths (out-of-rubric findings survive a rejected candidate).
# A critic-infra failure files nothing. Filings are honest-zero cost.
# ==========================================================================

_CRITIC_FILINGS = [
    {"title": "Extract shared range-base helper", "description": "two call sites duplicate it"},
    {"title": "Add regression test for X", "description": "uncovered", "evidence": "y.py:10"},
]


def ticket_filed_events(plane: RecordPlane, ticket_id: str) -> list[dict]:
    return [
        e
        for e in plane.read_events()
        if e["event_type"] == "ticket-filed" and e["ticket"] == ticket_id
    ]


def test_met_verdict_files_critic_proposals_and_lands(tmp_path, store, plane):
    staging_repo = make_staging_repo(tmp_path)
    bundle = make_bundle(tmp_path, staging_repo, name="t1", files={"feature.txt": "x\n"})
    add_parked_ticket(store, "t1", work_product=bundle)

    weaver = make_weaver(
        tmp_path,
        store,
        plane,
        staging_repo,
        run_checks_fn=FakeRunChecks([green_result()]),
        critic=make_critic(outcome="met", filed_tickets=_CRITIC_FILINGS),
    )
    result = weaver.weave(now=1000.0)[0]

    assert result.outcome == "landed"  # filing is additive, never blocks landing
    assert store.get_ticket("t1")["state"] == LANDED

    filed = store.list_filed_tickets(triaged=False)
    assert len(filed) == 2
    assert {f["title"] for f in filed} == {t["title"] for t in _CRITIC_FILINGS}
    for row in filed:
        assert row["origin_role"] == "critic"  # unapproved, critic-origin
        assert row["triaged"] == 0
        assert row["discovered_from"] == "dispatch-t1@t1"  # real dispatch@ticket provenance

    filings = ticket_filed_events(plane, "t1")
    assert len(filings) == 2
    for ev in filings:
        assert ev["computed_usd"] == 0.0  # honest zero — the LLM cost is on the GATE event
        assert ev["origin"]["role"] == "critic"


def test_unmet_verdict_still_files_proposals(tmp_path, store, plane):
    staging_repo = make_staging_repo(tmp_path)
    pinned = rev_parse(staging_repo, "refs/heads/staging")
    bundle = make_bundle(tmp_path, staging_repo, name="t2", files={"feature.txt": "x\n"})
    add_parked_ticket(store, "t2", work_product=bundle)

    weaver = make_weaver(
        tmp_path,
        store,
        plane,
        staging_repo,
        run_checks_fn=FakeRunChecks([green_result()]),
        critic=make_critic(outcome="unmet", reason="missing tests", filed_tickets=_CRITIC_FILINGS),
    )
    result = weaver.weave(now=1000.0)[0]

    assert result.outcome == "rejected"  # in-rubric failure -> rejected
    assert rev_parse(staging_repo, "refs/heads/staging") == pinned
    # ...but out-of-rubric proposals were still captured ("only a filed ticket survives").
    filed = store.list_filed_tickets(triaged=False)
    assert len(filed) == 2
    assert all(f["origin_role"] == "critic" for f in filed)


def test_gate_event_precedes_ticket_filed_events(tmp_path, store, plane):
    staging_repo = make_staging_repo(tmp_path)
    bundle = make_bundle(tmp_path, staging_repo, name="t3", files={"feature.txt": "x\n"})
    add_parked_ticket(store, "t3", work_product=bundle)

    weaver = make_weaver(
        tmp_path,
        store,
        plane,
        staging_repo,
        run_checks_fn=FakeRunChecks([green_result()]),
        critic=make_critic(outcome="met", filed_tickets=_CRITIC_FILINGS),
    )
    weaver.weave(now=1000.0)

    kinds = [e["event_type"] for e in plane.read_events() if e["ticket"] == "t3"]
    # verdict-first: the GATE event is recorded before ANY ticket-filed event.
    assert "gate" in kinds and "ticket-filed" in kinds
    assert kinds.index("gate") < kinds.index("ticket-filed")


def test_critic_infra_files_nothing(tmp_path, store, plane):
    staging_repo = make_staging_repo(tmp_path)
    bundle = make_bundle(tmp_path, staging_repo, name="t4", files={"feature.txt": "x\n"})
    add_parked_ticket(store, "t4", work_product=bundle)

    weaver = make_weaver(
        tmp_path,
        store,
        plane,
        staging_repo,
        run_checks_fn=FakeRunChecks([green_result()]),
        critic=make_infra_critic(),
    )
    result = weaver.weave(now=1000.0)[0]

    assert result.outcome == "infra"
    assert store.list_filed_tickets() == []  # no verdict -> no filings
    assert ticket_filed_events(plane, "t4") == []


def test_no_filings_when_critic_returns_none(tmp_path, store, plane):
    staging_repo = make_staging_repo(tmp_path)
    bundle = make_bundle(tmp_path, staging_repo, name="t5", files={"feature.txt": "x\n"})
    add_parked_ticket(store, "t5", work_product=bundle)

    weaver = make_weaver(
        tmp_path,
        store,
        plane,
        staging_repo,
        run_checks_fn=FakeRunChecks([green_result()]),
        critic=make_critic(outcome="met"),  # no filed_tickets -> tolerant []
    )
    result = weaver.weave(now=1000.0)[0]

    assert result.outcome == "landed"
    assert store.list_filed_tickets() == []
    assert ticket_filed_events(plane, "t5") == []


def test_count_cap_rejects_whole_critic_filing(tmp_path, store, plane):
    staging_repo = make_staging_repo(tmp_path)
    bundle = make_bundle(tmp_path, staging_repo, name="t6", files={"feature.txt": "x\n"})
    add_parked_ticket(store, "t6", work_product=bundle)

    too_many = [{"title": f"t{i}", "description": "d"} for i in range(4)]
    weaver = make_weaver(
        tmp_path,
        store,
        plane,
        staging_repo,
        run_checks_fn=FakeRunChecks([green_result()]),
        critic=make_critic(outcome="met", filed_tickets=too_many),
        filing_max_filings=2,  # 4 > 2 -> whole filing rejected (file_proposals' rule)
    )
    result = weaver.weave(now=1000.0)[0]

    assert result.outcome == "landed"  # still lands; filing is additive
    assert store.list_filed_tickets() == []  # nothing accepted
    filings = ticket_filed_events(plane, "t6")
    assert len(filings) == 1
    assert filings[0]["outcome"] == "rejected"
    assert filings[0]["reason"] == "count-cap-exceeded"


def test_filing_exception_never_crashes_the_weaver(tmp_path, store, plane, monkeypatch):
    # The weaver is the single serialized writer. file_proposals is contracted
    # to never raise, but an unexpected filing-side exception (after the GATE
    # event, before disposition) must never strand the weave at GATED — it is
    # swallowed (logged) and the ticket lands normally. Filing is additive.
    import stigmergy.weaver as weaver_mod

    def boom(*a, **k):
        raise RuntimeError("filing blew up")

    monkeypatch.setattr(weaver_mod, "file_proposals", boom)

    staging_repo = make_staging_repo(tmp_path)
    bundle = make_bundle(tmp_path, staging_repo, name="t7", files={"feature.txt": "x\n"})
    add_parked_ticket(store, "t7", work_product=bundle)
    weaver = make_weaver(
        tmp_path,
        store,
        plane,
        staging_repo,
        run_checks_fn=FakeRunChecks([green_result()]),
        critic=make_critic(outcome="met", filed_tickets=_CRITIC_FILINGS),
    )
    result = weaver.weave(now=1000.0)[0]  # must NOT raise
    assert result.outcome == "landed"
    assert store.get_ticket("t7")["state"] == LANDED


# ==========================================================================
# bead .107: WeaveResult.reason — a structured discriminator distinguishing
# critic-infra (gate-infra) from CAS-abort (concurrent-staging-move), both
# of which otherwise share outcome="infra" + FailureClass.INFRA. See
# build-107-109-spec.md §2(d).
# ==========================================================================


def test_landed_result_has_reason_none(tmp_path, store, plane):
    staging_repo = make_staging_repo(tmp_path)
    bundle = make_bundle(tmp_path, staging_repo, name="t-reason-landed", files={"f.txt": "x\n"})
    add_parked_ticket(store, "t-reason-landed", work_product=bundle)

    weaver = make_weaver(
        tmp_path,
        store,
        plane,
        staging_repo,
        run_checks_fn=FakeRunChecks([green_result()]),
        critic=make_critic(outcome="met"),
    )
    result = weaver.weave(now=1000.0)[0]

    assert result.outcome == "landed"
    assert result.reason is None


def test_critic_infra_result_has_reason_critic_infra(tmp_path, store, plane):
    staging_repo = make_staging_repo(tmp_path)
    bundle = make_bundle(tmp_path, staging_repo, name="t-reason-infra", files={"f.txt": "x\n"})
    add_parked_ticket(store, "t-reason-infra", work_product=bundle)

    weaver = make_weaver(
        tmp_path,
        store,
        plane,
        staging_repo,
        run_checks_fn=FakeRunChecks([green_result()]),
        critic=make_infra_critic(),
    )
    result = weaver.weave(now=1000.0)[0]

    assert result.outcome == "infra"
    assert result.reason == "critic-infra"


def test_cas_abort_result_has_reason_concurrent_staging_move(tmp_path, store, plane):
    """CAS-abort (concurrent staging move) is ALSO outcome="infra" but must
    carry a DIFFERENT `reason` than a genuine critic-infra failure — this is
    the whole point of the new field (regression guard for daemon.py's new
    critic-infra-only branch, which must NOT fire on this path)."""
    staging_repo = make_staging_repo(tmp_path)
    bundle = make_bundle(
        tmp_path, staging_repo, name="t-reason-cas", files={"feature.txt": "x\n"}
    )
    add_parked_ticket(store, "t-reason-cas", work_product=bundle)

    class RacingWeaver(Weaver):
        def _fetch_candidate_into_staging(self, candidate_dir, ticket_id):
            concurrent_dir = tmp_path / "concurrent-writer-reason"
            run_git(None, ["clone", "--quiet", "--", str(staging_repo), str(concurrent_dir)])
            run_git(concurrent_dir, ["checkout", "--quiet", "-b", "staging", "origin/staging"])
            (concurrent_dir / "concurrent.txt").write_text("a concurrent, out-of-band commit\n")
            run_git(concurrent_dir, ["add", "concurrent.txt"])
            run_git(concurrent_dir, ["commit", "--quiet", "-m", "concurrent move"])
            run_git(concurrent_dir, ["push", "--quiet", "origin", "staging"])
            super()._fetch_candidate_into_staging(candidate_dir, ticket_id)

    weaver = RacingWeaver(
        store=store,
        record_plane=plane,
        staging_repo=staging_repo,
        run_checks_fn=FakeRunChecks([green_result()]),
        critic=make_critic(outcome="met"),
        checker_image="unused",
        flake_reruns=0,
        protected_paths=[],
        journal_path=tmp_path / "journal-reason-cas.jsonl",
        ctx_of=stub_ctx_of,
        filing_max_filings=5,
        filing_max_bytes=16384,
    )
    result = weaver.weave(now=1000.0)[0]

    assert result.outcome == "infra"
    assert result.reason == "concurrent-staging-move"


def test_rejected_result_has_reason_gate_unmet(tmp_path, store, plane):
    staging_repo = make_staging_repo(tmp_path)
    bundle = make_bundle(tmp_path, staging_repo, name="t-reason-rej", files={"f.txt": "x\n"})
    add_parked_ticket(store, "t-reason-rej", work_product=bundle)

    weaver = make_weaver(
        tmp_path,
        store,
        plane,
        staging_repo,
        run_checks_fn=FakeRunChecks([green_result()]),
        critic=make_critic(outcome="unmet", reason="missing tests"),
    )
    result = weaver.weave(now=1000.0)[0]

    assert result.outcome == "rejected"
    assert result.reason == "gate-unmet"


# ==========================================================================
# bead .109: persist the CriticInfraError cause on the gate-infra
# INTEGRATION event, scoped to gate-infra only. See build-107-109-spec.md
# §3(a).
# ==========================================================================


def test_gate_infra_integration_event_carries_error_message(tmp_path, store, plane):
    """.109 #1: when `critic.judge` raises `CriticInfraError(...)`, the
    emitted gate-infra INTEGRATION event carries an `error` field equal to
    that exact message."""
    staging_repo = make_staging_repo(tmp_path)
    bundle = make_bundle(
        tmp_path, staging_repo, name="t-109-error", files={"f.txt": "x\n"}
    )
    add_parked_ticket(store, "t-109-error", work_product=bundle)

    weaver = make_weaver(
        tmp_path,
        store,
        plane,
        staging_repo,
        run_checks_fn=FakeRunChecks([green_result()]),
        critic=make_infra_critic(),
    )
    result = weaver.weave(now=1000.0)[0]

    assert result.outcome == "infra"
    assert result.reason == "critic-infra"

    events = integration_events(plane, "t-109-error", phase="gate-infra")
    assert len(events) == 1
    assert events[0]["error"] == "critic client call failed: RuntimeError: provider 503"


def test_non_gate_infra_integration_event_has_no_error_field(tmp_path, store, plane):
    """.109 #2: a NON-gate-infra integration event (e.g. the "apply"/"land"
    phase on a landed ticket) carries no `error` field (or `error is
    None`) — the field is scoped to gate-infra only."""
    staging_repo = make_staging_repo(tmp_path)
    bundle = make_bundle(
        tmp_path, staging_repo, name="t-109-noerror", files={"f.txt": "x\n"}
    )
    add_parked_ticket(store, "t-109-noerror", work_product=bundle)

    weaver = make_weaver(
        tmp_path,
        store,
        plane,
        staging_repo,
        run_checks_fn=FakeRunChecks([green_result()]),
        critic=make_critic(outcome="met"),
    )
    result = weaver.weave(now=1000.0)[0]
    assert result.outcome == "landed"

    events = integration_events(plane, "t-109-noerror")
    assert len(events) >= 1
    for event in events:
        assert event.get("error") is None


# --------------------------------------------------------------------------
# Trusted check evidence passed to critic
# --------------------------------------------------------------------------


def test_weaver_passes_check_results_as_evidence_to_critic(tmp_path, store, plane):
    """The weaver passes its computed check_results to the critic as trusted
    evidence, allowing the critic to rely on mechanical test results without
    requiring the artifact to re-prove them."""
    staging_repo = make_staging_repo(tmp_path)
    bundle = make_bundle(tmp_path, staging_repo, name="t-ev", files={"feature.txt": "x\n"})
    add_parked_ticket(store, "t-ev", work_product=bundle)

    # Create a spy critic that records what evidence it receives.
    class EvidenceSpy:
        def __init__(self, inner):
            self._inner = inner
            self.received_evidence = None

        def judge(self, artifact, rubric_items, *, check_evidence=None):
            self.received_evidence = check_evidence
            return self._inner.judge(artifact, rubric_items, check_evidence=check_evidence)

    inner_critic = make_critic(outcome="met")
    spy = EvidenceSpy(inner_critic)

    run_checks = FakeRunChecks([green_result()])
    weaver = make_weaver(
        tmp_path, store, plane, staging_repo, run_checks_fn=run_checks, critic=spy
    )
    result = weaver.weave(now=1000.0)[0]
    assert result.outcome == "landed"

    # Verify that check evidence was passed to the critic.
    assert spy.received_evidence is not None
    # The evidence should contain information about the check results.
    assert "Tier-1 Check Results" in spy.received_evidence
    assert "PASS" in spy.received_evidence


def test_weave_result_carries_real_gate_tokens_and_duration(tmp_path, store, plane):
    """Verify that WeaveResult carries real token usage, model, and wall-time
    duration extracted from the critic client response."""
    staging_repo = make_staging_repo(tmp_path)
    bundle = make_bundle(tmp_path, staging_repo, name="t-tokens", files={"f.txt": "x\n"})
    add_parked_ticket(store, "t-tokens", work_product=bundle)

    # Create a critic that returns usage tokens in the response
    usage_tokens = {"in": 150, "out": 75, "cached": 10, "reasoning": 0}
    response = {
        "outcome": "met",
        "tier": 1,
        "reason": "looks good",
        "severity": "none",
        "usage": usage_tokens,
    }
    from stigmergy.critic import Critic
    client = StubCriticClient(response=response)
    critic = Critic(
        client=client,
        model="test-model-v1",
        decoding_params={"temperature": 0.0},
        template="Judge the artifact.",
    )

    weaver = make_weaver(
        tmp_path,
        store,
        plane,
        staging_repo,
        run_checks_fn=FakeRunChecks([green_result()]),
        critic=critic,
    )
    result = weaver.weave(now=1000.0)[0]

    # Verify the WeaveResult carries the real gate tokens, model, and duration
    assert result.gate_model == "test-model-v1"
    assert result.gate_tokens == usage_tokens
    assert result.gate_duration is not None
    assert result.gate_duration > 0

    # Verify the GATE event carries the real tokens and wall_time
    gate_events_list = gate_events(plane, "t-tokens")
    assert len(gate_events_list) == 1
    gate_event = gate_events_list[0]
    assert gate_event["tokens"] == usage_tokens
    assert gate_event["wall_time_seconds"] > 0


def test_weave_result_carries_gate_data_on_rejected_verdict(tmp_path, store, plane):
    """Verify that WeaveResult carries gate_model/tokens/duration even for
    rejected verdicts (UNMET path), not just for landed verdicts."""
    staging_repo = make_staging_repo(tmp_path)
    bundle = make_bundle(tmp_path, staging_repo, name="t-rejected-tokens", files={"f.txt": "x\n"})
    add_parked_ticket(store, "t-rejected-tokens", work_product=bundle)

    # Create a critic that returns UNMET verdict with usage tokens
    usage_tokens = {"in": 100, "out": 50, "cached": 0, "reasoning": 0}
    response = {
        "outcome": "unmet",
        "tier": 2,
        "reason": "missing tests",
        "severity": "high",
        "usage": usage_tokens,
    }
    from stigmergy.critic import Critic
    client = StubCriticClient(response=response)
    critic = Critic(
        client=client,
        model="test-unmet-model",
        decoding_params={"temperature": 0.0},
        template="Judge the artifact.",
    )

    weaver = make_weaver(
        tmp_path,
        store,
        plane,
        staging_repo,
        run_checks_fn=FakeRunChecks([green_result()]),
        critic=critic,
    )
    result = weaver.weave(now=1000.0)[0]

    # Verify the rejected WeaveResult still carries the real gate data
    assert result.outcome == "rejected"
    assert result.gate_model == "test-unmet-model"
    assert result.gate_tokens == usage_tokens
    assert result.gate_duration is not None

    # Verify the GATE event was written and carries the tokens
    gate_events_list = gate_events(plane, "t-rejected-tokens")
    assert len(gate_events_list) == 1
    assert gate_events_list[0]["tokens"] == usage_tokens


def test_gate_event_carries_repair_attempts_and_hash(tmp_path, store, plane):
    """Verify that _append_gate_event passes through repair_attempts and
    repair_instruction_hash from gate_fields to the GATE event payload."""
    staging_repo = make_staging_repo(tmp_path)
    bundle = make_bundle(
        tmp_path, staging_repo, name="t-repair-fields", files={"f.txt": "x\n"}
    )
    add_parked_ticket(store, "t-repair-fields", work_product=bundle)

    # Create a critic with a clean response (no repair needed)
    response = {
        "outcome": "met",
        "tier": 1,
        "reason": "looks good",
        "severity": "none",
    }
    from stigmergy.critic import Critic

    client = StubCriticClient(response=response)
    critic = Critic(
        client=client,
        model="test-model",
        decoding_params={"temperature": 0.0},
        template="Judge the artifact.",
    )

    weaver = make_weaver(
        tmp_path,
        store,
        plane,
        staging_repo,
        run_checks_fn=FakeRunChecks([green_result()]),
        critic=critic,
    )
    weaver.weave(now=1000.0)

    # Verify the GATE event was written and carries repair fields
    gate_events_list = gate_events(plane, "t-repair-fields")
    assert len(gate_events_list) == 1
    gate_event = gate_events_list[0]

    # repair_attempts should be 0 (clean first pass)
    assert gate_event["repair_attempts"] == 0
    # repair_instruction_hash should be present and be a 64-char hex string (sha256)
    assert "repair_instruction_hash" in gate_event
    assert isinstance(gate_event["repair_instruction_hash"], str)
    assert len(gate_event["repair_instruction_hash"]) == 64


# ==========================================================================
# Scope-breach audit (bead .102a)
# ==========================================================================


def test_scope_breach_all_in_scope(tmp_path, store, plane):
    """Scope-breach audit: bundle whose touched files are ALL inside
    target_scope -> the apply INTEGRATION event carries scope_breach with
    empty out_of_scope_paths list, base_oid = pinned OID, and correct stamps."""
    staging_repo = make_staging_repo(tmp_path)
    pinned = rev_parse(staging_repo, "refs/heads/staging")

    bundle = make_bundle(
        tmp_path,
        staging_repo,
        name="all-in-scope",
        files={
            "src/feature.py": "def feature(): pass\n",
            "src/sub/nested.py": "x = 1\n",
        },
    )
    add_parked_ticket(
        store,
        "all-in-scope",
        work_product=bundle,
        target_scope=["src", "tests"],
    )

    weaver = make_weaver(
        tmp_path,
        store,
        plane,
        staging_repo,
        run_checks_fn=FakeRunChecks([green_result()]),
        critic=make_critic(outcome="met"),
    )
    weaver.weave(now=1000.0)

    apply_events = integration_events(plane, "all-in-scope", phase="apply")
    assert len(apply_events) == 1
    event = apply_events[0]

    assert "scope_breach" in event
    scope_breach = event["scope_breach"]
    assert scope_breach["out_of_scope_paths"] == []
    assert scope_breach["base_oid"] == pinned
    assert scope_breach["approval_hash"] == "approvalhash456"  # from stub_ctx_of
    assert scope_breach["algo_version"] == "102a-v1"


def test_scope_breach_mixed_in_and_out(tmp_path, store, plane):
    """Scope-breach audit: bundle that touches a mix of in-scope and out-of-
    scope files -> out_of_scope_paths lists exactly the out-of-scope ones,
    sorted, and the in-scope ones are excluded."""
    staging_repo = make_staging_repo(tmp_path)

    bundle = make_bundle(
        tmp_path,
        staging_repo,
        name="mixed-scope",
        files={
            "src/feature.py": "new feature\n",
            "docs/README.md": "docs\n",
            "src/sub/nested.py": "nested\n",
            "tools/script.sh": "#!/bin/bash\n",
        },
    )
    add_parked_ticket(
        store,
        "mixed-scope",
        work_product=bundle,
        target_scope=["src"],
    )

    weaver = make_weaver(
        tmp_path,
        store,
        plane,
        staging_repo,
        run_checks_fn=FakeRunChecks([green_result()]),
        critic=make_critic(outcome="met"),
    )
    weaver.weave(now=1000.0)

    apply_events = integration_events(plane, "mixed-scope", phase="apply")
    assert len(apply_events) == 1
    scope_breach = apply_events[0]["scope_breach"]

    # Only docs and tools should be out of scope
    assert scope_breach["out_of_scope_paths"] == ["docs/README.md", "tools/script.sh"]
    # src/feature.py and src/sub/nested.py are in scope, excluded from result


def test_scope_breach_directory_prefix_nesting(tmp_path, store, plane):
    """Scope-breach audit: directory-prefix nesting rule — a scope entry
    `src/foo` marks `src/foo/a.py` and `src/foo/sub/b.py` in scope, but
    NOT `src/foobar.py` (the `+ "/"` boundary)."""
    staging_repo = make_staging_repo(tmp_path)

    bundle = make_bundle(
        tmp_path,
        staging_repo,
        name="nesting-rule",
        files={
            "src/foo/a.py": "a\n",
            "src/foo/sub/b.py": "b\n",
            "src/foobar.py": "foobar\n",
            "src/other.py": "other\n",
        },
    )
    add_parked_ticket(
        store,
        "nesting-rule",
        work_product=bundle,
        target_scope=["src/foo"],
    )

    weaver = make_weaver(
        tmp_path,
        store,
        plane,
        staging_repo,
        run_checks_fn=FakeRunChecks([green_result()]),
        critic=make_critic(outcome="met"),
    )
    weaver.weave(now=1000.0)

    apply_events = integration_events(plane, "nesting-rule", phase="apply")
    scope_breach = apply_events[0]["scope_breach"]

    # src/foo/a.py and src/foo/sub/b.py are in scope
    # src/foobar.py and src/other.py are out of scope
    assert scope_breach["out_of_scope_paths"] == ["src/foobar.py", "src/other.py"]


def test_scope_breach_empty_target_scope(tmp_path, store, plane):
    """Scope-breach audit: target_scope is empty/None -> every touched path
    is recorded as out of scope (honest reading, never silence)."""
    staging_repo = make_staging_repo(tmp_path)

    bundle = make_bundle(
        tmp_path,
        staging_repo,
        name="empty-scope",
        files={
            "src/feature.py": "feature\n",
            "tests/test_feature.py": "test\n",
        },
    )
    add_parked_ticket(
        store,
        "empty-scope",
        work_product=bundle,
        target_scope=[],
    )

    weaver = make_weaver(
        tmp_path,
        store,
        plane,
        staging_repo,
        run_checks_fn=FakeRunChecks([green_result()]),
        critic=make_critic(outcome="met"),
    )
    weaver.weave(now=1000.0)

    apply_events = integration_events(plane, "empty-scope", phase="apply")
    scope_breach = apply_events[0]["scope_breach"]

    # All touched paths are out of scope
    assert scope_breach["out_of_scope_paths"] == [
        "src/feature.py",
        "tests/test_feature.py",
    ]


def test_scope_breach_none_target_scope(tmp_path, store, plane):
    """Scope-breach audit: target_scope is None (not in ticket) -> every
    touched path is recorded as out of scope, same as empty list."""
    staging_repo = make_staging_repo(tmp_path)

    bundle = make_bundle(
        tmp_path,
        staging_repo,
        name="none-scope",
        files={"src/feature.py": "feature\n"},
    )
    # Add ticket WITHOUT target_scope (None by default)
    add_parked_ticket(
        store,
        "none-scope",
        work_product=bundle,
        target_scope=None,
    )

    weaver = make_weaver(
        tmp_path,
        store,
        plane,
        staging_repo,
        run_checks_fn=FakeRunChecks([green_result()]),
        critic=make_critic(outcome="met"),
    )
    weaver.weave(now=1000.0)

    apply_events = integration_events(plane, "none-scope", phase="apply")
    scope_breach = apply_events[0]["scope_breach"]

    # All touched paths are out of scope
    assert scope_breach["out_of_scope_paths"] == ["src/feature.py"]


def test_scope_breach_independent_of_rejection_outcome(tmp_path, store, plane):
    """Scope-breach audit: a bundle that would be REJECTED but reaches outside
    scope -> the scope_breach field IS STILL on the apply event AND the
    disposition is still 'rejected' for the SAME reason (proving the breach
    is recording-only and independent of the gate outcome)."""
    staging_repo = make_staging_repo(tmp_path)

    bundle = make_bundle(
        tmp_path,
        staging_repo,
        name="out-of-scope-rejected",
        files={
            "src/feature.py": "feature\n",
            "external/bad.py": "bad\n",
        },
    )
    add_parked_ticket(
        store,
        "out-of-scope-rejected",
        work_product=bundle,
        target_scope=["src"],
    )

    # Critic will reject the candidate
    weaver = make_weaver(
        tmp_path,
        store,
        plane,
        staging_repo,
        run_checks_fn=FakeRunChecks([green_result()]),
        critic=make_critic(outcome="unmet", reason="fails rubric"),
    )
    results = weaver.weave(now=1000.0)

    # Verify rejected outcome
    result = results[0]
    assert result.outcome == "rejected"
    assert result.failure_class.value == "rejected"

    # Verify apply event still carries scope_breach
    apply_events = integration_events(plane, "out-of-scope-rejected", phase="apply")
    assert len(apply_events) == 1
    scope_breach = apply_events[0]["scope_breach"]
    assert scope_breach["out_of_scope_paths"] == ["external/bad.py"]

    # Verify disposition reason hasn't changed (still 'gate-unmet', not mentioning scope)
    dispositions = disposition_events(plane, "out-of-scope-rejected")
    assert len(dispositions) == 1
    assert dispositions[0]["disposition"] == "rejected"
    # The reason should not include any scope-breach mention in the string
    # (scope_breach is a separate structured field, not folded into reason)


def test_scope_breach_conflict_has_no_apply_event(tmp_path, store, plane):
    """Scope-breach audit: a bundle that fails to apply (merge conflict) ->
    there is NO apply INTEGRATION event at all (nothing to audit), and no
    scope_breach field — this is correct, not a gap."""
    staging_repo = make_staging_repo(tmp_path)

    # Make a conflicting bundle
    bundle = make_conflicting_bundle(tmp_path, staging_repo, name="conflict")
    add_parked_ticket(
        store,
        "conflict",
        work_product=bundle,
        target_scope=["src"],
    )

    weaver = make_weaver(
        tmp_path,
        store,
        plane,
        staging_repo,
        run_checks_fn=FakeRunChecks([green_result()]),
        critic=make_critic(outcome="met"),
    )
    results = weaver.weave(now=1000.0)

    result = results[0]
    assert result.outcome == "integration-conflict"

    # Verify NO apply event for this ticket
    apply_events = integration_events(plane, "conflict", phase="apply")
    assert len(apply_events) == 0


def test_scope_breach_separate_from_protected_paths(tmp_path, store, plane):
    """Scope-breach audit: a candidate with out-of-scope touched files but NO
    protected-path touched files still has flagged_for_human is False and a
    reason with no 'protected-path-touched:' marker (proving scope_breach is a
    separate, non-decision-changing channel)."""
    staging_repo = make_staging_repo(tmp_path)

    bundle = make_bundle(
        tmp_path,
        staging_repo,
        name="scope-but-no-protected",
        files={
            "src/feature.py": "feature\n",
            "docs/README.md": "docs\n",  # out of scope
        },
    )
    add_parked_ticket(
        store,
        "scope-but-no-protected",
        work_product=bundle,
        target_scope=["src"],
    )

    weaver = make_weaver(
        tmp_path,
        store,
        plane,
        staging_repo,
        run_checks_fn=FakeRunChecks([green_result()]),
        critic=make_critic(outcome="met"),
        protected_paths=["tests/test_acceptance.py"],  # not touched
    )
    results = weaver.weave(now=1000.0)

    result = results[0]
    # flagged_for_human should still be False (no protected paths touched)
    assert result.flagged_for_human is False

    # The reason string should NOT contain 'protected-path-touched'
    dispositions = disposition_events(plane, "scope-but-no-protected")
    assert len(dispositions) == 1
    reason = dispositions[0].get("reason")
    assert "protected-path-touched" not in (reason or "")

    # But scope_breach should still be on the apply event
    apply_events = integration_events(plane, "scope-but-no-protected", phase="apply")
    assert len(apply_events) == 1
    assert "scope_breach" in apply_events[0]
    assert apply_events[0]["scope_breach"]["out_of_scope_paths"] == ["docs/README.md"]
