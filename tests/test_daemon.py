"""Tests for stigmergy.daemon (SPEC.md §2 System Overview, §3 Stations, §9
Failure Handling & Loop Mechanics, §10 AC7/AC9; bead .22 build spec §6 —
frozen case list, cases 1-20).

Real, tiny on-disk rig (real `RigStore`, real `RecordPlane`, real small git
repos for `rig_repo`/`work_clone`/`staging_repo` — mirrors
`test_dispatch.py`'s and `test_weaver.py`'s real-git-repo fixture style). No
real podman/claude-code/network anywhere: `spawn_fn` and `run_checks_fn` are
always scripted/stubbed callables (mirrors `test_driver.py`'s
`CapturingRunOne`/`RaisingRunOne` pattern); where the real `Weaver` is
exercised (cases 11-13, 20), it is wired with a stubbed `run_checks_fn` and
a stubbed critic client exactly as `test_weaver.py` does.

Case numbering below matches the bead .22 build spec's frozen case list
(build spec §6, cases 1-20) verbatim in each test's docstring.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

import pytest

from stigmergy.approval import approve
from stigmergy.charter import load_charter
from stigmergy.checks import CheckOutcome, CheckResult
from stigmergy.daemon import (
    _BACKOFF_CAP_SECONDS,
    _CIRCUIT_BREAKER_THRESHOLD,
    Daemon,
    DaemonError,
)
from stigmergy.drivers.claude_code import DispatchResult, DispatchStatus
from stigmergy.notify import NotificationStore, NtfyNotifier
from stigmergy.records import RecordPlane
from stigmergy.recover import RecoveryError
from stigmergy.registry import load_registry
from stigmergy.relay import Capability, CapabilityStore
from stigmergy.rig import RigStore
from stigmergy.spend import Budgets, SpendLeash
from stigmergy.statemachine import ESCALATED, LANDED, PARKED, POOL
from stigmergy.weaver import Weaver

FIXTURES = Path(__file__).parent / "fixtures"
VALID_CHARTER_PATH = FIXTURES / "charter_valid.toml"
MODELS_REGISTRY_PATH = FIXTURES / "models.toml"
BASE_CHARTER_TOML = VALID_CHARTER_PATH.read_text()

PINNED_IMAGE = "localhost/stigmergy-worker@sha256:" + "a" * 64

PROMPT_TEMPLATE = (
    "Goal:\n$goal\n\nAcceptance criteria:\n$acceptance_criteria\n"
    "Tier 1 checks:\n$tier1_checks\n"
)

GIT_ENV_CFG = [
    "-c",
    "user.email=fixture@example.com",
    "-c",
    "user.name=Fixture User",
]


# ==========================================================================
# git fixture helpers (REAL git, host-safe: everything lives under tmp_path)
# ==========================================================================


def run_git(repo: Path | None, args: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    argv = ["git"]
    if repo is not None:
        argv += ["-C", str(repo)]
    argv += GIT_ENV_CFG + args
    result = subprocess.run(argv, capture_output=True, text=True, check=False, **kwargs)  # noqa: S603
    assert result.returncode == 0, f"git {args} failed: {result.stderr}"
    return result


def rev_parse(repo: Path, ref: str) -> str:
    return run_git(repo, ["rev-parse", ref]).stdout.strip()


def make_rig_repo(tmp_path: Path, name: str = "rig_repo") -> Path:
    """A real, small local git repo with one commit on `staging` (mirrors
    `test_dispatch.py`'s `make_repo` — this is what `prepare_dispatch`
    clones per dispatch)."""
    repo = tmp_path / name
    repo.mkdir(parents=True, exist_ok=True)
    run_git(repo, ["init", "--quiet", "-b", "staging"])
    (repo / "README.md").write_text("hello from the fixture repo\n")
    run_git(repo, ["add", "README.md"])
    run_git(repo, ["commit", "--quiet", "-m", "initial commit"])
    return repo


def make_staging_repo(tmp_path: Path, name: str = "staging_repo") -> Path:
    """A BARE loop-owned integration repo (mirrors `test_weaver.py`'s
    `make_staging_repo` exactly) — the weave station's landing target."""
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
    tmp_path: Path, staging_repo: Path, *, name: str, files: dict[str, str]
) -> Path:
    """A real `git bundle` carrying `refs/heads/work` (mirrors
    `test_weaver.py`'s `make_bundle`)."""
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


# ==========================================================================
# fakes / stubs
# ==========================================================================


class FakeClock:
    """A deterministic, strictly-increasing fake clock (never real time)."""

    def __init__(self, start: float = 1_000_000.0, step: float = 1.0):
        self.current = start
        self.step = step

    def __call__(self) -> float:
        self.current += self.step
        return self.current


class FakeWeaver:
    """A fake satisfying `Weaver`'s public shape (bead .22 build spec §6):
    `weave()` always returns an empty list and never touches the store —
    the default double for every test that does not itself exercise real
    weave machinery. `in_progress()`/`resolve()` satisfy the
    `WeaveJournalResolver` protocol `recover_on_start` needs."""

    def __init__(self) -> None:
        self.weave_calls = 0

    def in_progress(self) -> bool:
        return False

    def resolve(self) -> str:
        return "rolled-back"

    def weave(self, *, now: float) -> list:
        self.weave_calls += 1
        return []


class FakeReaper:
    """A fake `ContainerReaper`: nothing ever running, `reap()` a no-op."""

    def list_running(self) -> list[str]:
        return []

    def reap(self, dispatch_id: str) -> None:
        pass


class RecordingSender:
    """A fake ntfy `Sender` that always "succeeds" and records every call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, topic: str, title: str, message: str) -> None:
        self.calls.append((topic, title, message))


class SpyCapabilityStore(CapabilityStore):
    """Records every minted `Capability` (case 16/17 need the actual
    token) without changing behavior."""

    def __init__(self) -> None:
        super().__init__()
        self.minted: list[Capability] = []

    def mint(self, *args: Any, **kwargs: Any) -> Capability:  # type: ignore[override]
        cap = super().mint(*args, **kwargs)
        self.minted.append(cap)
        return cap


class SpySpendLeash(SpendLeash):
    """Counts `final_weave_allowed()` calls (case 20's one-shot-consumption
    assertion) without changing behavior."""

    def __init__(self, budgets: Budgets, registry: Any) -> None:
        super().__init__(budgets, registry)
        self.final_weave_allowed_calls = 0

    def final_weave_allowed(self) -> bool:
        self.final_weave_allowed_calls += 1
        return super().final_weave_allowed()


class ScriptedSpawn:
    """Deterministic injected `spawn_fn`: pops one canned `DispatchResult`
    per call, in order. Raises `AssertionError` (not silently reusing the
    last result) if called more times than scripted — a test-plan gap,
    never a silent pass."""

    def __init__(self, results: list[DispatchResult]):
        self._results = list(results)
        self.calls: list[tuple] = []

    def __call__(self, task_pack, work_clone, model_cfg, capability, budgets):
        self.calls.append((task_pack, work_clone, model_cfg, capability, budgets))
        if not self._results:
            raise AssertionError("ScriptedSpawn called more times than scripted")
        return self._results.pop(0)


class AlwaysInfraSpawn:
    """A `spawn_fn` that always returns `DispatchStatus.INFRA` (simulates
    repeated provider 429s, cases 2/3)."""

    def __init__(self):
        self.calls = 0

    def __call__(self, task_pack, work_clone, model_cfg, capability, budgets):
        self.calls += 1
        return make_result(DispatchStatus.INFRA, usage=zero_tokens(), detail="simulated 429")


class TokenEmbeddingSpawn:
    """A `spawn_fn` whose transcript embeds the capability token it was
    handed (case 17 — proves seal_transcript redacts it)."""

    def __call__(self, task_pack, work_clone, model_cfg, capability, budgets):
        transcript = f"worker transcript... secret={capability.token} ...end of transcript"
        return make_result(DispatchStatus.DONE, transcript=transcript)


class RaisingSpawn:
    """A `spawn_fn` that raises a bare `Exception` (case 16 — simulates a
    genuine bug, not a modeled `DispatchStatus`)."""

    def __call__(self, task_pack, work_clone, model_cfg, capability, budgets):
        raise RuntimeError("unexpected driver bug")


class ScriptedChecks:
    """Deterministic injected `run_checks_fn`: returns the same canned
    `CheckResult` list regardless of input (mirrors `test_weaver.py`'s
    `FakeRunChecks`)."""

    def __init__(self, results: list[CheckResult]):
        self._results = results
        self.calls: list[Any] = []

    def __call__(self, checks_dict, work_tree, *, image, flake_reruns):
        self.calls.append((dict(checks_dict), work_tree))
        return list(self._results)


def zero_tokens(**overrides: int) -> dict[str, int]:
    base = {"in": 0, "cached": 0, "out": 0, "reasoning": 0}
    base.update(overrides)
    return base


def make_result(
    status: DispatchStatus,
    *,
    usage: dict[str, int] | None = None,
    ceiling_trip: str | None = None,
    bundle_ref: str | None = None,
    transcript: str = "ok transcript, nothing secret here",
    cost: float | None = 0.001,
    detail: str = "scripted",
) -> DispatchResult:
    return DispatchResult(
        status=status,
        transcript=transcript,
        usage=usage if usage is not None else {"in": 10, "cached": 0, "out": 20, "reasoning": 0},
        reported_cost_usd=cost,
        bundle_ref=bundle_ref,
        ceiling_trip=ceiling_trip,
        detail=detail,
    )


def pass_result(name: str = "check") -> CheckResult:
    return CheckResult(
        name=name, outcome=CheckOutcome.PASS, runs=(0,), output="ok", wall_time_seconds=0.01
    )


def fail_result(name: str = "check") -> CheckResult:
    return CheckResult(
        name=name, outcome=CheckOutcome.FAIL, runs=(1,), output="boom", wall_time_seconds=0.01
    )


def flaky_result(name: str = "check") -> CheckResult:
    return CheckResult(
        name=name, outcome=CheckOutcome.FLAKY, runs=(1, 0), output="flaky", wall_time_seconds=0.01
    )


def base_steering(**overrides: Any) -> dict[str, Any]:
    steering: dict[str, Any] = {
        "ticket_text": "Implement foo() to spec.",
        "checks": {"named": ["pytest", "lint"], "paths": ["tests/test_foo.py"]},
        "rubric": ["foo() returns 42"],
        "lane": "default",
        "prompt_bytes": "code01-prompt-v1",
        "context_set": [],
    }
    steering.update(overrides)
    return steering


# ==========================================================================
# environment / daemon-construction helpers
# ==========================================================================


class Env:
    def __init__(self, tmp_path: Path):
        charter_dir = tmp_path / "charter_src"
        charter_dir.mkdir()
        charter_path = charter_dir / "charter.toml"
        charter_path.write_text(BASE_CHARTER_TOML)
        import shutil

        shutil.copy(MODELS_REGISTRY_PATH, charter_dir / "models.toml")
        self.charter = load_charter(charter_path, env={})
        self.charter.raw["rig"]["image"] = PINNED_IMAGE

        self.registry = load_registry(charter_dir / "models.toml")

        self.rig_repo = make_rig_repo(tmp_path)
        self.staging_repo = make_staging_repo(tmp_path)

        self.store = RigStore.create(tmp_path / "tickets.db")
        self.record_plane = RecordPlane(tmp_path / "records")
        self.capability_store = SpyCapabilityStore()
        self.notification_store = NotificationStore(tmp_path / "records" / "notifications.jsonl")
        self.sender = RecordingSender()
        self.notifier = NtfyNotifier("test-topic", sender=self.sender)
        self.reaper = FakeReaper()

        self.context_root = tmp_path / "context"
        self.context_root.mkdir()
        self.clones_root = tmp_path / "clones"
        self.prompts_dir = tmp_path / "prompts_dir"
        self.prompts_dir.mkdir()
        (self.prompts_dir / "code01").write_text(PROMPT_TEMPLATE)

        self.rig_paths = {
            "context_root": self.context_root,
            "repo_root": self.rig_repo,
            "clones_root": self.clones_root,
            "prompts_dir": self.prompts_dir,
            "records_dir": tmp_path / "records",
        }

        self.tmp_path = tmp_path
        self.steering_map: dict[str, dict[str, Any]] = {}

    def steering_of(self, ticket_id: str) -> dict[str, Any]:
        return self.steering_map[ticket_id]

    def close(self) -> None:
        self.store.close()


@pytest.fixture
def env(tmp_path: Path) -> Env:
    e = Env(tmp_path)
    yield e
    e.close()


def add_dispatchable_ticket(
    env: Env,
    ticket_id: str,
    *,
    state: str = "pool",
    attempts_used: int = 0,
    current_rung: str | None = None,
    integration_failures: int = 0,
    tier1_checks: list[str] | None = None,
) -> None:
    """Add + approve a ticket eligible for claim+dispatch (the dispatch-side
    tests, cases 1-10, 14-19)."""
    steering = base_steering()
    env.store.add_ticket(
        id=ticket_id,
        title=f"Ticket {ticket_id}",
        state=state,
        goal="Implement the feature end to end.",
        required_reading=[],
        target_scope=["src/foo.py"],
        acceptance_criteria=["foo() returns 42"],
        tier1_checks=tier1_checks if tier1_checks is not None else [],
        lane_hint=None,
        attempts_used=attempts_used,
        current_rung=current_rung,
        integration_failures=integration_failures,
    )
    approve(env.store, ticket_id, steering=steering)
    env.steering_map[ticket_id] = steering


def add_parked_ticket_for_weave(
    env: Env, ticket_id: str, *, work_product: str | Path
) -> None:
    """Add a ticket already at PARKED, ready for the real Weaver (cases
    11-13, 20) — mirrors `test_weaver.py`'s `add_parked_ticket`."""
    env.store.add_ticket(
        id=ticket_id,
        title=f"Ticket {ticket_id}",
        state=PARKED,
        work_product=str(work_product),
        tier1_checks={"tests": "true"},
        acceptance_criteria=["does the thing"],
    )


def stub_ctx_of(ticket_id: str) -> dict[str, Any]:
    """Mirrors `test_weaver.py`'s `stub_ctx_of` — a complete `_CTX_FIELDS`
    dict for the real `Weaver`'s own event emission."""
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


def make_real_weaver(env: Env, *, run_checks_fn: Any, critic: Any) -> Weaver:
    return Weaver(
        store=env.store,
        record_plane=env.record_plane,
        staging_repo=env.staging_repo,
        run_checks_fn=run_checks_fn,
        critic=critic,
        checker_image="unused:image",
        flake_reruns=0,
        protected_paths=[],
        journal_path=env.tmp_path / "weave-journal.jsonl",
        ctx_of=stub_ctx_of,
        filing_max_filings=5,
        filing_max_bytes=16384,
    )


def make_met_critic():
    from stigmergy.critic import Critic

    class _Client:
        def __call__(self, prompt, *, model, **kwargs):
            return {"outcome": "met", "tier": 1, "reason": "looks good", "severity": "none"}

    return Critic(
        client=_Client(), model="stub-model", decoding_params={"temperature": 0.0},
        template="Judge the artifact.",
    )


def make_unmet_critic():
    from stigmergy.critic import Critic

    class _Client:
        def __call__(self, prompt, *, model, **kwargs):
            return {"outcome": "unmet", "tier": 1, "reason": "missing tests", "severity": "low"}

    return Critic(
        client=_Client(), model="stub-model", decoding_params={"temperature": 0.0},
        template="Judge the artifact.",
    )


def make_infra_critic():
    from stigmergy.critic import Critic

    class _Client:
        def __call__(self, prompt, *, model, **kwargs):
            raise RuntimeError("provider 503")

    return Critic(
        client=_Client(), model="stub-model", decoding_params={"temperature": 0.0},
        template="Judge the artifact.",
    )


class FakeHandle:
    """Duck-typed stand-in for a per-dispatch relay/egress handle (bead .25):
    a socket_path + a stop() spy — enough for the daemon setup/teardown seams,
    no real server/thread."""

    def __init__(self, socket_path: Path):
        self.socket_path = socket_path
        self.stop_calls = 0

    def stop(self) -> None:
        self.stop_calls += 1


class FakeSetup:
    """A relay/egress setup_fn (bead .25): records call args and returns a
    FakeHandle with a deterministic socket_path per call. Accepts *args so it
    fits both the egress signature (provisional_id, policy, runtime_dir) and
    the relay signature (provisional_id, runtime_dir)."""

    def __init__(self, tag: str = "sock"):
        self._tag = tag
        self.calls: list[tuple] = []
        self.handles: list[FakeHandle] = []

    def __call__(self, provisional_id: str, *rest) -> FakeHandle:
        self.calls.append((provisional_id, *rest))
        runtime_dir = rest[-1]  # last positional is runtime_dir for both sigs
        handle = FakeHandle(Path(runtime_dir) / f"{provisional_id}-{self._tag}.sock")
        self.handles.append(handle)
        return handle


def make_daemon(
    env: Env,
    *,
    spawn_fn: Any,
    run_checks_fn: Any = None,
    weaver: Any = None,
    spend_leash: Any = None,
    now_fn: Any = None,
    min_disk_bytes: int = 0,
    relay_setup_fn: Any = None,
    relay_teardown_fn: Any = None,
    egress_setup_fn: Any = None,
    egress_teardown_fn: Any = None,
) -> Daemon:
    if run_checks_fn is None:
        run_checks_fn = ScriptedChecks([pass_result("lint"), pass_result("pytest")])
    if weaver is None:
        weaver = FakeWeaver()
    if spend_leash is None:
        budgets = Budgets(dispatches=1000, usd=1000.0, gate_calls=1000, reserve_usd=1.0)
        spend_leash = SpendLeash(budgets, env.registry)
    if now_fn is None:
        now_fn = FakeClock()
    extra: dict[str, Any] = {}
    if relay_setup_fn is not None:
        extra["relay_setup_fn"] = relay_setup_fn
    if relay_teardown_fn is not None:
        extra["relay_teardown_fn"] = relay_teardown_fn
    if egress_setup_fn is not None:
        extra["egress_setup_fn"] = egress_setup_fn
    if egress_teardown_fn is not None:
        extra["egress_teardown_fn"] = egress_teardown_fn
    return Daemon(
        store=env.store,
        record_plane=env.record_plane,
        notification_store=env.notification_store,
        notifier=env.notifier,
        spend_leash=spend_leash,
        charter=env.charter,
        registry=env.registry,
        rig_paths=env.rig_paths,
        capability_store=env.capability_store,
        checker_image="unused:checker-image",
        weaver=weaver,
        container_reaper=env.reaper,
        steering_of=env.steering_of,
        spawn_fn=spawn_fn,
        run_checks_fn=run_checks_fn,
        now_fn=now_fn,
        min_disk_bytes=min_disk_bytes,
        disk_path=str(env.rig_paths["repo_root"]),
        **extra,
    )


def dispatch_events(env: Env) -> list[dict]:
    return [e for e in env.record_plane.read_events() if e["event_type"] == "dispatch"]


def check_events(env: Env) -> list[dict]:
    return [e for e in env.record_plane.read_events() if e["event_type"] == "check"]


def disposition_events(env: Env, ticket_id: str | None = None) -> list[dict]:
    events = [e for e in env.record_plane.read_events() if e["event_type"] == "disposition"]
    if ticket_id is not None:
        events = [e for e in events if e.get("ticket") == ticket_id]
    return events


# ==========================================================================
# case 1: happy path
# ==========================================================================


def test_case1_happy_path_approved_ticket_ends_parked_with_full_event_trail(env: Env) -> None:
    """1. Happy path (bead's own VERIFY case): one approved, eligible
    ticket; stubbed spawn_fn returns DONE with usage; stubbed run_checks_fn
    returns all-PASS. One poll_once() call -> ticket ends at PARKED; event
    trail contains, in order, a DISPATCH event, CHECK event(s), and a
    DISPOSITION event; capability_store shows the capability was minted
    then revoked exactly once."""
    add_dispatchable_ticket(env, "t-1")
    spawn_fn = ScriptedSpawn([make_result(DispatchStatus.DONE, bundle_ref="/tmp/t-1-work.bundle")])
    run_checks_fn = ScriptedChecks([pass_result("lint"), pass_result("pytest")])
    daemon = make_daemon(env, spawn_fn=spawn_fn, run_checks_fn=run_checks_fn)

    summary = daemon.poll_once()

    assert summary.dispatched_ticket == "t-1"
    assert summary.dispatch_status == "done"
    assert env.store.get_ticket("t-1")["state"] == PARKED
    # bead .94: the driver's bundle must be persisted as work_product so the
    # weaver has something to apply (else the weave fails integration-conflict).
    assert env.store.get_ticket("t-1")["work_product"] == "/tmp/t-1-work.bundle"

    events = env.record_plane.read_events()
    types_in_order = [e["event_type"] for e in events]
    assert types_in_order[0] == "dispatch"
    assert "check" in types_in_order
    assert types_in_order[-1] == "disposition"
    # DISPATCH strictly before every CHECK, every CHECK strictly before the
    # terminal DISPOSITION.
    dispatch_idx = types_in_order.index("dispatch")
    check_idxs = [i for i, t in enumerate(types_in_order) if t == "check"]
    disposition_idx = len(types_in_order) - 1
    assert all(dispatch_idx < i < disposition_idx for i in check_idxs)

    assert len(env.capability_store.minted) == 1
    cap = env.capability_store.minted[0]
    assert env.capability_store.is_live(cap.token) is False


# ==========================================================================
# case 2: AC7 seeded 429 storm
# ==========================================================================


def test_case2_infra_storm_never_burns_attempts_or_escalates(env: Env) -> None:
    """2. AC7 seeded 429 storm: stubbed spawn_fn always returns INFRA
    (simulating repeated provider 429s) for _CIRCUIT_BREAKER_THRESHOLD - 1
    consecutive poll_once() calls -> every call: ticket stays claimable
    (attempts_used unchanged, integration_failures unchanged, ticket back
    at POOL each time), zero DISPOSITION events of any escalation kind,
    should_halt=False each time, backoff_seconds increases (or stays
    capped) each trip."""
    add_dispatchable_ticket(env, "t-2")
    spawn_fn = AlwaysInfraSpawn()
    daemon = make_daemon(env, spawn_fn=spawn_fn)

    backoffs = []
    n = _CIRCUIT_BREAKER_THRESHOLD - 1
    for _ in range(n):
        summary = daemon.poll_once()
        assert summary.should_halt is False
        assert summary.infra_trip is True
        backoffs.append(summary.backoff_seconds)

        row = env.store.get_ticket("t-2")
        assert row["state"] == POOL
        assert row["attempts_used"] == 0
        assert row["integration_failures"] == 0

    # Strictly increasing (or capped) backoff.
    for a, b in zip(backoffs, backoffs[1:], strict=False):
        assert b >= a
        assert b <= _BACKOFF_CAP_SECONDS

    escalation_dispositions = [
        e for e in disposition_events(env, "t-2") if e.get("disposition") == "escalated"
    ]
    assert escalation_dispositions == []


# ==========================================================================
# case 3: circuit breaker halt
# ==========================================================================


def test_case3_circuit_breaker_halts_loudly_past_threshold(env: Env) -> None:
    """3. Circuit breaker halt: one more INFRA trip past the threshold ->
    that poll_once() call returns should_halt=True; a
    NotificationIntent(kind="halt") is persisted in notification_store; a
    deliver_pending attempt was made (assert via the injected sender's call
    count, even if it's a no-op/failing fake)."""
    add_dispatchable_ticket(env, "t-3")
    spawn_fn = AlwaysInfraSpawn()
    daemon = make_daemon(env, spawn_fn=spawn_fn)

    summaries = [daemon.poll_once() for _ in range(_CIRCUIT_BREAKER_THRESHOLD)]

    for summary in summaries[:-1]:
        assert summary.should_halt is False
    assert summaries[-1].should_halt is True
    assert summaries[-1].consecutive_infra_trips == _CIRCUIT_BREAKER_THRESHOLD

    halt_intents = [i for i in env.notification_store.all_intents() if i.kind == "halt"]
    assert len(halt_intents) == 1

    assert len(env.sender.calls) >= 1


# ==========================================================================
# case 4: circuit breaker resets on a green dispatch
# ==========================================================================


def test_case4_circuit_breaker_resets_on_green_dispatch(env: Env) -> None:
    """4. Circuit breaker resets on a green dispatch: N-1 infra trips
    (below threshold), then one DONE dispatch -> the next INFRA trip
    afterward starts counting from 1 again, not N."""
    add_dispatchable_ticket(env, "t-a")
    add_dispatchable_ticket(env, "t-b")

    n = _CIRCUIT_BREAKER_THRESHOLD - 1
    results = [make_result(DispatchStatus.INFRA, usage=zero_tokens()) for _ in range(n)]
    results.append(make_result(DispatchStatus.DONE))
    results.append(make_result(DispatchStatus.INFRA, usage=zero_tokens()))
    spawn_fn = ScriptedSpawn(results)
    run_checks_fn = ScriptedChecks([pass_result("lint"), pass_result("pytest")])
    daemon = make_daemon(env, spawn_fn=spawn_fn, run_checks_fn=run_checks_fn)

    for _ in range(n):
        summary = daemon.poll_once()
        assert summary.dispatched_ticket == "t-a"
        assert summary.infra_trip is True

    green_summary = daemon.poll_once()
    assert green_summary.dispatched_ticket == "t-a"
    assert green_summary.dispatch_status == "done"
    assert green_summary.consecutive_infra_trips == 0
    assert env.store.get_ticket("t-a")["state"] == PARKED

    final_summary = daemon.poll_once()
    assert final_summary.dispatched_ticket == "t-b"
    assert final_summary.infra_trip is True
    assert final_summary.consecutive_infra_trips == 1  # NOT threshold/N


# ==========================================================================
# case 5: wedge -> retry ladder
# ==========================================================================


def test_case5_wedge_retries_same_rung(env: Env) -> None:
    """5. Wedge -> retry ladder: stubbed spawn_fn returns WEDGED -> ticket
    goes IN_FLIGHT->FAILED->POOL (same rung, attempts_used incremented by
    1), DISPOSITION event present with attempt_kind matching what
    decide_retry returned."""
    add_dispatchable_ticket(env, "t-5")
    spawn_fn = ScriptedSpawn([make_result(DispatchStatus.WEDGED, usage=zero_tokens())])
    daemon = make_daemon(env, spawn_fn=spawn_fn)

    summary = daemon.poll_once()

    assert summary.dispatch_status == "wedged"
    row = env.store.get_ticket("t-5")
    assert row["state"] == POOL
    assert row["attempts_used"] == 1
    assert row["current_rung"] == "cheap"

    dispositions = disposition_events(env, "t-5")
    assert len(dispositions) == 1
    assert dispositions[0]["attempt_kind"] == "clean-restart"
    assert dispositions[0]["disposition"] == "pool-reentry"


# ==========================================================================
# case 6: ceiling trip -> DEGENERATE
# ==========================================================================


def test_case6_ceiling_trip_uses_degenerate_failure_class(env: Env) -> None:
    """6. Ceiling trip -> DEGENERATE: stubbed spawn_fn returns FAILED with
    ceiling_trip="output_tokens" -> same retry-ladder shape as case 5, but
    confirm the daemon used FailureClass.DEGENERATE (pin the observable
    difference against an ordinary TIER1_FAIL: DEGENERATE and WEDGE share
    the identical "clean-restart" attempt_kind mapping in decide_retry,
    which is DISTINCT from TIER1_FAIL's "tier1-repair" — case 7 pins that
    contrast)."""
    add_dispatchable_ticket(env, "t-6")
    spawn_fn = ScriptedSpawn(
        [make_result(DispatchStatus.FAILED, ceiling_trip="output_tokens")]
    )
    daemon = make_daemon(env, spawn_fn=spawn_fn)

    summary = daemon.poll_once()

    assert summary.dispatch_status == "failed"
    row = env.store.get_ticket("t-6")
    assert row["state"] == POOL
    assert row["attempts_used"] == 1

    dispositions = disposition_events(env, "t-6")
    assert len(dispositions) == 1
    # The observable pin: DEGENERATE (like WEDGE) maps to "clean-restart",
    # never TIER1_FAIL's "tier1-repair" -- this is what proves the daemon
    # routed via DEGENERATE, not an ordinary TIER1_FAIL.
    assert dispositions[0]["attempt_kind"] == "clean-restart"

    dispatch_ev = dispatch_events(env)[0]
    assert dispatch_ev["ceiling_trip"] == "output_tokens"
    assert dispatch_ev["status"] == "failed"


# ==========================================================================
# case 7: ordinary FAILED (no ceiling, no bundle) -> TIER1_FAIL
# ==========================================================================


def test_case7_ordinary_failed_no_ceiling_no_bundle_handled_gracefully(env: Env) -> None:
    """7. Ordinary FAILED (no ceiling, no bundle) -> TIER1_FAIL path,
    bundle-absent handled gracefully: stubbed spawn_fn returns FAILED,
    ceiling_trip=None, bundle_ref=None -> ticket retries via
    FailureClass.TIER1_FAIL; no exception from a missing bundle_ref; the
    daemon does not attempt to pre-apply anything."""
    add_dispatchable_ticket(env, "t-7")
    spawn_fn = ScriptedSpawn(
        [make_result(DispatchStatus.FAILED, ceiling_trip=None, bundle_ref=None)]
    )
    daemon = make_daemon(env, spawn_fn=spawn_fn)

    summary = daemon.poll_once()  # must not raise

    assert summary.dispatch_status == "failed"
    row = env.store.get_ticket("t-7")
    assert row["state"] == POOL
    assert row["attempts_used"] == 1

    dispositions = disposition_events(env, "t-7")
    assert len(dispositions) == 1
    assert dispositions[0]["attempt_kind"] == "tier1-repair"  # ordinary TIER1_FAIL mapping


# ==========================================================================
# case 8: Tier-1 check failure
# ==========================================================================


def test_case8_tier1_check_failure_retries_never_reaches_parked(env: Env) -> None:
    """8. Tier-1 check failure: stubbed spawn_fn returns DONE; stubbed
    run_checks_fn returns one FAIL result among otherwise-PASS results ->
    ticket retries via FailureClass.TIER1_FAIL (never reaches
    TIER1_GREEN/PARKED)."""
    add_dispatchable_ticket(env, "t-8")
    spawn_fn = ScriptedSpawn([make_result(DispatchStatus.DONE)])
    run_checks_fn = ScriptedChecks([pass_result("lint"), fail_result("pytest")])
    daemon = make_daemon(env, spawn_fn=spawn_fn, run_checks_fn=run_checks_fn)

    summary = daemon.poll_once()

    assert summary.dispatch_status == "done"
    row = env.store.get_ticket("t-8")
    assert row["state"] == POOL
    assert row["attempts_used"] == 1

    dispositions = disposition_events(env, "t-8")
    assert len(dispositions) == 1
    assert dispositions[0]["attempt_kind"] == "tier1-repair"


# ==========================================================================
# case 9: FLAKY counts as green (prove-can-fail priority)
# ==========================================================================


def test_case9_flaky_check_still_counts_as_green(env: Env) -> None:
    """9. FLAKY counts as green: stubbed run_checks_fn returns one FLAKY
    (pass-on-rerun) among otherwise-PASS results -> ticket STILL reaches
    TIER1_GREEN->PARKED (prove-can-fail candidate)."""
    add_dispatchable_ticket(env, "t-9")
    spawn_fn = ScriptedSpawn([make_result(DispatchStatus.DONE)])
    run_checks_fn = ScriptedChecks([pass_result("lint"), flaky_result("pytest")])
    daemon = make_daemon(env, spawn_fn=spawn_fn, run_checks_fn=run_checks_fn)

    summary = daemon.poll_once()

    assert summary.dispatch_status == "done"
    assert env.store.get_ticket("t-9")["state"] == PARKED

    dispositions = disposition_events(env, "t-9")
    assert len(dispositions) == 1
    assert dispositions[0]["disposition"] == "parked"


# ==========================================================================
# case 10: ladder exhaustion -> escalation + notification
# ==========================================================================


def test_case10_ladder_exhaustion_escalates_and_notifies(env: Env) -> None:
    """10. Ladder exhaustion -> escalation + notification: seed a ticket
    already at attempts_used=attempts_per_rung-1, current_rung=<last
    ladder rung>; stubbed spawn_fn returns WEDGED one more time -> ticket
    reaches ESCALATED; a NotificationIntent(kind="escalation") is
    persisted."""
    attempts_per_rung = env.charter.raw["loop"]["retries"]["attempts_per_rung"]
    ladder = env.charter.raw["stepup"]["ladder"]
    add_dispatchable_ticket(
        env, "t-10", attempts_used=attempts_per_rung - 1, current_rung=ladder[-1]
    )
    spawn_fn = ScriptedSpawn([make_result(DispatchStatus.WEDGED, usage=zero_tokens())])
    daemon = make_daemon(env, spawn_fn=spawn_fn)

    summary = daemon.poll_once()

    assert summary.dispatch_status == "wedged"
    row = env.store.get_ticket("t-10")
    assert row["state"] == ESCALATED
    assert row["attempts_used"] == attempts_per_rung

    escalation_intents = [
        i for i in env.notification_store.all_intents() if i.kind == "escalation"
    ]
    assert len(escalation_intents) == 1
    assert escalation_intents[0].ticket == "t-10"

    dispositions = disposition_events(env, "t-10")
    assert len(dispositions) == 1
    assert dispositions[0]["disposition"] == "escalated"


# ==========================================================================
# case 11: weave-infra never routes through decide_retry/apply_decision
# (prove-can-fail priority)
# ==========================================================================


def test_case11_weave_infra_never_routes_through_retry_ladder(env: Env) -> None:
    """11. Weave-infra never routes through decide_retry/apply_decision:
    stub the injected weaver (a real Weaver wired to a critic stub that
    always raises CriticInfraError) so weave() returns a WeaveResult with
    outcome="infra" for a ticket already at PARKED -> after the daemon
    processes this result, the ticket is still PARKED (never touched via
    transition/decide_retry at all — assert no IllegalTransition was ever
    risked, and that attempts_used/integration_failures are unchanged).
    Infra-trip counter increments."""
    bundle = make_bundle(env.tmp_path, env.staging_repo, name="t11", files={"f.txt": "x\n"})
    add_parked_ticket_for_weave(env, "t11", work_product=bundle)

    weaver = make_real_weaver(
        env, run_checks_fn=ScriptedChecks([pass_result()]), critic=make_infra_critic()
    )
    daemon = make_daemon(env, spawn_fn=RaisingSpawn(), weaver=weaver)

    summary = daemon.poll_once()  # must not raise IllegalTransition

    assert summary.weave_ran is True
    assert summary.weave_results == ("t11",)
    assert summary.infra_trip is True
    assert summary.consecutive_infra_trips == 1

    row = env.store.get_ticket("t11")
    assert row["state"] == PARKED
    assert row["attempts_used"] == 0
    assert row["integration_failures"] == 0


# ==========================================================================
# case 12: weave rejected -> retry ladder
# ==========================================================================


def test_case12_weave_rejected_routes_through_retry_ladder(env: Env) -> None:
    """12. Weave rejected -> retry ladder: weaver.weave() returns a
    WeaveResult with outcome="rejected" -> ticket goes GATED->REJECTED->
    (POOL|ESCALATED) via decide_retry."""
    bundle = make_bundle(env.tmp_path, env.staging_repo, name="t12", files={"f.txt": "x\n"})
    add_parked_ticket_for_weave(env, "t12", work_product=bundle)

    weaver = make_real_weaver(
        env, run_checks_fn=ScriptedChecks([pass_result()]), critic=make_unmet_critic()
    )
    daemon = make_daemon(env, spawn_fn=RaisingSpawn(), weaver=weaver)

    summary = daemon.poll_once()

    assert summary.weave_ran is True
    assert summary.weave_results == ("t12",)

    row = env.store.get_ticket("t12")
    assert row["state"] == POOL  # not yet at ladder exhaustion
    assert row["attempts_used"] == 1

    dispositions = disposition_events(env, "t12")
    # weaver's own "rejected" disposition + this module's own pool-reentry
    # disposition (see daemon.py's module docstring, deviation 2).
    assert any(d["disposition"] == "rejected" for d in dispositions)
    assert any(d["disposition"] == "pool-reentry" for d in dispositions)


# ==========================================================================
# case 13: weave landed -> ticket rests at LANDED (bead .89)
# ==========================================================================


def test_case13_weave_landed_rests_at_landed(env: Env) -> None:
    """13. Weave landed: weave() returns outcome="landed" -> ticket
    GATED->LANDED and RESTS there (bead .89: v0 does not advance to DONE;
    DAG intake + spend key on state=="landed"), DISPOSITION(disposition=
    "landed") emitted."""
    bundle = make_bundle(env.tmp_path, env.staging_repo, name="t13", files={"f.txt": "x\n"})
    add_parked_ticket_for_weave(env, "t13", work_product=bundle)

    weaver = make_real_weaver(
        env, run_checks_fn=ScriptedChecks([pass_result()]), critic=make_met_critic()
    )
    daemon = make_daemon(env, spawn_fn=RaisingSpawn(), weaver=weaver)

    summary = daemon.poll_once()

    assert summary.weave_ran is True
    assert summary.weave_results == ("t13",)
    assert env.store.get_ticket("t13")["state"] == LANDED

    dispositions = disposition_events(env, "t13")
    assert any(d["disposition"] == "landed" for d in dispositions)


# ==========================================================================
# case 14: heartbeat written every poll
# ==========================================================================


def test_case14_heartbeat_increases_every_poll_even_with_nothing_dispatched(env: Env) -> None:
    """14. Heartbeat written every poll: store.get_meta("daemon_heartbeat_at")
    changes (increases) across successive poll_once() calls, including a
    poll where nothing was dispatched."""
    daemon = make_daemon(env, spawn_fn=RaisingSpawn())  # never called: nothing eligible

    daemon.poll_once()
    first = float(env.store.get_meta("daemon_heartbeat_at"))
    daemon.poll_once()
    second = float(env.store.get_meta("daemon_heartbeat_at"))

    assert second > first


# ==========================================================================
# case 15: recover_on_start propagates RecoveryError
# ==========================================================================


def test_case15_recover_on_start_propagates_recovery_error(env: Env) -> None:
    """15. recover_on_start propagates RecoveryError: an injected
    recover-path condition (a disk_path/min_disk_bytes combination that
    fails the headroom check) makes recover_on_start() raise — never
    swallowed, never converted into a PollSummary."""
    daemon = make_daemon(env, spawn_fn=RaisingSpawn(), min_disk_bytes=10**18)

    with pytest.raises(RecoveryError):
        daemon.recover_on_start()


# ==========================================================================
# case 16: capability revoked even when spawn_fn raises unexpectedly
# ==========================================================================


def test_case16_capability_revoked_even_when_spawn_raises(env: Env) -> None:
    """16. Capability revoked even when spawn_fn raises unexpectedly: an
    injected spawn_fn that raises a bare Exception — the finally around
    prepare->spawn->revoke still calls capability_store.revoke(dispatch_id)
    (assert via capability_store.is_live(cap.token) is False afterward)
    even though poll_once() itself may then propagate or wrap that
    exception (this daemon lets it propagate, undecorated)."""
    add_dispatchable_ticket(env, "t-16")
    daemon = make_daemon(env, spawn_fn=RaisingSpawn())

    with pytest.raises(RuntimeError):
        daemon.poll_once()

    assert len(env.capability_store.minted) == 1
    cap = env.capability_store.minted[0]
    assert env.capability_store.is_live(cap.token) is False


# ==========================================================================
# bead .25: credential-relay setup/teardown seam. Cases R1-R6 (incl. case 29).
# ==========================================================================


def test_relay_r1_no_relay_setup_fn_leaves_relay_socket_none(env: Env) -> None:
    """R1 (backward-compat gate): with no relay_setup_fn (default), the
    dispatch runs with no relay and model_cfg.relay_socket is None."""
    add_dispatchable_ticket(env, "t-r1")
    spawn = ScriptedSpawn([make_result(DispatchStatus.DONE)])
    daemon = make_daemon(env, spawn_fn=spawn)
    daemon.poll_once()
    assert spawn.calls[0][2].relay_socket is None


def test_relay_r2_socket_threaded_into_dispatch(env: Env) -> None:
    """R2: the relay handle's served socket_path (read from the handle) is
    exactly what prepare_dispatch threads into model_cfg.relay_socket — no
    drift between served and mounted."""
    add_dispatchable_ticket(env, "t-r2")
    spawn = ScriptedSpawn([make_result(DispatchStatus.DONE)])
    relay_setup = FakeSetup("relay")
    daemon = make_daemon(env, spawn_fn=spawn, relay_setup_fn=relay_setup)
    daemon.poll_once()
    assert len(relay_setup.handles) == 1
    assert spawn.calls[0][2].relay_socket == relay_setup.handles[0].socket_path


def test_relay_case29_teardown_revoke_relay_stop_and_egress(env: Env) -> None:
    """Case 29: a normal dispatch cycle revokes the capability AND stops the
    relay AND tears down egress."""
    add_dispatchable_ticket(env, "t-29")
    spawn = ScriptedSpawn([make_result(DispatchStatus.DONE)])
    relay_setup = FakeSetup("relay")
    egress_setup = FakeSetup("egress")
    daemon = make_daemon(env, spawn_fn=spawn, relay_setup_fn=relay_setup,
                         egress_setup_fn=egress_setup)
    daemon.poll_once()
    cap = env.capability_store.minted[0]
    assert env.capability_store.is_live(cap.token) is False   # revoked
    assert relay_setup.handles[0].stop_calls == 1             # relay stopped
    assert egress_setup.handles[0].stop_calls == 1            # egress torn down


def test_relay_case29b_teardown_runs_even_when_spawn_raises(env: Env) -> None:
    """Case 29b: spawn_fn raises -> the exception propagates AFTER the finally,
    and revoke + relay stop + egress stop all still ran."""
    add_dispatchable_ticket(env, "t-29b")
    relay_setup = FakeSetup("relay")
    egress_setup = FakeSetup("egress")
    daemon = make_daemon(env, spawn_fn=RaisingSpawn(), relay_setup_fn=relay_setup,
                         egress_setup_fn=egress_setup)
    with pytest.raises(RuntimeError):
        daemon.poll_once()
    cap = env.capability_store.minted[0]
    assert env.capability_store.is_live(cap.token) is False
    assert relay_setup.handles[0].stop_calls == 1
    assert egress_setup.handles[0].stop_calls == 1


def test_relay_r5_teardown_independence_relay_failure_still_tears_egress(env: Env) -> None:
    """R5: a relay_teardown_fn that raises must NOT skip egress teardown, and
    revoke must have run first regardless (each teardown independently
    guarded)."""
    add_dispatchable_ticket(env, "t-r5")
    spawn = ScriptedSpawn([make_result(DispatchStatus.DONE)])
    relay_setup = FakeSetup("relay")
    egress_setup = FakeSetup("egress")

    def exploding_relay_teardown(handle):
        raise RuntimeError("relay stop boom")

    daemon = make_daemon(env, spawn_fn=spawn, relay_setup_fn=relay_setup,
                         relay_teardown_fn=exploding_relay_teardown,
                         egress_setup_fn=egress_setup)
    daemon.poll_once()  # must NOT raise — teardown failure is swallowed+logged
    cap = env.capability_store.minted[0]
    assert env.capability_store.is_live(cap.token) is False   # revoke ran first
    assert egress_setup.handles[0].stop_calls == 1            # egress still torn down


def test_relay_egress_policy_uses_current_rung_not_entry_lane(env: Env, monkeypatch) -> None:
    """bead .25 audit (codex HIGH): a stepped-up ticket's egress policy must be
    resolved for its CURRENT RUNG, not the entry lane. Before the fix,
    _claim_and_prepare resolved the egress lane WITHOUT rung while
    prepare_dispatch used the rung -> mismatch. Spy policy_for_lane's lane_name."""
    import stigmergy.egress as egress_mod

    captured: list[str] = []
    real = egress_mod.policy_for_lane

    def spy(charter_raw, lane_name, **kw):
        captured.append(lane_name)
        return real(charter_raw, lane_name, **kw)

    monkeypatch.setattr(egress_mod, "policy_for_lane", spy)

    add_dispatchable_ticket(env, "t-rung", current_rung="exquisite")
    spawn = ScriptedSpawn([make_result(DispatchStatus.DONE)])
    daemon = make_daemon(env, spawn_fn=spawn,
                         relay_setup_fn=FakeSetup("relay"),
                         egress_setup_fn=FakeSetup("egress"))
    daemon.poll_once()

    assert captured == ["exquisite"], f"egress resolved wrong lane: {captured}"
    # sanity: the dispatch actually ran the exquisite rung's model
    assert spawn.calls[0][2].model == env.charter.raw["lanes"]["exquisite"]["model"]


def test_relay_r6_teardown_order_revoke_then_relay_then_egress(env: Env) -> None:
    """R6: teardown order is revoke -> relay stop -> egress stop (revoke is the
    load-bearing security act and goes first)."""
    add_dispatchable_ticket(env, "t-r6")
    spawn = ScriptedSpawn([make_result(DispatchStatus.DONE)])
    order: list[str] = []

    class OrderStore(SpyCapabilityStore):
        def revoke(self, dispatch_id):  # type: ignore[override]
            order.append("revoke")
            return super().revoke(dispatch_id)

    env.capability_store = OrderStore()

    def relay_setup_fn(pid, runtime_dir):
        return FakeHandle(Path(runtime_dir) / f"{pid}.sock")

    def relay_teardown(handle):
        order.append("relay")

    def egress_setup_fn(pid, policy, runtime_dir):
        return FakeHandle(Path(runtime_dir) / f"{pid}.sock")

    def egress_teardown(handle):
        order.append("egress")

    daemon = make_daemon(env, spawn_fn=spawn, relay_setup_fn=relay_setup_fn,
                         relay_teardown_fn=relay_teardown, egress_setup_fn=egress_setup_fn,
                         egress_teardown_fn=egress_teardown)
    daemon.poll_once()
    assert order == ["revoke", "relay", "egress"]


# ==========================================================================
# case 17: transcript sealed with the capability token redacted
# (prove-can-fail priority)
# ==========================================================================


def test_case17_transcript_sealed_with_capability_token_redacted(env: Env) -> None:
    """17. Transcript sealed with the capability token redacted: the
    stubbed DispatchResult.transcript contains the literal capability
    token string somewhere in it -> the sealed blob (read back from
    record_plane's transcript store) does NOT contain it."""
    add_dispatchable_ticket(env, "t-17")
    daemon = make_daemon(env, spawn_fn=TokenEmbeddingSpawn())

    daemon.poll_once()

    cap = env.capability_store.minted[0]
    dispatch_ev = dispatch_events(env)[0]
    transcript_ref = dispatch_ev["transcript_ref"]
    blob_path = env.record_plane.transcripts_dir / transcript_ref
    sealed_content = blob_path.read_text(encoding="utf-8")

    assert cap.token not in sealed_content


# ==========================================================================
# case 18: prompt_artifact_hash present on every DISPATCH event
# ==========================================================================


def test_case18_prompt_artifact_hash_present_on_dispatch_event(env: Env) -> None:
    """18. prompt_artifact_hash present on every DISPATCH event: assert
    the emitted DISPATCH event's prompt_artifact_hash field is a
    non-empty string matching sha256(prompt_template)."""
    add_dispatchable_ticket(env, "t-18")
    daemon = make_daemon(env, spawn_fn=ScriptedSpawn([make_result(DispatchStatus.DONE)]))

    daemon.poll_once()

    dispatch_ev = dispatch_events(env)[0]
    expected = hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest()
    assert dispatch_ev["prompt_artifact_hash"] == expected
    assert isinstance(dispatch_ev["prompt_artifact_hash"], str)
    assert dispatch_ev["prompt_artifact_hash"] != ""


# ==========================================================================
# case 19: workers=1 — never claims a second ticket while one is in flight
# ==========================================================================


def test_case19_at_most_one_ticket_dispatched_per_poll(env: Env) -> None:
    """19. Workers=1 — never claims a second ticket while one is in
    flight: since poll_once() is synchronous (spawn blocks until done),
    this is naturally satisfied by construction — seed two eligible
    tickets and confirm a single poll_once() call dispatches at most one
    of them (the other stays pool/eligible)."""
    add_dispatchable_ticket(env, "t-a")
    add_dispatchable_ticket(env, "t-b")
    spawn_fn = ScriptedSpawn([make_result(DispatchStatus.DONE)])
    run_checks_fn = ScriptedChecks([pass_result("lint"), pass_result("pytest")])
    daemon = make_daemon(env, spawn_fn=spawn_fn, run_checks_fn=run_checks_fn)

    summary = daemon.poll_once()

    assert summary.dispatched_ticket == "t-a"
    assert len(spawn_fn.calls) == 1
    assert env.store.get_ticket("t-a")["state"] == PARKED
    assert env.store.get_ticket("t-b")["state"] == POOL


# ==========================================================================
# case 20: spend exhaustion stops new claims but lets the final weave run
# ==========================================================================


def test_case20_spend_exhaustion_blocks_claims_but_allows_final_weave(env: Env) -> None:
    """20. Spend exhaustion stops new claims but lets the final weave run:
    seed a SpendLeash already at (one dispatch away from) its cap;
    poll_once() does not claim a new ticket once exhausted, but a pending
    weave trigger (parked tickets exist) still runs via
    final_weave_allowed() — assert final_weave_allowed() is consulted/
    consumed at most once across the test's poll_once() calls (its
    one-shot side-effecting nature).

    Note on trigger choice: an eligible-but-unclaimed ticket sitting in
    `pool` counts toward `triggers.ACTIVE_STATES`, so it structurally
    prevents the `queue-drained` trigger (`active_count == 0`) from firing
    — this is not a contradiction in the frozen case, just a reason to
    drive the weave via the `max-wait` trigger instead (a `now_fn` set far
    enough past the parked ticket's real (wall-clock) `created_at` that
    `staging_max_wait_seconds` has already elapsed on the very first poll).
    """
    add_dispatchable_ticket(env, "t-not-dispatched")  # must never be claimed

    bundle = make_bundle(env.tmp_path, env.staging_repo, name="t20", files={"f.txt": "x\n"})
    add_parked_ticket_for_weave(env, "t20", work_product=bundle)

    exhausted_budgets = Budgets(dispatches=0, usd=100.0, gate_calls=0, reserve_usd=0.0)
    spy_leash = SpySpendLeash(exhausted_budgets, env.registry)
    assert spy_leash.exhausted() is True

    weaver = make_real_weaver(
        env, run_checks_fn=ScriptedChecks([pass_result()]), critic=make_met_critic()
    )
    max_wait_seconds = env.charter.raw["loop"]["cadences"]["staging_max_wait_seconds"]
    import time as _time

    far_future_now = FakeClock(start=_time.time() + max_wait_seconds + 1000.0, step=1.0)
    daemon = make_daemon(
        env, spawn_fn=RaisingSpawn(), weaver=weaver, spend_leash=spy_leash, now_fn=far_future_now
    )

    first = daemon.poll_once()
    assert first.dispatched_ticket is None  # can_dispatch() False -> never claimed
    assert first.weave_ran is True
    assert first.weave_results == ("t20",)
    assert env.store.get_ticket("t20")["state"] == LANDED
    assert env.store.get_ticket("t-not-dispatched")["state"] == POOL

    second = daemon.poll_once()
    assert second.dispatched_ticket is None
    assert second.weave_ran is False  # nothing parked anymore

    assert spy_leash.final_weave_allowed_calls == 1


# ==========================================================================
# D14: worker ticket-filing harvest hook (bead workspace-e2uh.38, AC14 case 2
# end-to-end + fail-closed isolation). The harvest runs host-side AFTER the
# container is killed (i.e. after spawn_fn returns), reading the UNCOMMITTED
# `/work/.stigmergy/filed-tickets.json` from the work_clone.
# ==========================================================================


class FilingSpawn:
    """spawn_fn that writes an (uncommitted) filed-tickets.json into the
    work_clone, as a real worker would, then returns DONE."""

    def __init__(self, proposals: Any):
        self._proposals = proposals
        self.calls = 0

    def __call__(self, task_pack, work_clone, model_cfg, capability, budgets):
        import json

        self.calls += 1
        dest = Path(work_clone) / ".stigmergy" / "filed-tickets.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(self._proposals))
        return make_result(DispatchStatus.DONE)


class BadFilingSpawn:
    """spawn_fn that writes a MALFORMED filings file then returns DONE."""

    def __call__(self, task_pack, work_clone, model_cfg, capability, budgets):
        dest = Path(work_clone) / ".stigmergy" / "filed-tickets.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("{ not valid json ]]")
        return make_result(DispatchStatus.DONE)


def test_worker_filings_harvested_host_side_after_dispatch(env: Env) -> None:
    add_dispatchable_ticket(env, "t-1")
    spawn_fn = FilingSpawn([{"title": "flaky test", "description": "test_x flakes 1/10"}])
    daemon = make_daemon(env, spawn_fn=spawn_fn)

    daemon.poll_once()

    # harvested into filed_tickets: unapproved, un-claimable, worker origin.
    assert env.store.count_untriaged_filings() == 1
    filed = env.store.list_filed_tickets()[0]
    assert filed["origin_role"] == "worker"
    assert filed["origin_parent_ticket"] == "t-1"
    assert env.store.get_ticket(filed["id"]) is None  # never a claimable ticket

    # a ticket-filed event was emitted alongside the normal dispatch trail.
    filed_ev = [e for e in env.record_plane.read_events() if e["event_type"] == "ticket-filed"]
    assert len(filed_ev) == 1
    assert filed_ev[0]["outcome"] == "accepted"
    assert filed_ev[0]["origin"]["parent_ticket"] == "t-1"


def test_harvest_failure_is_isolated_from_the_dispatch_outcome(env: Env) -> None:
    """A malformed filings file must never corrupt or abort the dispatch: the
    ticket still reaches its normal terminal state (PARKED) and nothing files
    (fail-closed, isolated — build item 1)."""
    add_dispatchable_ticket(env, "t-1")
    daemon = make_daemon(env, spawn_fn=BadFilingSpawn())

    summary = daemon.poll_once()

    assert summary.dispatch_status == "done"
    assert env.store.get_ticket("t-1")["state"] == PARKED
    assert env.store.count_untriaged_filings() == 0


# ==========================================================================
# bead .90: recover_on_start fails loudly if the dispatch_base branch is
# absent from the rig repo (belt-and-suspenders for the scaffold fix)
# ==========================================================================


def test_recover_on_start_raises_when_dispatch_base_branch_absent(
    env: Env, tmp_path: Path
) -> None:
    """A rig repo missing the charter's dispatch_base branch ('staging') must
    fail LOUDLY at startup (DaemonError), not dispatch obscurely against
    base_oid=None."""
    no_staging = tmp_path / "no_staging_repo"
    run_git(None, ["init", "--quiet", "-b", "main", str(no_staging)])
    (no_staging / "README.md").write_text("x\n")
    run_git(no_staging, ["add", "README.md"])
    run_git(no_staging, ["commit", "--quiet", "-m", "init"])
    env.rig_paths["repo_root"] = no_staging
    daemon = make_daemon(env, spawn_fn=RaisingSpawn())
    with pytest.raises(DaemonError):
        daemon.recover_on_start()
