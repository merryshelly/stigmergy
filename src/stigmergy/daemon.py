"""Loop daemon — the poll-loop integration assembly (SPEC.md §2 System
Overview, §3 Stations, §9 Failure Handling & Loop Mechanics, §10 AC7/AC9;
bead .22 build spec).

**This is the integration bead** (spec gap #1, per the bead's own
description): it wires together every already-built station — `intake`,
`statemachine`, `recover`, `spend`, `triggers`, `weaver`, `checks`,
`dispatch`, `drivers.claude_code`, `relay`, `egress`, `records`, `status`,
`notify` — into one poll loop, per the frozen state-transition protocol
(bead .22 build spec §1), event emission contract (§2), and circuit-breaker
mechanics (§3).

**Testability contract.** :meth:`Daemon.poll_once` runs ONE poll cycle as a
plain function call — no `time.sleep`, no real I/O beyond what its injected
collaborators do. Tests drive N cycles by calling it N times directly.
:meth:`Daemon.run_forever` is the thin real-sleep-and-repeat wrapper for the
systemd unit; it is the ONLY place `sys.exit` is ever called from this
module. `daemon.py` itself never imports a provider SDK and never shells
out to `podman` directly — every host-touching or nondeterministic
collaborator (`spawn_fn`, `run_checks_fn`, `container_reaper`,
`egress_setup_fn`/`egress_teardown_fn`, the `notifier`'s `Sender`) is
injected, mirroring the whole codebase's DI discipline.

**Documented deviations from the build spec (read before trusting the
"exactly as written" framing of §1 — these are real discrepancies against
the actual shipped module source, resolved conservatively and flagged
here, not silently papered over; see the sub report for the full list):**

1. **§1.1 INFRA fast-path step 2.** The build spec offers two "equally
   acceptable" choices for the `INFRA` fast path: call `apply_decision`
   anyway (relying on idempotence) or skip straight to the disposition.
   The FIRST choice is actually **impossible**: step 1 already moves the
   ticket `IN_FLIGHT -> POOL`; `apply_decision`'s `decide_retry(INFRA)`
   branch always targets `next_state=POOL`, and `POOL -> POOL` is not a
   legal edge (`LEGAL_TRANSITIONS[POOL] == {ELIGIBLE, CLAIMED}`) — calling
   `apply_decision` here would raise `IllegalTransition`. This module
   therefore takes the second choice: `decide_retry`/`apply_decision` are
   never called on this path at all; `record_disposition` is called
   directly with the fixed, already-known `attempt_kind="infra-retry"`.
2. **§1.3 weave-outcome handling assumes states weaver.py has already
   left.** The build spec's §1.3 text describes ".22 transitions GATED ->
   LANDED -> DONE" and "GATED -> REJECTED (via decide_retry/apply_decision)"
   as if the ticket were still sitting at `GATED` when `.22` sees the
   `WeaveResult`. Reading `weaver.py`'s actual `_weave_one`/`_die` source:
   **`weaver.py` itself already performs the `GATED -> {LANDED, REJECTED,
   PARKED}` transition and already emits the terminal `DISPOSITION` event**
   before returning the `WeaveResult` (`transition(..., expected_from=GATED)`
   then `record_disposition(...)` — see `weaver.py`'s `_die`/`_weave_one`).
   By the time `.22` sees `outcome=="landed"`, the ticket is **already at
   `LANDED`** (not `GATED`); by the time `.22` sees a rejecting/infra
   outcome, the ticket is **already at `REJECTED`/`PARKED`** (not `GATED`).
   Literally re-running the build spec's stated transitions would
   immediately raise `IllegalTransition` (`GATED -> REJECTED` on a ticket
   already AT `REJECTED`; `GATED -> LANDED` on a ticket already at
   `LANDED`). This module therefore does ONLY the work weaver.py has not
   already done:
   - `"landed"`: NOTHING state-machine-wise (bead .89 — v0 rests at
     LANDED; the old LANDED->DONE advance broke DAG intake + the spend
     report, which key on state=="landed"). The ticket already sits at
     LANDED via weaver's own GATED->LANDED transition, and weaver's
     `"landed"` disposition already satisfies the "one DISPOSITION per
     weave-ticket outcome"
     contract).
   - `"infra"`: nothing state-machine-wise (ticket is already `PARKED`,
     weaver's own disposition already emitted) — just the infra-trip
     counter bump. This is exactly the guard rail the build spec's own
     prose flags: `PARKED -> POOL` is illegal and must never be risked.
   - `"rejected"`/`"integration-conflict"`/`"integration-regression"`: the
     ticket is already at `REJECTED` (weaver's own transition); `.22` calls
     `decide_retry`/`apply_decision` from `REJECTED` (a legal edge:
     `REJECTED: {POOL, ESCALATED}`) and — per an explicit, documented
     choice — emits its OWN additional `DISPOSITION` event for the
     resulting pool-reentry/escalation decision (distinct fact from
     weaver's own "rejected" disposition: weaver's disposition records the
     *gate verdict*, this module's records the *retry-ladder decision*,
     mirroring the dispatch-side 5-step shape and `record_disposition`'s
     own documented use for "pool-reentry-with-attempt_kind"/"escalated").
3. **`attempt_kind` for DISPATCH/CHECK events.** `attempt_kind` is a
   retry-decision concept computed by `decide_retry` — it is never
   persisted on the ticket row itself. This module reuses the MOST RECENT
   `DISPOSITION` event's `attempt_kind` for a ticket (the disposition that
   decided what this next attempt should look like), defaulting to
   `"initial"` if none exists yet (the ticket's very first dispatch).
4. **Rung/lane selection gap (residual, not fixed here).**
   `dispatch.select_lane` explicitly skips `entry=false` lanes — but
   `statemachine.decide_retry`'s step-up mechanism sets
   `ticket_row["current_rung"]` to exactly those `entry=false` lane names
   (e.g. `"exquisite"`). `prepare_dispatch` (which internally calls
   `select_lane` using only `ticket_row["lane_hint"]`, never
   `current_rung`) can therefore **structurally never** select a
   stepped-up rung's lane — a genuine pre-existing gap between `.16`'s
   ladder and `.21`'s lane selection that `.22` inherits. This module does
   not invent new `dispatch.py`/`select_lane` surface to bridge it (out of
   scope, mirrors the build spec's own instruction not to invent new
   `dispatch.py` surface for the `pre_apply_prior`/bundle-threading gap
   below) — flagged here as a residual for a future bead.
5. **`decision.pre_apply_prior`/`prior_as_reference` are not acted on.**
   `dispatch.py` has no surface at all for threading a prior bundle into a
   fresh dispatch as revision context — every retry dispatch goes through
   the identical `prepare_dispatch(...)` call regardless of these flags.
   Per the build spec's own instruction, this is a residual for a later
   bead's dogfood pass, not something to invent here.
6. **`ticket_row["tier1_checks"]` shape.** The build spec's §1.2 describes
   `tier1_checks` as a `list` of paths (`pytest -x -q {shlex.join(paths)}`);
   `weaver.py`'s own `_all_checks` docstring documents treating the SAME
   JSON column as a flat `dict[str, str]` for its own (different) purpose.
   This module follows the build spec's literal instruction for its own
   Tier-1 check-dict construction (checking `isinstance(..., list)`) —
   consistent with the existing precedent of each module independently,
   locally interpreting this deliberately generic JSON column.
7. **`claim()`'s `dispatch_id` vs `prepare_dispatch`'s self-generated
   `dispatch_id`/`worker_name` (bridging decision, not a spec deviation).**
   `intake.claim()` requires a `dispatch_id` up front (stored as the
   ticket's `lease_dispatch_id`); `dispatch.prepare_dispatch` generates its
   OWN `dispatch_id` (== `worker_name`) internally, with no parameter to
   inject a pre-chosen one. This module claims with a throwaway
   provisional id, then — once `prepare_dispatch` returns the REAL
   `dispatch_id` — reconciles `lease_dispatch_id` to match via a direct
   `store.update_ticket` call, so the DISPATCH event's `dispatch_id`, the
   minted `Capability.dispatch_id`, and the ticket's `lease_dispatch_id`
   all agree (load-bearing for `recover.py`'s orphan-vs-wedge detection,
   which keys off `lease_dispatch_id`).
8. **`checker_image` requirement note (build spec §1.2).** Tier-1's
   synthesized `"ticket-tests"` check assumes the checker image has
   `pytest` available — noted here as an IMAGE REQUIREMENT, not silently
   assumed to always hold.

**Egress (v0 pre-.25).** `egress_setup_fn` defaults to `None` (no egress
wiring yet, per the frozen interface's own comment). When an
`egress_setup_fn` IS injected, this module resolves the dispatch's lane
egress policy (`egress.policy_for_lane`) and calls it with a throwaway
per-dispatch id (independent of the ticket's eventual `dispatch_id`, which
does not exist yet at egress-setup time) — a small, untested-by-the-frozen-
case-list but functionally real wiring path for whichever future bead
actually exercises it live.
"""

from __future__ import annotations

import concurrent.futures
import logging
import secrets
import shlex
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stigmergy import checks, egress
from stigmergy import intake as intake_module
from stigmergy import recover as recover_module
from stigmergy import triggers as triggers_module
from stigmergy.charter import (
    CIRCUIT_BREAKER_THRESHOLD,
    Charter,
    resolve_check_resources,
)
from stigmergy.checks import CheckOutcome, CheckResult
from stigmergy.dispatch import (
    DispatchPlan,
    PriorEvidence,
    generate_worker_name,
    prepare_dispatch,
    select_lane,
)
from stigmergy.drivers import claude_code
from stigmergy.drivers.claude_code import DispatchResult, DispatchStatus
from stigmergy.egress import EgressHandle
from stigmergy.filing import harvest_worker_filings
from stigmergy.intake import LeaseError
from stigmergy.notify import NotificationStore, NtfyNotifier
from stigmergy.records import EventType, RecordPlane, make_event
from stigmergy.recover import ContainerReaper, RecoveryReport
from stigmergy.registry import Registry
from stigmergy.relay import Capability, CapabilityStore, build_redactor, derive_relay_profile
from stigmergy.relay_transport import RelayHandle
from stigmergy.rig import RigStore
from stigmergy.spend import SpendLeash
from stigmergy.statemachine import (
    CLAIMED,
    ESCALATED,
    FAILED,
    IN_FLIGHT,
    PARKED,
    POOL,
    TIER1_GREEN,
    AttemptDecision,
    CriticInfraDecision,
    FailureClass,
    apply_decision,
    decide_critic_infra,
    decide_retry,
    record_disposition,
    transition,
)
from stigmergy.weaver import Weaver, WeaveResult

_logger = logging.getLogger(__name__)

_ZERO_TOKENS: dict[str, int] = {"in": 0, "cached": 0, "out": 0, "reasoning": 0}

# Checks whose outcome counts as "green" for Tier-1 gating purposes (SPEC
# §9/bead .22 build spec §1.2): FLAKY is distinct from PASS for
# observability but is still, overall, a passing check.
_GREEN_CHECK_OUTCOMES = frozenset({CheckOutcome.PASS, CheckOutcome.FLAKY})


class DaemonError(Exception):
    """Raised on a `Daemon` construction/configuration violation (e.g. a
    `rig_paths` mapping missing a required key) — a caller bug, never a
    poll-cycle outcome."""


HALT_EXIT_CODE = 3

# Circuit-breaker constants (SPEC §9, bead .22 build spec §3) — fixed,
# not charter-configurable in v0, mirrors checks.py's fixed-constant
# pattern for what the charter doesn't expose a knob for yet.
_BACKOFF_BASE_SECONDS = 5.0
_BACKOFF_CAP_SECONDS = 300.0
# review-followup (.107/.109): the threshold's single source of truth now
# lives in charter.py (`CIRCUIT_BREAKER_THRESHOLD`) so charter validation
# can enforce `loop.retries.critic_infra < threshold` at charter-load time
# without a circular import (charter.py does not import daemon.py). This
# module-private alias keeps every existing `_CIRCUIT_BREAKER_THRESHOLD`
# reference in this file (and in tests) unchanged.
_CIRCUIT_BREAKER_THRESHOLD = CIRCUIT_BREAKER_THRESHOLD

_REQUIRED_RIG_PATH_KEYS = (
    "context_root",
    "repo_root",
    "clones_root",
    "prompts_dir",
    "records_dir",
)


def _default_secrets_for_capability(capability: Capability) -> frozenset[str]:
    """Safe minimum redaction set (bead .22 build spec §2): always redacts
    the capability token. A future live-relay wiring (`.25`'s concern) can
    inject a richer callable that also includes the real provider key once
    a live `CredentialRelay` exists."""
    return frozenset({capability.token})


def _default_relay_teardown(handle: RelayHandle) -> None:
    """Default per-dispatch relay teardown (bead .25): stop the relay server
    and remove its socket (idempotent). Mirrors `egress.teardown`'s role for
    the egress proxy — a dispatch whose relay is gone never falls back to
    unmetered/uninjected access."""
    handle.stop()


@dataclass(frozen=True)
class PollSummary:
    """Everything one `poll_once()` call did, for tests to assert against
    and for `run_forever()` to act on (bead .22 build spec §4).

    **Concurrency note (ticket .TBD):** `dispatched_tickets` is now a set
    of ticket ids (up to `workers` in one poll cycle). The legacy
    `dispatched_ticket` field is kept for backward compatibility but is
    DEPRECATED — use `dispatched_tickets` instead. Similarly,
    `dispatch_status` is DEPRECATED (it only made sense when at most one
    dispatch happened per poll).
    """

    dispatched_tickets: frozenset[str]
    dispatch_statuses: frozenset[str]
    weave_ran: bool
    weave_results: tuple[str, ...]
    infra_trip: bool
    consecutive_infra_trips: int
    should_halt: bool
    backoff_seconds: float

    # Backward compatibility aliases (DEPRECATED).
    @property
    def dispatched_ticket(self) -> str | None:
        """DEPRECATED: returns the first dispatched ticket (for tests still
        expecting the old single-dispatch-per-poll contract), or None."""
        return next(iter(self.dispatched_tickets), None) if self.dispatched_tickets else None

    @property
    def dispatch_status(self) -> str | None:
        """DEPRECATED: returns the first dispatch status, or None."""
        return next(iter(self.dispatch_statuses), None) if self.dispatch_statuses else None


class Daemon:
    """Wires every station into one poll loop (bead .22 build spec §4).

    Beyond the frozen constructor kwargs, this implementation adds a
    handful of additional DI seams the frozen interface's own text implies
    but doesn't name explicitly (`.22`'s own choice to complete): a
    `container_reaper` (recover.py's `ContainerReaper` Protocol — real
    podman reaping is explicitly out of this module's scope, "that's
    .13's/.14's job through their own injected executors" per the build
    spec's own §0), a `steering_of` callable (the human-triage steering
    dict a real approval tool computed — genuinely out of `.22`'s scope to
    invent; injected exactly like every other externally-owned judgment
    input in this codebase), and small recovery/lease/placement knobs
    (`git_repo_paths`, `min_disk_bytes`, `disk_path`, `owner`,
    `relay_base_url`) with conservative defaults documented inline.
    """

    def __init__(
        self,
        *,
        store: RigStore,
        record_plane: RecordPlane,
        notification_store: NotificationStore,
        notifier: NtfyNotifier,
        spend_leash: SpendLeash,
        charter: Charter,
        registry: Registry,
        rig_paths: dict[str, Path],
        capability_store: CapabilityStore,
        checker_image: str,
        weaver: Weaver,
        container_reaper: ContainerReaper,
        steering_of: Callable[[str], dict[str, Any]],
        spawn_fn: Callable[..., DispatchResult] = claude_code.spawn,
        run_checks_fn: Callable[..., list[CheckResult]] = checks.run_checks,
        egress_setup_fn: Callable[..., EgressHandle] | None = None,
        egress_teardown_fn: Callable[[EgressHandle], None] = egress.teardown,
        relay_setup_fn: Callable[..., RelayHandle] | None = None,
        relay_profile_cell: dict | None = None,
        relay_teardown_fn: Callable[[RelayHandle], None] = _default_relay_teardown,
        secrets_for_capability: Callable[[Capability], frozenset[str]] | None = None,
        now_fn: Callable[[], float] = time.time,
        git_repo_paths: list[str | Path] | None = None,
        min_disk_bytes: int = 100 * 1024 * 1024,
        disk_path: str | Path | None = None,
        owner: str = "daemon",
        relay_base_url: str = "http://127.0.0.1:0/stigmergy-relay-placeholder",
        image: str | None = None,
    ) -> None:
        missing = [k for k in _REQUIRED_RIG_PATH_KEYS if k not in rig_paths]
        if missing:
            raise DaemonError(f"rig_paths missing required key(s): {missing}")

        self._store = store
        self._record_plane = record_plane
        self._notification_store = notification_store
        self._notifier = notifier
        self._spend_leash = spend_leash
        self._charter = charter
        self._registry = registry
        self._rig_paths = rig_paths
        self._capability_store = capability_store
        self._checker_image = checker_image
        self._weaver = weaver
        self._container_reaper = container_reaper
        self._steering_of = steering_of
        self._spawn_fn = spawn_fn
        self._run_checks_fn = run_checks_fn
        self._egress_setup_fn = egress_setup_fn
        self._egress_teardown_fn = egress_teardown_fn
        self._relay_setup_fn = relay_setup_fn
        # bead .147 §1F: the per-dispatch RELAY PROFILE side-channel. The
        # cli (`_build_daemon`) creates this dict and passes it here; the
        # dispatch path writes the lane's derived `RelayProfile` into it
        # keyed by the REAL dispatch id, immediately before the
        # `_setup_relay` call. The cli's `relay_setup_fn` closure (whose
        # signature stays `fn(dispatch_id, runtime_dir)` — ten+ existing
        # test stubs pin those 2 args) READS `cell.get(dispatch_id)`,
        # falling back to the DEFAULT anthropic profile when absent.
        # Default `None` -> the daemon skips derivation and today's
        # behaviour holds (existing stubs + the .25 wiring test keep
        # working untouched).
        self._relay_profile_cell: dict | None = relay_profile_cell
        self._relay_teardown_fn = relay_teardown_fn
        self._secrets_for_capability = (
            secrets_for_capability
            if secrets_for_capability is not None
            else _default_secrets_for_capability
        )
        self._now_fn = now_fn
        self._git_repo_paths = (
            git_repo_paths if git_repo_paths is not None else [str(rig_paths["repo_root"])]
        )
        self._min_disk_bytes = min_disk_bytes
        self._disk_path = disk_path if disk_path is not None else str(rig_paths["repo_root"])
        self._owner = owner
        self._relay_base_url = relay_base_url

        # bead .79: the effective worker image — the per-rig built image
        # (base + provision deps) when the caller resolved one, else the
        # charter's base `[rig].image` (backward-compatible default).
        self._image = image if image is not None else charter.raw["rig"]["image"]
        self._rig_name = charter.raw.get("rig", {}).get("name")
        self._lease_ttl_seconds = charter.raw["loop"]["timers"]["lease_ttl_seconds"]
        self._poll_seconds = charter.raw["loop"]["timers"]["poll_seconds"]

        # Infra-trip counter protection (ticket .TBD): RLock allows the same
        # thread to acquire it multiple times (re-entrant). Protects every
        # increment (_bump_infra_trip_counter), reset (_reset_infra_trip_counter),
        # and the read that decides should_halt against lost updates from
        # concurrent dispatch threads. The lock is held only for the brief
        # read-modify-write; dispatch threads release it immediately after,
        # so the weaver's own calls to the bump/reset methods (on the main
        # thread) never contend.
        self._infra_trip_lock = threading.RLock()
        self._consecutive_infra_trips = 0

        self._prior_parked: dict[str, float] = {}

        # Bounded thread pool for concurrent dispatch (ticket .TBD): up to
        # `workers` in-flight dispatches at once. Each dispatch's body
        # (prepare, spawn, Tier-1 checks, state transition, event emission,
        # teardown) runs on a pool thread; the claim step (atomic) and
        # running on the pool are the only changes from the single-dispatch
        # world. The weaver stays on the main thread.
        workers = charter.raw.get("loop", {}).get("concurrency", {}).get("workers", 1)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)

    # -- recover_on_start ----------------------------------------------------

    def recover_on_start(self) -> RecoveryReport:
        """Calls `recover.recover(...)` exactly once (bead .22 build spec
        §4). `recover.RecoveryError` (disk headroom) propagates uncaught —
        fatal, per the session brief.

        Also fails loudly (bead .90) if the charter's `dispatch_base` branch
        is absent from the rig repo: workers dispatch FROM and the weaver
        lands ONTO it, and without it `_build_execution` resolves `base_oid`
        to `None` and every dispatch fails obscurely downstream. Scaffold
        creates it (`rig._ensure_dispatch_base_branch`); this is the
        belt-and-suspenders for a hand-crafted rig or a deleted branch."""
        dispatch_base = self._charter.raw["tiers"]["dispatch_base"]
        if self._git_rev_parse(Path(self._rig_paths["repo_root"]), dispatch_base) is None:
            raise DaemonError(
                f"dispatch_base branch {dispatch_base!r} does not exist in the rig repo "
                f"{self._rig_paths['repo_root']!r} — cannot dispatch or land. Scaffold a fresh "
                "rig (which creates it) or create the branch before starting the daemon."
            )
        now = self._now_fn()
        return recover_module.recover(
            self._store,
            self._record_plane,
            reaper=self._container_reaper,
            weave_resolver=self._weaver,
            git_repo_paths=self._git_repo_paths,
            min_disk_bytes=self._min_disk_bytes,
            disk_path=self._disk_path,
            now=now,
        )

    # -- poll_once ------------------------------------------------------------

    def poll_once(self) -> PollSummary:
        """One full poll cycle (bead .22 build spec §4): heartbeat,
        spend-gated claim+dispatch+checks (up to `workers` tickets, concurrently
        via a bounded thread pool), weave-trigger evaluation + weave if
        triggered (main thread, not pooled), notification delivery.

        Never sleeps. See the module docstring + build spec §1-§3 for the
        exact decision logic. The weaver stays single-threaded on the main
        thread; only dispatch is parallelized.
        """
        now = self._now_fn()
        self._store.set_meta("daemon_heartbeat_at", str(now))

        dispatched_tickets: set[str] = set()
        dispatch_statuses: set[str] = set()
        any_dispatch_infra_trip = False

        if self._spend_leash.can_dispatch():
            # Claim and dispatch up to `workers` tickets concurrently.
            dispatched_tickets, dispatch_statuses, any_dispatch_infra_trip = (
                self._dispatch_bounded_pool(now)
            )

        weave_ran, weave_result_tickets, weave_infra_trip = self._maybe_weave(now)

        infra_trip = any_dispatch_infra_trip or weave_infra_trip

        # Read the infra-trip counter with the lock held, to protect against
        # concurrent updates from dispatch threads.
        with self._infra_trip_lock:
            consecutive = self._consecutive_infra_trips
        should_halt = consecutive >= _CIRCUIT_BREAKER_THRESHOLD
        backoff_seconds = self._compute_backoff(consecutive) if infra_trip else 0.0

        if should_halt:
            # bead .109: log the halt cause to the daemon LOG (not just the
            # persisted ntfy intent) — `sys.exit` in `run_forever` is
            # otherwise silent, and this was the actual root cause of "the
            # daemon exited with no logged reason" (see build-107-109-spec.md
            # §1: the already-existing global circuit breaker tripping).
            _logger.error(
                "stigmergy daemon halting: circuit breaker tripped — "
                "%d consecutive infra trips >= threshold %d",
                consecutive,
                _CIRCUIT_BREAKER_THRESHOLD,
            )
            self._persist_halt_notification(now)

        # Notification delivery: every poll attempts to flush pending
        # intents (AC12 "persisted with retry") — this single call also
        # satisfies §3's "attempt one NtfyNotifier.deliver_pending call"
        # for a halt this cycle, since the halt intent was just persisted
        # above, before this line runs.
        self._notifier.deliver_pending(self._notification_store, now=now)

        return PollSummary(
            dispatched_tickets=frozenset(dispatched_tickets),
            dispatch_statuses=frozenset(dispatch_statuses),
            weave_ran=weave_ran,
            weave_results=weave_result_tickets,
            infra_trip=infra_trip,
            consecutive_infra_trips=consecutive,
            should_halt=should_halt,
            backoff_seconds=backoff_seconds,
        )

    # -- run_forever ------------------------------------------------------------

    def run_forever(self, *, sleep_fn: Callable[[float], None] = time.sleep) -> None:
        """`recover_on_start()` once, then `poll_once()` + `sleep_fn(...)`
        forever. Calls `sys.exit(HALT_EXIT_CODE)` the first time a
        `PollSummary.should_halt` is `True` — the only place this module
        ever exits the process."""
        self.recover_on_start()
        while True:
            summary = self.poll_once()
            if summary.should_halt:
                sys.exit(HALT_EXIT_CODE)
            sleep_seconds = (
                summary.backoff_seconds if summary.backoff_seconds > 0 else self._poll_seconds
            )
            sleep_fn(sleep_seconds)

    # -- concurrent dispatch (via bounded thread pool) --

    def _dispatch_bounded_pool(
        self, now: float
    ) -> tuple[set[str], set[str], bool]:
        """Claim and dispatch up to `workers` tickets concurrently via the
        bounded thread pool. Returns (dispatched_tickets, dispatch_statuses,
        any_infra_trip) where:
        - dispatched_tickets: the set of ticket ids that were dispatched
        - dispatch_statuses: the set of status strings from dispatch results
        - any_infra_trip: True iff any dispatch was an infra trip

        The claim step is atomic (via store.atomic_claim_ticket); the dispatch
        cycle runs on a pool thread. Each dispatch's failure handling
        (infra-trip counter bumps, state transitions, etc.) is per-dispatch,
        with the infra counter bumps protected by _infra_trip_lock.

        Only up to `workers` tasks are submitted to the pool in one poll cycle,
        matching the charter's `loop.concurrency.workers` limit.

        If any dispatch thread raises an exception, it is re-raised from here
        (propagates to poll_once, which may propagate it further). This ensures
        test cases that expect exceptions from spawn_fn to propagate (e.g.,
        case 16, case 29b) still see the exception at the poll_once level.
        """
        eligible_ids = intake_module.eligible(
            self._store, now=now, steering_of=self._steering_of
        )
        if not eligible_ids:
            return set(), set(), False

        dispatched: set[str] = set()
        statuses: set[str] = set()
        any_infra = False
        first_exception: Exception | None = None

        # Get the workers limit from the charter (same value used for max_workers
        # in the ThreadPoolExecutor, so we never submit more than max_workers tasks).
        workers = self._charter.raw.get("loop", {}).get("concurrency", {}).get("workers", 1)

        # Submit up to `workers` tasks to the pool: claim_and_dispatch each ticket.
        futures_to_ticket: dict[concurrent.futures.Future, str] = {}
        for ticket_id in eligible_ids[:workers]:
            future = self._executor.submit(self._claim_and_dispatch_one, ticket_id, now)
            futures_to_ticket[future] = ticket_id

        # Collect results from completed tasks (they may not finish in order).
        for future in concurrent.futures.as_completed(futures_to_ticket):
            ticket_id = futures_to_ticket[future]
            try:
                status, was_infra = future.result()
                if status is not None:
                    dispatched.add(ticket_id)
                    statuses.add(status)
                    any_infra = any_infra or was_infra
            except Exception as e:
                # A dispatch thread raised an exception. Save the first one to
                # re-raise later, but continue collecting results from other
                # dispatch threads so they can complete cleanly (e.g., running
                # their finally blocks for cleanup).
                if first_exception is None:
                    first_exception = e
                _logger.warning(
                    "exception in dispatch thread for ticket %s",
                    ticket_id,
                    exc_info=True,
                )

        # If any dispatch thread raised, re-raise the first exception now.
        if first_exception is not None:
            raise first_exception

        return dispatched, statuses, any_infra

    def _claim_and_dispatch_one(self, ticket_id: str, now: float) -> tuple[str | None, bool]:
        """Claim and dispatch one ticket; returns (status, was_infra_trip).
        If the claim fails (LeaseError), returns (None, False). If the dispatch
        cycle raises an exception, it propagates to the caller (which will
        handle it and re-raise from _dispatch_bounded_pool).

        This method runs on a thread pool thread."""
        claimed = self._claim_and_prepare(now, ticket_id)
        if claimed is None:
            return None, False
        ticket_id_claimed, plan, egress_handle, relay_handle = claimed
        status, infra_trip = self._run_dispatch_cycle(
            ticket_id_claimed, plan, egress_handle, relay_handle
        )
        return status, infra_trip

    # -- claim + prepare ------------------------------------------------------

    def _claim_and_prepare(
        self, now: float, ticket_id_override: str | None = None
    ) -> tuple[str, DispatchPlan, EgressHandle | None, RelayHandle | None] | None:
        """Claim and prepare the dispatch for one ticket. If
        ``ticket_id_override`` is provided (by _claim_and_dispatch_one on a
        pool thread), claim that specific ticket. Otherwise, find one eligible
        ticket from the eligible list. Returns `None` if nothing is eligible,
        or if a race makes the claim itself fail (never a bug — `LeaseError`
        is an ordinary "try again next poll" signal)."""
        if ticket_id_override is not None:
            ticket_id = ticket_id_override
        else:
            eligible_ids = intake_module.eligible(
                self._store, now=now, steering_of=self._steering_of
            )
            if not eligible_ids:
                return None
            ticket_id = eligible_ids[0]
        ticket_row = self._store.get_ticket(ticket_id)
        steering = self._steering_of(ticket_id)
        # bead .25 audit (codex HIGH): resolve the OPERATIONAL lane with the
        # ticket's current rung — the SAME resolution prepare_dispatch uses
        # (dispatch.py select_lane(..., rung=...)). Without rung, a stepped-up
        # retry would get the egress policy + execution snapshot of the ENTRY
        # lane, not the rung it actually runs on — an egress-policy mismatch
        # (masked in v0 only because all rungs share egress groups). Now the
        # egress policy, the execution metadata, and plan.lane all agree.
        lane = select_lane(self._charter, ticket_row, rung=ticket_row.get("current_rung"))
        execution = self._build_execution(lane)

        # Bridging decision (module docstring point 7): claim with a
        # throwaway provisional id, reconcile to the real one below.
        provisional_dispatch_id = f"claim-{secrets.token_hex(8)}"
        try:
            intake_module.claim(
                self._store,
                ticket_id,
                owner=self._owner,
                dispatch_id=provisional_dispatch_id,
                ttl_seconds=self._lease_ttl_seconds,
                now=now,
                steering=steering,
                execution=execution,
            )
        except LeaseError:
            return None

        transition(self._store, ticket_id, IN_FLIGHT, expected_from=CLAIMED)

        # Generate the REAL dispatch id BEFORE setting up relay/egress (ticket requirement).
        # This is the same id that prepare_dispatch would mint, minted early so both
        # relay and egress can log it from the very first request.
        real_dispatch_id = generate_worker_name(self._store, "worker", lane.model, lane.prompt)

        egress_handle = self._setup_egress(ticket_id, lane, now, real_dispatch_id)
        egress_socket = egress_handle.socket_path if egress_handle is not None else None

        # bead .25: the credential relay mirrors egress — set up BEFORE
        # prepare_dispatch with the real dispatch id, read the served socket path
        # back from the handle, thread it into the dispatch so spawn mounts
        # /run/relay.sock. The relay authorizes by capability TOKEN against the
        # shared CapabilityStore (not by dispatch_id), so it is safe to stand
        # up before the capability is minted inside prepare_dispatch — the
        # worker only issues a request once it is running, well after mint.
        relay_handle = self._setup_relay(ticket_id, now, real_dispatch_id, lane=lane)
        relay_socket = relay_handle.socket_path if relay_handle is not None else None

        fresh_row = self._store.get_ticket(ticket_id)
        plan = prepare_dispatch(
            charter=self._charter,
            ticket_row=fresh_row,
            store=self._store,
            capability_store=self._capability_store,
            rig_repo=Path(self._rig_paths["repo_root"]),
            context_root=Path(self._rig_paths["context_root"]),
            clones_root=Path(self._rig_paths["clones_root"]),
            prompts_dir=Path(self._rig_paths["prompts_dir"]),
            relay_base_url=self._relay_base_url,
            image=self._image,
            egress_socket=egress_socket,
            relay_socket=relay_socket,
            prior_evidence=self._gather_prior_evidence(ticket_id),
            dispatch_id=real_dispatch_id,
        )
        # Reconcile the ticket's lease_dispatch_id to the REAL dispatch
        # identity now that prepare_dispatch has generated it — see module
        # docstring point 7.
        self._store.update_ticket(ticket_id, lease_dispatch_id=plan.dispatch_id)
        return ticket_id, plan, egress_handle, relay_handle

    def _build_execution(self, lane: Any) -> dict[str, Any]:
        """Build the `execution` snapshot fields (SPEC §4: mechanism
        inputs — base OID, resolved model+version, image digest, egress
        policy — that may legitimately drift without re-approval)."""
        dispatch_base = self._charter.raw["tiers"]["dispatch_base"]
        base_oid = self._git_rev_parse(Path(self._rig_paths["repo_root"]), dispatch_base)
        model_entry = self._registry.resolve(lane.model)
        return {
            "base_oid": base_oid,
            "model": lane.model,
            "model_version": model_entry.version,
            "image_digest": self._image,
            "egress_policy": list(lane.egress),
        }

    @staticmethod
    def _git_rev_parse(repo: Path, ref: str) -> str | None:
        result = subprocess.run(  # noqa: S603
            ["git", "-C", str(repo), "rev-parse", ref],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    def _setup_egress(
        self, ticket_id: str, lane: Any, now: float, dispatch_id: str
    ) -> EgressHandle | None:
        """`None` unless `egress_setup_fn` is injected (wired for real by
        `cli._build_daemon` at `.25`)."""
        if self._egress_setup_fn is None:
            return None
        policy = egress.policy_for_lane(self._charter.raw, lane.name)
        runtime_dir = Path(self._rig_paths["clones_root"]) / "_egress_runtime"
        return self._egress_setup_fn(dispatch_id, policy, runtime_dir)

    def _setup_relay(
        self, ticket_id: str, now: float, dispatch_id: str, *, lane: Any = None
    ) -> RelayHandle | None:
        """bead .25: `None` unless `relay_setup_fn` is injected (wired for real
        by `cli._build_daemon`). Mirrors `_setup_egress`: the real dispatch id + a
        per-rig runtime dir; the injected fn constructs a `CredentialRelay`
        (sharing this daemon's `CapabilityStore`) and `start_relay`s it,
        returning a `RelayHandle` whose `socket_path` the caller threads into
        the dispatch. The relay authorizes by capability token, not by lane —
        BUT bead .147 §1F threads the lane's RELAY PROFILE through the
        side-channel: when `relay_profile_cell` is wired and `lane` is given,
        the profile is derived from the lane's model NOW (immediately before
        the setup call — the one point where lane + registry + charter are
        all in scope pre-`prepare_dispatch`) and written into the cell under
        the REAL dispatch id. The `relay_setup_fn` signature stays
        `(dispatch_id, runtime_dir)` (the 2-arg contract ten+ test stubs
        pin); the closure reads the cell and falls back to the default
        anthropic profile when absent. Derivation failure is a loud dispatch
        failure (unknown wire / missing openai base URL — a caller bug that
        must never silently fall back to a DIFFERENT upstream)."""
        if self._relay_setup_fn is None:
            return None
        if self._relay_profile_cell is not None and lane is not None:
            # Derivation failure is a loud dispatch failure (unknown wire /
            # missing openai base URL — a caller bug that must never silently
            # fall back to a DIFFERENT upstream).
            profile = derive_relay_profile(
                lane_model=lane.model,
                registry=self._registry,
                charter=self._charter,
            )
            self._relay_profile_cell[dispatch_id] = profile
        runtime_dir = Path(self._rig_paths["clones_root"]) / "_relay_runtime"
        return self._relay_setup_fn(dispatch_id, runtime_dir)

    # -- dispatch cycle ---------------------------------------------------------

    def _run_dispatch_cycle(
        self,
        ticket_id: str,
        plan: DispatchPlan,
        egress_handle: EgressHandle | None,
        relay_handle: RelayHandle | None = None,
    ) -> tuple[str, bool]:
        """Spawn, emit the DISPATCH event, then run the §1.1/§1.2 decision
        tree. `capability_store.revoke(dispatch_id)`, relay teardown, and
        egress teardown ALWAYS run in a `finally` around this whole block —
        covers every exit path, including an unexpected exception from
        `spawn_fn` (case 16: the exception propagates after `finally` runs;
        revoke still happened). bead .25 (case 29): teardown revokes the
        capability FIRST (the load-bearing security act — a leaked/replayed
        token is dead instantly), then stops the relay, then the egress
        proxy; each teardown is independently guarded so a failure in one
        never leaks the other's OS resources."""
        infra_trip = False
        dispatch_status_value = "none"
        try:
            t_start = self._now_fn()
            result = self._spawn_fn(
                plan.task_pack,
                plan.work_clone,
                plan.model_cfg,
                plan.capability,
                plan.budgets,
            )
            t_end = self._now_fn()
            dispatch_status_value = result.status.value

            ticket_row = self._store.get_ticket(ticket_id)
            attempt = (ticket_row.get("attempts_used") or 0) + 1
            attempt_kind = self._last_attempt_kind(ticket_id)
            rung = ticket_row.get("current_rung") or plan.lane.name
            model_entry = self._registry.resolve(plan.lane.model)

            ctx = {
                "rig": self._rig_name,
                "ticket": ticket_id,
                "dispatch_id": plan.dispatch_id,
                "attempt": attempt,
                "rung": rung,
                "worker": plan.worker_name,
                "charter_hash": self._charter.resolved_hash,
                "approval_hash": ticket_row.get("approval_hash"),
                "image_digest": plan.model_cfg.image,
                "model": plan.lane.model,
                "model_version": model_entry.version,
                "price_table_version": self._registry.version_hash,
            }

            computed_usd = self._spend_leash.record_dispatch(
                plan.lane.model, result.usage, ticket=ticket_id
            )

            secret_set = self._secrets_for_capability(plan.capability)
            redactor = build_redactor(secret_set)
            transcript_ref = self._record_plane.seal_transcript(
                result.transcript, redactor=redactor, must_not_contain=secret_set
            )

            dispatch_event = make_event(
                EventType.DISPATCH,
                **ctx,
                attempt_kind=attempt_kind,
                tokens=result.usage,
                computed_usd=computed_usd,
                wall_time_seconds=max(0.0, t_end - t_start),
                prompt_artifact_hash=plan.prompt_artifact_hash,
                transcript_ref=transcript_ref,
                status=result.status.value,
                ceiling_trip=result.ceiling_trip,
                bundle_ref=result.bundle_ref,
                detail=result.detail,
            )
            self._record_plane.append(dispatch_event)

            harvest_result = None
            try:
                limits = self._charter.raw["loop"]["dispatch_limits"]
                harvest_result = harvest_worker_filings(
                    plan.work_clone,
                    store=self._store,
                    record_plane=self._record_plane,
                    ctx=ctx,
                    attempt_kind=attempt_kind,
                    max_filings=limits["filed_tickets"],
                    max_bytes=limits["filed_ticket_bytes"],
                )
            except Exception:
                # Defensive belt only: harvest_worker_filings is contracted
                # to never raise (SPEC amendment A). A harvest failure must
                # never be able to break the dispatch outcome path.
                _logger.warning(
                    "worker filing harvest failed for dispatch %s (ticket %s)",
                    plan.dispatch_id,
                    ticket_id,
                    exc_info=True,
                )

            # The worker has explicitly declared the ticket impossible-as-scoped;
            # honor the declaration and route to the human triage floor REGARDLESS
            # of the Tier-1 result. The rare green-Tier-1-plus-escape case also
            # routes to triage — safe; the human resolves the contradiction. Cost
            # of a wrong escape = one triage glance; the code01 recognition bar +
            # human gate are the backstops (per the design's functional-first ruling)
            # — build NO anti-gaming adjudicator.
            if harvest_result is not None and harvest_result.escape is not None:
                self._dispatch_spec_failure(ticket_id, ctx, harvest_result.escape, now=t_end)
                infra_trip = False
            else:
                infra_trip = self._handle_dispatch_outcome(ticket_id, plan, result, ctx, now=t_end)
        finally:
            self._capability_store.revoke(plan.dispatch_id)
            if relay_handle is not None:
                try:
                    self._relay_teardown_fn(relay_handle)
                except Exception:  # noqa: BLE001 - a relay-stop failure must not skip egress teardown
                    _logger.warning(
                        "relay teardown failed for dispatch %s (ticket %s)",
                        plan.dispatch_id,
                        ticket_id,
                        exc_info=True,
                    )
            if egress_handle is not None:
                try:
                    self._egress_teardown_fn(egress_handle)
                except Exception:  # noqa: BLE001 - an egress-stop failure must not mask the outcome
                    _logger.warning(
                        "egress teardown failed for dispatch %s (ticket %s)",
                        plan.dispatch_id,
                        ticket_id,
                        exc_info=True,
                    )

        return dispatch_status_value, infra_trip

    def _handle_dispatch_outcome(
        self,
        ticket_id: str,
        plan: DispatchPlan,
        result: DispatchResult,
        ctx: dict[str, Any],
        *,
        now: float,
    ) -> bool:
        """The §1.1/§1.2 decision tree. Returns `True` iff this was an
        infra trip (dispatch-side)."""
        status = result.status

        if status is DispatchStatus.INFRA:
            # §1.1 INFRA fast path — see module docstring deviation 1: we
            # never call decide_retry/apply_decision here (would raise
            # IllegalTransition on POOL->POOL); attempt_kind is already the
            # fixed "infra-retry" value.
            transition(self._store, ticket_id, POOL, expected_from=IN_FLIGHT)
            record_disposition(
                self._record_plane, ctx=ctx, disposition="infra-retry", attempt_kind="infra-retry"
            )
            self._bump_infra_trip_counter()
            return True

        if status is DispatchStatus.WEDGED:
            self._dispatch_failure(ticket_id, FailureClass.WEDGE, ctx, now=now)
            self._reset_infra_trip_counter()
            return False

        if status is DispatchStatus.FAILED:
            # ceiling_trip set -> DEGENERATE; otherwise an ordinary
            # TIER1_FAIL (build-reconciliation note, module docstring
            # deviation 5: pre_apply_prior is never acted on regardless of
            # result.bundle_ref — dispatch.py has no pre-apply surface).
            failure_class = (
                FailureClass.DEGENERATE
                if result.ceiling_trip is not None
                else FailureClass.TIER1_FAIL
            )
            self._dispatch_failure(ticket_id, failure_class, ctx, now=now)
            self._reset_infra_trip_counter()
            return False

        # DONE -> Tier-1 checks (§1.2) decide the outcome.
        self._run_tier1_checks(ticket_id, plan, ctx, bundle_ref=result.bundle_ref, now=now)
        return False

    def _dispatch_failure(
        self, ticket_id: str, failure_class: FailureClass, ctx: dict[str, Any], *, now: float
    ) -> None:
        transition(self._store, ticket_id, FAILED, expected_from=IN_FLIGHT)
        self._apply_retry_decision(ticket_id, failure_class, ctx, now=now)

    def _dispatch_spec_failure(
        self, ticket_id: str, ctx: dict[str, Any], escape: dict[str, Any], *, now: float
    ) -> None:
        """Route a worker escape (blocks_ticket:true) to spec-failure escalation."""
        transition(self._store, ticket_id, FAILED, expected_from=IN_FLIGHT)
        self._apply_retry_decision(
            ticket_id, FailureClass.SPEC_FAILURE, ctx, now=now, spec_failure=escape
        )

    # -- Tier-1 checks (§1.2) ---------------------------------------------------

    def _run_tier1_checks(
        self,
        ticket_id: str,
        plan: DispatchPlan,
        ctx: dict[str, Any],
        *,
        bundle_ref: str | None,
        now: float,
    ) -> None:
        """Run Tier-1 checks directly against the dispatch's own
        `work_clone` (bead .22 build spec §1.2) and decide TIER1_GREEN vs
        TIER1_FAIL. {PASS, FLAKY} count as green; {FAIL, ERROR} as failed
        (FLAKY-counts-as-green is a deliberately frozen choice, case 9)."""
        charter = self._charter
        check_dict: dict[str, str] = {}
        for name in charter.raw["gates"]["attempt"]:
            check_dict[name] = charter.raw["checks"][name]["cmd"]

        ticket_row = self._store.get_ticket(ticket_id)
        # Module docstring deviation 6: tier1_checks is treated as a LIST
        # of paths here (per the build spec's literal §1.2 instruction),
        # a DIFFERENT interpretation of the same generic JSON column than
        # weaver.py's own dict-shaped convention — each module is free to
        # locally interpret this column (weaver.py's own precedent).
        tier1_checks = ticket_row.get("tier1_checks")
        if isinstance(tier1_checks, list) and tier1_checks:
            # Module docstring deviation 8: assumes the checker image has
            # pytest available — an IMAGE REQUIREMENT, not silently assumed.
            check_dict["ticket-tests"] = f"pytest -x -q {shlex.join(tier1_checks)}"

        flake_reruns = charter.raw["loop"]["retries"]["flake_reruns"]
        # bead .91: per-check charter-configurable resource bounds. The
        # synthesized "ticket-tests" name has no [checks.<name>] entry, so
        # it resolves to DEFAULT_CHECK_RESOURCES + [loop.check_resources]
        # overlay only — intentional, since a real pytest run needs more
        # than the checker's conservative 60s/256m/1cpu defaults.
        resources_map = {name: resolve_check_resources(charter, name) for name in check_dict}
        check_results = self._run_checks_fn(
            check_dict,
            work_tree=plan.work_clone,
            image=self._checker_image,
            flake_reruns=flake_reruns,
            resources=resources_map,
        )

        for cr in check_results:
            event = make_event(
                EventType.CHECK,
                **ctx,
                attempt_kind=self._last_attempt_kind(ticket_id),
                tokens=dict(_ZERO_TOKENS),
                computed_usd=0.0,
                wall_time_seconds=cr.wall_time_seconds,
                name=cr.name,
                outcome=cr.outcome.value,
                runs=list(cr.runs),
                output=cr.output,
            )
            self._record_plane.append(event)

        all_green = all(cr.outcome in _GREEN_CHECK_OUTCOMES for cr in check_results)

        if all_green:
            # bead .94: persist the worker's git bundle (created by the driver,
            # DispatchResult.bundle_ref) as the ticket's work_product BEFORE
            # parking — the weaver applies it at the staging gate. Without this
            # the ticket parks with work_product=None and the weave fails
            # integration-conflict (weaver._apply_bundle has nothing to apply).
            self._store.update_ticket(ticket_id, work_product=bundle_ref)
            transition(self._store, ticket_id, TIER1_GREEN, expected_from=IN_FLIGHT)
            transition(self._store, ticket_id, PARKED, expected_from=TIER1_GREEN)
            record_disposition(
                self._record_plane,
                ctx=ctx,
                disposition="parked",
                attempt_kind=self._last_attempt_kind(ticket_id),
            )
            self._reset_infra_trip_counter()
        else:
            transition(self._store, ticket_id, FAILED, expected_from=IN_FLIGHT)
            self._apply_retry_decision(ticket_id, FailureClass.TIER1_FAIL, ctx, now=now)
            self._reset_infra_trip_counter()

    # -- shared retry-ladder application -----------------------------------------

    def _apply_retry_decision(
        self,
        ticket_id: str,
        failure_class: FailureClass,
        ctx: dict[str, Any],
        *,
        now: float,
        spec_failure: dict[str, Any] | None = None,
    ) -> AttemptDecision:
        """`decide_retry` -> `apply_decision` -> disposition -> escalation
        notification (shared tail for both the dispatch-side FAILED/GATED
        retry paths, §1.1/§1.2, and the weave-side rejected/conflict/
        regression path, §1.3). The caller is responsible for any
        transition INTO `FAILED`/`REJECTED` that must happen first — this
        helper assumes the ticket is already sitting in `FAILED` or
        `REJECTED`, per `apply_decision`'s own precondition."""
        ticket_row = self._store.get_ticket(ticket_id)
        ladder = self._charter.raw["stepup"]["ladder"]
        attempts_per_rung = self._charter.raw["loop"]["retries"]["attempts_per_rung"]
        integration_failures_cap = self._charter.raw["loop"]["retries"]["integration_failures"]

        decision = decide_retry(
            ticket_row,
            failure_class,
            ladder=ladder,
            attempts_per_rung=attempts_per_rung,
            integration_failures_cap=integration_failures_cap,
        )
        apply_decision(self._store, ticket_id, decision)

        disposition = "escalated" if decision.next_state == ESCALATED else "pool-reentry"
        record_disposition(
            self._record_plane,
            ctx=ctx,
            disposition=disposition,
            attempt_kind=decision.attempt_kind,
            reason=decision.escalation_reason,
            spec_failure=spec_failure,
        )

        if decision.next_state == ESCALATED:
            self._persist_escalation_notification(ticket_id, decision, now=now)

        return decision

    def _last_attempt_kind(self, ticket_id: str) -> str:
        """Module docstring deviation 3: reuse the most recent DISPOSITION
        event's `attempt_kind` for this ticket (the disposition that
        decided what THIS next attempt should look like); `"initial"` for
        a ticket's very first dispatch."""
        for ev in reversed(self._record_plane.read_events()):
            is_disposition = ev.get("event_type") == EventType.DISPOSITION.value
            if is_disposition and ev.get("ticket") == ticket_id:
                kind = ev.get("attempt_kind")
                if isinstance(kind, str) and kind:
                    return kind
        return "initial"

    def _gather_prior_evidence(self, ticket_id: str) -> PriorEvidence | None:
        """Bead .29: reconstruct the ticket's most-recent prior-attempt
        failure evidence from the record plane, to thread into a RETRY
        dispatch's task pack (SPEC §9 "each [retry] carries the failure
        evidence"). `None` for a first (initial) dispatch. Evidence is
        selected by the retry kind (`_last_attempt_kind`), mirroring the
        `AttemptDecision` that produced it: a tier1/integration retry carries
        the prior failing check output; a critic-revision carries the
        critic's reason; any retry carries the most recent non-empty
        disposition reason as a failure note. This is the informational
        channel only — physically re-applying the prior work product to the
        fresh clone (`pre_apply_prior`) is deferred (see `PriorEvidence`)."""
        attempt_kind = self._last_attempt_kind(ticket_id)
        if attempt_kind == "initial":
            return None
        wants_check = attempt_kind in ("tier1-repair", "integration-reconcile")
        wants_critic = attempt_kind == "critic-revision"
        failure_note: str | None = None
        critic_reason: str | None = None
        check_output: str | None = None
        _failing = (CheckOutcome.FAIL.value, CheckOutcome.ERROR.value)
        for ev in reversed(self._record_plane.read_events()):
            if ev.get("ticket") != ticket_id:
                continue
            etype = ev.get("event_type")
            if failure_note is None and etype == EventType.DISPOSITION.value:
                reason = ev.get("reason")
                if isinstance(reason, str) and reason:
                    failure_note = reason
            elif wants_critic and critic_reason is None and etype == EventType.GATE.value:
                reason = ev.get("reason")
                if isinstance(reason, str) and reason:
                    critic_reason = reason
            elif wants_check and check_output is None and etype == EventType.CHECK.value:
                if ev.get("outcome") in _failing:
                    out = ev.get("output")
                    if isinstance(out, str) and out:
                        check_output = out
            if (
                failure_note is not None
                and (critic_reason is not None or not wants_critic)
                and (check_output is not None or not wants_check)
            ):
                break
        return PriorEvidence(
            attempt_kind=attempt_kind,
            check_output=check_output,
            critic_reason=critic_reason,
            failure_note=failure_note,
        )

    # -- weave ------------------------------------------------------------------

    def _maybe_weave(self, now: float) -> tuple[bool, tuple[str, ...], bool]:
        """Evaluate weave triggers (SPEC §9 "Weave triggers") and run
        `weaver.weave()` if triggered. Spend exhaustion still lets the
        ONE final weave cycle run via `final_weave_allowed()`'s one-shot
        gate (case 20) — consulted at most once, only when a trigger has
        actually fired AND the leash is exhausted."""
        loop_state = triggers_module.build_loop_state(self._store, now=now)
        cadences = triggers_module.cadences_from_charter(self._charter.raw)
        decision, next_prior_parked = triggers_module.evaluate_triggers(
            loop_state, cadences, self._prior_parked, now=now
        )
        self._prior_parked = next_prior_parked

        should_weave = decision.should_weave
        if should_weave and self._spend_leash.exhausted():
            should_weave = self._spend_leash.final_weave_allowed()

        if not should_weave:
            return False, (), False

        results = self._weaver.weave(now=now)
        weave_result_tickets = tuple(r.ticket for r in results)
        weave_infra_trip = self._process_weave_results(results, now=now)
        return True, weave_result_tickets, weave_infra_trip

    def _process_weave_results(self, results: list[WeaveResult], *, now: float) -> bool:
        """Process `weaver.weave()`'s results per §1.3 — see module
        docstring deviation 2 for exactly what state-machine work `.22`
        does (and, critically, does NOT do) on each branch, given
        weaver.py has already performed its own terminal transition +
        disposition. Returns `True` iff ANY weave-infra trip happened in
        this batch."""
        any_infra = False
        for result in results:
            # Record real gate-call usage on both MET and UNMET paths, before
            # any outcome branching. This mirrors record_dispatch's placement
            # before the outcome branch and ensures exactly one record_gate
            # per GATE event (no double-count against event-derived status).
            if result.gate_model is not None and result.gate_tokens is not None:
                self._spend_leash.record_gate(result.gate_model, result.gate_tokens)
            ticket_id = result.ticket
            if result.outcome == "landed":
                # bead .89: the ticket already RESTS at LANDED (weaver's own
                # GATED->LANDED transition). v0 does NOT advance it to DONE —
                # DAG intake (intake._deps_landed) and the spend report both
                # key on state=="landed", so a landed ticket must stay there.
                # A clean landing only breaks any consecutive-infra streak.
                self._reset_infra_trip_counter()
                # bead .107: a landed verdict resolves any prior consecutive
                # critic-infra streak for THIS ticket — reset its persisted
                # per-ticket counter too (a valid critic call, even one that
                # lands, breaks the streak).
                self._store.update_ticket(ticket_id, critic_infra_failures=0)
            elif result.outcome == "infra" and result.reason == "critic-infra":
                # bead .107: per-ticket critic-infra escalation cap — a
                # single poisoned ticket must escalate to the human floor
                # instead of taking down the whole daemon via the global
                # circuit breaker (see build-107-109-spec.md §2). Ticket is
                # ALREADY at PARKED (weaver's own contract).
                ticket_row = self._store.get_ticket(ticket_id) or {}
                cap = self._charter.raw["loop"]["retries"]["critic_infra"]
                decision = decide_critic_infra(ticket_row, critic_infra_cap=cap)
                self._store.update_ticket(
                    ticket_id, critic_infra_failures=decision.critic_infra_failures
                )
                if decision.escalate:
                    transition(self._store, ticket_id, ESCALATED, expected_from=PARKED)
                    ctx = self._weave_retry_ctx(ticket_id)
                    record_disposition(
                        self._record_plane,
                        ctx=ctx,
                        disposition="escalated",
                        attempt_kind="critic-infra",
                        reason=decision.escalation_reason,
                    )
                    self._persist_critic_infra_escalation_notification(
                        ticket_id, decision, now=now
                    )
                    # review-followup: do NOT reset the global infra-trip
                    # counter here. This branch already omits
                    # _bump_infra_trip_counter() above, which is enough to
                    # keep a single poisoned ticket's escalation from ever
                    # feeding the global halt on its own; resetting it too
                    # would ALSO erase any storm evidence already
                    # accumulated by OTHER tickets (dispatch-side infra or
                    # other parked tickets' below-cap critic-infra bumps)
                    # in this same poll or a prior one — masking a genuine
                    # multi-ticket infra storm. Neither feed nor forgive.
                else:
                    # Below cap: stays PARKED (re-weave next cadence), and
                    # still feeds the global storm backstop.
                    self._bump_infra_trip_counter()
                    any_infra = True
            elif result.outcome == "infra":
                # Non-critic-infra (e.g. CAS-abort/concurrent-staging-move):
                # unchanged behavior. Ticket is ALREADY at PARKED (weaver's
                # own contract). PARKED->POOL is illegal and must never be
                # risked: do NOTHING state-machine-wise, never call
                # decide_retry/apply_decision on this branch. Still feeds
                # the global storm backstop as always.
                self._bump_infra_trip_counter()
                any_infra = True
                # review-followup: a CAS-abort (reason=="concurrent-staging-
                # move") is produced ONLY after the critic returned a valid
                # MET verdict (weaver.py's CAS-land path is reached AFTER
                # critic.judge() succeeds) — the consecutive-critic-infra
                # streak is genuinely broken here, exactly like the landed/
                # rejected branches above. Reset the per-ticket counter too
                # (the global storm counter above is UNAFFECTED — CAS-abort
                # is still a transient infra event for storm accounting).
                if result.reason == "concurrent-staging-move":
                    self._store.update_ticket(ticket_id, critic_infra_failures=0)
            else:  # "rejected" | "integration-conflict" | "integration-regression"
                ctx = self._weave_retry_ctx(ticket_id)
                assert result.failure_class is not None  # noqa: S101 - non-landing => set
                self._apply_retry_decision(ticket_id, result.failure_class, ctx, now=now)
                self._reset_infra_trip_counter()
                # bead .107: a valid (even rejected) verdict resolves any
                # prior consecutive critic-infra streak for this ticket.
                self._store.update_ticket(ticket_id, critic_infra_failures=0)
        return any_infra

    def _weave_retry_ctx(self, ticket_id: str) -> dict[str, Any]:
        """A conservative identity ctx for the weave-side retry-ladder
        disposition — built purely from the ticket row + charter/registry,
        since no live dispatch identity exists at weave time (the ticket's
        lease fields were cleared long ago, the last time it re-entered
        `pool`)."""
        ticket_row = self._store.get_ticket(ticket_id) or {}
        ladder = self._charter.raw["stepup"]["ladder"]
        return {
            "rig": self._rig_name,
            "ticket": ticket_id,
            "dispatch_id": None,
            "attempt": (ticket_row.get("attempts_used") or 0) + 1,
            "rung": ticket_row.get("current_rung") or (ladder[0] if ladder else None),
            "worker": None,
            "charter_hash": self._charter.resolved_hash,
            "approval_hash": ticket_row.get("approval_hash"),
            "image_digest": None,
            "model": None,
            "model_version": None,
            "price_table_version": self._registry.version_hash,
        }

    # -- circuit breaker (SPEC §9, bead .22 build spec §3) -----------------------

    def _bump_infra_trip_counter(self) -> None:
        # Protected by lock: allows concurrent dispatch threads to safely increment
        # the counter without lost updates. The lock is held only for the brief
        # read-modify-write.
        with self._infra_trip_lock:
            self._consecutive_infra_trips += 1

    def _reset_infra_trip_counter(self) -> None:
        # Protected by lock: same reasoning as _bump_infra_trip_counter. The weaver
        # (main thread) and dispatch threads (pool) both call this; the lock
        # prevents lost updates when they race.
        with self._infra_trip_lock:
            self._consecutive_infra_trips = 0

    @staticmethod
    def _compute_backoff(consecutive_trips: int) -> float:
        if consecutive_trips <= 0:
            return 0.0
        backoff = _BACKOFF_BASE_SECONDS * (2 ** (consecutive_trips - 1))
        return min(backoff, _BACKOFF_CAP_SECONDS)

    def _persist_halt_notification(self, now: float) -> None:
        self._notification_store.record_intent(
            ticket=None,
            kind="halt",
            title="stigmergy daemon circuit breaker halt",
            message=(
                f"{self._consecutive_infra_trips} consecutive infra trips reached "
                f"the circuit-breaker threshold ({_CIRCUIT_BREAKER_THRESHOLD}); halting the loop."
            ),
            now=now,
        )

    def _persist_escalation_notification(
        self, ticket_id: str, decision: AttemptDecision, *, now: float
    ) -> None:
        self._notification_store.record_intent(
            ticket=ticket_id,
            kind="escalation",
            title=f"ticket {ticket_id} escalated",
            message=(
                f"escalation_reason={decision.escalation_reason}; "
                f"attempt_kind={decision.attempt_kind}"
            ),
            now=now,
        )

    def _persist_critic_infra_escalation_notification(
        self, ticket_id: str, decision: CriticInfraDecision, *, now: float
    ) -> None:
        """Bead .107: the per-ticket critic-infra escalation counterpart to
        `_persist_escalation_notification` — a single ticket's consecutive
        critic-infra failures reached `loop.retries.critic_infra`, so it
        was escalated to the human floor (PARKED->ESCALATED) instead of
        letting the whole loop take the global circuit-breaker halt."""
        self._notification_store.record_intent(
            ticket=ticket_id,
            kind="escalation",
            title=f"ticket {ticket_id} escalated (critic-infra)",
            message=(
                f"escalation_reason={decision.escalation_reason}; "
                "consecutive critic-infra failures reached cap"
            ),
            now=now,
        )
