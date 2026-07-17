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
   - `"landed"`: `transition(store, ticket, DONE, expected_from=LANDED)`
     only. No new DISPOSITION event (weaver's own `"landed"` disposition
     already satisfies the "one DISPOSITION per weave-ticket outcome"
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
9. **`spend_leash.record_gate` is never called anywhere in this module.**
   `weaver.py`'s `WeaveResult` carries no token-usage/cost data for its
   critic call (`critic.judge`'s `gate_fields` also carries no usage), so
   there is no accessible data for `.22` to feed `record_gate` with. This
   mirrors the fact that nothing else in the shipped codebase calls
   `record_gate` either — gate-call USD accounting is architecturally
   unwired as of this bead; flagged as a residual, not fixed here (no new
   `weaver.py`/`critic.py` surface invented to carry it).

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

import logging
import secrets
import shlex
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stigmergy import checks, egress
from stigmergy import intake as intake_module
from stigmergy import recover as recover_module
from stigmergy import triggers as triggers_module
from stigmergy.charter import Charter
from stigmergy.checks import CheckOutcome, CheckResult
from stigmergy.dispatch import DispatchPlan, prepare_dispatch, select_lane
from stigmergy.drivers import claude_code
from stigmergy.drivers.claude_code import DispatchResult, DispatchStatus
from stigmergy.egress import EgressHandle
from stigmergy.filing import harvest_worker_filings
from stigmergy.intake import LeaseError
from stigmergy.notify import NotificationStore, NtfyNotifier
from stigmergy.records import EventType, RecordPlane, make_event
from stigmergy.recover import ContainerReaper, RecoveryReport
from stigmergy.registry import Registry
from stigmergy.relay import Capability, CapabilityStore, build_redactor
from stigmergy.rig import RigStore
from stigmergy.spend import SpendLeash
from stigmergy.statemachine import (
    CLAIMED,
    DONE,
    ESCALATED,
    FAILED,
    IN_FLIGHT,
    LANDED,
    PARKED,
    POOL,
    TIER1_GREEN,
    AttemptDecision,
    FailureClass,
    apply_decision,
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
_CIRCUIT_BREAKER_THRESHOLD = 5

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


@dataclass(frozen=True)
class PollSummary:
    """Everything one `poll_once()` call did, for tests to assert against
    and for `run_forever()` to act on (bead .22 build spec §4)."""

    dispatched_ticket: str | None
    dispatch_status: str | None
    weave_ran: bool
    weave_results: tuple[str, ...]
    infra_trip: bool
    consecutive_infra_trips: int
    should_halt: bool
    backoff_seconds: float


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
        secrets_for_capability: Callable[[Capability], frozenset[str]] | None = None,
        now_fn: Callable[[], float] = time.time,
        git_repo_paths: list[str | Path] | None = None,
        min_disk_bytes: int = 100 * 1024 * 1024,
        disk_path: str | Path | None = None,
        owner: str = "daemon",
        relay_base_url: str = "http://127.0.0.1:0/stigmergy-relay-placeholder",
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

        self._image = charter.raw["rig"]["image"]
        self._rig_name = charter.raw.get("rig", {}).get("name")
        self._lease_ttl_seconds = charter.raw["loop"]["timers"]["lease_ttl_seconds"]
        self._poll_seconds = charter.raw["loop"]["timers"]["poll_seconds"]

        self._consecutive_infra_trips = 0
        self._prior_parked: dict[str, float] = {}

    # -- recover_on_start ----------------------------------------------------

    def recover_on_start(self) -> RecoveryReport:
        """Calls `recover.recover(...)` exactly once (bead .22 build spec
        §4). `recover.RecoveryError` (disk headroom) propagates uncaught —
        fatal, per the session brief."""
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
        spend-gated claim+dispatch+checks (at most one ticket, workers=1),
        weave-trigger evaluation + weave if triggered, notification
        delivery. Never sleeps. See the module docstring + build spec §1-§3
        for the exact decision logic."""
        now = self._now_fn()
        self._store.set_meta("daemon_heartbeat_at", str(now))

        dispatched_ticket: str | None = None
        dispatch_status_value: str | None = None
        dispatch_infra_trip = False

        if self._spend_leash.can_dispatch():
            claimed = self._claim_and_prepare(now)
            if claimed is not None:
                ticket_id, plan, egress_handle = claimed
                dispatched_ticket = ticket_id
                dispatch_status_value, dispatch_infra_trip = self._run_dispatch_cycle(
                    ticket_id, plan, egress_handle
                )

        weave_ran, weave_result_tickets, weave_infra_trip = self._maybe_weave(now)

        infra_trip = dispatch_infra_trip or weave_infra_trip
        consecutive = self._consecutive_infra_trips
        should_halt = consecutive >= _CIRCUIT_BREAKER_THRESHOLD
        backoff_seconds = self._compute_backoff(consecutive) if infra_trip else 0.0

        if should_halt:
            self._persist_halt_notification(now)

        # Notification delivery: every poll attempts to flush pending
        # intents (AC12 "persisted with retry") — this single call also
        # satisfies §3's "attempt one NtfyNotifier.deliver_pending call"
        # for a halt this cycle, since the halt intent was just persisted
        # above, before this line runs.
        self._notifier.deliver_pending(self._notification_store, now=now)

        return PollSummary(
            dispatched_ticket=dispatched_ticket,
            dispatch_status=dispatch_status_value,
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

    # -- claim + prepare ------------------------------------------------------

    def _claim_and_prepare(
        self, now: float
    ) -> tuple[str, DispatchPlan, EgressHandle | None] | None:
        """Find one eligible ticket, claim it, and prepare its dispatch.
        Returns `None` if nothing is eligible, or if a race makes the
        claim itself fail (never a bug — `LeaseError` is an ordinary
        "try again next poll" signal)."""
        eligible_ids = intake_module.eligible(
            self._store, now=now, steering_of=self._steering_of
        )
        if not eligible_ids:
            return None
        ticket_id = eligible_ids[0]
        ticket_row = self._store.get_ticket(ticket_id)
        steering = self._steering_of(ticket_id)
        lane = select_lane(self._charter, ticket_row)
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

        egress_handle = self._setup_egress(ticket_id, lane, now)
        egress_socket = egress_handle.socket_path if egress_handle is not None else None

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
        )
        # Reconcile the ticket's lease_dispatch_id to the REAL dispatch
        # identity now that prepare_dispatch has generated it — see module
        # docstring point 7.
        self._store.update_ticket(ticket_id, lease_dispatch_id=plan.dispatch_id)
        return ticket_id, plan, egress_handle

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

    def _setup_egress(self, ticket_id: str, lane: Any, now: float) -> EgressHandle | None:
        """v0 pre-.25: `None` unless `egress_setup_fn` is injected (see
        module docstring "Egress" note)."""
        if self._egress_setup_fn is None:
            return None
        policy = egress.policy_for_lane(self._charter.raw, lane.name)
        runtime_dir = Path(self._rig_paths["clones_root"]) / "_egress_runtime"
        provisional_id = f"egress-{ticket_id}-{int(now * 1000)}"
        return self._egress_setup_fn(provisional_id, policy, runtime_dir)

    # -- dispatch cycle ---------------------------------------------------------

    def _run_dispatch_cycle(
        self, ticket_id: str, plan: DispatchPlan, egress_handle: EgressHandle | None
    ) -> tuple[str, bool]:
        """Spawn, emit the DISPATCH event, then run the §1.1/§1.2 decision
        tree. `capability_store.revoke(dispatch_id)` and egress teardown
        ALWAYS run in a `finally` around this whole block — covers every
        exit path, including an unexpected exception from `spawn_fn`
        (case 16: the exception propagates after `finally` runs; revoke
        still happened)."""
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

            try:
                limits = self._charter.raw["loop"]["dispatch_limits"]
                harvest_worker_filings(
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

            infra_trip = self._handle_dispatch_outcome(ticket_id, plan, result, ctx, now=t_end)
        finally:
            self._capability_store.revoke(plan.dispatch_id)
            if egress_handle is not None:
                self._egress_teardown_fn(egress_handle)

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
        self._run_tier1_checks(ticket_id, plan, ctx, now=now)
        return False

    def _dispatch_failure(
        self, ticket_id: str, failure_class: FailureClass, ctx: dict[str, Any], *, now: float
    ) -> None:
        transition(self._store, ticket_id, FAILED, expected_from=IN_FLIGHT)
        self._apply_retry_decision(ticket_id, failure_class, ctx, now=now)

    # -- Tier-1 checks (§1.2) ---------------------------------------------------

    def _run_tier1_checks(
        self, ticket_id: str, plan: DispatchPlan, ctx: dict[str, Any], *, now: float
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
        check_results = self._run_checks_fn(
            check_dict,
            work_tree=plan.work_clone,
            image=self._checker_image,
            flake_reruns=flake_reruns,
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
        self, ticket_id: str, failure_class: FailureClass, ctx: dict[str, Any], *, now: float
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
            ticket_id = result.ticket
            if result.outcome == "landed":
                transition(self._store, ticket_id, DONE, expected_from=LANDED)
                self._reset_infra_trip_counter()
            elif result.outcome == "infra":
                # Ticket is ALREADY at PARKED (weaver's own contract).
                # PARKED->POOL is illegal and must never be risked: do
                # NOTHING state-machine-wise, never call
                # decide_retry/apply_decision on this branch.
                self._bump_infra_trip_counter()
                any_infra = True
            else:  # "rejected" | "integration-conflict" | "integration-regression"
                ctx = self._weave_retry_ctx(ticket_id)
                assert result.failure_class is not None  # noqa: S101 - non-landing => set
                self._apply_retry_decision(ticket_id, result.failure_class, ctx, now=now)
                self._reset_infra_trip_counter()
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
        self._consecutive_infra_trips += 1

    def _reset_infra_trip_counter(self) -> None:
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
