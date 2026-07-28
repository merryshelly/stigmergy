"""Ticket state machine + retry semantics (SPEC.md §9 "State machine
(normative)" + "Failure classes" + "Retry semantics", §10 AC7/AC9).

**The transition matrix is the spec** (SPEC §9: "illegal transitions are
loader-tested"). :data:`LEGAL_TRANSITIONS` is the literal normative artifact;
:func:`transition` is the only mechanism-side function permitted to move a
ticket's `state` column, and it rejects any edge not present in the matrix.

**Counters are lifetime-per-ticket** (SPEC §9 "Retry semantics"): `attempts_used`
and `integration_failures` never reset except on a legitimate step-up (a
harder rung is a fresh floor). Nothing in this module resets them on daemon
restart/resumption — that is `recover.py`'s job to *not* do, by construction
(it never calls :func:`apply_decision`, only `intake.expire_leases`, which
never touches counters either).

**`decide_retry` is pure** (SPEC §9 "Retry semantics" + AC7 ladder walk): it
reads a ticket row and a failure class and returns an :class:`AttemptDecision`
— it mutates nothing. :func:`apply_decision` is the only function that
persists a decision, and it always routes the state change back through
:func:`transition` so the legal-edge guard still applies even to
retry/step-up/escalation moves.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

from stigmergy.records import EventType, RecordPlane, make_event

# --- states (SPEC §9: "pool -> eligible -> claimed(lease) -> in-flight ->
# [tier1-green -> parked] | [failed -> pool] -> gated -> landed | rejected ->
# pool ... -> escalated | done") ---------------------------------------------

POOL = "pool"
ELIGIBLE = "eligible"
CLAIMED = "claimed"
IN_FLIGHT = "in_flight"
TIER1_GREEN = "tier1_green"
PARKED = "parked"
GATED = "gated"
LANDED = "landed"
REJECTED = "rejected"
FAILED = "failed"
ESCALATED = "escalated"
DONE = "done"

STATES: frozenset[str] = frozenset(
    {
        POOL,
        ELIGIBLE,
        CLAIMED,
        IN_FLIGHT,
        TIER1_GREEN,
        PARKED,
        GATED,
        LANDED,
        REJECTED,
        FAILED,
        ESCALATED,
        DONE,
    }
)

# `landed` is the authoritative terminal-success state and where v0 RESTS
# (bead .89): the spend report counts `state == "landed"` (SPEC §9) AND DAG
# intake eligibility (`intake._deps_landed`) unblocks a dependent only when
# every predecessor is `state == "landed"` — so a landed ticket must not move
# past it. `done` is a reserved post-land archival state, defined for
# vocabulary completeness but UNREACHABLE in v0 (nothing transitions into it
# — the LANDED edge set is empty; see below). Only LANDED and DONE are
# terminal (no outgoing edges). ESCALATED is no longer terminal as of bead .102c,
# which adds the sanctioned operator resume re-entry edge (ESCALATED->POOL via CLI).
TERMINAL_STATES: frozenset[str] = frozenset({LANDED, DONE})

# The four lease columns (RigStore schema, ticket .15) — cleared together
# whenever a ticket re-enters `pool` (recover/retry re-entry never leaves a
# stale lease pinned to a ticket that is once again claimable).
_LEASE_FIELDS: tuple[str, ...] = (
    "lease_owner",
    "lease_dispatch_id",
    "lease_expires_at",
    "lease_heartbeat_at",
)

# --- the transition matrix (NORMATIVE) --------------------------------------

LEGAL_TRANSITIONS: dict[str, frozenset[str]] = {
    # intake._CLAIMABLE_STATES == {"pool", "eligible"}; intake.claim() sets
    # state="claimed" from either, so both pool->claimed and eligible->claimed
    # are legal. pool->eligible is the intake eligibility query surfacing a
    # ticket (no lease taken yet).
    POOL: frozenset({ELIGIBLE, CLAIMED}),
    # eligible->pool is the approval-hash-mutation de-eligibilizing edge
    # (SPEC §10 AC11): a steering mutation invalidates the stored approval
    # hash and the ticket falls back out of the eligible set.
    ELIGIBLE: frozenset({CLAIMED, POOL}),
    # claimed->pool is the recover/expire_leases orphan path: a lease that
    # expires before dispatch actually starts.
    CLAIMED: frozenset({IN_FLIGHT, POOL}),
    # in_flight->tier1_green: checks pass. in_flight->failed: dispatch/check/
    # wedge/degenerate failure (the *class* of failure is metadata carried
    # alongside, not encoded in the state). in_flight->pool: the recover
    # orphan path (daemon crash mid-dispatch, container reaped + expired).
    IN_FLIGHT: frozenset({TIER1_GREEN, FAILED, POOL}),
    # tier1_green->parked: the park station stages a tier1-green bundle for
    # the weaver to pick up.
    TIER1_GREEN: frozenset({PARKED}),
    # parked->gated: a weave trigger (SPEC §9 "Weave triggers") fires and
    # picks up the parked bundle for gating. parked->escalated (bead .107):
    # a single ticket's per-ticket critic-infra failures reached the cap
    # (`decide_critic_infra`) — escalate to the human floor instead of
    # letting the whole loop take the global circuit-breaker halt for one
    # poisoned ticket.
    PARKED: frozenset({GATED, ESCALATED}),
    # gated->landed: critic pass + CAS land. gated->rejected: any gate
    # FAILURE class (critic quality-reject, integration-conflict,
    # integration-regression) — the single edge serves all three; the
    # FailureClass (not the state) decides which counter moves.
    # gated->parked: critic-call failure is INFRA (SPEC §9): it blocks
    # landing but is never recorded as a rejection, so the bundle just goes
    # back to parked to be re-gated later.
    GATED: frozenset({LANDED, REJECTED, PARKED}),
    # failed->pool: retry at the same rung, or step-up to the next rung
    # (both re-enter pool; the rung/attempt bookkeeping is a counter
    # change, not a different state). failed->escalated: ladder exhausted.
    FAILED: frozenset({POOL, ESCALATED}),
    # rejected->pool: retry (revision/reconcile). rejected->escalated:
    # integration_failures cap or rung/ladder exhaustion.
    REJECTED: frozenset({POOL, ESCALATED}),
    # bead .89: v0 rests at LANDED (terminal-success). The former LANDED->DONE
    # edge was taken by the daemon and advanced a landed ticket to DONE,
    # silently breaking DAG intake + the spend report (both key on
    # state=="landed"). DONE stays a reserved-but-unreachable archival state.
    LANDED: frozenset(),
    # bead .102c: ESCALATED is no longer terminal due to the sanctioned operator
    # resume re-entry edge (ESCALATED->POOL via CLI verb). An escalated ticket
    # that a human has triaged/re-approved can re-enter the pool without manual SQL.
    ESCALATED: frozenset({POOL}),
    # Terminal. No outgoing edges.
    DONE: frozenset(),
}


class IllegalTransition(Exception):
    """Raised by :func:`transition` (and, transitively, :func:`apply_decision`)
    on a missing ticket, an `expected_from` mismatch, or any (from, to) pair
    not present in :data:`LEGAL_TRANSITIONS`."""


def is_legal(from_state: str, to_state: str) -> bool:
    """True iff ``to_state`` is a legal destination from ``from_state`` per
    :data:`LEGAL_TRANSITIONS`. An unknown ``from_state`` has no legal
    destinations (fails closed, never raises)."""
    return to_state in LEGAL_TRANSITIONS.get(from_state, frozenset())


def transition(
    store: Any,
    ticket_id: str,
    to_state: str,
    *,
    expected_from: str | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Move ``ticket_id`` to ``to_state`` iff the edge is legal (SPEC §9).

    ``expected_from``, if given, is an optimistic-concurrency guard: the
    ticket's current state must equal it exactly, or this raises
    :class:`IllegalTransition` naming both the expected and actual state —
    even when ``to_state`` would otherwise be a legal destination from the
    ticket's *actual* current state.

    When ``to_state == POOL``, all four lease fields are cleared to `None`
    (recover/retry re-entry: a ticket back in the pool holds no stale lease).
    Counters (`attempts_used`, `integration_failures`, `current_rung`) are
    NEVER touched here — this is the happy-path state setter only;
    :func:`apply_decision` is where a retry decision's counter effects land.

    ``now`` is accepted for signature symmetry with the rest of this
    module's API (and for future use by callers that want to journal the
    transition's timestamp) but is not currently consulted — `store.
    update_ticket` already stamps its own `updated_at`.

    Returns ``{"ticket": ticket_id, "from": from_state, "to": to_state}``.
    Raises :class:`IllegalTransition` if the ticket doesn't exist, the
    `expected_from` guard fails, or the edge is illegal.
    """
    del now  # reserved, not yet consulted (see docstring)
    row = store.get_ticket(ticket_id)
    if row is None:
        raise IllegalTransition(f"cannot transition: no such ticket {ticket_id!r}")

    from_state = row["state"]
    if expected_from is not None and from_state != expected_from:
        raise IllegalTransition(
            f"ticket {ticket_id!r}: expected from-state {expected_from!r} "
            f"but current state is {from_state!r}"
        )
    if not is_legal(from_state, to_state):
        raise IllegalTransition(
            f"ticket {ticket_id!r}: illegal transition {from_state!r} -> {to_state!r}"
        )

    fields: dict[str, Any] = {"state": to_state}
    if to_state == POOL:
        fields.update(dict.fromkeys(_LEASE_FIELDS))
    store.update_ticket(ticket_id, **fields)

    return {"ticket": ticket_id, "from": from_state, "to": to_state}


# --- failure classes + retry decision (SPEC §9 "Failure classes" +
# "Retry semantics") ---------------------------------------------------------


class FailureClass(enum.Enum):
    """The seven failure classes a dispatch/gate outcome resolves to
    (SPEC §9 "Failure classes"). The class — never the state alone —
    decides which counter moves and what the next `attempt_kind` is."""

    TIER1_FAIL = "tier1-fail"
    WEDGE = "wedge"
    DEGENERATE = "degenerate"
    REJECTED = "rejected"
    INTEGRATION_CONFLICT = "integration-conflict"
    INTEGRATION_REGRESSION = "integration-regression"
    INFRA = "infra"


# Classes that burn a rung attempt (SPEC §9: "Only `rejected` burns rung
# attempts / drives step-up" for gate failures; tier1-fail/wedge/degenerate
# always do).
_RUNG_CONSUMING_CLASSES = frozenset(
    {
        FailureClass.TIER1_FAIL,
        FailureClass.WEDGE,
        FailureClass.DEGENERATE,
        FailureClass.REJECTED,
    }
)

# Classes that increment `integration_failures` instead (never a rung
# attempt): "conflict/regression increment integration_failures (lifetime
# cap -> escalate framed as triage failure)".
_INTEGRATION_CLASSES = frozenset(
    {FailureClass.INTEGRATION_CONFLICT, FailureClass.INTEGRATION_REGRESSION}
)

# Same-rung retry attempt_kind + pre-application policy, by failure class
# (SPEC §9 "Retry semantics" pre-application policy paragraph).
# pre_apply_prior=True: dispatch applies the prior work product to a fresh
# clone (revision, not re-roll) - or falls back to clean-restart on a dirty
# apply (.21's job, not this module's).
# prior_as_reference=True: prior bundle attached only as read-only reference
# (no usable product from a wedge/degenerate run to revise).
_SAME_RUNG_RETRY: dict[FailureClass, tuple[str, bool, bool]] = {
    FailureClass.TIER1_FAIL: ("tier1-repair", True, False),
    FailureClass.REJECTED: ("critic-revision", True, False),
    FailureClass.WEDGE: ("clean-restart", False, True),
    FailureClass.DEGENERATE: ("clean-restart", False, True),
}


@dataclass(frozen=True)
class AttemptDecision:
    """The pure output of :func:`decide_retry`: what the NEXT attempt looks
    like (or that the ticket escalates instead), plus the resulting counter
    values to persist. Never itself mutates anything — see
    :func:`apply_decision` for the persistence step."""

    attempt_kind: str
    rung: str
    next_state: str
    pre_apply_prior: bool
    prior_as_reference: bool
    escalation_reason: str | None
    attempts_used: int
    current_rung: str
    integration_failures: int


def decide_retry(
    ticket_row: dict[str, Any],
    failure_class: FailureClass,
    *,
    ladder: list[str],
    attempts_per_rung: int,
    integration_failures_cap: int,
) -> AttemptDecision:
    """Decide the next attempt (or escalation) for one failed dispatch
    (SPEC §9 "Retry semantics" + AC7 ladder walk). PURE: reads counters off
    ``ticket_row``, returns a decision, mutates nothing (not even
    ``ticket_row`` itself).

    ``ladder`` is the charter's ``stepup.ladder`` (ascending capability
    order, e.g. ``["cheap", "default", "exquisite"]``).
    ``ticket_row["current_rung"]``, if `None`/absent, is treated as
    ``ladder[0]`` (a ticket that has never stepped up starts at the floor).
    """
    current_rung = ticket_row.get("current_rung")
    if current_rung is None:
        current_rung = ladder[0]
    attempts_used = ticket_row.get("attempts_used") or 0
    integration_failures = ticket_row.get("integration_failures") or 0

    if failure_class is FailureClass.INFRA:
        # Infra consumes nothing: no rung attempt, no dispatch budget (SPEC
        # §9 "no rung attempts, no dispatch budget"). Bounded backoff lives
        # outside this pure function (.21/.22's job).
        return AttemptDecision(
            attempt_kind="infra-retry",
            rung=current_rung,
            next_state=POOL,
            pre_apply_prior=False,
            prior_as_reference=False,
            escalation_reason=None,
            attempts_used=attempts_used,
            current_rung=current_rung,
            integration_failures=integration_failures,
        )

    if failure_class in _INTEGRATION_CLASSES:
        return _decide_integration_retry(
            failure_class,
            current_rung=current_rung,
            attempts_used=attempts_used,
            integration_failures=integration_failures,
            integration_failures_cap=integration_failures_cap,
        )

    if failure_class in _RUNG_CONSUMING_CLASSES:
        return _decide_rung_consuming_retry(
            failure_class,
            ladder=ladder,
            current_rung=current_rung,
            attempts_used=attempts_used,
            integration_failures=integration_failures,
            attempts_per_rung=attempts_per_rung,
        )

    raise AssertionError(  # pragma: no cover
        f"unreachable: unhandled FailureClass {failure_class!r}"
    )


def _decide_integration_retry(
    failure_class: FailureClass,
    *,
    current_rung: str,
    attempts_used: int,
    integration_failures: int,
    integration_failures_cap: int,
) -> AttemptDecision:
    """`integration-conflict`/`integration-regression`: never burns a rung
    attempt; increments `integration_failures` and, past the cap, escalates
    as a triage failure (colliding tickets should have been serialized at
    triage — SPEC §9)."""
    del failure_class  # both integration classes behave identically here
    integration_failures += 1
    escalate = integration_failures >= integration_failures_cap
    return AttemptDecision(
        attempt_kind="integration-reconcile",
        rung=current_rung,
        next_state=ESCALATED if escalate else POOL,
        pre_apply_prior=False,
        prior_as_reference=True,
        escalation_reason="triage-failure" if escalate else None,
        attempts_used=attempts_used,
        current_rung=current_rung,
        integration_failures=integration_failures,
    )


def _decide_rung_consuming_retry(
    failure_class: FailureClass,
    *,
    ladder: list[str],
    current_rung: str,
    attempts_used: int,
    integration_failures: int,
    attempts_per_rung: int,
) -> AttemptDecision:
    """TIER1_FAIL / WEDGE / DEGENERATE / REJECTED: always burns a rung
    attempt; drives same-rung retry, step-up, or ladder-exhaustion
    escalation (SPEC §9 + AC7)."""
    attempts_used += 1
    rung_exhausted = attempts_used >= attempts_per_rung
    is_final_rung = current_rung == ladder[-1]

    if rung_exhausted and is_final_rung:
        # Ladder-exhausted: the attempt that WOULD have run (same mapping as
        # the same-rung-retry branch) is recorded for the disposition, but
        # next_state is ESCALATED instead of POOL — it never actually runs.
        kind, pre_apply_prior, prior_as_reference = _SAME_RUNG_RETRY[failure_class]
        return AttemptDecision(
            attempt_kind=kind,
            rung=current_rung,
            next_state=ESCALATED,
            pre_apply_prior=pre_apply_prior,
            prior_as_reference=prior_as_reference,
            escalation_reason="ladder-exhausted",
            attempts_used=attempts_used,
            current_rung=current_rung,
            integration_failures=integration_failures,
        )

    if rung_exhausted and not is_final_rung:
        # Step-up: advance to the next ladder entry, RESET attempts_used to
        # 0 (a legit transition — a harder rung is a fresh floor; contrast
        # daemon-restart resumption, which never resets this).
        next_rung = ladder[ladder.index(current_rung) + 1]
        return AttemptDecision(
            attempt_kind="stepup-initial",
            rung=next_rung,
            next_state=POOL,
            pre_apply_prior=False,
            prior_as_reference=True,
            escalation_reason=None,
            attempts_used=0,
            current_rung=next_rung,
            integration_failures=integration_failures,
        )

    # Same-rung retry (not exhausted). OVERRIDE (SPEC §9 "final attempt on
    # the final rung is always clean-restart" — anchor decontamination):
    # if the retry we're about to schedule is the LAST allowed attempt on
    # the FINAL rung, force clean-restart regardless of class.
    final_attempt_before_exhaustion = is_final_rung and attempts_used == attempts_per_rung - 1
    if final_attempt_before_exhaustion:
        kind, pre_apply_prior, prior_as_reference = "clean-restart", False, True
    else:
        kind, pre_apply_prior, prior_as_reference = _SAME_RUNG_RETRY[failure_class]

    return AttemptDecision(
        attempt_kind=kind,
        rung=current_rung,
        next_state=POOL,
        pre_apply_prior=pre_apply_prior,
        prior_as_reference=prior_as_reference,
        escalation_reason=None,
        attempts_used=attempts_used,
        current_rung=current_rung,
        integration_failures=integration_failures,
    )


def apply_decision(
    store: Any, ticket_id: str, decision: AttemptDecision, *, now: float | None = None
) -> None:
    """Persist an :class:`AttemptDecision` (the only place a decision from
    :func:`decide_retry` is ever written to the store).

    The state move happens FIRST, through :func:`transition` — so legality
    is still enforced on the state change (an illegal edge raises
    :class:`IllegalTransition` and leaves the counters entirely untouched,
    no partial write). Only once the state move succeeds are the resulting
    counters (`attempts_used`, `current_rung`, `integration_failures`)
    written via `store.update_ticket`.
    """
    transition(store, ticket_id, decision.next_state, now=now)
    store.update_ticket(
        ticket_id,
        attempts_used=decision.attempts_used,
        current_rung=decision.current_rung,
        integration_failures=decision.integration_failures,
    )


# --- per-ticket critic-infra escalation (bead .107) --------------------------


@dataclass(frozen=True)
class CriticInfraDecision:
    """The pure output of :func:`decide_critic_infra`: whether this
    ticket's per-ticket, PERSISTED consecutive-critic-infra streak has
    reached its cap, plus the resulting counter value to persist. Never
    itself mutates anything."""

    escalate: bool
    critic_infra_failures: int
    escalation_reason: str | None


def decide_critic_infra(
    ticket_row: dict[str, Any], *, critic_infra_cap: int
) -> CriticInfraDecision:
    """Decide the outcome of ONE critic-infra weave failure for a parked
    ticket (bead .107). PURE: reads `ticket_row["critic_infra_failures"]`
    (defaulting to 0 if absent/None, same convention as `decide_retry`'s
    own counter reads), increments it, and — at or past
    ``critic_infra_cap`` — escalates (PARKED->ESCALATED). Mutates nothing,
    not even ``ticket_row`` itself.

    This is a DIFFERENT, dedicated counter from `integration_failures`
    (conflating the two would corrupt the integration-failure escalation
    path — a different failure mode needs a different counter) and is
    per-ticket + persisted (survives daemon restarts; an in-memory counter
    would reset and re-livelock on a poisoned ticket).

    ``critic_infra_cap`` MUST be configured below the daemon's global
    circuit-breaker threshold (`daemon._CIRCUIT_BREAKER_THRESHOLD`, 5) —
    the charter default (`loop.retries.critic_infra` == 3) preserves this
    invariant so a single poisoned ticket escalates itself before a
    genuine multi-ticket infra storm would trip the global halt.
    """
    failures = (ticket_row.get("critic_infra_failures") or 0) + 1
    escalate = failures >= critic_infra_cap
    return CriticInfraDecision(
        escalate=escalate,
        critic_infra_failures=failures,
        escalation_reason="critic-infra-cap" if escalate else None,
    )


# --- disposition journaling helper -------------------------------------------

_ZERO_TOKENS: dict[str, int] = {"in": 0, "cached": 0, "out": 0, "reasoning": 0}

# The identity fields the caller must supply via `ctx` (SPEC §8 common
# fields minus the ones this helper fills in itself: tokens, computed_usd,
# wall_time_seconds, attempt_kind).
_DISPOSITION_CTX_FIELDS: tuple[str, ...] = (
    "rig",
    "ticket",
    "dispatch_id",
    "attempt",
    "rung",
    "worker",
    "charter_hash",
    "approval_hash",
    "image_digest",
    "model",
    "model_version",
    "price_table_version",
)


def record_disposition(
    record_plane: RecordPlane,
    *,
    ctx: dict[str, Any],
    disposition: str,
    attempt_kind: str,
    reason: str | None = None,
) -> None:
    """Journal one MEANINGFUL disposition event (SPEC §8/§9).

    Do NOT call this for mechanical transitions already implied by
    intake/dispatch events (pool->eligible, claimed->in_flight) — only for
    dispositions that carry real information: landed, rejected, escalated,
    parked, and pool re-entries carrying an `attempt_kind` decision.

    ``ctx`` supplies the live dispatch's identity fields (see
    :data:`_DISPOSITION_CTX_FIELDS`); this helper fills in the rest of the
    SPEC §8 common-field set with declared true zeros — a disposition
    genuinely spends nothing (`tokens` all-zero, `computed_usd=0.0`,
    `wall_time_seconds=0.0`). This is safe because the live spend leash
    (`spend.py`) is accrual-based and never sums events, so a declared-zero
    disposition event can never poison it.

    ``disposition`` (the to-state, e.g. `"landed"`/`"rejected"`/
    `"escalated"`/`"parked"`/`"pool"`) and the optional ``reason`` are
    carried as extra fields on the event alongside the SPEC §8 common set.
    """
    fields = {key: ctx.get(key) for key in _DISPOSITION_CTX_FIELDS}
    fields["attempt_kind"] = attempt_kind
    fields["tokens"] = dict(_ZERO_TOKENS)
    fields["computed_usd"] = 0.0
    fields["wall_time_seconds"] = 0.0
    fields["disposition"] = disposition
    if reason is not None:
        fields["reason"] = reason

    event = make_event(EventType.DISPOSITION, **fields)
    record_plane.append(event)
