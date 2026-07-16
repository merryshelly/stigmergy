"""Weave triggers — the OR-set that decides when the loop starts a weave
(SPEC.md §9 "Weave triggers (OR-ed — fixes the one-ticket deadlock)",
§5 `[loop.cadences]`, §10 AC13).

**Pure decision module** (mirrors :func:`stigmergy.statemachine.decide_retry`'s
discipline): given a snapshot of the rig's ticket states, the charter
cadences, and the prior poll's parked snapshot, decide whether the loop
should start a weave now, and via which trigger. No store writes and no
I/O in the core evaluation — :func:`build_loop_state` is the ONLY
store-touching function here; the daemon (.22) calls it once per poll and
threads the returned tracker forward into the next poll's call.

**The three triggers, OR-ed** (SPEC §9): a weave fires if ANY holds —

- **quiescent:** the count of *quiescent* parked tickets (present AND
  unchanged across the poll boundary) reaches `staging_quiescent_tickets`.
- **queue-drained (implicit, ALWAYS ON):** no active/in-flight work AND
  >=1 parked ticket. This is the rev-1 DEADLOCK FIX (AC13): a single
  parked ticket with a drained queue must weave immediately rather than
  wait for a quiescent threshold it can structurally never reach alone.
- **max-wait:** the oldest parked ticket's age reaches
  `staging_max_wait_seconds`.

A rig with zero parked tickets never weaves (nothing to gate).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from stigmergy.statemachine import CLAIMED, ELIGIBLE, IN_FLIGHT, PARKED, POOL, TIER1_GREEN

# States that mean "the queue is NOT drained" (in-flight/claimable work
# still exists). `gated`/`landed`/`rejected`/`failed`/`escalated`/`done`
# are deliberately excluded — none of them are active dispatch/claim work
# for trigger purposes.
ACTIVE_STATES = frozenset({POOL, ELIGIBLE, CLAIMED, IN_FLIGHT, TIER1_GREEN})

# Charter in-code defaults (charter.py DEFAULT_CHARTER["loop"]["cadences"]),
# duplicated here only as the defensive fallback `cadences_from_charter`
# uses when a key is missing from an already-resolved charter — the
# resolved charter is expected to already carry these via charter.py's own
# defaulting, so this should rarely, if ever, be exercised.
_DEFAULT_STAGING_QUIESCENT_TICKETS = 3
_DEFAULT_STAGING_MAX_WAIT_SECONDS = 7200.0


@dataclass(frozen=True)
class Cadences:
    """The two `[loop.cadences]` knobs this module consults (SPEC §5)."""

    staging_quiescent_tickets: int
    staging_max_wait_seconds: float


@dataclass(frozen=True)
class ParkedTicket:
    """One parked ticket's identity + the instant it entered `parked`.

    `parked_since` is the ticket's `updated_at` while parked — constant for
    as long as the ticket stays parked, which is exactly what lets the
    quiescent trigger detect "unchanged across a poll boundary".
    """

    ticket_id: str
    parked_since: float


@dataclass(frozen=True)
class LoopState:
    """One poll's snapshot of the state this module needs: the parked set
    plus a count of tickets doing active/claimable work."""

    parked: tuple[ParkedTicket, ...]
    active_count: int


@dataclass(frozen=True)
class WeaveDecision:
    """The pure output of :func:`evaluate_triggers`."""

    should_weave: bool
    trigger: str | None  # "quiescent" | "queue-drained" | "max-wait" | None
    quiescent_count: int
    reason: str


def evaluate_triggers(
    state: LoopState,
    cadences: Cadences,
    prior_parked: Mapping[str, float],
    *,
    now: float,
) -> tuple[WeaveDecision, dict[str, float]]:
    """PURE. Decide whether to weave now, and via which trigger (SPEC §9).

    Returns ``(decision, next_prior_parked)``. ``next_prior_parked`` is the
    CURRENT parked set reshaped as ``{ticket_id: parked_since}`` — the
    caller feeds it back in as ``prior_parked`` on the NEXT poll's call.
    This round-trip is how "unchanged for >=1 poll cycle" is measured: a
    parked ticket is quiescent iff it is present in ``prior_parked`` with
    the exact same ``parked_since`` (a ticket parked for the first time
    this poll is never quiescent yet; it becomes quiescent next poll if it
    is still parked, unchanged).

    Trigger-label priority when several fire simultaneously: quiescent >
    queue-drained > max-wait. ``should_weave`` is the OR of all three —
    the label only names the highest-priority one that fired, for
    status/logging purposes.

    **Guard (quiescent):** honor ``staging_quiescent_tickets`` as the
    threshold, but ALSO require ``quiescent_count >= 1`` — this is
    deliberate so a charter misconfigured with ``staging_quiescent_tickets
    <= 0`` can never fire the quiescent trigger on an empty (zero
    quiescent) staging area. ``staging_max_wait_seconds <= 0`` is NOT
    special-cased (that is the charter's business to police); it can make
    max-wait fire immediately given >=1 parked ticket, but the
    queue-drained/quiescent paths never depend on it either way.
    """
    next_prior_parked = {p.ticket_id: p.parked_since for p in state.parked}

    if not state.parked:
        # Nothing to gate — never weaves, regardless of active_count.
        decision = WeaveDecision(
            should_weave=False,
            trigger=None,
            quiescent_count=0,
            reason="no parked tickets",
        )
        return decision, next_prior_parked

    quiescent_count = sum(
        1
        for p in state.parked
        if p.ticket_id in prior_parked and prior_parked[p.ticket_id] == p.parked_since
    )

    quiescent_fires = (
        quiescent_count >= 1 and quiescent_count >= cadences.staging_quiescent_tickets
    )
    # queue-drained (implicit, ALWAYS ON — AC13 rev-1 deadlock fix): no
    # active/in-flight work AND at least one parked ticket (already
    # guaranteed by the `not state.parked` guard above).
    queue_drained_fires = state.active_count == 0
    oldest_parked_since = min(p.parked_since for p in state.parked)
    max_wait_fires = (now - oldest_parked_since) >= cadences.staging_max_wait_seconds

    should_weave = quiescent_fires or queue_drained_fires or max_wait_fires

    if quiescent_fires:
        trigger = "quiescent"
        reason = f"{quiescent_count} quiescent parked ticket(s) reached threshold"
    elif queue_drained_fires:
        trigger = "queue-drained"
        reason = "queue drained (no active work) with parked ticket(s) waiting"
    elif max_wait_fires:
        trigger = "max-wait"
        reason = "oldest parked ticket reached staging_max_wait_seconds"
    else:
        trigger = None
        reason = "no trigger fired"

    decision = WeaveDecision(
        should_weave=should_weave,
        trigger=trigger,
        quiescent_count=quiescent_count,
        reason=reason,
    )
    return decision, next_prior_parked


def cadences_from_charter(charter_resolved: Mapping[str, Any]) -> Cadences:
    """Extract `[loop.cadences]` from an already-resolved charter mapping.

    A RESOLVED charter (``Charter.raw`` from `charter.load_charter`) already
    carries `staging_quiescent_tickets`/`staging_max_wait_seconds` via
    `charter.py`'s own in-code defaulting — this function's fallback
    constants exist only as a defensive last resort for a malformed/partial
    mapping, not as a second source of truth to keep in sync by hand.
    """
    loop_cfg = charter_resolved.get("loop") if isinstance(charter_resolved, Mapping) else None
    cadences_cfg = loop_cfg.get("cadences") if isinstance(loop_cfg, Mapping) else None
    if not isinstance(cadences_cfg, Mapping):
        cadences_cfg = {}

    quiescent = cadences_cfg.get("staging_quiescent_tickets", _DEFAULT_STAGING_QUIESCENT_TICKETS)
    max_wait = cadences_cfg.get("staging_max_wait_seconds", _DEFAULT_STAGING_MAX_WAIT_SECONDS)
    return Cadences(
        staging_quiescent_tickets=quiescent,
        staging_max_wait_seconds=max_wait,
    )


def build_loop_state(store: Any, *, now: float | None = None) -> LoopState:
    """Query a `RigStore` for the current `LoopState` (the only
    store-touching function in this module; the daemon calls this once per
    poll, then hands the result to :func:`evaluate_triggers`).

    ``now`` is accepted for signature symmetry with the rest of this
    module's API (mirroring `statemachine.transition`'s unused `now`) but
    is not consulted here — the parked snapshot is read as-is from the
    store regardless of the caller's notion of "now".
    """
    del now  # reserved, not yet consulted (see docstring)
    parked = tuple(
        ParkedTicket(ticket_id=t["id"], parked_since=t["updated_at"])
        for t in store.list_tickets(state=PARKED)
    )
    active_count = sum(len(store.list_tickets(state=s)) for s in ACTIVE_STATES)
    return LoopState(parked=parked, active_count=active_count)
