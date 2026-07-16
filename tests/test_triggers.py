"""Tests for stigmergy.triggers (SPEC.md §9 "Weave triggers (OR-ed — fixes
the one-ticket deadlock)", §5 `[loop.cadences]`, §10 AC13).

Case numbering below matches the bead .19 build spec's frozen case list
(build spec §2, cases 1-12) exactly — one test per case, same intent.
"""

from __future__ import annotations

from pathlib import Path

from stigmergy.rig import RigStore
from stigmergy.statemachine import CLAIMED, ELIGIBLE, IN_FLIGHT, LANDED, PARKED, POOL, TIER1_GREEN
from stigmergy.triggers import (
    Cadences,
    LoopState,
    ParkedTicket,
    build_loop_state,
    evaluate_triggers,
)

# --- helpers -----------------------------------------------------------------


def parked_of(*tickets: tuple[str, float]) -> tuple[ParkedTicket, ...]:
    return tuple(ParkedTicket(ticket_id=tid, parked_since=since) for tid, since in tickets)


# case 1 -----------------------------------------------------------------------


def test_quiescent_fires_at_threshold():
    """3 quiescent (present-unchanged) parked tickets reach threshold 3;
    active work present (queue-drained off) and well within max-wait
    (max-wait off) -> only the quiescent trigger can fire."""
    now = 1000.0
    parked = parked_of(("t1", 900.0), ("t2", 900.0), ("t3", 900.0))
    prior_parked = {"t1": 900.0, "t2": 900.0, "t3": 900.0}
    state = LoopState(parked=parked, active_count=1)
    cadences = Cadences(staging_quiescent_tickets=3, staging_max_wait_seconds=7200.0)

    decision, _next_prior = evaluate_triggers(state, cadences, prior_parked, now=now)

    assert decision.should_weave is True
    assert decision.trigger == "quiescent"
    assert decision.quiescent_count == 3


# case 2 -----------------------------------------------------------------------


def test_quiescent_below_threshold_no_fire():
    """Only 2 quiescent parked tickets against a threshold of 3; active
    work present and within max-wait -> nothing fires."""
    now = 1000.0
    parked = parked_of(("t1", 900.0), ("t2", 900.0))
    prior_parked = {"t1": 900.0, "t2": 900.0}
    state = LoopState(parked=parked, active_count=1)
    cadences = Cadences(staging_quiescent_tickets=3, staging_max_wait_seconds=7200.0)

    decision, _next_prior = evaluate_triggers(state, cadences, prior_parked, now=now)

    assert decision.should_weave is False
    assert decision.quiescent_count == 2


# case 3 -----------------------------------------------------------------------


def test_quiescent_requires_unchanged_across_poll():
    """A parked ticket ABSENT from prior_parked (just parked this poll) is
    not quiescent; one whose parked_since DIFFERS from prior_parked is not
    quiescent either. With threshold 1, active work present (queue-drained
    off) and within max-wait: the first poll (empty prior) does not fire;
    feeding the returned tracker back into an identical second call makes
    it fire (exercises the cross-poll quiescence mechanism)."""
    now = 1000.0
    cadences = Cadences(staging_quiescent_tickets=1, staging_max_wait_seconds=7200.0)

    # Sub-case: ticket freshly parked, absent from prior_parked.
    state = LoopState(parked=parked_of(("t1", 1000.0)), active_count=1)
    decision, next_prior = evaluate_triggers(state, cadences, {}, now=now)
    assert decision.quiescent_count == 0
    assert decision.should_weave is False

    # Sub-case: ticket present in prior_parked but with a DIFFERENT
    # parked_since (i.e. it re-parked since the prior poll) -> not
    # quiescent.
    state_changed = LoopState(parked=parked_of(("t1", 1000.0)), active_count=1)
    decision_changed, _ = evaluate_triggers(
        state_changed, cadences, {"t1": 900.0}, now=now
    )
    assert decision_changed.quiescent_count == 0
    assert decision_changed.should_weave is False

    # Cross-poll mechanism: feed the tracker returned by the FIRST poll
    # (above) back in as prior_parked for an identical second poll -> t1
    # is now present-and-unchanged -> quiescent -> fires.
    decision2, _ = evaluate_triggers(state, cadences, next_prior, now=now)
    assert decision2.quiescent_count == 1
    assert decision2.should_weave is True
    assert decision2.trigger == "quiescent"


# case 4 -----------------------------------------------------------------------


def test_queue_drained_single_ticket_fires():
    """AC13 rev-1 DEADLOCK REGRESSION TEST.

    A single parked ticket with a drained queue (active_count == 0) must
    weave IMMEDIATELY via the queue-drained trigger — it must NEVER wait
    for the quiescent threshold (staging_quiescent_tickets=3), which one
    lone ticket can structurally never reach on its own. Before this
    trigger existed (rev-1), a rig that parked exactly one ticket and then
    drained its queue would sit forever waiting for 2 more quiescent
    tickets that would never arrive — a permanent deadlock. This test
    pins the fix: it must fire on the very first poll, with an EMPTY prior
    (so quiescent_count == 0) and well within max-wait.
    """
    now = 1000.0
    state = LoopState(parked=parked_of(("t1", 990.0)), active_count=0)
    cadences = Cadences(staging_quiescent_tickets=3, staging_max_wait_seconds=7200.0)

    decision, _next_prior = evaluate_triggers(state, cadences, {}, now=now)

    assert decision.should_weave is True
    assert decision.trigger == "queue-drained"
    assert decision.quiescent_count == 0


# case 5 -----------------------------------------------------------------------


def test_queue_drained_requires_a_parked_ticket():
    """A fully drained queue (active_count == 0) with ZERO parked tickets
    must never weave — there is nothing to gate."""
    now = 1000.0
    state = LoopState(parked=(), active_count=0)
    cadences = Cadences(staging_quiescent_tickets=3, staging_max_wait_seconds=7200.0)

    decision, _next_prior = evaluate_triggers(state, cadences, {}, now=now)

    assert decision.should_weave is False
    assert decision.trigger is None


# case 6 -----------------------------------------------------------------------


def test_not_drained_when_active_work_present():
    """Active work still exists (active_count > 0): the queue-drained
    trigger must not fire even with a parked ticket sitting below the
    quiescent threshold and within max-wait."""
    now = 1000.0
    state = LoopState(parked=parked_of(("t1", 990.0)), active_count=2)
    cadences = Cadences(staging_quiescent_tickets=3, staging_max_wait_seconds=7200.0)

    decision, _next_prior = evaluate_triggers(state, cadences, {}, now=now)

    assert decision.should_weave is False


# case 7 -----------------------------------------------------------------------


def test_max_wait_fires_on_old_parked():
    """The oldest (only) parked ticket's age has reached
    staging_max_wait_seconds: max-wait must fire, with active work present
    (queue-drained off) and quiescent not satisfied (empty prior)."""
    now = 1000.0
    cadences = Cadences(staging_quiescent_tickets=3, staging_max_wait_seconds=100.0)
    state = LoopState(parked=parked_of(("t1", 900.0)), active_count=1)  # age == 100

    decision, _next_prior = evaluate_triggers(state, cadences, {}, now=now)

    assert decision.should_weave is True
    assert decision.trigger == "max-wait"


# case 8 -----------------------------------------------------------------------


def test_max_wait_uses_oldest_parked():
    """With two parked tickets, max-wait fires on the OLDEST one's age even
    though the other is fresh. Control: when BOTH are fresh (neither
    reaches the threshold), nothing fires."""
    now = 1000.0
    cadences = Cadences(staging_quiescent_tickets=3, staging_max_wait_seconds=100.0)

    old_and_fresh = LoopState(
        parked=parked_of(("old", 900.0), ("fresh", 990.0)), active_count=1
    )
    decision, _next_prior = evaluate_triggers(old_and_fresh, cadences, {}, now=now)
    assert decision.should_weave is True
    assert decision.trigger == "max-wait"

    both_fresh = LoopState(
        parked=parked_of(("a", 950.0), ("b", 990.0)), active_count=1
    )
    control_decision, _ = evaluate_triggers(both_fresh, cadences, {}, now=now)
    assert control_decision.should_weave is False


# case 9 -----------------------------------------------------------------------


def test_no_parked_never_weaves():
    """Zero parked tickets never weave, regardless of active_count being
    zero (drained) or positive (busy) — nothing to gate either way."""
    cadences = Cadences(staging_quiescent_tickets=3, staging_max_wait_seconds=7200.0)

    drained = LoopState(parked=(), active_count=0)
    decision_drained, _ = evaluate_triggers(drained, cadences, {}, now=1000.0)
    assert decision_drained.should_weave is False

    busy = LoopState(parked=(), active_count=5)
    decision_busy, _ = evaluate_triggers(busy, cadences, {}, now=1000.0)
    assert decision_busy.should_weave is False


# case 10 ----------------------------------------------------------------------


def test_next_tracker_carries_current_parked():
    """The returned next_prior_parked equals {id: parked_since} for
    exactly the current parked set; feeding it back as prior on an
    unchanged next call makes those tickets quiescent."""
    now = 1000.0
    cadences = Cadences(staging_quiescent_tickets=2, staging_max_wait_seconds=7200.0)
    state = LoopState(parked=parked_of(("a", 950.0), ("b", 960.0)), active_count=1)

    decision, next_prior = evaluate_triggers(state, cadences, {}, now=now)
    assert next_prior == {"a": 950.0, "b": 960.0}
    assert decision.quiescent_count == 0

    decision2, _ = evaluate_triggers(state, cadences, next_prior, now=now)
    assert decision2.quiescent_count == 2
    assert decision2.should_weave is True
    assert decision2.trigger == "quiescent"


# case 11 ----------------------------------------------------------------------


def test_build_loop_state_from_store(tmp_path: Path):
    """build_loop_state, backed by a real RigStore: parked list matches the
    parked tickets (id + updated_at); active_count counts only tickets in
    ACTIVE_STATES (pool/eligible/claimed/in_flight/tier1_green) — a landed
    ticket must not be counted."""
    store = RigStore.create(tmp_path / "tickets.db")
    try:
        store.add_ticket(id="p1", title="Parked 1", state=PARKED)
        store.add_ticket(id="p2", title="Parked 2", state=PARKED)
        store.add_ticket(id="pool1", title="Pool 1", state=POOL)
        store.add_ticket(id="elig1", title="Eligible 1", state=ELIGIBLE)
        store.add_ticket(id="claimed1", title="Claimed 1", state=CLAIMED)
        store.add_ticket(id="inflight1", title="In flight 1", state=IN_FLIGHT)
        store.add_ticket(id="t1green1", title="Tier1 green 1", state=TIER1_GREEN)
        store.add_ticket(id="landed1", title="Landed 1", state=LANDED)

        loop_state = build_loop_state(store)

        expected_parked_rows = {t["id"]: t for t in store.list_tickets(state=PARKED)}
        assert {p.ticket_id for p in loop_state.parked} == {"p1", "p2"}
        for p in loop_state.parked:
            assert p.parked_since == expected_parked_rows[p.ticket_id]["updated_at"]

        # pool1, elig1, claimed1, inflight1, t1green1 == 5 active tickets;
        # landed1 must not be counted.
        assert loop_state.active_count == 5
    finally:
        store.close()


# case 12 ----------------------------------------------------------------------


def test_trigger_label_priority_when_multiple_fire():
    """When multiple triggers hold simultaneously, the label names the
    highest-priority one (quiescent > queue-drained > max-wait), while
    should_weave stays the OR of all of them."""
    now = 1000.0

    # quiescent AND queue-drained both hold -> label is "quiescent".
    cadences_a = Cadences(staging_quiescent_tickets=1, staging_max_wait_seconds=7200.0)
    state_a = LoopState(parked=parked_of(("t1", 990.0)), active_count=0)
    decision_a, _ = evaluate_triggers(state_a, cadences_a, {"t1": 990.0}, now=now)
    assert decision_a.should_weave is True
    assert decision_a.trigger == "quiescent"

    # queue-drained AND max-wait both hold (quiescent does not: empty
    # prior) -> label is "queue-drained".
    cadences_b = Cadences(staging_quiescent_tickets=3, staging_max_wait_seconds=100.0)
    state_b = LoopState(parked=parked_of(("t1", 900.0)), active_count=0)
    decision_b, _ = evaluate_triggers(state_b, cadences_b, {}, now=now)
    assert decision_b.should_weave is True
    assert decision_b.trigger == "queue-drained"
