"""Tests for stigmergy.statemachine (SPEC.md §9 "State machine (normative)"
+ "Failure classes" + "Retry semantics", §10 AC7/AC9).

Case numbering below matches the bead .16 design doc's exact case list
(design §3, cases 1-24):

Transition matrix / legality (1-6), retry decisions (7-19),
apply_decision (20-22), disposition journaling (23-24).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from stigmergy.records import ATTEMPT_KINDS, RecordError, RecordPlane
from stigmergy.rig import RigStore
from stigmergy.statemachine import (
    CLAIMED,
    DONE,
    ELIGIBLE,
    ESCALATED,
    FAILED,
    GATED,
    IN_FLIGHT,
    LANDED,
    LEGAL_TRANSITIONS,
    PARKED,
    POOL,
    REJECTED,
    STATES,
    TIER1_GREEN,
    AttemptDecision,
    FailureClass,
    IllegalTransition,
    apply_decision,
    decide_retry,
    is_legal,
    record_disposition,
    transition,
)

# --- fixtures / helpers ------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> RigStore:
    s = RigStore.create(tmp_path / "tickets.db")
    yield s
    s.close()


@pytest.fixture
def plane(tmp_path: Path) -> RecordPlane:
    return RecordPlane(tmp_path / "records")


def add_ticket(store: RigStore, ticket_id: str, *, state: str = POOL, **fields: Any) -> None:
    store.add_ticket(id=ticket_id, title=f"Ticket {ticket_id}", state=state, **fields)


LADDER = ["cheap", "default", "exquisite"]
ATTEMPTS_PER_RUNG = 3
INTEGRATION_FAILURES_CAP = 2


def row(
    *, attempts_used: int = 0, integration_failures: int = 0, current_rung: str | None = "cheap"
) -> dict[str, Any]:
    return {
        "attempts_used": attempts_used,
        "integration_failures": integration_failures,
        "current_rung": current_rung,
    }


def decide(row_dict: dict[str, Any], failure_class: FailureClass) -> AttemptDecision:
    return decide_retry(
        row_dict,
        failure_class,
        ladder=LADDER,
        attempts_per_rung=ATTEMPTS_PER_RUNG,
        integration_failures_cap=INTEGRATION_FAILURES_CAP,
    )


def disposition_ctx(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "rig": "shipyard",
        "ticket": "workspace-e2uh.16",
        "dispatch_id": "dispatch-0001",
        "attempt": 1,
        "rung": "cheap",
        "worker": "worker-haiku-code01-broom-casino-flock",
        "charter_hash": "charterhash123",
        "approval_hash": "approvalhash456",
        "image_digest": "sha256:deadbeef",
        "model": "haiku",
        "model_version": "haiku-3-5-20241022",
        "price_table_version": "modelshash789",
    }
    base.update(overrides)
    return base


# The transition matrix per design §1.2, hand-transcribed here as an
# INDEPENDENT oracle (never sourced from `LEGAL_TRANSITIONS`/`is_legal`
# themselves) so both cases 1 and 2 actually check the implementation
# against the frozen spec rather than against its own code.
_EXPECTED_LEGAL_TRANSITIONS: dict[str, set[str]] = {
    POOL: {ELIGIBLE, CLAIMED},
    ELIGIBLE: {CLAIMED, POOL},
    CLAIMED: {IN_FLIGHT, POOL},
    IN_FLIGHT: {TIER1_GREEN, FAILED, POOL},
    TIER1_GREEN: {PARKED},
    PARKED: {GATED},
    GATED: {LANDED, REJECTED, PARKED},
    FAILED: {POOL, ESCALATED},
    REJECTED: {POOL, ESCALATED},
    LANDED: {DONE},
    ESCALATED: set(),
    DONE: set(),
}


def _expected_legal(from_state: str, to_state: str) -> bool:
    """Independent oracle for legality, built from
    `_EXPECTED_LEGAL_TRANSITIONS` only -- never from `is_legal`/
    `LEGAL_TRANSITIONS` -- so case 2 cannot pass merely because the
    implementation is self-consistent with itself."""
    return to_state in _EXPECTED_LEGAL_TRANSITIONS.get(from_state, set())


# --- 1. test_legal_transitions_matrix_exact ---------------------------------


def test_legal_transitions_matrix_exact() -> None:
    assert set(LEGAL_TRANSITIONS.keys()) == STATES

    for state, expected_targets in _EXPECTED_LEGAL_TRANSITIONS.items():
        assert LEGAL_TRANSITIONS[state] == frozenset(expected_targets), state


# --- 2. test_full_illegal_transition_matrix ---------------------------------


def test_full_illegal_transition_matrix(store: RigStore) -> None:
    for from_state in STATES:
        for to_state in STATES:
            ticket_id = f"t-{from_state}-{to_state}"
            add_ticket(store, ticket_id, state=from_state)

            # Cross-check the module's own `is_legal` agrees with the
            # independent oracle (guards against `is_legal` and
            # `LEGAL_TRANSITIONS` silently drifting apart from each other).
            assert is_legal(from_state, to_state) == _expected_legal(from_state, to_state)

            if _expected_legal(from_state, to_state):
                result = transition(store, ticket_id, to_state)
                assert result == {"ticket": ticket_id, "from": from_state, "to": to_state}
                assert store.get_ticket(ticket_id)["state"] == to_state
            else:
                with pytest.raises(IllegalTransition):
                    transition(store, ticket_id, to_state)
                # No partial write on a rejected transition.
                assert store.get_ticket(ticket_id)["state"] == from_state


# --- 3. test_transition_expected_from_guard ---------------------------------


def test_transition_expected_from_guard(store: RigStore) -> None:
    add_ticket(store, "t-guard", state=CLAIMED)

    # in_flight would otherwise be a legal destination from claimed, but the
    # expected_from guard names a state the ticket isn't actually in.
    with pytest.raises(IllegalTransition):
        transition(store, "t-guard", IN_FLIGHT, expected_from=POOL)

    # The ticket is untouched by the failed guard check.
    assert store.get_ticket("t-guard")["state"] == CLAIMED

    # The correct expected_from lets the same edge through.
    transition(store, "t-guard", IN_FLIGHT, expected_from=CLAIMED)
    assert store.get_ticket("t-guard")["state"] == IN_FLIGHT


# --- 4. test_transition_to_pool_clears_lease --------------------------------


def test_transition_to_pool_clears_lease(store: RigStore) -> None:
    add_ticket(
        store,
        "t-clear",
        state=IN_FLIGHT,
        lease_owner="worker-1",
        lease_dispatch_id="dispatch-1",
        lease_expires_at=1000.0,
        lease_heartbeat_at=999.0,
        attempts_used=2,
        integration_failures=1,
        current_rung="default",
    )

    transition(store, "t-clear", POOL)

    row_after = store.get_ticket("t-clear")
    assert row_after["state"] == POOL
    assert row_after["lease_owner"] is None
    assert row_after["lease_dispatch_id"] is None
    assert row_after["lease_expires_at"] is None
    assert row_after["lease_heartbeat_at"] is None
    # Counters untouched by the happy-path state setter.
    assert row_after["attempts_used"] == 2
    assert row_after["integration_failures"] == 1
    assert row_after["current_rung"] == "default"


# --- 5. test_transition_missing_ticket_raises -------------------------------


def test_transition_missing_ticket_raises(store: RigStore) -> None:
    with pytest.raises(IllegalTransition):
        transition(store, "t-does-not-exist", POOL)


# --- 6. test_transition_does_not_touch_counters -----------------------------


def test_transition_does_not_touch_counters(store: RigStore) -> None:
    add_ticket(
        store,
        "t-counters",
        state=POOL,
        attempts_used=1,
        integration_failures=1,
        current_rung="default",
    )

    transition(store, "t-counters", CLAIMED)

    row_after = store.get_ticket("t-counters")
    assert row_after["state"] == CLAIMED
    assert row_after["attempts_used"] == 1
    assert row_after["integration_failures"] == 1
    assert row_after["current_rung"] == "default"


# --- 7. test_infra_consumes_nothing ------------------------------------------


def test_infra_consumes_nothing() -> None:
    input_row = row(attempts_used=1, integration_failures=1, current_rung="default")
    input_row_copy = dict(input_row)

    decision = decide(input_row, FailureClass.INFRA)

    assert decision.attempt_kind == "infra-retry"
    assert decision.next_state == POOL
    assert decision.rung == "default"
    assert decision.current_rung == "default"
    assert decision.attempts_used == 1
    assert decision.integration_failures == 1
    assert decision.escalation_reason is None
    assert decision.pre_apply_prior is False
    assert decision.prior_as_reference is False
    # Pure: the input row must be untouched.
    assert input_row == input_row_copy


# --- 8. test_tier1_fail_same_rung_retry -------------------------------------


def test_tier1_fail_same_rung_retry() -> None:
    input_row = row(attempts_used=0, current_rung="cheap")

    decision = decide(input_row, FailureClass.TIER1_FAIL)

    assert decision.attempts_used == 1
    assert decision.attempt_kind == "tier1-repair"
    assert decision.pre_apply_prior is True
    assert decision.prior_as_reference is False
    assert decision.next_state == POOL
    assert decision.rung == "cheap"
    assert decision.current_rung == "cheap"
    assert decision.escalation_reason is None


# --- 9. test_rejected_same_rung_retry ----------------------------------------


def test_rejected_same_rung_retry() -> None:
    input_row = row(attempts_used=0, current_rung="cheap")

    decision = decide(input_row, FailureClass.REJECTED)

    assert decision.attempt_kind == "critic-revision"
    assert decision.pre_apply_prior is True
    assert decision.prior_as_reference is False
    assert decision.attempts_used == 1
    assert decision.next_state == POOL


# --- 10. test_wedge_and_degenerate_clean_restart ----------------------------


@pytest.mark.parametrize("failure_class", [FailureClass.WEDGE, FailureClass.DEGENERATE])
def test_wedge_and_degenerate_clean_restart(failure_class: FailureClass) -> None:
    input_row = row(attempts_used=0, current_rung="cheap")

    decision = decide(input_row, failure_class)

    assert decision.attempt_kind == "clean-restart"
    assert decision.pre_apply_prior is False
    assert decision.prior_as_reference is True
    assert decision.attempts_used == 1
    assert decision.next_state == POOL


# --- 11. test_rung_exhaustion_steps_up ---------------------------------------


def test_rung_exhaustion_steps_up() -> None:
    # attempts_used=2 -> +1 = 3 == attempts_per_rung: exhausted, but "cheap"
    # is not the final rung, so this steps up rather than escalating.
    input_row = row(attempts_used=2, current_rung="cheap")

    decision = decide(input_row, FailureClass.TIER1_FAIL)

    assert decision.current_rung == "default"
    assert decision.rung == "default"
    assert decision.attempts_used == 0
    assert decision.attempt_kind == "stepup-initial"
    assert decision.next_state == POOL
    assert decision.pre_apply_prior is False
    assert decision.prior_as_reference is True
    assert decision.escalation_reason is None


# --- 12. test_ladder_exhaustion_escalates ------------------------------------


def test_ladder_exhaustion_escalates() -> None:
    input_row = row(attempts_used=2, current_rung="exquisite")

    decision = decide(input_row, FailureClass.TIER1_FAIL)

    assert decision.next_state == ESCALATED
    assert decision.escalation_reason == "ladder-exhausted"
    assert decision.attempts_used == 3
    assert decision.current_rung == "exquisite"


# --- 13. test_worst_case_eleven_dispatch_bound -------------------------------


def test_worst_case_eleven_dispatch_bound() -> None:
    """AC7 ladder walk (SPEC §9/§10 AC7): "one hard ticket floors itself in
    <=11 dispatches" == 3 rungs x 3 rung-consuming attempts (9) + the
    integration_failures cap (2). Drive both maximum-length walks and
    verify escalation happens EXACTLY at the floor in each dimension, and
    that they sum to the documented worst-case bound of 11.
    """
    # --- rung-consuming walk: cheap -> default -> exquisite, 3 attempts each
    walk_row = row(attempts_used=0, current_rung="cheap")
    rung_consuming_dispatches = 0
    escalated_at = None

    for _ in range(20):  # generous upper bound; the walk must terminate well before this
        decision = decide(walk_row, FailureClass.TIER1_FAIL)
        rung_consuming_dispatches += 1
        walk_row = row(
            attempts_used=decision.attempts_used,
            current_rung=decision.current_rung,
            integration_failures=walk_row["integration_failures"],
        )
        if decision.next_state == ESCALATED:
            escalated_at = rung_consuming_dispatches
            assert decision.escalation_reason == "ladder-exhausted"
            break
        # Must never escalate before all three rungs have been walked.
        assert decision.rung in LADDER

    assert escalated_at == 9  # 3 rungs * 3 attempts, exactly, not before/after
    assert walk_row["current_rung"] == LADDER[-1]

    # --- integration-failure walk: cap trips exactly at the 2nd failure ---
    integ_row = row(integration_failures=0, current_rung="cheap")
    integration_dispatches = 0
    integ_escalated_at = None

    for _ in range(20):
        decision = decide(integ_row, FailureClass.INTEGRATION_CONFLICT)
        integration_dispatches += 1
        integ_row = row(
            attempts_used=integ_row["attempts_used"],
            integration_failures=decision.integration_failures,
            current_rung=integ_row["current_rung"],
        )
        if decision.next_state == ESCALATED:
            integ_escalated_at = integration_dispatches
            assert decision.escalation_reason == "triage-failure"
            break

    assert integ_escalated_at == 2  # cap == 2, trips exactly on the 2nd, not the 1st

    # The documented worst-case bound: 9 rung-consuming + 2 integration.
    assert escalated_at + integ_escalated_at == 11


# --- 14. test_final_attempt_on_final_rung_is_clean_restart ------------------


def test_final_attempt_on_final_rung_is_clean_restart() -> None:
    # attempts_used=1 -> +1 = 2 == attempts_per_rung - 1: this is the LAST
    # allowed attempt on the final rung before ladder-exhaustion. A
    # TIER1_FAIL would normally be tier1-repair; the override forces
    # clean-restart instead (anchor decontamination, SPEC §9).
    input_row = row(attempts_used=1, current_rung="exquisite")

    decision = decide(input_row, FailureClass.TIER1_FAIL)

    assert decision.attempts_used == 2
    assert decision.next_state == POOL  # not yet exhausted -- this IS the retry
    assert decision.attempt_kind == "clean-restart"
    assert decision.pre_apply_prior is False
    assert decision.prior_as_reference is True


# --- 15. test_integration_conflict_increments_integration_failures_not_attempts


def test_integration_conflict_increments_integration_failures_not_attempts() -> None:
    input_row = row(attempts_used=1, integration_failures=0, current_rung="default")

    decision = decide(input_row, FailureClass.INTEGRATION_CONFLICT)

    assert decision.attempts_used == 1  # unchanged
    assert decision.integration_failures == 1  # 0 -> 1
    assert decision.attempt_kind == "integration-reconcile"
    assert decision.pre_apply_prior is False
    assert decision.prior_as_reference is True
    assert decision.next_state == POOL
    assert decision.rung == "default"


# --- 16. test_integration_regression_same_as_conflict -----------------------


def test_integration_regression_same_as_conflict() -> None:
    input_row = row(attempts_used=1, integration_failures=0, current_rung="default")

    conflict_decision = decide(dict(input_row), FailureClass.INTEGRATION_CONFLICT)
    regression_decision = decide(dict(input_row), FailureClass.INTEGRATION_REGRESSION)

    assert conflict_decision == regression_decision


# --- 17. test_integration_failure_cap_escalates_triage_failure --------------


def test_integration_failure_cap_escalates_triage_failure() -> None:
    input_row = row(integration_failures=1, current_rung="default")

    decision = decide(input_row, FailureClass.INTEGRATION_REGRESSION)

    assert decision.integration_failures == 2
    assert decision.next_state == ESCALATED
    assert decision.escalation_reason == "triage-failure"
    assert decision.attempt_kind == "integration-reconcile"


# --- 18. test_current_rung_none_defaults_to_first_ladder_entry --------------


def test_current_rung_none_defaults_to_first_ladder_entry() -> None:
    input_row = row(current_rung=None)

    decision = decide(input_row, FailureClass.INFRA)

    assert decision.rung == LADDER[0]
    assert decision.current_rung == LADDER[0]


# --- 19. test_all_returned_attempt_kinds_are_valid ---------------------------


def test_all_returned_attempt_kinds_are_valid() -> None:
    decisions = [
        decide(row(current_rung="cheap"), FailureClass.INFRA),
        decide(row(attempts_used=0, current_rung="cheap"), FailureClass.TIER1_FAIL),
        decide(row(attempts_used=0, current_rung="cheap"), FailureClass.REJECTED),
        decide(row(attempts_used=0, current_rung="cheap"), FailureClass.WEDGE),
        decide(row(attempts_used=0, current_rung="cheap"), FailureClass.DEGENERATE),
        # step-up
        decide(row(attempts_used=2, current_rung="cheap"), FailureClass.TIER1_FAIL),
        # ladder-exhausted
        decide(row(attempts_used=2, current_rung="exquisite"), FailureClass.TIER1_FAIL),
        decide(row(attempts_used=2, current_rung="exquisite"), FailureClass.WEDGE),
        decide(row(attempts_used=2, current_rung="exquisite"), FailureClass.REJECTED),
        # final-attempt override
        decide(row(attempts_used=1, current_rung="exquisite"), FailureClass.TIER1_FAIL),
        decide(row(integration_failures=0), FailureClass.INTEGRATION_CONFLICT),
        decide(row(integration_failures=0), FailureClass.INTEGRATION_REGRESSION),
        decide(row(integration_failures=1), FailureClass.INTEGRATION_CONFLICT),  # cap escalate
    ]
    for decision in decisions:
        assert decision.attempt_kind in ATTEMPT_KINDS, decision.attempt_kind


# --- 20. test_apply_decision_persists_counters_and_state --------------------


def test_apply_decision_persists_counters_and_state(store: RigStore) -> None:
    add_ticket(
        store,
        "t-apply",
        state=FAILED,
        attempts_used=0,
        current_rung="cheap",
        integration_failures=0,
    )
    input_row = row(attempts_used=0, current_rung="cheap")
    decision = decide(input_row, FailureClass.TIER1_FAIL)

    apply_decision(store, "t-apply", decision)

    row_after = store.get_ticket("t-apply")
    assert row_after["state"] == decision.next_state == POOL
    assert row_after["attempts_used"] == decision.attempts_used == 1
    assert row_after["current_rung"] == decision.current_rung == "cheap"
    assert row_after["integration_failures"] == decision.integration_failures == 0


# --- 21. test_apply_decision_escalation_moves_to_escalated ------------------


def test_apply_decision_escalation_moves_to_escalated(store: RigStore) -> None:
    add_ticket(
        store,
        "t-escalate",
        state=FAILED,
        attempts_used=2,
        current_rung="exquisite",
        integration_failures=0,
    )
    input_row = row(attempts_used=2, current_rung="exquisite")
    decision = decide(input_row, FailureClass.TIER1_FAIL)
    assert decision.next_state == ESCALATED

    apply_decision(store, "t-escalate", decision)

    row_after = store.get_ticket("t-escalate")
    assert row_after["state"] == ESCALATED
    assert row_after["attempts_used"] == 3
    assert row_after["current_rung"] == "exquisite"

    # Also exercise the REJECTED starting state.
    add_ticket(
        store,
        "t-escalate-rejected",
        state=REJECTED,
        attempts_used=2,
        current_rung="exquisite",
        integration_failures=0,
    )
    decision2 = decide(row(attempts_used=2, current_rung="exquisite"), FailureClass.REJECTED)
    apply_decision(store, "t-escalate-rejected", decision2)
    assert store.get_ticket("t-escalate-rejected")["state"] == ESCALATED


# --- 22. test_apply_decision_enforces_legal_state_edge ----------------------


def test_apply_decision_enforces_legal_state_edge(store: RigStore) -> None:
    # Ticket is in POOL, not FAILED/REJECTED -- POOL->ESCALATED is not a
    # legal edge, so apply_decision must raise and leave counters alone.
    add_ticket(
        store, "t-illegal-apply", state=POOL, attempts_used=1, current_rung="cheap"
    )
    decision = AttemptDecision(
        attempt_kind="tier1-repair",
        rung="cheap",
        next_state=ESCALATED,
        pre_apply_prior=False,
        prior_as_reference=False,
        escalation_reason="ladder-exhausted",
        attempts_used=3,
        current_rung="cheap",
        integration_failures=0,
    )

    with pytest.raises(IllegalTransition):
        apply_decision(store, "t-illegal-apply", decision)

    # No partial write: state and counters both untouched.
    row_after = store.get_ticket("t-illegal-apply")
    assert row_after["state"] == POOL
    assert row_after["attempts_used"] == 1


# --- 23. test_record_disposition_builds_valid_event -------------------------


def test_record_disposition_builds_valid_event(plane: RecordPlane) -> None:
    ctx = disposition_ctx()

    record_disposition(
        plane,
        ctx=ctx,
        disposition="landed",
        attempt_kind="initial",
        reason="critic approved",
    )

    events = plane.read_events()
    assert len(events) == 1
    event = events[0]
    assert event["event_type"] == "disposition"
    assert event["tokens"] == {"in": 0, "cached": 0, "out": 0, "reasoning": 0}
    assert event["computed_usd"] == 0.0
    assert event["wall_time_seconds"] == 0.0
    assert event["attempt_kind"] == "initial"
    assert event["disposition"] == "landed"
    assert event["reason"] == "critic approved"
    for key in ("rig", "ticket", "dispatch_id", "rung", "worker"):
        assert event[key] == ctx[key]


def test_record_disposition_reason_omitted_when_not_given(plane: RecordPlane) -> None:
    ctx = disposition_ctx(dispatch_id="dispatch-0002")

    record_disposition(plane, ctx=ctx, disposition="parked", attempt_kind="tier1-repair")

    events = plane.read_events()
    assert len(events) == 1
    assert "reason" not in events[0]


# --- 24. test_record_disposition_bad_attempt_kind_raises --------------------


def test_record_disposition_bad_attempt_kind_raises(plane: RecordPlane) -> None:
    ctx = disposition_ctx()

    with pytest.raises(RecordError):
        record_disposition(
            plane, ctx=ctx, disposition="landed", attempt_kind="not-a-real-attempt-kind"
        )

    # Nothing was appended for the rejected event.
    assert plane.read_events() == []
