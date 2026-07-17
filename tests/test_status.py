"""Tests for stigmergy.status (SPEC.md §9 "Notifications", §10 AC12, OQ3;
bead .23 build spec §2).

Case numbering below matches the bead .23 build spec's frozen case list
(build spec §3 "tests/test_status.py", cases 8-15 — notify.py's tests
occupy cases 1-7 of the same build-spec case list).

Governing invariant under test: spend is RECONSTRUCTED read-only from the
event log (never a live `SpendLeash` — this module doesn't even import
it), and every AC12 field (state counts, spend vs leashes, heartbeat age,
queue age, escalated-unnotified, disk headroom) is actually populated by
`gather_status`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from stigmergy.notify import NotificationStore
from stigmergy.records import EventType, RecordPlane, make_event
from stigmergy.rig import RigStore
from stigmergy.statemachine import STATES
from stigmergy.status import (
    gather_status,
    reconstruct_spend,
    render_ticket_detail,
    render_ticket_list,
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


@pytest.fixture
def notification_store(tmp_path: Path) -> NotificationStore:
    return NotificationStore(tmp_path / "records" / "notifications.jsonl")


CHARTER_RESOLVED: dict[str, Any] = {
    "loop": {"budgets": {"dispatches": 50, "usd": 25.0, "gate_calls": 30}}
}


def common_fields(**overrides: Any) -> dict[str, Any]:
    """A complete, valid set of SPEC §8 common fields (mirrors
    test_records.py's helper of the same name)."""
    base: dict[str, Any] = {
        "rig": "shipyard",
        "ticket": "workspace-e2uh.23",
        "dispatch_id": "dispatch-0001",
        "attempt": 1,
        "attempt_kind": "initial",
        "rung": "cheap",
        "worker": "worker-haiku-code01-broom-casino-flock",
        "charter_hash": "charterhash123",
        "approval_hash": "approvalhash456",
        "image_digest": "sha256:deadbeef",
        "model": "haiku",
        "model_version": "haiku-3-5-20241022",
        "price_table_version": "modelshash789",
        "tokens": {"in": 100, "cached": 0, "out": 50, "reasoning": 0},
        "computed_usd": 0.0123,
        "wall_time_seconds": 12.5,
    }
    base.update(overrides)
    return base


def make_dispatch_event(**overrides: Any):
    fields = common_fields(prompt_artifact_hash="code01-hashabc", **overrides)
    return make_event(EventType.DISPATCH, **fields)


def make_gate_event(**overrides: Any):
    fields = common_fields(
        prompt_artifact_hash="critic01-hashdef",
        decoding_params={"temperature": 0.0},
        **overrides,
    )
    return make_event(EventType.GATE, **fields)


def make_disposition_event(**overrides: Any):
    fields = common_fields(computed_usd=0.0, **overrides)
    fields.setdefault("disposition", "landed")
    return make_event(EventType.DISPOSITION, **fields)


def add_ticket(store: RigStore, ticket_id: str, *, state: str = "pool", **fields: Any) -> None:
    store.add_ticket(id=ticket_id, title=f"Ticket {ticket_id}", state=state, **fields)


# --- case 8: test_gather_status_all_ac12_fields_present ---------------------


def test_gather_status_all_ac12_fields_present(
    store: RigStore, plane: RecordPlane, notification_store: NotificationStore
) -> None:
    add_ticket(store, "t-pool", state="pool")
    add_ticket(store, "t-eligible", state="eligible")
    add_ticket(store, "t-parked", state="parked")
    add_ticket(store, "t-landed", state="landed")
    add_ticket(store, "t-escalated", state="escalated")

    plane.append(make_dispatch_event(dispatch_id="dispatch-0001"))
    plane.append(make_gate_event(dispatch_id="dispatch-0001"))
    plane.append(make_disposition_event(dispatch_id="dispatch-0001", disposition="landed"))

    now = time.time()
    store.set_meta("daemon_heartbeat_at", str(now - 15))

    notification_store.record_intent(
        ticket="t-escalated",
        kind="escalation",
        title="escalated",
        message="ladder exhausted",
        now=now,
    )

    status = gather_status(
        store=store,
        record_plane=plane,
        notification_store=notification_store,
        charter_resolved=CHARTER_RESOLVED,
        disk_path=str(store.db_path.parent),
        now=now,
    )

    # Every statemachine state present in state_counts, including the
    # AC12 set (pool/eligible/claimed/parked/landed/escalated).
    assert set(status.state_counts) == set(STATES)
    for ac12_state in ("pool", "eligible", "claimed", "parked", "landed", "escalated"):
        assert ac12_state in status.state_counts
    assert status.state_counts["pool"] == 1
    assert status.state_counts["eligible"] == 1
    assert status.state_counts["parked"] == 1
    assert status.state_counts["landed"] == 1
    assert status.state_counts["escalated"] == 1
    assert status.state_counts["claimed"] == 0

    assert len(status.last_verdicts) == 1
    assert status.last_verdicts[0]["ticket"] == "workspace-e2uh.23"
    assert status.last_verdicts[0]["disposition"] == "landed"

    assert status.spend["dispatches_used"] == 1
    assert status.spend["dispatches_cap"] == 50
    assert status.spend["gate_calls_used"] == 1
    assert status.spend["gate_calls_cap"] == 30
    assert status.spend["usd_cap"] == 25.0
    assert isinstance(status.spend["metered_spent"], float)

    assert status.daemon_heartbeat_age == pytest.approx(15.0, abs=1.0)
    assert status.queue_age is not None
    assert status.escalated_unnotified >= 1
    assert status.disk_free_bytes > 0
    assert status.disk_ok is True


# --- case 9: test_reconstruct_spend_from_events -----------------------------


def test_reconstruct_spend_from_events(plane: RecordPlane) -> None:
    plane.append(make_dispatch_event(dispatch_id="dispatch-0001", computed_usd=1.5))
    plane.append(make_dispatch_event(dispatch_id="dispatch-0002", computed_usd="unbudgetable"))
    plane.append(make_gate_event(dispatch_id="dispatch-0003", computed_usd=0.25))

    spend = reconstruct_spend(plane, usd_cap=25.0, dispatches_cap=50, gate_calls_cap=30)

    assert spend["metered_spent"] == pytest.approx(1.75)
    assert spend["dispatches_used"] == 2
    assert spend["gate_calls_used"] == 1
    assert spend["unbudgetable_events"] == 1
    assert spend["usd_cap"] == 25.0
    assert spend["dispatches_cap"] == 50
    assert spend["gate_calls_cap"] == 30


# --- case 10: test_escalated_unnotified_counts_pending_escalations ----------


def test_escalated_unnotified_counts_pending_escalations(
    store: RigStore, plane: RecordPlane, notification_store: NotificationStore
) -> None:
    now = time.time()

    pending_escalation = notification_store.record_intent(
        ticket="t1", kind="escalation", title="esc1", message="m1", now=now
    )
    delivered_escalation = notification_store.record_intent(
        ticket="t2", kind="escalation", title="esc2", message="m2", now=now
    )
    notification_store.mark_delivered(delivered_escalation.intent_id, now=now)
    notification_store.record_intent(
        ticket="t3", kind="budget-exhausted", title="budget", message="m3", now=now
    )

    status = gather_status(
        store=store,
        record_plane=plane,
        notification_store=notification_store,
        charter_resolved=CHARTER_RESOLVED,
        disk_path=str(store.db_path.parent),
        now=now,
    )

    assert status.escalated_unnotified == 1
    # Sanity: the pending escalation counted is the right one.
    remaining_pending = [i for i in notification_store.pending() if i.kind == "escalation"]
    assert [i.intent_id for i in remaining_pending] == [pending_escalation.intent_id]


# --- case 11: test_heartbeat_age_none_when_unset ----------------------------


def test_heartbeat_age_none_when_unset(
    store: RigStore, plane: RecordPlane, notification_store: NotificationStore
) -> None:
    now = time.time()

    status_unset = gather_status(
        store=store,
        record_plane=plane,
        notification_store=notification_store,
        charter_resolved=CHARTER_RESOLVED,
        disk_path=str(store.db_path.parent),
        now=now,
    )
    assert status_unset.daemon_heartbeat_age is None

    store.set_meta("daemon_heartbeat_at", str(now - 30))

    status_set = gather_status(
        store=store,
        record_plane=plane,
        notification_store=notification_store,
        charter_resolved=CHARTER_RESOLVED,
        disk_path=str(store.db_path.parent),
        now=now,
    )
    assert status_set.daemon_heartbeat_age == pytest.approx(30.0, abs=1.0)


# --- case 12: test_queue_age_from_oldest_pooled_ticket ----------------------


def test_queue_age_from_oldest_pooled_ticket(
    store: RigStore, plane: RecordPlane, notification_store: NotificationStore
) -> None:
    real_now = time.time()

    # Empty queue (only a non-queue-state ticket exists) -> None.
    add_ticket(store, "t-landed", state="landed")
    status_empty = gather_status(
        store=store,
        record_plane=plane,
        notification_store=notification_store,
        charter_resolved=CHARTER_RESOLVED,
        disk_path=str(store.db_path.parent),
        now=real_now,
    )
    assert status_empty.queue_age is None

    add_ticket(store, "t-pool", state="pool")
    add_ticket(store, "t-eligible", state="eligible")

    far_future = real_now + 10_000.0
    status = gather_status(
        store=store,
        record_plane=plane,
        notification_store=notification_store,
        charter_resolved=CHARTER_RESOLVED,
        disk_path=str(store.db_path.parent),
        now=far_future,
    )

    # Oldest pool/eligible ticket's created_at is within a couple seconds
    # of real_now (both were just inserted), so queue_age should be ~10000s.
    assert status.queue_age == pytest.approx(10_000.0, abs=5.0)


# --- case 13: test_disk_headroom_reported -----------------------------------


def test_disk_headroom_reported(
    store: RigStore, plane: RecordPlane, notification_store: NotificationStore, tmp_path: Path
) -> None:
    import shutil

    now = time.time()
    real_free = shutil.disk_usage(str(tmp_path)).free

    status_no_min = gather_status(
        store=store,
        record_plane=plane,
        notification_store=notification_store,
        charter_resolved=CHARTER_RESOLVED,
        disk_path=str(tmp_path),
        min_disk_bytes=None,
        now=now,
    )
    assert status_no_min.disk_free_bytes > 0
    assert status_no_min.disk_ok is True

    status_below = gather_status(
        store=store,
        record_plane=plane,
        notification_store=notification_store,
        charter_resolved=CHARTER_RESOLVED,
        disk_path=str(tmp_path),
        min_disk_bytes=real_free - 1024,
        now=now,
    )
    assert status_below.disk_ok is True

    status_above = gather_status(
        store=store,
        record_plane=plane,
        notification_store=notification_store,
        charter_resolved=CHARTER_RESOLVED,
        disk_path=str(tmp_path),
        min_disk_bytes=real_free + (10 * 1024 ** 4),  # absurdly large, must exceed free
        now=now,
    )
    assert status_above.disk_ok is False


# --- case 14: test_render_status_contains_ac12_fields -----------------------


def test_render_status_contains_ac12_fields(
    store: RigStore, plane: RecordPlane, notification_store: NotificationStore
) -> None:
    from stigmergy.status import render_status

    add_ticket(store, "t-pool", state="pool")
    add_ticket(store, "t-parked", state="parked")
    add_ticket(store, "t-landed", state="landed")
    add_ticket(store, "t-escalated", state="escalated")

    plane.append(make_dispatch_event(dispatch_id="dispatch-0001"))
    plane.append(make_gate_event(dispatch_id="dispatch-0002"))

    now = time.time()
    store.set_meta("daemon_heartbeat_at", str(now - 5))
    notification_store.record_intent(
        ticket="t-escalated", kind="escalation", title="esc", message="m", now=now
    )

    status = gather_status(
        store=store,
        record_plane=plane,
        notification_store=notification_store,
        charter_resolved=CHARTER_RESOLVED,
        disk_path=str(store.db_path.parent),
        now=now,
    )

    text = render_status(status)

    for ac12_state in ("pool", "eligible", "claimed", "parked", "landed", "escalated"):
        assert ac12_state in text
    assert "usd" in text
    assert "dispatches" in text
    assert "gate_calls" in text
    assert "heartbeat" in text
    assert "queue_age" in text
    assert "escalated_unnotified" in text
    assert "disk" in text


# --- case 15: test_render_ticket_list_and_detail ----------------------------


def test_render_ticket_list_and_detail(store: RigStore) -> None:
    add_ticket(store, "t-1", state="pool")
    store.add_ticket(id="t-2", title="Second ticket", state="eligible", difficulty="medium")

    listing = render_ticket_list(store)
    assert "t-1" in listing
    assert "[pool]" in listing
    assert "Ticket t-1" in listing
    assert "t-2" in listing
    assert "[eligible]" in listing
    assert "Second ticket" in listing

    detail = render_ticket_detail(store, "t-2")
    assert "Second ticket" in detail
    assert "eligible" in detail
    assert "medium" in detail

    missing = render_ticket_detail(store, "no-such-ticket-id")
    assert "no such ticket" in missing
    assert "no-such-ticket-id" in missing


# --- D14: untriaged filing count (bead workspace-e2uh.38, AC14 case 8) ------


def test_untriaged_filings_count_in_status_and_render(store, plane, notification_store) -> None:
    from stigmergy.status import render_status

    for n in (1, 2):
        store.add_filed_ticket(
            id=f"filed-d1-{n}",
            title=f"t{n}",
            description="d",
            origin_role="worker",
            origin_worker="w",
            origin_dispatch_id="d1",
            origin_parent_ticket="p",
            discovered_from="d1@p",
            proposal_hash=f"h{n}",
        )

    status = gather_status(
        store=store,
        record_plane=plane,
        notification_store=notification_store,
        charter_resolved=CHARTER_RESOLVED,
        disk_path=str(store.db_path.parent),
        now=1000.0,
    )
    assert status.untriaged_filings == 2

    text = render_status(status)
    assert "untriaged_filings" in text
    # value is text-labelled (deuteranopia: never colour-only).
    assert "untriaged_filings: 2" in text


def test_untriaged_filings_zero_when_none_filed(store, plane, notification_store) -> None:
    status = gather_status(
        store=store,
        record_plane=plane,
        notification_store=notification_store,
        charter_resolved=CHARTER_RESOLVED,
        disk_path=str(store.db_path.parent),
        now=1000.0,
    )
    assert status.untriaged_filings == 0
