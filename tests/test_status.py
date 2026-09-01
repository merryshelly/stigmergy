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
from stigmergy.quotagov import (
    QuotaGovernor,
    QuotaState,
    RollingWindow,
    decide,
)
from stigmergy.records import EventType, RecordPlane, make_event
from stigmergy.registry import ModelEntry, PricingClass, Registry
from stigmergy.rig import RigStore
from stigmergy.statemachine import STATES
from stigmergy.status import (
    gather_lane_parks,
    gather_status,
    reconstruct_spend,
    render_status,
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


# --- monitor command tests ---------------------------------------------------


def test_render_event_tail_empty_list() -> None:
    from stigmergy.status import render_event_tail

    result = render_event_tail([], 25)
    assert result == "(no events)"


def test_render_event_tail_bounded_to_n() -> None:
    from stigmergy.status import render_event_tail

    events = [
        {
            "event_type": "dispatch",
            "ticket": "t-1",
            "ts": 1000.0,
            "outcome": "succeeded",
        },
        {
            "event_type": "gate",
            "ticket": "t-2",
            "ts": 2000.0,
            "disposition": "approved",
        },
        {
            "event_type": "dispatch",
            "ticket": "t-3",
            "ts": 3000.0,
            "outcome": "failed",
        },
    ]

    result = render_event_tail(events, 2)
    lines = result.split("\n")
    assert len(lines) == 2
    assert "t-2" in lines[0]
    assert "t-3" in lines[1]


def test_render_event_tail_includes_warning_content() -> None:
    from stigmergy.status import render_event_tail

    events = [
        {
            "event_type": "disposition",
            "ticket": "t-warning",
            "ts": 1000.0,
            "outcome": "warning-level-issue",
        },
    ]

    result = render_event_tail(events, 25)
    assert "warning" in result
    assert "t-warning" in result


def test_render_daemon_liveness_no_pidfile(tmp_path: Path) -> None:
    from stigmergy.status import render_daemon_liveness

    rig_root = tmp_path / "rig"
    rig_root.mkdir()
    result = render_daemon_liveness(rig_root)
    assert "pidfile not found" in result


def test_render_daemon_liveness_invalid_pidfile_content(tmp_path: Path) -> None:
    from stigmergy.status import render_daemon_liveness

    rig_root = tmp_path / "rig"
    rig_root.mkdir()
    pidfile = rig_root / "daemon.pid"
    pidfile.write_text("not-a-number")

    result = render_daemon_liveness(rig_root)
    assert "invalid" in result


def test_render_daemon_liveness_alive_process(tmp_path: Path) -> None:
    import os

    from stigmergy.status import render_daemon_liveness

    rig_root = tmp_path / "rig"
    rig_root.mkdir()
    pidfile = rig_root / "daemon.pid"
    pidfile.write_text(str(os.getpid()))

    ps_responses = []

    def stub_ps_runner(cmd, **kwargs):
        class FakeResult:
            returncode = 0
            stdout = "1:23:45\n"

        ps_responses.append(cmd)
        return FakeResult()

    result = render_daemon_liveness(rig_root, ps_runner=stub_ps_runner)
    assert "alive" in result
    assert str(os.getpid()) in result
    assert "etime" in result
    assert len(ps_responses) == 1


def test_render_daemon_liveness_dead_process(tmp_path: Path) -> None:
    from stigmergy.status import render_daemon_liveness

    rig_root = tmp_path / "rig"
    rig_root.mkdir()
    pidfile = rig_root / "daemon.pid"
    fake_pid = 999999
    pidfile.write_text(str(fake_pid))

    result = render_daemon_liveness(rig_root)
    assert "DAEMON EXITED" in result
    assert str(fake_pid) in result


# --- governor park state per lane (quotagov park integration) ---------------
#
# The park-half of the spend/leash status reporting: `Status.lane_parks`
# carries one entry per charter lane sourced from the quota governor's
# state (the same `quotagov.decide` API the daemon parks claims on).
#
# CHARTER_RESOLVED has no [lanes] table (its AC12 cases predate lane
# governance), so the park-state cases below use a lane-bearing charter of
# their own.

LANE_CHARTER: dict[str, Any] = {
    "loop": {"budgets": {"dispatches": 50, "usd": 25.0, "gate_calls": 30}},
    "lanes": {
        "default": {"driver": "claude-code", "model": "metered-m"},
        "sub": {"driver": "openalph-exec", "model": "sub-m"},
    },
}


def _lane_registry() -> Registry:
    """A registry with one metered and one subscription model.

    The subscription entry deliberately sets NO ``oa_provider_key`` so the
    status path's ``oa_provider_key or provider`` fallback (mirroring the
    daemon's lane-governor lookup) is what actually keys the governor
    state — a provider-key shortcut would silently miss the park."""
    return Registry(
        entries={
            "metered-m": ModelEntry(
                name="metered-m",
                provider="p",
                family="f",
                version="1",
                pricing=PricingClass.METERED,
                input_usd_per_mtok=1.0,
                output_usd_per_mtok=2.0,
            ),
            "sub-m": ModelEntry(
                name="sub-m",
                provider="prov-sub",
                family="f",
                version="1",
                pricing=PricingClass.SUBSCRIPTION,
                marginal_usd=0.0,
            ),
        },
        version_hash="lane-reg-hash",
    )


def _governor_parking_provider(*, next_tick_at: str, limited: bool = True) -> QuotaGovernor:
    """A governor whose folded state shows quota-exhausted evidence
    (``limited`` + an upstream 429) for ``"prov-sub"`` only."""
    governor = QuotaGovernor()
    governor.providers["prov-sub"] = QuotaState(
        provider="prov-sub",
        rolling=RollingWindow(remaining=0.0, max=100.0, limited=limited, next_tick_at=next_tick_at),
        limited=limited,
        next_tick_at=next_tick_at,
        last_429_at=100.0,
        updated_at=200.0,
    )
    return governor


def test_lane_parks_parked_lane_renders_until_timestamp(
    store: RigStore, plane: RecordPlane, notification_store: NotificationStore
) -> None:
    """A parked subscription lane renders parked=yes with its until
    timestamp; the metered lane is explicitly not-parked and a provider
    with no quota state stays not-parked."""
    now = 1_000_000.0
    tick = 1_003_600.0  # +1h from now -> below the 8h escalation ceiling
    governor = _governor_parking_provider(next_tick_at=_iso_utc(tick))

    status = gather_status(
        store=store,
        record_plane=plane,
        notification_store=notification_store,
        charter_resolved=LANE_CHARTER,
        disk_path=str(store.db_path.parent),
        now=now,
        registry=_lane_registry(),
        governor=governor,
    )

    assert set(status.lane_parks) == {"default", "sub"}
    assert status.lane_parks["sub"] == {
        "parked": True,
        "parked_until": tick,
        "escalation_fired": False,
    }
    # The metered lane can never park, regardless of quota state.
    assert status.lane_parks["default"] == {
        "parked": False,
        "parked_until": None,
        "escalation_fired": False,
    }

    text = render_status(status)
    assert "sub: parked=yes  parked_until=1003600  escalation_fired=no" in text
    assert "default: parked=no  parked_until=-  escalation_fired=no" in text


def test_lane_parks_escalated_park_renders_escalation_fired(
    store: RigStore, plane: RecordPlane, notification_store: NotificationStore
) -> None:
    """A park whose wait exceeds the escalation ceiling renders the
    escalation-fired marker; the park itself (and its until-timestamp) is
    unchanged — escalation never clears the park."""
    now = 1_000_000.0
    tick = now + 9 * 3600.0  # 9h wait > 8h default ceiling
    governor = _governor_parking_provider(next_tick_at=_iso_utc(tick))

    status = gather_status(
        store=store,
        record_plane=plane,
        notification_store=notification_store,
        charter_resolved=LANE_CHARTER,
        disk_path=str(store.db_path.parent),
        now=now,
        registry=_lane_registry(),
        governor=governor,
    )

    assert status.lane_parks["sub"] == {
        "parked": True,
        "parked_until": tick,
        "escalation_fired": True,
    }

    text = render_status(status)
    assert f"sub: parked=yes  parked_until={tick:.0f}  escalation_fired=yes" in text


def _iso_utc(epoch_s: float) -> str:
    from datetime import UTC, datetime

    return datetime.fromtimestamp(epoch_s, tz=UTC).isoformat()


def test_lane_parks_no_feed_configured_all_lanes_not_parked(
    store: RigStore, plane: RecordPlane, notification_store: NotificationStore
) -> None:
    """No ``governor`` argument (no feed configured) OR a governor with no
    folded state (no quota evidence) -> every lane is explicitly
    not-parked/None/not-escalated, never a missing key; the metered lane
    never parks even with a fully exhausted governor."""
    now = 1_000_000.0

    status_no_governor = gather_status(
        store=store,
        record_plane=plane,
        notification_store=notification_store,
        charter_resolved=LANE_CHARTER,
        disk_path=str(store.db_path.parent),
        now=now,
    )
    expected_empty = {
        name: {"parked": False, "parked_until": None, "escalation_fired": False}
        for name in ("default", "sub")
    }
    assert status_no_governor.lane_parks == expected_empty

    status_empty_governor = gather_status(
        store=store,
        record_plane=plane,
        notification_store=notification_store,
        charter_resolved=LANE_CHARTER,
        disk_path=str(store.db_path.parent),
        now=now,
        registry=_lane_registry(),
        governor=QuotaGovernor(),  # fresh: no feed lines folded
    )
    assert status_empty_governor.lane_parks == expected_empty

    # A governor that only has quota state for ANOTHER provider also parks
    # nothing (no evidence for this lane's provider).
    other = QuotaGovernor()
    other.providers["other-prov"] = QuotaState(
        provider="other-prov", limited=True, last_429_at=1.0, updated_at=2.0
    )
    status_other = gather_status(
        store=store,
        record_plane=plane,
        notification_store=notification_store,
        charter_resolved=LANE_CHARTER,
        disk_path=str(store.db_path.parent),
        now=now,
        registry=_lane_registry(),
        governor=other,
    )
    assert status_other.lane_parks == expected_empty

    # A registry miss (the metered lane's model absent) is not-parked for
    # that lane, never an error — and the miss must not suppress the park
    # of the OTHER lane, whose model still resolves.
    miss_registry = Registry(
        entries={
            "sub-m": ModelEntry(
                name="sub-m",
                provider="prov-sub",
                family="f",
                version="1",
                pricing=PricingClass.SUBSCRIPTION,
                marginal_usd=0.0,
            ),
        },
        version_hash="h",
    )
    governor = _governor_parking_provider(next_tick_at=_iso_utc(now + 3600))
    status_miss = gather_status(
        store=store,
        record_plane=plane,
        notification_store=notification_store,
        charter_resolved=LANE_CHARTER,
        disk_path=str(store.db_path.parent),
        now=now,
        registry=miss_registry,
        governor=governor,
    )
    # The metered lane's model is missing from the registry -> not-parked
    # (explicit), but the sub lane's park still shows through the miss.
    assert status_miss.lane_parks["default"] == {
        "parked": False,
        "parked_until": None,
        "escalation_fired": False,
    }
    assert status_miss.lane_parks["sub"] == {
        "parked": True,
        "parked_until": now + 3600,
        "escalation_fired": False,
    }

    text = render_status(status_no_governor)
    assert "default: parked=no  parked_until=-  escalation_fired=no" in text
    assert "sub: parked=no  parked_until=-  escalation_fired=no" in text


def test_lane_parks_absent_leaves_preexisting_status_fields_unchanged(
    store: RigStore, plane: RecordPlane, notification_store: NotificationStore
) -> None:
    """A rig with no quota evidence renders every pre-existing AC12 field
    exactly as before the park-state addition: same state counts, spend
    totals, heartbeat/queue/escalation/disk fields, and the old render
    lines all present in the same order (the new === quota parks === block
    slots between spend and health)."""
    now = time.time()

    add_ticket(store, "t-pool", state="pool")
    add_ticket(store, "t-parked", state="parked")
    plane.append(make_dispatch_event(dispatch_id="dispatch-0001"))
    plane.append(make_gate_event(dispatch_id="dispatch-0001"))
    store.set_meta("daemon_heartbeat_at", str(now - 5))
    notification_store.record_intent(
        ticket="t-parked", kind="escalation", title="e", message="m", now=now
    )

    # Same rig, no quota inputs at all (the standalone status run shape).
    status = gather_status(
        store=store,
        record_plane=plane,
        notification_store=notification_store,
        charter_resolved=CHARTER_RESOLVED,  # no [lanes] table
        disk_path=str(store.db_path.parent),
        now=now,
    )

    # Every pre-existing field still populated exactly as before.
    assert status.state_counts["pool"] == 1
    assert status.state_counts["parked"] == 1
    assert set(status.state_counts) == set(STATES)
    assert status.spend["dispatches_used"] == 1
    assert status.spend["gate_calls_used"] == 1
    assert status.spend["usd_cap"] == 25.0
    assert status.daemon_heartbeat_age == pytest.approx(5.0, abs=1.0)
    assert status.escalated_unnotified >= 1
    assert status.disk_free_bytes > 0
    assert status.disk_ok is True

    # No lanes in the charter -> no per-lane entries (nothing invented),
    # but the block still renders explicitly.
    assert status.lane_parks == {}

    text = render_status(status)
    # All pre-existing rendered lines survive, in order.
    for line in (
        "=== spend vs budgets ===",
        "=== quota parks ===",
        "  (no lanes)",
        "=== health ===",
        "  usd: 0.0246 / cap 25.0",
        "  dispatches: 1 / cap 50",
        "  gate_calls: 1 / cap 30",
        "  unbudgetable_events: 0",
        "  daemon_heartbeat_age: ",
        "  queue_age: ",
        "  escalated_unnotified: 1",
        "  disk_free_bytes: ",
    ):
        assert line in text
    assert (
        text.index("=== spend vs budgets ===")
        < text.index("=== quota parks ===")
        < text.index("=== health ===")
    )


def test_gather_lane_parks_pure_function_matches_governor_decide() -> None:
    """`gather_lane_parks` is the same evidence rule as `decide`: a park
    with no computable next-tick deadline is still parked (until unknown,
    escalation never due)."""
    now = 500.0
    governor = QuotaGovernor()
    governor.providers["prov-sub"] = QuotaState(
        provider="prov-sub", limited=True, last_429_at=1.0, updated_at=2.0
    )
    parks = gather_lane_parks(
        charter_resolved=LANE_CHARTER,
        registry=_lane_registry(),
        governor=governor,
        now=now,
    )
    assert parks["sub"]["parked"] is True
    assert parks["sub"]["parked_until"] is None
    assert parks["sub"]["escalation_fired"] is False
    assert parks["default"]["parked"] is False

    # And the non-park branch matches decide() outcome-for-outcome.
    entry = _lane_registry().resolve("sub-m")
    d = decide(entry, governor.providers.get("prov-sub"), now=now)
    assert (d.decision != "dispatch") == parks["sub"]["parked"]
