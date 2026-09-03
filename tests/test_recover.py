"""Tests for stigmergy.recover (SPEC.md §3, §9 "Crash recovery", §10 AC9).

Case numbering below matches the bead .16 design doc's exact case list
(design §3, cases 25-33). Uses FAKE `ContainerReaper`/`WeaveJournalResolver`
collaborators plus a REAL `RigStore` + `RecordPlane` in a tmp dir (per the
design doc's explicit instruction) — recover.py's own logic is exercised
for real; only the host-touching protocols are faked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from stigmergy.records import EventType, RecordPlane, make_event
from stigmergy.recover import (
    DISPOSITION_TO_STATE,
    RecoveryError,
    RecoveryReport,
    project_ticket_states,
    recover,
)
from stigmergy.rig import RigStore
from stigmergy.statemachine import (
    ESCALATED,
    IN_FLIGHT,
    LANDED,
    PARKED,
    POOL,
    REJECTED,
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


def common_fields(**overrides: Any) -> dict[str, Any]:
    """A complete, valid SPEC §8 common-field set; override as needed."""
    base: dict[str, Any] = {
        "rig": "shipyard",
        "ticket": "workspace-e2uh.16",
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


def make_disposition_event(**overrides: Any):
    overrides.setdefault("tokens", {"in": 0, "cached": 0, "out": 0, "reasoning": 0})
    overrides.setdefault("computed_usd", 0.0)
    overrides.setdefault("wall_time_seconds", 0.0)
    fields = common_fields(**overrides)
    return make_event(EventType.DISPOSITION, **fields)


class FakeReaper:
    """Fake ContainerReaper: `running` lists live dispatch_ids; `reap()`
    records the call (optionally into a shared ``call_order`` list for the
    ordering assertion in case 25) and removes the id from `running`."""

    def __init__(self, running: list[str], call_order: list[tuple[str, str]] | None = None):
        self.running: list[str] = list(running)
        self.reaped: list[str] = []
        self._call_order = call_order

    def list_running(self) -> list[str]:
        return list(self.running)

    def reap(self, dispatch_id: str) -> None:
        self.reaped.append(dispatch_id)
        if self._call_order is not None:
            self._call_order.append(("reap", dispatch_id))
        if dispatch_id in self.running:
            self.running.remove(dispatch_id)


class FakeWeaveResolver:
    """Fake WeaveJournalResolver: `in_progress`/`resolution` are fixed at
    construction; `resolve_called` records whether `resolve()` was invoked."""

    def __init__(self, *, in_progress: bool, resolution: str = "landed"):
        self._in_progress = in_progress
        self._resolution = resolution
        self.resolve_called = False

    def in_progress(self) -> bool:
        return self._in_progress

    def resolve(self) -> str:
        self.resolve_called = True
        return self._resolution


class OrderTrackingStore:
    """Thin delegating wrapper around a real `RigStore` that additionally
    appends `("expire", ticket_id)` to a shared `call_order` list on every
    `update_ticket` call — used only by case 25 to prove reap-before-expire
    ordering. (Within `recover()`, the ONLY caller of `update_ticket` is
    `intake.expire_leases`; `recover.py` itself never writes to the store.)
    """

    def __init__(self, store: RigStore, call_order: list[tuple[str, str]]):
        self._store = store
        self._call_order = call_order

    def get_ticket(self, ticket_id: str) -> dict[str, Any] | None:
        return self._store.get_ticket(ticket_id)

    def list_tickets(self, *, state: str | None = None) -> list[dict[str, Any]]:
        return self._store.list_tickets(state=state)

    def update_ticket(self, ticket_id: str, **fields: Any) -> None:
        self._call_order.append(("expire", ticket_id))
        self._store.update_ticket(ticket_id, **fields)


def default_recover_kwargs(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "reaper": FakeReaper(running=[]),
        "weave_resolver": FakeWeaveResolver(in_progress=False),
        "git_repo_paths": [],
        "min_disk_bytes": 0,
        "disk_path": str(tmp_path),
        "now": 1000.0,
    }
    base.update(overrides)
    return base


# --- 25. test_recover_reaps_orphan_before_expiring_lease --------------------


def test_recover_reaps_orphan_before_expiring_lease(
    store: RigStore, plane: RecordPlane, tmp_path: Path
) -> None:
    store.add_ticket(
        id="t-order",
        title="Order",
        state=IN_FLIGHT,
        lease_owner="worker-1",
        lease_dispatch_id="dispatch-order",
        lease_expires_at=100.0,  # already expired (now=1000.0)
        lease_heartbeat_at=99.0,
    )

    call_order: list[tuple[str, str]] = []
    wrapped_store = OrderTrackingStore(store, call_order)
    reaper = FakeReaper(running=["dispatch-order"], call_order=call_order)

    recover(
        wrapped_store,
        plane,
        **default_recover_kwargs(tmp_path, reaper=reaper),
    )

    assert call_order == [("reap", "dispatch-order"), ("expire", "t-order")]


# --- 26. test_recover_expires_dead_leases_to_pool_counters_unchanged -------


def test_recover_expires_dead_leases_to_pool_counters_unchanged(
    store: RigStore, plane: RecordPlane, tmp_path: Path
) -> None:
    store.add_ticket(
        id="t-dead-lease",
        title="Dead lease",
        state=IN_FLIGHT,
        lease_owner="worker-1",
        lease_dispatch_id="dispatch-dead",
        lease_expires_at=100.0,
        lease_heartbeat_at=99.0,
        attempts_used=2,
    )

    report = recover(store, plane, **default_recover_kwargs(tmp_path))

    assert report.expired_leases == ["t-dead-lease"]
    row = store.get_ticket("t-dead-lease")
    assert row["state"] == POOL
    assert row["lease_owner"] is None
    assert row["lease_dispatch_id"] is None
    assert row["lease_expires_at"] is None
    assert row["lease_heartbeat_at"] is None
    assert row["attempts_used"] == 2  # untouched


# --- 27. test_recover_orphan_dispatch_sealed_counters_unchanged -----------


def test_recover_orphan_dispatch_sealed_counters_unchanged(
    store: RigStore, plane: RecordPlane, tmp_path: Path
) -> None:
    store.add_ticket(
        id="t-orphan",
        title="Orphan",
        state=IN_FLIGHT,
        lease_owner="worker-1",
        lease_dispatch_id="dispatch-orphan",
        lease_expires_at=100.0,
        lease_heartbeat_at=99.0,
        attempts_used=0,
    )
    plane.append(
        make_dispatch_event(dispatch_id="dispatch-orphan", ticket="t-orphan", attempt=1)
    )
    # No terminal event for dispatch-orphan -- daemon crashed mid-dispatch.

    report = recover(store, plane, **default_recover_kwargs(tmp_path))

    assert report.sealed_dispatches == ["dispatch-orphan"]

    events = plane.read_events()
    dispatch_events = [e for e in events if e["dispatch_id"] == "dispatch-orphan"]
    assert len(dispatch_events) == 2  # original DISPATCH + the new seal
    seal = [e for e in dispatch_events if e["event_type"] == "disposition"]
    assert len(seal) == 1
    assert seal[0]["disposition"] == "orphaned"
    assert seal[0]["ticket"] == "t-orphan"

    # Daemon-crash orphan is free: no attempt consumed.
    assert store.get_ticket("t-orphan")["attempts_used"] == 0


# --- 28. test_recover_wedge_already_sealed_untouched ------------------------


def test_recover_wedge_already_sealed_untouched(
    store: RigStore, plane: RecordPlane, tmp_path: Path
) -> None:
    store.add_ticket(
        id="t-wedge",
        title="Wedge",
        state=POOL,  # already moved on by the pre-crash retry decision
        attempts_used=1,  # the wedge's attempt was already consumed
    )
    plane.append(make_dispatch_event(dispatch_id="dispatch-wedge", ticket="t-wedge", attempt=1))
    # The terminal disposition was already sealed BEFORE the crash: the
    # wedge burned a rung attempt and the retry decision (clean-restart,
    # back to pool) was already journaled.
    plane.append(
        make_disposition_event(
            dispatch_id="dispatch-wedge",
            ticket="t-wedge",
            attempt=1,
            attempt_kind="clean-restart",
            disposition="pool",
            reason="wedge-timeout",
        )
    )
    events_before = plane.read_events()

    report = recover(store, plane, **default_recover_kwargs(tmp_path))

    assert report.sealed_dispatches == []
    events_after = plane.read_events()
    assert events_after == events_before  # nothing appended
    assert store.get_ticket("t-wedge")["attempts_used"] == 1  # untouched


# --- 29. test_recover_resumes_weave_journal_when_in_progress ---------------


def test_recover_resumes_weave_journal_when_in_progress(
    store: RigStore, plane: RecordPlane, tmp_path: Path
) -> None:
    resolver = FakeWeaveResolver(in_progress=True, resolution="landed")
    report = recover(
        store, plane, **default_recover_kwargs(tmp_path, weave_resolver=resolver)
    )
    assert resolver.resolve_called is True
    assert report.weave_resolution == "landed"


def test_recover_does_not_resolve_weave_when_not_in_progress(
    store: RigStore, plane: RecordPlane, tmp_path: Path
) -> None:
    resolver = FakeWeaveResolver(in_progress=False)
    report = recover(
        store, plane, **default_recover_kwargs(tmp_path, weave_resolver=resolver)
    )
    assert resolver.resolve_called is False
    assert report.weave_resolution is None


# --- 30. test_recover_clears_stale_git_lock_loop_repo_only ------------------


def test_recover_clears_stale_git_lock_loop_repo_only(
    store: RigStore, plane: RecordPlane, tmp_path: Path
) -> None:
    loop_repo = tmp_path / "loop-repo"
    other_repo = tmp_path / "other-repo"
    (loop_repo / ".git").mkdir(parents=True)
    (other_repo / ".git").mkdir(parents=True)
    loop_lock = loop_repo / ".git" / "index.lock"
    other_lock = other_repo / ".git" / "index.lock"
    loop_lock.write_text("stale")
    other_lock.write_text("stale")

    report = recover(
        store,
        plane,
        **default_recover_kwargs(tmp_path, git_repo_paths=[str(loop_repo)]),
    )

    assert report.cleared_git_locks == [str(loop_repo)]
    assert not loop_lock.exists()
    assert other_lock.exists()  # untouched -- not a loop-owned path given here


# --- 31. test_recover_disk_headroom_fail_closed -----------------------------


def test_recover_disk_headroom_fail_closed(
    store: RigStore, plane: RecordPlane, tmp_path: Path
) -> None:
    # Absurdly high floor -> guaranteed to exceed actual free space -> fails
    # closed by raising, rather than silently starting to poll.
    with pytest.raises(RecoveryError):
        recover(
            store,
            plane,
            **default_recover_kwargs(tmp_path, min_disk_bytes=10**18),
        )

    # Sufficient headroom (0 bytes required) -> proceeds, disk_ok True.
    report = recover(store, plane, **default_recover_kwargs(tmp_path, min_disk_bytes=0))
    assert report.disk_ok is True
    assert report.disk_free_bytes > 0


# --- 32. test_recover_full_report_shape -------------------------------------


def test_recover_full_report_shape(store: RigStore, plane: RecordPlane, tmp_path: Path) -> None:
    # One running container whose ticket is not terminal -> reaped.
    store.add_ticket(
        id="t-running",
        title="Running",
        state=IN_FLIGHT,
        lease_dispatch_id="dispatch-running",
        lease_expires_at=2000.0,  # not expired
        lease_heartbeat_at=999.0,
        lease_owner="worker-1",
    )
    # One dead lease.
    store.add_ticket(
        id="t-lease",
        title="Lease",
        state=IN_FLIGHT,
        lease_dispatch_id="dispatch-lease",
        lease_expires_at=100.0,
        lease_heartbeat_at=99.0,
        lease_owner="worker-2",
    )
    # One orphan dispatch (no terminal event).
    store.add_ticket(id="t-orphan", title="Orphan", state=IN_FLIGHT)
    plane.append(make_dispatch_event(dispatch_id="dispatch-orphan-full", ticket="t-orphan"))

    # One git lock.
    loop_repo = tmp_path / "repo"
    (loop_repo / ".git").mkdir(parents=True)
    (loop_repo / ".git" / "index.lock").write_text("stale")

    reaper = FakeReaper(running=["dispatch-running"])
    resolver = FakeWeaveResolver(in_progress=True, resolution="rolled-back")

    report = recover(
        store,
        plane,
        **default_recover_kwargs(
            tmp_path,
            reaper=reaper,
            weave_resolver=resolver,
            git_repo_paths=[str(loop_repo)],
        ),
    )

    assert isinstance(report, RecoveryReport)
    assert report.reaped_containers == ["dispatch-running"]
    assert report.sealed_dispatches == ["dispatch-orphan-full"]
    assert report.expired_leases == ["t-lease"]
    assert report.weave_resolution == "rolled-back"
    assert report.cleared_git_locks == [str(loop_repo)]
    assert report.disk_ok is True
    assert report.disk_free_bytes > 0


# --- 33. test_recover_idempotent ---------------------------------------------


def test_recover_idempotent(store: RigStore, plane: RecordPlane, tmp_path: Path) -> None:
    store.add_ticket(
        id="t-running",
        title="Running",
        state=IN_FLIGHT,
        lease_dispatch_id="dispatch-running",
        lease_expires_at=2000.0,
        lease_heartbeat_at=999.0,
        lease_owner="worker-1",
    )
    store.add_ticket(
        id="t-lease",
        title="Lease",
        state=IN_FLIGHT,
        lease_dispatch_id="dispatch-lease",
        lease_expires_at=100.0,
        lease_heartbeat_at=99.0,
        lease_owner="worker-2",
    )
    store.add_ticket(id="t-orphan", title="Orphan", state=IN_FLIGHT)
    plane.append(make_dispatch_event(dispatch_id="dispatch-orphan-idem", ticket="t-orphan"))

    reaper = FakeReaper(running=["dispatch-running"])
    resolver = FakeWeaveResolver(in_progress=False)

    first = recover(
        store, plane, **default_recover_kwargs(tmp_path, reaper=reaper, weave_resolver=resolver)
    )
    assert first.reaped_containers == ["dispatch-running"]
    assert first.sealed_dispatches == ["dispatch-orphan-idem"]
    assert first.expired_leases == ["t-lease"]

    events_after_first = plane.read_events()

    second = recover(
        store, plane, **default_recover_kwargs(tmp_path, reaper=reaper, weave_resolver=resolver)
    )

    assert second.reaped_containers == []
    assert second.sealed_dispatches == []
    assert second.expired_leases == []
    assert second.weave_resolution is None
    assert second.disk_ok is True

    # No duplicate seal events -- the event log is unchanged after the
    # second, no-op run.
    events_after_second = plane.read_events()
    assert events_after_second == events_after_first

    seal_events = [
        e
        for e in events_after_second
        if e["dispatch_id"] == "dispatch-orphan-idem" and e["event_type"] == "disposition"
    ]
    assert len(seal_events) == 1

    # And the previously-landed side effects are stable: t-running is not
    # magically terminal (still in_flight, unreaped-again), t-lease stayed
    # in pool.
    assert store.get_ticket("t-lease")["state"] == POOL
    assert store.get_ticket("t-orphan")["state"] == IN_FLIGHT


def test_recover_terminal_ticket_running_container_not_reaped(
    store: RigStore, plane: RecordPlane, tmp_path: Path
) -> None:
    """Not a numbered design case, but directly exercises the terminal-state
    exemption named in design §2.2 step 1 ("whose ticket is NOT in a
    terminal state") -- a running container for an already-`landed` ticket
    must not be reaped."""
    store.add_ticket(
        id="t-landed",
        title="Landed",
        state=LANDED,
        lease_dispatch_id="dispatch-landed",
    )
    reaper = FakeReaper(running=["dispatch-landed"])

    report = recover(store, plane, **default_recover_kwargs(tmp_path, reaper=reaper))

    assert report.reaped_containers == []
    assert reaper.reaped == []


# --- project_ticket_states (pure event-log -> ticket-table projection) -------


def test_disposition_to_state_mapping_is_named() -> None:
    # The named disposition-to-state mapping the projection is built from:
    # terminal dispositions name their state; every pool re-entry projects pool.
    assert DISPOSITION_TO_STATE == {
        "landed": LANDED,
        "rejected": REJECTED,
        "escalated": ESCALATED,
        "parked": PARKED,
        "pool-reentry": POOL,
        "infra-retry": POOL,
        "quota-parked": POOL,
        "orphaned": POOL,
    }


def test_project_ticket_states_landed_disposition_projects_landed() -> None:
    events = [
        make_dispatch_event(ticket="t-landed", dispatch_id="d-1").payload,
        make_disposition_event(
            ticket="t-landed", dispatch_id="d-1", disposition="landed"
        ).payload,
    ]
    assert project_ticket_states(events) == {"t-landed": LANDED}


@pytest.mark.parametrize("disposition", ["infra-retry", "quota-parked", "orphaned", "pool-reentry"])
def test_project_ticket_states_pool_reentry_projects_pool(disposition: str) -> None:
    events = [
        make_dispatch_event(ticket="t-retry", dispatch_id="d-2").payload,
        make_disposition_event(
            ticket="t-retry",
            dispatch_id="d-2",
            attempt_kind="infra-retry",
            disposition=disposition,
        ).payload,
    ]
    assert project_ticket_states(events) == {"t-retry": POOL}


def test_project_ticket_states_parked_disposition_projects_parked() -> None:
    events = [
        make_dispatch_event(ticket="t-parked", dispatch_id="d-3").payload,
        make_disposition_event(ticket="t-parked", dispatch_id="d-3", disposition="parked").payload,
    ]
    assert project_ticket_states(events) == {"t-parked": PARKED}


def test_project_ticket_states_open_dispatch_projects_in_flight() -> None:
    # A dispatch with no terminal disposition anywhere projects in-flight.
    events = [make_dispatch_event(ticket="t-open", dispatch_id="d-4").payload]
    assert project_ticket_states(events) == {"t-open": IN_FLIGHT}


def test_project_ticket_states_rejected_and_escalated_project_their_states() -> None:
    events = [
        make_dispatch_event(ticket="t-rej", dispatch_id="d-5").payload,
        make_disposition_event(ticket="t-rej", dispatch_id="d-5", disposition="rejected").payload,
        make_dispatch_event(ticket="t-esc", dispatch_id="d-6", attempt=2).payload,
        make_disposition_event(
            ticket="t-esc",
            dispatch_id="d-6",
            attempt=2,
            attempt_kind="tier1-repair",
            disposition="escalated",
            reason="ladder-exhausted",
        ).payload,
    ]
    assert project_ticket_states(events) == {"t-rej": REJECTED, "t-esc": ESCALATED}


def test_project_ticket_states_redispatch_after_reentry_reprojects_in_flight() -> None:
    # pool re-entry disposition, then a fresh dispatch: the ticket is open
    # again, so the projection moves back to in_flight.
    events = [
        make_dispatch_event(ticket="t-re", dispatch_id="d-7").payload,
        make_disposition_event(
            ticket="t-re", dispatch_id="d-7", attempt_kind="infra-retry", disposition="infra-retry"
        ).payload,
        make_dispatch_event(ticket="t-re", dispatch_id="d-8", attempt=2).payload,
    ]
    assert project_ticket_states(events) == {"t-re": IN_FLIGHT}


def test_project_ticket_states_tickets_without_events_are_absent() -> None:
    events = [make_dispatch_event(ticket="t-only", dispatch_id="d-9").payload]
    projection = project_ticket_states(events)
    assert projection == {"t-only": IN_FLIGHT}
    assert "t-no-events" not in projection

    assert project_ticket_states([]) == {}


def test_project_ticket_states_does_not_mutate_its_input() -> None:
    events = [
        make_dispatch_event(ticket="t-mut", dispatch_id="d-10").payload,
        make_disposition_event(ticket="t-mut", dispatch_id="d-10", disposition="parked").payload,
    ]
    snapshot = [dict(ev) for ev in events]
    assert project_ticket_states(events) == {"t-mut": PARKED}
    assert events == snapshot  # input list and its dicts are untouched
