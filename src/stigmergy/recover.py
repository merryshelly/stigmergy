"""The `recover` station (SPEC.md §3, §9 "Crash recovery", §10 AC9).

Runs once at every daemon start, BEFORE the first poll. All host-touching
operations (container reaping, weave journal resolution) are INJECTED as
:class:`typing.Protocol` collaborators so this module is pure-logic-testable
with fakes — zero host risk in tests. The daemon (.22) wires the real podman
reaper and the real weaver (.18) as the concrete implementations.

**Order is normative (SPEC §9):** reap orphaned containers, THEN seal
interrupted events, THEN expire dead leases, THEN resolve the weave journal,
THEN clear stale git locks, THEN verify disk headroom. Reaping must precede
lease expiry — otherwise a freed ticket could be re-claimed and
re-dispatched while its old container is still running (double-dispatch).

**Orphan vs wedge (SPEC §9):** an in-flight dispatch whose terminal event
was never written (daemon crashed mid-dispatch) is an ORPHAN — free, no
attempt consumed, counters unchanged, but the record plane must still be
sealed so the dispatch is no longer "open." A WEDGE (loop-detected timeout
kill) already recorded a terminal `failed` outcome with a consumed attempt
BEFORE the crash — that event is already sealed, and recovery must never
re-touch it or its counters. The distinguishing signal lives entirely in
the event log: does a DISPATCH event for this `dispatch_id` have a matching
terminal event (any :data:`ATTEMPT_KINDS`-carrying disposition/outcome
event for the same `dispatch_id`)? Sealed already -> leave it (wedge,
already counted). Not sealed -> orphan (seal it now, free).

**Lease expiry is not reimplemented here.** `intake.expire_leases` already
guarantees counters are untouched; this module calls it, never duplicates
its logic (SPEC §9 "expire dead leases back to `pool` (attempt counters
unchanged)").
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from stigmergy import intake
from stigmergy.records import EventType, RecordPlane, make_event
from stigmergy.statemachine import (
    ESCALATED,
    IN_FLIGHT,
    LANDED,
    PARKED,
    POOL,
    REJECTED,
    TERMINAL_STATES,
)

# Event types that count as a dispatch's terminal seal for orphan detection.
# A DISPOSITION event (whether written by the normal loop via
# `statemachine.record_disposition` for landed/rejected/escalated/parked/
# pool-reentry, or by this module's own orphan seal) closes the dispatch.
_TERMINAL_EVENT_TYPES = frozenset({EventType.DISPOSITION.value})

_ZERO_TOKENS: dict[str, int] = {"in": 0, "cached": 0, "out": 0, "reasoning": 0}

# Identity fields copied verbatim from the found DISPATCH event onto the
# orphan-seal DISPOSITION event, so the seal shares the exact same
# `dispatch_id`/`ticket`/etc identity (SPEC §8 common fields) without
# inventing any value the original dispatch didn't already carry.
_SEAL_COPY_FIELDS: tuple[str, ...] = (
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


class ContainerReaper(Protocol):
    """Injected collaborator: lists and kills worker containers by
    dispatch ID. The real implementation (.22) shells out to podman; tests
    inject a fake that just records calls."""

    def list_running(self) -> list[str]:
        """Return the dispatch_ids of every currently-running worker
        container."""
        ...

    def reap(self, dispatch_id: str) -> None:
        """Kill and remove the container for ``dispatch_id``."""
        ...


class WeaveJournalResolver(Protocol):
    """Injected collaborator implemented by the weaver (.18). `recover`
    only defines and calls this contract; the weaver owns the idempotent
    compare-and-swap land underneath it."""

    def in_progress(self) -> bool:
        """True iff there is an interrupted weave to resolve."""
        ...

    def resolve(self) -> str:
        """Resolve the interrupted weave. Returns `"landed"` or
        `"rolled-back"`. Idempotent: a repeat call on an already-resolved
        journal completes identically or leaves staging untouched."""
        ...


@dataclass
class RecoveryReport:
    """The result of one :func:`recover` run (SPEC §9/§10 AC9)."""

    reaped_containers: list[str] = field(default_factory=list)
    sealed_dispatches: list[str] = field(default_factory=list)
    expired_leases: list[str] = field(default_factory=list)
    weave_resolution: str | None = None
    cleared_git_locks: list[str] = field(default_factory=list)
    disk_ok: bool = True
    disk_free_bytes: int = 0


class RecoveryError(Exception):
    """Raised when recovery must fail closed (SPEC §9 "verify disk headroom
    before first poll") rather than let the daemon start polling."""


def recover(
    store: Any,
    record_plane: RecordPlane,
    *,
    reaper: ContainerReaper,
    weave_resolver: WeaveJournalResolver,
    git_repo_paths: list[str | os.PathLike[str]],
    min_disk_bytes: int,
    disk_path: str | os.PathLike[str],
    now: float,
) -> RecoveryReport:
    """Run the full ordered recovery sequence once, at daemon start (SPEC
    §9, §10 AC9). Order is normative — see module docstring.

    Raises :class:`RecoveryError` (fail closed) if disk headroom is below
    ``min_disk_bytes`` — the daemon must not begin polling on a full disk.
    All other steps are best-effort in the sense that an empty/no-op input
    (nothing running, nothing orphaned, nothing expired, no weave in
    progress, no stale locks) produces an all-empty, `disk_ok=True` report.
    """
    report = RecoveryReport()

    # 1. Reap orphaned containers FIRST, before touching leases (double-
    # dispatch guard: a ticket must never be freed back to pool while its
    # old container might still be running).
    report.reaped_containers = _reap_orphaned_containers(store, reaper)

    # 2. Seal interrupted events (orphan-vs-wedge, driven by the event log
    # alone — independent of what step 1 just reaped: a container can exit
    # naturally before its terminal event is written, and this must still
    # be sealed).
    report.sealed_dispatches = _seal_interrupted_dispatches(record_plane)

    # 3. Expire dead leases -> pool. Counters untouched (guaranteed by
    # intake.expire_leases itself) — never reimplemented here.
    report.expired_leases = intake.expire_leases(store, now=now)

    # 4. Resolve the weave journal, if one is in progress.
    if weave_resolver.in_progress():
        report.weave_resolution = weave_resolver.resolve()
    else:
        report.weave_resolution = None

    # 5. Clear stale git locks — loop-owned repos ONLY (never worker
    # clones, never anything outside `git_repo_paths`).
    report.cleared_git_locks = _clear_stale_git_locks(git_repo_paths)

    # 6. Verify disk headroom BEFORE first poll. Fail closed.
    report.disk_free_bytes = shutil.disk_usage(disk_path).free
    report.disk_ok = report.disk_free_bytes >= min_disk_bytes
    if not report.disk_ok:
        raise RecoveryError(
            f"disk headroom {report.disk_free_bytes} bytes at {disk_path!r} is below "
            f"the minimum {min_disk_bytes} bytes — refusing to start polling"
        )

    return report


def _reap_orphaned_containers(store: Any, reaper: ContainerReaper) -> list[str]:
    """Reap every running container whose ticket is not (yet known to be)
    terminal. A running dispatch_id with no matching ticket at all (a
    dangling id — e.g. the ticket has since been reclaimed under a new
    dispatch_id) cannot be proven terminal, so it is reaped too: fail
    closed on the double-dispatch guard, never skip on missing evidence.
    """
    ticket_state_by_dispatch: dict[str, str] = {}
    for row in store.list_tickets():
        dispatch_id = row.get("lease_dispatch_id")
        if dispatch_id is not None:
            ticket_state_by_dispatch[dispatch_id] = row.get("state")

    reaped: list[str] = []
    for dispatch_id in reaper.list_running():
        state = ticket_state_by_dispatch.get(dispatch_id)
        if state in TERMINAL_STATES:
            continue
        reaper.reap(dispatch_id)
        reaped.append(dispatch_id)
    return reaped


def _seal_interrupted_dispatches(record_plane: RecordPlane) -> list[str]:
    """Find every DISPATCH event whose dispatch_id has no matching terminal
    event, and append an orphan-seal DISPOSITION event for each (SPEC §9).

    Orphan (daemon crash, no terminal event sealed) -> counters UNCHANGED;
    this function never touches the ticket store at all, only the record
    plane. A dispatch that already carries a terminal event (a wedge,
    already counted) is left completely alone — this is what makes a
    second `recover()` run idempotent (case 33): the very seal this
    function appends IS a terminal event, so a repeat scan sees the
    dispatch as already closed.
    """
    events = record_plane.read_events()

    first_dispatch_event: dict[str, dict[str, Any]] = {}
    sealed_dispatch_ids: set[str] = set()
    for ev in events:
        dispatch_id = ev.get("dispatch_id")
        if not isinstance(dispatch_id, str):
            continue
        event_type = ev.get("event_type")
        if event_type == EventType.DISPATCH.value:
            first_dispatch_event.setdefault(dispatch_id, ev)
        elif event_type in _TERMINAL_EVENT_TYPES:
            sealed_dispatch_ids.add(dispatch_id)

    sealed_now: list[str] = []
    for dispatch_id, dispatch_event in first_dispatch_event.items():
        if dispatch_id in sealed_dispatch_ids:
            continue  # wedge: already sealed, already counted — leave it.
        _seal_orphan(record_plane, dispatch_event)
        sealed_now.append(dispatch_id)
    return sorted(sealed_now)


def _seal_orphan(record_plane: RecordPlane, dispatch_event: dict[str, Any]) -> None:
    """Append the terminal orphan-seal DISPOSITION event for one crashed,
    never-sealed dispatch. Copies identity fields from the original
    DISPATCH event; declares a true zero cost (a seal spends nothing) and
    `attempt_kind="infra-retry"` (free — no attempt consumed, mirroring
    INFRA's counter-effect: an orphan is the daemon's fault, not the
    ticket's)."""
    fields = {key: dispatch_event.get(key) for key in _SEAL_COPY_FIELDS}
    fields["attempt_kind"] = "infra-retry"
    fields["tokens"] = dict(_ZERO_TOKENS)
    fields["computed_usd"] = 0.0
    fields["wall_time_seconds"] = 0.0
    fields["disposition"] = "orphaned"
    fields["reason"] = "daemon-crash-recovery"

    event = make_event(EventType.DISPOSITION, **fields)
    record_plane.append(event)


def _clear_stale_git_lock(repo_path: str | os.PathLike[str]) -> str | None:
    """Remove ``<repo_path>/.git/index.lock`` if present. Returns the
    cleared path (as given) or `None` if there was nothing to clear."""
    lock_path = Path(repo_path) / ".git" / "index.lock"
    if lock_path.exists():
        lock_path.unlink()
        return str(repo_path)
    return None


def _clear_stale_git_locks(git_repo_paths: list[str | os.PathLike[str]]) -> list[str]:
    """Clear stale `.git/index.lock` files across the given LOOP-OWNED repo
    paths only (SPEC §9: "clear stale git locks (loop-owned repos only)").
    Never touches anything outside the given paths — worker clones hold no
    shared git metadata anyway and are never named here."""
    cleared: list[str] = []
    for repo_path in git_repo_paths:
        cleared_path = _clear_stale_git_lock(repo_path)
        if cleared_path is not None:
            cleared.append(cleared_path)
    return cleared


# --- event-log projection (read-only drift reporting) ------------------------


# The named disposition-to-state mapping (SPEC §9): a DISPOSITION event's
# `disposition` value is the ticket-table state the ticket is expected to
# hold afterwards. Terminal dispositions name their state directly; the
# pool re-entry dispositions — daemon infra fast path (`infra-retry`),
# quota-governor lane park (`quota-parked`), the retry-ladder re-entry
# (`pool-reentry`), and this module's own orphan seal (`orphaned`) — all
# project `pool`.
DISPOSITION_TO_STATE: dict[str, str] = {
    "landed": LANDED,
    "rejected": REJECTED,
    "escalated": ESCALATED,
    "parked": PARKED,
    "pool-reentry": POOL,
    "infra-retry": POOL,
    "quota-parked": POOL,
    "orphaned": POOL,
}


def project_ticket_states(events: list[dict[str, Any]]) -> dict[str, str]:
    """Pure projection: the full event log -> per-ticket projected
    ticket-table state.

    Walks ``events`` in log order and keeps, for each ticket, the state
    implied by that ticket's most recent state-defining event:

    - a DISPATCH event projects its ticket to ``in_flight`` (an open
      dispatch — no terminal disposition has been recorded for it yet);
    - a DISPOSITION event whose `disposition` appears in
      :data:`DISPOSITION_TO_STATE` projects the mapped state.

    A DISPATCH after a disposition is a RE-dispatch (pool re-entry
    already journaled) and correctly re-projects the ticket to
    ``in_flight``. Tickets with no events at all — or only events that
    define no state — are ABSENT from the returned mapping (there is
    nothing to compare a store row against).

    Pure: reads ``events`` only. It performs no mutation of the input
    list or its dicts, and never touches a ticket store or record plane.
    """
    projection: dict[str, str] = {}
    for ev in events:
        ticket = ev.get("ticket")
        if not isinstance(ticket, str):
            continue
        event_type = ev.get("event_type")
        if event_type == EventType.DISPATCH.value:
            projection[ticket] = IN_FLIGHT
        elif event_type == EventType.DISPOSITION.value:
            state = DISPOSITION_TO_STATE.get(ev.get("disposition"))
            if state is not None:
                projection[ticket] = state
    return projection
