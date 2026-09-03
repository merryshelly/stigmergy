"""Read-only observability: the AC12 status view + the operator's no-SQL
ticket-read surface (SPEC.md §9 "Notifications", §10 AC12, OQ3;
bead .23 build spec §2).

**Spend is RECONSTRUCTED, never live (bead .23 build spec §0 DECISION).**
`spend.SpendLeash` is stateful/accrual and lives only in the daemon's
in-memory process; a standalone `status` run has no access to it. SPEC §12
says spend is derived from events, so :func:`reconstruct_spend` sums
`computed_usd` and counts `DISPATCH`/`GATE` events straight from
`records.RecordPlane.read_events()` (already checksum-validated) and
compares the totals to the charter's `[loop.budgets]` caps. This module
never imports or instantiates `SpendLeash`.

**No new persistence, no `rig.py` schema changes.** Every function here
reads through the already-frozen interfaces of `rig.RigStore`,
`records.RecordPlane`, and `notify.NotificationStore` — nothing here
writes to any of them.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stigmergy import quotagov, statemachine

# Cap on how many recent DISPOSITION events `gather_status` surfaces in
# `last_verdicts` (SPEC §2 frozen interface: "last N (=5)").
_LAST_VERDICTS_LIMIT = 5

# Default tail length for render_event_tail.
_DEFAULT_EVENT_TAIL_LIMIT = 25

# The pool/eligible states that count toward queue age (a ticket waiting
# to be worked, not one already claimed/in-flight/parked/terminal).
_QUEUE_STATES: tuple[str, ...] = (statemachine.POOL, statemachine.ELIGIBLE)


@dataclass(frozen=True)
class Status:
    """The full AC12 status snapshot (SPEC §2 frozen interface)."""

    state_counts: dict[str, int]
    last_verdicts: list[dict[str, Any]]
    spend: dict[str, Any]
    lane_parks: dict[str, dict[str, Any]]
    daemon_heartbeat_age: float | None
    queue_age: float | None
    escalated_unnotified: int
    disk_free_bytes: int
    disk_ok: bool
    untriaged_filings: int


def reconstruct_spend(
    record_plane: Any,
    *,
    usd_cap: float,
    dispatches_cap: int,
    gate_calls_cap: int,
) -> dict[str, Any]:
    """Reconstruct spend totals purely from the event log (SPEC §12).

    ``metered_spent`` sums `computed_usd` over every event where it is a
    real number (bool excluded — mirrors `records._validate_computed_usd`'s
    own guard against a stray bool masquerading as a number).
    ``unbudgetable_events`` counts events whose `computed_usd` is exactly
    the string `"unbudgetable"`. ``dispatches_used``/``gate_calls_used``
    count `DISPATCH`/`GATE` typed events respectively (disposition/check/
    integration/notify/report events never count toward either).
    """
    events = record_plane.read_events()

    metered_spent = 0.0
    unbudgetable_events = 0
    dispatches_used = 0
    gate_calls_used = 0

    for ev in events:
        usd = ev.get("computed_usd")
        if isinstance(usd, str):
            if usd == "unbudgetable":
                unbudgetable_events += 1
        elif isinstance(usd, int | float) and not isinstance(usd, bool):
            metered_spent += float(usd)

        event_type = ev.get("event_type")
        if event_type == "dispatch":
            dispatches_used += 1
        elif event_type == "gate":
            gate_calls_used += 1

    return {
        "metered_spent": metered_spent,
        "usd_cap": usd_cap,
        "dispatches_used": dispatches_used,
        "dispatches_cap": dispatches_cap,
        "gate_calls_used": gate_calls_used,
        "gate_calls_cap": gate_calls_cap,
        "unbudgetable_events": unbudgetable_events,
    }


def _budgets_from_charter(charter_resolved: dict[str, Any]) -> dict[str, Any]:
    """Pull `[loop.budgets]`'s three caps out of an already-resolved charter
    mapping. No in-code defaults here (unlike `charter.py`'s own merge
    step) — `gather_status` is handed an already-resolved charter, so a
    missing key is a caller bug, not a legitimate absence; it surfaces as
    `None` rather than silently substituting a number that isn't the
    charter's own.
    """
    loop_cfg = charter_resolved.get("loop") if isinstance(charter_resolved, dict) else None
    budgets = loop_cfg.get("budgets") if isinstance(loop_cfg, dict) else None
    if not isinstance(budgets, dict):
        budgets = {}
    return {
        "usd": budgets.get("usd"),
        "dispatches": budgets.get("dispatches"),
        "gate_calls": budgets.get("gate_calls"),
    }


def _empty_lane_parks(lane_names: list[str]) -> dict[str, dict[str, Any]]:
    """The not-parked, no-evidence shape for a set of lanes: one entry per
    lane, every field explicitly empty (a lane whose governance inputs are
    absent is not-parked, never a missing key)."""
    return {
        name: {"parked": False, "parked_until": None, "escalation_fired": False}
        for name in lane_names
    }


def gather_lane_parks(
    *,
    charter_resolved: dict[str, Any],
    registry: Any = None,
    governor: quotagov.QuotaGovernor | None = None,
    escalation_ceiling_s: float = quotagov.DEFAULT_ESCALATION_CEILING_S,
    now: float,
) -> dict[str, dict[str, Any]]:
    """The governor's park state per charter lane — the park-half of the
    spend/leash status reporting (quotagov governor-state integration).

    Mirrors the daemon's claim filter exactly: each lane's model is resolved
    through the registry, and the governor's decision for the lane's
    provider quota state is taken from :func:`quotagov.decide` (the same
    pure API the daemon parks on). Every lane named in the charter gets an
    entry: ``parked`` (bool), ``parked_until`` (the epoch-seconds park
    deadline, or ``None``), and ``escalation_fired`` (bool — the governor's
    escalation-due signal for this park). Lanes whose pricing class is not
    subscription, whose model is missing from the registry, and the whole
    mapping when ``governor`` is ``None`` (no feed configured) or its
    provider map is empty (no quota state yet), are all
    not-parked/``None``/not-escalated — explicitly empty, never a missing
    key, and never a park without evidence.

    ``registry`` and ``governor`` default to ``None``: a standalone
    ``status`` run has no live feed state to fold (a ``QuotaGovernor`` is
    built and folded only in the daemon's in-memory process, exactly like
    ``SpendLeash``), so standalone runs report every lane as not-parked.
    """
    lanes_cfg = charter_resolved.get("lanes") or {}
    lane_names = sorted(lanes_cfg)
    parks = _empty_lane_parks(lane_names)
    if governor is None or not governor.providers or registry is None:
        return parks
    for name in lane_names:
        cfg = lanes_cfg.get(name)
        if not isinstance(cfg, dict):
            continue
        try:
            entry = registry.resolve(cfg.get("model"))
        except Exception:  # noqa: BLE001 - a resolve miss is not-parked, never a crash
            continue
        provider = entry.oa_provider_key or entry.provider
        decision = quotagov.decide(
            entry,
            governor.providers.get(provider),
            now=now,
            escalation_ceiling_s=escalation_ceiling_s,
        )
        if decision.decision != "dispatch":
            parks[name] = {
                "parked": True,
                "parked_until": decision.park_until,
                "escalation_fired": decision.escalation_due,
            }
    return parks


def gather_status(
    *,
    store: Any,
    record_plane: Any,
    notification_store: Any,
    charter_resolved: dict[str, Any],
    disk_path: str,
    min_disk_bytes: int | None = None,
    now: float,
    registry: Any = None,
    governor: quotagov.QuotaGovernor | None = None,
    escalation_ceiling_s: float = quotagov.DEFAULT_ESCALATION_CEILING_S,
) -> Status:
    """Assemble the full AC12 :class:`Status` snapshot from the rig's
    stores (SPEC §2 frozen interface).

    ``registry``/``governor`` (optional) feed :func:`gather_lane_parks` —
    the per-lane governor park state reported alongside the spend/leash
    totals. Both default to ``None`` (standalone status run: no live
    quota-state source), in which case every lane reports not-parked."""
    state_counts = {s: len(store.list_tickets(state=s)) for s in statemachine.STATES}

    disposition_events = [
        ev for ev in record_plane.read_events() if ev.get("event_type") == "disposition"
    ]
    # `read_events()` returns file/insertion order, which is chronological
    # (append-only log) — reverse it for newest-first rather than sorting
    # by `ts`, since `ts` defaults to `time.time()` and rapid test seeding
    # can produce identical/near-identical timestamps (unstable sort).
    newest_first = list(reversed(disposition_events))[:_LAST_VERDICTS_LIMIT]
    last_verdicts = [
        {
            "ticket": ev.get("ticket"),
            "disposition": ev.get("disposition"),
            "reason": ev.get("reason"),
        }
        for ev in newest_first
    ]

    caps = _budgets_from_charter(charter_resolved)
    spend = reconstruct_spend(
        record_plane,
        usd_cap=caps["usd"],
        dispatches_cap=caps["dispatches"],
        gate_calls_cap=caps["gate_calls"],
    )
    lane_parks = gather_lane_parks(
        charter_resolved=charter_resolved,
        registry=registry,
        governor=governor,
        escalation_ceiling_s=escalation_ceiling_s,
        now=now,
    )

    heartbeat_raw = store.get_meta("daemon_heartbeat_at")
    daemon_heartbeat_age = None if heartbeat_raw is None else now - float(heartbeat_raw)

    queue_tickets = [t for state in _QUEUE_STATES for t in store.list_tickets(state=state)]
    queue_age = None
    if queue_tickets:
        oldest_created_at = min(t["created_at"] for t in queue_tickets)
        queue_age = now - oldest_created_at

    escalated_unnotified = sum(
        1 for intent in notification_store.pending() if intent.kind == "escalation"
    )

    disk_free_bytes = shutil.disk_usage(disk_path).free
    disk_ok = True if min_disk_bytes is None else disk_free_bytes >= min_disk_bytes

    untriaged_filings = store.count_untriaged_filings()

    return Status(
        state_counts=state_counts,
        last_verdicts=last_verdicts,
        spend=spend,
        lane_parks=lane_parks,
        daemon_heartbeat_age=daemon_heartbeat_age,
        queue_age=queue_age,
        escalated_unnotified=escalated_unnotified,
        disk_free_bytes=disk_free_bytes,
        disk_ok=disk_ok,
        untriaged_filings=untriaged_filings,
    )


def _fmt_age(age: float | None, *, absent_label: str) -> str:
    return absent_label if age is None else f"{age:.1f}s ago"


def render_status(status: Status) -> str:
    """Readable multi-line status text (SPEC §2 frozen interface). Never
    relies on colour to carry meaning (deuteranopia note) — every value is
    prefixed with its own text label."""
    lines: list[str] = []

    lines.append("=== ticket states ===")
    for state in sorted(status.state_counts):
        lines.append(f"  {state}: {status.state_counts[state]}")

    lines.append("")
    lines.append("=== spend vs budgets ===")
    spend = status.spend
    lines.append(
        f"  usd: {spend['metered_spent']:.4f} / cap {spend['usd_cap']}"
    )
    lines.append(
        f"  dispatches: {spend['dispatches_used']} / cap {spend['dispatches_cap']}"
    )
    lines.append(
        f"  gate_calls: {spend['gate_calls_used']} / cap {spend['gate_calls_cap']}"
    )
    lines.append(f"  unbudgetable_events: {spend['unbudgetable_events']}")

    lines.append("")
    lines.append("=== quota parks ===")
    if status.lane_parks:
        for lane in sorted(status.lane_parks):
            park = status.lane_parks[lane]
            if not park["parked"]:
                lines.append(
                    f"  {lane}: parked=no  parked_until=-  escalation_fired=no"
                )
                continue
            until = (
                f"{park['parked_until']:.0f}"
                if park["parked_until"] is not None
                else "-"
            )
            fired = "yes" if park["escalation_fired"] else "no"
            lines.append(
                f"  {lane}: parked=yes  parked_until={until}  escalation_fired={fired}"
            )
    else:
        lines.append("  (no lanes)")

    lines.append("")
    lines.append("=== health ===")
    heartbeat_text = _fmt_age(status.daemon_heartbeat_age, absent_label="no heartbeat")
    lines.append(f"  daemon_heartbeat_age: {heartbeat_text}")
    lines.append(
        f"  queue_age: {_fmt_age(status.queue_age, absent_label='queue empty')}"
    )
    lines.append(f"  escalated_unnotified: {status.escalated_unnotified}")
    lines.append(
        f"  disk_free_bytes: {status.disk_free_bytes} "
        f"(disk_ok={'yes' if status.disk_ok else 'no'})"
    )
    lines.append(f"  untriaged_filings: {status.untriaged_filings}")

    lines.append("")
    lines.append("=== last verdicts (newest first) ===")
    if status.last_verdicts:
        for verdict in status.last_verdicts:
            reason = verdict.get("reason")
            reason_part = f" ({reason})" if reason else ""
            lines.append(f"  {verdict['ticket']}: {verdict['disposition']}{reason_part}")
    else:
        lines.append("  (none)")

    return "\n".join(lines)


# --- no-SQL ticket-read surface (SB oversight requirement, OQ3) ------------


def render_ticket_list(store: Any) -> str:
    """`bd-list`-style one-line-per-ticket rendering, ordered by
    `created_at` (SPEC §2 frozen interface)."""
    tickets = store.list_tickets()
    if not tickets:
        return "(no tickets)"

    lines: list[str] = []
    for t in tickets:
        difficulty = t.get("difficulty")
        difficulty_part = f"{difficulty}" if difficulty is not None else "-"
        rung = t.get("current_rung")
        rung_part = rung if rung is not None else "-"
        attempts = t.get("attempts_used") or 0
        approved = 1 if t.get("approved") else 0
        lines.append(
            f"{t['id']}  [{t['state']}]  {difficulty_part}  rung={rung_part}  "
            f"attempts={attempts}  approved={approved}  {t['title']}"
        )
    return "\n".join(lines)


def render_ticket_detail(store: Any, ticket_id: str) -> str:
    """`bd-show`-style rendering of every column for one ticket, one field
    per line (SPEC §2 frozen interface). A missing id returns a clear
    "no such ticket" line rather than raising."""
    ticket = store.get_ticket(ticket_id)
    if ticket is None:
        return f"no such ticket: {ticket_id}"

    lines: list[str] = []
    for key in sorted(ticket):
        lines.append(f"{key}: {ticket[key]}")
    return "\n".join(lines)


def render_event_tail(events: list[dict[str, Any]], n: int) -> str:
    """Render the last `n` events from a list, one line per event (chronological
    order in file = one-line-per-event, bounded to last n).

    Each line shows: human-readable timestamp, event_type, ticket, and
    outcome/disposition field (whichever is present).

    If events is empty, return a clear placeholder line.
    Never filter events based on content — every event in the trailing window
    is shown.
    """
    if not events:
        return "(no events)"

    trailing_events = list(reversed(events))[:n]
    trailing_events.reverse()

    lines: list[str] = []
    for ev in trailing_events:
        ts = ev.get("ts")
        ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else "unknown"

        event_type = ev.get("event_type", "unknown")
        ticket = ev.get("ticket", "-")

        outcome = ev.get("outcome")
        disposition = ev.get("disposition")
        result_part = ""
        if outcome is not None:
            result_part = f" outcome={outcome}"
        elif disposition is not None:
            result_part = f" disposition={disposition}"
        elif isinstance(ev.get("tickets"), list):
            # decompose-intake emission (bead .174): the count + D17 outcome.
            approved = ev.get("approved")
            result_part = (
                f" tickets={len(ev['tickets'])}"
                f" approved={'yes' if approved else 'no'}"
            )

        lines.append(f"{ts_str}  {event_type}  {ticket}{result_part}")

    return "\n".join(lines)


def render_daemon_liveness(rig_root: Path, *, ps_runner=subprocess.run) -> str:
    """Check daemon liveness from pidfile at <rig_root>/daemon.pid.

    Returns a human-readable status line covering:
    - Pidfile missing
    - Pidfile content unparseable
    - Process not running (daemon exited)
    - Process running (with elapsed runtime from ps if available)

    ps_runner is an injectable subprocess-runner parameter (defaults to
    subprocess.run) for testing.
    """
    pidfile_path = Path(rig_root) / "daemon.pid"

    if not pidfile_path.exists():
        return "daemon: pidfile not found"

    try:
        pid_str = pidfile_path.read_text(encoding="utf-8").strip()
        pid = int(pid_str)
    except (ValueError, OSError):
        return "daemon: pidfile content is invalid"

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return f"daemon: DAEMON EXITED (pid {pid})"
    except PermissionError:
        return f"daemon: alive (pid {pid}, etime unavailable)"

    result = ps_runner(
        ["ps", "-p", str(pid), "-o", "etime="],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode == 0:
        etime = result.stdout.strip() if result.stdout else "unavailable"
        return f"daemon: alive (pid {pid}, etime {etime})"
    else:
        return f"daemon: alive (pid {pid}, etime unavailable)"
