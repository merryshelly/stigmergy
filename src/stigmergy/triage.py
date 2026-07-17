"""Triage-promotion library (bead .42 — SPEC §6 item 10 / D2 / D15 / D1).

`promote_proposal` completes a filed proposal (bead .38's `filed_tickets`
row) into a real §6 ticket. The orchestrator (a trusted human-adjacent
role, D2) authors the completion — most importantly `functional_summary`,
the human-facing judgment field the operator later verdicts at `approve`
time. The new ticket ALWAYS lands UNAPPROVED: promotion and approval are
distinct acts (D2) — `store.promote_filed_ticket` enforces this atomically
(one transaction: ticket insert + filed-row tombstone).

`record_triage_event` writes the v0 attribution audit line (D1:
agent-asserted acting agent + operator session) for the human triage acts
(APPROVAL/UNAPPROVAL/TRIAGE_REJECTED) into the ONE event log.
"""

from __future__ import annotations

import time
from typing import Any

from stigmergy.records import EventType, RecordPlane, make_event

# Required §6 completion keys a promotion spec MUST carry (beyond the filed
# row's own origin/discovery fields, which are untouched). `functional_summary`
# is additionally required to be a NON-EMPTY string after `.strip()` — it is
# now a signed steering field (bead .42 §A), not merely descriptive text.
_REQUIRED_PROMOTION_KEYS = frozenset(
    {"id", "title", "functional_summary", "acceptance_criteria", "tier1_checks", "target_scope"}
)

# Optional §6 completion keys forwarded to `promote_filed_ticket` verbatim
# when present in the spec (excludes `id`/`title`, which have their own
# keyword-only parameters on `promote_filed_ticket`, and `blocks`, which is
# consumed here to build dependency edges, not a ticket column).
_OPTIONAL_PROMOTION_KEYS = frozenset(
    {
        "goal",
        "required_reading",
        "difficulty",
        "lane_hint",
        "rubric_only",
        "work_product",
    }
)

# The other required §6 fields (besides id/title, which map to
# promote_filed_ticket's own keyword params) forwarded as ticket fields.
_REQUIRED_TICKET_FIELD_KEYS = frozenset(
    {"functional_summary", "acceptance_criteria", "tier1_checks", "target_scope"}
)


class TriageError(Exception):
    """Raised on any triage-promotion validation failure (bead .42).

    Validation always happens BEFORE any write — a `TriageError` guarantees
    nothing was written: no ticket row created, and the filed row (if it
    exists) is left untriaged.
    """


def promote_proposal(store: Any, *, filed_id: str, spec: dict) -> str:
    """Complete a filed proposal into a §6 ticket (SPEC §6 item 10 / D2).

    ``spec`` is the orchestrator-authored completion. Required keys: ``id``,
    ``title``, ``functional_summary`` (must be a non-empty string after
    ``.strip()`` — TriageError if empty, blank, or missing), plus
    ``acceptance_criteria``, ``tier1_checks``, ``target_scope``. Optional
    keys: ``goal``, ``required_reading``, ``difficulty``, ``lane_hint``,
    ``rubric_only``, ``work_product``, ``blocks`` (a list of predecessor
    ticket ids).

    ``filed_id`` must name an existing, untriaged `filed_tickets` row
    (TriageError otherwise). All validation happens BEFORE any write —
    on any `TriageError`, nothing lands: no ticket row, filed row stays
    untriaged.

    Calls ``store.promote_filed_ticket(filed_id=filed_id, ticket_id=
    spec['id'], title=spec['title'], **<other §6 fields present in spec>)``
    — the ticket lands UNAPPROVED (promotion and approval are distinct
    acts). For each predecessor id in ``spec.get('blocks', [])``, records
    ``store.add_dep(spec['id'], predecessor_id)`` (SPEC §6 item 9:
    `deps_of(ticket_id)` returns these).

    Returns the new ticket id (``spec['id']``). Emits NO event — promotion
    is recorded structurally via the filed row's `resulting_ticket_id` +
    the new unapproved ticket; the authority act that carries attribution
    is the later `approve`.
    """
    # Validate the filed row exists and is untriaged BEFORE touching spec
    # contents, so a bad filed_id fails fast regardless of spec shape.
    filed_rows = store.list_filed_tickets(triaged=False)
    filed_ids = {row["id"] for row in filed_rows}
    if filed_id not in filed_ids:
        raise TriageError(f"no untriaged filed ticket: {filed_id!r}")

    missing = _REQUIRED_PROMOTION_KEYS - set(spec)
    if missing:
        raise TriageError(f"promotion spec missing required key(s): {sorted(missing)}")

    functional_summary = spec.get("functional_summary")
    if not isinstance(functional_summary, str) or not functional_summary.strip():
        raise TriageError(
            "promotion spec 'functional_summary' must be a non-empty string "
            f"(got {functional_summary!r})"
        )

    ticket_fields: dict[str, Any] = {}
    for key in _REQUIRED_TICKET_FIELD_KEYS:
        ticket_fields[key] = spec[key]
    for key in _OPTIONAL_PROMOTION_KEYS:
        if key in spec:
            ticket_fields[key] = spec[key]

    store.promote_filed_ticket(
        filed_id=filed_id,
        ticket_id=spec["id"],
        title=spec["title"],
        **ticket_fields,
    )

    for predecessor_id in spec.get("blocks", []):
        store.add_dep(spec["id"], predecessor_id)

    return spec["id"]


def record_triage_event(
    record_plane: RecordPlane,
    *,
    event_type: EventType,
    rig: str,
    subject_id: str,
    outcome: str,
    acting_agent: str,
    operator_session: str,
    approval_hash: str | None = None,
    reason: str | None = None,
    now: float | None = None,
) -> None:
    """Build and append ONE attribution audit event (bead .42, D1).

    ``event_type`` is one of APPROVAL/UNAPPROVAL/TRIAGE_REJECTED.
    ``subject_id`` is the ticket id (approve/unapprove) or the filed
    proposal id (reject). ``acting_agent``/``operator_session`` are the
    v0 agent-asserted audit line — no read-only-rig hardening, D1's policy
    boundary. ``approval_hash``/``reason`` are included in the built event
    only when not ``None`` (both are optional extras on the triage payload
    shape). ``ts`` defaults to ``time.time()`` if ``now`` is not supplied.
    """
    fields: dict[str, Any] = {
        "rig": rig,
        "subject_id": subject_id,
        "outcome": outcome,
        "acting_agent": acting_agent,
        "operator_session": operator_session,
        "ts": now if now is not None else time.time(),
    }
    if approval_hash is not None:
        fields["approval_hash"] = approval_hash
    if reason is not None:
        fields["reason"] = reason

    event = make_event(event_type, **fields)
    record_plane.append(event)
