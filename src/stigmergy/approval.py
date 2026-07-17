"""Approval-hash integrity mechanism (SPEC.md §4 "Approval integrity", §10 AC2/AC11).

Human triage approval signs a **content hash over the ticket's inputs**,
split into two groups with deliberately different mutability rules:

- **STEERING fields** (human-signed judgment inputs): ticket text, checks,
  rubric, target scope, lane, prompt bytes, context set. (``target_scope``
  was ratified 2026-07-16 as a 7th signed field beyond the original six:
  a post-approval change to a worker's declared file scope now forces
  re-approval. ``steering.derive_steering`` is the single authoritative
  producer of this dict, used identically at approval time and at the
  live eligibility re-check.) Any mutation of a steering
  field atomically invalidates approval — the ticket leaves the eligible
  pool until a human re-approves it. This is the mechanism, not a policy
  the loop could talk itself out of: the stored `approval_hash` is over
  steering alone, so a changed steering field always recomputes to a
  different hash and `is_approval_valid` always goes false.
- **EXECUTION fields** (mechanism inputs that may legitimately drift
  without re-approval): base OID, resolved model+version, image digest,
  egress policy. A fresh base OID after an integration failure, or a
  resolved model version bump, does not require a human — it re-enters
  through :func:`rehash_execution`, a *scoped* re-hash that proves the
  steering hash is unchanged (steering didn't move) while producing a new
  `snapshot_hash` over both groups for the record plane to log against
  (as an `integration-reconcile` attempt, SPEC §9 `attempt_kind`).

`approve()` is the single authoritative approval act (SPEC §3: "the only
authoritative `approved` — no LLM, including Merry, can apply it"). It is
a plain function precisely so no LLM-invoked code path can call it as part
of automated triage in v0 — it represents the human act itself, not a
judgment the loop performs.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _canonical_hash(data: dict[str, Any]) -> str:
    """sha256 over a canonical (sorted-key, whitespace-free JSON) serialization.

    Mirrors `registry.py`/`records.py`'s `_canonical_hash` — stable across
    key reordering at every nesting level, sensitive to any value change.
    """
    blob = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def steering_hash(steering: dict[str, Any]) -> str:
    """Canonical sha256 over the STEERING fields alone (sorted-key).

    This is exactly what a human's approval act signs, and exactly what
    `approval_hash` on a ticket row stores — never the execution fields,
    so execution drift can never (accidentally or otherwise) show up as a
    steering-hash change.
    """
    return _canonical_hash(steering)


def snapshot_hash(steering: dict[str, Any], execution: dict[str, Any]) -> str:
    """Full claim-time hash over BOTH steering and execution fields.

    Wrapped as ``{"steering": steering, "execution": execution}`` before
    hashing so `sort_keys=True` gives order-insensitivity at every level of
    both groups, not just the top one. This is the hash a claim snapshot
    and its record-plane events carry — the complete picture of what was
    actually dispatched, not just what a human signed.
    """
    return _canonical_hash({"steering": steering, "execution": execution})


def approve(store: Any, ticket_id: str, *, steering: dict[str, Any]) -> None:
    """The ONLY authoritative approval act (SPEC §3/§4).

    Sets ``approved=1`` and ``approval_hash=steering_hash(steering)`` on
    the ticket. Represents the human triage act — no LLM path should call
    this in production; it is a plain function here precisely so nothing
    else in this codebase can pretend to be a human by calling it as a
    side effect of some other automated judgment.
    """
    store.update_ticket(ticket_id, approved=1, approval_hash=steering_hash(steering))


def unapprove(store: Any, ticket_id: str) -> None:
    """Withdraw approval WITHOUT a steering edit (bead .42, SPEC §4 minors).

    Sets ``approved=0`` and ``approval_hash=None`` on the ticket via
    ``store.update_ticket`` — the symmetric inverse of :func:`approve`.
    Like `approve`, this is a plain, pure function that only touches the
    store: it represents the human withdraw act itself (not an automated
    judgment), so it stays free of any record-plane side effect — callers
    that want an audit trail append the attribution event separately
    (`triage.record_triage_event`).
    """
    store.update_ticket(ticket_id, approved=0, approval_hash=None)


def is_approval_valid(ticket_row: dict[str, Any], current_steering: dict[str, Any]) -> bool:
    """True iff the ticket is approved AND its stored hash matches steering now.

    ``ticket_row['approved']`` must be truthy (SQLite stores it as
    ``0``/``1``) and ``ticket_row['approval_hash']`` must equal
    ``steering_hash(current_steering)``. A mutated steering field recomputes
    to a different hash, so this — and nothing else — is what
    de-eligibilizes a ticket the instant its steering moves.
    """
    if not ticket_row.get("approved"):
        return False
    stored_hash = ticket_row.get("approval_hash")
    if not stored_hash:
        return False
    return stored_hash == steering_hash(current_steering)


def rehash_execution(steering: dict[str, Any], execution: dict[str, Any]) -> dict[str, str]:
    """Scoped re-hash for a changed execution field (e.g. a fresh base OID).

    Returns ``{'steering_hash': ..., 'snapshot_hash': ...}``.
    ``steering_hash`` is computed from ``steering`` alone and is therefore
    IDENTICAL to before the execution change — proof that steering didn't
    move and no human is needed. ``snapshot_hash`` folds in the new
    ``execution`` dict and so DOES change. Callers log this as an
    execution re-hash (SPEC §9 `attempt_kind` `integration-reconcile`).
    """
    return {
        "steering_hash": steering_hash(steering),
        "snapshot_hash": snapshot_hash(steering, execution),
    }
