"""Intake eligibility query + leases + red-at-birth (SPEC.md §3 `intake`
station, §9 state machine + leases, §6 test-authorship / red-at-birth,
§10 AC2/AC11).

**Eligibility is a deterministic query** (SPEC §3): a ticket is claimable
iff it is approved with a hash that still matches its current steering
(`approval.is_approval_valid`), every `blocks` predecessor has landed, and
it is not currently pinned under a live lease. `eligible()` never mutates
anything — it is read-only over the store, safe to call every poll cycle.

**Claims take a lease, never a bare flag** (SPEC §3/§9): `claim()`
re-verifies eligibility itself (never trusts a caller's earlier `eligible()`
call — the two can race across a poll interval) and, on success, freezes
the claim-time content (steering + execution + both hashes) into a
snapshot dict that is the *authority* for that dispatch. Later mutations to
the ticket row — a re-approval, a steering edit, an execution change — do
not reach back and change an in-flight dispatch's judgment; only a fresh
`claim()` sees fresh content.

**Lease expiry is recoverable, not punitive** (SPEC §9 `recover` station):
`expire_leases()` resets an orphaned ticket back to `pool` and clears only
the lease fields — `attempts_used`/`integration_failures` are lifetime
counters and survive expiry untouched, so an infra hiccup never gives a
ticket a discount on its retry budget.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Callable
from typing import Any

from stigmergy.approval import is_approval_valid, snapshot_hash, steering_hash
from stigmergy.checks import CheckOutcome

# States from which a ticket may be claimed (SPEC §9 state machine:
# `pool -> eligible -> claimed(lease) -> ...`). Any other state (claimed,
# in-flight, parked, gated, landed, rejected mid-transition, escalated,
# done) is not directly claimable here.
_CLAIMABLE_STATES = frozenset({"pool", "eligible"})

# The §6 ticket key vocabulary `intake` accepts (bead workspace-e2uh.152):
# the required five + the optional extras that ride straight into the
# ticket columns. `blocks` is a manifest-level wiring key consumed to build
# dependency edges — legal in a manifest entry but NOT a ticket column
# (mirrors `manifest.py`'s KNOWN_KEYS split).
_REQUIRED_INTAKE_KEYS = frozenset(
    {"id", "title", "functional_summary", "acceptance_criteria", "target_scope"}
)
_OPTIONAL_INTAKE_KEYS = frozenset(
    {"goal", "required_reading", "tier1_checks", "difficulty", "lane_hint",
     "rubric_only", "work_product"}
)


def ingest_manifest(store: Any, manifest: list) -> tuple[list[str], list[str]]:
    """Validate + insert a parsed manifest array (list of ticket dicts).

    Extracted verbatim from `cli._cmd_intake` (bead workspace-e2uh.152) so
    the decompose station driver can seed tickets through the SAME
    read-only-first validation the CLI uses — the CLI is now a thin
    prefix-wrapping delegate around this function, and its observable
    behavior (stdout/stderr/exit code) is byte-identical.

    Returns ``(inserted_ticket_ids, error_strings)``:

    - ``([], [])``-shape success: every entry validated and inserted
      (tickets inserted, `blocks` deps wired), ids in manifest order.
    - ``([], [err, ...])`` on ANY entry-level error: the function is
      READ-ONLY until the first insertion — a full first pass validates
      every entry (required keys, non-empty functional_summary, duplicate
      ids via the single O(n) id-count pass, store collisions, blocks
      resolution against store-or-manifest ids) BEFORE any insert. The
      first failing entry's message is returned and nothing is inserted
      (existing CLI behavior — one bad entry fails the whole manifest).

    The error strings are the CLI message bodies verbatim (no `stigmergy
    intake:` prefix, which the CLI adds when printing them to stderr).
    """
    # First pass: count each id's occurrences (single O(n) pass) and collect
    # the manifest id set for blocks resolution.
    id_counts = Counter()
    for entry in manifest:
        if isinstance(entry, dict) and "id" in entry:
            id_counts[entry["id"]] += 1
    manifest_ids = set(id_counts.keys())

    # Second pass: validate all entries before inserting anything.
    entries_to_insert: list[dict[str, Any]] = []

    for idx, entry in enumerate(manifest):
        if not isinstance(entry, dict):
            return [], [
                f"manifest entry {idx} must be a JSON object "
                f"(got {type(entry).__name__})"
            ]

        # Check required fields
        missing = _REQUIRED_INTAKE_KEYS - set(entry)
        if missing:
            entry_id = entry.get("id", f"<index {idx}>")
            sorted_missing = ", ".join(sorted(missing))
            return [], [
                f"manifest entry {idx} ({entry_id}) "
                f"missing required key(s): {sorted_missing}"
            ]

        # Check functional_summary is a non-empty string
        functional_summary = entry.get("functional_summary")
        if not isinstance(functional_summary, str) or not functional_summary.strip():
            entry_id = entry.get("id", f"<index {idx}>")
            return [], [
                f"manifest entry {idx} ({entry_id}) "
                f"'functional_summary' must be a non-empty string "
                f"(got {functional_summary!r})"
            ]

        # Check for duplicate ids within the manifest using pre-computed counts
        entry_id = entry["id"]
        if id_counts[entry_id] > 1:
            return [], [
                f"manifest entry {idx} ({entry_id}) has duplicate id"
            ]

        # Check if this id already exists in the store
        if store.get_ticket(entry_id) is not None:
            return [], [
                f"manifest entry {idx} ({entry_id}) ticket id already exists"
            ]

        # Validate blocks references (all must resolve to either store or
        # manifest entries)
        blocks = entry.get("blocks", [])
        for predecessor_id in blocks:
            store_has_it = store.get_ticket(predecessor_id) is not None
            manifest_has_it = predecessor_id in manifest_ids
            if not store_has_it and not manifest_has_it:
                return [], [
                    f"manifest entry {idx} ({entry_id}) "
                    f"blocks unresolved predecessor: {predecessor_id}"
                ]

        entries_to_insert.append(entry)

    # All entries validated — the write phase may begin.
    for entry in entries_to_insert:
        ticket_fields: dict[str, Any] = {}
        for key in ["functional_summary", "acceptance_criteria", "target_scope"]:
            ticket_fields[key] = entry[key]
        for key in _OPTIONAL_INTAKE_KEYS:
            if key in entry:
                ticket_fields[key] = entry[key]

        store.add_ticket(id=entry["id"], title=entry["title"], **ticket_fields)

    # Wire all dependencies
    for entry in entries_to_insert:
        blocks = entry.get("blocks", [])
        for predecessor_id in blocks:
            store.add_dep(entry["id"], predecessor_id)

    return [entry["id"] for entry in entries_to_insert], []


class LeaseError(Exception):
    """Raised by :func:`claim` when a ticket cannot be claimed: not
    eligible (unapproved, hash-stale, or a predecessor hasn't landed) or
    already held under a live, unexpired lease."""


def _deps_landed(store: Any, ticket_id: str) -> bool:
    """True iff every `blocks` predecessor of ``ticket_id`` is in the
    terminal `landed` state. A predecessor row that has vanished counts as
    not landed (fail closed, never fail open on a dangling dependency)."""
    for dep_id in store.deps_of(ticket_id):
        dep_row = store.get_ticket(dep_id)
        if dep_row is None or dep_row.get("state") != "landed":
            return False
    return True


def _lease_is_live(row: dict[str, Any], now: float) -> bool:
    """True iff ``row`` carries an unexpired lease. A `None`/absent
    `lease_expires_at` means unclaimed — never treated as live."""
    expires_at = row.get("lease_expires_at")
    return expires_at is not None and expires_at >= now


def _claimable_now(row: dict[str, Any], now: float) -> bool:
    """True iff ``row``'s state is claimable AND it holds no live lease."""
    return row.get("state") in _CLAIMABLE_STATES and not _lease_is_live(row, now)


def eligible(store: Any, *, now: float, steering_of: Callable[[str], dict[str, Any]]) -> list[str]:
    """Return the ids of every currently-claimable ticket (SPEC §3/§10 AC2).

    A ticket is eligible iff ALL of:

    1. ``is_approval_valid(row, steering_of(id))`` — approved and its
       stored hash still matches the ticket's *current* steering.
    2. Every predecessor in ``store.deps_of(id)`` is `landed`.
    3. It is not currently held under a live (unexpired) lease, and its
       state is claimable (`pool`/`eligible`).

    Pure query — reads ``store`` via ``get_ticket``/``list_tickets``/
    ``deps_of`` only, never writes. ``steering_of`` is a callable
    ``id -> current-steering-dict``, injected by callers (the real loop
    assembles it from the rig's tickets + context/checks/lane state; tests
    inject a plain dict lookup). The result is sorted by ticket id so the
    query is deterministic even if the store's own row order ties.
    """
    result: list[str] = []
    for row in store.list_tickets():
        ticket_id = row["id"]
        # Short-circuit on the raw `approved` flag before ever calling
        # `steering_of` — an unapproved ticket (e.g. an unlanded
        # predecessor that was never itself put up for triage) has no
        # obligation to have a steering entry at all.
        if not row.get("approved"):
            continue
        if not is_approval_valid(row, steering_of(ticket_id)):
            continue
        if not _deps_landed(store, ticket_id):
            continue
        if not _claimable_now(row, now):
            continue
        result.append(ticket_id)
    return sorted(result)


def claim(
    store: Any,
    ticket_id: str,
    *,
    owner: str,
    dispatch_id: str,
    ttl_seconds: float,
    now: float,
    steering: dict[str, Any],
    execution: dict[str, Any],
) -> dict[str, Any]:
    """Claim ``ticket_id`` under a fresh lease (ATOMIC conditional UPDATE
    with rowcount-based winner detection — SPEC §5 + ticket .TBD).

    Re-verifies eligibility itself against the *given* ``steering`` (never
    trusts that a caller's earlier `eligible()` call is still valid — the
    two can race across a poll interval): approved + hash-valid, all
    predecessors landed. Then attempts an ATOMIC conditional claim via
    ``store.atomic_claim_ticket()`` — a single SQL UPDATE statement that
    sets the lease fields and state ONLY if the ticket is currently
    unclaimed (lease_owner IS NULL) and in a claimable state. Winner is
    determined purely by the UPDATE's rowcount (==1 means this thread won).

    Any failure (eligibility or atomic claim loss) raises :class:`LeaseError`.

    On success: ``lease_owner``, ``lease_dispatch_id``,
    ``lease_expires_at = now + ttl_seconds``, ``lease_heartbeat_at = now``,
    ``state = 'claimed'`` have all been persisted. Returns the FROZEN claim
    snapshot — a fresh dict holding deep copies of ``steering``/``execution``
    plus their hashes. This snapshot, not the live row, is the authority for
    the dispatch: subsequent mutation of the ticket row (re-approval, steering
    edit, execution drift) cannot reach back and change what was already
    returned here.
    """
    row = store.get_ticket(ticket_id)
    if row is None:
        raise LeaseError(f"no such ticket: {ticket_id!r}")

    if not is_approval_valid(row, steering):
        raise LeaseError(
            f"ticket {ticket_id!r} is not eligible: unapproved or hash-stale steering"
        )
    if not _deps_landed(store, ticket_id):
        raise LeaseError(f"ticket {ticket_id!r} is not eligible: predecessor(s) not landed")

    lease_expires_at = now + ttl_seconds
    won = store.atomic_claim_ticket(
        ticket_id,
        owner=owner,
        dispatch_id=dispatch_id,
        lease_expires_at=lease_expires_at,
        lease_heartbeat_at=now,
    )

    if not won:
        raise LeaseError(
            f"ticket {ticket_id!r} claim failed: "
            "already claimed by another thread, or not in a claimable state"
        )

    return {
        "ticket_id": ticket_id,
        "owner": owner,
        "dispatch_id": dispatch_id,
        "lease_expires_at": lease_expires_at,
        "lease_heartbeat_at": now,
        "steering_hash": steering_hash(steering),
        "snapshot_hash": snapshot_hash(steering, execution),
        "steering": copy.deepcopy(steering),
        "execution": copy.deepcopy(execution),
    }


def heartbeat(store: Any, ticket_id: str, *, now: float, ttl_seconds: float) -> None:
    """Extend a held lease: ``lease_expires_at = now + ttl_seconds``,
    ``lease_heartbeat_at = now``. Does not touch `state` or any counter."""
    store.update_ticket(
        ticket_id,
        lease_expires_at=now + ttl_seconds,
        lease_heartbeat_at=now,
    )


#: The only states in which a ticket may legitimately carry a lease.
#: Anything else with a stale lease row is either terminal (`landed`/
#: `rejected`/`done`), mid-pipeline (`parked`/`gated`/`tier1_green` —
#: demotion would corrupt its pipeline position), or `escalated` (whose
#: re-entry is the operator's `resume` verb, never a silent expiry).
#: Caught live on quotagov01 2026-08-31: a daemon restart's recover()
#: sweep demoted 3 landed + 1 escalated tickets whose land/escalation
#: paths had left stale lease columns behind.
_EXPIRABLE_STATES: frozenset[str] = frozenset({"claimed", "in_flight"})


def expire_leases(store: Any, *, now: float) -> list[str]:
    """Reset every orphaned (expired-lease) ticket back to `pool`.

    A ticket qualifies iff it is in an active-dispatch state
    (`claimed`/`in_flight`) AND carries a non-``None` `lease_expires_at`
    strictly less than ``now``. For each: `state` resets to `pool` and all
    four lease fields (`lease_owner`, `lease_dispatch_id`,
    `lease_expires_at`, `lease_heartbeat_at`) clear to `None` — recoverable
    orphan, not a mystery (SPEC §9). `attempts_used`/`integration_failures`
    are lifetime-per-ticket counters (SPEC §9 "Retry semantics") and are
    never named here, so they are never touched by expiry.

    Tickets in any other state are never demoted, even if their rows carry
    stale lease columns (terminal states are final; escalated re-entry is
    operator-authority via `resume`).

    Returns the expired ticket ids, sorted for determinism.
    """
    expired: list[str] = []
    for row in store.list_tickets():
        expires_at = row.get("lease_expires_at")
        if row.get("state") not in _EXPIRABLE_STATES:
            continue
        if expires_at is not None and expires_at < now:
            store.update_ticket(
                row["id"],
                state="pool",
                lease_owner=None,
                lease_dispatch_id=None,
                lease_expires_at=None,
                lease_heartbeat_at=None,
            )
            expired.append(row["id"])
    return sorted(expired)


# Outcomes that unambiguously prove the acceptance tests already passed
# against the pre-ticket tree (vacuous — nothing left for the ticket to
# do). Only a clean, first-attempt PASS counts: FLAKY (passed on a rerun)
# and ERROR (couldn't even determine an outcome) cannot *prove* vacuity,
# and admission control fails closed on "not proven vacuous" -> red.
_VACUOUS_OUTCOMES = frozenset({CheckOutcome.PASS})


def red_at_birth_ok(acceptance_outcome: Any) -> bool:
    """SPEC §6 red-at-birth: True iff the acceptance tests FAILED against
    the pre-ticket tree (red => admit); False iff they unambiguously
    PASSED (vacuous => auto-reject).

    Accepts three input shapes:

    - :class:`stigmergy.checks.CheckOutcome` — only `PASS` is vacuous;
      `FAIL`, `FLAKY`, and `ERROR` all admit (fail closed on admission:
      an outcome that doesn't *prove* the test was already green must not
      be treated as if it were).
    - ``bool`` — the "did it pass" convention: ``True`` means passed
      (vacuous, rejected), ``False`` means failed (red, admitted).
    - ``int`` (process exit code) — the Unix convention: ``0`` means
      passed (vacuous, rejected), any nonzero code means failed (red,
      admitted).

    Raises :class:`TypeError` for any other input type — there is no
    silent default interpretation of an unrecognized outcome shape.
    """
    if isinstance(acceptance_outcome, CheckOutcome):
        return acceptance_outcome not in _VACUOUS_OUTCOMES
    if isinstance(acceptance_outcome, bool):
        return acceptance_outcome is False
    if isinstance(acceptance_outcome, int):
        return acceptance_outcome != 0
    raise TypeError(f"unsupported acceptance_outcome type: {type(acceptance_outcome)!r}")
