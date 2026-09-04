"""D14 worker/critic/range-critic ticket-filing: harvest, validate, file,
emit (SPEC.md §3 filing capability, §4 propagation edge, §5 filing caps,
§7 harvest, §8 `ticket-filed` event; README D14; bead workspace-e2uh.38).

**Security class: moderate.** Worker-authored ticket text (`title`,
`description`, `evidence`) is an injection surface — a compromised or
merely careless worker controls those strings end to end. Containment is
STRUCTURAL, not disciplinary: a filed proposal is written to the separate
`filed_tickets` table (`rig.RigStore.add_filed_ticket`), which
`intake.claim`/`intake.eligible` never read (they iterate `tickets`
only). A proposal that is never a `tickets` row cannot be claimed —
physics, not a filter that can regress. This module never inserts into
`tickets`; that is bead .42's later triage-promotion job.

**Two entry points, one shared core:**

- :func:`file_proposals` — the reusable core (count-cap -> per-proposal
  shape/size validation -> hash -> insert -> emit). Reused verbatim by
  `.39` (critic filing, `origin_role="critic"`) and `.41` (range-report
  filing, `origin_role="range-critic"`).
- :func:`harvest_worker_filings` — the `.38` worker-specific path: safely
  resolves the fixed host-side harvest path
  (`<worktree>/.stigmergy/filed-tickets.json`), reads + parses it,
  NORMALIZES it (bead workspace-e2uh.188 salvage pipeline), then
  delegates to :func:`file_proposals` with `origin_role="worker"`.

The transport file is parsed as the append-friendly JSONL stream the
`file_ticket` tool actually emits — one proposal object per line — with a
backward-compat exception: a file whose WHOLE content is a single JSON
array (the legacy shape) is still read as that proposal list. A line that
fails JSON parsing or is not an object is a PER-LINE `bad-shape`
rejection (never a whole-harvest rejection) and does not count against
the filing count-cap.

**Salvage semantics (worker harvest path ONLY — bead workspace-e2uh.188).**
The contexthandoff01 evidence: the in-cage OA file_ticket suite polluted
the production transport with a byte-identical fixture block repeated N
times, and the old whole-batch count-cap (amendment D, "suspicious
worker -> drop all") then dropped the WHOLE file — including the
worker's real diagnostics buried at the tail. An over-cap TRANSPORT on
the worker channel can only arise from harness pollution/duplication (the
tool's call-time cap already bounds what one worker run can emit), so
punishing it by dropping everything punished the harness's own bug.
The harvest therefore normalizes before delegating: exact duplicates
collapse to their first occurrence (ONE aggregate `duplicate` event +
the `FilingResult.duplicates_collapsed` counter), the count cap counts
UNIQUE INSERTABLE candidates (post-dedup, post-shape, post-size) with
earlier-filings-stand semantics (the excess gets ONE aggregate
`count-cap-exceeded` event + per-entry `rejected` entries), and
shape/size rejects stay per-entry and do not consume cap slots. The
legacy array shape follows the SAME rules. `file_proposals` itself is
untouched: its amendment-D whole-batch count-cap stays fully in force
for the direct `.39`/`.41` callers, and its per-entry rejections are
the single source of the per-entry rejection events on the harvest path
(no double-emission).

**Never raises into the dispatch teardown path.** `harvest_worker_filings`
runs post-container-kill, inside the daemon's per-dispatch teardown
(SPEC amendment A). Every failure mode — missing file (silent no-op, by
design), unreadable/unsafe path, undecodable (non-UTF-8) bytes
(`malformed-json`), unparseable or non-object line (per-line `bad-shape`),
bad proposal shape, oversize proposal, too many proposals — is captured as
a :class:`FilingResult` with a `rejected` entry and/or a `ticket-filed`
rejection event; none of them propagate an exception to the caller.

**Honest zeros, not double-counted spend.** The harvest itself is a
host-side mechanism call, not an LLM invocation — no tokens are consumed,
no wall time is spent running an LLM, and the worker's own cost was
already recorded on its DISPATCH event. Every `ticket-filed` event
declares `tokens={"in":0,"cached":0,"out":0,"reasoning":0}`,
`computed_usd=0.0`, `wall_time_seconds=0.0` explicitly (never a default —
`records.make_event` has no default for `computed_usd`) so spend
reconstruction (SPEC §12, `status.reconstruct_spend`) never double-counts
a dispatch's cost through its filing side-effect.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from stigmergy.pathsafety import PathSafetyError, reject_special, resolve_beneath
from stigmergy.records import EventType, make_event

# Loop-stamped (never worker-authored) provenance string format for
# `filed_tickets.discovered_from` (SPEC amendment B).
DISCOVERED_FROM_FMT = "{dispatch_id}@{parent_ticket}"

# Fixed, worker-facing path (relative to the worktree/`/work`) the worker
# writes filed proposals to inside its container; the loop reads this from
# the HOST worktree, host-side, after the container has already been
# killed (SPEC amendment A) — never part of `work_product` (the bundle).
FILED_TICKETS_REL = ".stigmergy/filed-tickets.json"

# The exact proposal-shape keys hashed for provenance (SPEC amendment F).
_PROPOSAL_HASH_KEYS = ("title", "description", "evidence")

# Honest, declared-zero cost fields for every `ticket-filed` event — the
# harvest costs no tokens/time; the worker's cost was already booked on
# the DISPATCH event (SPEC amendment E).
_ZERO_TOKENS: dict[str, int] = {"in": 0, "cached": 0, "out": 0, "reasoning": 0}


@dataclass(frozen=True)
class FilingResult:
    """The outcome of one filing attempt (SPEC amendment, frozen interface).

    `duplicates_collapsed` (bead workspace-e2uh.188): how many exact-
    duplicate candidates the worker harvest collapsed to their first
    occurrence (0 for any harvest without duplicates and for direct
    `file_proposals` callers). Collapsed duplicates are NOT in `rejected` —
    they were filed (as the first occurrence), not rejected; they surface
    via the single aggregate `duplicate` ticket-filed event and this
    counter only.
    """

    accepted_ids: list[str]
    rejected: list[dict[str, Any]]  # each: {"reason": str, "title": str | None}
    escape: dict[str, Any] | None = None
    duplicates_collapsed: int = 0


def proposal_hash(proposal: dict[str, Any]) -> str:
    """sha256 hex over the canonical JSON of exactly
    `{"title", "description", "evidence"}` (`evidence` defaulting to
    `None` when absent, so a missing-evidence and explicit-`None`-evidence
    proposal hash identically). Canonical = `json.dumps(..., sort_keys=True,
    separators=(",", ":"))`.

    Provenance ONLY — this is NOT a steering `approval_hash` (`.42`
    computes that later, over the completed 9-field ticket).
    """
    shape = {key: proposal.get(key) for key in _PROPOSAL_HASH_KEYS}
    blob = json.dumps(shape, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _emit(
    *,
    record_plane: Any,
    ctx: dict[str, Any],
    attempt_kind: str,
    origin: dict[str, Any],
    proposal_hash_value: str,
    outcome: str,
    reason: str | None,
    filed_ticket_id: str | None,
) -> None:
    """Build one `ticket-filed` event via `make_event` (full §8 common
    fields from `ctx` + honest zeros) and append it to `record_plane`."""
    ev = make_event(
        EventType.TICKET_FILED,
        **ctx,
        attempt_kind=attempt_kind,
        tokens=dict(_ZERO_TOKENS),
        computed_usd=0.0,
        wall_time_seconds=0.0,
        origin=origin,
        proposal_hash=proposal_hash_value,
        outcome=outcome,
        reason=reason,
        filed_ticket_id=filed_ticket_id,
    )
    record_plane.append(ev)


def _is_valid_shape(proposal: Any) -> bool:
    """The single ingest-side shape authority for the SIX contract fields
    (audit .162 HIGH: the daemon routes worker-escape on
    `blocks_ticket` / `suspected_out_of_scope_paths`, so the ingest — not
    just the OA tool's call-time defense-in-depth — must validate their
    types before anything persists):

    - `title` / `description`: non-empty str (required);
    - `evidence`: str-or-absent;
    - `blocks_ticket`: bool-or-absent — the .102.2 escape route keys on
      `is True`, and a truthy STRING ("yes"/"true") is exactly the
      misroute garbage the audit named;
    - `suspected_out_of_scope_paths`: list-of-str-or-absent;
    - `reason`: str-or-absent.

    Per-item isolation is preserved by the caller: a violation rejects ONLY
    the offending item (`bad-shape`), never the whole batch, and an
    escape-shaped item (`blocks_ticket is True`) still surfaces via
    `harvest_worker_filings`' escape routing when its other fields are
    well-typed.

    Unknown EXTRA keys are deliberately NOT rejected: the store insert
    takes NAMED fields only (`RigStore.add_filed_ticket`), so extra keys
    never reach the DB, and :func:`proposal_hash` hashes only the three
    provenance keys (`title`/`description`/`evidence`), so they don't
    perturb provenance either — the raw dict's only whole-shape consumers
    are the size-cap byte count and the raw-hash of the per-item rejection
    event.
    """
    if not isinstance(proposal, dict):
        return False
    title = proposal.get("title")
    description = proposal.get("description")
    if not isinstance(title, str) or not title:
        return False
    if not isinstance(description, str) or not description:
        return False
    evidence = proposal.get("evidence")
    if evidence is not None and not isinstance(evidence, str):
        return False
    blocks_ticket = proposal.get("blocks_ticket")
    if blocks_ticket is not None and not isinstance(blocks_ticket, bool):
        return False
    suspected_paths = proposal.get("suspected_out_of_scope_paths")
    if suspected_paths is not None and not (
        isinstance(suspected_paths, list)
        and all(isinstance(p, str) for p in suspected_paths)
    ):
        return False
    reason = proposal.get("reason")
    if reason is not None and not isinstance(reason, str):
        return False
    return True


def file_proposals(
    proposals: list[Any],
    *,
    store: Any,
    record_plane: Any,
    ctx: dict[str, Any],
    attempt_kind: str,
    origin_role: str,
    max_filings: int,
    max_bytes: int,
    now: float | None = None,
) -> FilingResult:
    """Validate + file a list of raw proposal dicts (the reusable D14
    core; reused verbatim by `.39`/`.41` with a different `origin_role`).

    `now` is accepted for interface symmetry with `harvest_worker_filings`
    but unused here — `RigStore.add_filed_ticket` stamps its own
    `created_at=time.time()`.
    """
    del now  # accepted for interface symmetry; store stamps its own created_at.

    origin = {
        "role": origin_role,
        "worker": ctx["worker"],
        "dispatch_id": ctx["dispatch_id"],
        "parent_ticket": ctx["ticket"],
    }
    discovered_from = DISCOVERED_FROM_FMT.format(
        dispatch_id=ctx["dispatch_id"], parent_ticket=ctx["ticket"]
    )

    # Count-cap: reject the WHOLE filing (amendment D), one event, nothing
    # lands. A worker emitting more than the cap is suspicious -> drop all.
    if len(proposals) > max_filings:
        _emit(
            record_plane=record_plane,
            ctx=ctx,
            attempt_kind=attempt_kind,
            origin=origin,
            proposal_hash_value=_whole_filing_hash(proposals),
            outcome="rejected",
            reason="count-cap-exceeded",
            filed_ticket_id=None,
        )
        return FilingResult(
            accepted_ids=[], rejected=[{"reason": "count-cap-exceeded", "title": None}]
        )

    accepted_ids: list[str] = []
    rejected: list[dict[str, Any]] = []

    for n, proposal in enumerate(proposals, start=1):
        if not _is_valid_shape(proposal):
            title = proposal.get("title") if isinstance(proposal, dict) else None
            title = title if isinstance(title, str) else None
            _emit(
                record_plane=record_plane,
                ctx=ctx,
                attempt_kind=attempt_kind,
                origin=origin,
                proposal_hash_value=_whole_filing_hash(proposal),
                outcome="rejected",
                reason="bad-shape",
                filed_ticket_id=None,
            )
            rejected.append({"reason": "bad-shape", "title": title})
            continue

        proposal_bytes = json.dumps(proposal).encode("utf-8")
        if len(proposal_bytes) > max_bytes:
            _emit(
                record_plane=record_plane,
                ctx=ctx,
                attempt_kind=attempt_kind,
                origin=origin,
                proposal_hash_value=proposal_hash(proposal),
                outcome="rejected",
                reason="size-cap-exceeded",
                filed_ticket_id=None,
            )
            rejected.append({"reason": "size-cap-exceeded", "title": proposal["title"]})
            continue

        # bead .39: the id namespace includes `origin_role`. The worker's
        # dispatch and the staging critic's gate share ONE real dispatch id
        # (the daemon reconciles the ticket's lease_dispatch_id to the worker's
        # plan.dispatch_id, and the production weaver's default ctx reads that
        # same lease id) — so a plain `filed-{dispatch_id}-{n}` collides across
        # roles: a critic proposal at index n would hit the worker proposal's
        # primary key and be dropped as `store-error`, silently losing the
        # (per D14, highest-value) crit-role discovery. Keying on role makes
        # cross-role filings on one dispatch structurally distinct, while a
        # re-run of the SAME role on the SAME dispatch still collides -> the
        # intended idempotency guard below.
        pid = f"filed-{origin_role}-{ctx['dispatch_id']}-{n}"
        ph = proposal_hash(proposal)
        try:
            store.add_filed_ticket(
                id=pid,
                title=proposal["title"],
                description=proposal["description"],
                evidence=proposal.get("evidence"),
                origin_role=origin_role,
                origin_worker=ctx["worker"],
                origin_dispatch_id=ctx["dispatch_id"],
                origin_parent_ticket=ctx["ticket"],
                discovered_from=discovered_from,
                proposal_hash=ph,
            )
        except Exception:
            # Defense in depth: harvest_worker_filings' contract is "never
            # raises into the dispatch teardown path" (SPEC amendment A),
            # and file_proposals is reused verbatim by that path. A store-
            # level failure (e.g. a `filed-<role>-<dispatch_id>-<n>` primary-
            # key collision on a re-run harvest for the same role+dispatch) must
            # become a per-proposal rejection, never an exception that
            # escapes to the caller.
            _emit(
                record_plane=record_plane,
                ctx=ctx,
                attempt_kind=attempt_kind,
                origin=origin,
                proposal_hash_value=ph,
                outcome="rejected",
                reason="store-error",
                filed_ticket_id=None,
            )
            rejected.append({"reason": "store-error", "title": proposal["title"]})
            continue
        _emit(
            record_plane=record_plane,
            ctx=ctx,
            attempt_kind=attempt_kind,
            origin=origin,
            proposal_hash_value=ph,
            outcome="accepted",
            reason=None,
            filed_ticket_id=pid,
        )
        accepted_ids.append(pid)

    return FilingResult(accepted_ids=accepted_ids, rejected=rejected)


def _whole_filing_hash(raw: Any) -> str:
    """sha256 hex over a best-effort JSON/bytes serialization of ``raw`` —
    used for whole-filing rejection events (count-cap, bad-shape-as-a-
    whole-item fallback) where there is no single well-formed proposal to
    hash via :func:`proposal_hash`."""
    try:
        blob = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError):
        blob = repr(raw).encode("utf-8", errors="replace")
    return hashlib.sha256(blob).hexdigest()


def _reject_whole_harvest(
    *,
    record_plane: Any,
    ctx: dict[str, Any],
    attempt_kind: str,
    origin_role: str,
    reason: str,
    raw_bytes: bytes | None,
) -> FilingResult:
    """Emit ONE whole-filing rejection event (proposal_hash = sha256 of the
    raw file bytes, or of b"" if unreadable) and return the corresponding
    `FilingResult`. Never raises."""
    origin = {
        "role": origin_role,
        "worker": ctx["worker"],
        "dispatch_id": ctx["dispatch_id"],
        "parent_ticket": ctx["ticket"],
    }
    digest = hashlib.sha256(raw_bytes or b"").hexdigest()
    _emit(
        record_plane=record_plane,
        ctx=ctx,
        attempt_kind=attempt_kind,
        origin=origin,
        proposal_hash_value=digest,
        outcome="rejected",
        reason=reason,
        filed_ticket_id=None,
    )
    return FilingResult(accepted_ids=[], rejected=[{"reason": reason, "title": None}])


def harvest_worker_filings(
    worktree: str | Path,
    *,
    store: Any,
    record_plane: Any,
    ctx: dict[str, Any],
    attempt_kind: str,
    max_filings: int,
    max_bytes: int,
    now: float | None = None,
) -> FilingResult:
    """`.38` worker harvest path: safely read+parse
    `<worktree>/.stigmergy/filed-tickets.json` (host-side, post-kill),
    normalize it (bead workspace-e2uh.188 salvage pipeline), and delegate
    to :func:`file_proposals` with `origin_role="worker"`.

    The file is accepted in either shape:

    - **JSONL (the shape the `file_ticket` tool emits):** one proposal
      object per line, appended. Blank lines (including the trailing
      newline) carry no proposal. Each non-blank line is validated
      independently: a line that fails JSON parsing or is not an object
      is a PER-LINE `bad-shape` rejection (one `ticket-filed` rejection
      event + one `FilingResult.rejected` entry) while well-formed lines
      from the same file are still filed — a corrupt tail line never
      rejects the whole harvest, and a corrupt/non-object line does not
      count against the filing count-cap.
    - **Legacy whole-file JSON array:** a file whose entire content
      parses as one JSON list is still read as that proposal list
      (backward compatibility).

    The worker channel then NORMALIZES before delegating (bead
    workspace-e2uh.188 — the contexthandoff01 salvage contract; the
    in-cage OA file_ticket suite once polluted the production transport
    with a byte-identical fixture block repeated N times, and the old
    whole-batch count-cap semantics dropped the WHOLE file — including
    the worker's real diagnostics buried at the tail):

    1. **Exact-duplicate collapse** to the FIRST occurrence; the dup key
       is canonical JSON (key-order-only repeats collapse too).
       Collapsed duplicates count against nothing and are NOT in
       `FilingResult.rejected` (they were filed, not rejected); they
       surface as ONE aggregate `ticket-filed` event
       (`outcome="collapsed"`, `reason="duplicate"`, proposal_hash over
       the list of collapsed objects; no event when nothing collapsed) +
       the `duplicates_collapsed` counter.
    2. **Per-entry shape/size rejects** use exactly `file_proposals`'
       own conventions (`_is_valid_shape` -> `bad-shape`; raw-dump byte
       length vs `max_bytes` -> `size-cap-exceeded`), one rejection
       event + one `rejected` entry per offender, and do NOT consume
       cap slots.
    3. **Truncate to cap, earlier filings stand:** the count cap counts
       UNIQUE INSERTABLE candidates (post-dedup, post-shape, post-size);
       the first `max_filings` are delegated, the excess gets ONE
       aggregate `count-cap-exceeded` event + one `rejected` entry per
       excess proposal. This is the worker channel's semantics — NOT
       `file_proposals`' amendment-D whole-batch rejection, which stays
       fully in force for its direct `.39`/`.41` callers.
    4. **Escape routing unchanged:** the first deduped dict with
       `blocks_ticket is True` surfaces on `FilingResult.escape`
       (built as before); the escape object is STILL filed when
       shape-valid (insert-then-tombstone is the house pattern —
       execdogfood01 precedent), and entries after it still process.
    5. The legacy whole-file JSON-array shape follows the SAME rules
       (divergent handling between the two shapes is a footgun).

    MUST NEVER RAISE — this runs in the dispatch teardown path (SPEC
    amendment A). Every failure mode is captured in the returned
    `FilingResult` (and, except for the missing-file case, in one
    `ticket-filed` rejection event per failure).
    """
    worktree_path = Path(worktree)

    try:
        resolved = resolve_beneath(worktree_path, FILED_TICKETS_REL)
    except PathSafetyError:
        return _reject_whole_harvest(
            record_plane=record_plane,
            ctx=ctx,
            attempt_kind=attempt_kind,
            origin_role="worker",
            reason="path-unsafe",
            raw_bytes=None,
        )
    except OSError:
        return _reject_whole_harvest(
            record_plane=record_plane,
            ctx=ctx,
            attempt_kind=attempt_kind,
            origin_role="worker",
            reason="unreadable",
            raw_bytes=None,
        )

    if not resolved.exists():
        # Silent no-op: no filing file means the worker filed nothing this
        # dispatch — not an error, no event at all. Checked BEFORE
        # reject_special (which os.stat()s and would itself raise on a
        # genuinely absent path).
        return FilingResult(accepted_ids=[], rejected=[])

    try:
        reject_special(resolved)
    except PathSafetyError:
        return _reject_whole_harvest(
            record_plane=record_plane,
            ctx=ctx,
            attempt_kind=attempt_kind,
            origin_role="worker",
            reason="path-unsafe",
            raw_bytes=None,
        )

    try:
        raw_bytes = resolved.read_bytes()
    except OSError:
        return _reject_whole_harvest(
            record_plane=record_plane,
            ctx=ctx,
            attempt_kind=attempt_kind,
            origin_role="worker",
            reason="unreadable",
            raw_bytes=None,
        )

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return _reject_whole_harvest(
            record_plane=record_plane,
            ctx=ctx,
            attempt_kind=attempt_kind,
            origin_role="worker",
            reason="malformed-json",
            raw_bytes=raw_bytes,
        )

    origin = {
        "role": "worker",
        "worker": ctx["worker"],
        "dispatch_id": ctx["dispatch_id"],
        "parent_ticket": ctx["ticket"],
    }

    # The file is either one whole JSON value (the legacy array shape, or a
    # single-line JSONL file whose content is one object/scalar) or not —
    # in which case it is read line-by-line as the JSONL stream.
    try:
        whole = json.loads(text)
    except json.JSONDecodeError:
        whole = None

    line_rejected: list[dict[str, Any]] = []
    if isinstance(whole, list):
        # Legacy backward-compat shape: one whole-file JSON array of
        # proposals — the SAME salvage pipeline as the JSONL stream
        # (contract rule 5: divergent handling between the two shapes
        # is a footgun).
        candidates: list[Any] = whole
    else:
        # JSONL: one proposal object per line. When the whole-file parse
        # succeeded (a dict or scalar), the entire content is that ONE
        # line; otherwise the file is not a single JSON value and is split
        # into lines.
        lines = [text] if whole is not None else text.splitlines()
        candidates = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                # Blank lines (e.g. the trailing newline of an appended
                # stream) carry no proposal — never a rejection.
                continue
            try:
                obj: Any = json.loads(stripped)
            except json.JSONDecodeError:
                obj = stripped  # unparseable line — per-line rejection below
            if isinstance(obj, dict):
                candidates.append(obj)
                continue
            # A line that fails JSON parsing or is not an object is a
            # PER-LINE bad-shape rejection: well-formed lines from the same
            # file are still filed (no whole-harvest rejection for a
            # corrupt tail line), and the line does not count against the
            # filing count-cap (which file_proposals enforces on
            # `candidates` only).
            _emit(
                record_plane=record_plane,
                ctx=ctx,
                attempt_kind=attempt_kind,
                origin=origin,
                proposal_hash_value=_whole_filing_hash(obj),
                outcome="rejected",
                reason="bad-shape",
                filed_ticket_id=None,
            )
            line_rejected.append({"reason": "bad-shape", "title": None})

    # --- bead workspace-e2uh.188 salvage pipeline (worker path only) -----
    # The worker channel's harvest semantics are "earlier filings stand"
    # (mirroring the OA tool's call-time cap), NOT amendment-D whole-batch
    # rejection: an over-cap worker TRANSPORT can only arise from harness
    # pollution/duplication (the tool's call-time cap already bounds what
    # one worker run can emit), and dropping the whole file there
    # punished the harness's own bug by losing the worker's real
    # diagnostics (contexthandoff01 T0). file_proposals itself stays
    # untouched for its direct .39/.41 callers.

    # 1. Exact-duplicate collapse to the FIRST occurrence. The dup key is
    #    canonical JSON (sort_keys + compact separators), so a
    #    key-order-only-differing repeat of the same object collapses too.
    #    Collapsed duplicates count against NOTHING (the cap counts unique
    #    insertable candidates) and are NOT in `rejected` — they were
    #    filed, as the first occurrence.
    seen_keys: set[str] = set()
    deduped: list[Any] = []
    collapsed: list[Any] = []
    for obj in candidates:
        try:
            key = json.dumps(obj, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError):
            # Unserializable content: degrade, never raise (amendment A).
            key = repr(obj)
        if key in seen_keys:
            collapsed.append(obj)
            continue
        seen_keys.add(key)
        deduped.append(obj)

    # ONE aggregate event for the collapse (operator legibility: an
    # N-fold repeated entry is a transport-pollution signal, not noise
    # to hide). No event at all when nothing collapsed.
    if collapsed:
        _emit(
            record_plane=record_plane,
            ctx=ctx,
            attempt_kind=attempt_kind,
            origin=origin,
            proposal_hash_value=_whole_filing_hash(collapsed),
            outcome="collapsed",
            reason="duplicate",
            filed_ticket_id=None,
        )

    # 2. Escape scan: the first DEDUPED dict with blocks_ticket is True
    #    surfaces as escape (same construction as before: reason coerced
    #    to str-or-None, paths coerced to list-or-[]). Scanning the
    #    deduped pre-filter list means a shape-invalid escape item still
    #    surfaces even though its filing is rejected per-entry below.
    #    This is purely additive: a shape-valid escape object is ALSO
    #    still filed as a normal proposal (its title/description persist
    #    to filed_tickets for triage — insert-then-tombstone is the house
    #    pattern, execdogfood01 precedent), and entries after the escape
    #    still process.
    escape_dict: dict[str, Any] | None = None
    try:
        for obj in deduped:
            if isinstance(obj, dict) and obj.get("blocks_ticket") is True:
                # Found the first escape object. Build the escape dict.
                reason = obj.get("reason")
                reason = reason if isinstance(reason, str) else None
                suspected_paths = obj.get("suspected_out_of_scope_paths")
                suspected_paths = suspected_paths if isinstance(suspected_paths, list) else []
                escape_dict = {
                    "reason": reason,
                    "suspected_out_of_scope_paths": suspected_paths,
                }
                break
    except Exception:
        # Malformed escape detection must never break the harvest. Default
        # to escape=None on any unexpected error.
        escape_dict = None

    # 3. Cap-truncate with earlier-filings-stand: the FIRST max_filings
    #    INSERTABLE candidates (dict + _is_valid_shape + under max_bytes)
    #    are delegated; the insertable excess is ONE aggregate
    #    count-cap-exceeded event + one rejected entry per excess
    #    proposal. NON-insertable candidates (bad shape / oversize /
    #    non-dict legacy-array entries) stay in the delegation list in
    #    place: file_proposals rejects them per-entry with its own
    #    conventions (exactly ONE event per rejection — the harvest does
    #    not pre-emit, so nothing double-fires), they do NOT consume cap
    #    slots, and keeping them positional keeps the pid ordinals stable
    #    across reharvests of the same file (a rejected entry still
    #    occupies its ordinal — the .162 idempotency pins rely on that).
    delegation: list[Any] = []
    excess: list[dict[str, Any]] = []
    insertable_kept = 0
    for obj in deduped:
        insertable = (
            isinstance(obj, dict)
            and _is_valid_shape(obj)
            and len(json.dumps(obj).encode("utf-8")) <= max_bytes
        )
        if insertable:
            if insertable_kept < max_filings:
                insertable_kept += 1
                delegation.append(obj)
            else:
                excess.append(obj)
        else:
            delegation.append(obj)

    excess_rejected: list[dict[str, Any]] = []
    if excess:
        _emit(
            record_plane=record_plane,
            ctx=ctx,
            attempt_kind=attempt_kind,
            origin=origin,
            proposal_hash_value=_whole_filing_hash(excess),
            outcome="rejected",
            reason="count-cap-exceeded",
            filed_ticket_id=None,
        )
        for obj in excess:
            title = obj.get("title")
            title = title if isinstance(title, str) else None
            excess_rejected.append({"reason": "count-cap-exceeded", "title": title})

    # 4. Delegate to file_proposals UNCHANGED. `max_filings` is passed as
    #    len(delegation) so its amendment-D whole-batch branch (still
    #    fully in force for the direct .39/.41 callers) is dead code on
    #    this path — the salvage truncation above already bounded the
    #    insertable count — while its per-entry shape/size rejections are
    #    the single source of the per-entry rejection events for the
    #    non-insertable entries kept in the list.
    result = file_proposals(
        delegation,
        store=store,
        record_plane=record_plane,
        ctx=ctx,
        attempt_kind=attempt_kind,
        origin_role="worker",
        max_filings=len(delegation),
        max_bytes=max_bytes,
        now=now,
    )
    result = replace(
        result,
        rejected=line_rejected + result.rejected + excess_rejected,
        duplicates_collapsed=len(collapsed),
    )
    if escape_dict is not None:
        return replace(result, escape=escape_dict)
    return result
