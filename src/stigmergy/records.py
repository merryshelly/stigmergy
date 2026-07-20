"""Event-sourced record plane (SPEC.md §8, plus §4 redaction/credentials and
§9 `attempt_kind`).

Rev 1's per-dispatch CV row tried to hold facts that don't exist at record
time (a parked dispatch may be rejected, reconciled, and landed much later).
This module records **immutable events** and derives projections from them:

- **Event log** (`records/events.jsonl`) — append-only, framed + checksummed
  lines, fsync'd on every append, read back through a *tolerant* reader that
  skips a torn/corrupt tail (crash mid-append) rather than raising.
- **CV projection** (`records/cv.jsonl`) — the per-dispatch analytical view,
  *rebuilt from events* on demand, written via temp-file + atomic rename.
  Never hand-edited; always reproducible from the event log.
- **Transcript blob store** (`records/transcripts/<sha256>`) — content-
  addressed, **redacted at seal** (§4): the redactor runs before anything
  touches disk, and sealing refuses outright if a caller-supplied
  "must-not-appear" sentinel survives redaction.

**`computed_usd` invariant (the whole point, §4/§8/§12):** every event's
`computed_usd` is either a non-negative float or the exact string
`"unbudgetable"` — never defaulted. There is no code path in this module
where an absent cost silently becomes `$0`; `0.0` is legal only when a
caller passes it explicitly (mirrors `registry.py`'s pricing philosophy).

**On the event-type list.** SPEC §8's prose paragraph enumerates six event
types (`dispatch, check, gate, integration, disposition, notify`), but SPEC
§4's prompt-artifact invariant and §8's field list both separately name
`report` as the LLM-invocation event behind `stigmergy range-report
--critic` (the `rangecrit01` prompt), and §12 requires "amortized
range/report spend" to be counted from events. The six-type prose list is a
paraphrase, not the full schema; :class:`EventType` includes `REPORT` as a
seventh member so range-report spend has an event to land in. D14 (bead
workspace-e2uh.38) adds an eighth member, `TICKET_FILED`, for the
host-side worker/critic/range-critic ticket-filing harvest (see
`filing.py`) — it is NOT an LLM-invocation event (the harvest is
mechanism-only; the worker's cognition was already recorded on the
DISPATCH event) so it carries no `prompt_artifact_hash`.

Bead `.42` (D1/D15) adds three more — `APPROVAL`, `UNAPPROVAL`,
`TRIAGE_REJECTED` — the human triage attribution events (acting agent +
operator session, agent-asserted in v0). They are the ONE-log audit line
D1 locates in the event plane, and are **exempt from the dispatch-shaped
common-field set** (human acts, not dispatches): they validate against
`_REQUIRED_TRIAGE_FIELDS` only. Eleven members total.
"""

from __future__ import annotations

import contextlib
import enum
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_STRICT_MODE = 0o600
_STRICT_DIR_MODE = 0o700


class RecordError(Exception):
    """Raised on any record-plane violation: bad schema, bad attempt_kind,
    a missing/invalid `computed_usd`, or a seal that would leak a secret."""


class EventType(enum.Enum):
    """Event-plane discriminant (SPEC.md §8).

    Eleven members. `dispatch`, `check`, `gate`, `integration`,
    `disposition`, `notify` are SPEC §8's prose list; `report` is added
    per SPEC §4 (prompt-artifact invariant: "every LLM-invoked role... the
    prompt hash is logged in every event that invocation produces") and
    §12 (range/report spend must be counted from events) — see module
    docstring for the full justification. `ticket-filed` (D14, bead
    workspace-e2uh.38) is added for the host-side ticket-filing harvest
    (see `filing.py`); it is mechanism-only, not an LLM invocation.
    `approval`/`unapproval`/`triage-rejected` (bead .42, D1/D15) are the
    human triage attribution events — exempt from the dispatch-shaped
    common-field set (see `_REQUIRED_TRIAGE_FIELDS` and `_validate_payload`).
    """

    DISPATCH = "dispatch"
    CHECK = "check"
    GATE = "gate"
    INTEGRATION = "integration"
    DISPOSITION = "disposition"
    NOTIFY = "notify"
    REPORT = "report"
    TICKET_FILED = "ticket-filed"
    # Triage attribution events (bead .42, D1/D15): human triage acts executed
    # by an agent at the operator's direction. They are NOT dispatch-adjacent —
    # no worker, model, dispatch, tokens, or cost — so they are EXEMPT from the
    # dispatch-shaped common-field set and validate against their own required
    # set (see `_REQUIRED_TRIAGE_FIELDS`). This deliberately loosens "every
    # event carries the full common-field set" (§8 prose, written when every
    # event was dispatch-adjacent), the same way `.38` documented TICKET_FILED's
    # own exemptions.
    APPROVAL = "approval"
    UNAPPROVAL = "unapproval"
    TRIAGE_REJECTED = "triage-rejected"


# LLM-invocation events (SPEC §4 prompt-artifact invariant): must carry the
# versioned prompt hash that produced them (code01/critic01/rangecrit01).
_LLM_INVOCATION_TYPES = frozenset({EventType.DISPATCH, EventType.GATE, EventType.REPORT})

# SPEC.md §9 "Retry semantics" attempt_kind enumeration — the exhaustive set.
# `report` (bead .42): a `range-report --critic` REPORT event is rig-level, not
# a ticket attempt; `initial` would be a recorded lie in an audit log, so a
# dedicated honest value is used. SPEC §9's enumeration gains this member — a
# tracked follow-up note for SB's next review (see bead42-build-spec.md).
# `critic-infra` (bead .107): a per-ticket critic-infra escalation
# (PARKED->ESCALATED, `decide_critic_infra`) is a distinct, queryable
# outcome from a dispatch-side `infra-retry` — reusing `infra-retry` here
# would be a recorded lie in an audit log (same precedent as `report`
# above). SPEC §9's enumeration gains this member too — another tracked
# follow-up note for SB's next review.
ATTEMPT_KINDS: frozenset[str] = frozenset(
    {
        "initial",
        "tier1-repair",
        "critic-revision",
        "integration-reconcile",
        "infra-retry",
        "stepup-initial",
        "clean-restart",
        "report",
        "critic-infra",
    }
)

# Common fields required on EVERY event (SPEC §8 field list), minus
# `event_type` and `ts` which get their own dedicated handling below.
_REQUIRED_COMMON_FIELDS: tuple[str, ...] = (
    "rig",
    "ticket",
    "dispatch_id",
    "attempt",
    "attempt_kind",
    "rung",
    "worker",
    "charter_hash",
    "approval_hash",
    "image_digest",
    "model",
    "model_version",
    "price_table_version",
    "tokens",
    "computed_usd",
    "wall_time_seconds",
)

_TOKEN_KEYS: tuple[str, ...] = ("in", "cached", "out", "reasoning")

# Triage attribution events (bead .42): the human-triage acts are EXEMPT from
# `_REQUIRED_COMMON_FIELDS` (no dispatch/worker/model/tokens/cost) and validate
# against this set instead. Each must be a non-empty str — a forgeable-by-
# omission audit line (D1's agent-asserted attribution) is not acceptable.
_TRIAGE_EVENT_TYPES = frozenset(
    {EventType.APPROVAL, EventType.UNAPPROVAL, EventType.TRIAGE_REJECTED}
)
_REQUIRED_TRIAGE_FIELDS: tuple[str, ...] = (
    "rig",
    "subject_id",
    "outcome",
    "acting_agent",
    "operator_session",
)


@dataclass(frozen=True)
class Event:
    """A validated, immutable event ready for `RecordPlane.append`.

    Deliberately a thin wrapper around one flat ``payload`` dict rather than
    ~20 individual dataclass fields plus a type-specific extras dict — SPEC
    §8's field list is common to every event, and the two per-type extras
    (`prompt_artifact_hash`, `decoding_params`) merge naturally into the
    same flat namespace with no collision risk against the common fields.
    """

    payload: dict[str, Any]

    @property
    def event_type(self) -> EventType:
        return EventType(self.payload["event_type"])

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


def make_event(event_type: EventType | str, **fields: Any) -> Event:
    """Build and validate one :class:`Event`.

    ``ts`` defaults to ``time.time()`` if omitted — it is bookkeeping, not a
    budget fact, so this is a genuine convenience default, not the forbidden
    kind. Every other required field (including `computed_usd`) has no
    default: omit one and this raises :class:`RecordError`.
    """
    try:
        resolved_type = event_type if isinstance(event_type, EventType) else EventType(event_type)
    except ValueError:
        raise RecordError(f"unknown event_type {event_type!r}") from None

    payload: dict[str, Any] = dict(fields)
    payload["event_type"] = resolved_type.value
    payload.setdefault("ts", time.time())

    _validate_payload(payload)
    return Event(payload=payload)


def _validate_payload(payload: dict[str, Any]) -> None:
    """Validate one event payload dict. Raises :class:`RecordError` on any
    violation. Called both at `make_event` build time and again at
    `RecordPlane.append` time (defense in depth: a payload that reaches
    `append` by some path other than `make_event` still gets checked)."""
    if "event_type" not in payload:
        raise RecordError("event missing required field 'event_type'")
    try:
        event_type = EventType(payload["event_type"])
    except ValueError:
        raise RecordError(f"unknown event_type {payload['event_type']!r}") from None

    if "ts" not in payload or not isinstance(payload["ts"], int | float):
        raise RecordError("event missing required numeric field 'ts'")

    # Triage attribution events (bead .42) validate against their own required
    # set and are EXEMPT from the dispatch-shaped common fields / LLM / gate
    # branches below (they are human acts, not dispatches — no worker, model,
    # tokens, or cost). Documented deviation from §8's "every event carries the
    # full common-field set" prose (see the EventType docstring).
    if event_type in _TRIAGE_EVENT_TYPES:
        _validate_triage_payload(payload)
        return

    missing = [f for f in _REQUIRED_COMMON_FIELDS if f not in payload]
    if missing:
        raise RecordError(f"event missing required common field(s): {missing}")

    attempt = payload["attempt"]
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
        raise RecordError(f"event field 'attempt' must be a non-negative int (got {attempt!r})")

    attempt_kind = payload["attempt_kind"]
    if attempt_kind not in ATTEMPT_KINDS:
        raise RecordError(
            f"invalid attempt_kind {attempt_kind!r}; must be one of {sorted(ATTEMPT_KINDS)}"
        )

    wall_time = payload["wall_time_seconds"]
    if isinstance(wall_time, bool) or not isinstance(wall_time, int | float) or wall_time < 0:
        raise RecordError(
            f"event field 'wall_time_seconds' must be a non-negative number (got {wall_time!r})"
        )

    _validate_tokens(payload["tokens"])
    _validate_computed_usd(payload["computed_usd"])

    if event_type in _LLM_INVOCATION_TYPES:
        prompt_hash = payload.get("prompt_artifact_hash")
        if not isinstance(prompt_hash, str) or not prompt_hash:
            raise RecordError(
                f"{event_type.value!r} event missing required 'prompt_artifact_hash' "
                "(SPEC §4 prompt-artifact invariant: code01/critic01/rangecrit01 "
                "hash must be logged on every LLM-invocation event)"
            )

    if event_type is EventType.GATE:
        # This IS the critic-prompt-template hash (critic01) — provenance of
        # which template judged the gate — already enforced above as
        # prompt_artifact_hash; decoding_params is the second gate-only
        # requirement (pinned + logged decoding params, SPEC §4/§8).
        decoding_params = payload.get("decoding_params")
        if not isinstance(decoding_params, dict):
            raise RecordError(
                "gate event missing required 'decoding_params' (must be a dict). An "
                "EMPTY dict is VALID (bead .81/.95): the deprecating-generation models "
                "(opus-4-8/sonnet-5) reject all sampling params, so {} is the correct "
                "logged provenance — 'no sampling params sent; model defaults + forced "
                "tool_choice'. Only a missing field or non-dict is rejected."
            )


def _validate_triage_payload(payload: dict[str, Any]) -> None:
    """Validate a triage attribution event (APPROVAL/UNAPPROVAL/TRIAGE_REJECTED,
    bead .42). Only the five required attribution strings are enforced — no
    dispatch-shaped common fields. `approval_hash`/`reason` are optional extras
    (no validation beyond whatever the caller supplied)."""
    for field in _REQUIRED_TRIAGE_FIELDS:
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise RecordError(
                f"triage event field {field!r} must be a non-empty string "
                f"(got {value!r}) — the v0 audit line (D1) must not be forgeable "
                "by omission"
            )


def _validate_tokens(tokens: Any) -> None:
    if not isinstance(tokens, dict):
        raise RecordError(f"event field 'tokens' must be a dict (got {tokens!r})")
    missing = [k for k in _TOKEN_KEYS if k not in tokens]
    if missing:
        raise RecordError(f"event field 'tokens' missing key(s): {missing}")
    for key in _TOKEN_KEYS:
        value = tokens[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RecordError(
                f"event field 'tokens.{key}' must be a non-negative int (got {value!r})"
            )


def _validate_computed_usd(value: Any) -> None:
    """Enforce the `computed_usd` invariant: non-negative float, or the
    exact string `"unbudgetable"` — never anything else, never a default."""
    if isinstance(value, str):
        if value == "unbudgetable":
            return
        raise RecordError(
            f"event field 'computed_usd' string value must be exactly 'unbudgetable' "
            f"(got {value!r})"
        )
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RecordError(
            "event field 'computed_usd' must be a non-negative float or the string "
            f"'unbudgetable' (got {value!r})"
        )
    if not math.isfinite(value) or value < 0:
        raise RecordError(
            f"event field 'computed_usd' must be a non-negative finite number (got {value!r})"
        )


def _canonical_hash(data: dict[str, Any]) -> str:
    """sha256 over a canonical (sorted-key, whitespace-free JSON) serialization.

    Mirrors `registry.py`/`charter.py`'s `_canonical_hash` — stable across
    key reordering, sensitive to any value change. Used both as the
    per-line checksum basis and for content-addressing transcript blobs.
    """
    blob = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` via temp-file (same dir) + `os.replace`,
    fsync'd before the rename, with strict `0o600` mode guaranteed
    regardless of umask."""
    directory = path.parent
    fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp_name, _STRICT_MODE)
        os.replace(tmp_name, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


class RecordPlane:
    """The event-sourced record plane for one rig's `records/` directory."""

    def __init__(self, records_dir: str | os.PathLike[str]) -> None:
        self.records_dir = Path(records_dir)
        self.records_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.records_dir, _STRICT_DIR_MODE)
        self.transcripts_dir = self.records_dir / "transcripts"
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.transcripts_dir, _STRICT_DIR_MODE)
        self.events_path = self.records_dir / "events.jsonl"
        self.cv_path = self.records_dir / "cv.jsonl"

    # -- event log -----------------------------------------------------

    def append(self, event: Event) -> None:
        """Validate, frame, checksum, and durably append one event.

        Writes one JSON line (event payload + a `checksum` field, sha256
        over the canonical payload sans `checksum`) to `events.jsonl`,
        then `flush()` + `fsync()`s the file — a crash right after this
        call cannot lose the write. `events.jsonl` is created (if absent)
        with strict mode `0o600`, re-asserted after every append so umask
        can never widen it.
        """
        if not isinstance(event, Event):
            raise RecordError("append() requires an Event built via make_event()")
        # Defense in depth: re-validate at the boundary that actually
        # touches disk, independent of how the caller obtained the Event.
        _validate_payload(event.payload)

        record = dict(event.payload)
        record["checksum"] = _canonical_hash(event.payload)
        line = json.dumps(record, sort_keys=True, separators=(",", ":"))

        fd = os.open(self.events_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, _STRICT_MODE)
        try:
            with os.fdopen(fd, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        finally:
            os.chmod(self.events_path, _STRICT_MODE)

    def read_events(self) -> list[dict[str, Any]]:
        """Tolerant read of `events.jsonl`: returns validated payload dicts
        (checksum field stripped) in file order.

        A line that fails to parse as JSON, isn't an object, is missing its
        `checksum` field, or whose checksum doesn't match the recomputed
        canonical hash of the rest of the line is silently skipped — this
        is exactly the torn-tail-from-a-crash-mid-append case, and it must
        never raise or abort the read of everything that IS intact.
        """
        events: list[dict[str, Any]] = []
        if not self.events_path.exists():
            return events

        with open(self.events_path, encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict) or "checksum" not in record:
                    continue
                checksum = record["checksum"]
                payload = {k: v for k, v in record.items() if k != "checksum"}
                try:
                    expected = _canonical_hash(payload)
                except (TypeError, ValueError):
                    continue
                if checksum != expected:
                    continue
                events.append(payload)
        return events

    # -- CV projection ---------------------------------------------------

    def rebuild_cv(self) -> None:
        """Rebuild `cv.jsonl` from `events.jsonl` — one row per dispatch.

        Purely a function of the (valid, checksum-passing) events currently
        on disk: same events in, byte-identical `cv.jsonl` out, every time
        (idempotent). Rows are ordered by `dispatch_id` (sorted) rather than
        first-seen order so this holds even if some future writer appends
        events for the same dispatch out of original order. Written via
        temp-file + `os.replace` (atomic), strict mode `0o600`.
        """
        events = self.read_events()

        by_dispatch: dict[str, list[dict[str, Any]]] = {}
        for ev in events:
            dispatch_id = ev.get("dispatch_id")
            if not isinstance(dispatch_id, str):
                continue
            by_dispatch.setdefault(dispatch_id, []).append(ev)

        rows = [
            _build_cv_row(dispatch_id, by_dispatch[dispatch_id])
            for dispatch_id in sorted(by_dispatch)
        ]

        lines = [json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows]
        content = ("\n".join(lines) + "\n") if lines else ""
        _atomic_write_bytes(self.cv_path, content.encode("utf-8"))

    # -- transcript blob store --------------------------------------------

    def seal_transcript(
        self,
        content: str,
        *,
        redactor: Any,
        must_not_contain: frozenset[str] = frozenset(),
    ) -> str:
        """Redact, then content-address, then durably store a transcript.

        The redactor runs FIRST. If, after redaction, the result still
        contains any sentinel from ``must_not_contain``, this raises
        :class:`RecordError` and writes NOTHING — no blob file of any kind
        is created (checked before any filesystem write, including a temp
        file). The error message never echoes which sentinel matched or
        any content, so the secret is never re-logged via the exception.

        Returns the blob ref: the sha256 hex digest of the redacted
        content, which is also its filename under `records/transcripts/`.
        A repeat seal of already-stored (identical) content is a no-op,
        not an error — content addressing makes it naturally idempotent.
        """
        redacted = redactor(content)

        for sentinel in must_not_contain:
            if sentinel in redacted:
                raise RecordError(
                    "seal_transcript refused: redacted content still contains a "
                    "must-not-appear sentinel — no blob written"
                )

        digest = hashlib.sha256(redacted.encode("utf-8")).hexdigest()
        blob_path = self.transcripts_dir / digest
        if blob_path.exists():
            return digest
        _atomic_write_bytes(blob_path, redacted.encode("utf-8"))
        return digest


def _build_cv_row(dispatch_id: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one dispatch's events into a single CV row.

    `final_outcome` is deliberately *not* a SPEC §8 common field — the
    state-machine/verdict schema (SPEC §9) that would define a canonical
    "outcome" enum is a later ticket. Rather than invent one here and risk
    colliding with it, this takes the `event_type` of the last event seen
    for the dispatch (in on-disk order) as a provisional summary signal —
    a CV-projection convenience, not part of the sealed event contract.
    """
    tokens_total = {key: 0 for key in _TOKEN_KEYS}
    computed_usd_total = 0.0
    has_unbudgetable = False

    for ev in events:
        tokens = ev.get("tokens")
        if isinstance(tokens, dict):
            for key in _TOKEN_KEYS:
                value = tokens.get(key, 0)
                if isinstance(value, int) and not isinstance(value, bool):
                    tokens_total[key] += value

        usd = ev.get("computed_usd")
        if isinstance(usd, str):
            if usd == "unbudgetable":
                has_unbudgetable = True
        elif isinstance(usd, int | float) and not isinstance(usd, bool):
            computed_usd_total += float(usd)

    first = events[0]
    last = events[-1]
    return {
        "dispatch_id": dispatch_id,
        "ticket": first.get("ticket"),
        "rung": first.get("rung"),
        "model": first.get("model"),
        "event_count": len(events),
        "tokens": tokens_total,
        "computed_usd_total": computed_usd_total,
        "has_unbudgetable": has_unbudgetable,
        "final_outcome": last.get("event_type"),
    }
