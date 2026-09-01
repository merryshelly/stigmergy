"""Quota governor — signal ingestion from the relay JSONL feed (bead .147 §1E
follow-up; the ingestion half of the governor).

The relay logs one JSON line per request with the bounded captures
``synthetic_quotas`` (the verbatim, capped ``x-synthetic-quotas`` response
header) and ``upstream_429_body`` (the verbatim, capped upstream 429 body).
This module folds those lines into a **current per-provider**
:class:`QuotaState` map:

- the reader is **offset-based**: it seeks to the previously returned byte
  offset and reads only what has been appended since, so the historical
  log is never re-read and never buffered;
- memory is **bounded by construction**: the module retains exactly one
  ``QuotaState`` per provider key that has ever reported (a fixed-size
  dict-of-dataclasses) and nothing else — no list of historical records,
  no growing per-line cache;
- **tolerance is total**: an invalid-JSON line, a truncated/partial record,
  or a record with missing/mistyped quota fields is skipped individually,
  and no exception path in :func:`read_feed` can reach its caller — a
  quota-signal parse gap must never crash the loop.

Wire shape (spike bundle 138): ``synthetic_quotas`` is a JSON object of
the form::

    {
      "rollingFiveHourLimit": {
        "remaining": <number>, "max": <number>,
        "limited": <bool>, "nextTickAt": <ISO-8601 string>
      },
      "weeklyTokenLimit": {
        "creditUsed": "$12.34", "creditRemaining": "$87.66", "creditLimit": "$100.00"
      }
    }

``upstream_429_body`` is the upstream 429's body, verbatim and capped: the
same JSON object when the upstream speaks the synthetic shape, otherwise an
unstructured error body — recorded as evidence only.

The decision half of the governor (:func:`decide`) is pure and
deterministic over the pricing class + a provider's :class:`QuotaState`
— no daemon plumbing, unit-testable without any daemon fixture.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stigmergy.registry import ModelEntry, PricingClass

# Dollar-string credit fields on the `weeklyTokenLimit` object (spike bundle
# 138): "creditUsed"/"creditRemaining"/"creditLimit" -> "12.34"-shaped
# dollar strings. Only these three are interpreted; anything else on the
# object is ignored, so the wire can grow fields without breaking the fold.
_CREDIT_FIELDS = ("creditUsed", "creditRemaining", "creditLimit")

# Provider-key fields accepted from a feed line, in precedence order. The
# .147 relay lines carry `wire` (openai/anthropic) and `dispatch_id`;
# `provider`/`provider_key` are accepted for feeds that key explicitly.
_KEY_FIELDS = ("provider", "provider_key", "dispatch_id", "wire")


@dataclass(frozen=True)
class RollingWindow:
    """The `rollingFiveHourLimit` object: tokens remaining in the rolling
    window, the window's max, whether the provider currently reports the
    window as limited, and the timestamp at which the next reset tick
    occurs (verbatim ISO-8601 string; ``None`` if absent/untyped)."""

    remaining: float
    max: float
    limited: bool
    next_tick_at: str | None


@dataclass(frozen=True)
class WeeklyCredit:
    """The `weeklyTokenLimit` object: dollar-string credit fields parsed to
    floats. A field that was absent or not parseable is ``None`` (e.g. the
    window reported only some of the three)."""

    credit_used: float | None
    credit_remaining: float | None
    credit_limit: float | None


@dataclass(frozen=True)
class QuotaState:
    """Current quota state for ONE provider key.

    Fields are the latest values seen for that provider; a field a record
    omitted or mistyped keeps the previous value (absent data never
    clobbers present data). ``last_429_at`` records the feed timestamp of
    the most recent ``upstream_429_body`` seen (``None`` until one
    arrives); ``updated_at`` is the feed timestamp of the most recent
    folded record for this provider."""

    provider: str
    rolling: RollingWindow | None = None
    weekly: WeeklyCredit | None = None
    limited: bool = False
    next_tick_at: str | None = None
    last_429_at: float | None = None
    updated_at: float | None = None
    # The reason of the most recent relay capability-deny capture seen for
    # this provider (``None`` until one arrives). These are relay POLICY
    # signals — a deny record NEVER sets ``limited`` or ``last_429_at``, and
    # the decision logic never turns one into park evidence.
    last_deny_reason: str | None = None


@dataclass
class QuotaGovernor:
    """The governor's current state: one :class:`QuotaState` per provider
    key that has ever reported. This dict — and nothing else — is what the
    module retains; it is bounded by the number of providers, not by the
    number of feed lines."""

    providers: dict[str, QuotaState] = field(default_factory=dict)

    def fold(self, record: dict[str, Any]) -> bool:
        """Fold one parsed feed-line record into the governor.

        Returns ``True`` if the record changed state. Records without a
        usable provider key, or whose quota fields are all absent/mistyped,
        leave the state untouched. Never raises."""
        try:
            key = _provider_key(record)
            if key is None:
                return False
            rolling, weekly = _parse_quotas(_as_str(record.get("synthetic_quotas")))
            body = _as_str(record.get("upstream_429_body"))
            # The 429 body is parsed as a quota source too: when the upstream
            # speaks the synthetic shape the body carries the same objects,
            # and the header capture may be absent on that same response.
            body_rolling, body_weekly = _parse_quotas(body)
            # A relay capability-deny record (bead .147 §1E marker) arrives
            # as status 429 but is a POLICY signal, not an upstream 429:
            # record its reason only — it must never become park evidence
            # (it sets neither `limited` nor `last_429_at`).
            deny_reason = _relay_deny_reason(record, body)
            if rolling is None and weekly is None and body is None and deny_reason is None:
                return False

            ts = _as_num(record.get("ts"))
            prev = self.providers.get(key)
            state = (
                prev
                if prev is not None
                else QuotaState(provider=key, updated_at=ts)
            )
            if rolling is None and body_rolling is not None:
                rolling = body_rolling
            if weekly is None and body_weekly is not None:
                weekly = body_weekly
            if rolling is not None:
                state = replace(state, rolling=rolling, limited=rolling.limited)
                if rolling.next_tick_at is not None:
                    state = replace(state, next_tick_at=rolling.next_tick_at)
            if weekly is not None:
                state = replace(state, weekly=weekly)
            if body is not None and deny_reason is None:
                state = replace(state, last_429_at=ts)
            if deny_reason is not None:
                state = replace(state, last_deny_reason=deny_reason)
            state = replace(state, updated_at=ts if ts is not None else state.updated_at)
            self.providers[key] = state
            return True
        except Exception:  # noqa: BLE001 - a fold must never crash the loop
            return False


# --------------------------------------------------------------------------- #
# Field extraction helpers — each returns None for absent/mistyped input.     #
# --------------------------------------------------------------------------- #
def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _as_num(v: Any) -> float | None:
    return float(v) if _is_num(v) else None


def _as_str(v: Any) -> str | None:
    if isinstance(v, str):
        s = v.strip()
        return s or None
    return None


def _as_bool(v: Any) -> bool | None:
    return v if isinstance(v, bool) else None


def _provider_key(record: dict[str, Any]) -> str | None:
    for key_field in _KEY_FIELDS:
        key = _as_str(record.get(key_field))
        if key is not None:
            return key
    return None


def _relay_deny_reason(record: dict[str, Any], body: str | None) -> str | None:
    """The relay capability-deny reason on a record, or ``None`` if the
    record is not a relay deny.

    A relay deny (bead .147 §1E) is identified by the
    ``x-stigmergy-deny-reason`` marker — the header name as a record key, or
    the ``stigmergy_relay_deny`` type inside the (verbatim-captured) 429
    body's JSON — NEVER by status alone: a relay deny may ride a 429 status
    (quota-calls / quota-tokens) without being upstream quota evidence.
    """
    reason = _as_str(record.get("x-stigmergy-deny-reason"))
    if reason is not None:
        return reason
    if body is not None:
        try:
            obj = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return None
        if isinstance(obj, dict):
            err = obj.get("error")
            if isinstance(err, dict) and err.get("type") == "stigmergy_relay_deny":
                r = _as_str(err.get("reason"))
                if r is not None:
                    return r
    return None


def _parse_dollar_string(v: Any) -> float | None:
    """Parse a dollar-string credit field ("$87.66") to a float; ``None``
    for anything that is not a plain numeric (optionally $-prefixed) string."""
    if not isinstance(v, str):
        return None
    s = v.strip()
    if s.startswith("$"):
        s = s[1:]
    try:
        return float(s)
    except ValueError:
        return None


def _parse_rolling(obj: Any) -> RollingWindow | None:
    """Fold the `rollingFiveHourLimit` object. Returns ``None`` unless the
    object is well-typed: numeric ``remaining``/``max``, boolean
    ``limited``; ``nextTickAt`` is a verbatim string or omitted."""
    if not isinstance(obj, dict):
        return None
    remaining = _as_num(obj.get("remaining"))
    max_ = _as_num(obj.get("max"))
    limited = _as_bool(obj.get("limited"))
    if remaining is None or max_ is None or limited is None:
        return None
    return RollingWindow(
        remaining=remaining,
        max=max_,
        limited=limited,
        next_tick_at=_as_str(obj.get("nextTickAt")),
    )


def _parse_weekly(obj: Any) -> WeeklyCredit | None:
    """Fold the `weeklyTokenLimit` object: the dollar-string credit fields.
    Returns ``None`` if the object is not a dict or carries no parseable
    credit field at all."""
    if not isinstance(obj, dict):
        return None
    used = _parse_dollar_string(obj.get("creditUsed"))
    rem = _parse_dollar_string(obj.get("creditRemaining"))
    lim = _parse_dollar_string(obj.get("creditLimit"))
    if used is None and rem is None and lim is None:
        return None
    return WeeklyCredit(credit_used=used, credit_remaining=rem, credit_limit=lim)


def _parse_quotas(raw: str | None) -> tuple[RollingWindow | None, WeeklyCredit | None]:
    """Parse the `synthetic_quotas` string (a JSON object, spike bundle 138
    shape). A truncated or non-object JSON body yields ``(None, None)`` —
    the tolerance contract."""
    if raw is None:
        return None, None
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None, None
    if not isinstance(obj, dict):
        return None, None
    return _parse_rolling(obj.get("rollingFiveHourLimit")), _parse_weekly(
        obj.get("weeklyTokenLimit")
    )


# --------------------------------------------------------------------------- #
# Offset-based feed reader                                                    #
# --------------------------------------------------------------------------- #
def _clamped_offset(offset: Any, size: int) -> int:
    """Clamp a previously returned offset into ``[0, size]``: non-numeric,
    non-finite, negative, or pre-truncation offsets fall back to a safe
    position instead of raising."""
    if not _is_num(offset):
        return 0
    o = int(offset)
    if o < 0:
        return 0
    return min(o, size)


def read_feed(
    path: str | Path, offset: int | float = 0
) -> tuple[QuotaGovernor, int]:
    """Read the relay JSONL feed from ``offset`` and fold the new lines.

    Args:
        path: the relay JSONL log path (the .147 §1E per-dispatch log).
        offset: the byte offset returned by the previous call (``0`` for
            the first read).

    Returns:
        ``(governor, next_offset)`` — the :class:`QuotaGovernor` holding
        the per-provider state folded from exactly the lines read since
        ``offset`` (a fresh governor per call — callers persist and merge
        across calls, and the module itself retains no state), and the
        byte offset to pass as ``offset`` on the next call (the end of
        the last complete line read).

    The reader seeks to ``offset`` rather than re-reading the file; only
    the appended bytes are read and only a per-line buffer exists at a
    time. A line that does not end in a newline (the writer was mid-line
    when the snapshot was taken) is NOT consumed — the next offset stops
    before it, so it is re-read once complete. Blank lines, invalid-JSON
    lines, and lines with no usable provider/quota data are skipped
    individually. This function never raises: a missing file or any other
    I/O failure yields an empty governor and offset ``0`` (the caller
    restarts from the beginning on the next read) rather than crashing.
    """
    try:
        p = Path(path)
        size = p.stat().st_size
        start = _clamped_offset(offset, size)
        next_offset = start
        try:
            with p.open("rb") as f:
                if start:
                    f.seek(start)
                data = f.read()
        except FileNotFoundError:
            data = b""
        lines = data.split(b"\n")
        tail = lines[-1]
        if tail:
            # The last line has no terminating newline (the writer was
            # mid-line at snapshot time): read only the complete lines and
            # stop the offset before the partial one, so it is re-read once
            # complete.
            chunks = lines[:-1]
            next_offset = start + len(data) - len(tail)
        else:
            # The chunk ends in a newline (or is empty): every line is
            # complete (the empty final element is skipped as blank).
            chunks = lines[:-1]
            next_offset = start + len(data)
        governor = QuotaGovernor()
        for chunk in chunks:
            line = chunk.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue  # invalid / truncated JSON: skip this line only
            if not isinstance(record, dict):
                continue
            governor.fold(record)
        return governor, next_offset
    except Exception:  # noqa: BLE001 - the feed reader must never crash a caller
        # Bounded degradation: nothing was folded and the caller restarts
        # from the beginning on the next read.
        pass
    return QuotaGovernor(), 0


def feed_snapshot(path: str | Path, offset: int | float = 0) -> tuple[QuotaGovernor, int]:
    """Alias for :func:`read_feed` — name for callers that prefer the
    "snapshot at an offset" framing over the streaming one."""
    return read_feed(path, offset)


# --------------------------------------------------------------------------- #
# Decision logic (pure, deterministic — no daemon plumbing, no daemon         #
# fixture: the inputs are the pricing class + the provider's QuotaState).     #
# --------------------------------------------------------------------------- #
# The `x-stigmergy-deny-reason` marker (relay_transport.py, bead .147 §1E):
# a per-dispatch capability deny (the relay synthesized its 429 itself from a
# CapabilityDenied — quota-calls/quota-tokens/revoked/unknown) carries this
# marker and is a POLICY signal, never upstream quota evidence. Such a record
# is folded into `QuotaState.last_deny_reason` only, and is NEVER transformed
# into park evidence anywhere in this module: `_exhausted_evidence` reads
# exclusively `limited` and `last_429_at`, and `fold` never sets either from
# a deny record.

#: Default park-wait ceiling: 8 hours. A park whose wait exceeds it carries
#: an escalation-due signal while remaining parked.
DEFAULT_ESCALATION_CEILING_S = 8 * 3600.0


@dataclass(frozen=True)
class GovernorDecision:
    """One governor decision for one provider's quota state.

    ``decision`` discriminates the three outcomes:

    - ``"dispatch"``            — dispatch normally (the default for every
                                   non-subscription pricing class, and for a
                                   subscription with no current exhausted
                                   evidence);
    - ``"park-until"``          — park until ``park_until`` (an explicit
                                   epoch-seconds timestamp computed from the
                                   quota state's next-tick timestamp);
    - ``"park-escalation-due"`` — parked as above AND the wait from ``now``
                                   to ``park_until`` exceeds the caller-
                                   supplied escalation ceiling.

    Escalation is exactly-once-per-park by construction: the signal is the
    pure predicate ``park_until - now > ceiling``, asserted at the moment of
    parking (wait exceeds the ceiling) and de-asserted exactly once, when
    the remaining wait shrinks down to the ceiling — it is never asserted
    twice and it NEVER clears the park (the escalation outcome keeps the
    same ``park_until``). No code path here selects a different provider or
    pricing class.
    """

    decision: str
    park_until: float | None = None
    escalation_due: bool = False


def _resolve_pricing_class(pricing: PricingClass | ModelEntry) -> PricingClass:
    """Accept a pricing class directly, or a :class:`ModelEntry` whose
    ``pricing`` resolves it — callers may pass either."""
    if isinstance(pricing, PricingClass):
        return pricing
    if isinstance(pricing, ModelEntry):
        return pricing.pricing
    raise TypeError(f"expected a PricingClass or ModelEntry, got {type(pricing).__name__}")


def _exhausted_evidence(state: QuotaState | None) -> bool:
    """True only on quota-exhausted EVIDENCE from the provider: the
    ``limited`` flag set, or an upstream 429 record for the provider
    (``last_429_at`` set). A capability-deny record is deliberately not
    evidence: it never enters this check and nothing in this module
    converts one into ``limited``/``last_429_at``."""
    if state is None:
        return False
    return state.limited or state.last_429_at is not None


def _parse_next_tick(next_tick_at: str | None) -> float | None:
    """The next-tick ISO-8601 timestamp as epoch seconds; ``None`` if absent
    or unparseable (a park with no computable end is a park with no
    escalation — the deadline is unknown, not overdue)."""
    if next_tick_at is None:
        return None
    s = next_tick_at.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


def decide(
    pricing: PricingClass | ModelEntry,
    state: QuotaState | None,
    *,
    now: float,
    escalation_ceiling_s: float = DEFAULT_ESCALATION_CEILING_S,
) -> GovernorDecision:
    """Pure, deterministic governor decision for ONE provider.

    Args:
        pricing: the pricing class of the dispatch's model (or a
            :class:`ModelEntry` — its ``pricing`` is resolved from it).
        state: the current :class:`QuotaState` for the provider (``None`` if
            the provider has never reported — treated as no evidence).
        now: the decision time, epoch seconds (explicit, for determinism —
            this function never calls a clock).
        escalation_ceiling_s: the wait beyond which a park carries the
            escalation-due signal (default 8 hours).

    Returns:
        A :class:`GovernorDecision`:

        - Non-subscription classes (``metered``/``local``): ALWAYS
          ``"dispatch"``, regardless of the quota state. There is no branch
          below that can park a metered or local lane.
        - Subscription without exhausted evidence (no ``limited`` flag, no
          upstream 429 record, provider never reported, or the state's only
          record is a capability deny): ``"dispatch"``. A capability-deny
          record sets only ``last_deny_reason`` (never ``limited``/
          ``last_429_at``), so it is structurally incapable of producing
          exhausted evidence here.
        - Subscription WITH exhausted evidence: parked. The park ends at the
          state's next-tick timestamp (``next_tick_at`` as epoch seconds);
          whenever the wait from ``now`` to that timestamp exceeds
          ``escalation_ceiling_s`` the decision is
          ``"park-escalation-due"`` — still parked, same ``park_until``.

    Invariants: escalation never clears the park (the escalation outcome
    keeps the identical ``park_until``); this function never selects or
    suggests a different provider or pricing class — it is a pure decision
    over the single (class, state) pair it is given.
    """
    cls = _resolve_pricing_class(pricing)

    # Non-subscription lanes NEVER park — metered and local quota exhaustion
    # is budgeted through the spend leash and dispatch ceilings elsewhere;
    # no quota-state input of any shape can park one.
    if cls is not PricingClass.SUBSCRIPTION:
        return GovernorDecision(decision="dispatch")

    if not _exhausted_evidence(state):
        return GovernorDecision(decision="dispatch")

    park_until = _parse_next_tick(state.next_tick_at) if state is not None else None
    if park_until is None:
        # Exhausted, but the state carries no computable next-tick deadline
        # (and the limited flag is not yet observed clearing): park with no
        # escalation — an unknown deadline is not an overdue one.
        return GovernorDecision(decision="park-until", park_until=None)

    escalation_due = (park_until - now) > escalation_ceiling_s
    if escalation_due:
        return GovernorDecision(
            decision="park-escalation-due", park_until=park_until, escalation_due=True
        )
    return GovernorDecision(decision="park-until", park_until=park_until)
