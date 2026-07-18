"""The LIVE credential-relay HTTP transport (bead .31; SPEC.md §4 credentials
"out of the worker namespace entirely", §10 AC4). SECURITY-CRITICAL — the
single highest-security component in the system (network-facing provider-
credential handling), reviewed line-by-line (bead31-build-spec.md).

**What this module is.** ``stigmergy.relay`` (bead .12, frozen) is the
offline, deterministic relay CORE: ``CredentialRelay.handle()`` is
synchronous and full-body-buffered. This module wraps that core in a real
asyncio HTTP server terminating a worker's inference request over a
per-dispatch unix domain socket, streaming the response back incrementally
(avoiding claude-code's TTFB-abort risk on large-context tickets — see
bead31-build-spec.md §0), and forwards upstream via ``urllib.request`` (the
smallest correct HTTP/TLS client available in the stdlib — a hand-rolled
HTTP/1.1 client would be the single largest attack-surface addition this
bead could make).

**Threat model (SPEC §4): every worker is assumed compromised.** The worker
holds ONLY a per-dispatch capability token; the real provider key lives only
inside the relay process, injected toward the upstream request only. Every
fail-closed invariant from ``relay.py`` (bead .12) carries through the
``prepare_upstream`` seam unchanged: a missing/denied capability never
reaches ``forwarder`` and never even fetches the real key
(``relay._key_provider``).

**The hard parts (bead31-build-spec.md AMENDMENT, normative):**

- *Response framing.* ``urllib``'s ``resp.read(N)`` yields the DE-CHUNKED,
  DECODED entity body — copying upstream's framing headers verbatim while
  streaming would corrupt the response. So every worker-facing response is
  written with :data:`HOP_BY_HOP` headers stripped, ``Connection: close``
  added, and NO ``Content-Length``/``Transfer-Encoding`` — the body is
  close-delimited (worker reads to EOF). This single incremental path
  handles both SSE (``stream:true``) and plain JSON (``stream:false``).
  ``content-type``/``content-encoding`` are preserved verbatim (this module
  never gunzips).
- *Usage-parser selection by response content-type*: ``text/event-stream``
  routes through an incremental, bounded :class:`_SseUsageMeter` (bead .56,
  F4 — fed chunk-by-chunk as they are forwarded, never retaining the whole
  body); anything else routes through ``relay.extract_usage`` on a bounded
  JSON accumulator (``_MAX_JSON_METER_BYTES``).
- *Reserve-then-reconcile metering (§C + concurrency TOCTOU)*: the call slot
  is charged (``calls=1``) NOW — atomically with ``prepare_upstream``'s
  ``authorize`` (no ``await`` between them, so the single-threaded event loop
  cannot interleave another connection in the gap) — BEFORE the forward, and
  output tokens are reconciled (``calls=0``) after the stream. This closes a
  TOCTOU: ``authorize`` alone left ``used_calls`` at 0 until the stream ended,
  so a compromised worker opening N concurrent connections would pass all N
  authorize checks and forward the real key N times, defeating the SPEC §4
  ``max_calls``/``max_output_tokens`` leash. Reserving up front bounds
  concurrent forwards to ``max_calls``. It also subsumes §C's "no unmetered
  retries": a call that fails upstream (synthesized 502) stays charged because
  the reserve already landed. A ``DenyResponse`` path charges nothing (never
  authorized). The .12 sync ``handle()`` path needs no reserve — single-shot,
  no concurrency window.
- *Bounded memory (§D) + fail-closed metering (bead .56, F4)*: the
  urllib-reading worker thread is bridged to the async caller via a BOUNDED
  queue (natural backpressure — a stalled worker cannot make this process
  buffer the whole upstream response). Usage parsing NEVER retains the whole
  body: SSE is metered incrementally with a bounded pending-event buffer
  (:class:`_SseUsageMeter`); JSON is accumulated into a capped buffer. When a
  **2xx** response's true usage cannot be determined (an oversized SSE event,
  an over-cap JSON body, or a truncated/timed-out stream) the call is charged
  **unbudgetable** (the remaining ``max_output_tokens`` — see
  ``CapabilityStore.charge_unbudgetable``), never ~0 — this closes the
  bead .56 leash-bypass finding (F4) where a truncated tee under-charged a
  call whose worker still received every token.
- *Accept cap + stream deadline (bead .56, F5)*: ``serve_relay``'s
  ``max_in_flight`` (default 1) bounds concurrent ``_handle_connection``s via
  an ``asyncio.Semaphore`` created inside the running loop; ``stream_deadline``
  (default ``_DEFAULT_STREAM_DEADLINE``) bounds how long the forward+stream
  phase may hold an upstream socket open via ``asyncio.timeout``.
- *Socket perms (§B, deliberately BETTER than ``egress_proxy``, which does
  no explicit chmod)*: the per-dispatch runtime dir is created ``0o700``
  and the socket is ``chmod``'d ``0o600`` immediately after bind.

**Known gaps (logged, not hidden — bead31-build-spec.md, updated bead .56):**
- Single-threaded relay daemon (one worker/dispatch at a time, v0);
  ``max_in_flight`` > 1 / true parallelism (v1) re-opens output-token
  overshoot (F6) and needs up-front token reservation before it is safe —
  see :func:`serve_relay`'s docstring.
- No request-body chunked-transfer-encoding support (claude-code sends
  ``Content-Length``); rejected explicitly, never silently mishandled.
"""

from __future__ import annotations

import asyncio
import http.client
import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from stigmergy.critic_client import _NoRedirectHandler
from stigmergy.relay import (
    CapabilityDenied,
    CredentialRelay,
    DenyResponse,
    RelayError,
    RelayRequest,
    RelayResponse,
    UpstreamRequest,
    _find_header,
    extract_usage,
    prepare_upstream,
)

# --------------------------------------------------------------------------- #
# Framing constants                                                            #
# --------------------------------------------------------------------------- #

# Headers stripped from every worker-facing response (bead31 amendment §A):
# hop-by-hop + any framing header that could corrupt the close-delimited
# transport this module always uses (a stale `transfer-encoding: chunked`
# forwarded over already-de-chunked bytes would make the worker
# double-de-chunk; a stale `content-length` would no longer match the bytes
# actually written).
HOP_BY_HOP: frozenset[str] = frozenset(
    {
        "connection",
        "keep-alive",
        "transfer-encoding",
        "content-length",
        "te",
        "trailer",
        "upgrade",
        "proxy-authenticate",
        "proxy-authorization",
    }
)

_BODY_METHODS: frozenset[str] = frozenset({"POST", "PUT", "PATCH"})

_HEAD_READ_CHUNK = 4096
_BODY_READ_CHUNK = 65536

_DEFAULT_MAX_HEAD_BYTES = 65536
_DEFAULT_MAX_BODY_BYTES = 32 * 1024 * 1024
_DEFAULT_REQUEST_TIMEOUT = 30.0

# Bound on how much of a non-SSE (JSON) upstream response body this module
# will accumulate for usage parsing (bead31 amendment §D; renamed from
# `_MAX_TEE_BYTES` in bead .56 — the SSE path no longer tees the whole body,
# see `_SseUsageMeter`, so "tee" is now a JSON-only name) — past this cap,
# chunks are still forwarded to the worker in full, just no longer
# accumulated for parsing. Bead .56 (F4): a JSON body whose usage field
# lands past this cap is no longer silently under-charged — it is now
# charged UNBUDGETABLE (fail-closed), never ~0.
_MAX_JSON_METER_BYTES = 8 * 1024 * 1024

# Bound on a single pending (not-yet-terminated) SSE event `_SseUsageMeter`
# will hold before giving up on it as unparseable (bead .56, F4). 1 MiB is
# enormously generous vs real Anthropic events (<64 KiB) — this fires only
# on genuinely malformed/adversarial framing, at which point the meter goes
# `_unmeterable` (fail-closed: the caller charges the remaining budget).
_MAX_SSE_EVENT_BYTES = 1 * 1024 * 1024

# Default wall-clock deadline (seconds) on the forward+stream phase of one
# connection (bead .56, F5) — generous so as not to cut a legitimate long
# generation short; a deadline breach tears down the upstream socket and
# (on an in-flight 2xx) charges the capability unbudgetable rather than
# holding a real-key-bearing connection open indefinitely.
_DEFAULT_STREAM_DEADLINE = 300.0

# Bound on the bridge queue between the blocking urllib-reading thread and
# the async consumer (bead31 amendment §D) — natural backpressure: a
# stalled worker blocks the upstream-reading thread, capping in-flight
# memory at roughly queue-depth * chunk_size, never the total response size.
_BRIDGE_QUEUE_SIZE = 8

_START_TIMEOUT_SECONDS = 10.0
_STOP_TIMEOUT_SECONDS = 10.0

# Passed to asyncio.start_unix_server's `limit=` (flow-control watermark for
# the underlying transport, not a hard cap — request_relay_request enforces
# the real bound itself via max_head_bytes/max_body_bytes).
_STREAM_LIMIT = 1 << 20


class RelayTransportError(Exception):
    """Raised by :func:`read_relay_request` on any request-parse/framing
    failure: malformed request line, missing ``Content-Length`` on a body
    method, a ``Transfer-Encoding`` request header (chunked request bodies
    are out of scope for v0), any duplicate header name (request-smuggling
    defense), an oversized head/body, or a read timeout. Callers MUST treat
    this as fail-closed: deny, never forward."""


class UpstreamError(Exception):
    """Raised by an injected forwarder on any connect/TLS/read/timeout/
    redirect-refused failure talking to the real upstream. Never carries the
    raw underlying exception's message (which could echo request context) —
    only the failing exception's class name — and never carries the real
    provider key, which is on the request that failed."""


@dataclass(frozen=True)
class UpstreamHead:
    """The status + RAW (pre-strip) response headers a forwarder resolved
    from the real upstream call. :func:`serve_relay` strips
    :data:`HOP_BY_HOP` names and adds ``Connection: close`` before writing
    anything derived from this toward the worker."""

    status: int
    headers: list[tuple[str, str]]


Forwarder = Callable[[UpstreamRequest], Awaitable[tuple[UpstreamHead, AsyncIterator[bytes]]]]


# --------------------------------------------------------------------------- #
# No-redirect opener (reuses .36's critic_client._NoRedirectHandler)          #
# --------------------------------------------------------------------------- #

# The real provider key rides the upstream request built from an
# UpstreamRequest — a 3xx MUST NOT be followed (urllib re-sends request
# headers, including the injected key, to the redirect target by default).
# Reuses critic_client's exact no-redirect handler class (same guard, same
# failure shape: the redirect raises HTTPError -> mapped to UpstreamError).
_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirectHandler)


# --------------------------------------------------------------------------- #
# sse_extract_usage / _SseUsageMeter — PURE, never raise (bead31 build spec    #
# cases 12-18; bead .56 F4 adds the incremental, bounded, cross-chunk meter)   #
# --------------------------------------------------------------------------- #

# Recognizes an SSE event terminator (a blank line) tolerantly across five
# framings: CRLF (`\r\n\r\n`), the two MIXED blank-line pairs
# (`\r\n\n`, `\n\r\n`), LF (`\n\n`), and bare-CR (`\r\r`). The `\r\n`-
# containing alternatives are matched FIRST (regex alternation tries
# left-to-right at each position) so a split `\r\n` across a chunk boundary
# never forms a false boundary via a shorter alternative matching first.
# Matched against RAW (never pre-normalized) bytes — normalizing `\r`->`\n`
# before searching for boundaries would let a `\r\n` split across a chunk
# boundary (the trailing `\r` normalized alone, then the next chunk's
# leading `\n` joins it) masquerade as a false `\n\n` blank line. Matching
# raw bytes over the full cumulative pending buffer sidesteps that
# entirely: an unresolved/ambiguous trailing `\r` simply never matches any
# alternative here and stays in `pending` until more bytes arrive to
# disambiguate it. Bead .56 round 2 (review-56-codex.md, MINOR "mixed
# newline"): exotic/mixed framing that this still fails to recognize does
# NOT silently under-charge — the tri-state metering (§8 fix A) makes a
# missed final delta fail CLOSED (finalize() -> None -> unbudgetable),
# never a silent zero/placeholder charge. This is no longer a claim of
# "behaviorally identical to the batch parser" for every exotic input, only
# that recognized framings are parsed identically and unrecognized framings
# fail closed rather than under-charge.
_SSE_EVENT_TERMINATOR = re.compile(rb"\r\n\r\n|\r\n\n|\n\r\n|\n\n|\r\r")


def _sse_event_usage_update(
    block: str, input_tokens: int | None, output_tokens: int | None
) -> tuple[int | None, int | None, str | None]:
    """Shared per-event body -> usage-update logic used by BOTH
    :func:`sse_extract_usage` (batch) and :class:`_SseUsageMeter`
    (incremental) so the two parsers stay behaviorally identical (bead .56).

    ``block`` is one already-delimited SSE event's text, LF-normalized
    (``\\r\\n``/``\\r`` already collapsed to ``\\n``) — collects its
    ``data:`` lines (skipping blank lines and ``:``-prefixed comments),
    ignores the ``data: [DONE]`` sentinel, and updates the running
    ``input_tokens``/``output_tokens`` per Anthropic's two usage shapes:
    ``message.usage.{input_tokens,output_tokens}`` (``output_tokens`` only
    if still ``None`` — the ``message_start`` placeholder) and top-level
    ``usage.{output_tokens,input_tokens}`` (``output_tokens`` ALWAYS
    overwrites — cumulative, last ``message_delta`` wins). Ints only;
    ``bool`` is rejected. Never raises: an unparseable event returns the
    running totals unchanged.

    Returns a 3-tuple ``(input_tokens, output_tokens, final_kind)`` where
    ``final_kind`` is the tri-state "positively determined" signal for
    THIS event's TOP-LEVEL ``usage.output_tokens`` (bead .56 round 3, §9
    fix H — latest-usage VALIDITY, not sticky-OR):

    - a valid non-negative int -> ``"valid"`` (recorded into
      ``output_tokens``, the cumulative ``message_delta`` measurement).
    - present but a negative int -> the value IS still recorded into
      ``output_tokens`` (BATCH-COMPAT: :func:`sse_extract_usage` must keep
      surfacing a negative value exactly as before — see its own
      docstring/tests), but the signal is ``"invalid"`` — a later,
      genuinely final delta that regresses to negative must not let an
      earlier valid value keep being trusted by the incremental meter.
    - present but not an int (or is a ``bool``) -> NOT recorded (also
      batch-compat — a non-int/bool never overwrote ``output_tokens``
      before either) and the signal is ``"invalid"``.
    - top-level ``usage`` present without an ``output_tokens`` key, OR the
      ``message_start`` (``message.usage``) placeholder path, OR no usage
      at all -> ``None`` (no opinion this event; the running
      "positively determined"/"unmeterable" state carries over unchanged).

    :func:`sse_extract_usage` ignores this 3rd element entirely, so its own
    observable behavior (INCLUDING surfacing a negative top-level value) is
    byte-identical to before round 3. :class:`_SseUsageMeter` uses it to
    track LATEST validity (not sticky-OR): ``"valid"`` sets
    ``_final_output_seen``; ``"invalid"`` sets ``_unmeterable`` (STICKY — a
    later valid value must never rescue an earlier invalid one, because the
    invalid one might itself have been the true, corrupt final answer).
    """
    data_lines: list[str] = []
    for line in block.split("\n"):
        if not line or line.startswith(":"):
            continue  # blank / keep-alive comment
        if line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
        # `event:` and any other field lines are informational only — this
        # parser keys off the parsed JSON shape, not the event name.

    if not data_lines:
        return input_tokens, output_tokens, None
    data_str = "\n".join(data_lines)
    if data_str == "[DONE]":
        return input_tokens, output_tokens, None

    try:
        parsed = json.loads(data_str)
    except Exception:  # noqa: BLE001 - never raise (bead .56 round 3, §9 fix J);
        # broadened from (JSONDecodeError, ValueError) so a RecursionError or
        # Python's int-string-digit-limit ValueError on a bounded event never
        # escapes this helper — that event simply contributes nothing.
        return input_tokens, output_tokens, None
    if not isinstance(parsed, dict):
        return input_tokens, output_tokens, None

    message = parsed.get("message")
    if isinstance(message, dict):
        usage = message.get("usage")
        if isinstance(usage, dict):
            it = usage.get("input_tokens")
            if isinstance(it, int) and not isinstance(it, bool):
                input_tokens = it
            if output_tokens is None:
                ot = usage.get("output_tokens")
                if isinstance(ot, int) and not isinstance(ot, bool):
                    output_tokens = ot
        return input_tokens, output_tokens, None

    usage = parsed.get("usage")
    final_kind: str | None = None
    if isinstance(usage, dict) and "output_tokens" in usage:
        # NB: membership check (not `.get()`), so a present-but-``null``
        # value is distinguished from an ABSENT key — both decode to Python
        # ``None`` via `.get()`, but only the former is a genuinely present,
        # invalid measurement (round 3, §9 fix H: `null` must signal
        # "invalid", not "no opinion").
        ot = usage["output_tokens"]
        if isinstance(ot, int) and not isinstance(ot, bool):
            output_tokens = ot  # cumulative -> last wins, always overwrite
            final_kind = "valid" if ot >= 0 else "invalid"
        else:
            final_kind = "invalid"
        it = usage.get("input_tokens")
        if isinstance(it, int) and not isinstance(it, bool):
            input_tokens = it
    elif isinstance(usage, dict):
        it = usage.get("input_tokens")
        if isinstance(it, int) and not isinstance(it, bool):
            input_tokens = it
    return input_tokens, output_tokens, final_kind


def sse_extract_usage(body: bytes) -> dict:
    """Parse Anthropic-style SSE usage out of an accumulated response body.

    Anthropic SSE carries usage across two event shapes:
    ``event: message_start`` -> ``data:`` JSON with
    ``.message.usage.{input_tokens,output_tokens}`` (the ``output_tokens``
    here is an early/placeholder value, NOT the final count), and
    ``event: message_delta`` -> ``data:`` JSON with a top-level
    ``.usage.output_tokens`` that is CUMULATIVE — the last ``message_delta``
    seen carries the true total (never summed across events).

    Tolerant by construction: comment lines (``:`` prefix), ``event: ping``,
    the ``data: [DONE]`` sentinel, blank lines, CRLF or LF framing, and
    truncated/garbage bytes are all handled without raising — an
    unparseable or missing usage field simply yields a partial (or empty)
    dict; the call still counts toward ``max_calls`` via
    :meth:`stigmergy.relay.CapabilityStore.charge`'s default ``calls=1``,
    it just charges fewer (or zero) output tokens for that call (the
    documented v0 under-meter gap on truncated streams).

    A plain JSON (``stream:false``) body has no SSE framing at all — this
    function then finds no ``data:`` lines and returns ``{}``, so the
    caller (:func:`serve_relay`) falls back to
    ``stigmergy.relay.extract_usage`` for that content-type.

    This function's own observable behavior is unchanged by bead .56 — it
    still parses the WHOLE ``body`` in one pass (batch parser). Its
    per-event logic is now shared with the incremental
    :class:`_SseUsageMeter` via :func:`_sse_event_usage_update` so the two
    stay behaviorally identical, but ``sse_extract_usage`` itself is not
    bounded/incremental (callers that must avoid whole-body retention use
    :class:`_SseUsageMeter` instead).
    """
    if not body:
        return {}
    try:
        text = body.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - never raise, always fail to {}
        return {}

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = normalized.split("\n\n")

    input_tokens: int | None = None
    output_tokens: int | None = None

    for block in blocks:
        input_tokens, output_tokens, _final_seen = _sse_event_usage_update(
            block, input_tokens, output_tokens
        )

    result: dict = {}
    if input_tokens is not None:
        result["input_tokens"] = input_tokens
    if output_tokens is not None:
        result["output_tokens"] = output_tokens
    return result


class _SseUsageMeter:
    """Stateful, bounded, cross-chunk incremental SSE usage parser (bead .56,
    F4) — the fail-closed alternative to metering only after the WHOLE
    response body has been accumulated (the leash-bypass this bead closes:
    a final ``message_delta``'s cumulative ``output_tokens`` landing past a
    byte cap was silently dropped, under-charging a call whose worker still
    received every token). Feed forwarded chunks as they stream; call
    :meth:`finalize` once the stream ends.

    Only a BOUNDED ``pending`` buffer (bytes not yet forming a complete SSE
    event) is ever retained — never the whole body. An event is terminated
    by a blank line, recognized tolerantly via :data:`_SSE_EVENT_TERMINATOR`
    (``\\n\\n``, ``\\r\\n\\r\\n``, mixed ``\\r\\n\\n``/``\\n\\r\\n``, or
    ``\\r\\r`` framing) matched against RAW, cumulative ``pending`` bytes —
    never pre-normalized, so a split ``\\r\\n`` is never mistaken for a
    blank line (see :data:`_SSE_EVENT_TERMINATOR`'s docstring). Usage is
    tracked with IDENTICAL shape logic to :func:`sse_extract_usage` (both
    call :func:`_sse_event_usage_update`) so the two parsers stay
    behaviorally identical on any recognized framing.

    Pure/never-raise, mirroring :func:`sse_extract_usage`'s tolerance: any
    decode/parse failure on one event is swallowed (that event contributes
    nothing), never raised; non-``bytes``/``bytearray`` input to
    :meth:`feed` also never raises — it transitions the meter to
    ``_unmeterable`` instead (bead .56 round 2, §8 fix F). A single COMPLETE
    (delimited) event whose own bytes exceed ``_MAX_SSE_EVENT_BYTES``, or an
    unterminated pending tail past the same bound, is unparseable — the
    meter goes ``_unmeterable`` and drops its buffer; the caller MUST then
    treat usage as unknown (:meth:`finalize` returns ``None``) and charge
    the capability unbudgetable (never ~0).

    **Tri-state (bead .56 round 2, §8 fix A).** ``finalize()`` returns an
    ``int`` ONLY when a valid final usage was POSITIVELY determined — a
    top-level ``usage.output_tokens`` (the cumulative ``message_delta``
    value) parsed as a valid non-negative int at least once. This is
    tracked via ``_final_output_seen``, set only by
    :func:`_sse_event_usage_update`'s 3rd return value (``"valid"``). The
    ``message_start`` placeholder (``message.usage.output_tokens``) never
    sets it. A stream that ends without ever positively determining a final
    usage — even if it parsed cleanly and hit EOF — is UNMETERABLE
    (``finalize()`` returns ``None``), never charged at the placeholder or
    at 0. This closes the residual F4 leash-bypass a cross-family review
    found in round 1: conflating "parser didn't raise and the iterator
    ended" with "true usage was determined".

    **Latest-usage validity, not sticky-OR (bead .56 round 3, §9 fix H).**
    ``output_tokens`` is CUMULATIVE — the last ``message_delta`` is
    authoritative. So an ``"invalid"`` ``final_kind`` (a present-but-
    negative/non-int/bool top-level ``output_tokens``, INCLUDING an
    explicit JSON ``null``) sets ``_unmeterable`` STICKILY: the concern is
    a genuinely LATER, corrupted cumulative delta being silently ignored in
    favor of an earlier, lower valid one (which would under-charge) — so
    once a final-usage signal has been invalid, no later valid value
    "rescues" the meter. Once ``_unmeterable``, :meth:`finalize` always
    returns ``None`` regardless of any further updates to ``output_tokens``.
    """

    def __init__(self) -> None:
        self.pending = bytearray()
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None
        self._unmeterable = False
        self._final_output_seen = False

    def feed(self, chunk: bytes) -> None:
        """Append ``chunk`` and extract every complete SSE event now
        available; a no-op once :meth:`finalize`-triggering unmeterable
        state has been reached (bytes are still forwarded to the worker by
        the caller regardless — metering just stays failed-closed). Never
        raises: non-``bytes``/``bytearray`` input transitions to
        ``_unmeterable`` instead (bead .56 round 2, §8 fix F)."""
        if self._unmeterable:
            return
        if not isinstance(chunk, (bytes, bytearray)):
            self._unmeterable = True
            self.pending = bytearray()
            return

        try:
            self.pending.extend(chunk)

            data = bytes(self.pending)
            pos = 0
            for match in _SSE_EVENT_TERMINATOR.finditer(data):
                # §8 fix F: the per-event size bound applies to a COMPLETE
                # (delimited) event too, not only an unterminated pending
                # tail — a single event whose own length exceeds the bound
                # is malformed/adversarial regardless of whether it was
                # eventually terminated.
                if match.start() - pos > _MAX_SSE_EVENT_BYTES:
                    self._unmeterable = True
                    self.pending = bytearray()
                    return
                self._consume_event(data[pos : match.start()])
                pos = match.end()
            if pos:
                del self.pending[:pos]

            if len(self.pending) > _MAX_SSE_EVENT_BYTES:
                self._unmeterable = True
                self.pending = bytearray()
        except Exception:  # noqa: BLE001 - never raise; fail closed instead
            self._unmeterable = True
            self.pending = bytearray()

    def _consume_event(self, event_bytes: bytes) -> None:
        try:
            text = event_bytes.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - never raise, that event contributes nothing
            return
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        self.input_tokens, self.output_tokens, final_kind = _sse_event_usage_update(
            normalized, self.input_tokens, self.output_tokens
        )
        # Round 3, §9 fix H: latest-usage VALIDITY, not sticky-OR. A
        # "valid" signal marks a final usage as positively determined;
        # an "invalid" signal (a present-but-negative/non-int/bool/null
        # top-level output_tokens) is STICKY-unmeterable — a later valid
        # value must never rescue an earlier invalid one, because usage is
        # cumulative and the invalid one may be the genuinely later,
        # corrupt final answer.
        if final_kind == "valid":
            self._final_output_seen = True
        elif final_kind == "invalid":
            self._unmeterable = True

    def finalize(self) -> int | None:
        """Process any remaining ``pending`` as a final (best-effort,
        possibly unterminated) event, then return the metered
        ``output_tokens`` — but ONLY if a valid final usage was positively
        determined (``_final_output_seen``) and the meter never went
        unmeterable; otherwise ``None`` (the caller must then charge
        unbudgetable, never ~0 — bead .56 round 2, §8 fix A tri-state)."""
        if not self._unmeterable and self.pending:
            tail = bytes(self.pending)
            self.pending = bytearray()
            if len(tail) > _MAX_SSE_EVENT_BYTES:
                self._unmeterable = True
            else:
                self._consume_event(tail)
        if self._unmeterable or not self._final_output_seen:
            return None
        return self.output_tokens if self.output_tokens is not None else 0


# --------------------------------------------------------------------------- #
# Origin-form path validation (F1, bead .55) — layer 1 of the netloc-pinning   #
# defense: a worker-controlled request path must never be able to steer the   #
# upstream connection target when string-concatenated onto ``base_url``.      #
# --------------------------------------------------------------------------- #


def _is_origin_form_path(path: str) -> bool:
    """Strict HTTP origin-form check: exactly one leading ``/`` (never
    ``//...``, which urllib/browsers can treat as a network-path reference),
    and none of: whitespace/control characters (including ``\\t``), ``@``
    (host/userinfo separator), ``\\`` (some HTTP stacks treat this as a path
    separator), or an embedded scheme (``://``). Legitimate relay paths
    (``/v1/messages``, ``/v1/messages?beta=1``) all pass; anything a
    compromised worker could use to redirect the real-key request's netloc
    (layer 2/3 in :func:`make_urllib_forwarder`) is rejected here first."""
    if not path.startswith("/") or path.startswith("//"):
        return False
    for ch in path:
        code_point = ord(ch)
        if code_point < 0x20 or 0x7F <= code_point <= 0x9F or ch.isspace():
            return False
    if "@" in path or "\\" in path or "://" in path:
        return False
    return True


# --------------------------------------------------------------------------- #
# read_relay_request — bounded, fail-closed request parsing (cases 6-11)      #
# --------------------------------------------------------------------------- #


async def read_relay_request(
    reader: asyncio.StreamReader,
    *,
    max_head_bytes: int,
    max_body_bytes: int,
    timeout: float,
) -> RelayRequest:
    """Bounded, fail-closed request-line + header + ``Content-Length``-
    delimited body parser (bead31 build spec cases 6-11).

    Any malformed input, oversized head/body, or a stalled peer past
    ``timeout`` raises :class:`RelayTransportError` — the caller MUST treat
    this as a fail-closed deny and never forward. Chunked REQUEST bodies are
    out of scope for v0 (claude-code always sends ``Content-Length``) and
    are explicitly rejected, as is ANY duplicate header name — a
    request-smuggling defense: a duplicated ``Content-Length`` or capability
    header is never silently resolved by "pick the first/last", it is a
    deny. GET/HEAD without a body parse fine with no ``Content-Length``;
    auth still gates everything downstream in :func:`prepare_upstream`.
    """
    try:
        return await asyncio.wait_for(
            _read_relay_request_inner(
                reader, max_head_bytes=max_head_bytes, max_body_bytes=max_body_bytes
            ),
            timeout=timeout,
        )
    except RelayTransportError:
        raise
    except TimeoutError as exc:
        raise RelayTransportError("request read timed out") from exc
    except (OSError, asyncio.IncompleteReadError, asyncio.LimitOverrunError) as exc:
        raise RelayTransportError(f"request read failed: {exc.__class__.__name__}") from exc


async def _read_relay_request_inner(
    reader: asyncio.StreamReader, *, max_head_bytes: int, max_body_bytes: int
) -> RelayRequest:
    buffer = bytearray()
    while b"\r\n\r\n" not in buffer:
        if len(buffer) > max_head_bytes:
            raise RelayTransportError("request head exceeds max_head_bytes")
        chunk = await reader.read(_HEAD_READ_CHUNK)
        if not chunk:
            raise RelayTransportError("connection closed before request head completed")
        buffer.extend(chunk)

    head_bytes, _, rest = bytes(buffer).partition(b"\r\n\r\n")
    if len(head_bytes) > max_head_bytes:
        raise RelayTransportError("request head exceeds max_head_bytes")

    text = head_bytes.decode("latin-1")
    lines = text.split("\r\n")
    request_line = lines[0]
    parts = request_line.split(" ")
    if len(parts) != 3:
        raise RelayTransportError("malformed request line")
    method, path, version = parts
    if not method or not path or not version.upper().startswith("HTTP/"):
        raise RelayTransportError("malformed request line")
    if not _is_origin_form_path(path):
        raise RelayTransportError("malformed request path")

    headers: dict[str, str] = {}
    seen_lower: set[str] = set()
    for line in lines[1:]:
        if not line or ":" not in line:
            raise RelayTransportError("malformed header line")
        name, _, value = line.partition(":")
        # F9 (bead .56, LOW) + round-2 §8 fix G: the head was partitioned on
        # `\r\n\r\n` then split on `\r\n`, so a bare `\n` (or a stray `\r`
        # not part of a `\r\n` pair) embedded inside a header VALUE *or
        # NAME* survives into this raw (pre-strip) segment instead of
        # ending the line. Reject both here as a clean 400 rather than
        # passing a control character through to `forwarder`/`key_provider`
        # (neither of which is ever reached on this path).
        if "\n" in name or "\r" in name:
            raise RelayTransportError("malformed header name")
        name = name.strip()
        if "\n" in value or "\r" in value:
            raise RelayTransportError("malformed header value")
        value = value.strip()
        if not name:
            raise RelayTransportError("malformed header line")
        lname = name.lower()
        if lname in seen_lower:
            raise RelayTransportError("duplicate header (request-smuggling defense)")
        seen_lower.add(lname)
        headers[name] = value

    if "transfer-encoding" in seen_lower:
        raise RelayTransportError("chunked/transfer-encoded request bodies are not supported")

    content_length_str: str | None = None
    for name, value in headers.items():
        if name.lower() == "content-length":
            content_length_str = value
            break

    method_upper = method.upper()
    if content_length_str is None:
        if method_upper in _BODY_METHODS:
            raise RelayTransportError(f"missing Content-Length for {method_upper} request")
        body_length = 0
    else:
        if not (content_length_str.isascii() and content_length_str.isdigit()):
            raise RelayTransportError("malformed Content-Length")
        try:
            body_length = int(content_length_str)
        except ValueError as exc:  # belt-and-braces; the isascii/isdigit guard above
            raise RelayTransportError("malformed Content-Length") from exc

    if body_length > max_body_bytes:
        raise RelayTransportError("request body exceeds max_body_bytes")

    # F10 (bead .56, LOW, doc-only): any bytes in `rest` beyond
    # `body_length` are dropped here, never buffered/replayed. Safe for the
    # v0 one-request-per-connection model (every response sets
    # `Connection: close`, so there is never a next request to misparse on
    # this connection) — revisit only if HTTP/1.1 pipelining is ever added.
    body = bytearray(rest[:body_length])
    while len(body) < body_length:
        remaining = body_length - len(body)
        chunk = await reader.read(min(_BODY_READ_CHUNK, remaining))
        if not chunk:
            raise RelayTransportError("connection closed before request body completed")
        body.extend(chunk)

    return RelayRequest(method=method, path=path, headers=headers, body=bytes(body))


# --------------------------------------------------------------------------- #
# make_urllib_forwarder — the production upstream transport                   #
# --------------------------------------------------------------------------- #


def make_urllib_forwarder(
    *,
    base_url: str,
    timeout: float = 60.0,
    chunk_size: int = 65536,
    opener: Any = _NO_REDIRECT_OPENER,
) -> Forwarder:
    """Build the production forwarder: a blocking ``urllib.request`` call to
    ``base_url + upstream.request.path`` run on a dedicated worker thread,
    bridged to the async caller via a BOUNDED queue (bead31 amendment §D) —
    a stalled/slow-draining consumer blocks the worker thread rather than
    letting this process buffer the whole upstream response in memory.

    Uses ``opener`` (default :data:`_NO_REDIRECT_OPENER`, reusing
    ``critic_client._NoRedirectHandler``) so a 3xx from upstream is refused
    rather than followed — the real provider key rides this request and must
    never be re-sent to a redirect target. ``opener`` is injectable so tests
    can force a redirect (or any other failure) deterministically.

    ANY urllib/connection/TLS/timeout/redirect failure is mapped to
    :class:`UpstreamError` carrying only the failing exception's class name
    — never the raw exception's message/args (which could echo request
    context) and never the real key.

    If the async consumer stops iterating the returned generator early
    (e.g. the worker connection reset mid-stream), the generator's
    ``finally`` unblocks the bridge so the background thread does not block
    forever holding an open upstream socket: it sets a stop flag and
    releases the bridge semaphore, and the worker thread — checked right
    after each semaphore acquire — closes the response and exits.

    **Netloc pinning (F1, bead .55 — the load-bearing guard, defense-in-
    depth even if a malformed path somehow slips past
    :func:`_is_origin_form_path`).** ``base_url`` is parsed ONCE, here, at
    forwarder-build time, to capture the expected netloc (host[:port]). The
    real-key-bearing request is built with ``url = base_url + path`` exactly
    as before (``path`` still drives routing for legitimate origin-form
    paths), but ``_worker`` re-parses the resulting ``url`` and compares its
    netloc against the pinned expected netloc BEFORE ever calling
    ``opener.open``. A worker-supplied path that manages to shift the netloc
    (host suffix hijack, port change, userinfo->host) is refused: no
    connection is ever opened, and the failure surfaces as the usual
    :class:`UpstreamError` fail-closed path.
    """
    expected_netloc = urlsplit(base_url).netloc

    async def forwarder(upstream: UpstreamRequest) -> tuple[UpstreamHead, AsyncIterator[bytes]]:
        loop = asyncio.get_running_loop()
        head_future: asyncio.Future = loop.create_future()
        queue: asyncio.Queue = asyncio.Queue(maxsize=_BRIDGE_QUEUE_SIZE)
        sem = threading.Semaphore(_BRIDGE_QUEUE_SIZE)
        stop_event = threading.Event()

        def _set_head_result(head: UpstreamHead) -> None:
            if not head_future.done():
                head_future.set_result(head)

        def _set_head_exception(exc: BaseException) -> None:
            if not head_future.done():
                head_future.set_exception(exc)

        def _worker() -> None:
            url = base_url.rstrip("/") + upstream.request.path

            # F1 layer 2/3 (bead .55): pin the connection netloc. Never let
            # opener.open see a URL whose netloc differs from the one
            # base_url was built from — this is the load-bearing guard even
            # if a malformed path somehow slipped past the parser's
            # origin-form check. Fail closed: no connection is opened.
            if urlsplit(url).netloc != expected_netloc:
                err = UpstreamError("upstream request failed: netloc mismatch")
                try:
                    loop.call_soon_threadsafe(_set_head_exception, err)
                except RuntimeError:
                    pass
                return

            try:
                # `method=` is passed explicitly so `Request.get_method()`
                # always honours the real HTTP method regardless of `data`
                # — but urllib still treats any non-None `data` (even
                # `b""`) as "this request has a body" (adds
                # Content-Length: 0, sends a body frame). Pass `None` for a
                # genuinely empty body so a GET is sent exactly as a GET,
                # not as a bodyless-but-body-framed request.
                body = upstream.request.body if upstream.request.body else None
                req = urllib.request.Request(
                    url,
                    data=body,
                    headers=dict(upstream.request.headers),
                    method=upstream.request.method,
                )
                resp = opener.open(req, timeout=timeout)
            except Exception as exc:  # noqa: BLE001 - fail closed, never leak exc details
                err = UpstreamError(f"upstream request failed: {exc.__class__.__name__}")
                try:
                    loop.call_soon_threadsafe(_set_head_exception, err)
                except RuntimeError:
                    pass
                return

            try:
                head = UpstreamHead(status=resp.status, headers=list(resp.headers.items()))
            except Exception as exc:  # noqa: BLE001
                err = UpstreamError(
                    f"upstream response inspection failed: {exc.__class__.__name__}"
                )
                try:
                    loop.call_soon_threadsafe(_set_head_exception, err)
                except RuntimeError:
                    pass
                try:
                    resp.close()
                except Exception:  # noqa: BLE001
                    pass
                return

            try:
                loop.call_soon_threadsafe(_set_head_result, head)
            except RuntimeError:
                try:
                    resp.close()
                except Exception:  # noqa: BLE001
                    pass
                return

            try:
                while True:
                    if stop_event.is_set():
                        break
                    sem.acquire()
                    if stop_event.is_set():
                        sem.release()
                        break
                    try:
                        chunk = resp.read(chunk_size)
                    except Exception as exc:  # noqa: BLE001
                        err = UpstreamError(f"upstream read failed: {exc.__class__.__name__}")
                        try:
                            loop.call_soon_threadsafe(queue.put_nowait, ("error", err))
                        except RuntimeError:
                            pass
                        break
                    if not chunk:
                        try:
                            loop.call_soon_threadsafe(queue.put_nowait, ("end", None))
                        except RuntimeError:
                            pass
                        break
                    try:
                        loop.call_soon_threadsafe(queue.put_nowait, ("chunk", chunk))
                    except RuntimeError:
                        break
            finally:
                try:
                    resp.close()
                except Exception:  # noqa: BLE001
                    pass

        thread = threading.Thread(
            target=_worker, name="relay-upstream-forwarder", daemon=True
        )
        thread.start()

        try:
            head = await head_future
        except BaseException:
            # §9 fix K (bead .56 round 3 — the one genuinely-new bug from
            # the round-2 fix-C restructure): a cancellation/timeout HERE
            # (e.g. the stream_deadline firing BEFORE the head ever
            # arrives) means this coroutine never returns a generator, so
            # `_handle_connection` has no `chunks` to `aclose()` — its
            # `finally` (which normally sets `stop_event`/releases `sem` via
            # `_gen`'s own `finally`) never runs either. Without this,
            # `_worker` (already started) later fills the bounded bridge
            # queue and blocks FOREVER on `sem.acquire()` (no consumer ever
            # drains the queue), holding the real-key-bearing upstream
            # socket open indefinitely. Setting `stop_event` here makes
            # `_worker` break out at the top of its read loop (checked
            # right after each acquire, and before the first `resp.read`
            # once the head is set); releasing `sem` unblocks it
            # immediately if it is already parked on `acquire()`.
            # `threading.Semaphore` is NOT bounded, so an extra `release()`
            # here (on top of whatever `_gen`'s `finally` would have done,
            # which never runs on this path) is safe.
            stop_event.set()
            sem.release()
            raise

        async def _gen() -> AsyncIterator[bytes]:
            try:
                while True:
                    kind, payload = await queue.get()
                    sem.release()
                    if kind == "chunk":
                        yield payload
                    elif kind == "end":
                        return
                    else:
                        raise payload
            finally:
                # Unblock an abandoned worker thread (early consumer abort,
                # e.g. worker connection reset mid-stream) so it never holds
                # an open upstream socket + real key in memory forever.
                stop_event.set()
                sem.release()

        return head, _gen()

    return forwarder


# --------------------------------------------------------------------------- #
# serve_relay — the per-connection pipeline                                   #
# --------------------------------------------------------------------------- #


def _lookup_header(headers: list[tuple[str, str]], name: str) -> str | None:
    target = name.lower()
    for key, value in headers:
        if key.lower() == target:
            return value
    return None


def _build_response_head(status: int, headers: list[tuple[str, str]]) -> bytes:
    """Build the worker-facing response status line + headers (bead31
    amendment §A): strip :data:`HOP_BY_HOP`, add ``Connection: close``, and
    emit NO ``Content-Length``/``Transfer-Encoding`` of our own — the body
    that follows is close-delimited by design."""
    phrase = http.client.responses.get(status, "")
    lines = [f"HTTP/1.1 {status} {phrase}".rstrip()]
    for name, value in headers:
        if name.lower() in HOP_BY_HOP:
            continue
        lines.append(f"{name}: {value}")
    lines.append("Connection: close")
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")


async def _write_deny_response(writer: asyncio.StreamWriter, status: int) -> None:
    phrase = http.client.responses.get(status, "Error")
    head = f"HTTP/1.1 {status} {phrase}\r\nConnection: close\r\n\r\n".encode("latin-1")
    try:
        writer.write(head)
        await writer.drain()
    except (OSError, ConnectionError):
        pass


def _log(log_path: str | Path, entry: dict[str, Any]) -> None:
    """Append one JSONL line. Mirrors ``egress_proxy._log`` exactly — small
    individual appends stay atomic on POSIX. NEVER pass headers, bodies, the
    capability token, or the real key into ``entry`` — only correlation
    metadata (dispatch id, decision, reason, status)."""
    line = json.dumps(entry, sort_keys=True)
    with open(log_path, "a") as fh:
        fh.write(line + "\n")


async def _handle_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    relay: CredentialRelay,
    forwarder: Forwarder,
    log_path: str | Path,
    dispatch_id: str | None,
    stream_deadline: float,
) -> None:
    decision = "error"
    reason = "handler-exception"
    status = 500
    head_written = False
    # bead .25: the worker-controlled anthropic-beta value is passed through to
    # upstream (claude-code needs it; see bead25-build-spec §9). Log it so the
    # worker's beta selection is AUDITED, not invisible — the observability half
    # of the decision-5b posture (server-side-tool betas would bypass the cage).
    beta: str | None = None
    try:
        try:
            request = await read_relay_request(
                reader,
                max_head_bytes=_DEFAULT_MAX_HEAD_BYTES,
                max_body_bytes=_DEFAULT_MAX_BODY_BYTES,
                timeout=_DEFAULT_REQUEST_TIMEOUT,
            )
        except RelayTransportError:
            status = 400
            decision = "deny"
            reason = "malformed-request"
            await _write_deny_response(writer, status)
            return

        beta = _find_header(request.headers, "anthropic-beta")

        prepared = prepare_upstream(relay, request)
        if isinstance(prepared, DenyResponse):
            status = prepared.status
            decision = "deny"
            reason = prepared.reason
            await _write_deny_response(writer, status)
            return

        # RESERVE the call slot NOW — atomically with authorize (no `await`
        # between prepare_upstream and this charge, so the single-threaded
        # event loop cannot interleave another connection in between). This
        # closes the max_calls TOCTOU: a compromised worker (threat model)
        # opening N concurrent connections would otherwise all pass
        # authorize() while used_calls is still 0 (no charge lands until the
        # stream completes) and all N would forward with the real key,
        # defeating the SPEC §4 max_calls leash. Reserving up front means
        # only max_calls forwards ever go out; output tokens are reconciled
        # (calls=0) after the stream. Also subsumes §C: a call that fails
        # upstream stays charged (no unmetered retries), because the reserve
        # already landed before the forward. The .12 sync `handle()` path
        # needs no reserve — it is single-shot with no concurrency window.
        try:
            relay._store.charge(prepared.token, output_tokens=0, calls=1)
        except CapabilityDenied as exc:
            # Only reachable if the token went unknown/revoked in the sync
            # window (not possible today, but fail closed regardless).
            status = 401
            decision = "deny"
            reason = exc.reason
            await _write_deny_response(writer, status)
            return

        # F5 round 2 (bead .56, §8 fix C / review-56-codex.md MAJOR-3): the
        # `stream_deadline` must bound `await forwarder(prepared)` too, not
        # only head-write + stream iteration — otherwise a forwarder that
        # stalls before ever returning a head can hold the real-key-bearing
        # request (and the sole in-flight slot) open indefinitely, past any
        # configured deadline. `head`/`chunks` start unset so the `finally`
        # below can tell whether a forwarder result ever existed.
        head = None
        chunks = None
        stream_error = False
        try:
            async with asyncio.timeout(stream_deadline):
                head, chunks = await forwarder(prepared)

                # §9 fix I (bead .56 round 3): classify the media type by
                # its TOKEN (the part before any `;` parameter), not a
                # substring match — `text/plain; note=text/event-stream`
                # must NOT be metered as SSE just because the substring
                # "text/event-stream" appears somewhere in the header.
                content_type = _lookup_header(head.headers, "content-type") or ""
                mtype = content_type.split(";", 1)[0].strip().lower()
                is_sse = mtype == "text/event-stream"
                is_json = mtype == "application/json"

                # §9 fix I: collect ALL Content-Encoding occurrences (a
                # forwarder — or a lookup helper that only returns the
                # FIRST match — could let a later, real encoding hide
                # behind an earlier "identity"). A present-but-empty value
                # counts as non-identity too.
                encodings = [
                    v.strip().lower() for (k, v) in head.headers if k.lower() == "content-encoding"
                ]
                # §8 fix B: an unsupported/compressed 2xx representation is
                # never trusted for metering — this relay never decompresses,
                # so a non-identity Content-Encoding means the parser (SSE or
                # JSON) would be fed compressed bytes and must not be
                # trusted even if it happens to "parse". Gated explicitly
                # here (rather than relying only on the parse failing) for
                # defense-in-depth + legibility.
                unsupported_encoding = any(e != "identity" for e in encodings)

                # F4 (bead .56): branch metering strategy on content-type
                # BEFORE the stream loop, and never retain the whole body.
                # SSE is metered by a bounded, incremental, cross-chunk
                # `_SseUsageMeter`; JSON is accumulated into a bounded buffer
                # (`json_tee`), and once it would exceed
                # `_MAX_JSON_METER_BYTES`, accumulation stops
                # (`json_overflow = True`) while every byte still gets
                # forwarded to the worker. §9 fix I: any OTHER (or absent)
                # media type is fed to neither accumulator — nothing is
                # retained for it, and it is unbudgetable below.
                meter = _SseUsageMeter() if is_sse else None
                json_tee = bytearray()
                json_overflow = False

                writer.write(_build_response_head(head.status, head.headers))
                head_written = True
                await writer.drain()
                async for chunk in chunks:
                    writer.write(chunk)
                    await writer.drain()
                    if is_sse:
                        meter.feed(chunk)
                    elif is_json and not json_overflow:
                        if len(json_tee) + len(chunk) > _MAX_JSON_METER_BYTES:
                            json_overflow = True
                            json_tee = bytearray()
                        else:
                            json_tee.extend(chunk)
        except Exception:  # noqa: BLE001 - forwarder/stream failure or deadline, fail closed
            if not head_written:
                # A timeout/failure BEFORE the head was ever written toward
                # the worker (forwarder stall/failure) — the call slot was
                # already reserved (calls=1) above; do NOT double-charge.
                # Nothing was generated, so this is NOT unbudgetable either
                # (§8 fix C: distinguished from a post-head failure via
                # `head_written`).
                status = 502
                decision = "error"
                reason = "upstream-error"
                # `finally` below closes `chunks` if the forwarder somehow
                # returned one before this branch was reached (it did not
                # here, but kept uniform for any future forwarder shape).
                await _write_deny_response(writer, status)
                return
            # A timeout/failure AFTER the head was already sent (mid-stream
            # truncation, or the deadline firing during the stream loop):
            # this can no longer become a 502. Stop writing and close —
            # close-delimited framing means the worker sees a truncated
            # body at EOF. On a 2xx response this becomes unbudgetable
            # below (F4 backstop) rather than the old silent under-meter.
            stream_error = True
        finally:
            # Deterministically (not just at GC/loop-teardown time) unblock
            # an abandoned upstream-reading worker thread on ANY early exit
            # from this loop (worker disconnect, write error, deadline,
            # etc.) — the forwarder generator's own `finally` sets the stop
            # flag and releases the bridge semaphore so the thread closes
            # its response and exits instead of blocking on the semaphore
            # forever while holding an open upstream socket + the real key.
            if chunks is not None:
                aclose = getattr(chunks, "aclose", None)
                if aclose is not None:
                    try:
                        await aclose()
                    except Exception:  # noqa: BLE001
                        pass
            if head_written:
                try:
                    writer.write_eof()
                except (OSError, RuntimeError):
                    pass

        is_success = 200 <= head.status < 300
        if not is_success:
            # Non-2xx: upstream generated nothing chargeable — the call slot
            # was already reserved (calls=1); do NOT penalize the leash for
            # an upstream error/redirect/etc.
            status = head.status
            decision = "allow"
            reason = "stream-error" if stream_error else "forwarded"
        else:
            # §9 fix J (bead .56 round 3): a bounded parser can still raise
            # AFTER a 2xx has already been delivered to the worker (e.g. a
            # >4300-digit int hitting Python's int-string-conversion limit
            # in `json.loads`/int coercion). That must fail closed
            # (unbudgetable), never escape to the generic handler-exception
            # `except` below (which would synthesize a 500 and, worse,
            # leave the capability's usage un-reconciled/live).
            try:
                if unsupported_encoding:
                    # §8 fix B: never trust a parsed usage value from a
                    # compressed/unrecognized 2xx representation.
                    output_tokens = None
                elif is_sse:
                    output_tokens = meter.finalize()
                elif is_json:
                    if json_overflow:
                        output_tokens = None
                    else:
                        usage = extract_usage(
                            RelayResponse(status=head.status, headers={}, body=bytes(json_tee))
                        )
                        output_tokens = (
                            usage.get("output_tokens") if isinstance(usage, dict) else None
                        )
                        # §8 fix A (JSON path): require a POSITIVELY-
                        # determined valid non-negative int;
                        # absent/non-int/bool/negative is NOT a measurement
                        # -> unmeterable (None), never a silent
                        # default-to-0 under-charge.
                        if (
                            not isinstance(output_tokens, int)
                            or isinstance(output_tokens, bool)
                            or output_tokens < 0
                        ):
                            output_tokens = None
                else:
                    # §9 fix I: any media type other than the two supported
                    # representations (text/event-stream, application/json)
                    # is unmeterable — nothing was accumulated for it, and
                    # this relay must never guess.
                    output_tokens = None
            except Exception:  # noqa: BLE001 - bounded-parser failure after a
                # delivered 200 fails closed (unbudgetable), never a 500 that
                # would leave the capability's usage un-reconciled/live.
                output_tokens = None

            if stream_error:
                # A 2xx cut short (worker abort, F5 deadline, or a benign
                # upstream blip) is unbudgetable — closes the
                # truncate-to-avoid-charge bypass (a compromised worker
                # aborting right before the final usage delta must not
                # dodge metering).
                output_tokens = None

            if output_tokens is None:
                try:
                    relay._store.charge_unbudgetable(prepared.token)
                except CapabilityDenied:
                    pass
                reason = "unbudgetable"
            else:
                # Reconcile output tokens only — the call slot was reserved
                # (calls=1) before the forward, so this must NOT re-count
                # the call (calls=0).
                try:
                    relay._store.charge(prepared.token, output_tokens=output_tokens, calls=0)
                except CapabilityDenied:
                    pass
                reason = "stream-error" if stream_error else "forwarded"

            status = head.status
            decision = "allow"
    except Exception:  # noqa: BLE001 - a handler bug must never hang or leak the real key
        decision = "error"
        reason = "handler-exception"
        status = 500
        # A response head (e.g. a 200) may already be in flight toward the
        # worker by the time an unexpected bug fires (mid-stream). Writing a
        # second status line on top of that would corrupt the framing the
        # worker is already parsing — only attempt a synthesized deny if no
        # head has gone out yet; otherwise just let `finally` close.
        if not head_written:
            try:
                await _write_deny_response(writer, 500)
            except Exception:  # noqa: BLE001
                pass
    finally:
        _log(
            log_path,
            {
                "ts": time.time(),
                "dispatch_id": dispatch_id,
                "decision": decision,
                "reason": reason,
                "status": status,
                "anthropic_beta": beta,
            },
        )
        try:
            writer.close()
        except Exception:  # noqa: BLE001
            pass


def _validate_serve_config(max_in_flight: int, stream_deadline: float) -> None:
    """Reject unsafe/invalid config BEFORE any socket I/O (bead .56 round 2,
    §8 fix E). ``max_in_flight`` must be exactly ``1`` in v0 — the
    docstrings on :func:`serve_relay`/:func:`start_relay` already say any
    other value reopens F6 (output-token overshoot); an API that silently
    accepts a known-unsafe value is unsafe-by-configuration. A non-``int``
    or ``bool`` is rejected too (``bool`` is an ``int`` subclass in Python
    but is never a meaningful concurrency count here).
    ``stream_deadline`` must be a positive, finite number — ``<= 0``,
    ``NaN``, or infinite all defeat the wall-clock bound F5 exists to
    provide."""
    if isinstance(max_in_flight, bool) or not isinstance(max_in_flight, int):
        raise RelayError(f"max_in_flight must be an int (got {max_in_flight!r})")
    if max_in_flight != 1:
        raise RelayError(
            f"max_in_flight must be exactly 1 in v0 (got {max_in_flight!r}) — "
            "raising it reopens F6 output-token overshoot; see serve_relay's docstring"
        )
    if isinstance(stream_deadline, bool) or not isinstance(stream_deadline, (int, float)):
        raise RelayError(
            f"stream_deadline must be a positive finite number (got {stream_deadline!r})"
        )
    if math.isnan(stream_deadline) or math.isinf(stream_deadline) or stream_deadline <= 0:
        raise RelayError(
            f"stream_deadline must be a positive finite number (got {stream_deadline!r})"
        )


async def serve_relay(
    socket_path: str | Path,
    relay: CredentialRelay,
    *,
    forwarder: Forwarder,
    log_path: str | Path,
    dispatch_id: str | None = None,
    max_in_flight: int = 1,
    stream_deadline: float = _DEFAULT_STREAM_DEADLINE,
) -> asyncio.Server:
    """Start the credential-relay HTTP server listening on a unix domain
    socket. Mirrors ``egress_proxy.serve``'s shape: per-connection, read ->
    :func:`prepare_upstream` -> (``DenyResponse``: write the synthesized
    deny, log, done) | (``UpstreamRequest``: ``forwarder`` -> frame + stream
    to the worker, meter usage incrementally/boundedly, ``store.charge`` /
    ``store.charge_unbudgetable``).

    Immediately after the socket binds, ``chmod``s it to ``0o600`` (bead31
    amendment §B — deliberately BETTER than ``egress_proxy``, which relies
    only on process umask + the runtime dir's own permissions).

    **F5 (bead .56) — accept cap + stream deadline.** ``max_in_flight``
    (default 1, the v0-safe single-in-flight model) bounds how many
    connections may be inside :func:`_handle_connection` concurrently: an
    ``asyncio.Semaphore(max_in_flight)`` is created HERE (inside the
    running loop, so it binds to it correctly) and acquired in
    ``_on_connect`` around the ENTIRE handler call — this bounds both
    concurrent request-body buffering (memory) and concurrent forwards.
    ``stream_deadline`` (default :data:`_DEFAULT_STREAM_DEADLINE`) bounds
    the wall-clock time the forward+stream phase may hold an upstream
    socket (and the real key) open; a breach tears the connection down and
    (on an in-flight 2xx) charges the capability unbudgetable rather than
    hanging indefinitely.

    **F6 (bead .56) note — no request-body parsing added.** With
    ``max_in_flight=1``, only one call is ever in flight against a
    capability, so concurrent output-token overshoot cannot occur (the v0
    fix is structural serialization, not up-front reservation). Raising
    ``max_in_flight`` above 1 — or any future true-parallelism (v1) design —
    RE-OPENS the output-token overshoot finding and requires up-front
    output-token reservation (parsed from the worker's declared request,
    e.g. ``max_tokens``) plus a reconcile-down step before it is safe;
    that is explicitly NOT built here.
    """
    _validate_serve_config(max_in_flight, stream_deadline)

    socket_path = Path(socket_path)
    if socket_path.exists():
        socket_path.unlink()

    sem = asyncio.Semaphore(max_in_flight)

    async def _on_connect(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            async with sem:
                await _handle_connection(
                    reader,
                    writer,
                    relay=relay,
                    forwarder=forwarder,
                    log_path=log_path,
                    dispatch_id=dispatch_id,
                    stream_deadline=stream_deadline,
                )
        except Exception:  # noqa: BLE001 - a handler bug must never leak an open connection
            try:
                writer.close()
            except Exception:  # noqa: BLE001
                pass

    server = await asyncio.start_unix_server(
        _on_connect, path=str(socket_path), limit=_STREAM_LIMIT
    )
    os.chmod(socket_path, 0o600)
    return server


# --------------------------------------------------------------------------- #
# Per-dispatch lifecycle (mirrors egress.setup_dispatch_egress/EgressHandle)   #
# --------------------------------------------------------------------------- #


@dataclass
class RelayHandle:
    """A running per-dispatch relay server.

    ``socket_path``/``log_path`` are what the caller feeds onward (the
    worker's ``127.0.0.1:PORT -> unix socket`` shim mount is a ``.25``/``.32``
    concern, not built here)."""

    dispatch_id: str
    socket_path: Path
    log_path: Path
    _loop: asyncio.AbstractEventLoop = field(repr=False)
    _thread: threading.Thread = field(repr=False)
    _stopped: bool = False

    def stop(self) -> None:
        """Stop serving and remove the socket file (idempotent).

        After this returns, any client ``connect()`` to ``socket_path``
        fails outright (no listener, no file) — the fail-closed
        credential-relay-down guarantee: a dispatch whose relay is gone
        never falls back to unfiltered/unmetered access.
        """
        if self._stopped:
            return
        self._stopped = True

        def _shutdown() -> None:
            self._loop.stop()

        try:
            self._loop.call_soon_threadsafe(_shutdown)
        except RuntimeError:
            pass  # loop already stopped/closed
        self._thread.join(timeout=_STOP_TIMEOUT_SECONDS)

        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass


def start_relay(
    dispatch_id: str,
    socket_path_or_runtime_dir: str | Path,
    relay: CredentialRelay,
    *,
    forwarder: Forwarder,
    log_path: str | Path,
    max_in_flight: int = 1,
    stream_deadline: float = _DEFAULT_STREAM_DEADLINE,
) -> RelayHandle:
    """Start the credential relay for one dispatch on a dedicated
    background event-loop thread and return its handle once actually
    listening (mirrors ``egress.setup_dispatch_egress``'s lifecycle
    exactly: block-until-listening, raise on start failure, never hand back
    a handle whose socket doesn't exist yet).

    ``socket_path_or_runtime_dir``: if it ends in ``.sock`` it is used as
    the exact socket path (its parent directory is created/chmod'd); any
    other path is treated as a per-dispatch runtime directory, created
    ``0o700`` (bead31 amendment §B), under which
    ``relay-<dispatch_id>.sock`` is created.

    ``max_in_flight``/``stream_deadline`` (bead .56, F5) are passed straight
    through to :func:`serve_relay` — exposed here so ``.25``/``.32``
    wiring can tune them without editing this module; the defaults
    (1 / :data:`_DEFAULT_STREAM_DEADLINE`) preserve today's behaviour.

    Bead .56 round 2 (§8 fix E): config is validated up front, before any
    directory creation or socket I/O — a bad ``max_in_flight``/
    ``stream_deadline`` raises :class:`stigmergy.relay.RelayError`
    immediately rather than after standing up a runtime directory/thread.
    """
    _validate_serve_config(max_in_flight, stream_deadline)

    given = Path(socket_path_or_runtime_dir)
    if given.suffix == ".sock":
        socket_path = given
        runtime_dir = given.parent
    else:
        runtime_dir = given
        socket_path = runtime_dir / f"relay-{dispatch_id}.sock"

    runtime_dir.mkdir(parents=True, exist_ok=True)
    # F8 (bead .56, LOW, doc-only): this chmod is the LOAD-BEARING half of
    # the socket-permission guarantee. The socket itself is chmod'd 0o600
    # right after bind (see serve_relay) restricting *open*, but a unix
    # socket's connect() is also gated by directory-traversal permission on
    # every ancestor directory — without this 0o700 here, another local
    # user could still reach the socket via a permissive parent directory.
    # Together, 0o700 (dir) + 0o600 (socket) are the fail-closed connector
    # restriction; neither alone is sufficient.
    os.chmod(runtime_dir, 0o700)

    ready = threading.Event()
    failure: dict[str, BaseException] = {}
    loop_box: dict[str, asyncio.AbstractEventLoop] = {}

    def _run() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop_box["loop"] = loop
        try:
            server = loop.run_until_complete(
                serve_relay(
                    socket_path,
                    relay,
                    forwarder=forwarder,
                    log_path=log_path,
                    dispatch_id=dispatch_id,
                    max_in_flight=max_in_flight,
                    stream_deadline=stream_deadline,
                )
            )
        except BaseException as exc:  # noqa: BLE001 - relay to the starting thread
            failure["error"] = exc
            ready.set()
            loop.close()
            return

        ready.set()
        try:
            loop.run_forever()
        finally:
            server.close()
            loop.run_until_complete(server.wait_closed())
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    thread = threading.Thread(target=_run, name=f"relay-transport-{dispatch_id}", daemon=True)
    thread.start()

    if not ready.wait(timeout=_START_TIMEOUT_SECONDS):
        raise RelayTransportError(
            f"relay for dispatch {dispatch_id!r} did not start within "
            f"{_START_TIMEOUT_SECONDS}s"
        )
    if "error" in failure:
        raise RelayTransportError(
            f"relay for dispatch {dispatch_id!r} failed to start"
        ) from failure["error"]

    return RelayHandle(
        dispatch_id=dispatch_id,
        socket_path=socket_path,
        log_path=Path(log_path),
        _loop=loop_box["loop"],
        _thread=thread,
    )
