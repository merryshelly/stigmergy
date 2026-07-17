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
  routes through :func:`sse_extract_usage`; anything else routes through
  ``relay.extract_usage`` on the accumulated (bounded-tee) body.
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
- *Bounded memory (§D)*: the urllib-reading worker thread is bridged to the
  async caller via a BOUNDED queue (natural backpressure — a stalled worker
  cannot make this process buffer the whole upstream response), and usage
  parsing tees into a capped buffer (past the cap: keep forwarding every
  byte to the worker, just stop trying to meter it precisely — the
  documented v0 under-meter gap, see below).
- *Socket perms (§B, deliberately BETTER than ``egress_proxy``, which does
  no explicit chmod)*: the per-dispatch runtime dir is created ``0o700``
  and the socket is ``chmod``'d ``0o600`` immediately after bind.

**Known gaps (logged, not hidden — bead31-build-spec.md):**
- A truncated/errored stream under-meters usage (best-effort tee parse) —
  fail-*open* for the USD/token leash. Acceptable at v0; ``max_calls`` still
  increments so a compromised worker cannot get unlimited free retries.
- Single-threaded relay daemon (one worker/dispatch at a time, v0).
- No request-body chunked-transfer-encoding support (claude-code sends
  ``Content-Length``); rejected explicitly, never silently mishandled.
"""

from __future__ import annotations

import asyncio
import http.client
import json
import os
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
    RelayRequest,
    RelayResponse,
    UpstreamRequest,
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

# Bound on how much of the upstream response body this module will tee for
# usage parsing (bead31 amendment §D) — past this cap, chunks are still
# forwarded to the worker in full, just no longer accumulated for parsing.
# A response whose final cumulative usage field lands past this cap
# under-meters (the documented, logged v0 gap); it never truncates the
# worker's actual answer.
_MAX_TEE_BYTES = 8 * 1024 * 1024

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
# sse_extract_usage — PURE, never raises (bead31 build spec cases 12-18)       #
# --------------------------------------------------------------------------- #


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
        data_lines: list[str] = []
        for line in block.split("\n"):
            if not line or line.startswith(":"):
                continue  # blank / keep-alive comment
            if line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
            # `event:` and any other field lines are informational only —
            # this parser keys off the parsed JSON shape, not the event name.

        if not data_lines:
            continue
        data_str = "\n".join(data_lines)
        if data_str == "[DONE]":
            continue

        try:
            parsed = json.loads(data_str)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(parsed, dict):
            continue

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
            continue

        usage = parsed.get("usage")
        if isinstance(usage, dict):
            ot = usage.get("output_tokens")
            if isinstance(ot, int) and not isinstance(ot, bool):
                output_tokens = ot  # cumulative -> last wins, always overwrite
            it = usage.get("input_tokens")
            if isinstance(it, int) and not isinstance(it, bool):
                input_tokens = it

    result: dict = {}
    if input_tokens is not None:
        result["input_tokens"] = input_tokens
    if output_tokens is not None:
        result["output_tokens"] = output_tokens
    return result


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
        name = name.strip()
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

        head = await head_future

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
) -> None:
    decision = "error"
    reason = "handler-exception"
    status = 500
    head_written = False
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

        try:
            head, chunks = await forwarder(prepared)
        except Exception:  # noqa: BLE001 - any forwarder failure -> 502, fail closed
            status = 502
            decision = "error"
            reason = "upstream-error"
            # The call slot was already reserved (calls=1) above — do NOT
            # double-charge here; §C's "no unmetered retries" is preserved
            # by the reserve, not by a charge on this path.
            await _write_deny_response(writer, status)
            return

        content_type = _lookup_header(head.headers, "content-type") or ""

        tee = bytearray()
        stream_error = False
        try:
            writer.write(_build_response_head(head.status, head.headers))
            head_written = True
            await writer.drain()
            async for chunk in chunks:
                writer.write(chunk)
                await writer.drain()
                if len(tee) < _MAX_TEE_BYTES:
                    room = _MAX_TEE_BYTES - len(tee)
                    tee.extend(chunk[:room])
        except Exception:  # noqa: BLE001 - a mid-stream failure truncates, never crashes
            # The 200 (or whatever) head was already sent; this can no
            # longer become a 502. Stop writing and close — close-delimited
            # framing means the worker sees a truncated body at EOF (the
            # documented under-meter gap; never silently corrupt further).
            stream_error = True
        finally:
            # Deterministically (not just at GC/loop-teardown time) unblock
            # an abandoned upstream-reading worker thread on ANY early exit
            # from this loop (worker disconnect, write error, etc.) — the
            # forwarder generator's own `finally` sets the stop flag and
            # releases the bridge semaphore so the thread closes its
            # response and exits instead of blocking on the semaphore
            # forever while holding an open upstream socket + the real key.
            aclose = getattr(chunks, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:  # noqa: BLE001
                    pass
            try:
                writer.write_eof()
            except (OSError, RuntimeError):
                pass

        if "text/event-stream" in content_type.lower():
            usage = sse_extract_usage(bytes(tee))
        else:
            usage = extract_usage(RelayResponse(status=head.status, headers={}, body=bytes(tee)))
        output_tokens = usage.get("output_tokens", 0) if isinstance(usage, dict) else 0
        if not isinstance(output_tokens, int) or isinstance(output_tokens, bool) or (
            output_tokens < 0
        ):
            output_tokens = 0
        # Reconcile output tokens only — the call slot was reserved (calls=1)
        # before the forward, so this must NOT re-count the call (calls=0).
        try:
            relay._store.charge(prepared.token, output_tokens=output_tokens, calls=0)
        except CapabilityDenied:
            pass

        status = head.status
        decision = "allow"
        reason = "stream-error" if stream_error else "forwarded"
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
            },
        )
        try:
            writer.close()
        except Exception:  # noqa: BLE001
            pass


async def serve_relay(
    socket_path: str | Path,
    relay: CredentialRelay,
    *,
    forwarder: Forwarder,
    log_path: str | Path,
    dispatch_id: str | None = None,
) -> asyncio.Server:
    """Start the credential-relay HTTP server listening on a unix domain
    socket. Mirrors ``egress_proxy.serve``'s shape: per-connection, read ->
    :func:`prepare_upstream` -> (``DenyResponse``: write the synthesized
    deny, log, done) | (``UpstreamRequest``: ``forwarder`` -> frame + stream
    to the worker, tee for usage parsing, ``store.charge``).

    Immediately after the socket binds, ``chmod``s it to ``0o600`` (bead31
    amendment §B — deliberately BETTER than ``egress_proxy``, which relies
    only on process umask + the runtime dir's own permissions).
    """
    socket_path = Path(socket_path)
    if socket_path.exists():
        socket_path.unlink()

    async def _on_connect(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await _handle_connection(
                reader,
                writer,
                relay=relay,
                forwarder=forwarder,
                log_path=log_path,
                dispatch_id=dispatch_id,
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
    """
    given = Path(socket_path_or_runtime_dir)
    if given.suffix == ".sock":
        socket_path = given
        runtime_dir = given.parent
    else:
        runtime_dir = given
        socket_path = runtime_dir / f"relay-{dispatch_id}.sock"

    runtime_dir.mkdir(parents=True, exist_ok=True)
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
