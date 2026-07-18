"""Adversarial tests for the LIVE credential-relay HTTP transport (bead .31;
SPEC.md §4 credentials "out of the worker namespace entirely", §10 AC4).

Authored by the orchestrator, not the implementor — this is the single
highest-security component in the system (network-facing credential handling),
so the case-list is the orchestrator's to own (bead31-build-spec.md, frozen +
2026-07-16 amendment). The implementation in `stigmergy.relay_transport` (and
the additive `prepare_upstream` seam in `stigmergy.relay`) must satisfy every
assertion here without weakening one to force a pass.

Threat model (SPEC §4): every worker is assumed compromised. The worker holds
ONLY a per-dispatch capability token; the real provider key lives only inside
the relay process and must never reach the worker via response bytes, the JSONL
relay log, or any exception message. A denied capability must never cause the
real key to be fetched (`key_provider` uncalled) or forwarded. DUMMY keys only.

Scope split (mirrors .11/.12 mechanism-vs-live): the wired
worker->shim->relay->api.anthropic.com path with a real `op` key + real TLS is
the .32/.25 live proof (case 30). Here the forwarder and key provider are
injected fakes — the transport, framing, SSE metering, auth/inject seam, and
per-dispatch lifecycle are exercised deterministically and offline.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
from pathlib import Path

import pytest

from stigmergy.relay import (
    CapabilityStore,
    CredentialRelay,
    DenyResponse,
    RelayRequest,
    RelayResponse,
    UpstreamRequest,
    extract_usage,
    prepare_upstream,
)
from stigmergy.relay_transport import (
    HOP_BY_HOP,
    RelayTransportError,
    UpstreamError,
    UpstreamHead,
    make_urllib_forwarder,
    read_relay_request,
    serve_relay,
    sse_extract_usage,
    start_relay,
)

# A stand-in for a real provider key — shaped so a leak is unmistakable and so
# that if it EVER lands in worker-facing bytes or a log the assertion screams.
DUMMY_REAL_KEY = "sk-ant-LEAKED-REAL-KEY-must-never-reach-worker"
CAP_HEADER = "x-api-key"


# --------------------------------------------------------------------------- #
# Fakes (mirror test_relay.py's injected-callable discipline)                  #
# --------------------------------------------------------------------------- #
class CountingKeyProvider:
    """Returns the dummy real key and counts fetches — a deny path must fetch
    ZERO times."""

    def __init__(self, key: str = DUMMY_REAL_KEY) -> None:
        self._key = key
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        return self._key


class NeverForwarder:
    """A forwarder that fails the test if it is ever invoked (deny paths must
    never reach upstream)."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, upstream: UpstreamRequest):  # noqa: ANN201
        self.calls += 1
        raise AssertionError("forwarder invoked on a path that must never forward")


class FakeForwarder:
    """Captures the UpstreamRequest and streams a canned response back as
    (UpstreamHead, async-iterator-of-chunks). ``delay`` inserts an await between
    chunks so incremental delivery / backpressure is observable. ``error``, if
    set, is raised instead of returning (upstream-failure path)."""

    def __init__(
        self,
        *,
        status: int = 200,
        headers: list[tuple[str, str]] | None = None,
        chunks: list[bytes] | None = None,
        delay: float = 0.0,
        error: Exception | None = None,
    ) -> None:
        self.status = status
        self.headers = headers if headers is not None else [("content-type", "application/json")]
        self.chunks = chunks if chunks is not None else [b"{}"]
        self.delay = delay
        self.error = error
        self.requests: list[UpstreamRequest] = []

    async def __call__(self, upstream: UpstreamRequest):  # noqa: ANN201
        self.requests.append(upstream)
        if self.error is not None:
            raise self.error
        head = UpstreamHead(status=self.status, headers=list(self.headers))

        async def _gen():
            for chunk in self.chunks:
                if self.delay:
                    await asyncio.sleep(self.delay)
                yield chunk

        return head, _gen()


def _relay(store: CapabilityStore, key_provider: CountingKeyProvider) -> CredentialRelay:
    # forwarder here is the .12 sync forwarder; prepare_upstream never calls it,
    # and serve_relay uses the INJECTED async forwarder, not this one.
    return CredentialRelay(
        store=store,
        key_provider=key_provider,
        forwarder=lambda req: (_ for _ in ()).throw(AssertionError("sync forwarder unused")),
        capability_header=CAP_HEADER,
    )


def _build_raw_request(
    *,
    token: str | None,
    method: str = "POST",
    path: str = "/v1/messages",
    body: bytes = b'{"model":"x"}',
    extra_lines: list[str] | None = None,
    content_length: int | None = None,
    include_cl: bool = True,
) -> bytes:
    lines = [f"{method} {path} HTTP/1.1", "host: relay"]
    if token is not None:
        lines.append(f"{CAP_HEADER}: {token}")
    lines.append("content-type: application/json")
    if include_cl:
        cl = content_length if content_length is not None else len(body)
        lines.append(f"content-length: {cl}")
    if extra_lines:
        lines.extend(extra_lines)
    head = "\r\n".join(lines) + "\r\n\r\n"
    return head.encode("latin-1") + body


async def _roundtrip(socket_path: Path, raw: bytes) -> bytes:
    reader, writer = await asyncio.open_unix_connection(path=str(socket_path))
    writer.write(raw)
    await writer.drain()
    data = await reader.read()  # Connection: close -> read to EOF
    writer.close()
    try:
        await writer.wait_closed()
    except OSError:
        pass
    return data


def _split_response(raw: bytes) -> tuple[bytes, bytes]:
    head, _, body = raw.partition(b"\r\n\r\n")
    return head, body


def _reader_with(data: bytes) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    reader.feed_data(data)
    reader.feed_eof()
    return reader


def test_hop_by_hop_set_covers_framing_headers():
    # The exact strip set is part of the contract (amendment §A): these must be
    # removed from the worker-facing response so framing can't be corrupted.
    for name in (
        "connection",
        "keep-alive",
        "transfer-encoding",
        "content-length",
        "te",
        "trailer",
        "upgrade",
        "proxy-authenticate",
        "proxy-authorization",
    ):
        assert name in HOP_BY_HOP


# =========================================================================== #
# prepare_upstream seam (cases 1-5) — the single authorize+inject authority     #
# =========================================================================== #
class TestPrepareUpstreamSeam:
    def test_missing_capability_header_denies_and_never_fetches_key(self):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        relay = _relay(store, kp)
        req = RelayRequest(
            method="POST", path="/v1/messages", headers={"host": "relay"}, body=b"{}"
        )

        result = prepare_upstream(relay, req)

        assert isinstance(result, DenyResponse)
        assert result.status == 401
        assert result.reason == "missing-capability"
        assert kp.calls == 0  # case 1: key never fetched on deny

    def test_unknown_token_denies_401_key_never_fetched(self):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        relay = _relay(store, kp)
        req = RelayRequest(
            method="POST", path="/v1/messages", headers={CAP_HEADER: "bogus"}, body=b"{}"
        )

        result = prepare_upstream(relay, req)

        assert isinstance(result, DenyResponse)
        assert result.status == 401
        assert result.reason == "unknown"
        assert kp.calls == 0

    def test_revoked_token_denies_401(self):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        relay = _relay(store, kp)
        cap = store.mint("d1", max_output_tokens=1000, max_calls=5)
        store.revoke("d1")
        req = RelayRequest(
            method="POST", path="/v1/messages", headers={CAP_HEADER: cap.token}, body=b"{}"
        )

        result = prepare_upstream(relay, req)

        assert isinstance(result, DenyResponse)
        assert result.status == 401
        assert result.reason == "revoked"
        assert kp.calls == 0

    def test_quota_calls_denies_429(self):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        relay = _relay(store, kp)
        cap = store.mint("d1", max_output_tokens=1000, max_calls=1)
        store.charge(cap.token, output_tokens=0, calls=1)  # exhaust calls
        req = RelayRequest(
            method="POST", path="/v1/messages", headers={CAP_HEADER: cap.token}, body=b"{}"
        )

        result = prepare_upstream(relay, req)

        assert isinstance(result, DenyResponse)
        assert result.status == 429  # case 2: quota -> 429
        assert result.reason == "quota-calls"
        assert kp.calls == 0

    def test_quota_tokens_denies_429(self):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        relay = _relay(store, kp)
        cap = store.mint("d1", max_output_tokens=10, max_calls=5)
        store.charge(cap.token, output_tokens=10, calls=0)  # exhaust tokens
        req = RelayRequest(
            method="POST", path="/v1/messages", headers={CAP_HEADER: cap.token}, body=b"{}"
        )

        result = prepare_upstream(relay, req)

        assert isinstance(result, DenyResponse)
        assert result.status == 429
        assert result.reason == "quota-tokens"
        assert kp.calls == 0

    def test_authorized_injects_real_key_and_strips_capability(self):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        relay = _relay(store, kp)
        cap = store.mint("d1", max_output_tokens=1000, max_calls=5)
        req = RelayRequest(
            method="POST",
            path="/v1/messages",
            headers={CAP_HEADER: cap.token, "content-type": "application/json", "host": "relay"},
            body=b'{"model":"x"}',
        )

        result = prepare_upstream(relay, req)

        assert isinstance(result, UpstreamRequest)  # case 4: distinct TYPE, not a flag
        assert result.token == cap.token
        # real key injected under the capability header name; capability token gone
        header_vals = {k.lower(): v for k, v in result.request.headers.items()}
        assert header_vals[CAP_HEADER] == DUMMY_REAL_KEY
        assert cap.token not in result.request.headers.values()
        # method/path/body pass through unchanged (case 3)
        assert result.request.method == "POST"
        assert result.request.path == "/v1/messages"
        assert result.request.body == b'{"model":"x"}'
        assert kp.calls == 1

    def test_deny_branch_type_cannot_reach_forward_path(self):
        # case 4 structural: a DenyResponse carries no request to forward.
        deny = DenyResponse(status=401, reason="unknown")
        assert not hasattr(deny, "request")
        assert not hasattr(deny, "token")


# =========================================================================== #
# sse_extract_usage — PURE (cases 12-18)                                        #
# =========================================================================== #
class TestSseExtractUsage:
    def test_full_valid_sse_input_and_output(self):
        body = (
            b"event: message_start\n"
            b'data: {"type":"message_start","message":'
            b'{"usage":{"input_tokens":25,"output_tokens":1}}}\n\n'
            b"event: message_delta\n"
            b'data: {"type":"message_delta","usage":{"output_tokens":42}}\n\n'
            b"event: message_stop\n"
            b'data: {"type":"message_stop"}\n\n'
        )
        usage = sse_extract_usage(body)
        assert usage["input_tokens"] == 25
        assert usage["output_tokens"] == 42  # case 12

    def test_output_tokens_from_last_delta_not_summed(self):
        body = (
            b"event: message_start\n"
            b'data: {"message":{"usage":{"input_tokens":10,"output_tokens":1}}}\n\n'
            b"event: message_delta\n"
            b'data: {"usage":{"output_tokens":10}}\n\n'
            b"event: message_delta\n"
            b'data: {"usage":{"output_tokens":25}}\n\n'
        )
        usage = sse_extract_usage(body)
        assert usage["output_tokens"] == 25  # case 13: cumulative LAST, not 10+25

    def test_comments_ping_and_done_ignored(self):
        body = (
            b": this is a keep-alive comment\n\n"
            b"event: ping\n"
            b'data: {"type":"ping"}\n\n'
            b"event: message_start\n"
            b'data: {"message":{"usage":{"input_tokens":5,"output_tokens":1}}}\n\n'
            b"event: message_delta\n"
            b'data: {"usage":{"output_tokens":7}}\n\n'
            b"data: [DONE]\n\n"
        )
        usage = sse_extract_usage(body)
        assert usage["output_tokens"] == 7  # case 14
        assert usage["input_tokens"] == 5

    def test_crlf_framing_parsed(self):
        body = (
            b"event: message_start\r\n"
            b'data: {"message":{"usage":{"input_tokens":3,"output_tokens":1}}}\r\n\r\n'
            b"event: message_delta\r\n"
            b'data: {"usage":{"output_tokens":9}}\r\n\r\n'
        )
        usage = sse_extract_usage(body)
        assert usage["output_tokens"] == 9  # case 15 (CRLF)
        assert usage["input_tokens"] == 3

    def test_truncated_stream_best_effort_never_raises(self):
        body = (
            b"event: message_start\n"
            b'data: {"message":{"usage":{"input_tokens":8,"output_tokens":1}}}\n\n'
            b"event: message_delta\n"
            b'data: {"usage":{"output_tok'  # cut mid-JSON
        )
        usage = sse_extract_usage(body)  # case 16: must not raise
        # best-effort: the message_start input is still recoverable; output falls
        # back to what parsed cleanly (1) — under-meter is the logged v0 gap.
        assert usage.get("input_tokens") == 8
        assert usage.get("output_tokens", 0) >= 0

    def test_non_sse_json_body_returns_empty_so_caller_falls_back(self):
        # case 17: a plain JSON body is NOT SSE; sse_extract_usage yields {},
        # and serve_relay routes non-event-stream bodies to relay.extract_usage.
        body = b'{"usage":{"input_tokens":4,"output_tokens":11}}'
        assert sse_extract_usage(body) == {}
        # relay.extract_usage is the fallback authority for non-SSE bodies:
        fallback = extract_usage(RelayResponse(status=200, headers={}, body=body))
        assert fallback == {"input_tokens": 4, "output_tokens": 11}

    def test_garbage_bytes_return_empty(self):
        assert sse_extract_usage(b"\x00\xff not sse at all \x01") == {}  # case 18
        assert sse_extract_usage(b"") == {}


# =========================================================================== #
# read_relay_request — bounded, fail-closed request parsing (cases 6-11)        #
# =========================================================================== #
class TestReadRelayRequest:
    def _read(self, data: bytes, **kw):
        async def _run():
            reader = _reader_with(data)
            return await read_relay_request(
                reader,
                max_head_bytes=kw.get("max_head_bytes", 65536),
                max_body_bytes=kw.get("max_body_bytes", 8 * 1024 * 1024),
                timeout=kw.get("timeout", 5.0),
            )

        return asyncio.run(_run())

    def test_wellformed_post_parses(self):
        raw = _build_raw_request(token="tok", body=b'{"hello":"world"}')
        req = self._read(raw)
        assert req.method == "POST"
        assert req.path == "/v1/messages"
        assert req.body == b'{"hello":"world"}'
        vals = {k.lower(): v for k, v in req.headers.items()}
        assert vals[CAP_HEADER] == "tok"  # case 6

    def test_oversized_head_denied(self):
        big = _build_raw_request(token="tok", extra_lines=["x-pad: " + "A" * 100000])
        with pytest.raises(RelayTransportError):
            self._read(big, max_head_bytes=4096)  # case 7 (head)

    def test_oversized_body_denied(self):
        raw = _build_raw_request(token="tok", body=b"B" * 5000)
        with pytest.raises(RelayTransportError):
            self._read(raw, max_body_bytes=1024)  # case 7 (body)

    def test_read_timeout_denied(self):
        # A reader that never delivers the header terminator -> timeout -> deny.
        async def _run():
            reader = asyncio.StreamReader()
            reader.feed_data(b"POST /v1/messages HTTP/1.1\r\nhost: relay\r\n")  # no blank line
            # deliberately no feed_eof()
            return await read_relay_request(
                reader, max_head_bytes=65536, max_body_bytes=1024, timeout=0.05
            )

        with pytest.raises(RelayTransportError):
            asyncio.run(_run())  # case 8

    def test_malformed_request_line_denied(self):
        raw = b"GARBAGE-NOT-HTTP\r\n\r\n"
        with pytest.raises(RelayTransportError):
            self._read(raw)  # case 9 (malformed line)

    def test_post_missing_content_length_denied(self):
        raw = _build_raw_request(token="tok", include_cl=False, body=b"{}")
        with pytest.raises(RelayTransportError):
            self._read(raw)  # case 9 (missing CL on body method)

    def test_chunked_request_body_denied(self):
        raw = _build_raw_request(
            token="tok",
            include_cl=False,
            extra_lines=["transfer-encoding: chunked"],
            body=b"0\r\n\r\n",
        )
        with pytest.raises(RelayTransportError):
            self._read(raw)  # case 9 (chunked request body rejected in v0)

    def test_duplicate_capability_header_denied(self):
        raw = _build_raw_request(token="tok", extra_lines=[f"{CAP_HEADER}: smuggled"])
        with pytest.raises(RelayTransportError):
            self._read(raw)  # case 10: never silently pick-first

    def test_duplicate_content_length_denied(self):
        # request-smuggling defense: any duplicate header name is rejected.
        raw = _build_raw_request(token="tok", body=b"{}", extra_lines=["content-length: 999"])
        with pytest.raises(RelayTransportError):
            self._read(raw)

    def test_get_without_body_parses_auth_gates_later(self):
        raw = _build_raw_request(
            token="tok", method="GET", path="/v1/models", include_cl=False, body=b""
        )
        req = self._read(raw)
        assert req.method == "GET"
        # case 11: path passes through; auth still gates later in prepare_upstream
        assert req.path == "/v1/models"
        assert req.body == b""


# =========================================================================== #
# serve_relay end-to-end over a real unix socket (cases 3,5,19-22,31-36)        #
# =========================================================================== #
class TestServeRelayEndToEnd:
    def _serve_and_call(
        self, tmp_path, *, forwarder, store, kp, raw_builder, log_name="relay.jsonl"
    ):
        socket_path = tmp_path / "relay.sock"
        log_path = tmp_path / log_name
        relay = _relay(store, kp)
        captured: dict[str, bytes] = {}

        async def _run():
            server = await serve_relay(
                socket_path, relay, forwarder=forwarder, log_path=log_path, dispatch_id="d1"
            )
            try:
                captured["resp"] = await _roundtrip(socket_path, raw_builder())
            finally:
                server.close()
                await server.wait_closed()

        asyncio.run(_run())
        return captured["resp"], log_path

    def test_streaming_forward_incremental_and_meters(self, tmp_path):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=10000, max_calls=5)
        sse_chunks = [
            b"event: message_start\n"
            b'data: {"message":{"usage":{"input_tokens":30,"output_tokens":1}}}\n\n',
            b'event: message_delta\ndata: {"usage":{"output_tokens":55}}\n\n',
            b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
        ]
        fwd = FakeForwarder(
            headers=[("content-type", "text/event-stream")], chunks=sse_chunks, delay=0.01
        )
        resp, log_path = self._serve_and_call(
            tmp_path,
            forwarder=fwd,
            store=store,
            kp=kp,
            raw_builder=lambda: _build_raw_request(token=cap.token),
        )
        head, body = _split_response(resp)
        # case 3: forwarder saw the real key, not the capability token
        up = fwd.requests[0]
        assert {k.lower(): v for k, v in up.request.headers.items()}[CAP_HEADER] == DUMMY_REAL_KEY
        assert cap.token not in up.request.headers.values()
        # case 19: full SSE stream delivered to worker intact
        assert body == b"".join(sse_chunks)
        # case 5 + 36: metered exactly once, output from SSE parse
        assert store.usage(cap.token) == {"output_tokens": 55, "calls": 1}
        # case 22: real key never in worker bytes or log
        assert DUMMY_REAL_KEY.encode() not in resp
        assert DUMMY_REAL_KEY.encode() not in log_path.read_bytes()

    def test_framing_strips_hop_by_hop_and_te_adds_close(self, tmp_path):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=10000, max_calls=5)
        fwd = FakeForwarder(
            headers=[
                ("content-type", "text/event-stream"),
                ("transfer-encoding", "chunked"),
                ("connection", "keep-alive"),
                ("keep-alive", "timeout=5"),
                ("content-length", "999"),
                ("proxy-authenticate", "Basic"),
            ],
            chunks=[b"event: message_stop\ndata: {}\n\n"],
        )
        resp, _ = self._serve_and_call(
            tmp_path, forwarder=fwd, store=store, kp=kp,
            raw_builder=lambda: _build_raw_request(token=cap.token),
        )
        head, _ = _split_response(resp)
        low = head.lower()
        assert b"connection: close" in low  # case 31
        assert b"transfer-encoding" not in low  # case 31 (stripped, no double-dechunk)
        assert b"content-length" not in low  # case 31 (close-delimited)
        assert b"keep-alive" not in low  # case 33
        assert b"proxy-authenticate" not in low  # case 33
        assert b"content-type" in low  # preserved

    def test_content_encoding_preserved_body_not_decoded(self, tmp_path):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=10000, max_calls=5)
        gz_bytes = b"\x1f\x8b\x08\x00fake-gzip-bytes-not-really"
        fwd = FakeForwarder(
            headers=[("content-type", "application/json"), ("content-encoding", "gzip")],
            chunks=[gz_bytes],
        )
        resp, _ = self._serve_and_call(
            tmp_path, forwarder=fwd, store=store, kp=kp,
            raw_builder=lambda: _build_raw_request(token=cap.token),
        )
        head, body = _split_response(resp)
        assert b"content-encoding: gzip" in head.lower()  # case 32
        assert body == gz_bytes  # relay never gunzips

    def test_non_sse_json_response_metered_via_extract_usage(self, tmp_path):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=10000, max_calls=5)
        payload = b'{"usage":{"input_tokens":12,"output_tokens":34},"content":[]}'
        fwd = FakeForwarder(headers=[("content-type", "application/json")], chunks=[payload])
        resp, _ = self._serve_and_call(
            tmp_path, forwarder=fwd, store=store, kp=kp,
            raw_builder=lambda: _build_raw_request(token=cap.token),
        )
        _, body = _split_response(resp)
        assert body == payload
        # case 36/17: JSON body routed to relay.extract_usage -> 34 output tokens
        assert store.usage(cap.token) == {"output_tokens": 34, "calls": 1}

    def test_deny_path_never_forwards_or_fetches_key(self, tmp_path):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        never = NeverForwarder()
        resp, log_path = self._serve_and_call(
            tmp_path, forwarder=never, store=store, kp=kp,
            raw_builder=lambda: _build_raw_request(token="bogus-unknown"),
        )
        assert resp.startswith(b"HTTP/1.1 401")  # unknown -> 401
        assert never.calls == 0  # cases 1/2: forwarder never reached
        assert kp.calls == 0  # key never fetched
        assert DUMMY_REAL_KEY.encode() not in resp

    def test_upstream_error_gives_502_and_still_charges_one_call(self, tmp_path):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=10000, max_calls=5)
        fwd = FakeForwarder(error=UpstreamError("connect refused"))
        resp, _ = self._serve_and_call(
            tmp_path, forwarder=fwd, store=store, kp=kp,
            raw_builder=lambda: _build_raw_request(token=cap.token),
        )
        assert resp.startswith(b"HTTP/1.1 502")  # case 20
        # case 34 (§C): authorized + forwarder invoked -> calls charged once even on error
        assert store.usage(cap.token) == {"output_tokens": 0, "calls": 1}
        assert DUMMY_REAL_KEY.encode() not in resp

    def test_malformed_request_denied_without_forwarding(self, tmp_path):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        never = NeverForwarder()
        resp, _ = self._serve_and_call(
            tmp_path, forwarder=never, store=store, kp=kp,
            raw_builder=lambda: b"NOT-HTTP-AT-ALL\r\n\r\n",
        )
        assert resp.startswith(b"HTTP/1.1 4")  # a 4xx deny, never a forward
        assert never.calls == 0
        assert kp.calls == 0


# =========================================================================== #
# Bounded memory / backpressure (case 35)                                       #
# =========================================================================== #
class TestBackpressure:
    def test_bounded_queue_does_not_buffer_whole_response(self, tmp_path):
        # A fast upstream producing many chunks against a slow-draining worker
        # must not balloon memory: the bridge queue is bounded, so the reader
        # blocks; every byte still arrives in order.
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=1_000_000, max_calls=5)
        chunks = [f"chunk-{i:04d};".encode() for i in range(200)]
        fwd = FakeForwarder(headers=[("content-type", "application/json")], chunks=chunks)
        socket_path = tmp_path / "relay.sock"
        log_path = tmp_path / "relay.jsonl"
        relay = _relay(store, kp)
        got: dict[str, bytes] = {}

        async def _run():
            server = await serve_relay(
                socket_path, relay, forwarder=fwd, log_path=log_path, dispatch_id="d1"
            )
            try:
                reader, writer = await asyncio.open_unix_connection(path=str(socket_path))
                writer.write(_build_raw_request(token=cap.token))
                await writer.drain()
                # drain slowly to keep the bridge queue under backpressure
                buf = bytearray()
                while True:
                    piece = await reader.read(8)
                    if not piece:
                        break
                    buf.extend(piece)
                    await asyncio.sleep(0)  # yield; simulate a slow consumer
                got["body"] = bytes(buf).partition(b"\r\n\r\n")[2]
                writer.close()
                await writer.wait_closed()
            finally:
                server.close()
                await server.wait_closed()

        asyncio.run(_run())
        assert got["body"] == b"".join(chunks)  # case 35: complete + ordered


# =========================================================================== #
# No-redirect on the REAL-key upstream call (case 21)                           #
# =========================================================================== #
class TestNoRedirectUpstream:
    def test_urllib_forwarder_refuses_redirect(self):
        # The real key rides this request; a 3xx must NOT be followed (urllib
        # re-sends headers to the redirect target by default). Inject a fake
        # opener that a redirect would trip -> forwarder must raise UpstreamError.
        class RedirectingOpener:
            def open(self, request, timeout=None):  # noqa: ANN001
                raise urllib.error.HTTPError(
                    request.full_url, 302, "redirect", {}, None
                )

        fwd = make_urllib_forwarder(
            base_url="https://api.anthropic.com", opener=RedirectingOpener()
        )
        up = UpstreamRequest(
            request=RelayRequest(
                method="POST", path="/v1/messages", headers={CAP_HEADER: DUMMY_REAL_KEY}, body=b"{}"
            ),
            token="tok",
        )

        async def _run():
            with pytest.raises(UpstreamError):
                await fwd(up)

        asyncio.run(_run())

    def test_forwarder_uses_no_redirect_handler(self):
        # Structural: the production opener carries a redirect handler that
        # raises rather than follows (reuse of .36's _NoRedirectHandler pattern).
        from stigmergy import relay_transport

        opener = relay_transport._NO_REDIRECT_OPENER
        handler = next(
            h for h in opener.handlers if isinstance(h, urllib.request.HTTPRedirectHandler)
        )
        with pytest.raises(urllib.error.HTTPError):
            handler.redirect_request(
                urllib.request.Request("https://api.anthropic.com/v1/messages"),
                None, 302, "Found", {}, "https://evil.example.com",
            )

    def test_forwarder_opener_ignores_proxy_env(self, monkeypatch):
        # bead .25 audit F-1 (HIGH, opus+codex convergent): the production
        # opener carries the REAL key on a DIRECT host call; it must NOT honor
        # an inherited *_proxy env var. Reload the module UNDER a proxied env
        # and assert the built opener installs no env-derived ProxyHandler
        # (the explicit ProxyHandler({}) guard suppresses build_opener's
        # default env-reading one). Auto-catches a revert to the unguarded
        # build_opener(_NoRedirectHandler).
        import importlib
        import urllib.request as _ur

        from stigmergy import relay_transport

        monkeypatch.setenv("HTTPS_PROXY", "http://should-not-be-used.example:9")
        monkeypatch.setenv("https_proxy", "http://should-not-be-used.example:9")
        reloaded = importlib.reload(relay_transport)
        try:
            leaked = [
                h for h in reloaded._NO_REDIRECT_OPENER.handlers
                if isinstance(h, _ur.ProxyHandler) and h.proxies
            ]
            assert not leaked, f"relay opener honors inherited HTTPS_PROXY: {leaked}"
            # contrast (non-vacuous): the UNGUARDED default WOULD pick it up.
            unguarded = _ur.build_opener(reloaded._NoRedirectHandler)
            assert any(
                isinstance(h, _ur.ProxyHandler) and h.proxies
                for h in unguarded.handlers
            ), "sanity: unguarded opener picks up HTTPS_PROXY (proves the guard matters)"
        finally:
            monkeypatch.undo()
            importlib.reload(relay_transport)  # restore the clean-env module opener


# =========================================================================== #
# Capability lifecycle integration (cases 23-25)                                #
# =========================================================================== #
class TestCapabilityLifecycleIntegration:
    def _call(self, socket_path, relay, forwarder, log_path, token):
        async def _run():
            server = await serve_relay(
                socket_path, relay, forwarder=forwarder, log_path=log_path, dispatch_id="d1"
            )
            try:
                return await _roundtrip(socket_path, _build_raw_request(token=token))
            finally:
                server.close()
                await server.wait_closed()

        return asyncio.run(_run())

    def test_mint_call_revoke_replay_denied(self, tmp_path):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        relay = _relay(store, kp)
        cap = store.mint("d1", max_output_tokens=1000, max_calls=5)
        log_path = tmp_path / "relay.jsonl"
        fwd = FakeForwarder(
            headers=[("content-type", "text/event-stream")],
            chunks=[b'event: message_delta\ndata: {"usage":{"output_tokens":5}}\n\n'],
        )
        # authorized call injects real key + charges
        resp1 = self._call(tmp_path / "r1.sock", relay, fwd, log_path, cap.token)
        assert resp1.startswith(b"HTTP/1.1 200")
        assert store.usage(cap.token)["calls"] == 1
        assert kp.calls == 1

        # revoke -> replay of the SAME token is dead (case 23 / AC4)
        store.revoke("d1")
        never = NeverForwarder()
        kp2_before = kp.calls
        resp2 = self._call(tmp_path / "r2.sock", relay, never, log_path, cap.token)
        assert resp2.startswith(b"HTTP/1.1 401")
        assert never.calls == 0
        assert kp.calls == kp2_before  # key never fetched on the dead-token replay

    def test_injected_secrets_survive_revoke_and_redact(self):
        # case 24: injected_secrets works AFTER revoke (seal happens post-revoke)
        store = CapabilityStore()
        kp = CountingKeyProvider()
        relay = _relay(store, kp)
        cap = store.mint("d1", max_output_tokens=1000, max_calls=5)
        store.revoke("d1")
        secrets = relay.injected_secrets(cap.token)
        assert cap.token in secrets
        assert DUMMY_REAL_KEY in secrets

    def test_quota_tokens_exhaustion_next_call_429(self, tmp_path):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        relay = _relay(store, kp)
        cap = store.mint("d1", max_output_tokens=50, max_calls=5)
        log_path = tmp_path / "relay.jsonl"
        # first call consumes 60 output tokens (> max 50)
        fwd = FakeForwarder(
            headers=[("content-type", "text/event-stream")],
            chunks=[b'event: message_delta\ndata: {"usage":{"output_tokens":60}}\n\n'],
        )
        self._call(tmp_path / "a.sock", relay, fwd, log_path, cap.token)
        # next authorize -> quota-tokens -> 429 (case 25)
        never = NeverForwarder()
        resp = self._call(tmp_path / "b.sock", relay, never, log_path, cap.token)
        assert resp.startswith(b"HTTP/1.1 429")
        assert never.calls == 0


# =========================================================================== #
# Per-dispatch lifecycle: start_relay / stop / socket perms (cases 26-29)       #
# =========================================================================== #
class TestRelayLifecycle:
    def test_start_relay_blocks_until_listening(self, tmp_path):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        relay = _relay(store, kp)
        fwd = FakeForwarder()
        handle = start_relay(
            "d1", tmp_path, relay, forwarder=fwd, log_path=tmp_path / "relay.jsonl"
        )
        try:
            assert handle.socket_path.exists()  # case 26: socket exists on return
        finally:
            handle.stop()

    def test_stop_is_idempotent_and_fail_closed(self, tmp_path):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        relay = _relay(store, kp)
        fwd = FakeForwarder()
        handle = start_relay(
            "d1", tmp_path, relay, forwarder=fwd, log_path=tmp_path / "relay.jsonl"
        )
        sock = handle.socket_path
        handle.stop()
        handle.stop()  # idempotent (case 27) — must not raise
        assert not sock.exists()

        # after stop, a worker connect fails outright (fail-closed)
        async def _connect():
            with pytest.raises((ConnectionRefusedError, FileNotFoundError, OSError)):
                await asyncio.open_unix_connection(path=str(sock))

        asyncio.run(_connect())

    def test_socket_perms_restrict_connectors(self, tmp_path):
        # case 28 (tightened per amendment §B): 0o600 socket, 0o700 runtime dir.
        store = CapabilityStore()
        kp = CountingKeyProvider()
        relay = _relay(store, kp)
        fwd = FakeForwarder()
        runtime_dir = tmp_path / "rt"
        handle = start_relay(
            "d1", runtime_dir, relay, forwarder=fwd, log_path=runtime_dir / "relay.jsonl"
        )
        try:
            sock_mode = handle.socket_path.stat().st_mode & 0o777
            dir_mode = runtime_dir.stat().st_mode & 0o777
            assert sock_mode == 0o600, oct(sock_mode)
            assert dir_mode == 0o700, oct(dir_mode)
        finally:
            handle.stop()


# =========================================================================== #
# Concurrency quota-leash — max_calls must hold against a compromised worker    #
# opening N parallel connections (case 37; TOCTOU between authorize + charge).   #
# =========================================================================== #
class TestConcurrencyQuotaLeash:
    def test_concurrent_connections_cannot_bypass_max_calls(self, tmp_path):
        # Threat model (SPEC §4): every worker is compromised. A naive
        # authorize-now / charge-after-the-stream relay lets N concurrent
        # connections all pass authorize() (used_calls still 0) before any
        # charge lands -> all N forward with the real key, defeating the
        # max_calls leash. The relay MUST reserve the call slot atomically
        # with authorize so only max_calls ever forward.
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=1_000_000, max_calls=1)
        relay = _relay(store, kp)
        # a deliberately slow forwarder holds the first call in flight so the
        # second connection races the (un)charged window.
        fwd = FakeForwarder(
            headers=[("content-type", "text/event-stream")],
            chunks=[b'event: message_delta\ndata: {"usage":{"output_tokens":5}}\n\n'],
            delay=0.25,
        )
        socket_path = tmp_path / "relay.sock"
        log_path = tmp_path / "relay.jsonl"
        results: list[bytes] = []

        async def _run():
            server = await serve_relay(
                socket_path, relay, forwarder=fwd, log_path=log_path, dispatch_id="d1"
            )
            try:

                async def one():
                    resp = await _roundtrip(socket_path, _build_raw_request(token=cap.token))
                    results.append(resp.split(b"\r\n", 1)[0])

                await asyncio.gather(one(), one())
            finally:
                server.close()
                await server.wait_closed()

        asyncio.run(_run())
        assert len(fwd.requests) == 1  # only ONE call ever forwarded with the real key
        statuses = sorted(line.split(b" ")[1] for line in results)
        assert statuses == [b"200", b"429"]  # the other is quota-denied
        assert store.usage(cap.token)["calls"] == 1  # charged exactly once total


# =========================================================================== #
# bead .25: anthropic-beta pass-through + audit logging (live finding §9)       #
# =========================================================================== #
class TestAnthropicBetaPassthrough:
    def _serve(self, tmp_path, relay, forwarder, raw):
        socket_path = tmp_path / "relay.sock"
        log_path = tmp_path / "relay.jsonl"
        captured: dict[str, bytes] = {}

        async def _run():
            server = await serve_relay(
                socket_path, relay, forwarder=forwarder, log_path=log_path, dispatch_id="d1"
            )
            try:
                captured["resp"] = await _roundtrip(socket_path, raw)
            finally:
                server.close()
                await server.wait_closed()

        asyncio.run(_run())
        return captured["resp"], log_path

    def test_anthropic_beta_forwarded_when_allowlisted(self, tmp_path):
        # bead .25 (SB-approved widening): with anthropic-beta in the allowlist,
        # the worker's beta value reaches upstream (claude-code needs it).
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=10000, max_calls=5)
        relay = CredentialRelay(
            store=store,
            key_provider=kp,
            forwarder=lambda req: (_ for _ in ()).throw(AssertionError("sync unused")),
            capability_header=CAP_HEADER,
            upstream_header_allowlist=frozenset({"content-type", "accept", "anthropic-beta"}),
        )
        fwd = FakeForwarder(
            headers=[("content-type", "application/json")],
            chunks=[b'{"usage":{"output_tokens":1}}'],
        )
        raw = _build_raw_request(
            token=cap.token, extra_lines=["anthropic-beta: context-management-2025-06-27"]
        )
        self._serve(tmp_path, relay, fwd, raw)
        up = {k.lower(): v for k, v in fwd.requests[0].request.headers.items()}
        assert up.get("anthropic-beta") == "context-management-2025-06-27"

    def test_anthropic_beta_dropped_under_tight_default(self, tmp_path):
        # DEFAULT allowlist stays tight (honors .55): anthropic-beta is dropped.
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=10000, max_calls=5)
        relay = _relay(store, kp)  # DEFAULT_UPSTREAM_HEADER_ALLOWLIST
        fwd = FakeForwarder(
            headers=[("content-type", "application/json")],
            chunks=[b'{"usage":{"output_tokens":1}}'],
        )
        raw = _build_raw_request(
            token=cap.token, extra_lines=["anthropic-beta: context-management-2025-06-27"]
        )
        self._serve(tmp_path, relay, fwd, raw)
        up = {k.lower(): v for k, v in fwd.requests[0].request.headers.items()}
        assert "anthropic-beta" not in up

    def test_anthropic_beta_value_logged(self, tmp_path):
        # The worker-controlled beta value is AUDITED in the relay JSONL
        # (decision-5b observability), regardless of the allowlist.
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=10000, max_calls=5)
        relay = _relay(store, kp)
        fwd = FakeForwarder(
            headers=[("content-type", "application/json")],
            chunks=[b'{"usage":{"output_tokens":1}}'],
        )
        raw = _build_raw_request(
            token=cap.token, extra_lines=["anthropic-beta: interleaved-thinking-2025-05-14"]
        )
        _resp, log_path = self._serve(tmp_path, relay, fwd, raw)
        entries = [json.loads(ln) for ln in log_path.read_text().splitlines() if ln.strip()]
        allow = [e for e in entries if e["decision"] == "allow"]
        assert allow
        assert allow[-1]["anthropic_beta"] == "interleaved-thinking-2025-05-14"

    def test_anthropic_beta_absent_logs_null(self, tmp_path):
        # No anthropic-beta on the request -> logged as null (not missing key).
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=10000, max_calls=5)
        relay = _relay(store, kp)
        fwd = FakeForwarder(
            headers=[("content-type", "application/json")],
            chunks=[b'{"usage":{"output_tokens":1}}'],
        )
        raw = _build_raw_request(token=cap.token)
        _resp, log_path = self._serve(tmp_path, relay, fwd, raw)
        entries = [json.loads(ln) for ln in log_path.read_text().splitlines() if ln.strip()]
        assert entries
        assert all("anthropic_beta" in e for e in entries)
        assert entries[-1]["anthropic_beta"] is None
