"""Adversarial tests for the credential-relay hardening pt2 (bead .56) — the
metering/DoS half of the multi-model relay audit (2026-07-17,
memory/projects/stigmergy/audit-relay-2026-07-17/AUDIT-REPORT.md).

Authored by the orchestrator, not the implementor — this is a leash-bypass
control (F4 is HIGH, blocks .25) and the contract is the orchestrator's to
own (mirrors test_relay.py / test_relay_hardening.py). The implementation in
`stigmergy.relay` / `stigmergy.relay_transport` must satisfy every assertion
without weakening one to force a pass. Frozen spec: bead56-build-spec.md.

Findings covered:
- F4 (HIGH): incremental SSE usage metering — the true final cumulative
  output_tokens is captured regardless of body size (the old 8 MiB usage-tee
  dropped a final delta past the cap => under-charge on demand). Backstop:
  an unmeterable 2xx response is charged UNBUDGETABLE (remaining
  max_output_tokens), never ~0.
- F5 (MEDIUM): accept cap (max_in_flight) + wall-clock stream deadline.
- F6 (MEDIUM): output-token overshoot — closed in v0 by F5 serialization.
- F9 (LOW): a bare LF/CR in a header VALUE is a clean 400.

Threat model (SPEC §4): every worker is assumed compromised. DUMMY keys only.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import pytest

from stigmergy.relay import (
    CapabilityDenied,
    CapabilityStore,
    CredentialRelay,
    RelayError,
    RelayRequest,
    UpstreamRequest,
)
from stigmergy.relay_transport import (
    _MAX_JSON_METER_BYTES,
    _MAX_SSE_EVENT_BYTES,
    UpstreamHead,
    _SseUsageMeter,
    make_urllib_forwarder,
    serve_relay,
    sse_extract_usage,
)

DUMMY_REAL_KEY = "[REDACTED:api_key]"
CAP_HEADER = "x-api-key"


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #
class CountingKeyProvider:
    def __init__(self, key: str = DUMMY_REAL_KEY) -> None:
        self._key = key
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        return self._key


class StreamForwarder:
    """A configurable async forwarder: returns (UpstreamHead, async gen).

    ``chunks`` are yielded in order; ``delay`` inserts an await between them
    (so a stream deadline can fire); ``raise_after`` raises mid-stream after
    that many chunks (a truncated stream). ``track`` counts concurrent
    in-flight generators so the F5 accept cap is observable."""

    def __init__(
        self,
        *,
        status: int = 200,
        headers: list[tuple[str, str]] | None = None,
        chunks: list[bytes] | None = None,
        delay: float = 0.0,
        head_delay: float = 0.0,
        raise_after: int | None = None,
        track: dict | None = None,
    ) -> None:
        self.status = status
        self.headers = headers if headers is not None else [("content-type", "text/event-stream")]
        self.chunks = chunks if chunks is not None else [b"event: message_stop\ndata: {}\n\n"]
        self.delay = delay
        self.head_delay = head_delay
        self.raise_after = raise_after
        self.track = track
        self.requests: list[UpstreamRequest] = []

    async def __call__(self, upstream: UpstreamRequest):  # noqa: ANN201
        self.requests.append(upstream)
        if self.head_delay:
            # stall BEFORE returning the head — exercises the F5 deadline
            # covering the forwarder call itself (bead .56 review, MAJOR-3).
            await asyncio.sleep(self.head_delay)
        head = UpstreamHead(status=self.status, headers=list(self.headers))
        chunks = self.chunks
        delay = self.delay
        raise_after = self.raise_after
        track = self.track

        async def _gen():
            if track is not None:
                track["cur"] = track.get("cur", 0) + 1
                track["max"] = max(track.get("max", 0), track["cur"])
            try:
                count = 0
                for chunk in chunks:
                    if delay:
                        await asyncio.sleep(delay)
                    yield chunk
                    count += 1
                    if raise_after is not None and count >= raise_after:
                        raise RuntimeError("upstream truncated mid-stream")
            finally:
                if track is not None:
                    track["cur"] -= 1

        return head, _gen()


def _relay(store: CapabilityStore, kp: CountingKeyProvider) -> CredentialRelay:
    return CredentialRelay(
        store=store,
        key_provider=kp,
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
) -> bytes:
    lines = [f"{method} {path} HTTP/1.1", "host: relay"]
    if token is not None:
        lines.append(f"{CAP_HEADER}: {token}")
    lines.append("content-type: application/json")
    lines.append(f"content-length: {len(body)}")
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


def _serve_and_call(
    tmp_path,
    *,
    forwarder,
    store,
    kp,
    raw_builder,
    max_in_flight: int = 1,
    stream_deadline: float = 300.0,
):
    socket_path = tmp_path / "relay.sock"
    log_path = tmp_path / "relay.jsonl"
    relay = _relay(store, kp)
    captured: dict[str, bytes] = {}

    async def _run():
        server = await serve_relay(
            socket_path,
            relay,
            forwarder=forwarder,
            log_path=log_path,
            dispatch_id="d1",
            max_in_flight=max_in_flight,
            stream_deadline=stream_deadline,
        )
        try:
            captured["resp"] = await asyncio.wait_for(
                _roundtrip(socket_path, raw_builder()), timeout=15.0
            )
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(_run())
    return captured["resp"], log_path


def _split(raw: bytes) -> tuple[bytes, bytes]:
    head, _, body = raw.partition(b"\r\n\r\n")
    return head, body


# =========================================================================== #
# _SseUsageMeter — incremental, bounded, cross-chunk (F4)                       #
# =========================================================================== #
class TestSseUsageMeterUnit:
    def _full_body(self, out: int) -> bytes:
        delta = b'event: message_delta\ndata: {"usage":{"output_tokens":' + str(out).encode()
        return (
            b"event: message_start\n"
            b'data: {"message":{"usage":{"input_tokens":30,"output_tokens":1}}}\n\n'
            + delta + b"}}\n\n"
            b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
        )

    def test_single_chunk_matches_sse_extract_usage(self):
        body = self._full_body(55)
        meter = _SseUsageMeter()
        meter.feed(body)
        assert meter.finalize() == 55
        # parity with the batch parser on the same bytes
        assert sse_extract_usage(body).get("output_tokens") == 55

    def test_final_delta_split_across_chunks_still_captured(self):
        # The core F4 property: a message_delta JSON straddling a TCP chunk
        # boundary must still be parsed (the batch parser run per-chunk would
        # miss it — a new bypass). Split at every single byte to be brutal.
        body = self._full_body(4242)
        meter = _SseUsageMeter()
        for i in range(0, len(body), 1):
            meter.feed(body[i : i + 1])
        assert meter.finalize() == 4242

    def test_final_delta_split_mid_json(self):
        body = self._full_body(999)
        # split right inside the final delta's JSON payload
        cut = body.index(b'"output_tokens":999') + 5
        meter = _SseUsageMeter()
        meter.feed(body[:cut])
        meter.feed(body[cut:])
        assert meter.finalize() == 999

    def test_crlf_framing_split_across_chunks(self):
        body = self._full_body(77).replace(b"\n", b"\r\n")
        meter = _SseUsageMeter()
        # split so a \r\n straddles the boundary
        for i in range(0, len(body), 3):
            meter.feed(body[i : i + 3])
        assert meter.finalize() == 77

    def test_last_delta_wins_not_summed(self):
        body = (
            b'event: message_delta\ndata: {"usage":{"output_tokens":10}}\n\n'
            b'event: message_delta\ndata: {"usage":{"output_tokens":25}}\n\n'
        )
        meter = _SseUsageMeter()
        meter.feed(body)
        assert meter.finalize() == 25

    def test_message_start_placeholder_alone_is_unmeterable(self):
        # F4 tri-state (bead .56 review): the message_start `output_tokens` is
        # a PLACEHOLDER, not a determined final usage. A stream that ends
        # without a real (top-level) message_delta cumulative usage is
        # UNMETERABLE => finalize() None => the caller charges unbudgetable,
        # NOT the low placeholder (the residual F4 leash bypass).
        body = (
            b"event: message_start\n"
            b'data: {"message":{"usage":{"input_tokens":30,"output_tokens":7}}}\n\n'
        )
        meter = _SseUsageMeter()
        meter.feed(body)
        assert meter.finalize() is None
        # the BATCH parser's own observable behavior is UNCHANGED (still
        # surfaces the placeholder) — only the incremental meter is fail-closed.
        assert sse_extract_usage(body).get("output_tokens") == 7

    def test_no_usage_clean_stream_is_unmeterable(self):
        body = b"event: ping\ndata: {}\n\nevent: message_stop\ndata: {}\n\n"
        meter = _SseUsageMeter()
        meter.feed(body)
        assert meter.finalize() is None

    def test_garbage_never_raises_and_is_unmeterable(self):
        meter = _SseUsageMeter()
        meter.feed(b"\x00\x01\x02 not sse at all \xff\xfe")
        assert meter.finalize() is None  # no valid final usage; and never raised

    def test_zero_is_a_determined_value_not_unmeterable(self):
        # a positively-parsed 0 is a DETERMINED value (charge 0), distinct from
        # "could not determine" (None => unbudgetable).
        body = self._full_body(0)
        meter = _SseUsageMeter()
        meter.feed(body)
        assert meter.finalize() == 0

    def test_nonint_final_delta_is_unmeterable(self):
        body = self._full_body(55).replace(b'"output_tokens":55', b'"output_tokens":"55"')
        meter = _SseUsageMeter()
        meter.feed(body)
        assert meter.finalize() is None

    def test_bool_final_delta_is_unmeterable(self):
        body = self._full_body(55).replace(b'"output_tokens":55', b'"output_tokens":true')
        meter = _SseUsageMeter()
        meter.feed(body)
        assert meter.finalize() is None

    def test_negative_final_delta_is_unmeterable(self):
        body = self._full_body(55).replace(b'"output_tokens":55', b'"output_tokens":-5')
        meter = _SseUsageMeter()
        meter.feed(body)
        assert meter.finalize() is None

    def test_complete_event_over_bound_is_unmeterable(self):
        # the per-event bound applies to a COMPLETE (delimited) event too, not
        # just an unterminated pending tail.
        huge = b"event: message_delta\ndata: {" + b"x" * (_MAX_SSE_EVENT_BYTES + 16) + b"}\n\n"
        meter = _SseUsageMeter()
        meter.feed(huge)
        assert meter.finalize() is None

    def test_feed_nonbytes_never_raises_and_is_unmeterable(self):
        meter = _SseUsageMeter()
        meter.feed("not bytes")  # type: ignore[arg-type]  # defensive never-raise
        assert meter.finalize() is None

    def test_mixed_newline_terminators_recognized(self):
        # CRLF+LF and LF+CRLF blank lines are recognized as event boundaries
        # (the split-CRLF false-boundary hazard is still avoided).
        for sep in (b"\r\n\n", b"\n\r\n"):
            body = (
                b'event: message_delta\ndata: {"usage":{"output_tokens":88}}' + sep
                + b"event: message_stop\ndata: {}\n\n"
            )
            meter = _SseUsageMeter()
            meter.feed(body)
            assert meter.finalize() == 88

    def test_valid_low_then_invalid_final_is_unmeterable(self):
        # F4 round-3 (review r2 MAJOR-1): usage is CUMULATIVE, last delta wins.
        # A valid low value followed by a LATER invalid/malformed cumulative
        # update must NOT keep charging the earlier low value — the authoritative
        # final is corrupt => unmeterable => caller charges unbudgetable.
        for bad in (b'"999"', b"true", b"-5", b"null"):
            body = (
                b'event: message_delta\ndata: {"usage":{"output_tokens":1}}\n\n'
                b'event: message_delta\ndata: {"usage":{"output_tokens":' + bad + b"}}\n\n"
            )
            meter = _SseUsageMeter()
            meter.feed(body)
            assert meter.finalize() is None, f"bad={bad!r} should be unmeterable"

    def test_valid_then_valid_takes_last(self):
        body = (
            b'event: message_delta\ndata: {"usage":{"output_tokens":1}}\n\n'
            b'event: message_delta\ndata: {"usage":{"output_tokens":25}}\n\n'
        )
        meter = _SseUsageMeter()
        meter.feed(body)
        assert meter.finalize() == 25

    def test_bounded_huge_int_event_never_raises(self):
        # A single bounded event whose JSON carries a >4300-digit integer must
        # not raise (Python's int-string limit ValueError is swallowed per the
        # never-raise contract) — it simply contributes no valid usage.
        huge = b"1" * 4301
        body = b'event: message_delta\ndata: {"usage":{"output_tokens":' + huge + b"}}\n\n"
        meter = _SseUsageMeter()
        meter.feed(body)
        assert meter.finalize() is None  # unparseable-as-int => no valid final

    def test_batch_parser_negative_value_behavior_preserved(self):
        # §8 A/H regression: sse_extract_usage (the BATCH parser) must keep its
        # HEAD observable behavior — it records a negative top-level value —
        # even though the incremental meter treats the same input as invalid.
        body = b'event: message_delta\ndata: {"usage":{"output_tokens":-5}}\n\n'
        assert sse_extract_usage(body).get("output_tokens") == -5
        meter = _SseUsageMeter()
        meter.feed(body)
        assert meter.finalize() is None  # meter is fail-closed on the negative

    def test_oversized_single_event_is_unmeterable(self):
        # A single event that never terminates and exceeds the per-event bound
        # is unparseable => finalize() is None (=> caller charges unbudgetable).
        meter = _SseUsageMeter()
        blob = b"event: message_delta\ndata: {" + (b"x" * (_MAX_SSE_EVENT_BYTES + 4096))
        meter.feed(blob)
        assert meter.finalize() is None

    def test_bounded_no_whole_body_retention(self):
        # Feeding many MiB of well-formed terminated events must not retain the
        # whole body — the pending buffer drains after each event terminator.
        meter = _SseUsageMeter()
        one = b"event: ping\ndata: {\"type\":\"ping\"}\n\n"
        for _ in range(200_000):  # ~7 MiB of events
            meter.feed(one)
        meter.feed(b'event: message_delta\ndata: {"usage":{"output_tokens":321}}\n\n')
        # never went unmeterable, captured the final delta, retained ~nothing
        assert meter.finalize() == 321


# =========================================================================== #
# F4 — the leash-bypass RED test: final usage past the OLD 8 MiB tee cap        #
# =========================================================================== #
class TestF4LeashBypass:
    def test_final_usage_past_8mib_is_metered_not_undercharged(self, tmp_path):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=1_000_000, max_calls=5)
        # message_start (early placeholder output=1), then > 8 MiB of valid
        # filler events, then the TRUE final delta (past the old cap), then stop.
        start = (
            b"event: message_start\n"
            b'data: {"message":{"usage":{"input_tokens":30,"output_tokens":1}}}\n\n'
        )
        filler_event = b"event: ping\ndata: {\"type\":\"ping\"}\n\n"
        filler_chunk = filler_event * 2000  # ~66 KiB of complete events
        n_filler = 140  # ~9.2 MiB total, comfortably past the 8 MiB cap
        final = b'event: message_delta\ndata: {"usage":{"output_tokens":777}}\n\n'
        stop = b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
        chunks = [start] + [filler_chunk] * n_filler + [final, stop]
        fwd = StreamForwarder(
            headers=[("content-type", "text/event-stream")], chunks=chunks
        )
        resp, log_path = _serve_and_call(
            tmp_path, forwarder=fwd, store=store, kp=kp,
            raw_builder=lambda: _build_raw_request(token=cap.token),
        )
        _, body = _split(resp)
        # F4: the true final cumulative output_tokens is captured, not the
        # placeholder-1 the old truncated-tee parser would have charged.
        assert store.usage(cap.token) == {"output_tokens": 777, "calls": 1}
        # delivery intact: the worker still received every byte (incl. the tail)
        assert body.endswith(stop)
        assert len(body) == sum(len(c) for c in chunks)
        # real key never leaked
        assert DUMMY_REAL_KEY.encode() not in resp
        assert DUMMY_REAL_KEY.encode() not in log_path.read_bytes()


# =========================================================================== #
# F4 backstop — unmeterable 2xx => charge remaining, NEVER ~0                    #
# =========================================================================== #
class TestF4Backstop:
    def _exhausted(self, store, token) -> bool:
        try:
            store.authorize(token)
        except CapabilityDenied as exc:
            return exc.reason == "quota-tokens"
        return False

    def test_oversized_sse_event_charges_unbudgetable(self, tmp_path):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=5000, max_calls=5)
        # one 2xx SSE event exceeding the per-event bound, no parseable usage
        blob = b"event: message_delta\ndata: {" + (b"y" * (_MAX_SSE_EVENT_BYTES + 8192))
        fwd = StreamForwarder(
            headers=[("content-type", "text/event-stream")], chunks=[blob]
        )
        resp, _ = _serve_and_call(
            tmp_path, forwarder=fwd, store=store, kp=kp,
            raw_builder=lambda: _build_raw_request(token=cap.token),
        )
        assert resp.startswith(b"HTTP/1.1 200")
        assert store.usage(cap.token)["output_tokens"] == 5000  # saturated, not ~0
        assert self._exhausted(store, cap.token)

    def test_oversized_json_body_charges_unbudgetable(self, tmp_path):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=5000, max_calls=5)
        # a 2xx JSON body larger than the JSON meter cap => cannot parse usage
        big = b'{"usage":{"output_tokens":3},"content":"' + (b"z" * (_MAX_JSON_METER_BYTES + 4096))
        fwd = StreamForwarder(
            headers=[("content-type", "application/json")], chunks=[big]
        )
        resp, _ = _serve_and_call(
            tmp_path, forwarder=fwd, store=store, kp=kp,
            raw_builder=lambda: _build_raw_request(token=cap.token),
        )
        assert resp.startswith(b"HTTP/1.1 200")
        assert store.usage(cap.token)["output_tokens"] == 5000  # NOT the fake 3, NOT 0
        assert self._exhausted(store, cap.token)

    def test_truncated_2xx_stream_charges_unbudgetable(self, tmp_path):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=5000, max_calls=5)
        # a 2xx stream that dies AFTER the head + first chunk, BEFORE the final
        # usage delta — a compromised worker could abort here to dodge metering.
        chunks = [
            b"event: message_start\n"
            b'data: {"message":{"usage":{"input_tokens":30,"output_tokens":1}}}\n\n',
            b'event: content_block_delta\ndata: {"delta":{"text":"partial"}}\n\n',
        ]
        fwd = StreamForwarder(
            headers=[("content-type", "text/event-stream")], chunks=chunks, raise_after=2
        )
        resp, _ = _serve_and_call(
            tmp_path, forwarder=fwd, store=store, kp=kp,
            raw_builder=lambda: _build_raw_request(token=cap.token),
        )
        # head already 200 by the time truncation hit; leash still fails closed
        assert store.usage(cap.token)["output_tokens"] == 5000
        assert self._exhausted(store, cap.token)

    def test_non_2xx_response_does_not_penalize_leash(self, tmp_path):
        # An upstream 4xx/5xx generated nothing chargeable — the call slot is
        # reserved (calls=1) but the token budget is NOT nuked (no false
        # unbudgetable), so a legitimate retry within max_calls still works.
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=5000, max_calls=5)
        fwd = StreamForwarder(
            status=400,
            headers=[("content-type", "application/json")],
            chunks=[b'{"error":{"type":"invalid_request_error"}}'],
        )
        resp, _ = _serve_and_call(
            tmp_path, forwarder=fwd, store=store, kp=kp,
            raw_builder=lambda: _build_raw_request(token=cap.token),
        )
        assert resp.startswith(b"HTTP/1.1 400")
        assert store.usage(cap.token) == {"output_tokens": 0, "calls": 1}
        assert store.is_live(cap.token) is True  # still usable


# =========================================================================== #
# F5 — accept cap (max_in_flight) + stream deadline                             #
# =========================================================================== #
class TestF5AcceptCap:
    def test_max_in_flight_one_serializes_handlers(self, tmp_path):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=1_000_000, max_calls=5)
        track: dict = {}
        fwd = StreamForwarder(
            headers=[("content-type", "text/event-stream")],
            chunks=[b'event: message_delta\ndata: {"usage":{"output_tokens":5}}\n\n'],
            delay=0.15,
            track=track,
        )
        socket_path = tmp_path / "relay.sock"
        log_path = tmp_path / "relay.jsonl"
        relay = _relay(store, kp)
        statuses: list[bytes] = []

        async def _run():
            server = await serve_relay(
                socket_path, relay, forwarder=fwd, log_path=log_path,
                dispatch_id="d1", max_in_flight=1,
            )
            try:
                async def one():
                    resp = await asyncio.wait_for(
                        _roundtrip(socket_path, _build_raw_request(token=cap.token)), timeout=15.0
                    )
                    statuses.append(resp.split(b" ")[1])  # the status code, phrase-agnostic

                await asyncio.gather(one(), one())
            finally:
                server.close()
                await server.wait_closed()

        asyncio.run(_run())
        # with max_in_flight=1 only ONE handler streams at a time
        assert track.get("max") == 1
        # both calls still served (max_calls=5), both 200
        assert len(fwd.requests) == 2
        assert sorted(statuses) == [b"200", b"200"]

    def test_stream_deadline_tears_down_slow_drain_and_fails_closed(self, tmp_path):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=5000, max_calls=5)
        # head + first chunk arrive, then the upstream stalls PAST the deadline
        chunks = [
            b"event: message_start\n"
            b'data: {"message":{"usage":{"input_tokens":30,"output_tokens":1}}}\n\n',
            b'event: message_delta\ndata: {"usage":{"output_tokens":900}}\n\n',
        ]
        fwd = StreamForwarder(
            headers=[("content-type", "text/event-stream")], chunks=chunks, delay=2.0
        )
        # deadline well under the 2.0s inter-chunk stall
        resp, _ = _serve_and_call(
            tmp_path, forwarder=fwd, store=store, kp=kp,
            raw_builder=lambda: _build_raw_request(token=cap.token),
            stream_deadline=0.2,
        )
        # connection was torn down (did not hang — _serve_and_call has a 15s
        # wait_for), and the metered call fails closed (unbudgetable), never ~0
        assert store.usage(cap.token)["output_tokens"] == 5000
        with pytest.raises(CapabilityDenied):
            store.authorize(cap.token)


# =========================================================================== #
# F9 — bare LF/CR in a header VALUE is a clean 400 deny                          #
# =========================================================================== #
class TestF9HeaderValueControlChars:
    def test_bare_lf_in_header_value_denied_without_forwarding(self, tmp_path):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=5000, max_calls=5)

        class BoomForwarder:
            calls = 0

            async def __call__(self, upstream):  # noqa: ANN001, ANN201
                BoomForwarder.calls += 1
                raise AssertionError("must never forward a malformed request")

        resp, _ = _serve_and_call(
            tmp_path, forwarder=BoomForwarder(), store=store, kp=kp,
            raw_builder=lambda: _build_raw_request(
                token=cap.token, extra_lines=["x-foo: bar\nevil: injected"]
            ),
        )
        assert resp.startswith(b"HTTP/1.1 4")  # a 4xx deny
        assert BoomForwarder.calls == 0
        assert kp.calls == 0

    def test_bare_lf_in_header_name_denied_without_forwarding(self, tmp_path):
        # F9 extension (bead .56 review): a bare LF in a header NAME is also a
        # clean 4xx deny, not passed through to metadata processing.
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=5000, max_calls=5)

        class BoomForwarder:
            calls = 0

            async def __call__(self, upstream):  # noqa: ANN001, ANN201
                BoomForwarder.calls += 1
                raise AssertionError("must never forward a malformed request")

        resp, _ = _serve_and_call(
            tmp_path, forwarder=BoomForwarder(), store=store, kp=kp,
            raw_builder=lambda: _build_raw_request(
                token=cap.token, extra_lines=["x-fo\no: bar"]
            ),
        )
        assert resp.startswith(b"HTTP/1.1 4")
        assert BoomForwarder.calls == 0
        assert kp.calls == 0


# =========================================================================== #
# Round-2 (cross-family review, review-56-codex.md): tri-state metering so a    #
# 2xx whose true usage cannot be determined is unbudgetable, NOT under-charged. #
# =========================================================================== #
class TestF4TriStateMetering:
    def _exhausted(self, store, token) -> bool:
        try:
            store.authorize(token)
        except CapabilityDenied as exc:
            return exc.reason == "quota-tokens"
        return False

    def test_clean_sse_without_final_delta_is_unbudgetable(self, tmp_path):
        # 2xx SSE: message_start placeholder + delivered model output +
        # message_stop, but NO final message_delta usage (a worker can force
        # this via a short Content-Length upstream close that urllib does not
        # surface as an error). The worker got the tokens; usage is NOT
        # determined => unbudgetable, NOT the placeholder-1 under-charge.
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=5000, max_calls=5)
        chunks = [
            b"event: message_start\n"
            b'data: {"message":{"usage":{"input_tokens":30,"output_tokens":1}}}\n\n',
            b'event: content_block_delta\ndata: {"delta":{"text":"model output kept"}}\n\n',
            b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
        ]
        fwd = StreamForwarder(headers=[("content-type", "text/event-stream")], chunks=chunks)
        resp, _ = _serve_and_call(
            tmp_path, forwarder=fwd, store=store, kp=kp,
            raw_builder=lambda: _build_raw_request(token=cap.token),
        )
        assert resp.startswith(b"HTTP/1.1 200")
        assert store.usage(cap.token)["output_tokens"] == 5000  # saturated, not 1
        assert self._exhausted(store, cap.token)

    def test_nonint_final_delta_is_unbudgetable(self, tmp_path):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=5000, max_calls=5)
        chunks = [b'event: message_delta\ndata: {"usage":{"output_tokens":"999"}}\n\n']
        fwd = StreamForwarder(headers=[("content-type", "text/event-stream")], chunks=chunks)
        resp, _ = _serve_and_call(
            tmp_path, forwarder=fwd, store=store, kp=kp,
            raw_builder=lambda: _build_raw_request(token=cap.token),
        )
        assert resp.startswith(b"HTTP/1.1 200")
        assert store.usage(cap.token)["output_tokens"] == 5000
        assert self._exhausted(store, cap.token)

    def test_valid_final_delta_is_charged_exactly(self, tmp_path):
        # the positive path: a well-formed final delta on a 2xx is metered at
        # its exact value (regression guard that tri-state did not over-reach).
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=5000, max_calls=5)
        chunks = [
            b"event: message_start\n"
            b'data: {"message":{"usage":{"input_tokens":30,"output_tokens":1}}}\n\n',
            b'event: message_delta\ndata: {"usage":{"output_tokens":123}}\n\n',
            b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
        ]
        fwd = StreamForwarder(headers=[("content-type", "text/event-stream")], chunks=chunks)
        resp, _ = _serve_and_call(
            tmp_path, forwarder=fwd, store=store, kp=kp,
            raw_builder=lambda: _build_raw_request(token=cap.token),
        )
        assert resp.startswith(b"HTTP/1.1 200")
        assert store.usage(cap.token) == {"output_tokens": 123, "calls": 1}
        assert store.is_live(cap.token) is True

    def test_206_partial_content_still_metered(self, tmp_path):
        # 2xx breadth: a 206 with a valid final delta is a success and metered.
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=5000, max_calls=5)
        chunks = [b'event: message_delta\ndata: {"usage":{"output_tokens":42}}\n\n']
        fwd = StreamForwarder(
            status=206, headers=[("content-type", "text/event-stream")], chunks=chunks
        )
        resp, _ = _serve_and_call(
            tmp_path, forwarder=fwd, store=store, kp=kp,
            raw_builder=lambda: _build_raw_request(token=cap.token),
        )
        assert store.usage(cap.token) == {"output_tokens": 42, "calls": 1}


class TestF4UnsupportedRepresentation:
    def _exhausted(self, store, token) -> bool:
        try:
            store.authorize(token)
        except CapabilityDenied as exc:
            return exc.reason == "quota-tokens"
        return False

    def test_gzip_sse_is_unbudgetable_even_if_body_would_parse(self, tmp_path):
        # content-encoding gzip on a 2xx: the relay never decompresses, so it
        # must NOT trust the (declared-compressed) body's usage — unbudgetable
        # even though these plaintext bytes WOULD parse to 999.
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=5000, max_calls=5)
        chunks = [b'event: message_delta\ndata: {"usage":{"output_tokens":999}}\n\n']
        fwd = StreamForwarder(
            headers=[("content-type", "text/event-stream"), ("content-encoding", "gzip")],
            chunks=chunks,
        )
        resp, _ = _serve_and_call(
            tmp_path, forwarder=fwd, store=store, kp=kp,
            raw_builder=lambda: _build_raw_request(token=cap.token),
        )
        assert resp.startswith(b"HTTP/1.1 200")
        assert store.usage(cap.token)["output_tokens"] == 5000  # not 999
        assert self._exhausted(store, cap.token)

    def test_gzip_json_is_unbudgetable(self, tmp_path):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=5000, max_calls=5)
        chunks = [b'{"usage":{"output_tokens":999}}']
        fwd = StreamForwarder(
            headers=[("content-type", "application/json"), ("content-encoding", "gzip")],
            chunks=chunks,
        )
        resp, _ = _serve_and_call(
            tmp_path, forwarder=fwd, store=store, kp=kp,
            raw_builder=lambda: _build_raw_request(token=cap.token),
        )
        assert store.usage(cap.token)["output_tokens"] == 5000
        assert self._exhausted(store, cap.token)

    def test_text_plain_2xx_is_unbudgetable(self, tmp_path):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=5000, max_calls=5)
        fwd = StreamForwarder(
            headers=[("content-type", "text/plain")],
            chunks=[b"a large plain-text model answer the worker keeps"],
        )
        resp, _ = _serve_and_call(
            tmp_path, forwarder=fwd, store=store, kp=kp,
            raw_builder=lambda: _build_raw_request(token=cap.token),
        )
        assert store.usage(cap.token)["output_tokens"] == 5000
        assert self._exhausted(store, cap.token)

    def test_missing_content_type_2xx_is_unbudgetable(self, tmp_path):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=5000, max_calls=5)
        fwd = StreamForwarder(headers=[], chunks=[b"body with no content-type header"])
        resp, _ = _serve_and_call(
            tmp_path, forwarder=fwd, store=store, kp=kp,
            raw_builder=lambda: _build_raw_request(token=cap.token),
        )
        assert store.usage(cap.token)["output_tokens"] == 5000
        assert self._exhausted(store, cap.token)


class TestF5ForwarderDeadline:
    def test_pre_head_stall_hits_deadline_and_502(self, tmp_path):
        # MAJOR-3: the stream_deadline must bound `await forwarder(...)` too.
        # A stall BEFORE the head is returned must not hold the real-key-bearing
        # request open indefinitely — it hits the deadline and becomes a 502
        # (the call was reserved; no output charged; capability still live —
        # nothing was generated, so no unbudgetable penalty).
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=5000, max_calls=5)
        fwd = StreamForwarder(
            headers=[("content-type", "text/event-stream")],
            chunks=[b'event: message_delta\ndata: {"usage":{"output_tokens":900}}\n\n'],
            head_delay=3.0,
        )
        resp, _ = _serve_and_call(
            tmp_path, forwarder=fwd, store=store, kp=kp,
            raw_builder=lambda: _build_raw_request(token=cap.token),
            stream_deadline=0.2,
        )
        assert resp.startswith(b"HTTP/1.1 502")
        assert store.usage(cap.token) == {"output_tokens": 0, "calls": 1}
        assert store.is_live(cap.token) is True


class TestF5ConfigValidation:
    def _relay_for(self, tmp_path):
        return _relay(CapabilityStore(), CountingKeyProvider())

    def test_max_in_flight_other_than_one_rejected(self, tmp_path):
        # v0: max_in_flight > 1 reopens F6 (output-token overshoot); an API
        # that silently accepts a known-unsafe value is unsafe-by-config.
        relay = self._relay_for(tmp_path)
        for bad in (0, 2, 5):
            async def _run(mif=bad):
                return await serve_relay(
                    tmp_path / f"s{mif}.sock", relay, forwarder=StreamForwarder(),
                    log_path=tmp_path / "l.jsonl", max_in_flight=mif,
                )
            with pytest.raises((RelayError, ValueError)):
                asyncio.run(_run())

    def test_nonpositive_stream_deadline_rejected(self, tmp_path):
        relay = self._relay_for(tmp_path)
        for bad in (0.0, -1.0):
            async def _run(d=bad):
                return await serve_relay(
                    tmp_path / "s.sock", relay, forwarder=StreamForwarder(),
                    log_path=tmp_path / "l.jsonl", stream_deadline=d,
                )
            with pytest.raises((RelayError, ValueError)):
                asyncio.run(_run())


# =========================================================================== #
# Round-3 (re-review, review-56-codex-round2.md): fail-closed hardening of the  #
# metering core against unsupported/malformed upstream representations + a       #
# genuinely-new pre-head-cancellation resource leak from the round-2 restructure.#
# =========================================================================== #
class TestF4Round3TriStateAndRepresentation:
    def _exhausted(self, store, token) -> bool:
        try:
            store.authorize(token)
        except CapabilityDenied as exc:
            return exc.reason == "quota-tokens"
        return False

    def test_valid_low_then_invalid_final_delta_is_unbudgetable(self, tmp_path):
        # E2E of the sticky-flag fix (H): an early valid delta followed by a
        # malformed final cumulative delta must be unbudgetable, not charged low.
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=5000, max_calls=5)
        chunks = [
            b'event: message_delta\ndata: {"usage":{"output_tokens":1}}\n\n',
            b'event: message_delta\ndata: {"usage":{"output_tokens":"999999"}}\n\n',
            b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
        ]
        fwd = StreamForwarder(headers=[("content-type", "text/event-stream")], chunks=chunks)
        resp, _ = _serve_and_call(
            tmp_path, forwarder=fwd, store=store, kp=kp,
            raw_builder=lambda: _build_raw_request(token=cap.token),
        )
        assert resp.startswith(b"HTTP/1.1 200")
        assert store.usage(cap.token)["output_tokens"] == 5000  # not 1
        assert self._exhausted(store, cap.token)

    def test_text_plain_with_valid_json_body_is_unbudgetable(self, tmp_path):
        # media-type token gate (I): a text/plain 2xx whose body HAPPENS to be
        # valid JSON with a valid usage field must NOT be trusted/charged — only
        # text/event-stream and application/json are meterable representations.
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=5000, max_calls=5)
        body = b'{"usage":{"output_tokens":3},"content":"delivered to the worker"}'
        fwd = StreamForwarder(headers=[("content-type", "text/plain")], chunks=[body])
        resp, _ = _serve_and_call(
            tmp_path, forwarder=fwd, store=store, kp=kp,
            raw_builder=lambda: _build_raw_request(token=cap.token),
        )
        assert resp.startswith(b"HTTP/1.1 200")
        assert store.usage(cap.token)["output_tokens"] == 5000  # not the fake 3
        assert self._exhausted(store, cap.token)

    def test_deceptive_substring_content_type_not_metered_as_sse(self, tmp_path):
        # `text/plain; note=text/event-stream` must be classified by its media
        # TYPE token (text/plain), not a substring match => unbudgetable.
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=5000, max_calls=5)
        chunks = [b'event: message_delta\ndata: {"usage":{"output_tokens":999}}\n\n']
        fwd = StreamForwarder(
            headers=[("content-type", "text/plain; note=text/event-stream")], chunks=chunks
        )
        resp, _ = _serve_and_call(
            tmp_path, forwarder=fwd, store=store, kp=kp,
            raw_builder=lambda: _build_raw_request(token=cap.token),
        )
        assert store.usage(cap.token)["output_tokens"] == 5000
        assert self._exhausted(store, cap.token)

    def test_duplicate_encoding_identity_then_gzip_is_unbudgetable(self, tmp_path):
        # collect-ALL Content-Encoding (I): a first `identity` must not mask a
        # later `gzip` — any present non-identity coding => unbudgetable.
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=5000, max_calls=5)
        chunks = [b'event: message_delta\ndata: {"usage":{"output_tokens":999}}\n\n']
        fwd = StreamForwarder(
            headers=[
                ("content-type", "text/event-stream"),
                ("content-encoding", "identity"),
                ("content-encoding", "gzip"),
            ],
            chunks=chunks,
        )
        resp, _ = _serve_and_call(
            tmp_path, forwarder=fwd, store=store, kp=kp,
            raw_builder=lambda: _build_raw_request(token=cap.token),
        )
        assert store.usage(cap.token)["output_tokens"] == 5000
        assert self._exhausted(store, cap.token)

    def test_bounded_huge_int_json_after_200_is_unbudgetable_not_500(self, tmp_path):
        # metering totality (J): a bounded JSON body whose output_tokens is a
        # >4300-digit int raises ValueError under Python's int-string limit
        # AFTER the 200 was delivered. It must fail closed (unbudgetable), NOT
        # escape to the generic handler-exception 500 and leave the token live.
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=5000, max_calls=5)
        body = b'{"usage":{"output_tokens":' + b"1" * 4301 + b"}}"
        fwd = StreamForwarder(headers=[("content-type", "application/json")], chunks=[body])
        resp, _ = _serve_and_call(
            tmp_path, forwarder=fwd, store=store, kp=kp,
            raw_builder=lambda: _build_raw_request(token=cap.token),
        )
        assert resp.startswith(b"HTTP/1.1 200")  # head was already delivered
        assert store.usage(cap.token)["output_tokens"] == 5000  # fail-closed, not 0
        assert self._exhausted(store, cap.token)


class TestForwarderPreHeadCancellation:
    def test_pre_head_cancellation_releases_bridge_thread_and_closes_upstream(self):
        # Round-3 fix K (the one genuinely-new bug from the fix-C restructure):
        # when the widened stream_deadline cancels the forwarder's pre-head
        # `await head_future`, make_urllib_forwarder must signal its already-
        # started bridge thread (set stop_event + release the bridge semaphore)
        # so the thread closes the real-key-bearing upstream response and exits
        # instead of blocking forever on the semaphore.
        release = threading.Event()
        closed = {"v": False}

        class _Headers:
            def items(self):
                return [("content-type", "text/event-stream")]

        class _FakeResp:
            status = 200

            def __init__(self):
                self.headers = _Headers()

            def read(self, _n):
                # NEVER EOF — a well-behaved upstream that keeps producing.
                # Without fix K the worker fills the bridge queue then blocks
                # forever on the semaphore (no consumer); with fix K it breaks
                # on stop_event at the loop top before ever reading.
                return b"x" * 64

            def close(self):
                closed["v"] = True

        class _FakeOpener:
            def open(self, _req, timeout=None):  # noqa: ANN001
                # simulate a pre-head stall: block until released
                release.wait(5)
                return _FakeResp()

        fwd = make_urllib_forwarder(base_url="http://upstream.local", opener=_FakeOpener())
        upstream = UpstreamRequest(
            request=RelayRequest(method="POST", path="/v1/messages", headers={}, body=b"{}"),
            token="tok",
        )

        async def _run():
            # cancel the forwarder's pre-head `await head_future` via the deadline
            with pytest.raises((TimeoutError, asyncio.CancelledError)):
                async with asyncio.timeout(0.3):
                    await fwd(upstream)
            # let the stalled opener return so the worker observes stop_event.
            # Poll while THIS loop is still alive (exiting asyncio.run first
            # would close the loop and mask the bug via the loop-closed path).
            release.set()
            for _ in range(100):
                if closed["v"]:
                    break
                await asyncio.sleep(0.05)

        asyncio.run(_run())
        assert closed["v"], "pre-head cancellation must lead the bridge thread to close upstream"
        # give the daemon thread a moment to fully unwind, then confirm it exited
        deadline = time.time() + 2.0
        while time.time() < deadline and any(
            t.name == "relay-upstream-forwarder" for t in threading.enumerate()
        ):
            time.sleep(0.05)
        assert "relay-upstream-forwarder" not in [
            t.name for t in threading.enumerate()
        ], "bridge thread must have exited"
