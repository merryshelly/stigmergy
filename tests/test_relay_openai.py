"""bead .147 — OpenAI-wire pass-through: bearer credential contract, path
composition, OpenAI usage metering (JSON + SSE), `stream_options` policy
enforcement, machine-readable deny markers, quota-governor JSONL feed, and the
plain-http / proxy-disabled upstream transport (spec §3 items 2-9).

Self-contained (own fakes + harness, the suite's convention — no cross-test
imports). DUMMY keys only. The Anthropic lane is covered by test_relay*.py;
this file exercises the OpenAI shapes and the §1C/§1D/§1E re-scoped
behaviour end to end (relay core `handle()` for the sync path, `serve_relay`
over a real unix socket for the transport path).
"""

from __future__ import annotations

import asyncio
import http.server
import json
import threading
import urllib.error
import urllib.request
from urllib.parse import urlsplit

import pytest

from stigmergy import relay_transport as _rt
from stigmergy.relay import (
    CapabilityStore,
    CredentialRelay,
    RelayError,
    RelayRequest,
    RelayResponse,
    UpstreamRequest,
    _openai_json_usage_extractor,
    extract_openai_usage,
    openai_output_tokens,
    parse_bearer_token,
)
from stigmergy.relay_transport import (
    UpstreamHead,
    _OpenAiSseUsageMeter,
    make_urllib_forwarder,
    serve_relay,
)

DUMMY_REAL_KEY = "[REDACTED:api_key]"
CAP_HEADER = "authorization"
OPENAI_ENDPOINTS = frozenset({("POST", "/chat/completions")})
OPENAI_ALLOWLIST = frozenset({"content-type", "accept"})

# The fixed-vocabulary deny reasons (spec §1E): the CapabilityDenied reasons,
# the transport's fixed reasons, and the .147 policy deny. Never free text.
DENY_REASON_VOCABULARY = frozenset(
    {
        "missing-capability",
        "unknown",
        "revoked",
        "quota-calls",
        "quota-tokens",
        "forbidden-endpoint",
        "malformed-request",
        "missing-usage-requested",
        "upstream-error",
        "handler-exception",
    }
)


# --------------------------------------------------------------------------- #
# Fakes (mirror the injected-callable discipline of test_relay*.py)           #
# --------------------------------------------------------------------------- #
class CountingKeyProvider:
    """Returns the (dummy) real key and counts fetches — a deny path must
    fetch ZERO times. `key=None` models a keyless (auth="none") lane."""

    def __init__(self, key: str | None = DUMMY_REAL_KEY) -> None:
        self._key = key
        self.calls = 0

    def __call__(self) -> str | None:
        self.calls += 1
        return self._key


class SyncForwarder:
    """The .12 sync forwarder for `CredentialRelay.handle` tests: captures
    the UpstreamRequest and returns a canned RelayResponse."""

    def __init__(self, response: RelayResponse | None = None) -> None:
        self._response = response or RelayResponse(status=200, headers={}, body=b"{}")
        self.requests: list[RelayRequest] = []

    def __call__(self, request: RelayRequest) -> RelayResponse:
        self.requests.append(request)
        return self._response


class AsyncFakeForwarder:
    """The streaming transport forwarder: captures UpstreamRequests and
    streams a canned response as (UpstreamHead, async chunk iterator)."""

    def __init__(
        self,
        *,
        status: int = 200,
        headers: list[tuple[str, str]] | None = None,
        chunks: list[bytes] | None = None,
    ) -> None:
        self.status = status
        self.headers = headers if headers is not None else [("content-type", "application/json")]
        self.chunks = chunks if chunks is not None else [b"{}"]
        self.requests: list[UpstreamRequest] = []

    async def __call__(self, upstream: UpstreamRequest) -> tuple[UpstreamHead, object]:
        self.requests.append(upstream)
        head = UpstreamHead(status=self.status, headers=list(self.headers))

        async def _gen():
            for chunk in self.chunks:
                yield chunk

        return head, _gen()


class NeverForwarder:
    """Fails the test if ever invoked (deny paths must never reach upstream)."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, upstream: UpstreamRequest):  # noqa: ANN201
        self.calls += 1
        raise AssertionError("forwarder invoked on a path that must never forward")


def _unused_sync_forwarder(req):  # noqa: ANN001, ANN202
    raise AssertionError("sync forwarder slot must never run in these tests")


def _openai_relay(
    store: CapabilityStore,
    *,
    kp: CountingKeyProvider | None = None,
    sync_fwd: SyncForwarder | None = None,
    pricing: str = "local",
    auth: str = "bearer",
    key: str | None = DUMMY_REAL_KEY,
) -> CredentialRelay:
    """An OpenAI-wire relay core (spec §1A/§1B): authorization capability
    header, bearer/none auth, pricing-class re-scope, the OpenAI JSON usage
    extractor, the `/chat/completions` endpoint allowlist, no version pin."""
    return CredentialRelay(
        store=store,
        key_provider=kp if kp is not None else CountingKeyProvider(key),
        forwarder=sync_fwd if sync_fwd is not None else _unused_sync_forwarder,
        usage_extractor=_openai_json_usage_extractor,
        capability_header=CAP_HEADER,
        upstream_header_allowlist=OPENAI_ALLOWLIST,
        allowed_endpoints=OPENAI_ENDPOINTS,
        auth=auth,
        pricing_class=pricing,
        wire="openai",
    )


def _openai_request(
    token: str,
    *,
    body: bytes = b'{"model":"m","messages":[]}',
    bearer: bool = True,
    path: str = "/chat/completions",
    extra_headers: dict[str, str] | None = None,
) -> RelayRequest:
    headers = {
        CAP_HEADER: f"Bearer {token}" if bearer else token,
        "content-type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    return RelayRequest(method="POST", path=path, headers=headers, body=body)


def _openai_body(payload: dict) -> bytes:
    return json.dumps(payload).encode("utf-8")


# =========================================================================== #
# §3.2 Bearer credential contract                                             #
# =========================================================================== #
class TestBearerContract:
    def test_parse_bearer_prefix_variants(self):
        # Optional prefix, case-insensitive scheme token.
        assert parse_bearer_token("Bearer abc.def-123") == "abc.def-123"
        assert parse_bearer_token("bearer abc.def-123") == "abc.def-123"
        assert parse_bearer_token("BEARER abc.def-123") == "abc.def-123"
        # No prefix: the bare token is a valid capability credential.
        assert parse_bearer_token("abc.def-123") == "abc.def-123"
        # A different scheme is NOT the bearer form — returned unchanged
        # (the store lookup then fails `unknown`, exactly like a bare
        # unknown token).
        assert parse_bearer_token("Basic abc") == "Basic abc"
        # No space after the scheme: not the bearer form.
        assert parse_bearer_token("Bearerabc") == "Bearerabc"
        # Stray padding around the token is stripped.
        assert parse_bearer_token("Bearer   abc  ") == "abc"

    def test_bearer_real_key_injected_upstream(self):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=1000, max_calls=5)
        fwd = SyncForwarder(RelayResponse(status=200, headers={}, body=b'{"usage":{}}'))
        relay = _openai_relay(store, kp=kp, sync_fwd=fwd, auth="bearer")

        # Worker sends `Authorization: Bearer <capability-token>`.
        relay.handle(_openai_request(cap.token))

        assert len(fwd.requests) == 1
        upstream = {k.lower(): v for k, v in fwd.requests[0].headers.items()}
        # The REAL key is injected as `Bearer <real-key>` upstream ...
        assert upstream[CAP_HEADER] == f"Bearer {DUMMY_REAL_KEY}"
        # ... and the capability token is NOT copied toward upstream.
        assert cap.token not in fwd.requests[0].headers.values()
        assert kp.calls == 1

    def test_bearer_prefix_optional_raw_token_authenticates(self):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=1000, max_calls=5)
        fwd = SyncForwarder(RelayResponse(status=200, headers={}, body=b'{"usage":{}}'))
        relay = _openai_relay(store, kp=kp, sync_fwd=fwd, auth="bearer")

        # A raw (prefix-less) capability token in the authorization header
        # authenticates identically (the prefix is OPTIONAL).
        resp = relay.handle(_openai_request(cap.token, bearer=False))
        assert resp.status == 200
        assert len(fwd.requests) == 1
        assert kp.calls == 1

    def test_auth_none_omits_header_upstream_entirely(self):
        store = CapabilityStore()
        cap = store.mint("d1", max_output_tokens=1000, max_calls=5)
        fwd = SyncForwarder(RelayResponse(status=200, headers={}, body=b'{"usage":{}}'))
        # key_provider returns None (keyless lane) — the header is OMITTED
        # upstream: never `Bearer None`, never a placeholder credential.
        relay = _openai_relay(store, kp=CountingKeyProvider(None), sync_fwd=fwd, auth="none")

        resp = relay.handle(_openai_request(cap.token))
        assert resp.status == 200
        assert len(fwd.requests) == 1
        upstream = fwd.requests[0]
        assert CAP_HEADER not in {k.lower() for k in upstream.headers}
        assert all("Bearer None" != v for v in upstream.headers.values())
        # method/path/body pass through unchanged.
        assert upstream.method == "POST"
        assert upstream.path == "/chat/completions"

    def test_auth_none_with_non_none_key_is_a_wiring_bug(self):
        store = CapabilityStore()
        cap = store.mint("d1", max_output_tokens=1000, max_calls=5)
        # A key_provider that returns a REAL key on a none-auth lane is a
        # wiring bug: prepare_upstream must raise RelayError (never silently
        # omit or inject). Use a LIVE token so authorize passes and the check
        # is actually reached.
        relay = _openai_relay(store, kp=CountingKeyProvider(DUMMY_REAL_KEY), auth="none")
        with pytest.raises(RelayError):
            relay.handle(_openai_request(cap.token))

    def test_capability_header_never_copied_toward_upstream(self):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=1000, max_calls=5)
        fwd = SyncForwarder(RelayResponse(status=200, headers={}, body=b'{"usage":{}}'))
        relay = _openai_relay(store, kp=kp, sync_fwd=fwd, auth="bearer")

        # The worker smuggles credentialing/routing headers: only the
        # allowlist (content-type/accept) + the injected credential may
        # reach upstream.
        relay.handle(
            _openai_request(
                cap.token,
                extra_headers={
                    "anthropic-beta": "smuggled",
                    "x-evil": "smuggled",
                    "accept": "application/json",
                    "host": "relay",
                },
            )
        )
        upstream = {k.lower() for k in fwd.requests[0].headers}
        assert upstream == {CAP_HEADER, "content-type", "accept"}
        values = {k.lower(): v for k, v in fwd.requests[0].headers.items()}
        assert values[CAP_HEADER] == f"Bearer {DUMMY_REAL_KEY}"
        assert cap.token not in values.values()


# =========================================================================== #
# §3.9 injected_secrets with a None key (sealing on keyless lanes)            #
# =========================================================================== #
class TestInjectedSecretsNoneKey:
    def test_none_key_yields_token_only(self):
        store = CapabilityStore()
        cap = store.mint("d1", max_output_tokens=1000, max_calls=5)
        relay = _openai_relay(store, kp=CountingKeyProvider(None), auth="none")

        secrets = relay.injected_secrets(cap.token)
        # Nothing else was ever put on the wire on a keyless lane.
        assert secrets == frozenset({cap.token})

    def test_sealing_path_works_on_keyless_lane(self, tmp_path):
        from stigmergy.records import RecordPlane
        from stigmergy.relay import build_redactor

        store = CapabilityStore()
        cap = store.mint("d1", max_output_tokens=1000, max_calls=5)
        relay = _openai_relay(store, kp=CountingKeyProvider(None), auth="none")
        store.revoke("d1")  # sealing happens AFTER dispatch end (post-revoke)

        secrets = relay.injected_secrets(cap.token)
        rp = RecordPlane(tmp_path / "records")
        ref = rp.seal_transcript(
            f"leak attempt: {cap.token}\nbenign line\n",
            redactor=build_redactor(secrets),
            must_not_contain=secrets,
        )
        stored = (rp.transcripts_dir / ref).read_text(encoding="utf-8")
        assert cap.token not in stored
        assert "benign line" in stored

    def test_bearer_lane_still_carries_real_key(self):
        store = CapabilityStore()
        cap = store.mint("d1", max_output_tokens=1000, max_calls=5)
        relay = _openai_relay(store, kp=CountingKeyProvider(), auth="bearer")
        assert relay.injected_secrets(cap.token) == frozenset({cap.token, DUMMY_REAL_KEY})


# =========================================================================== #
# §3.4 OpenAI JSON usage (sync `handle()` path) — tri-state charge            #
# =========================================================================== #
def _openai_json_response(completion_tokens: object | None) -> RelayResponse:
    usage: dict = {}
    if completion_tokens is not None:
        usage["completion_tokens"] = completion_tokens
    return RelayResponse(status=200, headers={}, body=json.dumps({"usage": usage}).encode())


class TestOpenAiJsonUsage:
    def test_charged_completion_tokens(self):
        store = CapabilityStore()
        cap = store.mint("d1", max_output_tokens=10000, max_calls=5)
        fwd = SyncForwarder(_openai_json_response(37))
        relay = _openai_relay(store, sync_fwd=fwd, pricing="local")

        resp = relay.handle(_openai_request(cap.token))
        assert resp.status == 200
        assert store.usage(cap.token) == {"output_tokens": 37, "calls": 1}
        # The response's observability slot carries the charged usage.
        assert resp.usage is not None
        assert resp.usage.get("output_tokens") == 37

    @pytest.mark.parametrize("pricing", ["metered", "local"])
    def test_undetermined_by_pricing_class(self, pricing):
        store = CapabilityStore()
        cap = store.mint("d1", max_output_tokens=10000, max_calls=5)
        relay = _openai_relay(
            store, sync_fwd=SyncForwarder(RelayResponse(status=200, headers={}, body=b"{}")),
            pricing=pricing,
        )

        # No usage at all in the body -> undetermined (not a measurement).
        relay.handle(_openai_request(cap.token))

        if pricing == "metered":
            # Fail-closed: saturate (kill switch), never ~0.
            assert store.usage(cap.token)["output_tokens"] == 10000
            assert store.is_live(cap.token) is False
        else:
            # Local: accounting-only 0 — a $0 capability is never saturated.
            assert store.usage(cap.token) == {"output_tokens": 0, "calls": 1}
            assert store.is_live(cap.token) is True

    @pytest.mark.parametrize(
        "bad_value",
        ["37", -3, True, None, [37]],  # non-int / negative / bool / null / list
        ids=["str", "negative", "bool", "null", "list"],
    )
    @pytest.mark.parametrize("pricing", ["metered", "local"])
    def test_invalid_completion_tokens_is_undetermined(self, pricing, bad_value):
        store = CapabilityStore()
        cap = store.mint("d1", max_output_tokens=10000, max_calls=5)
        fwd = SyncForwarder(_openai_json_response(bad_value))
        relay = _openai_relay(store, sync_fwd=fwd, pricing=pricing)

        relay.handle(_openai_request(cap.token))

        usage = store.usage(cap.token)
        if pricing == "metered":
            assert usage["output_tokens"] == 10000  # saturated
        else:
            assert usage == {"output_tokens": 0, "calls": 1}

    def test_zero_completion_tokens_is_a_valid_measurement(self):
        # 0 is a valid non-negative int — it charges exactly 0 (and counts
        # the call), it is NOT undetermined.
        store = CapabilityStore()
        cap = store.mint("d1", max_output_tokens=10000, max_calls=5)
        fwd = SyncForwarder(_openai_json_response(0))
        relay = _openai_relay(store, sync_fwd=fwd, pricing="local")

        relay.handle(_openai_request(cap.token))
        assert store.usage(cap.token) == {"output_tokens": 0, "calls": 1}
        assert store.is_live(cap.token) is True

    def test_openai_output_tokens_tristate_helper(self):
        assert openai_output_tokens({"completion_tokens": 5}) == 5
        assert openai_output_tokens({"completion_tokens": 0}) == 0
        assert openai_output_tokens({}) is None
        assert openai_output_tokens({"completion_tokens": -1}) is None
        assert openai_output_tokens({"completion_tokens": "5"}) is None
        assert openai_output_tokens({"completion_tokens": True}) is None
        assert openai_output_tokens(None) is None
        assert openai_output_tokens("not a dict") is None

    def test_extract_openai_usage_never_raises(self):
        resp = RelayResponse(status=200, headers={}, body=b'{"usage":{"completion_tokens":9}}')
        assert extract_openai_usage(resp) == {"completion_tokens": 9}
        assert extract_openai_usage(RelayResponse(status=200, headers={}, body=b"junk")) == {}
        assert extract_openai_usage(RelayResponse(status=200, headers={}, body=b"")) == {}
        assert extract_openai_usage(RelayResponse(status=200, headers={}, body=b'[]')) == {}


# =========================================================================== #
# §3.5 OpenAI SSE meter — unit level (_OpenAiSseUsageMeter)                   #
# =========================================================================== #
def _sse(*chunks: bytes) -> bytes:
    return b"".join(chunks)


CONTENT_CHUNK = (
    b'data: {"id":"1","object":"chat.completion.chunk",'
    b'"choices":[{"index":0,"delta":{"content":"hi"}}]}\n\n'
)


def _usage_chunk(payload: dict) -> bytes:
    return (b'data: ' + json.dumps({"choices": [], "usage": payload}).encode() + b"\n\n")


USAGE_TOP_REASONING = _usage_chunk(
    {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
        # Synthetic/SGLang shape: top-level reasoning_tokens.
        "reasoning_tokens": 3,
        "prompt_tokens_details": {"cached_tokens": 5},
    }
)
USAGE_NESTED_REASONING = _usage_chunk(
    {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
        # OpenAI-standard shape: nested completion_tokens_details.
        "completion_tokens_details": {"reasoning_tokens": 9},
    }
)
USAGE_BOTH_SHAPES = _usage_chunk(
    {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "reasoning_tokens": 3,
        "completion_tokens_details": {"reasoning_tokens": 9},
    }
)
DONE = b"data: [DONE]\n\n"


class TestOpenAiSseMeterUnit:
    def test_final_usage_chunk_top_level_reasoning(self):
        meter = _OpenAiSseUsageMeter()
        meter.feed(CONTENT_CHUNK)
        meter.feed(USAGE_TOP_REASONING)
        meter.feed(DONE)
        usage = meter.finalize()
        assert usage is not None
        assert usage["completion_tokens"] == 7
        # The 4-key CV/quota-governor feed (spec §1C), both reasoning shapes.
        snap = meter.usage_snapshot()
        assert snap["prompt_tokens"] == 11
        assert snap["completion_tokens"] == 7
        assert snap["reasoning_tokens"] == 3
        assert snap["cached_tokens"] == 5

    def test_final_usage_chunk_nested_reasoning(self):
        meter = _OpenAiSseUsageMeter()
        meter.feed(CONTENT_CHUNK)
        meter.feed(USAGE_NESTED_REASONING)
        meter.feed(DONE)
        assert meter.finalize() is not None
        assert meter.usage_snapshot()["reasoning_tokens"] == 9

    def test_both_reasoning_shapes_top_level_wins(self):
        meter = _OpenAiSseUsageMeter()
        meter.feed(USAGE_BOTH_SHAPES)
        meter.finalize()
        # Both shapes present: top-level `usage.reasoning_tokens` wins.
        assert meter.usage_snapshot()["reasoning_tokens"] == 3

    def test_done_sentinel_is_informational_only(self):
        # [DONE] carries no usage and must not corrupt the meter; the usage
        # chunk after it (or before it) is what counts.
        meter = _OpenAiSseUsageMeter()
        meter.feed(DONE)
        assert meter.finalize() is None  # no usage at all -> undetermined
        meter2 = _OpenAiSseUsageMeter()
        meter2.feed(USAGE_TOP_REASONING)
        meter2.feed(DONE)
        assert meter2.finalize()["completion_tokens"] == 7

    def test_no_usage_chunk_finalizes_undetermined(self):
        meter = _OpenAiSseUsageMeter()
        meter.feed(CONTENT_CHUNK)
        meter.feed(b'data: {"choices":[{"delta":{"content":"more"}}]}\n\n')
        meter.feed(DONE)
        # A clean stream that never carried a usage event is UNDETERMINED
        # (the caller applies the pricing-class re-scope), never 0-by-default.
        assert meter.finalize() is None

    def test_negative_completion_tokens_sticky_undetermined(self):
        meter = _OpenAiSseUsageMeter()
        meter.feed(_usage_chunk({"prompt_tokens": 1, "completion_tokens": -5}))
        assert meter.finalize() is None

    def test_non_dict_usage_is_undetermined(self):
        # `usage: null` is a present-but-invalid measurement, not "no opinion".
        meter = _OpenAiSseUsageMeter()
        meter.feed(b'data: {"choices":[],"usage":null}\n\n')
        assert meter.finalize() is None

    def test_usage_event_split_across_chunk_boundary(self):
        # The event is split mid-JSON across two feeds — cross-chunk
        # tolerance (same discipline as the Anthropic meter).
        chunk = USAGE_TOP_REASONING
        cut = len(chunk) // 2
        meter = _OpenAiSseUsageMeter()
        meter.feed(chunk[:cut])
        meter.feed(chunk[cut:])
        assert meter.finalize()["completion_tokens"] == 7

    def test_later_usage_event_replaces_earlier(self):
        meter = _OpenAiSseUsageMeter()
        meter.feed(_usage_chunk({"prompt_tokens": 1, "completion_tokens": 2}))
        meter.feed(_usage_chunk({"prompt_tokens": 1, "completion_tokens": 8}))
        assert meter.finalize()["completion_tokens"] == 8


# =========================================================================== #
# §3.3 Path composition — both live upstream URLs, netloc pin still enforced  #
# =========================================================================== #
class CapturingOpener:
    """Records the FULL URL of every opener.open call, then raises (never
    really connects) — we assert on the composed URL, not the response."""

    def __init__(self) -> None:
        self.urls: list[str] = []

    def open(self, request, timeout=None):  # noqa: ANN001, ANN201
        self.urls.append(request.full_url)
        raise urllib.error.URLError("captured (test opener never connects upstream)")


async def _drive_forwarder(forwarder, upstream: UpstreamRequest) -> None:
    """Run a forwarder, swallowing the expected UpstreamError (the capturing
    opener always raises) or collecting the response if it succeeds."""
    # NB: resolve UpstreamError through the module object, NOT a from-import.
    # tests/test_relay_transport.py's proxy-env test importlib.reload()s this
    # module mid-session, REBINDING its class attributes — a from-imported
    # UpstreamError would be a stale class object and the except below would
    # miss (cross-file pollution, observed 2026-08-30). Module-attr lookup
    # always sees the class the current forwarder actually raises.
    try:
        head, gen = await forwarder(upstream)
        async for _ in gen:
            pass
    except _rt.UpstreamError:
        pass


def _upstream(path: str, body: bytes = b'{"model":"m"}') -> UpstreamRequest:
    return UpstreamRequest(
        request=RelayRequest(
            method="POST", path=path, headers={"content-type": "application/json"}, body=body
        ),
        token="tok",
    )


class TestPathComposition:
    # The two live upstream classes (spec §1A) — pinned as test constants.
    @pytest.mark.parametrize(
        "base_url",
        [
            "http://10.0.20.111:8000/v1",  # blackwell (LAN http, no auth)
            "https://api.synthetic.new/openai/v1",  # Synthetic (bearer, metered)
        ],
        ids=["blackwell", "synthetic"],
    )
    def test_composes_exactly_to_live_url(self, base_url):
        opener = CapturingOpener()
        fwd = make_urllib_forwarder(base_url=base_url, opener=opener)
        asyncio.run(_drive_forwarder(fwd, _upstream("/chat/completions")))
        # NO rewrite mechanism: `base_url + path` is exact for BOTH live
        # upstreams (the base carries /v1, the worker path does not).
        assert opener.urls == [base_url + "/chat/completions"]

    def test_netloc_pin_still_enforced_for_openai_bases(self):
        # The two live OpenAI bases carry a path prefix (/v1, /openai/v1), so
        # a legit `/chat/completions` composes to EXACTLY the pinned netloc
        # (the load-bearing security property for these bases — composition
        # cannot steer the connection off-host).
        for base_url in ("http://10.0.20.111:8000/v1", "https://api.synthetic.new/openai/v1"):
            expected_netloc = urlsplit(base_url).netloc
            opener = CapturingOpener()
            fwd = make_urllib_forwarder(base_url=base_url, opener=opener)
            asyncio.run(_drive_forwarder(fwd, _upstream("/chat/completions")))
            assert all(
                urlsplit(u).netloc == expected_netloc for u in opener.urls
            ), (f"legit path composed off the pinned netloc for {base_url}: {opener.urls}")

        # The guard is the SAME code path for every profile and MUST still fire
        # on a netloc shift (proving it is live, not dead code — a prefixed
        # base structurally can't be steered off-host by concatenation, so we
        # exercise the guard with a no-prefix base the way the .55 hardening
        # suite does: a host-extending path changes the composed netloc).
        no_prefix_base = "https://api.synthetic.new"  # no /openai/v1 prefix
        opener = CapturingOpener()
        fwd = make_urllib_forwarder(base_url=no_prefix_base, opener=opener)
        asyncio.run(_drive_forwarder(fwd, _upstream(".evil.com/chat/completions")))
        # THE security property: a netloc-shifting path never reaches
        # `opener.open` — the netloc pin refuses it before any connection is
        # opened (if the pin were removed, the evil netloc WOULD be recorded
        # here, turning this assertion RED).
        assert opener.urls == [], (
            f"key-bearing request opened against a non-pinned netloc: {opener.urls}"
        )


# =========================================================================== #
# §3.8 Plain-http upstream through the PRODUCTION opener (proxy env set)      #
# =========================================================================== #
class TestPlainHttpUpstreamProductionOpener:
    def test_direct_http_call_wins_over_inherited_proxy_env(self, monkeypatch):
        # A real local http server (the blackwell class is LAN http). With
        # HTTP_PROXY/HTTPS_PROXY set to a DEAD proxy in the test env, the
        # production opener's explicit ProxyHandler({}) must still connect
        # DIRECT: any proxy honoring would fail (nothing listens on :9).
        class _EchoHandler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("content-length") or 0)
                body = self.rfile.read(length)
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):  # silence
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _EchoHandler)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
            monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
            monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
            monkeypatch.setenv("https_proxy", "http://127.0.0.1:9")

            # NON-VACUOUS contrast: a fresh UNguarded opener built right now
            # WOULD honor the env proxy (proves the env is in effect).
            unguarded = urllib.request.build_opener()
            assert any(
                isinstance(h, urllib.request.ProxyHandler) and h.proxies
                for h in unguarded.handlers
            ), "sanity: unguarded opener picks up HTTP_PROXY (proves the env matters)"

            # The production forwarder (default opener = _NO_REDIRECT_OPENER,
            # with the explicit empty ProxyHandler) connects DIRECT.
            fwd = make_urllib_forwarder(base_url=f"http://127.0.0.1:{port}/v1")
            payload = b'{"echo":true}'

            async def _run():
                head, gen = await fwd(_upstream("/chat/completions", body=payload))
                body = b""
                async for chunk in gen:
                    body += chunk
                return head, body

            head, body = asyncio.run(_run())
            assert head.status == 200
            assert body == payload  # plain http round-trips unchanged
        finally:
            server.shutdown()
            server.server_close()


# =========================================================================== #
# §1E — upstream non-2xx through the PRODUCTION forwarder (real HTTP, no      #
# fakes): a 4xx/5xx is a RESPONSE (forwarded verbatim, 429 body logged), a   #
# 3xx redirect is a FAILURE (UpstreamError — the key must not ride a re-send) #
# =========================================================================== #
class TestUpstreamNon2xxThroughProductionForwarder:
    def _fake_upstream(self, status: int, body: bytes, handler_class=None):
        if handler_class is None:
            class _Handler(http.server.BaseHTTPRequestHandler):
                def do_POST(self):  # noqa: N802
                    self.rfile.read(int(self.headers.get("content-length") or 0))
                    self.send_response(status)
                    self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                def log_message(self, *args):  # silence
                    pass

            handler_class = _Handler
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_class)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server, f"http://127.0.0.1:{port}/v1"

    def test_upstream_429_is_a_real_response_not_upstream_error(self):
        # spec §1E: "Upstream non-2xx bodies are forwarded verbatim (unchanged)"
        # and the 429 body is logged to the relay JSONL — impossible if the
        # forwarder converts the 429 into a synthesized 502 upstream-error.
        # Regression for the production-opener bug: urllib raises HTTPError
        # for 4xx/5xx (the _NoRedirectHandler chain carries HTTPError
        # processing); the forwarder must treat a 4xx/5xx HTTPError as the
        # REAL response, not a transport failure.
        flat_429 = b'{"error":"quota exhausted"}'
        server, base_url = self._fake_upstream(429, flat_429)
        try:
            fwd = make_urllib_forwarder(base_url=base_url)

            async def _run():
                head, gen = await fwd(
                    UpstreamRequest(
                        request=RelayRequest(
                            method="POST",
                            path="/chat/completions",
                            headers={"content-type": "application/json"},
                            body=b'{"model":"m"}',
                        ),
                        token="tok",
                    )
                )
                body = b""
                async for chunk in gen:
                    body += chunk
                return head, body

            head, body = asyncio.run(_run())
            # The 429 (not a 502 upstream-error) reached the transport with
            # its REAL status and its VERBATIM body.
            assert head.status == 429, f"429 folded into an error: status={head.status}"
            assert body == flat_429, f"body not verbatim: {body!r}"
        finally:
            server.shutdown()
            server.server_close()

    def test_upstream_400_is_a_real_response_too(self):
        server, base_url = self._fake_upstream(400, b'{"error":"bad request"}')
        try:
            fwd = make_urllib_forwarder(base_url=base_url)

            async def _run():
                head, gen = await fwd(
                    UpstreamRequest(
                        request=RelayRequest(
                            method="POST",
                            path="/chat/completions",
                            headers={"content-type": "application/json"},
                            body=b'{"model":"m"}',
                        ),
                        token="tok",
                    )
                )
                body = b""
                async for chunk in gen:
                    body += chunk
                return head, body

            head, body = asyncio.run(_run())
            assert head.status == 400
            assert body == b'{"error":"bad request"}'
        finally:
            server.shutdown()
            server.server_close()

    def test_redirect_still_a_failure_not_a_response(self):
        # The 3xx refusal is UNCHANGED by the non-2xx fix: a redirect is not
        # a terminal response (the key-bearing request must never be
        # re-sent), so it stays an UpstreamError (the .55/.36 guard).
        class _Redirecting(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                self.rfile.read(int(self.headers.get("content-length") or 0))
                self.send_response(302)
                self.send_header("location", "http://127.0.0.1:1/steal")
                self.send_header("content-length", "0")
                self.end_headers()

            def log_message(self, *args):  # silence
                pass

        server, base_url = self._fake_upstream(200, b"", handler_class=_Redirecting)
        try:
            fwd = make_urllib_forwarder(base_url=base_url)

            async def _run():
                await fwd(
                    UpstreamRequest(
                        request=RelayRequest(
                            method="POST",
                            path="/chat/completions",
                            headers={"content-type": "application/json"},
                            body=b'{"model":"m"}',
                        ),
                        token="tok",
                    )
                )

            with pytest.raises(_rt.UpstreamError):
                asyncio.run(_run())
        finally:
            server.shutdown()
            server.server_close()


# =========================================================================== #
# Transport end-to-end harness (unix socket, raw HTTP)                        #
# =========================================================================== #
def _openai_raw(
    token: str | None,
    *,
    body: bytes = b'{"model":"m","messages":[]}',
    bearer: bool = True,
    path: str = "/chat/completions",
    extra_lines: list[str] | None = None,
) -> bytes:
    head = [f"POST {path} HTTP/1.1", "host: relay"]
    if token is not None:
        head.append(f"{CAP_HEADER}: {('Bearer ' + token) if bearer else token}")
    head.append("content-type: application/json")
    head.append(f"content-length: {len(body)}")
    if extra_lines:
        head.extend(extra_lines)
    return ("\r\n".join(head) + "\r\n\r\n").encode("latin-1") + body


def _serve_openai_roundtrip(
    tmp_path,
    *,
    relay: CredentialRelay,
    forwarder,
    raw: bytes,
    socket_name: str = "relay.sock",
    log_name: str = "relay.jsonl",
) -> tuple[bytes, object]:
    socket_path = tmp_path / socket_name
    log_path = tmp_path / log_name

    async def _run():
        server = await serve_relay(
            socket_path, relay, forwarder=forwarder, log_path=log_path, dispatch_id="d1"
        )
        try:
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
        finally:
            server.close()
            await server.wait_closed()

    return asyncio.run(_run()), log_path


def _split_response(raw: bytes) -> tuple[bytes, bytes]:
    head, _, body = raw.partition(b"\r\n\r\n")
    return head, body


def _log_lines(log_path) -> list[dict]:
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines]


def _deny_reason_from_head(head: bytes) -> str | None:
    for line in head.decode("latin-1").split("\r\n")[1:]:
        name, sep, value = line.partition(":")
        if sep and name.strip().lower() == "x-stigmergy-deny-reason":
            return value.strip()
    return None


# =========================================================================== #
# §3.6 stream_options enforcement (spec §1D) — metered OpenAI lanes           #
# =========================================================================== #
class TestStreamOptionsEnforcement:
    def _relay(self, store, *, kp, pricing="metered", auth="bearer"):
        return _openai_relay(store, kp=kp, pricing=pricing, auth=auth)

    def test_metered_stream_without_flag_denies_422_nothing_charged(self, tmp_path):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=10000, max_calls=5)
        never = NeverForwarder()
        relay = self._relay(store, kp=kp, pricing="metered")

        body = _openai_body({"model": "m", "stream": True})  # NO stream_options
        resp, log_path = _serve_openai_roundtrip(
            tmp_path, relay=relay, forwarder=never, raw=_openai_raw(cap.token, body=body)
        )
        head, body_bytes = _split_response(resp)
        assert head.startswith(b"HTTP/1.1 422")
        assert _deny_reason_from_head(head) == "missing-usage-requested"
        # The forwarder was NEVER called and the key was NEVER fetched ...
        assert never.calls == 0
        assert kp.calls == 0
        # ... and NOTHING was charged (no reserve, no reconcile): the
        # capability is completely untouched by the policy deny.
        assert store.usage(cap.token) == {"output_tokens": 0, "calls": 0}
        assert store.is_live(cap.token) is True
        line = _log_lines(log_path)[0]
        assert line["status"] == 422
        assert line["decision"] == "deny"
        assert line["reason"] == "missing-usage-requested"
        assert line["wire"] == "openai"
        assert line["pricing_class"] == "metered"

    def test_metered_stream_with_flag_forwards(self, tmp_path):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=10000, max_calls=5)
        fwd = AsyncFakeForwarder(
            headers=[("content-type", "text/event-stream")],
            chunks=[CONTENT_CHUNK, USAGE_TOP_REASONING, DONE],
        )
        relay = self._relay(store, kp=kp, pricing="metered")

        body = _openai_body(
            {"model": "m", "stream": True, "stream_options": {"include_usage": True}}
        )
        resp, _ = _serve_openai_roundtrip(
            tmp_path, relay=relay, forwarder=fwd, raw=_openai_raw(cap.token, body=body)
        )
        head, body_bytes = _split_response(resp)
        assert head.startswith(b"HTTP/1.1 200")
        assert body_bytes == CONTENT_CHUNK + USAGE_TOP_REASONING + DONE
        assert len(fwd.requests) == 1
        # Metered exactly the final usage chunk.
        assert store.usage(cap.token) == {"output_tokens": 7, "calls": 1}
        # The real bearer key rode the forwarded request.
        up = {k.lower(): v for k, v in fwd.requests[0].request.headers.items()}
        assert up[CAP_HEADER] == f"Bearer {DUMMY_REAL_KEY}"
        assert cap.token not in up.values()

    def test_metered_nonstream_without_flag_forwards(self, tmp_path):
        # The flag is a STREAMING requirement: a non-stream request (JSON
        # usage in the 200 body) never needs stream_options.
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=10000, max_calls=5)
        payload = b'{"usage":{"prompt_tokens":4,"completion_tokens":9}}'
        fwd = AsyncFakeForwarder(chunks=[payload])
        relay = self._relay(store, kp=kp, pricing="metered")

        body = _openai_body({"model": "m", "stream": False})
        resp, _ = _serve_openai_roundtrip(
            tmp_path, relay=relay, forwarder=fwd, raw=_openai_raw(cap.token, body=body)
        )
        head, body_bytes = _split_response(resp)
        assert head.startswith(b"HTTP/1.1 200")
        assert body_bytes == payload
        assert store.usage(cap.token) == {"output_tokens": 9, "calls": 1}

    def test_local_stream_without_flag_forwards(self, tmp_path):
        # `local` lanes: NO requirement (accounting-only metering tolerates a
        # missing usage event) — no 422, the stream goes through.
        store = CapabilityStore()
        kp = CountingKeyProvider(None)
        cap = store.mint("d1", max_output_tokens=10000, max_calls=5)
        fwd = AsyncFakeForwarder(
            headers=[("content-type", "text/event-stream")], chunks=[CONTENT_CHUNK, DONE]
        )
        relay = self._relay(store, kp=kp, pricing="local", auth="none")

        body = _openai_body({"model": "m", "stream": True})  # no flag, allowed for local
        resp, _ = _serve_openai_roundtrip(
            tmp_path, relay=relay, forwarder=fwd, raw=_openai_raw(cap.token, body=body)
        )
        head, _ = _split_response(resp)
        assert head.startswith(b"HTTP/1.1 200")
        assert len(fwd.requests) == 1
        # No usage chunk arrived -> local accounting-only 0 (call still counts).
        assert store.usage(cap.token) == {"output_tokens": 0, "calls": 1}

    def test_anthropic_wire_gets_no_body_inspection(self, tmp_path):
        # `wire="anthropic"`: no body inspection AT ALL — a streaming body
        # without stream_options must forward (today's byte-for-byte shape).
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=10000, max_calls=5)
        fwd = AsyncFakeForwarder(
            headers=[("content-type", "text/event-stream")],
            chunks=[b'event: message_delta\ndata: {"usage":{"output_tokens":5}}\n\n'],
        )
        relay = CredentialRelay(
            store=store,
            key_provider=kp,
            forwarder=_unused_sync_forwarder,
            capability_header="x-api-key",
        )  # defaults: anthropic wire, metered, x-api-key

        body = b'{"model":"claude","messages":[],"stream":true}'  # no stream_options
        raw = (
            f"POST /v1/messages HTTP/1.1\r\nhost: relay\r\nx-api-key: {cap.token}\r\n"
            f"content-type: application/json\r\ncontent-length: {len(body)}\r\n\r\n"
        ).encode("latin-1") + body
        resp, _ = _serve_openai_roundtrip(
            tmp_path, relay=relay, forwarder=fwd, raw=raw, socket_name="anthropic.sock"
        )
        head, _ = _split_response(resp)
        assert head.startswith(b"HTTP/1.1 200")
        assert len(fwd.requests) == 1
        assert store.usage(cap.token) == {"output_tokens": 5, "calls": 1}

    def test_metered_unparseable_body_denies_422_never_guess(self, tmp_path):
        # A metered OpenAI lane cannot verify the flag against an unparseable
        # body — "never guess" (spec §1D): deny, do not forward.
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=10000, max_calls=5)
        never = NeverForwarder()
        relay = self._relay(store, kp=kp, pricing="metered")

        resp, _ = _serve_openai_roundtrip(
            tmp_path, relay=relay, forwarder=never, raw=_openai_raw(cap.token, body=b"{not json")
        )
        head, _ = _split_response(resp)
        assert head.startswith(b"HTTP/1.1 422")
        assert _deny_reason_from_head(head) == "missing-usage-requested"
        assert never.calls == 0
        assert kp.calls == 0
        assert store.usage(cap.token) == {"output_tokens": 0, "calls": 0}

    def test_metered_revoked_token_falls_to_canonical_401_not_422(self, tmp_path):
        # §1D ordering: a DEAD token falls through to the canonical auth
        # deny — the stream_options check only applies to a LIVE token, so a
        # revoked capability is 401 `revoked`, never 422.
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=10000, max_calls=5)
        store.revoke("d1")
        never = NeverForwarder()
        relay = self._relay(store, kp=kp, pricing="metered")

        body = _openai_body({"model": "m", "stream": True})  # no flag
        resp, _ = _serve_openai_roundtrip(
            tmp_path, relay=relay, forwarder=never, raw=_openai_raw(cap.token, body=body)
        )
        head, _ = _split_response(resp)
        assert head.startswith(b"HTTP/1.1 401")
        assert _deny_reason_from_head(head) == "revoked"
        assert never.calls == 0
        assert kp.calls == 0

    def test_metered_unknown_token_falls_to_canonical_401(self, tmp_path):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        never = NeverForwarder()
        relay = self._relay(store, kp=kp, pricing="metered")

        body = _openai_body({"model": "m", "stream": True})
        resp, _ = _serve_openai_roundtrip(
            tmp_path,
            relay=relay,
            forwarder=never,
            raw=_openai_raw("totally-bogus-token", body=body),
        )
        head, _ = _split_response(resp)
        assert head.startswith(b"HTTP/1.1 401")
        assert _deny_reason_from_head(head) == "unknown"
        assert kp.calls == 0

    def test_metered_quota_exhausted_falls_to_canonical_429(self, tmp_path):
        # A quota-exhausted (but once-valid) token is a canonical 429, not a
        # 422 policy deny — the pre-check must distinguish live vs dead.
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=100, max_calls=5)
        store.charge(cap.token, output_tokens=100, calls=0)  # exhaust tokens
        never = NeverForwarder()
        relay = self._relay(store, kp=kp, pricing="metered")

        body = _openai_body({"model": "m", "stream": True})
        resp, _ = _serve_openai_roundtrip(
            tmp_path, relay=relay, forwarder=never, raw=_openai_raw(cap.token, body=body)
        )
        head, _ = _split_response(resp)
        assert head.startswith(b"HTTP/1.1 429")
        assert _deny_reason_from_head(head) == "quota-tokens"
        assert never.calls == 0
        assert kp.calls == 0


# =========================================================================== #
# §3.7 Deny markers (spec §1E) — body + header on every relay deny            #
# =========================================================================== #
class TestDenyMarkers:
    def _deny_case(self, tmp_path, name, *, raw_builder, relay, forwarder, status, reason):
        resp, log_path = _serve_openai_roundtrip(
            tmp_path,
            relay=relay,
            forwarder=forwarder,
            raw=raw_builder(),
            socket_name=f"{name}.sock",
            log_name=f"{name}.jsonl",
        )
        head, body = _split_response(resp)
        assert head.startswith(f"HTTP/1.1 {status}".encode()), f"{name}: {head!r}"
        # The x-stigmergy-deny-reason header carries the fixed-vocabulary reason.
        assert _deny_reason_from_head(head) == reason, f"{name}: {head!r}"
        assert reason in DENY_REASON_VOCABULARY
        # The machine-readable JSON body (additive, status-preserving).
        parsed = json.loads(body)
        assert parsed == {"error": {"type": "stigmergy_relay_deny", "reason": reason}}
        # The JSONL line carries the same reason + status.
        line = _log_lines(log_path)[0]
        assert line["decision"] == "deny"
        assert line["reason"] == reason
        assert line["status"] == status
        assert line["wire"] == "openai"

    def _openai_relay(self, store, kp, **cfg):
        return _openai_relay(store, kp=kp, **cfg)

    def test_unknown_token_401(self, tmp_path):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        never = NeverForwarder()
        self._deny_case(
            tmp_path,
            "unknown",
            raw_builder=lambda: _openai_raw("bogus", body=b"{}"),
            relay=self._openai_relay(store, kp),
            forwarder=never,
            status=401,
            reason="unknown",
        )
        assert never.calls == 0
        assert kp.calls == 0

    def test_revoked_token_401(self, tmp_path):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=100, max_calls=5)
        store.revoke("d1")
        never = NeverForwarder()
        self._deny_case(
            tmp_path,
            "revoked",
            raw_builder=lambda: _openai_raw(cap.token),
            relay=self._openai_relay(store, kp),
            forwarder=never,
            status=401,
            reason="revoked",
        )
        assert never.calls == 0
        assert kp.calls == 0

    def test_missing_capability_header_401(self, tmp_path):
        store = CapabilityStore()
        store.mint("d1", max_output_tokens=100, max_calls=5)
        kp = CountingKeyProvider()
        never = NeverForwarder()
        self._deny_case(
            tmp_path,
            "missing",
            raw_builder=lambda: _openai_raw(None, body=b"{}"),  # no authorization header
            relay=self._openai_relay(store, kp),
            forwarder=never,
            status=401,
            reason="missing-capability",
        )
        assert never.calls == 0
        assert kp.calls == 0

    def test_forbidden_endpoint_403(self, tmp_path):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=100, max_calls=5)
        never = NeverForwarder()
        self._deny_case(
            tmp_path,
            "forbidden",
            raw_builder=lambda: _openai_raw(cap.token, path="/v1/models"),
            relay=self._openai_relay(store, kp),
            forwarder=never,
            status=403,
            reason="forbidden-endpoint",
        )
        assert never.calls == 0
        assert kp.calls == 0

    def test_quota_calls_429(self, tmp_path):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=1000, max_calls=1)
        store.charge(cap.token, output_tokens=0, calls=1)  # exhaust calls
        never = NeverForwarder()
        self._deny_case(
            tmp_path,
            "quotacalls",
            raw_builder=lambda: _openai_raw(cap.token),
            relay=self._openai_relay(store, kp),
            forwarder=never,
            status=429,
            reason="quota-calls",
        )
        assert never.calls == 0
        assert kp.calls == 0

    def test_quota_tokens_429(self, tmp_path):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=10, max_calls=5)
        store.charge(cap.token, output_tokens=10, calls=0)  # exhaust tokens
        never = NeverForwarder()
        self._deny_case(
            tmp_path,
            "quotatokens",
            raw_builder=lambda: _openai_raw(cap.token),
            relay=self._openai_relay(store, kp),
            forwarder=never,
            status=429,
            reason="quota-tokens",
        )
        assert never.calls == 0
        assert kp.calls == 0

    def test_missing_usage_422(self, tmp_path):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=1000, max_calls=5)
        never = NeverForwarder()
        self._deny_case(
            tmp_path,
            "missingusage",
            raw_builder=lambda: _openai_raw(cap.token, body=_openai_body({"stream": True})),
            relay=self._openai_relay(store, kp, pricing="metered"),
            forwarder=never,
            status=422,
            reason="missing-usage-requested",
        )
        assert never.calls == 0
        assert kp.calls == 0
        # Nothing charged on the policy deny.
        assert store.usage(cap.token) == {"output_tokens": 0, "calls": 0}


# =========================================================================== #
# §1C/§1E JSONL feed — wire / pricing_class / usage / quota captures          #
# =========================================================================== #
class TestJsonlFeed:
    def test_json_line_carries_profile_fields_and_openai_usage(self, tmp_path):
        store = CapabilityStore()
        kp = CountingKeyProvider(None)
        cap = store.mint("d1", max_output_tokens=10000, max_calls=5)
        payload = (
            b'{"usage":{"prompt_tokens":12,"completion_tokens":34,'
            b'"prompt_tokens_details":{"cached_tokens":8}}}'
        )
        fwd = AsyncFakeForwarder(chunks=[payload])
        relay = _openai_relay(store, kp=kp, pricing="local", auth="none")

        body = _openai_body({"model": "m"})
        _serve_openai_roundtrip(
            tmp_path, relay=relay, forwarder=fwd, raw=_openai_raw(cap.token, body=body)
        )
        line = _log_lines(tmp_path / "relay.jsonl")[0]
        # Profile fields explicit on EVERY line (spec §1E).
        assert line["wire"] == "openai"
        assert line["pricing_class"] == "local"
        assert line["decision"] == "allow"
        assert line["status"] == 200
        assert line["reason"] == "forwarded"
        # The full parsed usage dict is the CV/quota-governor feed.
        assert line["usage"] == {
            "prompt_tokens": 12,
            "completion_tokens": 34,
            "prompt_tokens_details": {"cached_tokens": 8},
        }
        # Quota-governor captures present (absent upstream signals -> null).
        assert line["synthetic_quotas"] is None
        assert line["upstream_429_body"] is None

    def test_sse_usage_logged_as_full_parsed_dict(self, tmp_path):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=10000, max_calls=5)
        fwd = AsyncFakeForwarder(
            headers=[("content-type", "text/event-stream")],
            chunks=[CONTENT_CHUNK, USAGE_TOP_REASONING, DONE],
        )
        relay = _openai_relay(store, kp=kp, pricing="metered")

        body = _openai_body(
            {"model": "m", "stream": True, "stream_options": {"include_usage": True}}
        )
        _serve_openai_roundtrip(
            tmp_path, relay=relay, forwarder=fwd, raw=_openai_raw(cap.token, body=body)
        )
        line = _log_lines(tmp_path / "relay.jsonl")[0]
        # The FULL final usage dict (all keys, incl. both reasoning shapes'
        # source data) lands in the JSONL per response.
        assert line["usage"]["prompt_tokens"] == 11
        assert line["usage"]["completion_tokens"] == 7
        assert line["usage"]["reasoning_tokens"] == 3
        assert line["usage"]["prompt_tokens_details"]["cached_tokens"] == 5
        assert line["reason"] == "forwarded"
        assert store.usage(cap.token) == {"output_tokens": 7, "calls": 1}

    def test_metered_sse_without_usage_charges_unbudgetable(self, tmp_path):
        # The flag was requested (policy passes) but the upstream never sent
        # a usage chunk: undetermined -> metered saturates (kill switch).
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=500, max_calls=5)
        fwd = AsyncFakeForwarder(
            headers=[("content-type", "text/event-stream")], chunks=[CONTENT_CHUNK, DONE]
        )
        relay = _openai_relay(store, kp=kp, pricing="metered")

        body = _openai_body(
            {"model": "m", "stream": True, "stream_options": {"include_usage": True}}
        )
        resp, log_path = _serve_openai_roundtrip(
            tmp_path, relay=relay, forwarder=fwd, raw=_openai_raw(cap.token, body=body)
        )
        head, _ = _split_response(resp)
        assert head.startswith(b"HTTP/1.1 200")
        assert store.usage(cap.token)["output_tokens"] == 500  # saturated
        assert store.is_live(cap.token) is False
        line = _log_lines(log_path)[0]
        assert line["reason"] == "unbudgetable"
        assert line["usage"] is None

    def test_local_sse_without_usage_is_accounting_only(self, tmp_path):
        store = CapabilityStore()
        kp = CountingKeyProvider(None)
        cap = store.mint("d1", max_output_tokens=500, max_calls=5)
        fwd = AsyncFakeForwarder(
            headers=[("content-type", "text/event-stream")], chunks=[CONTENT_CHUNK, DONE]
        )
        relay = _openai_relay(store, kp=kp, pricing="local", auth="none")

        body = _openai_body({"model": "m", "stream": True})  # local: no flag needed
        resp, log_path = _serve_openai_roundtrip(
            tmp_path, relay=relay, forwarder=fwd, raw=_openai_raw(cap.token, body=body)
        )
        head, _ = _split_response(resp)
        assert head.startswith(b"HTTP/1.1 200")
        # Accounting-only: 0 charged, the call still counted, never saturated.
        assert store.usage(cap.token) == {"output_tokens": 0, "calls": 1}
        assert store.is_live(cap.token) is True
        line = _log_lines(log_path)[0]
        assert line["reason"] == "accounting-only"
        assert line["usage"] is None

    def test_upstream_429_forwarded_verbatim_and_logged(self, tmp_path):
        # The Synthetic 429 shape is unobserved — the FIRST occurrence must be
        # recoverable from the JSONL (verbatim, bounded).
        flat_429 = b'{"error":"quota exhausted"}'
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=10000, max_calls=5)
        fwd = AsyncFakeForwarder(
            status=429,
            headers=[
                ("content-type", "application/json"),
                ("x-synthetic-quotas", "5h:12/100;7d:40/500"),
            ],
            chunks=[flat_429],
        )
        relay = _openai_relay(store, kp=kp, pricing="metered")

        body = _openai_body({"model": "m"})
        resp, log_path = _serve_openai_roundtrip(
            tmp_path, relay=relay, forwarder=fwd, raw=_openai_raw(cap.token, body=body)
        )
        head, body_bytes = _split_response(resp)
        assert head.startswith(b"HTTP/1.1 429")
        # Forwarded VERBATIM to the worker (upstream non-2xx bodies pass through).
        assert body_bytes == flat_429
        line = _log_lines(log_path)[0]
        assert line["status"] == 429
        assert line["decision"] == "allow"
        assert line["upstream_429_body"] == flat_429.decode("utf-8")
        assert line["synthetic_quotas"] == "5h:12/100;7d:40/500"
        # A non-2xx upstream error does not meter tokens; the reserved call
        # slot still counts (no unmetered retries — §C).
        assert store.usage(cap.token) == {"output_tokens": 0, "calls": 1}

    def test_upstream_429_body_capped_but_forwarded_in_full(self, tmp_path):
        # The LOG capture is bounded (4 KiB) while the WORKER still gets every
        # byte (verbatim forwarding is unbounded by design).
        big = b'{"error":"' + b"x" * 10_000 + b'"}'
        store = CapabilityStore()
        kp = CountingKeyProvider()
        cap = store.mint("d1", max_output_tokens=10000, max_calls=5)
        fwd = AsyncFakeForwarder(status=429, chunks=[big])
        relay = _openai_relay(store, kp=kp, pricing="metered")

        resp, log_path = _serve_openai_roundtrip(
            tmp_path, relay=relay, forwarder=fwd, raw=_openai_raw(cap.token)
        )
        _, body_bytes = _split_response(resp)
        assert body_bytes == big  # worker got the FULL body
        line = _log_lines(log_path)[0]
        assert len(line["upstream_429_body"]) == 4096
        assert line["upstream_429_body"] == big[:4096].decode("utf-8")
