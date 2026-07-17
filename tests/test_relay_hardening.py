"""Hardening tests for the credential relay (bead .55) — the auth / routing /
injection surface findings of the 2026-07-17 multi-model adversarial audit
(tmp/code-audit/stigmergy-relay/AUDIT-REPORT.md, commit 83e3dbe).

AUTHORED BY THE ORCHESTRATOR (Merry), not the implementor — this is the single
highest-security component in the system. Every assertion here reproduces a
concrete finding; the implementation must turn each RED test GREEN WITHOUT
weakening one. Threat model (SPEC §4): every worker is fully compromised and
actively trying to (a) exfiltrate the real provider key, (b) reach the whole
provider API with a single call quota, (c) smuggle credentialing/routing
headers onto the real-key request. DUMMY keys only.

Covers F1 (CRITICAL path->host key-exfil), F2 (header allowlist / relay owns
version+beta), F3 (method+path allowlist), F7 (Content-Length ascii). The
metering/streaming/DoS findings (F4/F5/F6) are bead .56, not here.
"""

from __future__ import annotations

import asyncio
import urllib.error
from urllib.parse import urlsplit

import pytest

from stigmergy.relay import (
    CapabilityStore,
    CredentialRelay,
    DenyResponse,
    RelayRequest,
    UpstreamRequest,
    prepare_upstream,
)
from stigmergy.relay_transport import (
    RelayTransportError,
    UpstreamError,
    make_urllib_forwarder,
    read_relay_request,
)

DUMMY_REAL_KEY = "[REDACTED:api_key]"
CAP_HEADER = "x-api-key"
BASE_URL = "https://api.anthropic.com"
EXPECTED_NETLOC = "api.anthropic.com"


# --------------------------------------------------------------------------- #
# Fakes / helpers (self-contained — this file owns its harness)               #
# --------------------------------------------------------------------------- #
class CountingKeyProvider:
    """Returns the dummy real key and counts fetches. A deny path (missing
    capability, forbidden endpoint, denied quota) must fetch ZERO times."""

    def __init__(self, key: str = DUMMY_REAL_KEY) -> None:
        self._key = key
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        return self._key


def _unused_sync_forwarder(req):  # noqa: ANN001, ANN202
    raise AssertionError("the .12 sync forwarder is never used by prepare_upstream")


def _make_relay(store: CapabilityStore, kp: CountingKeyProvider, **cfg) -> CredentialRelay:
    """Build a relay with the .55 security config. Passing the new keyword
    args is intentional: until they exist on CredentialRelay this raises
    (RED), which is the contract driving the implementation."""
    return CredentialRelay(
        store=store,
        key_provider=kp,
        forwarder=_unused_sync_forwarder,
        capability_header=CAP_HEADER,
        **cfg,
    )


class CapturingOpener:
    """A urllib-style opener that RECORDS the netloc of every URL it is asked
    to open, then raises (never really connects). The security property under
    test: the key-bearing request must NEVER be opened against a netloc other
    than the pinned upstream, no matter what path the worker supplies."""

    def __init__(self) -> None:
        self.opened_netlocs: list[str] = []

    def open(self, request, timeout=None):  # noqa: ANN001, ANN201
        self.opened_netlocs.append(urlsplit(request.full_url).netloc)
        raise urllib.error.URLError("captured (test opener never connects upstream)")


def _upstream(path: str, *, method: str = "POST", body: bytes = b'{"model":"x"}') -> UpstreamRequest:
    return UpstreamRequest(
        request=RelayRequest(
            method=method, path=path, headers={CAP_HEADER: DUMMY_REAL_KEY}, body=body
        ),
        token="tok",
    )


async def _drive_forwarder(forwarder, upstream: UpstreamRequest) -> None:
    """Run a forwarder to completion, swallowing the expected UpstreamError
    (the capturing opener always raises). We assert on opened_netlocs, not on
    the return value."""
    try:
        head, gen = await forwarder(upstream)
        async for _ in gen:
            pass
    except UpstreamError:
        pass


def _reader_from(raw: bytes) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    reader.feed_data(raw)
    reader.feed_eof()
    return reader


def _read_raw(raw: bytes) -> RelayRequest:
    async def _run() -> RelayRequest:
        return await read_relay_request(
            _reader_from(raw),
            max_head_bytes=65536,
            max_body_bytes=32 * 1024 * 1024,
            timeout=5.0,
        )

    return asyncio.run(_run())


def _raw(method: str, path: str, *, headers: dict[str, str], body: bytes = b"") -> bytes:
    lines = [f"{method} {path} HTTP/1.1"]
    lines += [f"{k}: {v}" for k, v in headers.items()]
    head = ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")
    return head + body


# =========================================================================== #
# F1 (CRITICAL) — worker request path must never redirect the real-key         #
# request off the pinned upstream host                                         #
# =========================================================================== #
class TestF1PathHostPinning:
    # Paths that, string-concatenated onto BASE_URL, move the netloc to an
    # attacker-controlled host / port / userinfo (the confirmed exploit).
    EVIL_PATHS = [
        ".evil.com:443/v1/messages",   # -> api.anthropic.com.evil.com (host suffix hijack)
        ".evil.com/v1/messages",       # -> api.anthropic.com.evil.com
        ":9999/v1/messages",           # -> api.anthropic.com:9999 (port change)
        "@evil.com/v1/messages",       # -> userinfo api.anthropic.com@, host evil.com
    ]

    @pytest.mark.parametrize("evil_path", EVIL_PATHS)
    def test_forwarder_never_opens_a_non_pinned_netloc(self, evil_path):
        opener = CapturingOpener()
        forwarder = make_urllib_forwarder(base_url=BASE_URL, opener=opener)
        asyncio.run(_drive_forwarder(forwarder, _upstream(evil_path)))
        # THE security property: the key-bearing request was never opened to
        # any netloc other than the pinned upstream.
        assert all(nl == EXPECTED_NETLOC for nl in opener.opened_netlocs), (
            f"real-key request opened against non-pinned netloc(s): "
            f"{opener.opened_netlocs} (path={evil_path!r})"
        )

    def test_forwarder_still_reaches_pinned_host_for_a_legit_path(self):
        # Guard against an over-aggressive fix: a legitimate origin-form path
        # must still reach the pinned upstream netloc.
        opener = CapturingOpener()
        forwarder = make_urllib_forwarder(base_url=BASE_URL, opener=opener)
        asyncio.run(_drive_forwarder(forwarder, _upstream("/v1/messages")))
        assert opener.opened_netlocs == [EXPECTED_NETLOC]

    @pytest.mark.parametrize(
        "bad_path",
        [
            ".evil.com/v1/messages",     # no leading slash
            "evil.com/v1/messages",      # no leading slash
            "http://evil.com/v1/messages",  # absolute-form / scheme
            "//evil.com/v1/messages",    # network-path reference (double slash)
            "/v1/\tmessages",            # C0 control char (tab)
            "/v1/\u0080messages",        # C1 control byte (0x80, latin-1 decoded head)
            "/v1\\messages",             # backslash
        ],
    )
    def test_parser_rejects_non_origin_form_path(self, bad_path):
        raw = _raw(
            "POST",
            bad_path,
            headers={CAP_HEADER: "tok", "content-type": "application/json", "content-length": "2"},
            body=b"{}",
        )
        with pytest.raises(RelayTransportError):
            _read_raw(raw)

    @pytest.mark.parametrize("good_path", ["/v1/messages", "/v1/messages?beta=true"])
    def test_parser_accepts_origin_form_path(self, good_path):
        raw = _raw(
            "POST",
            good_path,
            headers={CAP_HEADER: "tok", "content-type": "application/json", "content-length": "2"},
            body=b"{}",
        )
        req = _read_raw(raw)
        assert req.path == good_path


# =========================================================================== #
# F2 (HIGH) — request-header allowlist; the RELAY owns version/beta            #
# =========================================================================== #
class TestF2HeaderAllowlist:
    def _authorized_result(self):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        relay = _make_relay(
            store,
            kp,
            upstream_header_allowlist=frozenset({"content-type", "accept"}),
            upstream_headers_pinned={"anthropic-version": "2023-06-01"},
        )
        cap = store.mint("d1", max_output_tokens=1000, max_calls=5)
        req = RelayRequest(
            method="POST",
            path="/v1/messages",
            headers={
                CAP_HEADER: cap.token,
                "content-type": "application/json",
                "accept": "*/*",
                # hostile / worker-controlled headers that must NOT reach upstream:
                "authorization": "Bearer attacker-secret",
                "anthropic-beta": "worker-chosen-feature",
                "anthropic-version": "1999-01-01",
                "host": "spoofed.example",
                "x-forwarded-for": "1.2.3.4",
                "cookie": "session=steal",
            },
            body=b'{"model":"x"}',
        )
        result = prepare_upstream(relay, req)
        assert isinstance(result, UpstreamRequest)
        return {k.lower(): v for k, v in result.request.headers.items()}

    def test_real_key_injected_and_allowlisted_headers_survive(self):
        hdrs = self._authorized_result()
        assert hdrs[CAP_HEADER] == DUMMY_REAL_KEY
        assert hdrs["content-type"] == "application/json"
        assert hdrs["accept"] == "*/*"

    def test_relay_owns_anthropic_version_not_the_worker(self):
        hdrs = self._authorized_result()
        # worker sent 1999-01-01; the relay's pinned value wins.
        assert hdrs["anthropic-version"] == "2023-06-01"

    @pytest.mark.parametrize(
        "banned", ["authorization", "anthropic-beta", "host", "x-forwarded-for", "cookie"]
    )
    def test_hostile_worker_headers_are_dropped(self, banned):
        hdrs = self._authorized_result()
        assert banned not in hdrs, f"{banned!r} must not reach the real-key upstream request"


# =========================================================================== #
# F3 (HIGH) — method+path allowlist (a valid capability must not reach the      #
# entire provider API)                                                         #
# =========================================================================== #
class TestF3EndpointAllowlist:
    def _relay_and_cap(self):
        store = CapabilityStore()
        kp = CountingKeyProvider()
        relay = _make_relay(
            store,
            kp,
            allowed_endpoints=frozenset({("POST", "/v1/messages")}),
        )
        cap = store.mint("d1", max_output_tokens=1000, max_calls=5)
        return relay, store, kp, cap

    def test_forbidden_method_denied_without_fetching_key(self):
        relay, _store, kp, cap = self._relay_and_cap()
        req = RelayRequest(
            method="GET", path="/v1/messages", headers={CAP_HEADER: cap.token}, body=b""
        )
        result = prepare_upstream(relay, req)
        assert isinstance(result, DenyResponse)
        assert result.status == 403
        assert result.reason == "forbidden-endpoint"
        assert kp.calls == 0

    def test_forbidden_path_denied_without_fetching_key(self):
        relay, _store, kp, cap = self._relay_and_cap()
        req = RelayRequest(
            method="POST", path="/v1/models", headers={CAP_HEADER: cap.token}, body=b"{}"
        )
        result = prepare_upstream(relay, req)
        assert isinstance(result, DenyResponse)
        assert result.status == 403
        assert result.reason == "forbidden-endpoint"
        assert kp.calls == 0

    def test_allowed_endpoint_authorizes_and_injects(self):
        relay, _store, kp, cap = self._relay_and_cap()
        req = RelayRequest(
            method="POST", path="/v1/messages", headers={CAP_HEADER: cap.token}, body=b"{}"
        )
        result = prepare_upstream(relay, req)
        assert isinstance(result, UpstreamRequest)
        assert kp.calls == 1

    def test_allowed_endpoint_matches_despite_query_string(self):
        relay, _store, kp, cap = self._relay_and_cap()
        req = RelayRequest(
            method="POST", path="/v1/messages?beta=1", headers={CAP_HEADER: cap.token}, body=b"{}"
        )
        result = prepare_upstream(relay, req)
        assert isinstance(result, UpstreamRequest)
        assert kp.calls == 1


# =========================================================================== #
# F7 (MEDIUM) — Content-Length validated ascii, malformed => clean 400          #
# =========================================================================== #
class TestF7ContentLengthAscii:
    @pytest.mark.parametrize("bad_cl", ["\u00b2", "1\u00b2", "\u00b3"])  # superscripts: isdigit()=True, int()=ValueError
    def test_unicode_content_length_is_relay_transport_error_not_uncaught(self, bad_cl):
        raw = _raw(
            "POST",
            "/v1/messages",
            headers={CAP_HEADER: "tok", "content-type": "application/json", "content-length": bad_cl},
            body=b"xx",
        )
        with pytest.raises(RelayTransportError):
            _read_raw(raw)

    def test_ascii_content_length_parses(self):
        body = b'{"model":"x"}'
        raw = _raw(
            "POST",
            "/v1/messages",
            headers={
                CAP_HEADER: "tok",
                "content-type": "application/json",
                "content-length": str(len(body)),
            },
            body=body,
        )
        req = _read_raw(raw)
        assert req.method == "POST"
        assert req.path == "/v1/messages"
        assert req.body == body
