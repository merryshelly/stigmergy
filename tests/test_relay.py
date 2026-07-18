"""Adversarial tests for the credential relay + capability tokens (SPEC.md §4
credentials "out of the worker namespace entirely", §10 AC4).

Authored by the orchestrator, not the implementor — the credential contract is
security-critical and is the orchestrator's to own (bead .12 build spec §2, an
orchestrator-frozen case-list; the implementation in `stigmergy.relay` must
satisfy every assertion without weakening one to force a pass).

Threat model (SPEC §4): every worker is assumed compromised. The worker must
hold ONLY a per-dispatch capability token — never a real provider key — and
that token must be dead the moment the dispatch ends (replay fails). The real
provider key lives only inside the relay process and must never reach a worker
env var, mount, event, transcript, log, or prompt. DUMMY keys only in tests.

Scope (mirrors bead .11's mechanism-vs-live split): this exercises the offline,
deterministic relay CORE with an injected forwarder + injected key provider
(the `critic.py` DI discipline — no provider SDK). The live wired path
(worker→shim→relay→api.anthropic.com, real key via `op`, SSE usage parsing) is
a .25 deliverable.
"""

from __future__ import annotations

import json

import pytest

from stigmergy.records import RecordError, RecordPlane
from stigmergy.relay import (
    REDACTED_PLACEHOLDER,
    Capability,
    CapabilityDenied,
    CapabilityStore,
    CredentialRelay,
    RelayError,
    RelayRequest,
    RelayResponse,
    build_redactor,
    extract_usage,
    worker_credential_env,
)

# A stand-in for a real provider key — this string must NEVER appear in any
# worker-facing artifact or any sealed transcript. It is deliberately shaped
# like an Anthropic key so a leak is obvious.
DUMMY_REAL_KEY = "sk-ant-REAL-xxxxDUMMYxxxx"

BASE_URL = "http://127.0.0.1:8931"


class FakeForwarder:
    """Captures the upstream request it is handed and returns a canned response
    (mirrors test_critic.StubClient). Records call count so a deny path can be
    asserted to never reach upstream."""

    def __init__(self, response: RelayResponse | None = None):
        self._response = response or RelayResponse(status=200, headers={}, body=b"{}")
        self.requests: list[RelayRequest] = []

    def __call__(self, request: RelayRequest) -> RelayResponse:
        self.requests.append(request)
        return self._response


class CountingKeyProvider:
    """Returns the dummy real key and counts how often it was asked for it — a
    deny path must fetch the key ZERO times."""

    def __init__(self, key: str = DUMMY_REAL_KEY):
        self._key = key
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        return self._key


def _usage_body(output_tokens: int) -> bytes:
    return json.dumps(
        {"id": "msg_x", "usage": {"input_tokens": 10, "output_tokens": output_tokens}}
    ).encode("utf-8")


def _relay(store, *, forwarder=None, key_provider=None) -> CredentialRelay:
    return CredentialRelay(
        store=store,
        key_provider=key_provider or CountingKeyProvider(),
        forwarder=forwarder or FakeForwarder(),
    )


def _request(token: str, *, header: str = "x-api-key") -> RelayRequest:
    return RelayRequest(
        method="POST",
        path="/v1/messages",
        headers={header: token, "content-type": "application/json"},
        body=b'{"model":"claude","messages":[]}',
    )


# ==========================================================================
# Capability lifecycle
# ==========================================================================


def test_mint_returns_unguessable_scoped_capability():
    store = CapabilityStore()
    cap = store.mint("disp-1", max_output_tokens=1000, max_calls=5)
    assert isinstance(cap, Capability)
    assert cap.dispatch_id == "disp-1"
    assert cap.max_output_tokens == 1000
    assert cap.max_calls == 5
    # High-entropy, not a trivially guessable value.
    assert isinstance(cap.token, str) and len(cap.token) >= 32
    # Distinct dispatches get distinct tokens.
    cap2 = store.mint("disp-2", max_output_tokens=1000, max_calls=5)
    assert cap2.token != cap.token


def test_authorize_live_capability_returns_it():
    store = CapabilityStore()
    cap = store.mint("disp-1", max_output_tokens=1000, max_calls=5)
    got = store.authorize(cap.token)
    assert got.token == cap.token
    assert got.dispatch_id == "disp-1"


def test_authorize_after_revoke_fails_replay_dead():
    # AC4 core: the capability is dead after dispatch end; a replayed (e.g.
    # exfiltrated) token no longer authorizes.
    store = CapabilityStore()
    cap = store.mint("disp-1", max_output_tokens=1000, max_calls=5)
    assert store.authorize(cap.token).token == cap.token  # live before end
    store.revoke("disp-1")
    with pytest.raises(CapabilityDenied) as exc:
        store.authorize(cap.token)
    assert exc.value.reason == "revoked"
    assert store.is_live(cap.token) is False


def test_revoke_is_idempotent():
    store = CapabilityStore()
    cap = store.mint("disp-1", max_output_tokens=1000, max_calls=5)
    store.revoke("disp-1")
    store.revoke("disp-1")  # second revoke: no raise
    store.revoke("never-existed")  # unknown dispatch: no raise
    assert store.is_live(cap.token) is False


def test_quota_calls_enforced():
    store = CapabilityStore()
    cap = store.mint("disp-1", max_output_tokens=10_000, max_calls=2)
    # Two authorize+charge cycles are allowed.
    store.authorize(cap.token)
    store.charge(cap.token, output_tokens=1, calls=1)
    store.authorize(cap.token)
    store.charge(cap.token, output_tokens=1, calls=1)
    # The third authorize is denied — the call budget is spent.
    with pytest.raises(CapabilityDenied) as exc:
        store.authorize(cap.token)
    assert exc.value.reason == "quota-calls"


def test_quota_output_tokens_enforced():
    store = CapabilityStore()
    cap = store.mint("disp-1", max_output_tokens=100, max_calls=100)
    store.authorize(cap.token)
    store.charge(cap.token, output_tokens=100, calls=1)
    with pytest.raises(CapabilityDenied) as exc:
        store.authorize(cap.token)
    assert exc.value.reason == "quota-tokens"


def test_authorize_unknown_token_denied():
    store = CapabilityStore()
    store.mint("disp-1", max_output_tokens=100, max_calls=5)
    with pytest.raises(CapabilityDenied) as exc:
        store.authorize("not-a-real-token-deadbeef")
    assert exc.value.reason == "unknown"


def test_mint_twice_same_dispatch_raises():
    store = CapabilityStore()
    store.mint("disp-1", max_output_tokens=100, max_calls=5)
    with pytest.raises(RelayError):
        store.mint("disp-1", max_output_tokens=100, max_calls=5)


def test_charge_dead_token_raises():
    store = CapabilityStore()
    cap = store.mint("disp-1", max_output_tokens=100, max_calls=5)
    store.revoke("disp-1")
    with pytest.raises(CapabilityDenied):
        store.charge(cap.token, output_tokens=1, calls=1)
    with pytest.raises(CapabilityDenied):
        store.charge("unknown-token", output_tokens=1, calls=1)


# ==========================================================================
# charge_unbudgetable (F4, bead .56): a call whose true output-token usage
# could not be metered saturates the token budget so the next authorize is
# quota-denied — NEVER charges ~0 (the "unbudgetable, never $0" principle
# from .51). Fail-closed leash under an unmeterable stream.
# ==========================================================================


def test_charge_unbudgetable_saturates_and_next_authorize_denied():
    store = CapabilityStore()
    cap = store.mint("disp-1", max_output_tokens=500, max_calls=5)
    store.authorize(cap.token)
    store.charge(cap.token, output_tokens=0, calls=1)  # a call reserved, unmeterable
    store.charge_unbudgetable(cap.token)
    # used_output_tokens is now saturated to the ceiling ...
    assert store.usage(cap.token)["output_tokens"] == 500
    # ... so the leash trips: the next authorize is quota-tokens denied.
    with pytest.raises(CapabilityDenied) as exc:
        store.authorize(cap.token)
    assert exc.value.reason == "quota-tokens"


def test_charge_unbudgetable_never_charges_zero_on_fresh_capability():
    # The whole point: an unmeterable call must not leave the worker with its
    # full token budget intact (the F4 leash bypass). Even from zero usage,
    # the capability ends quota-exhausted.
    store = CapabilityStore()
    cap = store.mint("disp-1", max_output_tokens=1000, max_calls=5)
    store.charge_unbudgetable(cap.token)
    assert store.usage(cap.token)["output_tokens"] == 1000
    assert store.is_live(cap.token) is False


def test_charge_unbudgetable_leaves_calls_untouched():
    store = CapabilityStore()
    cap = store.mint("disp-1", max_output_tokens=500, max_calls=5)
    store.charge(cap.token, output_tokens=0, calls=1)
    store.charge_unbudgetable(cap.token)
    assert store.usage(cap.token)["calls"] == 1  # only tokens saturated, not calls


def test_charge_unbudgetable_is_idempotent():
    store = CapabilityStore()
    cap = store.mint("disp-1", max_output_tokens=500, max_calls=5)
    store.charge_unbudgetable(cap.token)
    store.charge_unbudgetable(cap.token)  # second call must not double past the ceiling
    assert store.usage(cap.token)["output_tokens"] == 500


def test_charge_unbudgetable_saturates_from_partial_usage():
    store = CapabilityStore()
    cap = store.mint("disp-1", max_output_tokens=500, max_calls=5)
    store.charge(cap.token, output_tokens=120, calls=1)
    store.charge_unbudgetable(cap.token)
    assert store.usage(cap.token)["output_tokens"] == 500  # exactly the ceiling, never over


def test_charge_unbudgetable_unknown_and_revoked_raise():
    store = CapabilityStore()
    cap = store.mint("disp-1", max_output_tokens=500, max_calls=5)
    with pytest.raises(CapabilityDenied) as exc:
        store.charge_unbudgetable("not-a-real-token")
    assert exc.value.reason == "unknown"
    store.revoke("disp-1")
    with pytest.raises(CapabilityDenied) as exc2:
        store.charge_unbudgetable(cap.token)
    assert exc2.value.reason == "revoked"


# ==========================================================================
# Relay handle(): inject real key toward UPSTREAM only; deny is fail-closed
# ==========================================================================


def test_handle_injects_real_key_upstream_and_strips_capability():
    store = CapabilityStore()
    cap = store.mint("disp-1", max_output_tokens=10_000, max_calls=10)
    forwarder = FakeForwarder(RelayResponse(status=200, headers={}, body=_usage_body(5)))
    relay = _relay(store, forwarder=forwarder)

    relay.handle(_request(cap.token))

    assert len(forwarder.requests) == 1
    upstream = forwarder.requests[0]
    upstream_values = list(upstream.headers.values())
    # The real key is injected toward upstream...
    assert DUMMY_REAL_KEY in upstream_values
    # ...and the capability token is NOT forwarded upstream (it was swapped out).
    assert cap.token not in upstream_values
    # Method/path/body pass through unchanged.
    assert upstream.method == "POST"
    assert upstream.path == "/v1/messages"
    assert upstream.body == b'{"model":"claude","messages":[]}'


def test_handle_denied_capability_never_calls_forwarder():
    # Fail-closed: a denied capability must never reach upstream and must never
    # cause the real key to be fetched (mirrors egress "never pipes on deny").
    store = CapabilityStore()
    cap = store.mint("disp-1", max_output_tokens=10_000, max_calls=10)
    store.revoke("disp-1")
    forwarder = FakeForwarder()
    keyprov = CountingKeyProvider()
    relay = _relay(store, forwarder=forwarder, key_provider=keyprov)

    resp = relay.handle(_request(cap.token))
    assert resp.status == 401  # revoked/unknown -> 401
    assert len(forwarder.requests) == 0
    assert keyprov.calls == 0

    # Unknown token: same fail-closed behaviour.
    resp2 = relay.handle(_request("totally-unknown-token"))
    assert resp2.status == 401
    assert len(forwarder.requests) == 0
    assert keyprov.calls == 0


def test_handle_response_to_worker_carries_no_real_key():
    store = CapabilityStore()
    cap = store.mint("disp-1", max_output_tokens=10_000, max_calls=10)
    forwarder = FakeForwarder(RelayResponse(status=200, headers={}, body=_usage_body(5)))
    relay = _relay(store, forwarder=forwarder)

    resp = relay.handle(_request(cap.token))
    assert DUMMY_REAL_KEY not in json.dumps(dict(resp.headers))
    assert DUMMY_REAL_KEY.encode("utf-8") not in resp.body


def test_handle_meters_usage_against_quota():
    store = CapabilityStore()
    cap = store.mint("disp-1", max_output_tokens=100, max_calls=10)
    forwarder = FakeForwarder(RelayResponse(status=200, headers={}, body=_usage_body(60)))
    relay = _relay(store, forwarder=forwarder)

    relay.handle(_request(cap.token))  # charges 60
    relay.handle(_request(cap.token))  # charges 60 -> total 120 >= 100
    assert len(forwarder.requests) == 2
    # The third call is denied on the token-quota pre-check; upstream not touched.
    resp = relay.handle(_request(cap.token))
    assert resp.status == 429
    assert len(forwarder.requests) == 2


def test_handle_missing_capability_header_denied():
    store = CapabilityStore()
    store.mint("disp-1", max_output_tokens=100, max_calls=5)
    forwarder = FakeForwarder()
    keyprov = CountingKeyProvider()
    relay = _relay(store, forwarder=forwarder, key_provider=keyprov)

    req = RelayRequest(method="POST", path="/v1/messages", headers={"content-type": "x"}, body=b"")
    resp = relay.handle(req)
    assert resp.status == 401
    assert len(forwarder.requests) == 0
    assert keyprov.calls == 0


# ==========================================================================
# Credential vacancy (AC4): the real key never reaches the worker
# ==========================================================================


def test_worker_credential_env_has_capability_not_real_key():
    store = CapabilityStore()
    cap = store.mint("disp-1", max_output_tokens=100, max_calls=5)
    env = worker_credential_env(cap, base_url=BASE_URL)
    values = list(env.values())
    assert cap.token in values
    assert BASE_URL in values
    # The real provider key appears in NO value.
    assert all(DUMMY_REAL_KEY not in v for v in values)


def test_worker_credential_env_base_url_none_omits_base_url():
    # bead .25: when the relay is wired, the .63 worker entrypoint OWNS
    # ANTHROPIC_BASE_URL (it points claude at the loopback relay shim). The
    # daemon must then NOT put a host base URL in cred_env — only the
    # capability token. `base_url=None` (the new default) yields exactly that.
    store = CapabilityStore()
    cap = store.mint("disp-1", max_output_tokens=100, max_calls=5)
    env = worker_credential_env(cap, base_url=None)
    assert env == {"ANTHROPIC_API_KEY": cap.token}
    assert "ANTHROPIC_BASE_URL" not in env
    # default (omitted) behaves like base_url=None
    assert worker_credential_env(cap) == {"ANTHROPIC_API_KEY": cap.token}
    # real key never present either way
    assert DUMMY_REAL_KEY not in cap.token


def test_no_worker_facing_artifact_contains_real_key():
    # Everything the relay module hands toward the worker for a dispatch must be
    # clean of the real key — it lives only in the relay process.
    store = CapabilityStore()
    cap = store.mint("disp-1", max_output_tokens=100, max_calls=5)
    _relay(store, key_provider=CountingKeyProvider())  # relay holds the key internally

    surfaces = [
        json.dumps(worker_credential_env(cap, base_url=BASE_URL)),
        cap.token,
        cap.dispatch_id,
        str(cap.max_output_tokens),
        str(cap.max_calls),
        str(store.is_live(cap.token)),
        json.dumps(store.usage(cap.token)),
    ]
    assert all(DUMMY_REAL_KEY not in s for s in surfaces)


# ==========================================================================
# Redaction at seal (AC4 second proof) — integrates with records.seal_transcript
# ==========================================================================


def test_build_redactor_scrubs_all_injected_values():
    store = CapabilityStore()
    cap = store.mint("disp-1", max_output_tokens=100, max_calls=5)
    relay = _relay(store)
    secrets = relay.injected_secrets(cap.token)
    assert cap.token in secrets
    assert DUMMY_REAL_KEY in secrets

    redactor = build_redactor(secrets)
    text = f"log line: key={DUMMY_REAL_KEY} token={cap.token} rest is fine"
    out = redactor(text)
    assert DUMMY_REAL_KEY not in out
    assert cap.token not in out
    assert REDACTED_PLACEHOLDER in out
    assert "rest is fine" in out  # benign text untouched


def test_seal_transcript_with_relay_redactor_stores_no_secret(tmp_path):
    store = CapabilityStore()
    cap = store.mint("disp-1", max_output_tokens=100, max_calls=5)
    relay = _relay(store)
    secrets = relay.injected_secrets(cap.token)

    rp = RecordPlane(tmp_path / "records")
    transcript = (
        f"worker said: my key is {DUMMY_REAL_KEY} and my token is {cap.token}\n"
        "some ordinary transcript content\n"
    )
    ref = rp.seal_transcript(
        transcript, redactor=build_redactor(secrets), must_not_contain=secrets
    )
    stored = (rp.transcripts_dir / ref).read_text(encoding="utf-8")
    assert DUMMY_REAL_KEY not in stored
    assert cap.token not in stored
    assert "some ordinary transcript content" in stored


def test_seal_refuses_when_redaction_skipped(tmp_path):
    # Prove-can-fail (up front): with a pass-through redactor the sentinel
    # survives, so seal_transcript MUST refuse and write nothing. This is the
    # enforcement path — if it ever passed, an un-redacted secret would land.
    store = CapabilityStore()
    cap = store.mint("disp-1", max_output_tokens=100, max_calls=5)
    relay = _relay(store)
    secrets = relay.injected_secrets(cap.token)

    rp = RecordPlane(tmp_path / "records")
    transcript = f"leak: {DUMMY_REAL_KEY} {cap.token}\n"
    before = set(p.name for p in rp.transcripts_dir.iterdir())
    with pytest.raises(RecordError):
        rp.seal_transcript(
            transcript, redactor=lambda s: s, must_not_contain=secrets
        )
    after = set(p.name for p in rp.transcripts_dir.iterdir())
    assert before == after  # no blob written on refusal


def test_build_redactor_handles_empty_and_overlapping():
    # An empty secret must be skipped (never replace "").
    redactor = build_redactor(["", "abc"])
    assert redactor("xabcx") == f"x{REDACTED_PLACEHOLDER}x"
    # Overlapping secrets (one a substring of the other) both fully redacted.
    redactor2 = build_redactor(["secret", "topsecret"])
    out = redactor2("the topsecret value and the secret value")
    assert "secret" not in out
    assert "topsecret" not in out


# ==========================================================================
# usage extraction
# ==========================================================================


def test_extract_usage_parses_anthropic_body():
    resp = RelayResponse(status=200, headers={}, body=_usage_body(42))
    usage = extract_usage(resp)
    assert usage.get("output_tokens") == 42
    # Garbage / bodyless / streaming responses never raise; yield {}.
    assert extract_usage(RelayResponse(status=200, headers={}, body=b"not json")) == {}
    assert extract_usage(RelayResponse(status=200, headers={}, body=b"")) == {}
    assert extract_usage(RelayResponse(status=200, headers={}, body=b'{"no":"usage"}')) == {}
