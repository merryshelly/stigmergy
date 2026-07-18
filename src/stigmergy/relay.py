"""Credential relay + capability tokens (SPEC.md §4 credentials "out of the
worker namespace entirely", §10 AC4; bead .12 build spec §0/§1).

**Architecture (bead .12 build spec §0):** a CONNECT proxy (bead .11) tunnels
TLS opaquely and cannot see or inject an ``Authorization``/``x-api-key``
header, so credential injection cannot live in the egress proxy. This module
is therefore a distinct, HTTP-terminating relay that sits on the dispatch's
inference path: the worker holds only a per-dispatch **capability token**
(quota-bound, revoked at dispatch end); the real provider key lives only
inside the relay process and is injected toward the upstream request only —
never returned toward the worker in any form (env, response, transcript).

**Threat model (SPEC §4): every worker is assumed compromised.** A capability
token that leaks (env dump, log, transcript) must be worthless the instant
the dispatch ends — ``revoke()`` kills it, and a replayed token raises
``CapabilityDenied("revoked")``. Every authorization failure is **fail
closed** (mirrors ``egress_proxy.py``'s "any doubt = deny, deny never
touches upstream"): ``CredentialRelay.handle`` never calls the injected
``forwarder`` and never calls the injected ``key_provider`` on a denied
capability — a rejected/expired/unknown/quota-exhausted token gets a
synthesized deny response, full stop.

**Scope (mirrors bead .11's mechanism-vs-live split).** This module is the
offline, deterministic relay CORE: capability lifecycle, ``handle()``'s
authorize/inject/forward/meter pipeline, usage extraction, and the redaction
wiring into ``records.RecordPlane.seal_transcript``. It mirrors
``critic.py``'s injected-client discipline exactly — ``key_provider`` and
``forwarder`` are injected callables; this module NEVER imports a provider
SDK and never performs real network I/O. The wired worker->shim->relay->
``api.anthropic.com`` HTTP transport (real key sourced via ``op``, SSE
usage-parse fidelity) is a .25 deliverable that wraps this synchronous core
and gets its own security review when built — not slipped in unreviewed
here.
"""

from __future__ import annotations

import json
import secrets
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

REDACTED_PLACEHOLDER = "«redacted»"

# What claude-code sends as the request-level credential when
# ANTHROPIC_API_KEY is set (SPEC §4 / bead .12 build spec §0 diagram).
CAPABILITY_HEADER_DEFAULT = "x-api-key"

# --------------------------------------------------------------------------- #
# bead .55 hardening defaults (F2/F3) — deliberately tight. The capability    #
# header must NEVER appear in the allowlist (the real key is injected under  #
# it directly, never copied from the worker's value). Production `.25`      #
# wiring will set `upstream_headers_pinned={"anthropic-version": "<current>"}`#
# and may extend the allowlist if a live smoke shows claude-code needs more  #
# — exact membership is a `.26` concern, not decided here.                   #
# --------------------------------------------------------------------------- #
DEFAULT_UPSTREAM_HEADER_ALLOWLIST: frozenset[str] = frozenset({"content-type", "accept"})
DEFAULT_ALLOWED_ENDPOINTS: frozenset[tuple[str, str]] = frozenset({("POST", "/v1/messages")})


class RelayError(Exception):
    """Base for all relay errors (caller bugs, malformed inputs) — distinct
    from :class:`CapabilityDenied`, which is the expected, fail-closed
    authorization-denial path."""


class CapabilityDenied(RelayError):
    """Raised by :class:`CapabilityStore` (and surfaced through
    :meth:`CredentialRelay.handle`) whenever a capability token is not
    currently live. ``reason`` is one of ``"unknown"``, ``"revoked"``,
    ``"quota-calls"``, ``"quota-tokens"`` — never anything else — so
    callers can map it to a deny status without string-sniffing beyond
    this fixed vocabulary."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class Capability:
    """A minted, per-dispatch capability. ``token`` is the ONLY secret the
    worker ever receives — high-entropy (``secrets.token_urlsafe(32)``),
    unguessable, and dead the instant :meth:`CapabilityStore.revoke` is
    called for its ``dispatch_id``. Every field here is safe to expose to
    the worker (and is exercised as such by
    ``test_no_worker_facing_artifact_contains_real_key``)."""

    token: str
    dispatch_id: str
    max_output_tokens: int
    max_calls: int


@dataclass(frozen=True)
class RelayRequest:
    """One HTTP-shaped request through the relay. ``headers`` names are
    compared case-insensitively everywhere this module reads them."""

    method: str
    path: str
    headers: Mapping[str, str]
    body: bytes = b""


@dataclass(frozen=True)
class RelayResponse:
    """One HTTP-shaped response. ``usage`` is the usage dict the relay
    charged against the capability's quota for this call, kept for
    observability only — it is never treated as anything else by
    :meth:`CredentialRelay.handle` (the actual charge happens against
    the store directly)."""

    status: int
    headers: Mapping[str, str]
    body: bytes = b""
    usage: dict | None = None


@dataclass
class _LiveCapability:
    """Internal mutable per-token bookkeeping. Never exposed outside this
    module — the public :class:`Capability` is immutable and carries none
    of the mutable usage/revocation state."""

    capability: Capability
    used_output_tokens: int = 0
    used_calls: int = 0
    revoked: bool = False


class CapabilityStore:
    """In-memory per-dispatch capability authority (bead .12 build spec
    §1). Single-threaded (the relay daemon is single-threaded, DEFERRED
    #12 to .25); no locks. Token lookup is O(1) by dict key; tokens are
    256-bit (``secrets.token_urlsafe(32)``) so lookup timing is not a
    practical guessing oracle.

    ``mint`` additionally tracks which ``dispatch_id``s have EVER had a
    capability minted (``_ever_minted``), not just which currently have a
    live one — a stricter reading than "no live duplicate": one dispatch
    gets exactly one capability, ever, ordinary or revoked. This is a
    deliberate hardening beyond the literal "second LIVE capability"
    spec wording (one dispatch = one capability, permanently — a caller
    that tries to mint again for the same dispatch_id, live or dead, has
    a bug and must fail loud, never silently hand out a second token for
    an identity that's already been through the relay).
    """

    def __init__(self) -> None:
        self._by_token: dict[str, _LiveCapability] = {}
        self._token_by_dispatch: dict[str, str] = {}
        self._ever_minted: set[str] = set()

    def mint(self, dispatch_id: str, *, max_output_tokens: int, max_calls: int) -> Capability:
        """Mint a fresh, live capability for ``dispatch_id``.

        Raises :class:`RelayError` if ``dispatch_id`` has ever had a
        capability minted before (see class docstring) — one dispatch is
        one capability; this is a caller bug, never silently replaced.
        """
        if dispatch_id in self._ever_minted:
            raise RelayError(
                f"dispatch_id {dispatch_id!r} already has a minted capability; "
                "one dispatch = one capability"
            )

        token = secrets.token_urlsafe(32)
        capability = Capability(
            token=token,
            dispatch_id=dispatch_id,
            max_output_tokens=max_output_tokens,
            max_calls=max_calls,
        )
        self._by_token[token] = _LiveCapability(capability=capability)
        self._token_by_dispatch[dispatch_id] = token
        self._ever_minted.add(dispatch_id)
        return capability

    def _exists(self, token: str) -> _LiveCapability | None:
        return self._by_token.get(token)

    def authorize(self, token: str) -> Capability:
        """Return the live :class:`Capability` for ``token``, or raise
        :class:`CapabilityDenied` with the first-applicable reason, in
        this fixed order: ``"unknown"`` -> ``"revoked"`` ->
        ``"quota-calls"`` -> ``"quota-tokens"``. Pure pre-check — does not
        itself charge usage (see :meth:`charge`)."""
        live = self._exists(token)
        if live is None:
            raise CapabilityDenied("unknown")
        if live.revoked:
            raise CapabilityDenied("revoked")
        if live.used_calls >= live.capability.max_calls:
            raise CapabilityDenied("quota-calls")
        if live.used_output_tokens >= live.capability.max_output_tokens:
            raise CapabilityDenied("quota-tokens")
        return live.capability

    def charge(self, token: str, *, output_tokens: int, calls: int = 1) -> None:
        """Increment usage for ``token``. Raises :class:`CapabilityDenied`
        if the token is unknown or revoked (charging a dead token is a
        caller bug). ``output_tokens``/``calls`` must be non-negative ints
        (else :class:`RelayError`) — this method does not itself re-check
        the quota; :meth:`authorize` is the gate."""
        if isinstance(output_tokens, bool) or not isinstance(output_tokens, int) or (
            output_tokens < 0
        ):
            raise RelayError(f"output_tokens must be a non-negative int (got {output_tokens!r})")
        if isinstance(calls, bool) or not isinstance(calls, int) or calls < 0:
            raise RelayError(f"calls must be a non-negative int (got {calls!r})")

        live = self._exists(token)
        if live is None:
            raise CapabilityDenied("unknown")
        if live.revoked:
            raise CapabilityDenied("revoked")

        live.used_output_tokens += output_tokens
        live.used_calls += calls

    def charge_unbudgetable(self, token: str) -> None:
        """Fail-closed metering (F4, bead .56): for a call whose true output-
        token usage could not be determined (usage past a parse bound, an
        over-cap JSON body, or a truncated/timed-out 2xx stream), saturate
        ``used_output_tokens`` to ``max_output_tokens`` so the capability is
        quota-exhausted and the next :meth:`authorize` denies (429
        ``"quota-tokens"``). NEVER charge ~0 for an unmeterable metered call
        — that is the leash bypass this closes. Same "unbudgetable, never
        $0" principle as .51's REPORT-event usage. Raises
        :class:`CapabilityDenied` for an unknown/revoked token (charging a
        dead token is a caller bug), exactly like :meth:`charge`. Idempotent
        — touches only ``used_output_tokens``, never ``used_calls``."""
        live = self._exists(token)
        if live is None:
            raise CapabilityDenied("unknown")
        if live.revoked:
            raise CapabilityDenied("revoked")
        live.used_output_tokens = max(
            live.used_output_tokens, live.capability.max_output_tokens
        )

    def revoke(self, dispatch_id: str) -> None:
        """Kill the capability for ``dispatch_id``. Idempotent: revoking
        twice, or an unknown ``dispatch_id``, is a no-op (never raises) —
        SPEC's dispatch-end teardown must be safe to call unconditionally.
        """
        token = self._token_by_dispatch.get(dispatch_id)
        if token is None:
            return
        live = self._by_token.get(token)
        if live is not None:
            live.revoked = True

    def is_live(self, token: str) -> bool:
        """Non-raising predicate: True iff ``token`` currently authorizes
        (mirrors :meth:`authorize`'s LIVE definition exactly, but never
        raises)."""
        try:
            self.authorize(token)
        except CapabilityDenied:
            return False
        return True

    def usage(self, token: str) -> dict:
        """Return ``{"output_tokens": int, "calls": int}`` for ``token``.
        Raises :class:`RelayError` (not ``CapabilityDenied``) for an
        unknown token — this is an observability read, not an
        authorization decision."""
        live = self._exists(token)
        if live is None:
            raise RelayError(f"unknown token {token!r}")
        return {"output_tokens": live.used_output_tokens, "calls": live.used_calls}


def _find_header(headers: Mapping[str, str], name: str) -> str | None:
    """Case-insensitive header lookup."""
    target = name.lower()
    for key, value in headers.items():
        if key.lower() == target:
            return value
    return None


def extract_usage(response: RelayResponse) -> dict:
    """Parse an Anthropic-style JSON body's top-level ``"usage"`` object.

    Never raises: an unparseable, empty, or bodyless response (or one with
    no ``"usage"`` key) yields ``{}`` — the call still counts toward
    ``max_calls`` via :meth:`CapabilityStore.charge`'s default ``calls=1``,
    it simply charges zero output tokens for that call. Streaming/SSE
    usage-parse fidelity is a .25 concern (bead .12 build spec §0).
    """
    if not response.body:
        return {}
    try:
        parsed = json.loads(response.body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    usage = parsed.get("usage")
    if not isinstance(usage, dict):
        return {}
    return usage


def build_redactor(secrets: Iterable[str]) -> Callable[[str], str]:
    """Build ``f(text) -> text`` replacing every NON-EMPTY secret in
    ``secrets`` with :data:`REDACTED_PLACEHOLDER`.

    Empty strings are skipped (never replace ``""`` — that would insert
    the placeholder between every character). Secrets are replaced
    longest-first so that one secret which is a substring of another
    (e.g. ``"secret"`` inside ``"topsecret"``) does not leave a mangled
    partial match: the longer string is fully redacted first, then the
    shorter one is redacted wherever it still independently occurs.
    """
    non_empty = sorted({s for s in secrets if s}, key=len, reverse=True)

    def _redact(text: str) -> str:
        for secret in non_empty:
            text = text.replace(secret, REDACTED_PLACEHOLDER)
        return text

    return _redact


def worker_credential_env(capability: Capability, *, base_url: str | None = None) -> dict[str, str]:
    """The claude-code-shaped worker env for one dispatch's capability:
    ``{"ANTHROPIC_API_KEY": capability.token}``, plus ``ANTHROPIC_BASE_URL``
    ONLY when ``base_url`` is given.

    Provably contains NO real provider key — only the capability token
    (worthless after :meth:`CapabilityStore.revoke`) and, optionally, a base
    URL. This is the AC4 credential-vacancy target (bead .12 build spec §1).

    bead .25: in production the relay is wired and the ``.63`` worker
    entrypoint OWNS ``ANTHROPIC_BASE_URL`` (it points claude at the in-cage
    loopback relay shim once ``/run/relay.sock`` is mounted). The daemon
    must therefore pass ``base_url=None`` so no host base URL lands in the
    worker env — the entrypoint's value would win anyway, but a dead host
    URL in ``cred_env`` is misleading. ``base_url`` is retained (optional)
    for the legacy/no-relay path (a direct ``ANTHROPIC_BASE_URL``).
    """
    env = {"ANTHROPIC_API_KEY": capability.token}
    if base_url is not None:
        env["ANTHROPIC_BASE_URL"] = base_url
    return env


# Reasons whose deny status is 429 (quota exhaustion) rather than 401
# (identity/authenticity failure) — every other CapabilityDenied reason
# ("unknown", "revoked") maps to 401.
_QUOTA_REASON_PREFIX = "quota-"


def _deny_status(reason: str) -> int:
    return 429 if reason.startswith(_QUOTA_REASON_PREFIX) else 401


@dataclass(frozen=True)
class UpstreamRequest:
    """The result of a successful :func:`prepare_upstream` call: a
    :class:`RelayRequest` with the capability header REMOVED and the real
    provider key injected under the same header name, ready to hand to a
    forwarder. ``token`` is carried alongside (not embedded in the request)
    so the caller can meter the *original* capability after the response
    comes back — the request itself no longer carries the capability token
    anywhere."""

    request: RelayRequest
    token: str


@dataclass(frozen=True)
class DenyResponse:
    """The result of a failed :func:`prepare_upstream` call: a synthesized
    deny outcome, carrying no request/token — the forward path is
    structurally unreachable from this branch (bead .31 build spec case 4).
    ``status`` is 401 for a missing/unknown/revoked capability, 429 for any
    ``quota-*`` reason; ``reason`` is ``"missing-capability"`` or one of
    :class:`CapabilityDenied`'s fixed reason vocabulary."""

    status: int
    reason: str


def prepare_upstream(
    relay: CredentialRelay, request: RelayRequest
) -> UpstreamRequest | DenyResponse:
    """The single authorize+inject authority shared by :meth:`CredentialRelay.handle`
    (synchronous, full-body) and the live streaming transport
    (``relay_transport.serve_relay``, bead .31).

    Reuses ``relay``'s own internals (``_find_header``, ``_store.authorize``,
    ``_key_provider``, ``_capability_header``, and the bead .55 hardening
    config: ``_allowed_endpoints``, ``_upstream_header_allowlist``,
    ``_upstream_headers_pinned``) so both callers observe IDENTICAL
    fail-closed behaviour. Order of checks (bead .55 build spec, normative —
    steps 1-2 must never fetch the real key):

    1. Capability header present? Else :class:`DenyResponse` (401,
       ``"missing-capability"``).
    2. ``(method.upper(), path-without-query)`` in
       ``relay._allowed_endpoints``? Else :class:`DenyResponse` (403,
       ``"forbidden-endpoint"``) — WITHOUT calling ``authorize`` or
       ``key_provider`` (F3: a valid capability must not reach the whole
       provider API).
    3. ``relay._store.authorize(token)`` — a denied capability returns a
       :class:`DenyResponse` (401/429) and calls neither ``relay._forwarder``
       NOR ``relay._key_provider``: the real key is never even fetched on a
       deny path.
    4. Only on success: ``key_provider()`` is invoked and an
       :class:`UpstreamRequest` is built. The upstream header set (F2) is
       built by allowlist, not by copy-all-minus-capability: ONLY worker
       headers whose lowercased name is in ``relay._upstream_header_allowlist``
       are copied, then ``relay._upstream_headers_pinned`` is overlaid (the
       relay's pinned values WIN over anything the worker sent), then the
       real key is injected under ``relay._capability_header``. Everything
       else the worker sent (``authorization``, ``anthropic-beta``, ``host``,
       ``x-*``, cookies, proxy-*, ...) is dropped. Method/path/body pass
       through unchanged.
    """
    token = _find_header(request.headers, relay._capability_header)
    if token is None:
        return DenyResponse(status=401, reason="missing-capability")

    endpoint_path = request.path.split("?", 1)[0]
    if (request.method.upper(), endpoint_path) not in relay._allowed_endpoints:
        return DenyResponse(status=403, reason="forbidden-endpoint")

    try:
        relay._store.authorize(token)
    except CapabilityDenied as exc:
        return DenyResponse(status=_deny_status(exc.reason), reason=exc.reason)

    real_key = relay._key_provider()

    cap_header_lower = relay._capability_header.lower()
    upstream_headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() != cap_header_lower and key.lower() in relay._upstream_header_allowlist
    }
    # The relay's pinned headers win over anything the (allowlisted) worker
    # value already carries — e.g. the worker's own `anthropic-version` is
    # never allowlisted, but even if it were, the pinned value overlays it.
    for pinned_name, pinned_value in relay._upstream_headers_pinned.items():
        upstream_headers = {
            key: value
            for key, value in upstream_headers.items()
            if key.lower() != pinned_name.lower()
        }
        upstream_headers[pinned_name] = pinned_value
    upstream_headers[relay._capability_header] = real_key

    upstream_request = RelayRequest(
        method=request.method,
        path=request.path,
        headers=upstream_headers,
        body=request.body,
    )
    return UpstreamRequest(request=upstream_request, token=token)


class CredentialRelay:
    """Terminates the worker's inference request, swaps its capability
    token for the real provider key toward upstream, and never lets the
    real key travel back toward the worker (bead .12 build spec §1).

    ``key_provider``/``forwarder`` are injected callables — mirrors
    ``critic.py``'s injected-``client`` discipline exactly. This class
    never imports a provider SDK and never performs real network I/O;
    production wires a real HTTPS forwarder and an ``op``-backed key
    provider (.25), tests inject fakes.
    """

    def __init__(
        self,
        *,
        store: CapabilityStore,
        key_provider: Callable[[], str],
        forwarder: Callable[[RelayRequest], RelayResponse],
        usage_extractor: Callable[[RelayResponse], dict] = extract_usage,
        capability_header: str = CAPABILITY_HEADER_DEFAULT,
        upstream_header_allowlist: frozenset[str] = DEFAULT_UPSTREAM_HEADER_ALLOWLIST,
        upstream_headers_pinned: Mapping[str, str] | None = None,
        allowed_endpoints: frozenset[tuple[str, str]] = DEFAULT_ALLOWED_ENDPOINTS,
    ) -> None:
        self._store = store
        self._key_provider = key_provider
        self._forwarder = forwarder
        self._usage_extractor = usage_extractor
        self._capability_header = capability_header
        # bead .55 hardening config (F2/F3) — see module-level defaults'
        # docstring for the production-wiring note on pinned headers.
        self._upstream_header_allowlist = upstream_header_allowlist
        self._upstream_headers_pinned: Mapping[str, str] = (
            upstream_headers_pinned if upstream_headers_pinned is not None else {}
        )
        self._allowed_endpoints = allowed_endpoints

    def handle(self, request: RelayRequest) -> RelayResponse:
        """Authorize, inject, forward, meter — fail closed at every step.

        1. Extract the capability token from ``request.headers`` (case-
           insensitive lookup of ``capability_header``). A missing header
           is treated exactly like an unknown token: deny, status 401,
           ``forwarder``/``key_provider`` never called.
        2. ``store.authorize(token)``. On :class:`CapabilityDenied`,
           return a synthesized deny :class:`RelayResponse` — status 401
           for ``"unknown"``/``"revoked"``, 429 for any ``"quota-*"``
           reason — WITHOUT calling ``forwarder`` and WITHOUT calling
           ``key_provider`` (the real key is never even fetched on a deny
           path; mirrors ``egress_proxy.py``'s "deny never touches
           upstream").
        3. Only once authorized: build the upstream request by copying
           the headers, REMOVING the capability header, and INJECTING the
           real key (from ``key_provider()``) under the same header name.
           Method/path/body pass through unchanged.
        4. ``forwarder(upstream_request)`` -> the real response.
        5. Meter usage: ``usage_extractor(response)`` then
           ``store.charge(token, output_tokens=...)`` (defaults to 0 if
           the extractor found nothing; the call still counts via
           ``charge``'s default ``calls=1``).
        6. Return the forwarder's response AS-IS toward the worker — the
           real key is never added to anything returned here.

        As of bead .31, this delegates the authorize+inject decision to the
        module-level :func:`prepare_upstream` (the single authority shared
        with the live streaming transport) — behaviour is unchanged, this
        is a pure refactor.
        """
        prepared = prepare_upstream(self, request)
        if isinstance(prepared, DenyResponse):
            return RelayResponse(status=prepared.status, headers={}, body=b"")

        response = self._forwarder(prepared.request)

        usage = self._usage_extractor(response)
        output_tokens = usage.get("output_tokens", 0) if isinstance(usage, dict) else 0
        if not isinstance(output_tokens, int) or isinstance(output_tokens, bool) or (
            output_tokens < 0
        ):
            output_tokens = 0
        self._store.charge(prepared.token, output_tokens=output_tokens)

        return response

    def injected_secrets(self, token: str) -> frozenset[str]:
        """The values this relay put on the wire for ``token`` that MUST
        be redacted from any sealed transcript: the capability token
        itself and the real provider key. Raises :class:`RelayError` for
        a token that was never minted.

        Deliberately checks raw registration (via
        ``CapabilityStore._exists``) rather than ``authorize`` — sealing a
        transcript happens AFTER dispatch end, i.e. after
        :meth:`CapabilityStore.revoke` has already run, so this must
        still work for a revoked token; routing through ``authorize``
        would wrongly raise :class:`CapabilityDenied` for the exact case
        this method exists to serve.
        """
        live = self._store._exists(token)
        if live is None:
            raise RelayError(f"unknown token {token!r}")
        return frozenset({token, self._key_provider()})
