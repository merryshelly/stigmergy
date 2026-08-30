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
from dataclasses import dataclass, field
from typing import Any

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


# --------------------------------------------------------------------------- #
# bead .147 — relay endpoint profiles (per-dispatch upstream selection).      #
#                                                                             #
# Pre-`.147` the relay served exactly ONE upstream (Anthropic /v1/messages,  #
# x-api-key header, Anthropic-shaped metering) as module constants. `.147`   #
# makes that choice PER-DISPATCH: the daemon derives a frozen RelayProfile   #
# from the dispatch lane's registry entry + pricing class, and the cli       #
# closure builds the CredentialRelay/forwarder from it (spec §1A/§1F). The   #
# DEFAULT profile below reproduces the pre-`.147` constants EXACTLY          #
# (regression gate: the full existing suite must pass with zero edits —      #
# every code path that does not supply a profile observes today's            #
# behaviour byte-for-byte).                                                  #
# --------------------------------------------------------------------------- #

# The pre-`.147` upstream constants, now the DEFAULT profile's values.
_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
_ANTHROPIC_VERSION = "2023-06-01"
# OpenAI wire: worker-facing path is `/chat/completions` WITHOUT `/v1` (the
# driver sets the worker's OPENAI_BASE_URL to the shim root) while
# `upstream_base_url` CARRIES `/v1`, so `base_url + path` composition in
# make_urllib_forwarder is exact (spec §1A: NO rewrite mechanism).
_OPENAI_ENDPOINT: tuple[str, str] = ("POST", "/chat/completions")

#: The `stream_options.include_usage` deny's fixed marker reason (bead .147
#: §1D, verify-don't-mutate). The only non-:class:`CapabilityDenied` reason
#: in the relay deny vocabulary: a POLICY deny (status 422, nothing
#: forwarded, nothing charged — distinct from the .56 §C upstream-failure
#: charges), issued on a metered OpenAI lane when a streaming request does
#: not request usage.
MISSING_USAGE_REQUESTED = "missing-usage-requested"


def _derive_pricing_class(pricing: Any) -> str:
    """Map a registry pricing value to the relay's two-valued pricing class
    (bead .147 §1A/§1F): ``subscription`` -> ``metered`` (the subscription
    quota IS the metering leash — .56 §C's no-unmetered-retry property
    applies), ``local`` -> ``local``, and EVERYTHING else (any other value,
    a future class, or missing/``None``) -> ``metered``. D10 fail-closed:
    ``$0`` (i.e. ``local``) is only ever a declared value, never a fallback
    — an unknown pricing must never weaken the metering guarantee."""
    if pricing == "local":
        return "local"
    # "subscription", "metered", and any other/missing value -> metered.
    return "metered"


@dataclass(frozen=True)
class RelayProfile:
    """One dispatch's relay endpoint profile (bead .147 §1A, immutable).

    Constructed per dispatch by :func:`derive_relay_profile` (daemon side)
    or taken from :data:`DEFAULT_RELAY_PROFILE` (the cli fallback). The
    forwarder keeps composing ``upstream_base_url + worker path`` with
    netloc pinning — there is NO path-rewrite mechanism; ``allowed_endpoints``
    plus the base URL's own prefix define the exact upstream surface a
    capability may reach (one capability = one upstream = one endpoint).
    """

    upstream_base_url: str  # origin+prefix, e.g. "http://10.0.20.111:8000/v1"
    wire: str  # "anthropic" | "openai" — metering/contract selection axis
    auth: str  # "x-api-key" | "bearer" | "none" — upstream credential contract
    capability_header: str  # derived, not independent: where the capability rides
    pricing_class: str  # "metered" | "local" — DERIVED (see derive_relay_profile)
    allowed_endpoints: frozenset[tuple[str, str]]  # worker-facing (method, path) pairs
    pinned_headers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # bead .147 §2.5: unknown wire/auth/pricing is a BUILD error (profile
        # construction), never a request-time surprise — and the pricing
        # class can only ever be one of the two valid values, so there is no
        # "unknown pricing" fallthrough to `local` (D10 fail-closed).
        if self.wire not in ("anthropic", "openai"):
            raise RelayError(
                f"unknown relay wire {self.wire!r} (expected 'anthropic' or 'openai')"
            )
        if self.auth not in ("x-api-key", "bearer", "none"):
            raise RelayError(
                f"unknown relay auth {self.auth!r} (expected 'x-api-key', 'bearer' or 'none')"
            )
        if self.pricing_class not in ("metered", "local"):
            raise RelayError(
                f"unknown pricing_class {self.pricing_class!r} (expected 'metered' or 'local')"
            )
        if not self.upstream_base_url or "://" not in self.upstream_base_url:
            raise RelayError(
                "upstream_base_url must be an absolute scheme://host[/prefix] URL "
                f"(got {self.upstream_base_url!r})"
            )
        if not self.capability_header:
            raise RelayError("capability_header must be a non-empty header name")

    @property
    def is_metered(self) -> bool:
        """True iff undeterminable usage on a 2xx saturates the capability
        (``charge_unbudgetable``). ``local`` lanes are accounting-only
        (bead .147 §1C): best-effort charge of the parsed value, 0 when
        unparseable, JSONL reason ``accounting-only`` — never saturated."""
        return self.pricing_class == "metered"


def derive_relay_profile(
    *,
    lane_model: str,
    registry: Any,
    charter: Any = None,
) -> RelayProfile:
    """Derive the per-dispatch :class:`RelayProfile` from the lane's resolved
    registry entry (bead .147 §1A/§1F — called by the daemon in the dispatch
    path, immediately before the ``_setup_relay`` call).

    Inputs: ``lane_model`` (the charter lane's model name — the registry
    entry name), ``registry`` (a :class:`stigmergy.registry.Registry`), and
    ``charter`` (optional; accepted so the derivation surface is complete for
    future charter-level knobs — the v0 derivation reads ONLY the registry
    entry, the charter's lane fields being the same model name the registry
    entry is keyed by).

    Derivation (normative, spec §1A):

    - ``pricing_class``: the entry's ``pricing`` value mapped
      ``subscription`` -> ``metered`` (the subscription quota IS the metering
      leash), ``local`` -> ``local``, anything else/missing -> ``metered``
      (D10: `$0` only ever as a declared value, never a fallback).
    - ``wire``: the entry's ``oa_type`` (bead .143 OA wiring axis) —
      ``"anthropic"`` or ``"openai"``; anything else (including missing/
      ``None``) is a BUILD error (:class:`RelayError`), never a
      request-time guess.
    - ``upstream_base_url``: the entry's ``oa_base_url`` when the wire is
      ``openai`` (REQUIRED there — a base-less OpenAI upstream is unbudgetable
      to relay to); the fixed ``_ANTHROPIC_BASE_URL`` for ``anthropic``
      (today's exact value, whether or not the entry declares one — the
      anthropic lane keeps today's byte-for-byte behaviour).
    - ``auth``: for the ``openai`` wire, DERIVED FROM THE PRICING CLASS
      (bead .147 review fix F1, 2026-08-30): ``local`` (declared $0, the
      blackwell class) -> ``"none"`` (keyless — the header is omitted
      upstream entirely); every metered class -> ``"bearer"`` (the OpenAI
      credential contract, spec §1B). Fail-safe direction: a $0 upstream
      can never receive a foreign provider's key (the bug this closes: a
      `local` openai lane deriving `bearer` would inject the Synthetic key
      toward blackwell); a local upstream that DID need a key fails loudly
      at the upstream (401 -> infra), never by leaking. A future
      keyed-local upstream will need an explicit registry auth field —
      not invented silently here. ``"x-api-key"`` for ``anthropic``.
      (``"none"`` can also be constructed directly for keyless upstreams;
      ``derive_relay_profile`` derives it only from declared
      ``pricing="local"``.)
    - ``capability_header``: ``"authorization"`` for the ``openai`` wire,
      ``"x-api-key"`` for ``anthropic`` (derived, not independent).
    - ``allowed_endpoints``: ``{("POST", "/v1/messages")}`` for ``anthropic``
      (today's exact set, incl. the `.25` ``anthropic-beta`` handling living
      in the allowlist, not here), ``{("POST", "/chat/completions")}`` for
      ``openai``.
    - ``pinned_headers``: ``{"anthropic-version": _ANTHROPIC_VERSION}`` for
      ``anthropic`` (today's exact pin), ``{}`` for ``openai`` (no version
      pin on the OpenAI wire).

    Raises :class:`RelayError` on an unknown/missing wire, a missing
    ``oa_base_url`` on an openai entry, or any registry failure (the
    registry's own :class:`UnbudgetableError` propagates — an unresolvable
    model is unbudgetable, and the daemon's existing registry-error handling
    owns that path).
    """
    entry = registry.resolve(lane_model)
    pricing_raw = getattr(entry, "pricing", None)
    pricing_value = getattr(pricing_raw, "value", pricing_raw)
    pricing_class = _derive_pricing_class(pricing_value)

    oa_type = getattr(entry, "oa_type", None)
    if oa_type == "anthropic":
        return RelayProfile(
            upstream_base_url=_ANTHROPIC_BASE_URL,
            wire="anthropic",
            auth="x-api-key",
            capability_header=CAPABILITY_HEADER_DEFAULT,
            pricing_class=pricing_class,
            allowed_endpoints=DEFAULT_ALLOWED_ENDPOINTS,
            pinned_headers={"anthropic-version": _ANTHROPIC_VERSION},
        )
    if oa_type == "openai":
        base_url = getattr(entry, "oa_base_url", None)
        if not base_url:
            raise RelayError(
                f"model {lane_model!r} declares wire 'openai' but has no "
                "oa_base_url — an OpenAI relay upstream must be declared "
                "explicitly (silent base-URL guessing is unbudgetable)"
            )
        # Review fix F1 (bead .147, 2026-08-30): auth derives from the
        # PRICING CLASS, fail-safe. `local` ($0 declared — the blackwell
        # class) is KEYLESS: deriving `bearer` there would inject the
        # Synthetic key toward a foreign LAN upstream (the exact wiring bug
        # this closes). A local upstream that actually needed a key fails
        # loudly at the upstream instead of leaking one. Metered classes
        # carry the OpenAI Bearer credential contract (spec §1B).
        auth: str = "none" if pricing_class == "local" else "bearer"
        return RelayProfile(
            upstream_base_url=base_url,
            wire="openai",
            auth=auth,
            capability_header="authorization",
            pricing_class=pricing_class,
            allowed_endpoints=frozenset({_OPENAI_ENDPOINT}),
            pinned_headers={},
        )
    # Unknown/missing wire: BUILD error (spec §1A/§2.5), never a request-
    # time surprise. `charter` is unused in v0 (the lane's model IS the
    # registry key) but is part of the derivation surface.
    raise RelayError(
        f"model {lane_model!r}: unknown/missing OA wire {oa_type!r} — a relay "
        "profile requires a declared 'anthropic' or 'openai' wire "
        "(silent wire-type guessing is unbudgetable)"
    )


#: The pre-`.147` default endpoint profile — byte-identical to the module
#: constants it replaces (``_ANTHROPIC_BASE_URL`` + ``DEFAULT_ALLOWED_
#: ENDPOINTS`` + ``CAPABILITY_HEADER_DEFAULT`` + the `.25` pinned
#: ``anthropic-version``). The cli closure falls back to THIS when the
#: daemon's per-dispatch profile cell carries no entry, so existing stubs
#: and wiring tests keep observing today's relay shape.
DEFAULT_RELAY_PROFILE = RelayProfile(
    upstream_base_url=_ANTHROPIC_BASE_URL,
    wire="anthropic",
    auth="x-api-key",
    capability_header=CAPABILITY_HEADER_DEFAULT,
    pricing_class="metered",
    allowed_endpoints=DEFAULT_ALLOWED_ENDPOINTS,
    pinned_headers={"anthropic-version": _ANTHROPIC_VERSION},
)


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


def parse_bearer_token(header_value: str) -> str:
    """bead .147 §1B: strip an OPTIONAL ``Bearer `` prefix from the worker's
    ``Authorization`` header value for the capability lookup.

    Case-insensitive on the scheme token (``bearer``/``BEARER``/
    ``Bearer`` all strip; a different scheme or any other value is NOT a
    capability credential and is returned unchanged — the store lookup
    then fails ``unknown``, exactly as a bare unknown token would). A
    scheme with NO space after it (``Bearer<token>``) is not the Bearer
    form and is likewise returned unchanged. Surrounding whitespace is
    stripped in all cases (the header parser strips outer whitespace; this
    tolerates stray internal padding)."""
    value = header_value.strip()
    # The 7-char prefix already ends in the required space; any further
    # whitespace belongs to (or pads) the token and is stripped below.
    if len(value) > 7 and value[:7].lower() == "bearer ":
        return value[7:].strip()
    return value


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


def extract_openai_usage(response: RelayResponse) -> dict:
    """Parse an OpenAI-wire JSON body's top-level ``"usage"`` object
    (bead .147 §1C — the JSON / non-stream shape selected when
    ``profile.wire == "openai"``).

    Same tri-state discipline as :func:`extract_usage`: never raises; an
    unparseable, empty, bodyless, or usage-less response yields ``{}``.
    The CALLER applies the charge: the capability charge is
    ``usage.completion_tokens`` (a valid non-negative int, else
    undetermined — see :meth:`CredentialRelay._meter_usage` / the
    transport's OpenAI JSON branch), while the FULL parsed usage dict
    (including ``prompt_tokens`` / ``prompt_tokens_details.cached_tokens``
    / ``reasoning_tokens``) is what the relay JSONL records as the
    CV/quota-governor feed.
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


def openai_output_tokens(usage: Any) -> int | None:
    """bead .147 §1C: the OpenAI-wire output measurement — the ``usage``
    dict's ``completion_tokens`` when POSITIVELY determined (a valid
    non-negative int; bool rejected), else ``None`` (undetermined — the
    caller applies the pricing-class re-scope: ``metered`` ->
    ``charge_unbudgetable``, ``local`` -> 0).

    Shared by the JSON path (:meth:`CredentialRelay._meter_usage`, via the
    extractor's ``output_tokens`` alias below) and the SSE meter's
    finalize, so both OpenAI shapes observe one tri-state rule."""
    if not isinstance(usage, dict):
        return None
    raw = usage.get("completion_tokens")
    if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
        return raw
    return None


def openai_output_tokens_alias(usage: Any) -> int | None:
    """Alias mapping the OpenAI ``usage.completion_tokens`` field onto the
    ``output_tokens`` key the shared charge code reads (see
    :func:`openai_output_tokens` — same tri-state rule, different field
    name)."""
    return openai_output_tokens(usage)


def _openai_json_usage_extractor(response: RelayResponse) -> dict:
    """usage_extractor factory output for OpenAI-wire JSON relays: the
    full parsed ``usage`` dict with its ``output_tokens`` slot set to the
    tri-state-validated ``completion_tokens`` (``None`` when undetermined)
    so the shared charge code (:meth:`CredentialRelay._meter_usage`) can
    stay wire-agnostic."""
    usage = extract_openai_usage(response)
    if not usage:
        return usage
    out = dict(usage)
    out["output_tokens"] = openai_output_tokens(usage)
    return out


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
       real key is injected per ``relay._auth`` (bead .147 §1B):
       ``"x-api-key"`` injects the key under ``relay._capability_header``
       (today's exact behaviour); ``"bearer"`` injects
       ``"Bearer <real-key>"`` under the (authorization) capability header;
       ``"none"`` OMITS the header entirely (a non-None key there is a
       wiring bug -> :class:`RelayError`). Everything else the worker sent
       (``authorization``, ``anthropic-beta``, ``host``, ``x-*``, cookies,
       proxy-*, ...) is dropped. Method/path/body pass through unchanged.
    """
    token = _find_header(request.headers, relay._capability_header)
    if token is None:
        return DenyResponse(status=401, reason="missing-capability")
    # bead .147 §1B: on the `authorization` (bearer) capability header the
    # worker may send `Bearer <capability-token>` — strip the OPTIONAL
    # scheme prefix (case-insensitive) for the lookup. On `x-api-key` this
    # is a no-op (no "Bearer " prefix is ever a valid x-api-key value), so
    # the anthropic lane is byte-for-byte unchanged.
    if relay._capability_header.lower() == "authorization":
        token = parse_bearer_token(token)

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

    # Upstream credential injection (bead .147 §1B):
    #  - auth="x-api-key": today's exact behaviour — the real key under the
    #    (removed) capability header name.
    #  - auth="bearer": `Authorization: Bearer <real-key>` — the OpenAI
    #    wire's credential form (the worker's own `Authorization` header
    #    was already dropped by the allowlist above; the injected value
    #    REPLACES it, never appends).
    #  - auth="none": the header is OMITTED upstream entirely (blackwell is
    #    keyless) — NEVER `Bearer None`, never a placeholder credential.
    #    `real_key` is None here (the profile's key_provider returns None);
    #    a non-None key on a none-auth relay is a wiring bug and still
    #    omits the header (omission is the only safe shape).
    if relay._auth == "bearer":
        upstream_headers[relay._capability_header] = f"Bearer {real_key}"
    elif relay._auth == "x-api-key":
        upstream_headers[relay._capability_header] = real_key
    else:  # "none"
        if real_key is not None:
            raise RelayError(
                "auth='none' relay profile with a non-None key — the key "
                "provider must return None on keyless lanes (the header "
                "is omitted upstream, never `Bearer <key>`)"
            )

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
        key_provider: Callable[[], str | None],
        forwarder: Callable[[RelayRequest], RelayResponse],
        usage_extractor: Callable[[RelayResponse], dict] = extract_usage,
        capability_header: str = CAPABILITY_HEADER_DEFAULT,
        upstream_header_allowlist: frozenset[str] = DEFAULT_UPSTREAM_HEADER_ALLOWLIST,
        upstream_headers_pinned: Mapping[str, str] | None = None,
        allowed_endpoints: frozenset[tuple[str, str]] = DEFAULT_ALLOWED_ENDPOINTS,
        auth: str = "x-api-key",
        pricing_class: str = "metered",
        wire: str = "anthropic",
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
        # bead .147 §1A/§1B: the per-profile wire, credential contract, and
        # pricing class. Defaults are TODAY'S exact shape (anthropic wire,
        # x-api-key, metered), so a relay built without them (all
        # pre-`.147` construction sites) behaves byte-for-byte as before.
        self._wire = wire
        self._auth = auth
        self._pricing_class = pricing_class

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
           real key (from ``key_provider()``) per ``self._auth`` (see
           :func:`prepare_upstream`). Method/path/body pass through
           unchanged.
        4. ``forwarder(upstream_request)`` -> the real response.
        5. Meter usage (bead .147 §1C re-scope, gated on
           ``self._pricing_class``): the extractor's ``output_tokens``
           must be a POSITIVELY-determined valid non-negative int.
           - ``metered``: a determined value charges exactly; an
             undetermined value (absent/non-int/bool/negative) charges
             ``charge_unbudgetable`` (saturate = kill switch — the .56 F4
             leash, unchanged from today's behaviour).
           - ``local``: accounting-only — a determined value charges
             best-effort; an undetermined value charges 0 (never
             saturate a $0 capability for a metering gap).
           The call still counts via ``charge``'s default ``calls=1``.
        6. Return the forwarder's response AS-IS toward the worker — the
           real key is never added to anything returned here; the
           response's ``usage`` slot carries the charged usage dict
           (the full parsed ``usage`` object on OpenAI wires, ``{}`` when
           nothing was determined) for observability only.

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
        charged_usage = self._meter_usage(prepared.token, usage)
        return RelayResponse(
            status=response.status,
            headers=response.headers,
            body=response.body,
            usage=charged_usage,
        )

    def _meter_usage(self, token: str, usage: dict) -> dict:
        """bead .147 §1C: the pricing-class-gated metering decision shared
        by :meth:`handle` (the .12 sync full-body path).

        The tri-state validity rule is IDENTICAL to the transport's JSON
        path (bead .56 §8 fix A): ``output_tokens`` is a POSITIVELY
        determined measurement only when present as a valid non-negative
        int (bool rejected). A valid value charges exactly; an
        undetermined value is ``charge_unbudgetable`` (saturate) on
        ``metered`` lanes and a 0-charge on ``local`` lanes (accounting-
        only — never saturate a $0 capability). Returns the charged usage
        dict (the full parsed ``usage`` when a measurement was determined,
        ``{}`` otherwise) for the response's observability slot."""
        output_tokens: int | None
        raw = usage.get("output_tokens") if isinstance(usage, dict) else None
        if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
            output_tokens = raw
        else:
            output_tokens = None

        if output_tokens is None:
            if self._pricing_class == "metered":
                # .56 F4 fail-closed: an unmeterable metered call never
                # charges ~0 — it saturates the capability (kill switch).
                try:
                    self._store.charge_unbudgetable(token)
                except CapabilityDenied:
                    pass  # dead token in the sync window: fail closed
                return {}
            # local: accounting-only, best-effort 0 (JSONL reason
            # "accounting-only" is the transport's concern; this sync path
            # has no JSONL).
            self._store.charge(token, output_tokens=0)
            return {}
        self._store.charge(token, output_tokens=output_tokens)
        return usage

    def injected_secrets(self, token: str) -> frozenset[str]:
        """The values this relay put on the wire for ``token`` that MUST
        be redacted from any sealed transcript: the capability token
        itself and the real provider key. Raises :class:`RelayError` for
        a token that was never minted.

        bead .147 §1B: on a keyless (``auth="none"``) lane the key
        provider returns ``None`` — the secrets set is just the capability
        token (nothing else was ever put on the wire), so sealing works
        unchanged on no-auth lanes.

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
        real_key = self._key_provider()
        if real_key is None:
            return frozenset({token})
        return frozenset({token, real_key})
