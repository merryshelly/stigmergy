"""Model pricing registry (SPEC.md §5 `[models]`, §4 credentials).

Parses a `models.toml` file mapping model name -> pricing entry. Three pricing
classes exist:

- ``metered``      — per-token USD (``input_usd_per_mtok`` / ``output_usd_per_mtok``
                      required, both strictly > 0; optional ``cached_usd_per_mtok`` /
                      ``reasoning_usd_per_mtok`` if present must also be > 0). Counts
                      against the run's USD spend leash.
- ``local``        — DECLARED marginal cost (``marginal_usd``, may be exactly 0.0)
                      that additionally requires ``approved = true`` — an unapproved
                      local entry is a declared-but-unapproved cost and is rejected.
- ``subscription`` — ``marginal_usd`` must be exactly 0.0, plus a ``quota`` descriptor
                      string; quota exhaustion is handled elsewhere (infra backoff).

**Governing invariant:** `$0` is legal ONLY as an explicitly declared `local` or
`subscription` value — never as a fallback for missing data. A registry miss, or
any missing/invalid required price field, raises :class:`UnbudgetableError`. There
is no code path in this module where absent price data silently resolves to `$0`
or any other default.
"""

from __future__ import annotations

import enum
import hashlib
import json
import math
import os
import tomllib
from dataclasses import dataclass, replace
from typing import Any


class UnbudgetableError(Exception):
    """Raised when a model cannot be priced.

    Covers both a registry miss (unknown model name) and a registry entry
    that fails validation (missing/invalid/forbidden price data). Either way
    the model is "unbudgetable" and the caller must refuse to run it.
    """


class PricingClass(enum.Enum):
    METERED = "metered"
    LOCAL = "local"
    SUBSCRIPTION = "subscription"


@dataclass(frozen=True)
class ModelEntry:
    """A single validated registry entry.

    Only the fields relevant to ``pricing`` are populated; the rest stay
    ``None``. Which fields apply is determined entirely by ``pricing``:

    - ``METERED``: ``input_usd_per_mtok``, ``output_usd_per_mtok`` (always set),
      ``cached_usd_per_mtok``, ``reasoning_usd_per_mtok`` (set only if declared).
    - ``LOCAL``: ``marginal_usd`` (>= 0.0), ``approved`` (always ``True``).
    - ``SUBSCRIPTION``: ``marginal_usd`` (always exactly 0.0), ``quota``.

    The ``oa_*`` fields (bead .143) are the OA provider-layer wiring axis —
    ALWAYS populated by the loader: absent in the TOML means
    derive-by-convention (Anthropic entries default to
    ``oa_provider_key="anthropic"`` / ``oa_type="anthropic"`` /
    ``oa_base_url=None``, exactly reproducing the pre-.143 routing; a
    non-Anthropic ``provider`` REQUIRES ``oa_type`` — see
    :func:`_oa_wiring_fields`).
    """

    name: str
    provider: str
    family: str
    version: str
    pricing: PricingClass
    input_usd_per_mtok: float | None = None
    output_usd_per_mtok: float | None = None
    cached_usd_per_mtok: float | None = None
    reasoning_usd_per_mtok: float | None = None
    marginal_usd: float | None = None
    approved: bool | None = None
    quota: str | None = None
    # bead .143 — OA provider-layer wiring (additive; see class docstring).
    oa_provider_key: str | None = None
    oa_type: str | None = None
    oa_base_url: str | None = None


class Registry:
    """Resolved, validated model registry."""

    def __init__(self, entries: dict[str, ModelEntry], version_hash: str) -> None:
        self.entries = entries
        self.version_hash = version_hash

    def resolve(self, name: str) -> ModelEntry:
        """Look up a model entry by registry alias, or by the QUALIFIED
        ``provider/version`` form (bead .172: the decompose CLI's default
        model string is qualified, not an alias, and must still price).

        On an alias miss, entries are scanned in registry (source) order and
        the FIRST entry whose (provider, version) matches is returned —
        deterministic; two aliases may share one provider/version. A miss on
        both axes raises :class:`UnbudgetableError` — an unknown model is
        unbudgetable, never defaulted to $0 or any other price.
        """
        entry = self.entries.get(name)
        if entry is not None:
            return entry
        provider, sep, version = name.partition("/")
        if sep:
            for candidate in self.entries.values():
                if candidate.provider == provider and candidate.version == version:
                    return candidate
        raise UnbudgetableError(
            f"model {name!r} is not in the registry: unbudgetable, refusing"
        )


def _canonical_hash(data: dict[str, Any]) -> str:
    """sha256 over a canonical (sorted-key, whitespace-free) JSON serialization.

    ``sort_keys=True`` sorts dict keys at every nesting level, so the hash is
    invariant to the order entries or fields appear in the source TOML.
    """
    blob = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _require_str(name: str, raw: dict[str, Any], field: str) -> str:
    if field not in raw:
        raise UnbudgetableError(f"model {name!r}: missing required field {field!r}")
    value = raw[field]
    if not isinstance(value, str) or not value:
        raise UnbudgetableError(f"model {name!r}: field {field!r} must be a non-empty string")
    return value


def _positive_float(
    name: str, raw: dict[str, Any], field: str, *, required: bool
) -> float | None:
    """Fetch a strictly-positive numeric price field.

    Missing required fields raise rather than defaulting to 0.0. A present
    but non-positive value (e.g. explicit ``0``) is rejected too — for
    metered pricing, ``$0`` is the forbidden fallback disguised as data.
    """
    if field not in raw:
        if required:
            raise UnbudgetableError(f"model {name!r}: missing required metered field {field!r}")
        return None
    value = raw[field]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise UnbudgetableError(f"model {name!r}: field {field!r} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise UnbudgetableError(f"model {name!r}: field {field!r} must be a finite number")
    if value <= 0:
        raise UnbudgetableError(
            f"model {name!r}: metered field {field!r} must be > 0 (got {value!r}); "
            "$0 metered pricing is forbidden — $0 is only legal as a declared "
            "local/subscription value"
        )
    return value


def _numeric(name: str, raw: dict[str, Any], field: str) -> float:
    if field not in raw:
        raise UnbudgetableError(f"model {name!r}: missing required field {field!r}")
    value = raw[field]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise UnbudgetableError(f"model {name!r}: field {field!r} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise UnbudgetableError(f"model {name!r}: field {field!r} must be a finite number")
    return value


def _build_metered(
    name: str, raw: dict[str, Any], provider: str, family: str, version: str
) -> ModelEntry:
    input_price = _positive_float(name, raw, "input_usd_per_mtok", required=True)
    output_price = _positive_float(name, raw, "output_usd_per_mtok", required=True)
    cached_price = _positive_float(name, raw, "cached_usd_per_mtok", required=False)
    reasoning_price = _positive_float(name, raw, "reasoning_usd_per_mtok", required=False)
    return ModelEntry(
        name=name,
        provider=provider,
        family=family,
        version=version,
        pricing=PricingClass.METERED,
        input_usd_per_mtok=input_price,
        output_usd_per_mtok=output_price,
        cached_usd_per_mtok=cached_price,
        reasoning_usd_per_mtok=reasoning_price,
    )


def _build_local(
    name: str, raw: dict[str, Any], provider: str, family: str, version: str
) -> ModelEntry:
    marginal = _numeric(name, raw, "marginal_usd")
    if marginal < 0:
        raise UnbudgetableError(f"model {name!r}: 'marginal_usd' must be >= 0 (got {marginal!r})")

    approved = raw.get("approved", False)
    if approved is not True:
        raise UnbudgetableError(
            f"model {name!r}: local pricing declared but not human-approved "
            "(requires `approved = true`); refusing to load"
        )

    return ModelEntry(
        name=name,
        provider=provider,
        family=family,
        version=version,
        pricing=PricingClass.LOCAL,
        marginal_usd=marginal,
        approved=True,
    )


def _build_subscription(
    name: str, raw: dict[str, Any], provider: str, family: str, version: str
) -> ModelEntry:
    marginal = _numeric(name, raw, "marginal_usd")
    if marginal != 0.0:
        raise UnbudgetableError(
            f"model {name!r}: subscription 'marginal_usd' must be exactly 0.0 "
            f"(got {marginal!r})"
        )

    quota = raw.get("quota")
    if not isinstance(quota, str) or not quota:
        raise UnbudgetableError(f"model {name!r}: subscription pricing requires 'quota'")

    return ModelEntry(
        name=name,
        provider=provider,
        family=family,
        version=version,
        pricing=PricingClass.SUBSCRIPTION,
        marginal_usd=0.0,
        quota=quota,
    )


def _build_entry(name: str, raw: Any) -> ModelEntry:
    if not isinstance(raw, dict):
        raise UnbudgetableError(f"model {name!r}: entry must be a table")

    provider = _require_str(name, raw, "provider")
    family = _require_str(name, raw, "family")
    version = _require_str(name, raw, "version")

    if "pricing" not in raw:
        raise UnbudgetableError(f"model {name!r}: missing required field 'pricing'")
    pricing_raw = raw["pricing"]
    try:
        pricing = PricingClass(pricing_raw)
    except ValueError:
        raise UnbudgetableError(f"model {name!r}: unknown pricing class {pricing_raw!r}") from None

    if pricing is PricingClass.METERED:
        entry = _build_metered(name, raw, provider, family, version)
    elif pricing is PricingClass.LOCAL:
        entry = _build_local(name, raw, provider, family, version)
    elif pricing is PricingClass.SUBSCRIPTION:
        entry = _build_subscription(name, raw, provider, family, version)
    else:
        raise AssertionError("unreachable: PricingClass has no other members")  # pragma: no cover

    oa_provider_key, oa_type, oa_base_url = _oa_wiring_fields(name, raw, provider)
    return replace(entry, oa_provider_key=oa_provider_key, oa_type=oa_type, oa_base_url=oa_base_url)


def _oa_wiring_fields(
    name: str, raw: dict[str, Any], provider: str
) -> tuple[str, str, str | None]:
    """Resolve the optional ``oa_*`` provider-wiring fields (bead .143).

    Derive-by-convention, fail-loud:

    - ``provider == "anthropic"``: absent fields default to
      ``oa_provider_key="anthropic"`` / ``oa_type="anthropic"`` /
      ``oa_base_url=None`` — byte-parity with the pre-.143 routing (the
      registry's ``provider`` WAS the OA routing decision then).
    - ``provider == "local"``: no remote provider is routed, so the OA
      wiring stays ``None``/``None``/``None`` by default (a local model
      never enters a remote provider config). Explicit ``oa_*`` fields
      are still accepted and validated (a local entry COULD be exposed
      through a remote wire — declared, never guessed).
    - any other (remote, non-Anthropic) ``provider``: ``oa_type`` is
      REQUIRED — a missing wire type is :class:`UnbudgetableError`
      (registration must not silently guess a wire type);
      ``oa_provider_key`` defaults to the ``provider`` value when
      absent; ``oa_base_url`` may be absent (``None``) — some wire
      types carry a built-in base URL.
    - present fields use the same non-empty-string discipline as the
      other str fields (:func:`_require_str`).
    """
    oa_type: str | None = None
    if "oa_type" in raw:
        oa_type = _require_str(name, raw, "oa_type")
    oa_provider_key: str | None = None
    if "oa_provider_key" in raw:
        oa_provider_key = _require_str(name, raw, "oa_provider_key")
    oa_base_url: str | None = None
    if "oa_base_url" in raw:
        oa_base_url = _require_str(name, raw, "oa_base_url")

    if provider == "anthropic":
        return (
            oa_provider_key if oa_provider_key is not None else "anthropic",
            oa_type if oa_type is not None else "anthropic",
            oa_base_url,
        )
    if provider == "local":
        # local: no remote provider routed; absent == None (validated
        # explicit fields, if any, are honored).
        return (
            oa_provider_key,
            oa_type,
            oa_base_url,
        )
    if oa_type is None:
        raise UnbudgetableError(
            f"model {name!r}: provider {provider!r} requires 'oa_type' — the OA "
            "wire type must be declared explicitly (silent wire-type guessing "
            "is unbudgetable)"
        )
    return (
        oa_provider_key if oa_provider_key is not None else provider,
        oa_type,
        oa_base_url,
    )


def load_registry(path: str | os.PathLike[str]) -> Registry:
    """Parse and validate a `models.toml` registry file.

    Raises :class:`UnbudgetableError` on any violation: missing required
    fields, forbidden $0 metered pricing, unapproved local pricing, non-zero
    subscription marginal cost, or an unknown pricing class. Never defaults
    a missing price to $0.
    """
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)

    entries: dict[str, ModelEntry] = {}
    for name, entry_raw in raw.items():
        entries[name] = _build_entry(name, entry_raw)

    version_hash = _canonical_hash(raw)
    return Registry(entries=entries, version_hash=version_hash)
