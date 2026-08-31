"""Charter loader/validator (SPEC.md §5, §10 AC1).

The charter is the whole run configured as one TOML file. The loader
pipeline is strictly:

    parse TOML file -> apply in-code defaults -> apply `SG_*` env overrides
    -> validate the MERGED result.

Validation runs LAST, over the merged config, and fails closed: any
violation raises :class:`CharterError`. The one exception is the
same-family critic-vs-rung check (D10/D11), which is a warn-only
preference — it is logged and recorded on :attr:`Charter.warnings`, never
raised.

**Env override convention.** Env vars are matched by the `SG_` prefix,
followed by the dotted config path with a double underscore (``__``) as the
level separator — chosen so that single-underscore key names (e.g.
``poll_seconds``) stay unambiguous. Path components are lower-cased to
match TOML key casing. Examples::

    SG_LOOP__BUDGETS__USD=10.0            -> loop.budgets.usd = 10.0
    SG_LOOP__TIMERS__POLL_SECONDS=30      -> loop.timers.poll_seconds = 30

Override values are parsed to match the *existing* (post file+defaults)
value's type at that path (bool/int/float/str, checked in that order so
that a bool target is never mistaken for an int). An override that targets
a path absent from both defaults and the charter file is carried through
as a raw string and is then rejected by the normal unknown-key validation
— "rejected like any other bad key."

**Resolved-config hash.** :attr:`Charter.resolved_hash` is a sha256 over a
canonical (sorted-key, whitespace-free JSON) serialization of the fully
merged and validated config — stable across key reordering, sensitive to
any value change. :func:`classify_diff` compares two such resolved dicts
and reports whether the diff expands security or spend surface.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stigmergy.checks import DEFAULT_CHECK_RESOURCES, CheckResources
from stigmergy.registry import Registry, UnbudgetableError, load_registry

logger = logging.getLogger(__name__)


class CharterError(Exception):
    """Raised on any charter validation failure. Fail closed."""


# bead .107/review-followup: the daemon's global circuit-breaker threshold
# (SPEC §9, bead .22 build spec §3) — the SINGLE source of truth for both
# `daemon.py` (which imports this constant rather than defining its own)
# and this module's own `_validate_loop` check that `loop.retries.
# critic_infra` stays strictly below it. Defined here, not in daemon.py,
# so charter validation can enforce the invariant without daemon.py
# importing charter.py importing daemon.py (a cycle) — charter.py has no
# dependency on daemon.py either way.
CIRCUIT_BREAKER_THRESHOLD = 5


# --- in-code defaults (SPEC §5) -------------------------------------------
#
# Only `[loop.*]` policy knobs get in-code defaults — everything else
# (rig identity, lanes, stepup ladder, models, egress, ...) describes the
# actual run and has no sensible default; its absence is caught by the
# relevant validation rule (e.g. no lanes -> zero selector-less lanes).

DEFAULT_CHARTER: dict[str, Any] = {
    "loop": {
        "concurrency": {"workers": 1},
        "budgets": {"dispatches": 50, "usd": 25.0, "gate_calls": 30},
        "dispatch_limits": {
            "output_tokens": 200000,
            "driver_turns": 100,
            "filed_tickets": 5,
            "filed_ticket_bytes": 16384,
        },
        # bead .107: `critic_infra` (default 3) MUST stay strictly below
        # `CIRCUIT_BREAKER_THRESHOLD` (5) — a single poisoned ticket must
        # escalate itself to the human floor before a genuine multi-ticket
        # infra storm would trip the global halt. `_validate_loop` enforces
        # this invariant at charter-load time (fail closed), not just here
        # in the comment.
        "retries": {
            "attempts_per_rung": 3,
            "integration_failures": 2,
            "flake_reruns": 2,
            "critic_infra": 3,
        },
        "cadences": {"staging_quiescent_tickets": 3, "staging_max_wait_seconds": 7200},
        "timers": {
            "poll_seconds": 15,
            "dispatch_timeout_seconds": 3600,
            "lease_ttl_seconds": 4500,
        },
    },
}


# --- known-schema key sets (rule 1: unknown keys anywhere -> reject) -----

_KNOWN_TOP_LEVEL = {
    "rig",
    "tiers",
    "checks",
    "gates",
    "loop",
    "prompts",
    "lanes",
    "stepup",
    "roles",
    "models",
    "egress",
    "notify",
    "provision",
}

_KNOWN_RIG_KEYS = {"name", "repo", "image"}
_KNOWN_TIERS_KEYS = {"dispatch_base"}
# bead .91: charter-configurable checker resource bounds. `_RESOURCE_KEYS`
# names the CheckResources fields as they appear in TOML (both under
# `[checks.<name>]`, alongside `cmd`, and under `[loop.check_resources]`).
_RESOURCE_KEYS = {"timeout_seconds", "memory", "cpus", "scratch_size", "pids_limit"}
_KNOWN_CHECK_KEYS = {"cmd"} | _RESOURCE_KEYS
_KNOWN_GATES_KEYS = {"attempt", "staging"}
_KNOWN_LOOP_KEYS = {
    "concurrency",
    "budgets",
    "dispatch_limits",
    "retries",
    "cadences",
    "timers",
    "check_resources",
}
_KNOWN_CONCURRENCY_KEYS = {"workers"}
_KNOWN_BUDGETS_KEYS = {"dispatches", "usd", "gate_calls"}
_KNOWN_DISPATCH_LIMITS_KEYS = {
    "output_tokens",
    "driver_turns",
    "filed_tickets",
    "filed_ticket_bytes",
}
_KNOWN_RETRIES_KEYS = {
    "attempts_per_rung",
    "integration_failures",
    "flake_reruns",
    "critic_infra",
}
_KNOWN_CADENCES_KEYS = {"staging_quiescent_tickets", "staging_max_wait_seconds"}
# loop.timers is intentionally open-ended: any key is legal as long as it is
# `_seconds`-suffixed (rule 6) — there is no fixed key set for this section.
_KNOWN_PROMPTS_KEYS = {"dir"}
_KNOWN_LANE_KEYS = {"selector", "driver", "model", "prompt", "egress", "entry", "effort"}
# bead .149: the closed driver vocabulary. `claude-code` is the v0 default;
# `openalph-exec` is the .149 in-cage `openalph exec` worker driver (spec
# §4.2: a concrete union, no Protocol — an unknown driver value is a
# charter-load error, never a runtime mystery).
_KNOWN_LANE_DRIVERS = frozenset({"claude-code", "openalph-exec"})
# bead .149 (spec §2 decision 8): the card-native effort vocabulary
# (kdsn.301). `high`/`max` are charter-load ERRORS (collapsed onto
# `medium`/`xhigh` at the provider layer — a warn-spam rung no human will
# ever notice); only the four card values are legal. `effort` is an
# openalph-exec concept ONLY — on a claude-code lane it is meaningless and
# rejected at charter-load.
_KNOWN_LANE_EFFORTS = frozenset({"none", "low", "medium", "xhigh"})
_KNOWN_STEPUP_KEYS = {"ladder"}
_KNOWN_ROLES_KEYS = {"critic"}
_KNOWN_CRITIC_KEYS = {"model", "max_tokens"}
_KNOWN_MODELS_KEYS = {"registry"}
_KNOWN_EGRESS_GROUP_KEYS = {"hosts"}
_KNOWN_NOTIFY_KEYS = {"ntfy_topic"}
# bead .79: the per-rig worker-image provision table. `pip` is a list of
# package specs (may carry version pins) the provision station installs into
# the per-rig worker image on top of the charter's pinned base (SPEC §3
# provision). bead .149 adds `oa_wheelhouse` (bool): when true, the provision
# station also bakes openalph (from a host-built wheelhouse, installed fully
# offline) + the stigmergy-worker agent TOML, enabling the `openalph exec`
# worker driver.
_KNOWN_PROVISION_KEYS = {"pip", "oa_wheelhouse"}

_ENV_PREFIX = "SG_"


@dataclass(frozen=True)
class Charter:
    """A fully merged, validated charter."""

    raw: dict[str, Any]
    resolved_hash: str
    warnings: tuple[str, ...]


def load_charter(
    path: str | os.PathLike[str],
    *,
    env: Mapping[str, str] | None = None,
    defaults: dict[str, Any] | None = None,
) -> Charter:
    """Load, merge, override, and validate a charter TOML file.

    Pipeline: parse TOML -> apply in-code defaults -> apply `SG_*` env
    overrides -> validate the merged result. Raises :class:`CharterError`
    on any violation (fail closed).

    ``env`` defaults to :data:`os.environ`; pass an explicit mapping (e.g.
    ``{}``) in tests to avoid reading the real process environment.
    ``defaults`` overrides the in-code defaults (:data:`DEFAULT_CHARTER`),
    for tests that need different baseline knobs.
    """
    charter_path = Path(path)

    try:
        with open(charter_path, "rb") as fh:
            file_cfg = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise CharterError(f"malformed TOML in {charter_path}: {exc}") from exc
    except OSError as exc:
        raise CharterError(f"cannot read charter file {charter_path}: {exc}") from exc

    base_defaults = DEFAULT_CHARTER if defaults is None else defaults
    merged = _deep_merge(base_defaults, file_cfg)

    env_map = os.environ if env is None else env
    merged = _apply_env_overrides(merged, env_map)

    charter_dir = charter_path.resolve().parent
    warnings = _validate(merged, charter_dir)

    resolved_hash = _canonical_hash(merged)

    return Charter(raw=merged, resolved_hash=resolved_hash, warnings=tuple(warnings))


def classify_diff(prev_resolved: dict[str, Any], curr_resolved: dict[str, Any]) -> str:
    """Classify a resolved-config diff as ``"expanding"`` or ``"safe"``.

    ``"expanding"`` means the diff expands security or spend surface:
    any of ``loop.budgets.{usd,dispatches,gate_calls}`` increased, any
    ``egress.*.hosts`` list gained an entry, or ``loop.concurrency.workers``
    increased. Anything else (equal, or a strict decrease/removal) is
    ``"safe"``. This is the classification mechanism behind "security- or
    spend-expanding diffs require interactive confirmation" — the actual
    confirmation prompt is wired elsewhere.

    Note: missing values fall back to ``0`` (via ``or 0``), which assumes
    inputs are already-validated resolved configs where budgets/workers are
    always positive — a legitimate ``0`` is impossible post-validation. This
    function is not itself a substitute for charter validation.
    """
    prev_usd = _get_path(prev_resolved, "loop", "budgets", "usd") or 0.0
    curr_usd = _get_path(curr_resolved, "loop", "budgets", "usd") or 0.0
    if curr_usd > prev_usd:
        return "expanding"

    prev_dispatches = _get_path(prev_resolved, "loop", "budgets", "dispatches") or 0
    curr_dispatches = _get_path(curr_resolved, "loop", "budgets", "dispatches") or 0
    if curr_dispatches > prev_dispatches:
        return "expanding"

    prev_gate_calls = _get_path(prev_resolved, "loop", "budgets", "gate_calls") or 0
    curr_gate_calls = _get_path(curr_resolved, "loop", "budgets", "gate_calls") or 0
    if curr_gate_calls > prev_gate_calls:
        return "expanding"

    prev_workers = _get_path(prev_resolved, "loop", "concurrency", "workers") or 0
    curr_workers = _get_path(curr_resolved, "loop", "concurrency", "workers") or 0
    if curr_workers > prev_workers:
        return "expanding"

    prev_egress = _get_path(prev_resolved, "egress") or {}
    curr_egress = _get_path(curr_resolved, "egress") or {}
    if isinstance(curr_egress, dict):
        for group_name, group_cfg in curr_egress.items():
            curr_hosts = set(group_cfg.get("hosts", [])) if isinstance(group_cfg, dict) else set()
            prev_group_cfg = (
                prev_egress.get(group_name, {}) if isinstance(prev_egress, dict) else {}
            )
            prev_hosts = (
                set(prev_group_cfg.get("hosts", []))
                if isinstance(prev_group_cfg, dict)
                else set()
            )
            if curr_hosts - prev_hosts:
                return "expanding"

    return "safe"


def resolve_check_resources(charter: Charter, name: str) -> CheckResources:
    """Resolve the effective :class:`~stigmergy.checks.CheckResources` for
    check ``name`` (bead .91: charter-configurable checker resource
    bounds).

    Resolution order (lowest -> highest precedence):
      1. :data:`~stigmergy.checks.DEFAULT_CHECK_RESOURCES` (the legacy
         hardcoded 60s/256m/1cpu/64m/64pids bounds).
      2. ``[loop.check_resources]`` — applies to EVERY check, including a
         synthesized name absent from ``[checks.*]`` (e.g. the daemon's
         `ticket-tests`).
      3. ``[checks.<name>]`` resource keys — per-check override, only for
         a named charter check; the `cmd` key is ignored here.

    Only keys actually PRESENT at each layer are overlaid; an unset key
    falls through to the next lower-precedence layer. Final values are
    normalized to the :class:`CheckResources` field types (`cpus`/`memory`/
    `scratch_size` as `str`, `timeout_seconds`/`pids_limit` as `int`).
    """
    merged: dict[str, Any] = {
        "timeout_seconds": DEFAULT_CHECK_RESOURCES.timeout_seconds,
        "memory": DEFAULT_CHECK_RESOURCES.memory,
        "cpus": DEFAULT_CHECK_RESOURCES.cpus,
        "scratch_size": DEFAULT_CHECK_RESOURCES.scratch_size,
        "pids_limit": DEFAULT_CHECK_RESOURCES.pids_limit,
    }

    loop_overlay = charter.raw.get("loop", {}).get("check_resources", {})
    for key in _RESOURCE_KEYS:
        if key in loop_overlay:
            merged[key] = loop_overlay[key]

    check_overlay = charter.raw.get("checks", {}).get(name, {})
    for key in _RESOURCE_KEYS:
        if key in check_overlay:
            merged[key] = check_overlay[key]

    merged["timeout_seconds"] = int(merged["timeout_seconds"])
    merged["pids_limit"] = int(merged["pids_limit"])
    merged["cpus"] = str(merged["cpus"])
    merged["memory"] = str(merged["memory"])
    merged["scratch_size"] = str(merged["scratch_size"])

    return CheckResources(**merged)


# --- merge / env-override machinery ---------------------------------------


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` over ``base``; ``overlay`` wins conflicts."""
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _apply_env_overrides(merged: dict[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    result = copy.deepcopy(merged)
    for raw_key, raw_value in env.items():
        if not raw_key.startswith(_ENV_PREFIX):
            continue
        path_str = raw_key[len(_ENV_PREFIX) :]
        parts = [part.lower() for part in path_str.split("__") if part]
        if not parts:
            continue
        _set_override(result, parts, raw_value)
    return result


def _set_override(cfg: dict[str, Any], parts: list[str], raw_value: str) -> None:
    cur = cfg
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    leaf = parts[-1]
    existing = cur.get(leaf)
    cur[leaf] = _coerce_env_value(raw_value, existing)


def _coerce_env_value(raw_value: str, existing: Any) -> Any:
    """Parse ``raw_value`` to match ``existing``'s type.

    Checked in bool -> int -> float -> str order (bool is a subclass of
    int in Python, so it must be checked first). An ``existing`` of
    ``None`` means the override path is unknown; the raw string is passed
    through so ordinary unknown-key validation rejects it.
    """
    if isinstance(existing, bool):
        low = raw_value.strip().lower()
        if low in ("true", "1", "yes", "on"):
            return True
        if low in ("false", "0", "no", "off"):
            return False
        raise CharterError(f"cannot parse {raw_value!r} as bool for env override")
    if isinstance(existing, int):
        try:
            return int(raw_value)
        except ValueError as exc:
            raise CharterError(f"cannot parse {raw_value!r} as int for env override") from exc
    if isinstance(existing, float):
        try:
            return float(raw_value)
        except ValueError as exc:
            raise CharterError(f"cannot parse {raw_value!r} as float for env override") from exc
    return raw_value


def _get_path(d: Mapping[str, Any] | None, *path: str) -> Any:
    cur: Any = d
    for part in path:
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _canonical_hash(data: dict[str, Any]) -> str:
    """sha256 over a canonical (sorted-key, whitespace-free JSON) serialization."""
    blob = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# --- validation ------------------------------------------------------------


def _validate(merged: dict[str, Any], charter_dir: Path) -> list[str]:
    _validate_top_level_keys(merged)
    _validate_optional_keys(merged.get("rig"), _KNOWN_RIG_KEYS, "rig")
    _validate_optional_keys(merged.get("tiers"), _KNOWN_TIERS_KEYS, "tiers")
    _validate_checks(merged.get("checks"))
    _validate_optional_keys(merged.get("gates"), _KNOWN_GATES_KEYS, "gates")
    _validate_loop(merged["loop"])
    _validate_optional_keys(merged.get("prompts"), _KNOWN_PROMPTS_KEYS, "prompts")

    lanes = merged.get("lanes")
    if lanes is None:
        lanes = {}
    _validate_lanes_keys(lanes)
    _validate_lanes_values(lanes)

    stepup = merged.get("stepup")
    if stepup is None:
        stepup = {}
    _validate_keys(stepup, _KNOWN_STEPUP_KEYS, "stepup")

    roles = merged.get("roles")
    if roles is None:
        roles = {}
    _validate_roles_keys(roles)

    _validate_optional_keys(merged.get("models"), _KNOWN_MODELS_KEYS, "models")
    _validate_egress_keys(merged.get("egress"))
    _validate_optional_keys(merged.get("notify"), _KNOWN_NOTIFY_KEYS, "notify")
    _validate_provision(merged.get("provision"))

    registry = _load_registry_for_charter(merged, charter_dir)

    _validate_model_resolvable(lanes, roles, registry)
    _validate_lane_topology(lanes, stepup)
    warnings = _check_same_family_warning(lanes, stepup, roles, registry)

    return warnings


def _validate_keys(cfg: Any, known: set[str], ctx: str) -> None:
    if not isinstance(cfg, dict):
        raise CharterError(f"[{ctx}] must be a table")
    unknown = set(cfg) - known
    if unknown:
        raise CharterError(f"unknown key(s) in [{ctx}]: {sorted(unknown)}")


def _validate_optional_keys(cfg: Any, known: set[str], ctx: str) -> None:
    if cfg is None:
        return
    _validate_keys(cfg, known, ctx)


def _validate_top_level_keys(merged: dict[str, Any]) -> None:
    unknown = set(merged) - _KNOWN_TOP_LEVEL
    if unknown:
        raise CharterError(f"unknown top-level key(s): {sorted(unknown)}")


def _validate_checks(checks_cfg: Any) -> None:
    if checks_cfg is None:
        return
    if not isinstance(checks_cfg, dict):
        raise CharterError("[checks] must be a table")
    for name, entry in checks_cfg.items():
        _validate_keys(entry, _KNOWN_CHECK_KEYS, f"checks.{name}")
        _validate_check_resources(entry, f"checks.{name}")


def _validate_provision(provision_cfg: Any) -> None:
    """Validate the optional `[provision]` table (bead .79). Absent entirely
    is fine (a rig with no [provision] falls back to the base worker image
    unchanged). If present, it must be a table, unknown keys are rejected,
    and `pip` (if present) must be a list of non-empty strings -- package
    specs, which may carry version pins (e.g. ``"ruff==0.15.22"``)."""
    if provision_cfg is None:
        return
    _validate_keys(provision_cfg, _KNOWN_PROVISION_KEYS, "provision")
    if "pip" in provision_cfg:
        pip_cfg = provision_cfg["pip"]
        if not isinstance(pip_cfg, list):
            raise CharterError(f"provision.pip must be a list (got {pip_cfg!r})")
        for spec in pip_cfg:
            if not isinstance(spec, str) or not spec:
                raise CharterError(
                    f"provision.pip entries must be non-empty strings (got {spec!r})"
                )
    if "oa_wheelhouse" in provision_cfg:
        # bead .149: a strict boolean (a truthy-but-non-bool value like "true"
        # would be a silent typo — fail closed).
        if not isinstance(provision_cfg["oa_wheelhouse"], bool):
            raise CharterError(
                "provision.oa_wheelhouse must be a boolean "
                f"(got {provision_cfg['oa_wheelhouse']!r})"
            )


_MEMORY_SIZE_RE = re.compile(r"^[0-9]+[bkmgBKMG]?$")


def _validate_check_resources(cfg: dict[str, Any], ctx: str) -> None:
    """Validate the VALUE of each resource key that is PRESENT in ``cfg``
    (bead .91). Does NOT enforce the key set itself — the caller validates
    the allowed keys (via :func:`_validate_keys`) before or after calling
    this; this only rejects bad values for whichever resource keys showed
    up. Non-resource keys (e.g. `cmd`) are ignored."""
    if "timeout_seconds" in cfg:
        _validate_positive_int(cfg, "timeout_seconds", f"{ctx}.timeout_seconds")
    if "pids_limit" in cfg:
        _validate_positive_int(cfg, "pids_limit", f"{ctx}.pids_limit")
    if "cpus" in cfg:
        _validate_positive_number_or_numeric_string(cfg["cpus"], f"{ctx}.cpus")
    if "memory" in cfg:
        _validate_size_string(cfg["memory"], f"{ctx}.memory")
    if "scratch_size" in cfg:
        _validate_size_string(cfg["scratch_size"], f"{ctx}.scratch_size")


def _validate_positive_number_or_numeric_string(value: Any, ctx: str) -> None:
    if isinstance(value, bool):
        raise CharterError(f"{ctx} must be a positive number (got {value!r})")
    if isinstance(value, int | float):
        if value <= 0:
            raise CharterError(f"{ctx} must be a positive number (got {value!r})")
        return
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError as exc:
            raise CharterError(f"{ctx} must be a positive number (got {value!r})") from exc
        if parsed <= 0:
            raise CharterError(f"{ctx} must be a positive number (got {value!r})")
        return
    raise CharterError(f"{ctx} must be a positive number (got {value!r})")


def _validate_size_string(value: Any, ctx: str) -> None:
    if isinstance(value, bool) or not isinstance(value, str) or not value:
        raise CharterError(f"{ctx} must be a non-empty size string (got {value!r})")
    if not _MEMORY_SIZE_RE.match(value):
        raise CharterError(
            f"{ctx} must match ^[0-9]+[bkmgBKMG]?$ (a positive integer optionally "
            f"followed by one unit letter) (got {value!r})"
        )


def _validate_lanes_keys(lanes_cfg: Any) -> None:
    if not isinstance(lanes_cfg, dict):
        raise CharterError("[lanes] must be a table")
    for name, entry in lanes_cfg.items():
        _validate_keys(entry, _KNOWN_LANE_KEYS, f"lanes.{name}")


def _validate_lanes_values(lanes_cfg: dict[str, Any]) -> None:
    """bead .149: per-lane VALUE checks (the key-set check is
    :func:`_validate_lanes_keys`; this one closes the value vocabulary):

    - ``driver`` (when present) must be in the closed set
      :data:`_KNOWN_LANE_DRIVERS` — an unknown driver is a charter-load
      error (fail closed; there is no runtime fallback).
    - ``effort`` (when present) must be a card-native value from
      :data:`_KNOWN_LANE_EFFORTS` — ``high``/``max`` are rejected naming
      the kdsn.301 collapse (spec §2 decision 8).
    - ``effort`` on a ``claude-code`` lane is rejected (meaningless there
      — the claude-code driver has no effort concept).
    """
    for name, entry in lanes_cfg.items():
        if not isinstance(entry, dict):
            raise CharterError(f"lanes.{name} must be a table")
        driver = entry.get("driver", "claude-code")
        if driver not in _KNOWN_LANE_DRIVERS:
            raise CharterError(
                f"lanes.{name}.driver {driver!r} is not a known driver "
                f"(known: {sorted(_KNOWN_LANE_DRIVERS)})"
            )
        if "effort" not in entry:
            continue
        effort = entry["effort"]
        if not isinstance(effort, str) or effort not in _KNOWN_LANE_EFFORTS:
            raise CharterError(
                f"lanes.{name}.effort {effort!r} is not a card-native effort "
                f"(known: {sorted(_KNOWN_LANE_EFFORTS)}) — `high`/`max` collapse "
                "at the provider layer (kdsn.301) and are charter-load errors, "
                "not rungs"
            )
        if driver == "claude-code":
            raise CharterError(
                f"lanes.{name}.effort is meaningless on a claude-code lane "
                f"(effort is an openalph-exec driver concept)"
            )


def _validate_roles_keys(roles_cfg: Any) -> None:
    if not isinstance(roles_cfg, dict):
        raise CharterError("[roles] must be a table")
    _validate_keys(roles_cfg, _KNOWN_ROLES_KEYS, "roles")
    critic_cfg = roles_cfg.get("critic")
    if critic_cfg is not None:
        _validate_keys(critic_cfg, _KNOWN_CRITIC_KEYS, "roles.critic")
        # bead .118: optional per-rig verdict-critic output budget. Must be a
        # positive int when present; absent -> critic_client.DEFAULT_MAX_TOKENS.
        if "max_tokens" in critic_cfg:
            _validate_positive_int(critic_cfg, "max_tokens", "roles.critic.max_tokens")


def _validate_egress_keys(egress_cfg: Any) -> None:
    if egress_cfg is None:
        return
    if not isinstance(egress_cfg, dict):
        raise CharterError("[egress] must be a table")
    for name, entry in egress_cfg.items():
        _validate_keys(entry, _KNOWN_EGRESS_GROUP_KEYS, f"egress.{name}")


def _validate_loop(loop_cfg: Any) -> None:
    if not isinstance(loop_cfg, dict):
        raise CharterError("[loop] must be a table")
    unknown = set(loop_cfg) - _KNOWN_LOOP_KEYS
    if unknown:
        raise CharterError(f"unknown key(s) in [loop]: {sorted(unknown)}")

    concurrency = loop_cfg.get("concurrency", {})
    _validate_keys(concurrency, _KNOWN_CONCURRENCY_KEYS, "loop.concurrency")
    _validate_workers(concurrency)

    budgets = loop_cfg.get("budgets", {})
    _validate_keys(budgets, _KNOWN_BUDGETS_KEYS, "loop.budgets")
    _validate_positive_int(budgets, "dispatches", "loop.budgets.dispatches")
    _validate_positive_int(budgets, "gate_calls", "loop.budgets.gate_calls")
    _validate_positive_number(budgets, "usd", "loop.budgets.usd")

    dispatch_limits = loop_cfg.get("dispatch_limits", {})
    _validate_keys(dispatch_limits, _KNOWN_DISPATCH_LIMITS_KEYS, "loop.dispatch_limits")
    _validate_positive_int(dispatch_limits, "output_tokens", "loop.dispatch_limits.output_tokens")
    _validate_positive_int(dispatch_limits, "driver_turns", "loop.dispatch_limits.driver_turns")
    _validate_positive_int(dispatch_limits, "filed_tickets", "loop.dispatch_limits.filed_tickets")
    _validate_positive_int(
        dispatch_limits, "filed_ticket_bytes", "loop.dispatch_limits.filed_ticket_bytes"
    )

    retries = loop_cfg.get("retries", {})
    _validate_keys(retries, _KNOWN_RETRIES_KEYS, "loop.retries")
    _validate_positive_int(retries, "attempts_per_rung", "loop.retries.attempts_per_rung")
    _validate_positive_int(retries, "integration_failures", "loop.retries.integration_failures")
    _validate_positive_int(retries, "flake_reruns", "loop.retries.flake_reruns")
    _validate_positive_int(retries, "critic_infra", "loop.retries.critic_infra")
    if retries["critic_infra"] >= CIRCUIT_BREAKER_THRESHOLD:
        raise CharterError(
            f"loop.retries.critic_infra ({retries['critic_infra']}) must be strictly less "
            f"than the circuit-breaker threshold ({CIRCUIT_BREAKER_THRESHOLD}) — otherwise "
            "the global halt can fire before a single poisoned ticket escalates itself, "
            "reconstituting the single-ticket livelock-then-global-halt incident bead .107 "
            "fixed"
        )

    cadences = loop_cfg.get("cadences", {})
    _validate_keys(cadences, _KNOWN_CADENCES_KEYS, "loop.cadences")

    timers = loop_cfg.get("timers", {})
    _validate_timers(timers)

    check_resources = loop_cfg.get("check_resources", {})
    _validate_keys(check_resources, _RESOURCE_KEYS, "loop.check_resources")
    _validate_check_resources(check_resources, "loop.check_resources")


def _validate_workers(concurrency_cfg: dict[str, Any]) -> None:
    workers = concurrency_cfg.get("workers", 1)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise CharterError(f"loop.concurrency.workers must be a positive integer (got {workers!r})")


def _validate_timers(timers_cfg: Any) -> None:
    if not isinstance(timers_cfg, dict):
        raise CharterError("[loop.timers] must be a table")
    for key, value in timers_cfg.items():
        if not key.endswith("_seconds"):
            raise CharterError(f"loop.timers key {key!r} must be `_seconds`-suffixed")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise CharterError(f"loop.timers.{key} must be a positive int (got {value!r})")


def _validate_positive_int(cfg: dict[str, Any], key: str, ctx: str) -> None:
    value = cfg.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CharterError(f"{ctx} must be a positive int (got {value!r})")


def _validate_positive_number(cfg: dict[str, Any], key: str, ctx: str) -> None:
    value = cfg.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise CharterError(f"{ctx} must be a positive number (got {value!r})")


def _load_registry_for_charter(merged: dict[str, Any], charter_dir: Path) -> Registry:
    models_cfg = merged.get("models")
    if not isinstance(models_cfg, dict) or "registry" not in models_cfg:
        raise CharterError("[models].registry is required to resolve lane/critic models")
    registry_rel = models_cfg["registry"]
    if not isinstance(registry_rel, str) or not registry_rel:
        raise CharterError("[models].registry must be a non-empty string path")
    registry_path = Path(registry_rel)
    if not registry_path.is_absolute():
        registry_path = charter_dir / registry_path
    try:
        return load_registry(registry_path)
    except (UnbudgetableError, OSError) as exc:
        raise CharterError(f"failed to load model registry at {registry_path}: {exc}") from exc


def _validate_model_resolvable(
    lanes: dict[str, Any], roles: dict[str, Any], registry: Registry
) -> None:
    for lane_name, lane_cfg in lanes.items():
        if not isinstance(lane_cfg, dict):
            raise CharterError(f"lanes.{lane_name} must be a table")
        model_name = lane_cfg.get("model")
        if not isinstance(model_name, str) or not model_name:
            raise CharterError(f"lanes.{lane_name}.model is required")
        try:
            registry.resolve(model_name)
        except UnbudgetableError as exc:
            raise CharterError(
                f"lanes.{lane_name} names model {model_name!r} which is not in the registry"
            ) from exc

    critic_cfg = roles.get("critic") if isinstance(roles, dict) else None
    if not isinstance(critic_cfg, dict):
        raise CharterError("[roles.critic] is required")
    critic_model = critic_cfg.get("model")
    if not isinstance(critic_model, str) or not critic_model:
        raise CharterError("roles.critic.model is required")
    try:
        registry.resolve(critic_model)
    except UnbudgetableError as exc:
        raise CharterError(
            f"roles.critic names model {critic_model!r} which is not in the registry"
        ) from exc


def _validate_lane_topology(lanes: dict[str, Any], stepup: dict[str, Any]) -> None:
    ladder = stepup.get("ladder", [])
    if not isinstance(ladder, list):
        raise CharterError("stepup.ladder must be a list")
    for rung in ladder:
        if not isinstance(rung, str):
            raise CharterError(f"stepup.ladder entries must be strings (got {rung!r})")

    entry_selectorless: list[str] = []
    for lane_name, lane_cfg in lanes.items():
        entry = lane_cfg.get("entry", True)
        if not isinstance(entry, bool):
            raise CharterError(f"lanes.{lane_name}.entry must be a boolean")
        has_selector = lane_cfg.get("selector") is not None

        if entry is False:
            if lane_name not in ladder:
                raise CharterError(
                    f"lanes.{lane_name} has entry=false but is absent from stepup.ladder"
                )
            if has_selector:
                raise CharterError(
                    f"lanes.{lane_name} has entry=false but carries a selector "
                    "(entry=false lanes must not carry a selector)"
                )
        else:
            if not has_selector:
                entry_selectorless.append(lane_name)

    if len(entry_selectorless) == 0:
        raise CharterError(
            "no selector-less fallthrough lane defined; exactly one entry lane "
            "must omit `selector`"
        )
    if len(entry_selectorless) > 1:
        raise CharterError(
            f"multiple selector-less fallthrough lanes defined: {sorted(entry_selectorless)}; "
            "exactly one is allowed"
        )

    for rung in ladder:
        if rung not in lanes:
            raise CharterError(f"stepup.ladder references undefined lane {rung!r}")


def _check_same_family_warning(
    lanes: dict[str, Any],
    stepup: dict[str, Any],
    roles: dict[str, Any],
    registry: Registry,
) -> list[str]:
    """D10/D11: warn (never raise) if the critic shares a model family with
    any ladder rung's model — including the exact-same-model case."""
    warnings: list[str] = []
    critic_cfg = roles.get("critic", {}) if isinstance(roles, dict) else {}
    critic_model_name = critic_cfg.get("model") if isinstance(critic_cfg, dict) else None
    if not critic_model_name:
        return warnings
    critic_entry = registry.resolve(critic_model_name)

    ladder = stepup.get("ladder", [])
    for rung in ladder:
        lane_cfg = lanes.get(rung)
        if not isinstance(lane_cfg, dict):
            continue
        rung_model_name = lane_cfg.get("model")
        if not rung_model_name:
            continue
        rung_entry = registry.resolve(rung_model_name)
        if rung_entry.family == critic_entry.family:
            msg = (
                f"critic model {critic_model_name!r} (family={critic_entry.family!r}) "
                f"shares a family with ladder rung {rung!r} model {rung_model_name!r} "
                "(family/model overlap is a warn-only preference, not a hard rule)"
            )
            logger.warning(msg)
            warnings.append(msg)
    return warnings
