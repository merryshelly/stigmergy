"""OA provider-layer critic clients — bead workspace-e2uh.143 (A′).

SB's adjudicated ruling (openalph-exec-design.md §4, A′, 2026-08-30): the
critic/range-critic become **in-process forced-tool calls through OA's
provider layer** with the shared key-bearing-call hardening flag ON. This
module is that seam — the ONLY `src/stigmergy/` module that imports
`openalph.*`, and it does so LAZILY, inside the client factories,
fail-closed: an unavailable OA layer raises
:class:`CriticOAUnavailableError` at RIG LAUNCH (factory build), never as
a `.107` critic-infra trip on the first gate.

**One persistent event loop per process.** OA's provider client cache
(`openalph.provider._client_cache`) retains SDK clients whose httpx pools
bind to the loop that created them; spawning/destroying a loop per gate
call (e.g. `asyncio.run` per call) would orphan the pooled hardened client
on a dead loop and the second gate call would fail. The bridge below runs
ONE `asyncio.new_event_loop()` on ONE daemon thread for the process
lifetime (matching the daemon's lifecycle) and submits each gate call via
`asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=...)` with a
600 s ceiling — deliberately ABOVE the 120 s transport timeout so OA's
own timeouts (incl. its SDK retry budget, worst case ≈ 2.25 min) fire
first.

**Guard equivalence.** The legacy bare-urllib client's three key-exfil
guards all map onto kdsn.304's `hardened=True` transport (no-redirect
follow, no proxy-env inheritance, bounded response read) — this adapter
passes `hardened=True` UNCONDITIONALLY; there is no unhardened code path.
Residual Stigmergy-side responsibilities owned here (spec §3.2): the
120 s transport timeout (`_OA_TIMEOUT_SECONDS`, replacing the bare
client's 60 s default), the `.118` stop_reason mapping
(`{"length","max_tokens"}` -> `roles.critic.max_tokens`; `refusal` keeps
its own specific error), and the exactly-one-tool-call arity/name/
dict-input gate.

**Redaction invariants (spec §3.3).** The provider key, sourced via the
injected `key_provider` (the repo's `make_op_key_provider` mechanism —
fetch-once, never-in-exception-text), is:

- fetched LAZILY at first call (an idle daemon that never gates makes no
  `op` round-trip) and cached per factory (a mid-run fetch failure is not
  cached — the next call retries, as the mechanism guarantees);
- placed ONLY in `ProviderConfig.api_key`; the `ProviderConfig` is
  assembled per call (never shared/cached across models, so a key can
  never be observed for the wrong model);
- ABSENT from every returned dict, every raised error message, and every
  `gate_fields` provenance value. OA's own `_sanitize_error` additionally
  scrubs keys from provider errors before they can reach us.

**Prompt placement (spec §3.6a).** `system=""` + a single user message
carrying the prompt VERBATIM — the bytes `critic.py` hashed into
`prompt_artifact_hash` are exactly the bytes sent (OA's agent-preamble
path is off the bare-`complete` surface).

**decoding_params must be `{}` (spec §3.6b).** The legacy client spread
them into the request body; `complete()` has no kwargs spread. A
non-empty dict raises `CriticClientError` — fail LOUD, never
silently-dropped (the provenance field still records them verbatim,
shape unchanged, in `gate_fields`).

**Usage (spec §3.6c/UNVERIFIED-5).** The response's `Usage` maps onto
the canonical 4-key shape used across the repo
(`in = input_tokens`, `cached = cache_read_tokens`, `out =
output_tokens`, `reasoning = 0` — the reasoning channel is pinned at 0
until OA verifiably surfaces Kimi's `reasoning_content` count). NOTE:
OA's `Response.__post_init__` fabricates `Usage(0, 0)` when a provider
omits usage, so the adapter cannot distinguish "absent" from "zero" and
maps the zeros as real data — MOOT for the migrated path: the `{}`
unbudgetable sentinel only bites METERED entries (`cli._emit_report_event`)
and the critic entries route to SUBSCRIPTION. The verdict client ADDS
the channel under the `usage` key (closing the §2.4 provenance gap:
`tokens` was absent from gate_fields because the bare client dropped
usage); the range client returns it in its `{text, filed_tickets,
usage}` shape 1:1 with the legacy `_extract_range_review` contract.

**Tool-call ids are OPAQUE.** Spike .142 observed Kimi's `name:0`
scheme (`submit_verdict:0`); this module never inspects or branches on
`ToolCall.id` (kdsn.304 AC-10/11 carried to the consumer).
"""

from __future__ import annotations

import asyncio
import tempfile
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stigmergy.registry import ModelEntry, Registry

# ===========================================================================
# Moved constants / errors / schema builders (from critic_client.py; the
# schema dicts are BYTE-IDENTICAL — the spike-verified wire contract).
# ===========================================================================

# bead .118: a verdict `reason` for a real artifact routinely exceeds 1024
# output tokens; at 1024 the tool_use JSON truncates (stop_reason=max_tokens)
# and drops trailing required fields -> spurious CriticInfraError. Generous
# default; charter-overridable via roles.critic.max_tokens.
DEFAULT_MAX_TOKENS = 4096
# Range prose + proposals are larger than a verdict's 1024-token budget.
DEFAULT_RANGE_MAX_TOKENS = 4096
VERDICT_TOOL_NAME = "submit_verdict"
RANGE_REVIEW_TOOL_NAME = "submit_range_review"
# bead .152 (Decision 17): the decomposer band's validation critic — the
# structured decomposer-manifest validation verdict (gate 2 of 2 of the
# decomposer station). Rig-level, no ticket (same precedent as `report` /
# `ticket-filed`); its record-plane event is `EventType.DECOMPOSE`.
DECOMPOSE_VALIDATION_TOOL_NAME = "submit_validation"

# §3.2.1: a forced-tool verdict is a few-KB response, but spike .142
# observed 1661 pre-tool reasoning tokens and up to 18.1 s latency on
# blackwell — 120 s absorbs pre-tool deliberation with 6x headroom while
# not wedging a gate for ten minutes (replaces the bare client's 60 s
# default; OA's own 600 s ProviderConfig default is NOT used).
_OA_TIMEOUT_SECONDS = 120.0

# Ceiling for the run_coroutine_threadsafe future (§4.5): above the
# transport timeout + OA's SDK retry budget (worst case ≈ 2.25 min) so
# OA's own timeouts fire first.
_BRIDGE_FUTURE_TIMEOUT_SECONDS = 600.0


class CriticOAUnavailableError(Exception):
    """The OA provider layer (`openalph.provider` / `openalph.tools`)
    cannot be imported. Raised at FACTORY BUILD (rig launch) by
    `make_oa_critic_client` / `make_oa_range_critic_client` — fail-closed
    (Decision 1): a rig without OA available refuses to start rather
    than degrading mid-run. Distinct from `CriticInfraError` on purpose:
    this is a startup failure, not a `.107` gate-infra trip, and it
    bypasses `Critic.judge`'s wrapper by construction."""


class CriticClientError(Exception):
    """Raised when the critic client itself cannot produce a raw verdict
    input dict (or the range-review shape): a provider response whose
    `stop_reason` signals truncation/refusal, or whose tool-call
    arity/name/dict-input is malformed. `Critic.judge` wraps this (like
    any client exception) into `CriticInfraError` — never a rejection
    verdict. Never carries the provider key in its message (spec §3.3)."""


def build_verdict_tool() -> dict[str, Any]:
    """The synthetic tool definition that forces the model to return
    exactly one structured verdict. Mirrors `Outcome`/`Severity`
    (`verdicts.py`) but is only a HINT to the model — `_parse_verdict`
    (`critic.py`) remains the authority we actually trust; this schema is
    never itself treated as validation.

    D14 (bead `.39`): the schema also carries an OPTIONAL `filed_tickets`
    array — identical in shape to `build_range_review_tool()`'s — for the
    critic to propose out-of-rubric follow-up tickets alongside its
    verdict. It is deliberately NOT in the top-level `required` list (a
    clean verdict files nothing) and, like the rest of this schema, is
    only a HINT: `critic.py` tolerantly extracts it and never treats it
    as validation, mirroring the strict-verdict/tolerant-filings split.
    """
    return {
        "name": VERDICT_TOOL_NAME,
        # bead .118: strict tool use compiles this schema into a grammar and
        # constrains sampling, so the model cannot omit a `required` field
        # (server-side enforcement, not a prompting hint). Requires
        # additionalProperties:false on every object below. GA on the current
        # opus/sonnet generation; empirically verified with tool_choice=
        # {type:tool,name:submit_verdict} + the optional filed_tickets field
        # (2026-07-20 spike). NOTE strict does NOT override max_tokens
        # truncation -- see DEFAULT_MAX_TOKENS.
        "strict": True,
        "description": (
            "Submit the final structured verdict for the artifact under review. "
            "Call this exactly once with your judgment."
        ),
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                # Field order is deliberate: `reason` precedes `outcome` so the
                # model emits its evidence-anchored justification BEFORE it
                # commits to a verdict token (reason-before-verdict; lowers
                # flip rate per the llm-critic-reliability research; SB .26
                # sign-off 2026-07-18). Parsing is key-based / order-independent.
                "tier": {
                    "type": "integer",
                    "description": "Which rubric tier this verdict is judging.",
                },
                "reason": {
                    "type": "string",
                    "description": "Non-empty human-readable justification.",
                },
                "outcome": {
                    "type": "string",
                    "enum": ["met", "unmet"],
                    "description": "Whether the artifact meets the rubric.",
                },
                "severity": {
                    "type": "string",
                    "enum": ["none", "low", "medium", "high"],
                    "description": "Severity of any defect found (recorded, not gating).",
                },
                "filed_tickets": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "evidence": {"type": "string"},
                        },
                        "required": ["title", "description"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["outcome", "tier", "reason", "severity"],
        },
    }


def build_range_review_tool() -> dict[str, Any]:
    """The synthetic tool definition that forces the model to return
    exactly one structured range review carrying BOTH advisory prose
    (`findings`) and proposed follow-up tickets (`filed_tickets`).
    `filed_tickets` is deliberately NOT in the top-level `required` list —
    a clean range that raises no cross-cutting/interlink concerns files
    nothing. This schema is only a HINT to the model; `RangeCritic.review`
    (`rangereport.py`) remains the sole authority we actually trust — this
    module never re-validates `findings`/`filed_tickets` semantics.
    """
    return {
        "name": RANGE_REVIEW_TOOL_NAME,
        "description": (
            "Submit the range review for the operator: advisory prose findings, "
            "plus any cross-cutting/interlink issues worth filing as follow-up "
            "tickets. Call this exactly once."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "findings": {
                    "type": "string",
                    "description": "Advisory prose review for the operator.",
                },
                "filed_tickets": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "evidence": {"type": "string"},
                        },
                        "required": ["title", "description"],
                    },
                },
            },
            "required": ["findings"],
        },
    }


def build_validation_tool() -> dict[str, Any]:
    """The synthetic tool definition that forces the decomposer-station
    validation critic to return exactly one structured validation verdict
    for the decomposer manifest under review (bead workspace-e2uh.152,
    Decision 17 — the decomposer band's gate 2 of 2, the LLM half of the
    two-gate decomposer station). Mirrors the house style of
    :func:`build_verdict_tool` / :func:`build_range_review_tool`:
    `strict: True` + `additionalProperties: False` on every object, so the
    forced-tool grammar constrains sampling server-side (a `required` field
    cannot be omitted).

    The schema is only a HINT to the model — the decomposer station's
    manifest validator / repair-round driver remains the sole authority we
    actually trust; this module never re-validates `verdict` / `findings`
    semantics (mirrors the split where `critic.py` owns verdict parsing and
    `rangereport.py` owns range-review semantics).

    `verdict` is `accept` (the manifest may enter the ticket pool — minor
    findings ride along recorded) or `repair` (the findings must be
    addressed by a repair round). Each `findings` item is advisory: an
    `aspect` (judgment dimension), a `severity`, the `tickets` it touches
    (empty array for manifest-level findings), the `evidence` it rests on,
    and an advisory `direction` for a repair round — `direction` is
    deliberately NOT in the item's `required` list (a finding with no
    repair direction is still well-formed).
    """
    return {
        "name": DECOMPOSE_VALIDATION_TOOL_NAME,
        "strict": True,
        "description": (
            "Submit the structured validation verdict for the decomposer "
            "manifest under review. Call this exactly once with your judgment."
        ),
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": ["accept", "repair"],
                    "description": (
                        "accept = the manifest may enter the ticket pool (minor "
                        "findings ride along recorded); repair = the findings "
                        "must be addressed by a repair round."
                    ),
                },
                "summary": {
                    "type": "string",
                    "description": "Non-empty human-readable overall judgment.",
                },
                "findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "aspect": {
                                "type": "string",
                                "enum": [
                                    "fidelity",
                                    "coverage",
                                    "sizing",
                                    "rubric_quality",
                                    "hedges",
                                    "notes",
                                    "other",
                                ],
                                "description": (
                                    "Which judgment dimension the finding belongs to."
                                ),
                            },
                            "severity": {
                                "type": "string",
                                "enum": ["critical", "major", "minor"],
                            },
                            "tickets": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Manifest ticket ids the finding touches; "
                                    "empty array for manifest-level findings."
                                ),
                            },
                            "evidence": {
                                "type": "string",
                                "description": (
                                    "The specific spec clause, manifest field, "
                                    "or quote the finding rests on."
                                ),
                            },
                            "direction": {
                                "type": "string",
                                "description": (
                                    "What a repair round should do about it "
                                    "(advisory)."
                                ),
                            },
                        },
                        "required": ["aspect", "severity", "tickets", "evidence"],
                    },
                },
            },
            "required": ["verdict", "summary", "findings"],
        },
    }


# ===========================================================================
# The OA import seam (lazy, fail-closed — Decision 1)
# ===========================================================================


def _import_oa_provider() -> tuple[Any, Any, Any, Any]:
    """Lazily import the OA provider layer, resolving the four symbols the
    adapter needs: the `openalph.provider` module (its `complete`), plus
    the `ToolDef` / `ProviderConfig` / `AgentConfig` dataclasses.

    Called once per factory build (NOT per call). Any `ImportError`
    (module missing, OR a symbol missing from a half-installed OA) is
    wrapped in :class:`CriticOAUnavailableError` naming `openalph` —
    loud rig-launch failure, never a silent mid-run degradation.

    THIS is the only place in the module that imports openalph.*; the
    factories call it at build time so the import-failure check happens
    even when `complete_fn` is injected (tests may monkeypatch THIS
    function to simulate an OA-less environment).
    """
    try:
        import openalph.provider as provider
        from openalph.tools import ToolDef
    except ImportError as exc:
        raise CriticOAUnavailableError(
            f"openalph provider layer unavailable — {exc!s}; the critic clients "
            "route through OA's provider layer (bead workspace-e2uh.143, A′); "
            "install openalph or the rig cannot start"
        ) from exc
    try:
        complete_fn = provider.complete  # existence probe — must carry kdsn.304
        provider_config_cls = provider.ProviderConfig
        agent_config_cls = provider.AgentConfig
    except AttributeError as exc:
        raise CriticOAUnavailableError(
            f"openalph provider layer present but missing {exc!s} — the "
            "installed openalph does not carry the kdsn.304 forced-tool "
            "provider support; the rig cannot start"
        ) from exc
    del complete_fn  # probe only — calls resolve through the module handle
    return provider, ToolDef, provider_config_cls, agent_config_cls


# ===========================================================================
# The one persistent event loop per process (spec §4.5 / AC10)
# ===========================================================================

_LOOP_THREAD_NAME = "stigmergy-oa-critic-loop"
_scratch_workspace: Path | None = None
_scratch_lock = threading.Lock()


def _scratch_workspace_dir() -> Path:
    """A recognizable scratch `workspace` for the minimal AgentConfig —
    the critic never touches it (AgentConfig requires the field,
    config.py:81); a per-process temp dir, created once."""
    global _scratch_workspace
    if _scratch_workspace is None:
        with _scratch_lock:
            if _scratch_workspace is None:
                _scratch_workspace = Path(
                    tempfile.mkdtemp(prefix="stigmergy-critic-ws-")
                )
    return _scratch_workspace


def _default_loop_runner(coro: Awaitable[Any]) -> Any:
    """The production bridge: ONE persistent event loop on ONE daemon
    thread, started lazily at first use and shared by all factories (and
    for the process lifetime). Submits via
    `asyncio.run_coroutine_threadsafe(coro, loop).result(600 s)` — the
    600 s ceiling sits above `_OA_TIMEOUT_SECONDS` + OA's SDK retry
    budget so OA's own transport timeouts fire first (a hung future
    here would only mean OA itself is wedged).

    This is the orphaned-pooled-client fix: OA's `_client_cache`
    retains httpx-bound SDK clients, and a per-call ephemeral loop would
    destroy the loop under the pooled client, failing the second gate
    call.
    """
    return _loop_runner_singleton.run(coro)


class _LoopRunnerSingleton:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def run(self, coro: Awaitable[Any]) -> Any:
        loop = self._ensure_started()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=_BRIDGE_FUTURE_TIMEOUT_SECONDS)

    def _ensure_started(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None or self._loop.is_closed():
                loop = asyncio.new_event_loop()

                def _run() -> None:
                    asyncio.set_event_loop(loop)
                    loop.run_forever()

                thread = threading.Thread(target=_run, name=_LOOP_THREAD_NAME, daemon=True)
                thread.start()
                self._thread = thread
                self._loop = loop
            return self._loop


_loop_runner_singleton = _LoopRunnerSingleton()


# ===========================================================================
# Per-call assembly (spec §3.4/§3.6c)
# ===========================================================================


def _tooldict_to_tooldef(
    tooldef_cls: Any, tool_dict: dict[str, Any]
) -> Any:
    """§3.5 bridge: mechanical dict-rename of the Anthropic-shaped tool
    dict into OA's ToolDef shape `{name, description, parameters,
    config}`. `parameters` IS the source `input_schema` object (same
    identity — no mutation, no copy; the `required` arrays ride inside,
    unmodified); `config` is `{}` (tool-harness metadata, not
    serialized)."""
    return tooldef_cls(
        name=tool_dict["name"],
        description=tool_dict["description"],
        parameters=tool_dict["input_schema"],
        config={},
    )


def _map_oa_usage(usage: Any) -> dict[str, int]:
    """Canonical 4-key usage mapping from an OA `Usage` object
    (`{"in", "cached", "out", "reasoning"}`). `reasoning` is pinned at 0
    until OA verifiably surfaces the reasoning-token channel
    (spec UNVERIFIED-5; the in/cached/out keys are the load-bearing ones
    for SpendLeash). Tolerant of missing cache fields (None -> 0)."""
    def _nonneg_int(value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return 0
        return value

    if usage is None:
        return {"in": 0, "cached": 0, "out": 0, "reasoning": 0}
    return {
        "in": _nonneg_int(getattr(usage, "input_tokens", 0)),
        "cached": _nonneg_int(getattr(usage, "cache_read_tokens", 0)),
        "out": _nonneg_int(getattr(usage, "output_tokens", 0)),
        "reasoning": 0,
    }


def _check_stop_reason(response: Any, station: str) -> None:
    """§3.2.2 / the .118 contract: a truncation stop_reason means the
    forced tool-input JSON was cut off — surface THAT (naming the
    `roles.critic.max_tokens` config), not a misleading downstream
    'missing required fields'. OA's normalized stop_reason vocabulary
    (spec UNVERIFIED-1) is matched as a SET so both the OpenAI-surface
    `length` and the Anthropic-surface `max_tokens` forms are covered;
    `refusal` keeps its own specific message (legacy behavior)."""
    stop_reason = getattr(response, "stop_reason", "")
    if stop_reason in ("length", "max_tokens"):
        raise CriticClientError(
            f"critic verdict truncated at stop_reason={stop_reason!r} (the "
            "forced tool-input JSON was cut off, dropping trailing required "
            f"fields) — raise roles.critic.max_tokens (stop_reason={stop_reason!r})"
        )
    if stop_reason == "refusal":
        raise CriticClientError(f"{station} refused to produce a verdict (stop_reason=refusal)")


def _tool_call_or_raise(response: Any, expected_tool: str) -> Any:
    """§3.2.2: exactly ONE tool call, with the EXPECTED name and a DICT
    input — mirrors the legacy extractor's fail-closed gate
    (critic_client.py:254-273 / :421-455). ToolCall.id is never
    inspected (opaque, any scheme — incl. Kimi's `name:0`)."""
    tool_calls = getattr(response, "tool_calls", None)
    if not isinstance(tool_calls, list) or len(tool_calls) != 1:
        count = len(tool_calls) if isinstance(tool_calls, list) else type(tool_calls).__name__
        raise CriticClientError(
            f"critic response has {count} tool call(s); exactly one "
            f"'{expected_tool}' call is required "
            f"(stop_reason={getattr(response, 'stop_reason', None)!r})"
        )
    call = tool_calls[0]
    if getattr(call, "name", None) != expected_tool:
        raise CriticClientError(
            f"critic response tool call name {getattr(call, 'name', None)!r} "
            f"is not '{expected_tool}' (stop_reason={getattr(response, 'stop_reason', None)!r})"
        )
    if not isinstance(call.input, dict):
        raise CriticClientError(
            f"critic response '{expected_tool}' tool call has non-dict input "
            f"(got {type(call.input).__name__})"
        )
    return call


def _provider_config(
    provider_config_cls: Any, entry: ModelEntry, key_provider: Callable[[], str]
) -> Any:
    """§3.4: the one-provider ProviderConfig. The key is taken from the
    injected `key_provider` (already fetched + cached by the caller) and
    placed in `api_key` — the ONLY place it lives. Built per call
    (never cached across models)."""
    return provider_config_cls(
        key=entry.oa_provider_key,  # type: ignore[arg-type]
        type=entry.oa_type,  # type: ignore[arg-type]
        api_key=key_provider(),
        base_url=entry.oa_base_url,
        timeout=_OA_TIMEOUT_SECONDS,
    )


def _minimal_agent_config(
    agent_config_cls: Any, provider_cfg: Any, entry: ModelEntry, max_tokens: int
) -> Any:
    """§3.4 minimum-viable AgentConfig: one provider, the fully-qualified
    default model, the charter max_tokens, and a scratch workspace the
    critic never touches."""
    return agent_config_cls(
        name="stigmergy-critic",
        default_model=f"{entry.oa_provider_key}/{entry.version}",
        max_tokens=max_tokens,
        providers={entry.oa_provider_key: provider_cfg},
        workspace=_scratch_workspace_dir(),
    )


# ===========================================================================
# The factories
# ===========================================================================

_UNSET_KEY = object()


@dataclass
class _FactoryState:
    """Per-factory state: the lazy key (fetch-once, failures never
    cached — mirrors `make_op_key_provider`'s posture) and the resolved
    OA symbols."""

    key_provider: Callable[[], str]
    registry: Registry
    max_tokens: int
    complete_fn: Callable[..., Awaitable[Any]]
    loop_runner: Callable[[Awaitable[Any]], Any]
    tooldef_cls: Any
    provider_config_cls: Any
    agent_config_cls: Any
    tool_name: str
    tool_dict: dict[str, Any]
    station: str
    key: Any = _UNSET_KEY

    def fetch_key(self) -> str:
        if self.key is _UNSET_KEY:
            self.key = self.key_provider()
        return self.key


def _make_oa_client(state: _FactoryState, is_range: bool) -> Callable[..., dict[str, Any]]:
    def client(prompt: str, *, model: str, **decoding_params: Any) -> dict[str, Any]:
        # §3.6b: fail LOUD on non-empty decoding params (the legacy client
        # spread them into the request body; complete() has no such spread).
        # Checked FIRST — before any provider call, registry resolve, or
        # key fetch.
        if decoding_params:
            raise CriticClientError(
                "critic client received non-empty decoding_params "
                f"{sorted(decoding_params)} — the OA provider-layer path has no "
                "decoding-params spread; roles.critic.decoding_params must be {} "
                "(fail-loud, not silently dropped)"
            )

        entry = state.registry.resolve(model)  # UnbudgetableError propagates

        cfg = _provider_config(state.provider_config_cls, entry, state.fetch_key)
        agent_cfg = _minimal_agent_config(
            state.agent_config_cls, cfg, entry, state.max_tokens
        )
        tooldef = _tooldict_to_tooldef(state.tooldef_cls, state.tool_dict)
        response = state.loop_runner(
            state.complete_fn(
                agent_cfg,
                system="",
                messages=[{"role": "user", "content": prompt}],
                tools=[tooldef],
                max_tokens=state.max_tokens,
                model=f"{entry.oa_provider_key}/{entry.version}",
                tool_choice=state.tool_name,
                strict=True,
                hardened=True,
            )
        )

        # §3.2.2 gate: truncation/refusal, then exactly-one-tool-call.
        _check_stop_reason(response, state.station)
        call = _tool_call_or_raise(response, state.tool_name)

        if is_range:
            # 1:1 with the legacy `_extract_range_review` return shape:
            # raw tool input (tolerant: `text` may be None/non-str,
            # `filed_tickets` may be None/malformed — RangeCritic.review
            # owns semantic validation) + the mapped usage.
            return {
                "text": call.input.get("findings"),
                "filed_tickets": call.input.get("filed_tickets"),
                "usage": _map_oa_usage(getattr(response, "usage", None)),
            }

        # Verdict: the raw tool-input dict VERBATIM (_parse_verdict is the
        # sole shape authority), plus the ADDITIVE usage channel (§2.4 gap
        # closed — judge tolerates an absent usage and records a present
        # one in gate_fields["tokens"]).
        result: dict[str, Any] = dict(call.input)
        result["usage"] = _map_oa_usage(getattr(response, "usage", None))
        return result

    return client


def make_oa_critic_client(
    *,
    key_provider: Callable[[], str],
    registry: Registry,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    complete_fn: Callable[..., Awaitable[Any]] | None = None,
    loop_runner: Callable[[Awaitable[Any]], Any] | None = None,
) -> Callable[..., dict[str, Any]]:
    """Build the OA-layer verdict client: `client(prompt, *, model,
    **decoding_params) -> dict` (the EXACT injected-callable signature
    `Critic.judge` ducks — returns the raw `submit_verdict` tool-input
    dict plus an additive `usage` key).

    `model` is the charter registry NAME (e.g. `"opus"`, `"kimi3"`); the
    call resolves it and sends `f"{entry.oa_provider_key}/
    {entry.version}"` fully qualified to `openalph.provider.complete`.

    Factory build is FAIL-CLOSED (Decision 1): `_import_oa_provider()`
    runs HERE, so an OA-less environment raises
    `CriticOAUnavailableError` at rig launch — never at the first gate.
    The build performs NO registry lookup and NO key fetch (both are
    per-call / lazy).

    `complete_fn` / `loop_runner` are test seams: production resolves
    `complete` via the OA import and the ONE persistent per-process
    event loop; tests substitute a stub `async def` complete and a
    synchronous runner (no production thread in unit tests).
    """
    try:
        provider, tooldef_cls, provider_config_cls, agent_config_cls = _import_oa_provider()
    except ImportError as exc:
        raise CriticOAUnavailableError(
            f"openalph provider layer unavailable — {exc!s}; the critic clients "
            "route through OA's provider layer (bead workspace-e2uh.143, A′); "
            "install openalph or the rig cannot start"
        ) from exc
    state = _FactoryState(
        key_provider=key_provider,
        registry=registry,
        max_tokens=max_tokens,
        complete_fn=complete_fn if complete_fn is not None else provider.complete,
        loop_runner=loop_runner if loop_runner is not None else _default_loop_runner,
        tooldef_cls=tooldef_cls,
        provider_config_cls=provider_config_cls,
        agent_config_cls=agent_config_cls,
        tool_name=VERDICT_TOOL_NAME,
        tool_dict=build_verdict_tool(),
        station="critic",
    )
    return _make_oa_client(state, is_range=False)


def make_oa_range_critic_client(
    *,
    key_provider: Callable[[], str],
    registry: Registry,
    max_tokens: int = DEFAULT_RANGE_MAX_TOKENS,
    complete_fn: Callable[..., Awaitable[Any]] | None = None,
    loop_runner: Callable[[Awaitable[Any]], Any] | None = None,
) -> Callable[..., dict[str, Any]]:
    """Build the OA-layer range-critic client: same signature and
    fail-closed build as :func:`make_oa_critic_client`, forcing
    `submit_range_review` instead, returning `{text, filed_tickets,
    usage}` 1:1 with the legacy `_extract_range_review` contract.
    (Beads .51 + .41 split: the client returns raw tool input + usage;
    `rangereport.RangeCritic.review` owns semantic validation.)
    """
    try:
        provider, tooldef_cls, provider_config_cls, agent_config_cls = _import_oa_provider()
    except ImportError as exc:
        raise CriticOAUnavailableError(
            f"openalph provider layer unavailable — {exc!s}; the critic clients "
            "route through OA's provider layer (bead workspace-e2uh.143, A′); "
            "install openalph or the rig cannot start"
        ) from exc
    state = _FactoryState(
        key_provider=key_provider,
        registry=registry,
        max_tokens=max_tokens,
        complete_fn=complete_fn if complete_fn is not None else provider.complete,
        loop_runner=loop_runner if loop_runner is not None else _default_loop_runner,
        tooldef_cls=tooldef_cls,
        provider_config_cls=provider_config_cls,
        agent_config_cls=agent_config_cls,
        tool_name=RANGE_REVIEW_TOOL_NAME,
        tool_dict=build_range_review_tool(),
        station="range-critic",
    )
    return _make_oa_client(state, is_range=True)


def make_oa_decompose_critic_client(
    *,
    key_provider: Callable[[], str],
    registry: Registry,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    complete_fn: Callable[..., Awaitable[Any]] | None = None,
    loop_runner: Callable[[Awaitable[Any]], Any] | None = None,
) -> Callable[..., dict[str, Any]]:
    """Build the OA-layer decomposer-station validation critic client
    (bead workspace-e2uh.152, Decision 17 — the decomposer band's gate 2 of
    2, the LLM decomposition-validation critic). Same signature and
    fail-closed build as :func:`make_oa_critic_client`, forcing
    `submit_validation` instead and naming the station `decompose-critic`.

    The client returns the RAW `submit_validation` tool-input dict VERBATIM
    plus an ADDITIVE `usage` key — no semantic parsing here; the decomposer
    station's manifest validator / repair-round driver owns the judgment
    semantics, exactly as `Critic.judge` owns verdict parsing and
    `RangeCritic.review` owns range-review semantics.

    Rig-level, no ticket (Decision 17): the decomposer band's LLM
    invocations — the decomposer exec run and this validation critic — have
    no parent ticket, the same justification precedent as `report` and
    `ticket-filed`; their record-plane events land under
    `EventType.DECOMPOSE` (bead .152), which carries a hash-bearing
    `prompt_artifact_hash` (SPEC §4).

    Factory build is FAIL-CLOSED (Decision 1): `_import_oa_provider()` runs
    HERE, so an OA-less environment raises `CriticOAUnavailableError` at
    rig launch — never at the first validation call. The build performs NO
    registry lookup and NO key fetch (both are per-call / lazy).

    `complete_fn` / `loop_runner` are test seams: production resolves
    `complete` via the OA import and the ONE persistent per-process event
    loop; tests substitute a stub `async def` complete and a synchronous
    runner (no production thread in unit tests).
    """
    try:
        provider, tooldef_cls, provider_config_cls, agent_config_cls = _import_oa_provider()
    except ImportError as exc:
        raise CriticOAUnavailableError(
            f"openalph provider layer unavailable — {exc!s}; the critic clients "
            "route through OA's provider layer (bead workspace-e2uh.143, A′); "
            "install openalph or the rig cannot start"
        ) from exc
    state = _FactoryState(
        key_provider=key_provider,
        registry=registry,
        max_tokens=max_tokens,
        complete_fn=complete_fn if complete_fn is not None else provider.complete,
        loop_runner=loop_runner if loop_runner is not None else _default_loop_runner,
        tooldef_cls=tooldef_cls,
        provider_config_cls=provider_config_cls,
        agent_config_cls=agent_config_cls,
        tool_name=DECOMPOSE_VALIDATION_TOOL_NAME,
        tool_dict=build_validation_tool(),
        station="decompose-critic",
    )
    return _make_oa_client(state, is_range=False)
