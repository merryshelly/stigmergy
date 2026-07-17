"""Real provider-calling critic clients (bead .36 verdict client; beads
.51 + .41 range-critic client; SPEC.md §7 "Direct call, no tool loop").

`.27` wired the critic with a LOUD placeholder client
(`cli._unwired_critic_client`) that raised unconditionally;
`Critic.judge()` converts ANY client exception into `CriticInfraError` (a
visible circuit-breaker trip, never a silent wrong gate verdict). This
module builds the real client and is a **direct one-shot, no-tools**
provider call in the SPEC §7 sense — it does not run a tool-use loop with
the model. It forces exactly one round of *structured output* via a
synthetic `submit_verdict` tool + `tool_choice`, purely as a mechanism to
get a validated-shape response back, not as an agentic tool loop.

The client only has two jobs: (a) return the raw tool-input dict, or (b)
raise. `critic._parse_verdict` remains the SOLE authority on verdict shape
— this module never re-implements that validation, and a malformed/missing
tool_use block is `CriticClientError` (infra), never a fabricated verdict.
`build_verdict_tool`'s schema also carries an OPTIONAL `filed_tickets`
channel (D14, bead `.39`) — the raw tool-input dict simply carries it
verbatim when the model populates it; this module does not extract or
validate it (`critic.py`'s tolerant `_extract_filed_tickets` does).

`make_range_critic_client` (beads .51 + .41) is a SECOND, sibling direct
client for the range-critic role. `.51` was a wiring bug: production
`range-report --critic` used `make_critic_client` (the verdict client),
whose `{outcome,tier,reason,severity}` shape does not match what
`RangeCritic.review` needs. This adapter forces a dedicated
`submit_range_review` tool whose schema carries BOTH advisory prose
(`findings`) and proposed follow-up tickets (`filed_tickets`) — one API
call, one combined structured response (`.41`'s combined-schema
decision). It reuses every hardened primitive below verbatim
(`_default_http_post`, `_NO_REDIRECT_OPENER`, `_MAX_RESPONSE_BYTES`,
`ANTHROPIC_VERSION`, `DEFAULT_BASE_URL`, `DEFAULT_TIMEOUT`) and follows
the same fail-closed/no-validation-here split as `make_critic_client`:
this module returns the raw tool input + a provenance-gated usage
mapping, never validating `findings`/`filed_tickets` semantics itself —
that is `rangereport.RangeCritic.review`'s job.

Mirrors the repo's injected-callable discipline (`relay.py`'s
`key_provider`/`forwarder`, `critic.py`'s `client`, `notify.py`'s `Sender`):
the real HTTP transport (`_default_http_post`, stdlib `urllib.request`
only — this repo is deliberately zero third-party dependency) is a thin
injected callable, defaulted to the real implementation, replaced by a
fake in tests.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from stigmergy.registry import Registry

ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_BASE_URL = "https://api.anthropic.com"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT = 60.0
VERDICT_TOOL_NAME = "submit_verdict"
RANGE_REVIEW_TOOL_NAME = "submit_range_review"
# Range prose + proposals are larger than a verdict's 1024-token budget.
DEFAULT_RANGE_MAX_TOKENS = 4096
# Cap the response body read (defense in depth: a compromised/misconfigured
# endpoint could otherwise stream an unbounded body into memory). A one-shot
# verdict response is a few KB; 10 MiB is generous. An over-cap body is
# truncated -> json.loads fails -> CriticInfraError (fail closed).
_MAX_RESPONSE_BYTES = 10 * 1024 * 1024


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse to follow 3xx redirects. `urllib` re-sends request headers —
    including `x-api-key` — to the redirect target by default, so a
    compromised or misconfigured endpoint returning a 302 to an attacker
    host would leak the provider key on the second hop. Treating any
    redirect as an error keeps the key on the single intended (TLS-verified)
    hop; the raised `HTTPError` propagates -> `CriticInfraError`."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        raise urllib.error.HTTPError(
            req.full_url, code, "refusing redirect (key-exfil guard)", headers, fp
        )


# A module-level opener that verifies TLS (urllib default) and does NOT follow
# redirects. Injected transports in tests replace `open` on this object.
_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirectHandler)


class CriticClientError(Exception):
    """Raised when the critic client itself cannot produce a raw verdict
    input dict: a non-Anthropic registry entry, or a provider response
    that lacks a well-formed `submit_verdict` tool_use block.
    `Critic.judge` wraps this (like any client exception) into
    `CriticInfraError` — never a rejection verdict."""


def build_verdict_tool() -> dict[str, Any]:
    """The synthetic Anthropic tool definition that forces the model to
    return exactly one structured verdict. Mirrors `Outcome`/`Severity`
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
        "description": (
            "Submit the final structured verdict for the artifact under review. "
            "Call this exactly once with your judgment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "outcome": {
                    "type": "string",
                    "enum": ["met", "unmet"],
                    "description": "Whether the artifact meets the rubric.",
                },
                "tier": {
                    "type": "integer",
                    "description": "Which rubric tier this verdict is judging.",
                },
                "reason": {
                    "type": "string",
                    "description": "Non-empty human-readable justification.",
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
                    },
                },
            },
            "required": ["outcome", "tier", "reason", "severity"],
        },
    }


def _default_http_post(
    url: str, *, headers: dict[str, str], body: bytes, timeout: float
) -> dict[str, Any]:
    """The production HTTP transport: stdlib `urllib.request` only. Builds
    a POST (inferred by passing `data=body`), passes an EXPLICIT `timeout`
    (urllib has no default of its own — an un-timed-out call would wedge the
    daemon loop forever), and does NOT swallow a non-2xx response:
    `urllib.error.HTTPError` propagates uncaught (the caller, `Critic.judge`,
    turns it into `CriticInfraError`). Uses `_NO_REDIRECT_OPENER` so a 3xx
    can never re-send `x-api-key` to a redirect target, and bounds the body
    read at `_MAX_RESPONSE_BYTES`. Returns the parsed JSON response as a dict.
    """
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with _NO_REDIRECT_OPENER.open(request, timeout=timeout) as response:
        return json.loads(response.read(_MAX_RESPONSE_BYTES))


def _extract_verdict_input(response: dict[str, Any]) -> dict[str, Any]:
    """Pull the raw `submit_verdict` tool_use block's `input` dict out of
    an Anthropic Messages response, verbatim — never validating its
    fields (that's `_parse_verdict`'s job in `critic.py`).

    Raises :class:`CriticClientError` if `content` is missing/not a list,
    no `type=="tool_use"`/`name==VERDICT_TOOL_NAME` block exists, or the
    matching block's `input` is not a dict.
    """
    content = response.get("content")
    if not isinstance(content, list):
        raise CriticClientError(
            f"critic response 'content' is not a list (got {type(content).__name__})"
        )

    for block in content:
        if (
            isinstance(block, dict)
            and block.get("type") == "tool_use"
            and block.get("name") == VERDICT_TOOL_NAME
        ):
            tool_input = block.get("input")
            if not isinstance(tool_input, dict):
                raise CriticClientError(
                    f"critic response '{VERDICT_TOOL_NAME}' tool_use has non-dict input "
                    f"(got {type(tool_input).__name__})"
                )
            return tool_input

    raise CriticClientError(
        f"critic response has no '{VERDICT_TOOL_NAME}' tool_use block "
        f"(stop_reason={response.get('stop_reason')!r})"
    )


def make_critic_client(
    *,
    key_provider: Callable[[], str],
    registry: Registry,
    base_url: str = DEFAULT_BASE_URL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: float = DEFAULT_TIMEOUT,
    http_post: Callable[..., dict[str, Any]] = _default_http_post,
) -> Callable[..., dict[str, Any]]:
    """Build the real critic `client` callable that `Critic.judge` invokes
    as `client(prompt, model=self.model, **decoding_params)`.

    `model` is the charter registry NAME (e.g. `"opus"`), never a provider
    API model id directly. Per call:

    1. `entry = registry.resolve(model)` — a registry miss raises
       `UnbudgetableError`, propagated uncaught (before any network call;
       `Critic.judge` wraps it into `CriticInfraError`).
    2. `entry.provider != "anthropic"` -> `CriticClientError` (this v0
       direct client is Anthropic-only) — also before any network call.
    3. `api_model = entry.version` — production convention: the registry
       `version` field IS the exact provider API model id.
    4. Build the Anthropic Messages request body, forcing the
       `submit_verdict` tool via `tool_choice`.
    5. Source the real key via `key_provider()` and send it as
       `x-api-key`.
    6. POST via `http_post` (default: stdlib `urllib.request`) and return
       `_extract_verdict_input(response)` — the raw tool input dict,
       verbatim; validation is `_parse_verdict`'s job, not this module's.
    """

    def client(prompt: str, *, model: str, **decoding_params: Any) -> dict[str, Any]:
        entry = registry.resolve(model)
        if entry.provider != "anthropic":
            raise CriticClientError(
                f"model {model!r} has provider {entry.provider!r}; the direct critic "
                "client only supports 'anthropic'"
            )

        # decoding_params (pinned charter config) is spread FIRST so the
        # structural keys below always win — a caller (or future reuse) can
        # never clobber model/tool_choice/tools/messages via decoding_params.
        body: dict[str, Any] = {
            **decoding_params,
            "model": entry.version,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "tools": [build_verdict_tool()],
            "tool_choice": {"type": "tool", "name": VERDICT_TOOL_NAME},
        }

        key = key_provider()
        headers = {
            "x-api-key": key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

        response = http_post(
            f"{base_url}/v1/messages",
            headers=headers,
            body=json.dumps(body).encode("utf-8"),
            timeout=timeout,
        )
        return _extract_verdict_input(response)

    return client


# ==========================================================================
# make_range_critic_client — combined-schema range-critic adapter
# (beads .51 + .41).
# ==========================================================================


def build_range_review_tool() -> dict[str, Any]:
    """The synthetic Anthropic tool definition that forces the model to
    return exactly one structured range review carrying BOTH advisory
    prose (`findings`) and proposed follow-up tickets (`filed_tickets`).
    `filed_tickets` is deliberately NOT in the top-level `required` list —
    a clean range that raises no cross-cutting/interlink concerns files
    nothing. This schema is only a HINT to the model; `RangeCritic.review`
    (`rangereport.py`) remains the sole authority we actually trust — this
    module never re-validates `findings`/`filed_tickets` semantics."""
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


def _coerce_nonneg_int(raw: dict[str, Any], key: str) -> int:
    """Local copy of `drivers/claude_code._coerce_nonneg_int`'s idiom —
    this module must not import from `drivers/` (a critic client is not a
    driver). `raw.get(key, 0)`, coerced to 0 if not a non-negative `int`
    (bool excluded)."""
    value = raw.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _map_range_usage(raw: Any) -> dict[str, int]:
    """Provenance-gated canonical usage mapping for the range-critic
    response's top-level `usage` object (beads .51 + .41 build spec §3.1).

    The empty-dict sentinel `{}` is the ONLY signal `cli._emit_report_event`
    uses to decide "unbudgetable" — this function must never fabricate a
    usage number:

    - `raw` is not a dict -> `{}` (no usage object at all).
    - `raw` lacks a valid non-negative `int` `output_tokens` (bool
      excluded, negative excluded) -> `{}` (a usage object with no
      trustworthy completion-token count is not real metered accounting —
      note this is a stricter, un-coerced check than `_coerce_nonneg_int`,
      deliberately: coercing a bad `output_tokens` to 0 would make an
      invalid usage object look like a valid `out=0` reading).
    - Otherwise, map onto the canonical 4-key shape (mirrors
      `drivers/claude_code._map_usage`): `"in"` = `input_tokens` +
      `cache_creation_input_tokens`; `"cached"` = `cache_read_input_tokens`;
      `"out"` = `output_tokens`; `"reasoning"` = `0`.
    """
    if not isinstance(raw, dict):
        return {}
    output_tokens = raw.get("output_tokens")
    if isinstance(output_tokens, bool) or not isinstance(output_tokens, int):
        return {}
    if output_tokens < 0:
        return {}
    return {
        "in": _coerce_nonneg_int(raw, "input_tokens")
        + _coerce_nonneg_int(raw, "cache_creation_input_tokens"),
        "cached": _coerce_nonneg_int(raw, "cache_read_input_tokens"),
        "out": output_tokens,
        "reasoning": 0,
    }


def _extract_range_review(response: dict[str, Any]) -> dict[str, Any]:
    """Pull the raw `submit_range_review` tool_use block's `input` dict out
    of an Anthropic Messages response (mirrors `_extract_verdict_input`),
    then combine it with the mapped usage into the client's return shape.

    Raises :class:`CriticClientError` if `content` is missing/not a list,
    no `type=="tool_use"`/`name==RANGE_REVIEW_TOOL_NAME` block exists, or
    the matching block's `input` is not a dict.

    Does NOT validate `findings`/`filed_tickets` semantics — `text` may be
    `None`/non-str here, `filed_tickets` may be `None`/malformed here;
    that's `RangeCritic.review`'s job (mirrors the `.36` split: client
    returns raw tool input + usage, `rangereport.py` owns semantic
    validation).
    """
    content = response.get("content")
    if not isinstance(content, list):
        raise CriticClientError(
            f"range-critic response 'content' is not a list (got {type(content).__name__})"
        )

    for block in content:
        if (
            isinstance(block, dict)
            and block.get("type") == "tool_use"
            and block.get("name") == RANGE_REVIEW_TOOL_NAME
        ):
            tool_input = block.get("input")
            if not isinstance(tool_input, dict):
                raise CriticClientError(
                    f"range-critic response '{RANGE_REVIEW_TOOL_NAME}' tool_use has "
                    f"non-dict input (got {type(tool_input).__name__})"
                )
            return {
                "text": tool_input.get("findings"),
                "filed_tickets": tool_input.get("filed_tickets"),
                "usage": _map_range_usage(response.get("usage")),
            }

    raise CriticClientError(
        f"range-critic response has no '{RANGE_REVIEW_TOOL_NAME}' tool_use block "
        f"(stop_reason={response.get('stop_reason')!r})"
    )


def make_range_critic_client(
    *,
    key_provider: Callable[[], str],
    registry: Registry,
    base_url: str = DEFAULT_BASE_URL,
    max_tokens: int = DEFAULT_RANGE_MAX_TOKENS,
    timeout: float = DEFAULT_TIMEOUT,
    http_post: Callable[..., dict[str, Any]] = _default_http_post,
) -> Callable[..., dict[str, Any]]:
    """Build the real range-critic `client` callable that
    `RangeCritic.review` invokes as
    `client(prompt, model=self.model, **decoding_params)`.

    Mirrors `make_critic_client` exactly, except the forced tool is
    `submit_range_review` (`build_range_review_tool`) and the response is
    extracted via `_extract_range_review` into `{text, filed_tickets,
    usage}` rather than a raw verdict-tool-input dict:

    1. `entry = registry.resolve(model)` — a registry miss raises
       `UnbudgetableError`, propagated uncaught, before any network call.
    2. `entry.provider != "anthropic"` -> `CriticClientError`, also before
       any network call.
    3. Build the Anthropic Messages request body, forcing the
       `submit_range_review` tool via `tool_choice`. `decoding_params` is
       spread FIRST so the structural keys always win.
    4. Source the real key via `key_provider()` and send it as
       `x-api-key`.
    5. POST via `http_post` and return `_extract_range_review(response)`.
    """

    def client(prompt: str, *, model: str, **decoding_params: Any) -> dict[str, Any]:
        entry = registry.resolve(model)
        if entry.provider != "anthropic":
            raise CriticClientError(
                f"model {model!r} has provider {entry.provider!r}; the direct range-"
                "critic client only supports 'anthropic'"
            )

        body: dict[str, Any] = {
            **decoding_params,
            "model": entry.version,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "tools": [build_range_review_tool()],
            "tool_choice": {"type": "tool", "name": RANGE_REVIEW_TOOL_NAME},
        }

        key = key_provider()
        headers = {
            "x-api-key": key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

        response = http_post(
            f"{base_url}/v1/messages",
            headers=headers,
            body=json.dumps(body).encode("utf-8"),
            timeout=timeout,
        )
        return _extract_range_review(response)

    return client
