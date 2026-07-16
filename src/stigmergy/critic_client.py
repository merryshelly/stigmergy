"""Real provider-calling critic client (bead .36; SPEC.md §7 "Direct call,
no tool loop").

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
    never itself treated as validation."""
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
