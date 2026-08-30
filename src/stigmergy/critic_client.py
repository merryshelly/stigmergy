"""Legacy critic-client module — bead .36/.51/.41/.118, migrated by
bead workspace-e2uh.143 (A′).

The real provider-calling machinery (the bare-urllib Anthropic Messages
clients `make_critic_client` / `make_range_critic_client`, their HTTP
transport, response extractors, usage mapper, and the tool-schema
builders / constants / error class they carried) now lives in
`stigmergy.oa_critic` — the OA provider-layer seam:

- `build_verdict_tool` / `build_range_review_tool` (moved, BYTE-IDENTICAL
  — the spike-verified wire contract), `CriticClientError`,
  `VERDICT_TOOL_NAME`, `RANGE_REVIEW_TOOL_NAME`, `DEFAULT_MAX_TOKENS`,
  `DEFAULT_RANGE_MAX_TOKENS` are re-exported below as DEPRECATED
  aliases; existing imports (this repo's tests, external consumers)
  keep working, but new code must import from `stigmergy.oa_critic`.
- The replaced urllib machinery (`make_critic_client`,
  `make_range_critic_client`, `_default_http_post`,
  `_extract_verdict_input`, `_extract_range_review`,
  `_map_range_usage`, `_coerce_nonneg_int`, `ANTHROPIC_VERSION`,
  `DEFAULT_BASE_URL`, `DEFAULT_TIMEOUT`, `_MAX_RESPONSE_BYTES`,
  `_NO_REDIRECT_OPENER`) is DELETED. Its three key-exfil guards
  (no-redirect, no-proxy, bounded read) are subsumed 1:1 by kdsn.304's
  `hardened=True` transport in the OA provider layer (see
  `oa_critic`'s module docstring).

KEEP (load-bearing for the relay): `_NoRedirectHandler` —
`relay_transport.py` imports it and builds the relay's own no-redirect
opener on top of it (relay_transport.py:104, :241). The class is now
relay-owned-by-consumption; a follow-up bead may move it into
`relay_transport.py` and delete this module entirely (deferred here to
keep this bead's blast radius off the relay test surface).
"""

from __future__ import annotations

import urllib.error
import urllib.request

# DEPRECATED re-export aliases (bead workspace-e2uh.143): the source of
# truth moved to stigmergy.oa_critic. Kept so existing imports keep
# working; new code must import from stigmergy.oa_critic directly.
from stigmergy.oa_critic import (  # noqa: F401  (re-export by design)
    DEFAULT_MAX_TOKENS,
    DEFAULT_RANGE_MAX_TOKENS,
    RANGE_REVIEW_TOOL_NAME,
    VERDICT_TOOL_NAME,
    CriticClientError,
    build_range_review_tool,
    build_verdict_tool,
)

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_RANGE_MAX_TOKENS",
    "RANGE_REVIEW_TOOL_NAME",
    "VERDICT_TOOL_NAME",
    "CriticClientError",
    "build_range_review_tool",
    "build_verdict_tool",
    "_NoRedirectHandler",
]


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse to follow 3xx redirects. `urllib` re-sends request headers —
    including `x-api-key` — to the redirect target by default, so a
    compromised or misconfigured endpoint returning a 302 to an attacker
    host would leak the provider key on the second hop. Treating any
    redirect as an error keeps the key on the single intended (TLS-verified)
    hop; the raised `HTTPError` propagates to the caller as a transport
    failure (fail closed).

    bead .143: the critic's bare-urllib transport that originally
    instantiated this handler is gone (OA provider layer, hardened=True);
    the class survives because `relay_transport.py` reuses it for the
    relay's own key-bearing upstream call (relay-owned-by-consumption)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        raise urllib.error.HTTPError(
            req.full_url, code, "refusing redirect (key-exfil guard)", headers, fp
        )
