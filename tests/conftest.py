"""Shared test seams (bead workspace-e2uh.143).

The critic seam (`stigmergy.oa_critic`) imports `openalph.*` LAZILY at
factory build. Rigs run under an OA editable install, but the test
environment may not have one on this python's import path. To keep the
suite runnable in BOTH environments (Decision 1's packaging posture —
the suite must not hard-require OA) this conftest installs a minimal
STRUCTURAL fake of the `openalph` provider layer into `sys.modules` ONLY
when the real package is not importable.

The fake mirrors the kdsn.304 shapes (openalph b3d95b8) exactly where
the adapter touches them: `provider.complete(config, *, system,
messages, tools, max_tokens, model, tool_choice, strict, hardened)`,
`tools.ToolDef{name, description, parameters, config}`,
`provider.ProviderConfig{key, type, api_key, base_url, timeout}`,
`provider.AgentConfig{name, default_model, max_tokens, providers,
workspace}`. The fake's `complete` RAISES if actually invoked — every
test that drives a client injects its own `complete_fn` stub, so the
fake exists purely so the lazy import seam resolves (and so a
half-installed/renamed real OA is not silently required).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from typing import Any


def _openalph_importable() -> bool:
    return importlib.util.find_spec("openalph") is not None


def _install_fake_openalph() -> None:
    openalph = types.ModuleType("openalph")
    openalph.__path__ = []  # mark as a package
    tools = types.ModuleType("openalph.tools")
    provider = types.ModuleType("openalph.provider")

    @dataclass
    class ToolDef:
        """Structural twin of `openalph.tools.ToolDef` (tools/__init__.py:123-127)."""

        name: str
        description: str
        parameters: dict
        config: dict

    @dataclass
    class ProviderConfig:
        """Structural twin of `openalph.config.ProviderConfig` (config.py:47-60)."""

        key: str
        type: str
        api_key: str
        base_url: str | None = None
        timeout: float = 600.0

    @dataclass
    class AgentConfig:
        """Structural twin of `openalph.config.AgentConfig` (config.py:76-111) —
        the fields the adapter's minimal config sets, nothing more."""

        name: str
        default_model: str
        max_tokens: int
        providers: dict
        workspace: Any

    async def complete(  # pragma: no cover - tests inject their own complete_fn
        config: Any,
        system: str,
        messages: list[dict],
        tools: list | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        thinking: str | None = None,
        cache_ttl: str | None = None,
        room_id: str | None = None,
        tool_choice: str | None = None,
        strict: bool = False,
        hardened: bool = False,
    ) -> Any:
        raise AssertionError(
            "fake openalph.provider.complete invoked — tests must inject "
            "complete_fn (no live provider calls in the unit suite)"
        )

    tools.ToolDef = ToolDef
    provider.ToolDef = ToolDef
    provider.ProviderConfig = ProviderConfig
    provider.AgentConfig = AgentConfig
    provider.complete = complete
    openalph.tools = tools
    openalph.provider = provider

    sys.modules.setdefault("openalph", openalph)
    sys.modules.setdefault("openalph.tools", tools)
    sys.modules.setdefault("openalph.provider", provider)


if not _openalph_importable():
    _install_fake_openalph()
