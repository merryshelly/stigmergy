"""Shared secret-sourcing seam (bead .36; reused by bead .31's live relay
`key_provider`).

`keyprovider` is the one place stigmergy shells out to `op` (1Password CLI)
for a live provider key. Two security-relevant invariants dominate this
module:

- **A fetched secret NEVER lands in an exception message.** A partial secret
  echoed into a raised error would flow straight into logs/events — the
  transcript redaction machinery (`relay.build_redactor`) covers sealed
  transcripts, not exceptions, so this module must never hand a secret to
  `Exception.__init__` in the first place. On a non-zero `op` exit the
  message cites only the exit code, never stdout/stderr (stderr could, in
  principle, itself echo the value depending on how `op` fails).
- **A failure NEVER silently resolves to an empty/placeholder key.** Fail
  closed: raise :class:`KeyProviderError`, never return `""`.

Mirrors the repo's injected-callable discipline (`relay.py`'s
`key_provider`/`forwarder`, `critic.py`'s `client`, `notify.py`'s `Sender`):
the real subprocess boundary (`subprocess.run`) is a thin injected
callable, defaulted to the real implementation, replaced by a fake in
tests.
"""

from __future__ import annotations

import os
import subprocess
import threading
from collections.abc import Callable, Mapping
from pathlib import Path

# Standard on-disk locations for a 1Password service-account token, checked
# in order. Neither file's presence/absence is itself sensitive; only their
# contents are.
_TOKEN_FILE_CANDIDATES: tuple[Path, ...] = (
    Path.home() / ".config" / "op" / "service-account-token",
    Path.home() / ".config" / "op-token",
)

_SERVICE_ACCOUNT_TOKEN_VAR = "OP_SERVICE_ACCOUNT_TOKEN"

# Sentinel distinguishing "never fetched yet" from "fetched, value is falsy"
# (a real op secret should never be empty, but the sentinel keeps the cache
# logic honest regardless).
_UNSET = object()


class KeyProviderError(Exception):
    """Raised on any failure to source a secret: `op` binary missing, a
    non-zero exit, empty output, or a timeout. Never carries the secret
    value itself in its message."""


def _read_service_account_token() -> str | None:
    """Read a 1Password service-account token from the first candidate
    token file that exists and has non-empty content, or return `None` if
    none do. Monkeypatched wholesale by tests (by this exact module-level
    name) — never invoked directly in the test suite otherwise."""
    for path in _TOKEN_FILE_CANDIDATES:
        try:
            content = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if content:
            return content
    return None


def _build_child_env(env: Mapping[str, str] | None) -> dict[str, str]:
    """Build the child process env for `op read`: starts from `env`
    (default `os.environ`), and if `OP_SERVICE_ACCOUNT_TOKEN` is
    absent/empty, sources it from the standard token file and injects it
    into the child env ONLY — the passed-in `env`/`os.environ` mapping is
    never mutated in place. An already-present token is never overwritten.
    """
    child_env = dict(os.environ if env is None else env)
    if not child_env.get(_SERVICE_ACCOUNT_TOKEN_VAR):
        token = _read_service_account_token()
        if token:
            child_env[_SERVICE_ACCOUNT_TOKEN_VAR] = token
    return child_env


def op_read(
    ref: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    timeout: float = 15.0,
    env: Mapping[str, str] | None = None,
) -> str:
    """Fetch one secret from 1Password via `op read <ref>`.

    Runs `["op", "read", ref]` through `runner` (injected; default
    `subprocess.run`, `capture_output=True, text=True, check=False`). The
    child env sources `OP_SERVICE_ACCOUNT_TOKEN` itself (see
    `_build_child_env`) so callers do not need to pre-populate it.

    Returns `stdout.strip()` on a zero exit with non-empty output. Raises
    :class:`KeyProviderError` — never returning `""` — when: the `op`
    binary is missing (`FileNotFoundError`), the process times out
    (`subprocess.TimeoutExpired`), the exit code is non-zero, or the
    stripped stdout is empty. The non-zero-exit message cites only the ref
    and exit code, never stdout (the secret) or stderr.
    """
    child_env = _build_child_env(env)

    try:
        result = runner(
            ["op", "read", ref],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=child_env,
            check=False,
        )
    except FileNotFoundError as exc:
        raise KeyProviderError(f"op binary not found while reading {ref!r}") from exc
    except subprocess.TimeoutExpired as exc:
        raise KeyProviderError(f"op read timed out for {ref!r} (timeout={timeout!r})") from exc

    if result.returncode != 0:
        raise KeyProviderError(f"op read failed for {ref!r} (exit {result.returncode})")

    value = result.stdout.strip()
    if not value:
        raise KeyProviderError(f"op read returned empty output for {ref!r}")
    return value


def make_op_key_provider(
    ref: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    timeout: float = 15.0,
) -> Callable[[], str]:
    """Build a zero-arg callable that fetches `ref` via :func:`op_read`
    exactly once, caches the value, and returns the cached value on every
    subsequent call. Rotation mid-process is unsupported by design — the
    daemon is expected to restart to pick up a rotated key.

    Thread-safe (double-checked locking around a `threading.Lock`): N
    concurrent first-calls collapse into a single underlying `op_read`.
    A failed fetch is NEVER cached — the failure propagates and the cache
    stays empty so a later call can retry.
    """
    lock = threading.Lock()
    cached: list[object] = [_UNSET]

    def provider() -> str:
        if cached[0] is not _UNSET:
            return cached[0]  # type: ignore[return-value]
        with lock:
            if cached[0] is not _UNSET:
                return cached[0]  # type: ignore[return-value]
            value = op_read(ref, runner=runner, timeout=timeout)
            cached[0] = value
            return value

    return provider
