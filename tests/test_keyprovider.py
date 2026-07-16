"""Tests for the shared secret-sourcing seam (bead .36; reused by bead .31).

Authored by the orchestrator, not the implementor. `keyprovider` is the one
place stigmergy shells out to `op` (1Password) for a live provider key. Two
security-relevant invariants dominate this suite:

  * a fetched secret NEVER lands in an exception message (a partial secret in
    a raised error would flow into logs/events — redaction covers transcripts,
    not exceptions);
  * a failure NEVER silently resolves to an empty/placeholder key (fail closed:
    raise `KeyProviderError`, never return "").

The real `op` binary is never invoked here — the subprocess runner is an
injected callable (mirrors the repo's injected-callable discipline in
`relay.py`/`critic.py`/`notify.py`). These assertions are the fixed contract;
the implementation must satisfy them without weakening.
"""

from __future__ import annotations

import subprocess
import threading
import time

import pytest

from stigmergy.keyprovider import (
    KeyProviderError,
    make_op_key_provider,
    op_read,
)

REF = "op://shelly/API Credential - stigmergy rig 00/credential"
SECRET = "sk-ant-TESTSECRET-do-not-log-me"


class FakeRunner:
    """Stands in for `subprocess.run`. Records every call's args/kwargs and
    returns a canned `CompletedProcess` — or raises a canned exception."""

    def __init__(self, *, returncode=0, stdout=SECRET, stderr="", raises=None):
        self._returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self._raises = raises
        self.calls: list[dict] = []

    def __call__(self, args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        if self._raises is not None:
            raise self._raises
        return subprocess.CompletedProcess(
            args=args, returncode=self._returncode, stdout=self._stdout, stderr=self._stderr
        )


# --------------------------------------------------------------------------
# op_read — command construction + success
# --------------------------------------------------------------------------


def test_op_read_invokes_op_read_with_the_ref():
    runner = FakeRunner()
    op_read(REF, runner=runner)
    assert runner.calls[0]["args"] == ["op", "read", REF]


def test_op_read_returns_stripped_stdout():
    runner = FakeRunner(stdout=f"  {SECRET}\n")
    assert op_read(REF, runner=runner) == SECRET


def test_op_read_passes_timeout_to_runner():
    runner = FakeRunner()
    op_read(REF, runner=runner, timeout=3.5)
    assert runner.calls[0]["kwargs"].get("timeout") == 3.5


# --------------------------------------------------------------------------
# op_read — child env sources the service-account token itself
# --------------------------------------------------------------------------


def test_op_read_injects_token_from_file_when_absent(monkeypatch):
    monkeypatch.setattr(
        "stigmergy.keyprovider._read_service_account_token", lambda: "TOK-FROM-FILE"
    )
    runner = FakeRunner()
    op_read(REF, runner=runner, env={"PATH": "/usr/bin"})
    child_env = runner.calls[0]["kwargs"]["env"]
    assert child_env["OP_SERVICE_ACCOUNT_TOKEN"] == "TOK-FROM-FILE"


def test_op_read_does_not_overwrite_token_already_in_env(monkeypatch):
    monkeypatch.setattr(
        "stigmergy.keyprovider._read_service_account_token", lambda: "SHOULD-NOT-USE"
    )
    runner = FakeRunner()
    op_read(REF, runner=runner, env={"OP_SERVICE_ACCOUNT_TOKEN": "TOK-FROM-ENV"})
    child_env = runner.calls[0]["kwargs"]["env"]
    assert child_env["OP_SERVICE_ACCOUNT_TOKEN"] == "TOK-FROM-ENV"


# --------------------------------------------------------------------------
# op_read — fail closed, never leak the secret in the error
# --------------------------------------------------------------------------


def test_op_read_nonzero_exit_raises_and_does_not_leak_secret():
    # stdout carries the secret even on a nonzero exit — the error must not echo it.
    runner = FakeRunner(returncode=1, stdout=SECRET, stderr="some op diagnostic")
    with pytest.raises(KeyProviderError) as exc_info:
        op_read(REF, runner=runner)
    assert SECRET not in str(exc_info.value)


def test_op_read_empty_output_raises_never_returns_empty():
    runner = FakeRunner(returncode=0, stdout="   \n")
    with pytest.raises(KeyProviderError):
        op_read(REF, runner=runner)


def test_op_read_missing_binary_raises_keyprovider_error():
    runner = FakeRunner(raises=FileNotFoundError("op"))
    with pytest.raises(KeyProviderError):
        op_read(REF, runner=runner)


def test_op_read_timeout_raises_keyprovider_error():
    runner = FakeRunner(raises=subprocess.TimeoutExpired(cmd=["op", "read", REF], timeout=1.0))
    with pytest.raises(KeyProviderError):
        op_read(REF, runner=runner)


# --------------------------------------------------------------------------
# make_op_key_provider — cache once, never cache failures, thread-safe
# --------------------------------------------------------------------------


def test_provider_caches_after_first_fetch():
    runner = FakeRunner()
    provider = make_op_key_provider(REF, runner=runner)
    first = provider()
    second = provider()
    assert first == second == SECRET
    assert len(runner.calls) == 1  # cached — runner invoked exactly once


def test_provider_does_not_cache_failures():
    # First fetch fails; a later fetch with a healthy runner must succeed.
    class FlakyRunner:
        def __init__(self):
            self.calls = 0

        def __call__(self, args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise FileNotFoundError("op")
            return subprocess.CompletedProcess(args, 0, stdout=SECRET, stderr="")

    runner = FlakyRunner()
    provider = make_op_key_provider(REF, runner=runner)
    with pytest.raises(KeyProviderError):
        provider()
    assert provider() == SECRET  # retried, not stuck on the cached failure


def test_provider_is_thread_safe_single_fetch():
    class SlowRunner:
        def __init__(self):
            self.calls = 0
            self._lock = threading.Lock()

        def __call__(self, args, **kwargs):
            with self._lock:
                self.calls += 1
            time.sleep(0.05)  # widen the race window
            return subprocess.CompletedProcess(args, 0, stdout=SECRET, stderr="")

    runner = SlowRunner()
    provider = make_op_key_provider(REF, runner=runner)
    results: list[str] = []
    results_lock = threading.Lock()

    def worker():
        val = provider()
        with results_lock:
            results.append(val)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results == [SECRET] * 8
    assert runner.calls == 1  # concurrent first-calls collapse to a single fetch
