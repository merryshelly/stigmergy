"""Tests for stigmergy.checks (SPEC.md §3 `check` station, §4
"Judgment-surface hardening").

Two tiers, matching `test_container.py`'s convention:

- **Protocol tests** (deterministic): exercise the flake protocol and
  output-bounding logic via an injected ``run_one`` returning scripted
  exit codes — no container involved, no nondeterminism.
- **Live tests** (real rootless podman): a pinned image resolved
  programmatically (never hand-copied) proves the real default runner
  actually contains a check — no network egress at all, and the source
  work tree is never mutated.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from stigmergy.checks import (
    CheckOutcome,
    CheckResult,
    run_check,
    run_checks,
)
from stigmergy.container import worker_env

PODMAN = shutil.which("podman")
requires_podman = pytest.mark.skipif(PODMAN is None, reason="podman not installed")

LIVE_IMAGE = "docker.io/library/python:3.12-alpine"


@pytest.fixture(scope="module")
def pinned_live_image():
    """A digest-pinned ref for LIVE_IMAGE, resolved programmatically (never
    hand-copied — digests are redacted in this environment's view)."""
    if PODMAN is None:
        pytest.skip("podman not installed")
    subprocess.run(
        ["podman", "pull", LIVE_IMAGE],
        env=worker_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    ref = subprocess.run(
        ["podman", "inspect", "--format", "{{index .RepoDigests 0}}", LIVE_IMAGE],
        env=worker_env(),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert "@sha256:" in ref
    return ref


class ScriptedRunOne:
    """Deterministic stand-in for the real containerized single-attempt
    executor: returns exit codes from a scripted list (one per call, the
    last value repeats if there are more attempts than scripted codes) and
    records every call so tests can assert a fresh invocation happened per
    attempt — proving the flake protocol never reuses a prior result."""

    def __init__(self, codes: list[int], output: str = "scripted output"):
        self.codes = list(codes)
        self.output = output
        self.calls: list[tuple[str, object, str]] = []

    def __call__(self, command, work_tree, *, image):
        self.calls.append((command, work_tree, image))
        idx = len(self.calls) - 1
        code = self.codes[idx] if idx < len(self.codes) else self.codes[-1]
        return code, self.output


# --------------------------------------------------------------------------
# Protocol tests (deterministic, injected run_one — no container)
# --------------------------------------------------------------------------


def test_always_pass_is_outcome_pass(tmp_path):
    runner = ScriptedRunOne([0])
    result = run_check(
        "lint",
        "ruff check .",
        tmp_path,
        image="unused",
        flake_reruns=2,
        run_one=runner,
    )
    assert result.outcome is CheckOutcome.PASS
    assert result.runs == (0,)
    assert len(runner.calls) == 1


def test_fail_then_pass_is_outcome_flaky_not_pass(tmp_path):
    runner = ScriptedRunOne([1, 0])
    result = run_check(
        "pytest",
        "pytest -x -q",
        tmp_path,
        image="unused",
        flake_reruns=2,
        run_one=runner,
    )
    assert result.outcome is CheckOutcome.FLAKY
    assert result.outcome is not CheckOutcome.PASS
    assert result.runs == (1, 0)


def test_always_fail_is_outcome_fail_with_all_reruns_consumed(tmp_path):
    runner = ScriptedRunOne([1, 1, 1])
    result = run_check(
        "pytest",
        "pytest -x -q",
        tmp_path,
        image="unused",
        flake_reruns=2,
        run_one=runner,
    )
    assert result.outcome is CheckOutcome.FAIL
    assert len(result.runs) == 1 + 2
    assert result.runs == (1, 1, 1)


def test_run_one_invoked_fresh_each_attempt(tmp_path):
    # Proves fresh-container-per-run: a rerun is a brand-new invocation,
    # not a cached/replayed result. flake_reruns is generous (5) so the
    # protocol must stop the moment attempt 3 exits 0.
    runner = ScriptedRunOne([1, 1, 0])
    result = run_check(
        "pytest",
        "pytest -x -q",
        tmp_path,
        image="unused",
        flake_reruns=5,
        run_one=runner,
    )
    assert len(runner.calls) == 3
    assert result.runs == (1, 1, 0)
    assert result.outcome is CheckOutcome.FLAKY


def test_run_checks_returns_one_result_per_check_preserving_names(tmp_path, monkeypatch):
    # run_checks has no run_one parameter (per interface) — it always uses
    # the real default runner. Keep this deterministic and container-free
    # by monkeypatching the module-level default executor: each check's
    # scripted outcome is looked up by the check's *command* string.
    import stigmergy.checks as checks_module

    codes_by_command = {
        "lint-cmd": [0],
        "pytest-cmd": [1, 0],
        "typecheck-cmd": [1, 1],
    }
    runners = {cmd: ScriptedRunOne(codes) for cmd, codes in codes_by_command.items()}

    def fake_default_run_one(command, work_tree, *, image):
        return runners[command](command, work_tree, image=image)

    monkeypatch.setattr(checks_module, "_default_run_one", fake_default_run_one)

    checks = {"lint": "lint-cmd", "pytest": "pytest-cmd", "typecheck": "typecheck-cmd"}
    results = run_checks(checks, tmp_path, image="unused", flake_reruns=2)

    assert [r.name for r in results] == list(checks.keys())
    assert all(isinstance(r, CheckResult) for r in results)
    assert results[0].outcome is CheckOutcome.PASS
    assert results[1].outcome is CheckOutcome.FLAKY
    assert results[2].outcome is CheckOutcome.FAIL


def test_output_is_captured_and_bounded(tmp_path):
    huge = "x" * 20000
    runner = ScriptedRunOne([0], output=huge)
    result = run_check(
        "lint",
        "ruff check .",
        tmp_path,
        image="unused",
        flake_reruns=0,
        run_one=runner,
    )
    assert isinstance(result.output, str)
    assert len(result.output.encode("utf-8")) <= 4096
    assert len(result.output) < len(huge)
    assert result.output == huge[-len(result.output) :]


# --------------------------------------------------------------------------
# Live tests (real rootless podman, pinned image resolved programmatically)
# --------------------------------------------------------------------------


@requires_podman
def test_live_success_command_passes(tmp_path, pinned_live_image):
    work_tree = tmp_path / "src"
    work_tree.mkdir()
    (work_tree / "README.txt").write_text("hello\n")

    result = run_check(
        "trivial-pass",
        "true",
        work_tree,
        image=pinned_live_image,
        flake_reruns=1,
    )
    assert result.outcome is CheckOutcome.PASS
    assert result.runs == (0,)


@requires_podman
def test_live_failing_command_fails_after_reruns(tmp_path, pinned_live_image):
    work_tree = tmp_path / "src"
    work_tree.mkdir()
    (work_tree / "README.txt").write_text("hello\n")

    result = run_check(
        "trivial-fail",
        "false",
        work_tree,
        image=pinned_live_image,
        flake_reruns=1,
    )
    assert result.outcome is CheckOutcome.FAIL
    assert len(result.runs) == 1 + 1
    assert all(code != 0 for code in result.runs)


@requires_podman
def test_live_checker_container_has_no_network_egress(tmp_path, pinned_live_image):
    # No-network proof: a connection attempt from inside the checker
    # container must fail to connect — there is no route out at all
    # (network="none"), stricter than a worker's proxied netns. Asserting
    # merely "nonzero exit" would be too weak (any unrelated wget usage
    # error is also nonzero) — assert the *specific* failure signature
    # busybox wget prints when there is no network path to even resolve
    # DNS (no route -> no resolver reachable -> "bad address"), not e.g.
    # "wget: not found" or a usage error.
    work_tree = tmp_path / "src"
    work_tree.mkdir()
    (work_tree / "README.txt").write_text("hello\n")

    result = run_check(
        "no-egress",
        # -q suppressed so the failure text lands in captured output.
        "wget -T 2 -O- http://example.com",
        work_tree,
        image=pinned_live_image,
        flake_reruns=0,
    )
    assert result.outcome is CheckOutcome.FAIL
    assert result.runs == (result.runs[0],)
    assert result.runs[0] != 0
    lowered = result.output.lower()
    network_failure_signatures = (
        "bad address",
        "could not resolve",
        "unreachable",
        "timed out",
        "network is unreachable",
        "name or service not known",
    )
    assert any(sig in lowered for sig in network_failure_signatures), (
        f"expected a network-failure signature in output, got: {result.output!r}"
    )


@requires_podman
def test_live_work_tree_immutability(tmp_path, pinned_live_image):
    # A check that writes into /work must never mutate the HOST source
    # work_tree — it is copied into a fresh temp dir per run, never
    # mounted directly.
    work_tree = tmp_path / "src"
    work_tree.mkdir()
    original = work_tree / "original.txt"
    original.write_text("do not touch\n")

    before_listing = sorted(p.name for p in work_tree.iterdir())
    before_content = original.read_text()

    results = run_checks(
        {"mutator": "echo mutated > /work/mutated.txt && echo done"},
        work_tree,
        image=pinned_live_image,
        flake_reruns=0,
    )

    assert results[0].outcome is CheckOutcome.PASS
    after_listing = sorted(p.name for p in work_tree.iterdir())
    after_content = original.read_text()

    assert after_listing == before_listing
    assert "mutated.txt" not in after_listing
    assert after_content == before_content
