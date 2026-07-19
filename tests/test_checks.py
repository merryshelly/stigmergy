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
    DEFAULT_CHECK_RESOURCES,
    CheckOutcome,
    CheckResources,
    CheckResult,
    run_check,
    run_checks,
)
from stigmergy.container import (
    ContainerProfile,
    build_image,
    build_run_argv,
    worker_env,
)

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
        self.resources_seen: list[object] = []

    def __call__(self, command, work_tree, *, image, resources=None):
        self.calls.append((command, work_tree, image))
        self.resources_seen.append(resources)
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

    def fake_default_run_one(command, work_tree, *, image, resources=None):
        return runners[command](command, work_tree, image=image, resources=resources)

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


# --------------------------------------------------------------------------
# Bead .87: checks run against the egress-gated WORKER image, whose ENTRYPOINT
# is the fail-closed egress gatekeeper (exit 69 with no /run/egress.sock). A
# checker is network=none with no egress mount, so without the entrypoint
# bypass NO check could ever run. These live tests use a hermetic image that
# reproduces that gatekeeper — the plain LIVE_IMAGE (no gatekeeper) is exactly
# why this class of bug went undetected.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gatekeeper_image(tmp_path_factory):
    """Build a hermetic image whose ENTRYPOINT mimics the worker image's
    fail-closed egress gatekeeper: exit 69 unless /run/egress.sock is a
    socket, else exec "$@". Digest-pinned base resolved programmatically."""
    if PODMAN is None:
        pytest.skip("podman not installed")
    subprocess.run(
        ["podman", "pull", LIVE_IMAGE],
        env=worker_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    base = subprocess.run(
        ["podman", "inspect", "--format", "{{index .RepoDigests 0}}", LIVE_IMAGE],
        env=worker_env(),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert "@sha256:" in base
    cf_dir = tmp_path_factory.mktemp("gatekeeper-image")
    (cf_dir / "gate.sh").write_text(
        "#!/bin/sh\n"
        '[ -S /run/egress.sock ] || { echo "gatekeeper: egress socket absent" >&2; exit 69; }\n'
        'exec "$@"\n'
    )
    (cf_dir / "Containerfile").write_text(
        f"FROM {base}\n"
        "COPY gate.sh /gate.sh\n"
        "RUN chmod +x /gate.sh\n"
        'ENTRYPOINT ["/gate.sh"]\n'
    )
    return build_image(cf_dir, "localhost/stigmergy-checktest:gatekeeper")


@requires_podman
def test_live_egress_gatekeeper_fires_without_bypass(tmp_path, gatekeeper_image):
    # The bug .87 fixes, kept documented: WITHOUT the entrypoint bypass, a
    # network=none checker (no egress socket) hits the gatekeeper entrypoint
    # which fail-closes at exit 69 before the check command ever runs.
    work = tmp_path / "work"
    work.mkdir()
    (work / "f").write_text("SMOKE-OK\n")
    task = tmp_path / "task"
    task.mkdir()
    profile = ContainerProfile(
        image=gatekeeper_image,
        work_clone=work,
        task_pack=task,
        scratch_size="16m",
        pids_limit=16,
        memory="128m",
        cpus="1",
        timeout_seconds=30,
        network="none",
    )
    argv = build_run_argv(profile, command=["sh", "-c", "grep -q SMOKE /work/f"])
    result = subprocess.run(
        argv, env=worker_env(), capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 69


@requires_podman
def test_live_check_bypasses_egress_gatekeeper(tmp_path, gatekeeper_image):
    # The fix: run_check bypasses the gatekeeper -> the real check command
    # runs to a real exit code. Matching content -> PASS, runs == (0,).
    work = tmp_path / "src"
    work.mkdir()
    (work / "DOGFOOD-SMOKE.md").write_text("SMOKE-OK content\n")
    result = run_check(
        "smoke",
        "grep -q '^SMOKE-OK' /work/DOGFOOD-SMOKE.md",
        work,
        image=gatekeeper_image,
        flake_reruns=0,
    )
    assert result.outcome is CheckOutcome.PASS
    assert result.runs == (0,)


@requires_podman
def test_live_check_real_fail_not_infra_against_gatekeeper(tmp_path, gatekeeper_image):
    # Non-matching content -> the grep actually ran and returned 1 (a real
    # FAIL), NOT the gatekeeper's 69 and NOT ERROR — proving the check command
    # executed, not the entrypoint.
    work = tmp_path / "src"
    work.mkdir()
    (work / "DOGFOOD-SMOKE.md").write_text("nope\n")
    result = run_check(
        "smoke",
        "grep -q '^SMOKE-OK' /work/DOGFOOD-SMOKE.md",
        work,
        image=gatekeeper_image,
        flake_reruns=0,
    )
    assert result.outcome is CheckOutcome.FAIL
    assert result.runs == (1,)


# --------------------------------------------------------------------------
# .91: charter-configurable checker resource bounds
# --------------------------------------------------------------------------
# `resources: CheckResources` threads through run_checks -> run_check ->
# _default_run_one -> ContainerProfile -> build_run_argv/podman. When unset
# it is DEFAULT_CHECK_RESOURCES (the legacy 60s/256m/1cpu/64m/64pids), so a
# resource-free caller behaves exactly as pre-.91.


def test_default_check_resources_pin_legacy_values():
    # Backward-compat pin: the historical hardcoded bounds are the defaults.
    assert DEFAULT_CHECK_RESOURCES.timeout_seconds == 60
    assert DEFAULT_CHECK_RESOURCES.memory == "256m"
    assert DEFAULT_CHECK_RESOURCES.cpus == "1"
    assert DEFAULT_CHECK_RESOURCES.scratch_size == "64m"
    assert DEFAULT_CHECK_RESOURCES.pids_limit == 64


def test_run_check_passes_resources_to_executor(tmp_path):
    runner = ScriptedRunOne([0])
    res = CheckResources(
        timeout_seconds=1800, memory="4g", cpus="4", scratch_size="2g", pids_limit=512
    )
    run_check(
        "pytest",
        "pytest -x -q",
        tmp_path,
        image="unused",
        flake_reruns=0,
        resources=res,
        run_one=runner,
    )
    assert runner.resources_seen == [res]


def test_run_check_defaults_resources_when_unset(tmp_path):
    runner = ScriptedRunOne([0])
    run_check("lint", "ruff check .", tmp_path, image="unused", flake_reruns=0, run_one=runner)
    assert runner.resources_seen == [DEFAULT_CHECK_RESOURCES]


def test_run_checks_maps_per_check_resources(tmp_path, monkeypatch):
    import stigmergy.checks as checks_module

    seen = {}

    def fake_default_run_one(command, work_tree, *, image, resources=None):
        seen[command] = resources
        return 0, ""

    monkeypatch.setattr(checks_module, "_default_run_one", fake_default_run_one)
    big = CheckResources(
        timeout_seconds=1800, memory="4g", cpus="4", scratch_size="2g", pids_limit=512
    )
    run_checks(
        {"pytest": "pytest-cmd", "lint": "lint-cmd"},
        tmp_path,
        image="unused",
        flake_reruns=0,
        resources={"pytest": big},
    )
    assert seen["pytest-cmd"] == big
    # a check absent from the resources map falls back to the default.
    assert seen["lint-cmd"] == DEFAULT_CHECK_RESOURCES


def test_run_checks_defaults_all_when_no_resources_map(tmp_path, monkeypatch):
    import stigmergy.checks as checks_module

    seen = {}

    def fake_default_run_one(command, work_tree, *, image, resources=None):
        seen[command] = resources
        return 0, ""

    monkeypatch.setattr(checks_module, "_default_run_one", fake_default_run_one)
    run_checks({"a": "cmd-a", "b": "cmd-b"}, tmp_path, image="unused", flake_reruns=0)
    assert seen["cmd-a"] == DEFAULT_CHECK_RESOURCES
    assert seen["cmd-b"] == DEFAULT_CHECK_RESOURCES


def test_default_run_one_builds_profile_from_resources(tmp_path, monkeypatch):
    # The resolved bounds must reach the ContainerProfile (hence podman), and
    # the subprocess backstop must derive from the configured timeout — while
    # the network="none" checker invariant is preserved.
    import stigmergy.checks as checks_module

    captured = {}

    def fake_build_run_argv(profile, *, command, entrypoint_override=None, env=None):
        captured["profile"] = profile
        return ["true"]

    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(argv, **kwargs):
        captured["subprocess_timeout"] = kwargs.get("timeout")
        return _Result()

    monkeypatch.setattr(checks_module, "build_run_argv", fake_build_run_argv)
    monkeypatch.setattr(checks_module.subprocess, "run", fake_run)

    work = tmp_path / "src"
    work.mkdir()
    (work / "f").write_text("x")
    res = CheckResources(
        timeout_seconds=1800, memory="4g", cpus="4", scratch_size="2g", pids_limit=512
    )
    checks_module._default_run_one("grep x f", work, image="img", resources=res)

    p = captured["profile"]
    assert p.timeout_seconds == 1800
    assert p.memory == "4g"
    assert p.cpus == "4"
    assert p.scratch_size == "2g"
    assert p.pids_limit == 512
    assert p.network == "none"  # safety invariant preserved regardless of bounds
    assert captured["subprocess_timeout"] == 1800 + 30


@requires_podman
def test_live_check_honors_configured_timeout(tmp_path, pinned_live_image):
    # Fidelity: the configured timeout actually reaches podman --timeout. A 2s
    # bound must kill a 30s sleep long before it completes.
    work = tmp_path / "src"
    work.mkdir()
    (work / "f").write_text("x")
    res = CheckResources(timeout_seconds=2)
    result = run_check(
        "slow",
        "sleep 30",
        work,
        image=pinned_live_image,
        flake_reruns=0,
        resources=res,
    )
    assert result.outcome in (CheckOutcome.FAIL, CheckOutcome.ERROR)
    assert result.wall_time_seconds < 25
