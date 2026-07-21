"""Tests for `stigmergy.drivers.claude_code` and the `container.build_run_argv`
`env=` extension it depends on (bead .13 build spec §2/§3; SPEC §7 driver
interface, §9 failure classes, §10 AC7).

Deterministic, offline (injected `run_one`; no real podman/claude-code
except the one gated live smoke, §3). Fixture JSON strings model real
claude-code `--output-format json` shapes.

**Where the `env=` regression tests live (bead .13 build spec §0.2):** this
mirrors bead .11's own precedent — its `build_run_argv` regression tests for
`egress_socket` live in `tests/test_egress.py` (the file for the FEATURE
that motivated the addition), not scattered into `tests/test_container.py`.
The `env=` parameter exists because of THIS bead's credential-delivery need,
so its tests live here.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from stigmergy.container import ContainerError, ContainerProfile, build_run_argv
from stigmergy.drivers.claude_code import (
    Budgets,
    DispatchStatus,
    DriverError,
    ModelConfig,
    spawn,
)
from stigmergy.relay import Capability

# --------------------------------------------------------------------------
# shared fixtures / helpers
# --------------------------------------------------------------------------

PINNED_IMAGE = "localhost/stigmergy-worker@sha256:" + "a" * 64


def _profile(tmp_path, **overrides):
    """Mirrors test_container.py's/test_egress.py's `_profile` helper —
    duplicated here (not imported) since these tests own their own fixture
    surface, per the module docstring."""
    work = tmp_path / "profile-work"
    work.mkdir(parents=True, exist_ok=True)
    task = tmp_path / "profile-task"
    task.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        image=PINNED_IMAGE,
        work_clone=work,
        task_pack=task,
        scratch_size="64m",
        pids_limit=32,
        memory="256m",
        cpus="1",
        timeout_seconds=120,
    )
    kwargs.update(overrides)
    return ContainerProfile(**kwargs)


def _model_cfg(**overrides) -> ModelConfig:
    kwargs = dict(
        model="claude-sonnet-4-5",
        image=PINNED_IMAGE,
        relay_base_url="http://relay.local:9191",
        timeout_seconds=60,
    )
    kwargs.update(overrides)
    return ModelConfig(**kwargs)


def _capability(**overrides) -> Capability:
    kwargs = dict(
        token="cap-tok-abc123",  # nosec - test fixture, not a real credential
        dispatch_id="disp-1",
        max_output_tokens=100000,
        max_calls=50,
    )
    kwargs.update(overrides)
    return Capability(**kwargs)


def _budgets(**overrides) -> Budgets:
    kwargs = dict(output_tokens=1000, driver_turns=10)
    kwargs.update(overrides)
    return Budgets(**kwargs)


def _task_pack(tmp_path, name="task_pack", *, prompt_text="do the thing") -> Path:
    pack = tmp_path / name
    pack.mkdir(parents=True, exist_ok=True)
    (pack / "prompt.md").write_text(prompt_text, encoding="utf-8")
    return pack


def _task_pack_missing_prompt(tmp_path, name="task_pack_bad") -> Path:
    pack = tmp_path / name
    pack.mkdir(parents=True, exist_ok=True)
    return pack


def _git(repo, args, *, check=True):
    return subprocess.run(  # noqa: S603
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=check
    )


def _make_work_clone(tmp_path, name="work_clone", *, with_work_branch=True) -> Path:
    """A real (small, host-safe, under tmp_path) git repo mirroring the
    `.21`-prepared `work_clone` convention: `main` at a base commit, and —
    if requested — `refs/heads/work` checked out with one further commit
    (the branch the worker is expected to commit to, per bead .13 build
    spec §0.4)."""
    repo = tmp_path / name
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, ["init", "--quiet", "-b", "main"])
    _git(repo, ["config", "user.email", "test@example.com"])
    _git(repo, ["config", "user.name", "Test User"])
    (repo / "README.md").write_text("base\n")
    _git(repo, ["add", "README.md"])
    _git(repo, ["commit", "--quiet", "-m", "base"])
    if with_work_branch:
        _git(repo, ["checkout", "--quiet", "-b", "work"])
        (repo / "feature.txt").write_text("feature\n")
        _git(repo, ["add", "feature.txt"])
        _git(repo, ["commit", "--quiet", "-m", "work commit"])
    return repo


def _success_json(*, out_tokens=50, in_tokens=10, cache_creation=2, cache_read=5, cost=0.01) -> str:
    return json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "total_cost_usd": cost,
            "usage": {
                "input_tokens": in_tokens,
                "cache_creation_input_tokens": cache_creation,
                "cache_read_input_tokens": cache_read,
                "output_tokens": out_tokens,
            },
        }
    )


class CapturingRunOne:
    """Deterministic injected executor (mirrors checks.py's ScriptedRunOne
    pattern): records every call's `(argv, env, timeout)` and returns a
    scripted `subprocess.CompletedProcess`."""

    def __init__(self, stdout: str = "", *, returncode: int = 0, stderr: str = ""):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr
        self.calls: list[tuple[list[str], dict[str, str], int]] = []

    def __call__(self, argv, env, timeout):
        self.calls.append((argv, env, timeout))
        return subprocess.CompletedProcess(
            args=argv, returncode=self.returncode, stdout=self.stdout, stderr=self.stderr
        )


class RaisingRunOne:
    """Injected executor that always raises a scripted exception (proves
    the timeout/OSError launch-failure paths without any real subprocess)."""

    def __init__(self, exc: BaseException):
        self.exc = exc
        self.calls: list[tuple[list[str], dict[str, str], int]] = []

    def __call__(self, argv, env, timeout):
        self.calls.append((argv, env, timeout))
        raise self.exc


# ==========================================================================
# build_run_argv(env=...) extension (container.py, bead .13 build spec §0.2)
# — regression + additive correctness. Cases 1-5.
# ==========================================================================


def test_build_run_argv_adds_env_flags_when_given(tmp_path):
    profile = _profile(tmp_path)
    env = {"ANTHROPIC_API_KEY": "cap-tok", "ANTHROPIC_BASE_URL": "http://x"}
    argv = build_run_argv(profile, command=["true"], env=env)
    assert "--env=ANTHROPIC_API_KEY=cap-tok" in argv
    assert "--env=ANTHROPIC_BASE_URL=http://x" in argv
    image_index = argv.index(profile.image)
    assert argv.index("--env=ANTHROPIC_API_KEY=cap-tok") < image_index
    assert argv.index("--env=ANTHROPIC_BASE_URL=http://x") < image_index
    # sorted by key
    assert argv.index("--env=ANTHROPIC_API_KEY=cap-tok") < argv.index(
        "--env=ANTHROPIC_BASE_URL=http://x"
    )


def test_build_run_argv_unchanged_when_env_none(tmp_path):
    profile = _profile(tmp_path)
    with_none = build_run_argv(profile, command=["true"], env=None)
    without_param = build_run_argv(profile, command=["true"])
    assert with_none == without_param


def test_build_run_argv_env_rejects_bad_key(tmp_path):
    profile = _profile(tmp_path)
    with pytest.raises(ContainerError):
        build_run_argv(profile, command=["true"], env={"BAD=KEY": "v"})
    with pytest.raises(ContainerError):
        build_run_argv(profile, command=["true"], env={"": "v"})


def test_build_run_argv_env_and_egress_socket_together(tmp_path):
    profile = _profile(tmp_path)
    sock = tmp_path / "egress.sock"
    env = {"ANTHROPIC_API_KEY": "cap-tok", "ANTHROPIC_BASE_URL": "http://x"}
    argv = build_run_argv(profile, command=["true"], egress_socket=sock, env=env)
    volumes = [a for a in argv if a.startswith("--volume=")]
    assert len([v for v in volumes if "egress.sock" in v]) == 1
    assert f"--volume={sock}:/run/egress.sock:ro" in argv
    assert "--env=ANTHROPIC_API_KEY=cap-tok" in argv
    assert "--env=ANTHROPIC_BASE_URL=http://x" in argv


def test_build_run_argv_no_extra_volume_from_env(tmp_path):
    profile = _profile(tmp_path)
    env = {"ANTHROPIC_API_KEY": "cap-tok", "ANTHROPIC_BASE_URL": "http://x"}
    argv = build_run_argv(profile, command=["true"], egress_socket=None, env=env)
    volumes = [a for a in argv if a.startswith("--volume=")]
    assert len(volumes) == 2


# ==========================================================================
# bead .25: relay_socket threading + cred_env base-URL drop. Cases B4-B6.
# ==========================================================================


def test_model_cfg_relay_socket_field_defaults_none():
    cfg = _model_cfg()
    assert cfg.relay_socket is None
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.relay_socket = "/x"  # type: ignore[misc]


def test_spawn_with_relay_socket_mounts_it_and_drops_base_url(tmp_path):
    # bead .25: production relay path. The relay socket is mounted at the .63
    # in-cage path, and the worker gets ONLY the capability token — the .63
    # entrypoint owns ANTHROPIC_BASE_URL (loopback relay shim), so cred_env
    # must NOT carry a host base URL.
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    sock = tmp_path / "relay.sock"
    runner = CapturingRunOne(stdout=_success_json())
    cap = _capability()
    spawn(pack, work, _model_cfg(relay_socket=sock), cap, _budgets(), run_one=runner)
    argv, _env, _timeout = runner.calls[0]
    assert f"--volume={sock}:/run/relay.sock:ro" in argv
    assert f"--env=ANTHROPIC_API_KEY={cap.token}" in argv
    assert not any(a.startswith("--env=ANTHROPIC_BASE_URL=") for a in argv)


def test_spawn_without_relay_socket_is_backward_compatible(tmp_path):
    # Legacy/no-relay path: no relay volume; ANTHROPIC_BASE_URL still injected
    # from relay_base_url (existing .13 behavior preserved).
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    runner = CapturingRunOne(stdout=_success_json())
    cap = _capability()
    spawn(pack, work, _model_cfg(relay_base_url="http://relay.local:9191"),
          cap, _budgets(), run_one=runner)
    argv, _env, _timeout = runner.calls[0]
    assert not any(":/run/relay.sock:ro" in a for a in argv)
    assert f"--env=ANTHROPIC_API_KEY={cap.token}" in argv
    assert "--env=ANTHROPIC_BASE_URL=http://relay.local:9191" in argv


# ==========================================================================
# Task-pack / work-clone contract (bead .13 build spec §0.3/§0.4). Cases 6-7.
# ==========================================================================


def test_spawn_missing_prompt_file_raises_driver_error(tmp_path):
    pack = _task_pack_missing_prompt(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    runner = CapturingRunOne(stdout=_success_json())
    with pytest.raises(DriverError):
        spawn(pack, work, _model_cfg(), _capability(), _budgets(), run_one=runner)
    assert runner.calls == []


def test_spawn_reads_prompt_file_verbatim_into_dash_p(tmp_path):
    prompt_text = 'echo `whoami`; $(rm -rf /); "quoted"; done'
    pack = _task_pack(tmp_path, prompt_text=prompt_text)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    runner = CapturingRunOne(stdout=_success_json())
    spawn(pack, work, _model_cfg(), _capability(), _budgets(), run_one=runner)
    argv, _env, _timeout = runner.calls[0]
    p_index = argv.index("-p")
    assert argv[p_index + 1] == prompt_text


# ==========================================================================
# Status classification -- done. Cases 8-9.
# ==========================================================================


def test_spawn_success_subtype_returns_done(tmp_path):
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    runner = CapturingRunOne(stdout=_success_json())
    result = spawn(
        pack, work, _model_cfg(), _capability(), _budgets(output_tokens=1000), run_one=runner
    )
    assert result.status is DispatchStatus.DONE
    assert result.ceiling_trip is None
    assert result.usage == {"in": 12, "cached": 5, "out": 50, "reasoning": 0}
    assert result.reported_cost_usd == 0.01


def test_spawn_result_as_json_array_finds_terminal_result(tmp_path):
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    events = [
        {"type": "system", "subtype": "init"},
        {"type": "assistant", "message": {"content": "working..."}},
        json.loads(_success_json()),
    ]
    runner = CapturingRunOne(stdout=json.dumps(events))
    result = spawn(
        pack, work, _model_cfg(), _capability(), _budgets(output_tokens=1000), run_one=runner
    )
    assert result.status is DispatchStatus.DONE
    assert result.ceiling_trip is None
    assert result.usage == {"in": 12, "cached": 5, "out": 50, "reasoning": 0}
    assert result.reported_cost_usd == 0.01


# ==========================================================================
# Status classification -- failed (ordinary). Cases 10-14.
# ==========================================================================


def test_spawn_is_error_true_generic_subtype_returns_failed(tmp_path):
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    stdout = json.dumps({"type": "result", "is_error": True, "subtype": "some_other_value"})
    runner = CapturingRunOne(stdout=stdout)
    result = spawn(pack, work, _model_cfg(), _capability(), _budgets(), run_one=runner)
    assert result.status is DispatchStatus.FAILED
    assert result.ceiling_trip is None


def test_spawn_error_max_budget_usd_returns_failed_no_ceiling_trip(tmp_path):
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    stdout = json.dumps({"type": "result", "is_error": True, "subtype": "error_max_budget_usd"})
    runner = CapturingRunOne(stdout=stdout)
    result = spawn(pack, work, _model_cfg(), _capability(), _budgets(), run_one=runner)
    assert result.status is DispatchStatus.FAILED
    assert result.ceiling_trip is None


def test_spawn_error_max_structured_output_retries_returns_failed(tmp_path):
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    stdout = json.dumps(
        {"type": "result", "is_error": True, "subtype": "error_max_structured_output_retries"}
    )
    runner = CapturingRunOne(stdout=stdout)
    result = spawn(pack, work, _model_cfg(), _capability(), _budgets(), run_one=runner)
    assert result.status is DispatchStatus.FAILED
    assert result.ceiling_trip is None


def test_spawn_error_during_execution_no_infra_marker_returns_failed(tmp_path):
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    stdout = json.dumps(
        {
            "type": "result",
            "subtype": "error_during_execution",
            "is_error": True,
            "result": "TypeError: undefined is not a function",
        }
    )
    runner = CapturingRunOne(stdout=stdout)
    result = spawn(pack, work, _model_cfg(), _capability(), _budgets(), run_one=runner)
    assert result.status is DispatchStatus.FAILED
    assert result.ceiling_trip is None


def test_spawn_is_error_missing_defaults_to_failed(tmp_path):
    # Sharpest fail-closed edge case: no `is_error` key at all, subtype
    # superficially says "success" -- must NOT be treated as a clean
    # success (fail-closed default, bead .13 build spec §1.2).
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    stdout = json.dumps({"type": "result", "subtype": "success"})
    runner = CapturingRunOne(stdout=stdout)
    result = spawn(pack, work, _model_cfg(), _capability(), _budgets(), run_one=runner)
    assert result.status is DispatchStatus.FAILED
    assert result.ceiling_trip is None


# ==========================================================================
# Bead .64: API-error classification (subtype="success" + is_error=true +
# api_error_status), the shape REAL claude-code emits on an API/transport
# failure (observed live 2026-07-18). The pre-.64 classifier fell these through
# to FAILED; the unambiguous provider/infra statuses must be INFRA (SPEC D9).
# ==========================================================================


def _run_result(stdout_obj, tmp_path, budgets=None):
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    runner = CapturingRunOne(stdout=json.dumps(stdout_obj))
    return spawn(pack, work, _model_cfg(), _capability(), budgets or _budgets(), run_one=runner)


def test_spawn_api_error_status_500_returns_infra(tmp_path):
    # Discriminates the api_error_status path SPECIFICALLY: 500 is in
    # _INFRA_HTTP_STATUSES but "500" is NOT an _INFRA_MARKERS substring, and the
    # result text carries no marker -> INFRA can only come from the status path.
    result = _run_result(
        {"type": "result", "subtype": "success", "is_error": True,
         "api_error_status": 500, "result": "the upstream request did not complete",
         "terminal_reason": "api_error"},
        tmp_path,
    )
    assert result.status is DispatchStatus.INFRA


def test_spawn_api_error_status_408_returns_infra(tmp_path):
    # 408 likewise is a status-path-only signal (no "408" marker, marker-free text).
    result = _run_result(
        {"type": "result", "subtype": "success", "is_error": True,
         "api_error_status": 408, "result": "the request did not complete in time"},
        tmp_path,
    )
    assert result.status is DispatchStatus.INFRA


def test_spawn_api_error_status_403_returns_infra(tmp_path):
    # Observed-live denied-egress shape. SB ruled 2026-07-18 (bead .64): auth
    # statuses (401/403) are INFRA -- in the credential-relay model the worker
    # never holds the real key, so an auth failure is ours to fix at the infra
    # level, not a capability failure. This is the exact captured denied-egress
    # result object.
    result = _run_result(
        {"type": "result", "subtype": "success", "is_error": True,
         "api_error_status": 403,
         "result": "Failed to authenticate. API Error: 403 status code (no body)",
         "terminal_reason": "api_error"},
        tmp_path,
    )
    assert result.status is DispatchStatus.INFRA


def test_spawn_api_error_status_401_returns_infra(tmp_path):
    result = _run_result(
        {"type": "result", "subtype": "success", "is_error": True,
         "api_error_status": 401, "result": "unauthorized"},
        tmp_path,
    )
    assert result.status is DispatchStatus.INFRA


def test_spawn_api_error_status_400_returns_failed(tmp_path):
    # A client/request error (e.g. malformed request, context too long) is a
    # capability failure, NOT infra.
    result = _run_result(
        {"type": "result", "subtype": "success", "is_error": True,
         "api_error_status": 400, "result": "API Error: 400 bad_request"},
        tmp_path,
    )
    assert result.status is DispatchStatus.FAILED


def test_spawn_is_error_success_with_connection_marker_returns_infra(tmp_path):
    # A transport failure that carried NO status but a marker-bearing result
    # text (e.g. a dead egress surfaced as a connection error) -> INFRA.
    result = _run_result(
        {"type": "result", "subtype": "success", "is_error": True,
         "result": "request failed: Connection refused"},
        tmp_path,
    )
    assert result.status is DispatchStatus.INFRA


# ==========================================================================
# Status classification -- ceiling trips (SPEC §9). Cases 15-16.
# ==========================================================================


def test_spawn_output_tokens_ceiling_trip_overrides_success(tmp_path):
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    stdout = _success_json(out_tokens=500)
    runner = CapturingRunOne(stdout=stdout)
    result = spawn(
        pack, work, _model_cfg(), _capability(), _budgets(output_tokens=500), run_one=runner
    )
    assert result.status is DispatchStatus.FAILED
    assert result.ceiling_trip == "output_tokens"


def test_spawn_error_max_turns_returns_failed_with_driver_turns_ceiling_flag(tmp_path):
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    stdout = json.dumps(
        {
            "type": "result",
            "subtype": "error_max_turns",
            "is_error": False,  # the documented real-world quirk
            "num_turns": 3,
            "usage": {
                "input_tokens": 1,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 1,
            },
        }
    )
    runner = CapturingRunOne(stdout=stdout)
    result = spawn(pack, work, _model_cfg(), _capability(), _budgets(), run_one=runner)
    assert result.status is DispatchStatus.FAILED
    assert result.ceiling_trip == "driver_turns"


# ==========================================================================
# Status classification -- infra. Cases 17-19.
# ==========================================================================


def test_spawn_rate_limit_marker_returns_infra(tmp_path):
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    stdout = json.dumps(
        {
            "type": "result",
            "subtype": "error_during_execution",
            "is_error": True,
            "result": (
                'API Error (529 {"type":"error","error":{"type":"overloaded_error",'
                '"message":"Overloaded"}})'
            ),
        }
    )
    runner = CapturingRunOne(stdout=stdout)
    result = spawn(pack, work, _model_cfg(), _capability(), _budgets(), run_one=runner)
    assert result.status is DispatchStatus.INFRA
    assert result.ceiling_trip is None


def test_spawn_429_marker_returns_infra(tmp_path):
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    stdout = json.dumps(
        {
            "type": "result",
            "subtype": "error_during_execution",
            "is_error": True,
            "result": "API Error: 429 Too Many Requests",
        }
    )
    runner = CapturingRunOne(stdout=stdout)
    result = spawn(pack, work, _model_cfg(), _capability(), _budgets(), run_one=runner)
    assert result.status is DispatchStatus.INFRA


def test_spawn_unknown_subtype_never_becomes_infra(tmp_path):
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    stdout = json.dumps(
        {"type": "result", "subtype": "something_new_v99", "is_error": True}
    )
    runner = CapturingRunOne(stdout=stdout)
    result = spawn(pack, work, _model_cfg(), _capability(), _budgets(), run_one=runner)
    assert result.status is DispatchStatus.FAILED
    assert result.status is not DispatchStatus.INFRA


# ==========================================================================
# Status classification -- unparseable / launch failure / wedge. Cases 20-24.
# ==========================================================================


def test_spawn_unparseable_stdout_nonzero_exit_returns_failed(tmp_path):
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    runner = CapturingRunOne(stdout="Traceback (most recent call last):\n  boom\n", returncode=1)
    result = spawn(pack, work, _model_cfg(), _capability(), _budgets(), run_one=runner)
    assert result.status is DispatchStatus.FAILED


def test_spawn_exit_137_unparseable_returns_wedged(tmp_path):
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    runner = CapturingRunOne(stdout="", returncode=137)
    result = spawn(pack, work, _model_cfg(), _capability(), _budgets(), run_one=runner)
    assert result.status is DispatchStatus.WEDGED


def test_spawn_exit_69_cage_unavailable_returns_infra(tmp_path):
    # Bead .64: the worker-image entrypoint exits 69 on ANY fail-closed path
    # (egress socket absent / a shim never came up). That is a broken-cage INFRA
    # condition, not a capability FAILED — an EXPLICIT dead-cage -> INFRA signal
    # that does NOT depend on claude-code emitting an _INFRA_MARKERS substring.
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    runner = CapturingRunOne(stdout="", returncode=69)
    result = spawn(pack, work, _model_cfg(), _capability(), _budgets(), run_one=runner)
    assert result.status is DispatchStatus.INFRA
    assert "cage egress setup failed" in result.detail


def test_spawn_subprocess_timeout_returns_wedged(tmp_path):
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    runner = RaisingRunOne(subprocess.TimeoutExpired(cmd=["podman"], timeout=10))
    result = spawn(pack, work, _model_cfg(), _capability(), _budgets(), run_one=runner)
    assert result.status is DispatchStatus.WEDGED
    assert "backstop" in result.detail.lower()


def test_spawn_oserror_on_launch_returns_infra(tmp_path):
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    runner = RaisingRunOne(OSError("podman: command not found"))
    result = spawn(pack, work, _model_cfg(), _capability(), _budgets(), run_one=runner)
    assert result.status is DispatchStatus.INFRA


def test_spawn_unpinned_image_propagates_container_error(tmp_path):
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    runner = CapturingRunOne(stdout=_success_json())
    cfg = _model_cfg(image="localhost/stigmergy-worker:latest")
    with pytest.raises(ContainerError):
        spawn(pack, work, cfg, _capability(), _budgets(), run_one=runner)
    # never reached the executor -- the ContainerError comes from
    # build_run_argv, before any subprocess call.
    assert runner.calls == []


# ==========================================================================
# Credential vacancy (mirrors AC4). Cases 25-26.
# ==========================================================================


def test_spawn_worker_env_never_contains_real_key(tmp_path):
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    cap = _capability(token="cap-tok-xyz")  # nosec - test fixture
    cfg = _model_cfg(relay_base_url="http://relay.local:9999")
    runner = CapturingRunOne(stdout=_success_json())
    spawn(pack, work, cfg, cap, _budgets(), run_one=runner)

    argv, _env, _timeout = runner.calls[0]
    env_tokens: dict[str, str] = {}
    for token in argv:
        if token.startswith("--env="):
            key, _, value = token[len("--env=") :].partition("=")
            env_tokens[key] = value

    assert env_tokens == {
        "ANTHROPIC_API_KEY": cap.token,
        "ANTHROPIC_BASE_URL": cfg.relay_base_url,
    }


def test_spawn_argv_never_contains_capability_token_in_prompt_position(tmp_path):
    prompt_text = "innocuous prompt content, nothing capability-shaped here"
    pack = _task_pack(tmp_path, prompt_text=prompt_text)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    cap = _capability(token="super-secret-cap-token-999")  # nosec - test fixture
    runner = CapturingRunOne(stdout=_success_json())
    spawn(pack, work, _model_cfg(), cap, _budgets(), run_one=runner)

    argv, _env, _timeout = runner.calls[0]
    p_index = argv.index("-p")
    prompt_arg = argv[p_index + 1]
    assert prompt_arg == prompt_text
    assert cap.token not in prompt_arg


# ==========================================================================
# Bead .34 build spec: build_run_argv(dispatch_id=...) threaded through
# spawn(). Case 8 (frozen case list, Part B).
# ==========================================================================


def test_spawn_argv_contains_dispatch_id_label_from_capability(tmp_path):
    # spawn()'s build_run_argv(...) call gains dispatch_id=capability.
    # dispatch_id -- the captured argv contains --label=stigmergy.
    # dispatch_id=<capability.dispatch_id> with the SAME value as the
    # Capability passed into spawn().
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    cap = _capability(dispatch_id="disp-bead34-xyz")
    runner = CapturingRunOne(stdout=_success_json())
    spawn(pack, work, _model_cfg(), cap, _budgets(), run_one=runner)

    argv, _env, _timeout = runner.calls[0]
    assert f"--label=stigmergy.dispatch_id={cap.dispatch_id}" in argv


# ==========================================================================
# Bundle creation (bead .13 build spec §0.6, AC6 git-metadata isolation).
# Cases 27-31.
# ==========================================================================


def test_bundle_created_when_work_ref_exists(tmp_path):
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=True)
    cap = _capability(dispatch_id="disp-27")
    runner = CapturingRunOne(stdout=_success_json())
    bundle_dir = tmp_path / "bundles"
    result = spawn(pack, work, _model_cfg(), cap, _budgets(), run_one=runner, bundle_dir=bundle_dir)

    assert result.bundle_ref is not None
    assert Path(result.bundle_ref).is_file()
    # `git bundle verify` resolves the bundle's prerequisite commits against a
    # repository, so it must run INSIDE one -- verify against `work`, the clone
    # the bundle was created from. Passing cwd explicitly (rather than relying
    # on the ambient process cwd happening to sit in a git repo) keeps this
    # deterministic when the suite runs in a checker cage whose /work copy may
    # or may not carry .git.
    verify = subprocess.run(  # noqa: S603
        ["git", "bundle", "verify", result.bundle_ref],
        cwd=work,
        capture_output=True,
        text=True,
    )
    assert verify.returncode == 0, verify.stderr


def test_bundle_none_when_work_ref_absent(tmp_path):
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    cap = _capability(dispatch_id="disp-28")
    runner = CapturingRunOne(stdout=_success_json())
    result = spawn(pack, work, _model_cfg(), cap, _budgets(), run_one=runner)

    assert result.bundle_ref is None
    assert result.status is DispatchStatus.DONE


def test_bundle_created_regardless_of_failed_status(tmp_path):
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=True)
    cap = _capability(dispatch_id="disp-29")
    stdout = json.dumps({"type": "result", "is_error": True, "subtype": "some_other_value"})
    runner = CapturingRunOne(stdout=stdout)
    result = spawn(pack, work, _model_cfg(), cap, _budgets(), run_one=runner)

    assert result.status is DispatchStatus.FAILED
    assert result.bundle_ref is not None
    assert Path(result.bundle_ref).is_file()


def test_bundle_seeded_hook_never_fires(tmp_path):
    """AC6 adversarial case. `_create_work_bundle` only ever runs `rev-parse
    --verify` and `bundle create` against `work_clone` — plain git facts:
    neither operation invokes repository hooks at all, with or without
    `-c core.hooksPath=/dev/null` (verified experimentally this session;
    `git bundle create`/`rev-parse` are hook-free operations by design,
    unlike `merge`/`commit`). This test still seeds a hook and asserts the
    sentinel never fires (belt-and-suspenders regression guard matching
    the spec's literal case), and separately proves the seeded hook is
    genuinely live (would fire on a real commit) so the seeding mechanism
    itself isn't silently broken -- see the report's "concerns" section for
    why this can't be a true proof of the `-c core.hooksPath=/dev/null`
    override's necessity for *this* function specifically.
    """
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=True)
    sentinel = tmp_path / "SENTINEL"
    hooks_dir = work / ".git" / "hooks"
    post_commit_hook = hooks_dir / "post-commit"
    post_commit_hook.write_text(f"#!/bin/sh\necho FIRED > {sentinel}\n")
    post_commit_hook.chmod(0o755)

    cap = _capability(dispatch_id="disp-30")
    runner = CapturingRunOne(stdout=_success_json())
    result = spawn(pack, work, _model_cfg(), cap, _budgets(), run_one=runner)

    assert not sentinel.exists(), "seeded hook fired during _create_work_bundle"
    assert result.bundle_ref is not None

    # --- CONTROL: prove the hook is genuinely live and WOULD fire --------
    control_sentinel = tmp_path / "CONTROL_SENTINEL"
    post_commit_hook.write_text(f"#!/bin/sh\necho FIRED > {control_sentinel}\n")
    post_commit_hook.chmod(0o755)
    _git(work, ["commit", "--allow-empty", "--quiet", "-m", "control commit"])
    assert control_sentinel.exists(), (
        "control case failed: the seeded post-commit hook did not fire even on a "
        "real commit -- this would make the main assertion above vacuous"
    )


def test_bundle_dir_not_cleaned_up_by_spawn(tmp_path):
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=True)
    cap = _capability(dispatch_id="disp-31")
    runner = CapturingRunOne(stdout=_success_json())
    bundle_dir = tmp_path / "bundles-31"
    result = spawn(pack, work, _model_cfg(), cap, _budgets(), run_one=runner, bundle_dir=bundle_dir)

    assert result.bundle_ref is not None
    assert Path(result.bundle_ref).exists(), "spawn() must not clean up its own bundle tempdir"


# ==========================================================================
# Live E2E smoke (bead .25 AC13 + AC4) -- gated, skipped by default. Drives
# the FULL production credential path: a real claude-code haiku dispatch in
# the .63 worker image, reaching Anthropic THROUGH the wired credential relay
# (capability token as ANTHROPIC_API_KEY; real key host-side, injected by the
# relay) with the per-dispatch egress proxy also live. Proves credential-swap
# (AC13 DONE), key-absence + token-death (AC4). RUNS on diodati with the gate
# set; skips cleanly where any prerequisite is absent.
# ==========================================================================

_LIVE_WORKER_IMAGE = "localhost/stigmergy-worker:latest"


def _live_worker_image_id() -> str | None:
    """Bare sha256 image id of the built .63 worker image, or None if absent
    (so the live test skips cleanly where the image was never built)."""
    if shutil.which("podman") is None:
        return None
    uid = os.getuid()
    env = {**os.environ, "XDG_RUNTIME_DIR": f"/run/user/{uid}",
           "DBUS_SESSION_BUS_ADDRESS": f"unix:path=/run/user/{uid}/bus"}
    r = subprocess.run(  # noqa: S603
        ["podman", "inspect", "--format", "{{.Id}}", _LIVE_WORKER_IMAGE],
        env=env, capture_output=True, text=True, check=False,
    )
    raw = r.stdout.strip()
    if r.returncode != 0 or not raw:
        return None
    return raw if raw.startswith("sha256:") else f"sha256:{raw}"


@pytest.mark.skipif(
    not os.environ.get("STIGMERGY_LIVE_SMOKE"),
    reason="live smoke gated behind STIGMERGY_LIVE_SMOKE=1",
)
@pytest.mark.skipif(
    shutil.which("podman") is None or shutil.which("op") is None
    or _live_worker_image_id() is None,
    reason="requires podman + op + the built localhost/stigmergy-worker:latest image",
)
def test_live_smoke_one_trivial_dispatch(tmp_path):
    """bead .25 AC13 + AC4. One real haiku dispatch through the FULL wired
    path: capability token in the cage, real key host-side in the relay, the
    relay swaps it and forwards to api.anthropic.com. Asserts DONE (swap
    works), the real key is absent from every worker-visible surface (argv/
    transcript/relay-log), and the capability token is dead after revoke."""
    import stigmergy.relay_transport as _rt
    from stigmergy.egress import setup_dispatch_egress
    from stigmergy.egress_proxy import EgressPolicy
    from stigmergy.keyprovider import make_op_key_provider
    from stigmergy.relay import CapabilityDenied, CapabilityStore, CredentialRelay

    # The relay's urllib forwarder runs in THIS process -- don't let an
    # inherited proxy env var reroute the upstream call.
    for _v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(_v, None)

    key_provider = make_op_key_provider(
        "op://shelly/API Credential - stigmergy rig 00/credential"
    )
    real_key = key_provider()
    assert real_key and real_key.startswith(("sk-", "sk-ant"))

    image = _live_worker_image_id()
    did = "live-smoke-25"
    store = CapabilityStore()
    cap = store.mint(did, max_output_tokens=200000, max_calls=100)
    assert real_key not in cap.token  # the worker's ANTHROPIC_API_KEY is NOT the real key

    relay = CredentialRelay(
        store=store,
        key_provider=key_provider,
        forwarder=lambda req: (_ for _ in ()).throw(RuntimeError("sync unused")),
        upstream_headers_pinned={"anthropic-version": "2023-06-01"},
        upstream_header_allowlist=frozenset({"content-type", "accept", "anthropic-beta"}),
    )
    forwarder = _rt.make_urllib_forwarder(base_url="https://api.anthropic.com")

    egress_rt = tmp_path / "egress_rt"
    egress_rt.mkdir()
    relay_rt = tmp_path / "relay_rt"
    relay_rt.mkdir()
    # deny-all egress: inference flows via the loopback relay (NO_PROXY-exempt),
    # so the egress proxy governs nothing legitimate here -- but its socket must
    # exist (the .63 entrypoint fail-closes without it).
    policy = EgressPolicy(allowed_hosts=frozenset(), allowed_ports=frozenset({443}))
    egress_handle = setup_dispatch_egress(did, policy, egress_rt)
    relay_handle = _rt.start_relay(did, relay_rt, relay, forwarder=forwarder,
                                   log_path=relay_rt / "relay.jsonl")
    try:
        pack = _task_pack(tmp_path, prompt_text="Reply with exactly: OK")
        work = _make_work_clone(tmp_path, with_work_branch=False)
        cfg = _model_cfg(
            model="claude-haiku-4-5-20251001",
            image=image,
            egress_socket=egress_handle.socket_path,
            relay_socket=relay_handle.socket_path,
            timeout_seconds=180,
        )
        result = spawn(pack, work, cfg, cap, _budgets(output_tokens=200000, driver_turns=5))

        # AC13: the credential swap worked end-to-end.
        assert result.status is DispatchStatus.DONE, (
            f"expected DONE, got {result.status}: {result.detail!r}"
        )

        # AC4: the real key is absent from every worker-visible surface.
        relay_log = (relay_handle.log_path.read_text()
                     if relay_handle.log_path.exists() else "")
        assert real_key not in result.transcript
        assert real_key not in relay_log
        # the relay actually forwarded at least one call (proof it was used)
        assert '"decision": "allow"' in relay_log
    finally:
        # AC4: token death -- revoke, then a replay must be denied.
        store.revoke(did)
        with pytest.raises(CapabilityDenied):
            store.authorize(cap.token)
        relay_handle.stop()
        egress_handle.teardown()
        uid = os.getuid()
        penv = {**os.environ, "XDG_RUNTIME_DIR": f"/run/user/{uid}",
                "DBUS_SESSION_BUS_ADDRESS": f"unix:path=/run/user/{uid}/bus"}
        ids = subprocess.run(  # noqa: S603
            ["podman", "ps", "-aq", "--filter", f"label=stigmergy.dispatch_id={did}"],
            env=penv, capture_output=True, text=True, check=False,
        ).stdout.split()
        if ids:
            subprocess.run(["podman", "rm", "-f", *ids], env=penv,  # noqa: S603
                           capture_output=True, text=True, check=False)
