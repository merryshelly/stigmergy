"""Tests for `stigmergy.drivers.openalph_exec` (bead .149 build spec §4.1/§4.4).

The `openalph exec` worker driver: same `spawn()` signature, `DispatchResult`
type, and process-level failure parity as `claude_code.spawn` (injected
`run_one` + `bundle_dir`), but a different in-cage command (the OA `exec`
subcommand), a different credential env (OpenAI-wire shim: `OPENAI_API_KEY`
capability token + `OPENAI_BASE_URL` loopback shim root), and a different
result-JSON classification table (spec §4.1, closed conservative
vocabulary — unknown statuses never resolve to INFRA).

Deterministic, offline: every executor is an injected `run_one` script
(mirrors `tests/test_driver.py`'s `CapturingRunOne`/`RaisingRunOne`
pattern); no real podman/openalph/relay. The single in-cage shim constant
(`RELAY_PORT = 18081`, `worker_image/shim.py`) is asserted as a literal
constant against the shipped shim module. Bead .162: the in-cage tool
inventory gains `file_ticket` (one filing channel fleet-wide) and the cage
env gains the two non-credential filing vars (`FILE_TICKET_TRANSPORT` /
`FILE_TICKET_MAX_FILINGS`) on the same channel as the credential pair.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import stigmergy.drivers.claude_code as claude_code_module
from stigmergy.drivers import openalph_exec
from stigmergy.drivers.claude_code import Budgets, DispatchStatus, DriverError
from stigmergy.drivers.openalph_exec import ModelConfig, spawn
from stigmergy.relay import Capability
from stigmergy.worker_image import shim

# --------------------------------------------------------------------------
# shared fixtures / helpers (mirror test_driver.py's own, duplicated here —
# this module owns its own fixture surface)
# --------------------------------------------------------------------------

PINNED_IMAGE = "localhost/stigmergy-worker@sha256:" + "a" * 64

WORKER_TOOLS = "shell,file_read,file_write,file_edit,file_patch,glob,grep,file_ticket"


def _model_cfg(**overrides) -> ModelConfig:
    kwargs = dict(
        model="qwen38",
        worker_model="blackwell/qwen38-27b-fp8",
        effort="medium",
        image=PINNED_IMAGE,
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


def _git(repo, args, *, check=True):
    return subprocess.run(  # noqa: S603
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=check
    )


def _make_work_clone(tmp_path, name="work_clone", *, with_work_branch=True) -> Path:
    """A real (small, host-safe, under tmp_path) git repo mirroring the
    `.21`-prepared `work_clone` convention: `main` at a base commit, and —
    if requested — `refs/heads/work` with one further commit."""
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


def _exec_json(
    *,
    status="done",
    content="finished",
    usage=None,
    stop_reason=None,
    ceiling_trip=None,
    deny_reason=None,
    detail="",
) -> str:
    """One `openalph exec` stdout JSON line (spec §3.9 shape)."""
    return json.dumps(
        {
            "status": status,
            "content": content,
            "usage": {"in": 10, "cached": 2, "out": 50, "reasoning": 3}
            if usage is None
            else usage,
            "stop_reason": stop_reason,
            "ceiling_trip": ceiling_trip,
            "deny_reason": deny_reason,
            "detail": detail,
        }
    )


class CapturingRunOne:
    """Deterministic injected executor (mirrors test_driver.py's
    CapturingRunOne): records every call's `(argv, env, timeout)` and
    returns a scripted `subprocess.CompletedProcess`."""

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


def _argv_tail(argv: list[str]) -> list[str]:
    """The in-cage command: everything after the image ref."""
    return argv[argv.index(PINNED_IMAGE) + 1 :]


# ==========================================================================
# ModelConfig shape (spec §4.1: mirror claude_code's container-profile fields)
# ==========================================================================


def test_model_config_defaults_match_claude_code_container_profile():
    # spec §4.1: mirror claude_code.ModelConfig's container-profile fields —
    # image REQUIRED (like claude_code's), sockets default None, the
    # scratch/pids/memory/cpus/timeout defaults identical.
    with pytest.raises(TypeError):
        ModelConfig(model="qwen38", worker_model="blackwell/qwen38-27b-fp8", effort="medium")
    cfg = ModelConfig(
        model="qwen38", worker_model="blackwell/qwen38-27b-fp8", effort="medium", image=PINNED_IMAGE
    )
    assert cfg.image == PINNED_IMAGE
    assert cfg.relay_socket is None
    assert cfg.egress_socket is None
    assert cfg.scratch_size == "1g"
    assert cfg.pids_limit == 1024
    assert cfg.memory == "4g"
    assert cfg.cpus == "4"
    assert cfg.timeout_seconds == 3600


def test_model_config_is_frozen():
    cfg = ModelConfig(model="m", worker_model="w", effort="medium", image=PINNED_IMAGE)
    with pytest.raises(AttributeError):
        cfg.effort = "low"  # type: ignore[misc]


# ==========================================================================
# cred env (spec §4.1: the ONE path a capability credential takes into the
# worker env — the token, and only the token, plus the shim base URL)
# ==========================================================================


def test_credential_env_is_token_and_shim_base_url_only():
    cap = _capability(token="cap-tok-XYZ")
    env = openalph_exec._credential_env(cap)
    assert env == {
        "OPENAI_API_KEY": "cap-tok-XYZ",
        "OPENAI_BASE_URL": "http://127.0.0.1:18081",
    }


def test_credential_env_base_url_matches_shipped_shim_constant():
    # The constant must track worker_image/shim.py's RELAY_PORT — a drift
    # here points the worker's OpenAI client at a dead port.
    expected = f"http://127.0.0.1:{shim.RELAY_PORT}"
    assert openalph_exec._CRED_BASE_URL == expected
    assert openalph_exec._credential_env(_capability())["OPENAI_BASE_URL"] == expected


def test_credential_env_never_carries_a_non_token_key_value():
    # AC4 parity: the env's ONLY secret value is the capability token itself.
    cap = _capability(token="cap-tok-SECRET-123")
    env = openalph_exec._credential_env(cap)
    assert set(env) == {"OPENAI_API_KEY", "OPENAI_BASE_URL"}
    non_token_values = [v for k, v in env.items() if k != "OPENAI_API_KEY"]
    assert cap.token not in non_token_values
    assert len(non_token_values) == 1


# ==========================================================================
# argv shape (spec §1 in-cage command + §4.1 workdir)
# ==========================================================================


def test_spawn_argv_in_cage_command_shape(tmp_path):
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    runner = CapturingRunOne(stdout=_exec_json())
    spawn(pack, work, _model_cfg(), _capability(), _budgets(), run_one=runner)
    argv, _env, _timeout = runner.calls[0]
    command = _argv_tail(argv)
    assert command == [
        "openalph",
        "exec",
        "--agent",
        "stigmergy-worker",
        "--task-file",
        "/task/prompt.md",
        "--model",
        "blackwell/qwen38-27b-fp8",
        "--effort",
        "medium",
        "--max-turns",
        "10",
        "--tools",
        WORKER_TOOLS,
    ]


def test_spawn_argv_workdir_flag_present(tmp_path):
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    runner = CapturingRunOne(stdout=_exec_json())
    spawn(pack, work, _model_cfg(), _capability(), _budgets(), run_one=runner)
    argv, _env, _timeout = runner.calls[0]
    assert "--workdir=/work" in argv
    # spec §4.2: frozen position — right after --network (the dispatch label
    # owns the first after-network slot when dispatch_id is set), right
    # before the first --volume.
    network_index = next(i for i, a in enumerate(argv) if a.startswith("--network="))
    from stigmergy.container import DISPATCH_ID_LABEL_KEY

    assert argv[network_index + 1] == f"--label={DISPATCH_ID_LABEL_KEY}=disp-1"
    assert argv[network_index + 2] == "--workdir=/work"
    assert argv[network_index + 3].startswith("--volume=")


def test_spawn_argv_env_flags_carry_credential(tmp_path):
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    cap = _capability(token="cap-tok-ENV")
    runner = CapturingRunOne(stdout=_exec_json())
    spawn(pack, work, _model_cfg(), cap, _budgets(), run_one=runner)
    argv, _env, _timeout = runner.calls[0]
    assert "--env=OPENAI_API_KEY=cap-tok-ENV" in argv
    assert "--env=OPENAI_BASE_URL=http://127.0.0.1:18081" in argv
    image_index = argv.index(PINNED_IMAGE)
    assert argv.index("--env=OPENAI_API_KEY=cap-tok-ENV") < image_index


def test_spawn_tools_argv_contains_file_ticket(tmp_path):
    # bead .162: the worker files discovered work through the SAME
    # file_ticket builtin as the stations — appended to the v0 tool
    # inventory (order preserved; the existing seven tools untouched).
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    runner = CapturingRunOne(stdout=_exec_json())
    spawn(pack, work, _model_cfg(), _capability(), _budgets(), run_one=runner)
    argv, _env, _timeout = runner.calls[0]
    command = _argv_tail(argv)
    tools = command[command.index("--tools") + 1]
    assert tools == WORKER_TOOLS
    assert "file_ticket" in tools.split(",")
    assert tools == "shell,file_read,file_write,file_edit,file_patch,glob,grep,file_ticket"


def test_spawn_cage_env_carries_file_ticket_transport_and_cap(tmp_path):
    # bead .162: the file_ticket builtin's file-transport sink path + the
    # per-run count cap ride the SAME cage env channel as the credential
    # pair (one --env= token per entry, sorted, before the image ref).
    # The byte cap is ABSENT here: this Budgets carries no
    # max_filing_bytes (default None), and an absent env var means the
    # tool applies no call-time size check (OA tool default) — the
    # harvest-side size-cap stays the backstop.
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    runner = CapturingRunOne(stdout=_exec_json())
    spawn(pack, work, _model_cfg(), _capability(), _budgets(), run_one=runner)
    argv, _env, _timeout = runner.calls[0]
    transport_flag = "--env=FILE_TICKET_TRANSPORT=/work/.stigmergy/filed-tickets.json"
    cap_flag = "--env=FILE_TICKET_MAX_FILINGS=8"
    assert transport_flag in argv
    assert cap_flag in argv
    assert not any(a.startswith("--env=FILE_TICKET_MAX_BYTES=") for a in argv)
    image_index = argv.index(PINNED_IMAGE)
    assert argv.index(transport_flag) < image_index
    assert argv.index(cap_flag) < image_index
    # the cap here is the getattr DEFAULT: this test's Budgets stub does not
    # carry max_filings (older stubs stay valid). The charter-threaded path
    # is pinned in test_spawn_cage_env_filing_cap_from_budgets below.
    # the transport must track the harvest's /work-mount path (filing.py
    # owns the transport literal — a drift would orphan every worker filing).
    from stigmergy import filing

    assert openalph_exec._FILE_TICKET_TRANSPORT == "/work/" + filing.FILED_TICKETS_REL


def test_spawn_cage_env_file_ticket_max_bytes_from_budgets(tmp_path):
    # bead .162 audit fix: the per-filing byte cap rides the SAME cage env
    # channel (FILE_TICKET_MAX_BYTES from budgets.max_filing_bytes <-
    # charter [loop.dispatch_limits].filed_ticket_bytes) so the tool
    # size-checks at call time, matching the harvest-side size-cap. Present
    # with the exact value when Budgets carries max_filing_bytes; ABSENT
    # (never a fabricated default) when it is None — the default stub case
    # is pinned in test_spawn_cage_env_carries_file_ticket_transport_and_cap.
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=False)

    # present + correct value when the Budgets carries the cap.
    runner = CapturingRunOne(stdout=_exec_json())
    spawn(
        pack,
        work,
        _model_cfg(),
        _capability(),
        _budgets(max_filings=5, max_filing_bytes=20480),
        run_one=runner,
    )
    argv, _env, _timeout = runner.calls[0]
    bytes_flag = "--env=FILE_TICKET_MAX_BYTES=20480"
    assert bytes_flag in argv
    assert "--env=FILE_TICKET_MAX_FILINGS=5" in argv
    image_index = argv.index(PINNED_IMAGE)
    assert argv.index(bytes_flag) < image_index

    # explicit None -> ABSENT env var (unchecked, OA tool default).
    runner2 = CapturingRunOne(stdout=_exec_json())
    spawn(
        pack,
        work,
        _model_cfg(),
        _capability(),
        _budgets(max_filing_bytes=None),
        run_one=runner2,
    )
    argv2, _env2, _timeout2 = runner2.calls[0]
    assert not any(a.startswith("--env=FILE_TICKET_MAX_BYTES=") for a in argv2)


def test_spawn_cage_env_filing_cap_from_budgets(tmp_path):
    # bead .162: the charter filing cap rides budgets.max_filings (threaded
    # through prepare_dispatch from [loop.dispatch_limits].filed_tickets)
    # so the tool-side cap EQUALS the harvest-side count-cap — a tool cap
    # above the harvest cap would let a worker file N good proposals the
    # harvest then whole-batch-rejects.
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    runner = CapturingRunOne(stdout=_exec_json())
    spawn(
        pack,
        work,
        _model_cfg(),
        _capability(),
        _budgets(max_filings=5),
        run_one=runner,
    )
    argv, _env, _timeout = runner.calls[0]
    assert "--env=FILE_TICKET_MAX_FILINGS=5" in argv


def test_spawn_argv_relay_and_egress_sockets_mounted(tmp_path):
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    relay_sock = tmp_path / "relay.sock"
    egress_sock = tmp_path / "egress.sock"
    runner = CapturingRunOne(stdout=_exec_json())
    spawn(
        pack,
        work,
        _model_cfg(relay_socket=relay_sock, egress_socket=egress_sock),
        _capability(),
        _budgets(),
        run_one=runner,
    )
    argv, _env, _timeout = runner.calls[0]
    assert f"--volume={relay_sock}:/run/relay.sock:ro" in argv
    assert f"--volume={egress_sock}:/run/egress.sock:ro" in argv


def test_spawn_argv_dispatch_id_named(tmp_path):
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    runner = CapturingRunOne(stdout=_exec_json())
    spawn(pack, work, _model_cfg(), _capability(dispatch_id="disp-7"), _budgets(), run_one=runner)
    argv, _env, _timeout = runner.calls[0]
    assert "--name=disp-7" in argv


def test_spawn_max_turns_is_budgets_driver_turns(tmp_path):
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    runner = CapturingRunOne(stdout=_exec_json())
    spawn(pack, work, _model_cfg(), _capability(), _budgets(driver_turns=77), run_one=runner)
    argv, _env, _timeout = runner.calls[0]
    command = _argv_tail(argv)
    i = command.index("--max-turns")
    assert command[i + 1] == "77"


def test_spawn_model_flag_is_worker_model_from_registry_mapping(tmp_path):
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    runner = CapturingRunOne(stdout=_exec_json())
    spawn(
        pack,
        work,
        _model_cfg(model="kimi3", worker_model="synthetic/hf:moonshotai/Kimi-K3", effort="none"),
        _capability(),
        _budgets(),
        run_one=runner,
    )
    argv, _env, _timeout = runner.calls[0]
    command = _argv_tail(argv)
    mi = command.index("--model")
    assert command[mi + 1] == "synthetic/hf:moonshotai/Kimi-K3"
    ei = command.index("--effort")
    assert command[ei + 1] == "none"


def test_spawn_timeout_is_timeout_seconds_plus_slack(tmp_path):
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    runner = CapturingRunOne(stdout=_exec_json())
    spawn(pack, work, _model_cfg(timeout_seconds=60), _capability(), _budgets(), run_one=runner)
    _argv, _env, timeout = runner.calls[0]
    assert timeout == 90  # 60 + the 30s backstop slack (claude_code parity)


# ==========================================================================
# structural contract (parity with claude_code.spawn)
# ==========================================================================


def test_spawn_missing_prompt_md_raises_driver_error(tmp_path):
    pack = tmp_path / "pack"
    pack.mkdir()
    work = _make_work_clone(tmp_path, with_work_branch=False)
    with pytest.raises(DriverError):
        spawn(pack, work, _model_cfg(), _capability(), _budgets(), run_one=CapturingRunOne())


# ==========================================================================
# classification table (spec §4.1 — implement EXACTLY)
# ==========================================================================


def _classify_spawn(tmp_path, stdout: str, *, returncode: int = 0, budgets=None):
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    runner = CapturingRunOne(stdout=stdout, returncode=returncode)
    return spawn(pack, work, _model_cfg(), _capability(), budgets or _budgets(), run_one=runner)


def test_classify_done_is_done(tmp_path):
    result = _classify_spawn(tmp_path, _exec_json())
    assert result.status is DispatchStatus.DONE
    assert result.ceiling_trip is None
    assert result.transcript == _exec_json()  # raw stdout, never sealed here


def test_classify_done_at_output_ceiling_is_failed_output_tokens(tmp_path):
    # claude_code parity: done + usage.out >= budgets.output_tokens ->
    # FAILED/ceiling_trip=output_tokens (the output_tokens ceiling).
    usage = {"in": 1, "cached": 0, "out": 1000, "reasoning": 0}
    result = _classify_spawn(tmp_path, _exec_json(usage=usage))
    assert result.status is DispatchStatus.FAILED
    assert result.ceiling_trip == "output_tokens"


def test_classify_done_above_output_ceiling_is_failed_output_tokens(tmp_path):
    usage = {"in": 1, "cached": 0, "out": 5000, "reasoning": 0}
    result = _classify_spawn(tmp_path, _exec_json(usage=usage))
    assert result.status is DispatchStatus.FAILED
    assert result.ceiling_trip == "output_tokens"


def test_classify_done_truncated_max_tokens_is_failed_output_tokens(tmp_path):
    # A done result whose final turn ended on a truncated completion is
    # NEVER read as a success: FAILED with the output_tokens ceiling, even
    # with usage far under budget (a provider can cut the turn early).
    usage = {"in": 1, "cached": 0, "out": 10, "reasoning": 0}  # << 1000 budget
    result = _classify_spawn(
        tmp_path, _exec_json(status="done", stop_reason="max_tokens", usage=usage)
    )
    assert result.status is DispatchStatus.FAILED
    assert result.status is not DispatchStatus.DONE
    assert result.ceiling_trip == "output_tokens"
    # The offending stop_reason value rides the detail provenance channel
    # (the same channel as the 'relay-deny:<reason>' details).
    assert "max_tokens" in result.detail


def test_classify_done_truncated_length_is_failed_output_tokens(tmp_path):
    # The OpenAI-surface truncation form: same classification.
    usage = {"in": 1, "cached": 0, "out": 10, "reasoning": 0}  # << 1000 budget
    result = _classify_spawn(
        tmp_path, _exec_json(status="done", stop_reason="length", usage=usage)
    )
    assert result.status is DispatchStatus.FAILED
    assert result.status is not DispatchStatus.DONE
    assert result.ceiling_trip == "output_tokens"
    assert "length" in result.detail


def test_classify_done_truncated_ignores_usage_at_ceiling(tmp_path):
    # The truncation row is read regardless of the usage values — usage at
    # the ceiling does not change the classification (still FAILED with the
    # output_tokens ceiling, stop_reason named in detail).
    usage = {"in": 1, "cached": 0, "out": 1000, "reasoning": 0}
    result = _classify_spawn(
        tmp_path, _exec_json(status="done", stop_reason="max_tokens", usage=usage)
    )
    assert result.status is DispatchStatus.FAILED
    assert result.ceiling_trip == "output_tokens"
    assert "max_tokens" in result.detail


def test_classify_done_non_truncation_stop_reason_is_done(tmp_path):
    # A non-truncation stop_reason at done (usage under budget) still
    # classifies DONE — only max_tokens/length are truncation signals.
    result = _classify_spawn(tmp_path, _exec_json(status="done", stop_reason="end_turn"))
    assert result.status is DispatchStatus.DONE
    assert result.ceiling_trip is None


def test_classify_quota_calls_deny_is_failed_driver_turns(tmp_path):
    # spec §2 design decision 2: a quota-calls deny is Stigmergy's OWN
    # per-dispatch budget speaking — a ceiling trip (DEGENERATE rung attempt),
    # NEVER INFRA. The .134 provenance rides in the DISPATCH event detail.
    result = _classify_spawn(
        tmp_path, _exec_json(status="failed", deny_reason="quota-calls", detail="call limit")
    )
    assert result.status is DispatchStatus.FAILED
    assert result.ceiling_trip == "driver_turns"
    assert result.detail.startswith("relay-deny:quota-calls")


def test_classify_quota_tokens_deny_is_failed_output_tokens(tmp_path):
    result = _classify_spawn(
        tmp_path, _exec_json(status="failed", deny_reason="quota-tokens", detail="token limit")
    )
    assert result.status is DispatchStatus.FAILED
    assert result.ceiling_trip == "output_tokens"
    assert result.detail.startswith("relay-deny:quota-tokens")


@pytest.mark.parametrize(
    "reason",
    ["unknown", "revoked", "missing-capability", "forbidden-endpoint", "malformed-request"],
)
def test_classify_other_deny_reasons_failed_with_provenance(tmp_path, reason):
    # Capability failure (not our budget): FAILED, rung attempt consumed,
    # detail carries the deny reason as provenance.
    result = _classify_spawn(
        tmp_path, _exec_json(status="failed", deny_reason=reason, detail=f"denied: {reason}")
    )
    assert result.status is DispatchStatus.FAILED
    assert result.ceiling_trip is None
    assert result.detail.startswith(f"relay-deny:{reason}")


def test_classify_iteration_cap_sentinel_is_failed_driver_turns(tmp_path):
    # spec §2.5: max_iterations exhaustion surfaces as ceiling_trip
    # "driver_turns" in exec's result JSON.
    result = _classify_spawn(tmp_path, _exec_json(status="failed", ceiling_trip="driver_turns"))
    assert result.status is DispatchStatus.FAILED
    assert result.ceiling_trip == "driver_turns"


def test_classify_infra_is_infra_with_detail_carried(tmp_path):
    # Genuinely-forwarded upstream/transport failure (no deny marker): INFRA.
    result = _classify_spawn(tmp_path, _exec_json(status="infra", detail="upstream 503"))
    assert result.status is DispatchStatus.INFRA
    assert result.ceiling_trip is None
    assert "upstream 503" in result.detail


def test_classify_wedged_is_wedged(tmp_path):
    result = _classify_spawn(tmp_path, _exec_json(status="wedged"))
    assert result.status is DispatchStatus.WEDGED
    assert result.ceiling_trip is None


def test_classify_unknown_status_is_failed_never_infra(tmp_path):
    # Closed conservative vocabulary: an unrecognized status string is a
    # capability/quality FAILED, NEVER an infra backoff.
    result = _classify_spawn(tmp_path, _exec_json(status="mystery"))
    assert result.status is DispatchStatus.FAILED
    assert result.status is not DispatchStatus.INFRA
    assert result.ceiling_trip is None


# ==========================================================================
# process-level parity (claude_code parity: timeout / launch / exit codes)
# ==========================================================================


def test_spawn_timeout_expired_is_wedged(tmp_path):
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    exc = subprocess.TimeoutExpired(cmd=["podman"], timeout=90)
    result = spawn(pack, work, _model_cfg(), _capability(), _budgets(), run_one=RaisingRunOne(exc))
    assert result.status is DispatchStatus.WEDGED
    assert result.usage == {"in": 0, "cached": 0, "out": 0, "reasoning": 0}


def test_spawn_oserror_launch_is_infra(tmp_path):
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    raise_one = RaisingRunOne(OSError("no podman"))
    result = spawn(pack, work, _model_cfg(), _capability(), _budgets(), run_one=raise_one)
    assert result.status is DispatchStatus.INFRA
    assert "OSError" in result.detail


def test_spawn_returncode_69_is_infra(tmp_path):
    # Entrypoint fail-closed (cage egress never came up) — INFRA parity.
    result = _classify_spawn(tmp_path, "entrypoint: egress socket never ready\n", returncode=69)
    assert result.status is DispatchStatus.INFRA
    assert "69" in result.detail


def test_spawn_returncode_137_no_parse_is_wedged(tmp_path):
    # podman --timeout killed the container — WEDGED parity.
    result = _classify_spawn(tmp_path, "killed\n", returncode=137)
    assert result.status is DispatchStatus.WEDGED
    assert "137" in result.detail


def test_spawn_unparseable_stdout_other_exit_is_failed(tmp_path):
    result = _classify_spawn(tmp_path, "not json at all", returncode=1)
    assert result.status is DispatchStatus.FAILED
    assert "openalph-exec" in result.detail


# ==========================================================================
# usage mapping (exec's 4-key usage verbatim; coerced non-negative)
# ==========================================================================


def test_usage_mapped_verbatim_4_key(tmp_path):
    result = _classify_spawn(
        tmp_path, _exec_json(usage={"in": 12, "cached": 4, "out": 99, "reasoning": 7})
    )
    assert result.usage == {"in": 12, "cached": 4, "out": 99, "reasoning": 7}


def test_usage_coerced_non_negative(tmp_path):
    # Missing keys, negatives, non-ints, and bools all coerce to 0 — never
    # negative, never fabricated (records.py `_validate_tokens` invariant).
    result = _classify_spawn(
        tmp_path, _exec_json(usage={"in": -5, "cached": "x", "out": 3, "reasoning": True})
    )
    assert result.usage["in"] == 0
    assert result.usage["cached"] == 0
    assert result.usage["out"] == 3
    assert result.usage["reasoning"] == 0
    assert set(result.usage) == {"in", "cached", "out", "reasoning"}


def test_usage_non_dict_coerces_to_zeros(tmp_path):
    raw = json.dumps({"status": "done", "usage": "garbage"})
    result = _classify_spawn(tmp_path, raw)
    assert result.usage == {"in": 0, "cached": 0, "out": 0, "reasoning": 0}


# ==========================================================================
# bundle reuse (spec §4.1: reuse claude_code._create_work_bundle — do not
# duplicate) + the "regardless of status" bundle rule
# ==========================================================================


def test_spawn_reuses_claude_code_work_bundle(tmp_path):
    assert openalph_exec._create_work_bundle is claude_code_module._create_work_bundle


def test_spawn_creates_bundle_when_work_branch_exists(tmp_path):
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=True)
    bundle_dir = tmp_path / "bundles"
    result = spawn(
        pack, work, _model_cfg(), _capability(), _budgets(), run_one=CapturingRunOne(_exec_json()),
        bundle_dir=bundle_dir,
    )
    assert result.status is DispatchStatus.DONE
    assert result.bundle_ref is not None
    assert result.bundle_ref.endswith("disp-1.bundle")
    assert (tmp_path / "bundles" / "disp-1.bundle").is_file()


def test_spawn_bundle_absent_when_no_work_branch(tmp_path):
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    result = spawn(
        pack, work, _model_cfg(), _capability(), _budgets(), run_one=CapturingRunOne(_exec_json())
    )
    assert result.bundle_ref is None


def test_spawn_bundle_attempted_regardless_of_failed_status(tmp_path):
    # A ceiling-tripped dispatch that still committed work gets a bundle
    # (the claude_code §0.6 rule, reused).
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=True)
    result = spawn(
        pack,
        work,
        _model_cfg(),
        _capability(),
        _budgets(),
        run_one=CapturingRunOne(_exec_json(status="failed", deny_reason="quota-calls")),
        bundle_dir=tmp_path / "bundles2",
    )
    assert result.status is DispatchStatus.FAILED
    assert result.bundle_ref is not None


def test_spawn_empty_effort_raises_driver_error(tmp_path):
    # Defensive caller-bug guard: an empty/non-str effort would render an
    # invalid in-cage `--effort` flag — fail loud, not a mystery container
    # failure.
    pack = _task_pack(tmp_path)
    work = _make_work_clone(tmp_path, with_work_branch=False)
    with pytest.raises(DriverError):
        spawn(
            pack, work, _model_cfg(effort=""), _capability(), _budgets(),
            run_one=CapturingRunOne(),
        )
