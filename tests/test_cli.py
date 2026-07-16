"""Tests for stigmergy.cli's `daemon run --rig <name>` subcommand and its
collaborator-wiring helpers (bead .27 build spec §2, §2.1; frozen case list
§4 cases 13-21).

`_run_daemon` (which loops forever) is ALWAYS monkeypatched to a recording
stub — the real one is never called in a test. Rig fixtures go through the
REAL create_rig (real local git clone) so `repo/prompts/{code01,critic01}`
exist for `_build_daemon`'s `Critic.from_prompt_file` and `derive_steering`.
No real podman / network / provider calls anywhere.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

import stigmergy.cli as cli
from stigmergy.cli import (
    _DEFAULT_PROTECTED_PATHS,
    _build_daemon,
    _make_steering_of,
    main,
)
from stigmergy.container import PodmanContainerReaper
from stigmergy.critic import Critic, CriticInfraError
from stigmergy.daemon import Daemon
from stigmergy.rig import create_rig, resolve_rig
from stigmergy.steering import derive_steering

FIXTURES = Path(__file__).parent / "fixtures"
VALID_CHARTER_PATH = FIXTURES / "charter_valid.toml"
MODELS_REGISTRY_PATH = FIXTURES / "models.toml"
BASE_CHARTER_TOML = VALID_CHARTER_PATH.read_text()


# ==========================================================================
# fixtures (real git clone; matches test_rig.py's own helper style)
# ==========================================================================


def make_local_repo_with_prompts(tmp_path: Path, name: str = "source_repo") -> Path:
    repo_dir = tmp_path / name
    (repo_dir / "prompts").mkdir(parents=True)
    (repo_dir / "prompts" / "code01").write_text("code01 template: $goal\n")
    (repo_dir / "prompts" / "critic01").write_text("critic01 template\n")
    (repo_dir / "README.md").write_text("hello from the fixture repo\n")
    env_cfg = ["-c", "user.email=test@example.com", "-c", "user.name=Test User"]
    subprocess.run(["git", "init", "-q", str(repo_dir)], check=True)
    subprocess.run(["git", *env_cfg, "-C", str(repo_dir), "add", "-A"], check=True)
    subprocess.run(
        ["git", *env_cfg, "-C", str(repo_dir), "commit", "-q", "-m", "initial commit"],
        check=True,
    )
    return repo_dir


def make_charter(tmp_path: Path, repo: Path | str) -> Path:
    charter_dir = tmp_path / "charter_src"
    charter_dir.mkdir(exist_ok=True)
    text = BASE_CHARTER_TOML.replace('repo = "path-or-url"', f'repo = "{repo}"')
    charter_path = charter_dir / "charter.toml"
    charter_path.write_text(text)
    shutil.copy(MODELS_REGISTRY_PATH, charter_dir / "models.toml")
    return charter_path


def scaffold_rig(tmp_path: Path, rigs_root: Path | None = None) -> Path:
    """Scaffold a real rig named 'shipyard' and return its rigs_root."""
    repo = make_local_repo_with_prompts(tmp_path)
    charter_path = make_charter(tmp_path, repo)
    if rigs_root is None:
        rigs_root = tmp_path / "rigs"
    create_rig(charter_path, base_dir=rigs_root)
    return rigs_root


# --- case 13: `daemon run` builds and hands off a real Daemon ---------------


def test_main_daemon_run_builds_real_daemon(tmp_path: Path, monkeypatch) -> None:
    rigs_root = scaffold_rig(tmp_path)
    recorded: list[Daemon] = []
    monkeypatch.setattr(cli, "_run_daemon", recorded.append)

    rc = main(["daemon", "run", "--rig", "shipyard", "--rigs-root", str(rigs_root)])

    assert rc == 0
    assert len(recorded) == 1
    daemon = recorded[0]
    try:
        assert isinstance(daemon, Daemon)
        # Real, non-placeholder collaborators where it matters:
        assert isinstance(daemon._container_reaper, PodmanContainerReaper)
        # The critic is wired with the REAL provider-calling client (bead .36),
        # NOT a silently-wrong stub. It's a make_critic_client closure; its own
        # request/response + fail-closed behavior is covered in
        # test_critic_client.py. Daemon construction must NOT shell out to
        # 1Password (make_op_key_provider is lazy) -- reaching this assertion
        # here without any op/network access is itself that regression guard.
        assert callable(daemon._weaver.critic._client)
        assert daemon._weaver.critic.model == "opus"  # charter [roles.critic].model
    finally:
        daemon._store.close()


# --- case 14: unknown rig -> exit 1, stderr, loop NEVER entered -------------


def test_main_daemon_run_unknown_rig_returns_1(tmp_path: Path, monkeypatch, capsys) -> None:
    called: list[Daemon] = []
    monkeypatch.setattr(cli, "_run_daemon", called.append)

    rc = main(["daemon", "run", "--rig", "ghost", "--rigs-root", str(tmp_path)])

    assert rc == 1
    assert called == []  # regression guard: construction failed, never looped
    err = capsys.readouterr().err
    assert "ghost" in err


# --- case 15: --rigs-root is threaded through -------------------------------


def test_main_daemon_run_rigs_root_override(tmp_path: Path, monkeypatch) -> None:
    custom_root = tmp_path / "custom_rigs_location"
    scaffold_rig(tmp_path, rigs_root=custom_root)
    recorded: list[Daemon] = []
    monkeypatch.setattr(cli, "_run_daemon", recorded.append)

    # The rig exists ONLY under the non-default custom root.
    rc = main(["daemon", "run", "--rig", "shipyard", "--rigs-root", str(custom_root)])

    assert rc == 0
    assert len(recorded) == 1
    recorded[0]._store.close()


# --- case 16: _build_daemon wiring points -----------------------------------


def test_build_daemon_wiring(tmp_path: Path) -> None:
    rigs_root = scaffold_rig(tmp_path)
    resolved = resolve_rig("shipyard", rigs_root=rigs_root)
    # _build_daemon takes ownership of resolved.store via the Daemon.
    daemon = _build_daemon(resolved)
    try:
        # checker_image == charter [rig].image (v0: one image for worker+checker)
        assert daemon._checker_image == "stigmergy-worker:py312"

        # spend leash budgets: charter values + reserve_usd v0 default of 0.0
        budgets = daemon._spend_leash._budgets
        assert budgets.reserve_usd == 0.0
        assert budgets.dispatches == 50
        assert budgets.usd == 25.0
        assert budgets.gate_calls == 30

        # weaver protected paths (fixed tuple, incl. prompts/) + journal path
        assert daemon._weaver.protected_paths == list(_DEFAULT_PROTECTED_PATHS)
        assert "prompts/" in _DEFAULT_PROTECTED_PATHS
        assert (
            daemon._weaver.journal_path
            == resolved.rig_paths["records_dir"] / "weave_journal.jsonl"
        )

        # notifier topic from charter [notify].ntfy_topic
        assert daemon._notifier.topic == "stigmergy"
    finally:
        daemon._store.close()


# --- case 17: _make_steering_of == derive_steering (no silent divergence) ---


def test_make_steering_of_matches_derive_steering(tmp_path: Path) -> None:
    rigs_root = scaffold_rig(tmp_path)
    resolved = resolve_rig("shipyard", rigs_root=rigs_root)
    try:
        resolved.store.add_ticket(
            id="t-1",
            title="Do the thing",
            goal="ship it",
            required_reading=[],
            target_scope=["src/foo.py"],
            acceptance_criteria=["works"],
            tier1_checks=[],
        )
        prompts_dir = resolved.rig_paths["prompts_dir"]
        steering_of = _make_steering_of(resolved.store, resolved.charter, prompts_dir)

        via_closure = steering_of("t-1")
        direct = derive_steering(
            resolved.store.get_ticket("t-1"), resolved.charter, prompts_dir
        )
        assert via_closure == direct
    finally:
        resolved.store.close()


# --- case 18: any raising critic client surfaces as CriticInfraError ---------


def test_raising_critic_client_surfaces_as_infra_error() -> None:
    """A critic client that raises must surface as a critic-INFRA trip
    (non-crashing), never a raw exception and never a silent gate verdict —
    proved against the REAL Critic.judge(). (bead .36 removed the old
    _unwired_critic_client placeholder; the real provider-calling client's own
    fail-closed paths — transport error, malformed/missing verdict — are
    covered in test_critic_client.py. This preserves the invariant that ANY
    client exception is infra, independent of which client is wired.)"""

    def raising_client(prompt: str, *, model: str, **decoding_params: object) -> object:
        raise RuntimeError("provider unreachable")

    critic = Critic(
        client=raising_client,
        model="opus",
        decoding_params={"temperature": 0.0},
        template="judge the artifact",
    )
    with pytest.raises(CriticInfraError):
        critic.judge("some artifact", ["some rubric item"])


# --- case 20: bare invocation still prints usage + returns 0 ----------------


def test_main_bare_invocation_prints_usage(capsys) -> None:
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "usage" in out.lower()


# --- case 21: the new subcommand is discoverable in usage -------------------


def test_usage_mentions_daemon_run(capsys) -> None:
    main([])
    out = capsys.readouterr().out
    assert "daemon run --rig" in out
