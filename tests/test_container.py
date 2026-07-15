"""Adversarial tests for the hardened worker container profile (SPEC.md §4).

These tests are the security specification for bead .10. They are authored by
the orchestrator, not the implementor: a hardened profile that a worker could
trivially escape is worse than none, so the *assertions* here — which flags are
mandatory, which are forbidden, and what the container can actually reach — are
the fixed pass bar. The implementation in ``stigmergy.container`` must satisfy
them; it must not weaken them.

Two tiers:
  * deterministic argv-construction tests (no containers) — the regression guard
    on the hardening flag set and the mount table (SPEC §4 normative mount table);
  * live containment tests (real rootless podman) — AC5: fork-bomb and disk-fill
    are contained, a concurrent workload is unaffected. These skip when podman is
    unavailable.

Canonical flag forms are pinned here on purpose; the implementation emits exactly
these tokens so the security surface is asserted by identity, not by fuzzy match.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from stigmergy.container import (
    ContainerError,
    ContainerProfile,
    build_image,
    build_run_argv,
    worker_env,
)

PODMAN = shutil.which("podman")
requires_podman = pytest.mark.skipif(PODMAN is None, reason="podman not installed")

# A syntactically valid digest-pinned image ref for the deterministic tests.
PINNED_IMAGE = "localhost/stigmergy-worker@sha256:" + "a" * 64
# A small real image for live tests (pulled during environment setup).
LIVE_IMAGE = "docker.io/library/python:3.12-alpine"


@pytest.fixture(scope="module")
def pinned_live_image():
    """A digest-pinned ref for LIVE_IMAGE, resolved programmatically.

    The live containment tests need a *pinned* image because build_run_argv
    (correctly) refuses any unpinned ``profile.image``. The digest is captured
    by subprocess and never hand-copied, so no redacted placeholder can leak.
    """
    if PODMAN is None:
        pytest.skip("podman not installed")
    subprocess.run(
        ["podman", "pull", LIVE_IMAGE],
        env={**worker_env()},
        capture_output=True,
        text=True,
        check=False,
    )
    ref = subprocess.run(
        ["podman", "inspect", "--format", "{{index .RepoDigests 0}}", LIVE_IMAGE],
        env={**worker_env()},
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert "@sha256:" in ref
    return ref


def _profile(tmp_path, **overrides):
    work = tmp_path / "work"
    work.mkdir(parents=True, exist_ok=True)
    task = tmp_path / "task"
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


def _argv(tmp_path, command=None, **overrides):
    return build_run_argv(_profile(tmp_path, **overrides), command=command or ["true"])


# --------------------------------------------------------------------------
# Deterministic: mandatory hardening flags are present (SPEC §4 worker cage)
# --------------------------------------------------------------------------


def test_argv_is_podman_run(tmp_path):
    argv = _argv(tmp_path)
    assert argv[0] == "podman"
    assert argv[1] == "run"


def test_argv_drops_all_capabilities(tmp_path):
    assert "--cap-drop=ALL" in _argv(tmp_path)


def test_argv_no_new_privileges(tmp_path):
    assert "--security-opt=no-new-privileges" in _argv(tmp_path)


def test_argv_read_only_rootfs(tmp_path):
    assert "--read-only" in _argv(tmp_path)


def test_argv_removes_the_container_afterwards(tmp_path):
    # ephemeral: one dispatch, no lingering container state
    assert "--rm" in _argv(tmp_path)


def test_argv_cgroup_limits_present(tmp_path):
    argv = _argv(tmp_path)
    assert "--pids-limit=32" in argv
    assert "--memory=256m" in argv
    assert "--cpus=1" in argv


def test_argv_timeout_present(tmp_path):
    # podman kills the whole container (its cgroup) after this many seconds
    assert "--timeout=120" in _argv(tmp_path)


def test_argv_network_none_by_default(tmp_path):
    # .10 baseline is no network; the egress bead (.11) overrides with the
    # internal netns routed through the proxy. Default must never be open.
    assert "--network=none" in _argv(tmp_path)


def test_argv_network_override_is_honored(tmp_path):
    argv = _argv(tmp_path, network="ns:/run/netns/dispatch-x")
    assert "--network=ns:/run/netns/dispatch-x" in argv
    assert "--network=none" not in argv


# --------------------------------------------------------------------------
# Deterministic: the normative mount table, and NOTHING else (SPEC §4 table)
# --------------------------------------------------------------------------


def test_argv_mounts_work_rw(tmp_path):
    argv = _argv(tmp_path)
    work = str((tmp_path / "work").resolve())
    assert f"--volume={work}:/work:rw" in argv


def test_argv_mounts_task_ro(tmp_path):
    argv = _argv(tmp_path)
    task = str((tmp_path / "task").resolve())
    assert f"--volume={task}:/task:ro" in argv


def test_argv_scratch_is_size_capped_tmpfs(tmp_path):
    argv = _argv(tmp_path)
    joined = " ".join(argv)
    # a tmpfs at /scratch with the configured size cap
    assert "--tmpfs=/scratch:" in joined
    assert "size=64m" in joined


def test_argv_no_extra_bind_mounts(tmp_path):
    # Exactly two --volume binds (work, task). No host $HOME, workspace,
    # /opt/openalph, or any socket is mounted.
    argv = _argv(tmp_path)
    volumes = [a for a in argv if a.startswith("--volume=")]
    assert len(volumes) == 2, f"expected exactly work+task binds, got {volumes}"


def test_argv_mounts_no_forbidden_host_paths(tmp_path):
    joined = " ".join(_argv(tmp_path))
    for forbidden in (
        "docker.sock",
        "/var/run/docker",
        "/run/user",  # runtime dir goes in env, never mounted into the worker
        "/home/oa-merry/workspace",
        "/opt/openalph",
        "/srv/openalph",
        ".ssh",
        ".git-credentials",
        ".config/op",
    ):
        assert forbidden not in joined, f"forbidden host path leaked into mounts: {forbidden}"


# --------------------------------------------------------------------------
# Deterministic: forbidden relaxations never appear (charter cannot relax v0)
# --------------------------------------------------------------------------


def test_argv_never_privileged(tmp_path):
    assert "--privileged" not in _argv(tmp_path)


def test_argv_never_disables_seccomp(tmp_path):
    joined = " ".join(_argv(tmp_path))
    # default seccomp profile must be retained — never unconfined
    assert "seccomp=unconfined" not in joined
    assert "--security-opt=seccomp=unconfined" not in _argv(tmp_path)


def test_argv_never_adds_capabilities(tmp_path):
    assert not any(a.startswith("--cap-add") for a in _argv(tmp_path))


def test_argv_never_shares_host_namespaces(tmp_path):
    joined = " ".join(_argv(tmp_path))
    for bad in ("--pid=host", "--ipc=host", "--uts=host", "--userns=host", "--network=host"):
        assert bad not in joined


def test_profile_rejects_unpinned_image(tmp_path):
    # supply-chain: the profile image must be digest-pinned (@sha256:...)
    with pytest.raises(ContainerError):
        build_run_argv(
            _profile(tmp_path, image="localhost/stigmergy-worker:latest"),
            command=["true"],
        )


# --------------------------------------------------------------------------
# Deterministic: worker invocation environment (rootless cgroup enforcement)
# --------------------------------------------------------------------------


def test_worker_env_points_at_user_runtime(tmp_path):
    env = worker_env()
    # rootless podman must talk to the user systemd instance so cgroup
    # limits are actually enforced (else it silently falls back to cgroupfs).
    assert env.get("XDG_RUNTIME_DIR")
    assert "DBUS_SESSION_BUS_ADDRESS" in env


# --------------------------------------------------------------------------
# Live containment (AC5) — real rootless podman + cgroup v2 enforcement
# --------------------------------------------------------------------------


def _run_live(profile, command, timeout=90):
    argv = build_run_argv(profile, command=command)
    return subprocess.run(
        argv,
        env={**worker_env()},
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


# --------------------------------------------------------------------------
# Provision (`provision` station): supply-chain rules on image build (SPEC §3/§4)
# --------------------------------------------------------------------------


def test_build_image_rejects_unpinned_base(tmp_path):
    # A Containerfile whose FROM is not digest-pinned (@sha256:) must be
    # rejected BEFORE any build runs — pinned base-image digests are a
    # non-negotiable supply-chain rule (SPEC §3 provision). No podman needed:
    # the check parses the Containerfile first.
    cf_dir = tmp_path / "images" / "worker"
    cf_dir.mkdir(parents=True)
    (cf_dir / "Containerfile").write_text("FROM python:3.12-alpine\nRUN true\n")
    with pytest.raises(ContainerError):
        build_image(cf_dir, "localhost/stigmergy-worker:test")


def test_build_image_rejects_multistage_unpinned_base(tmp_path):
    # Every FROM must be pinned, including later stages in a multi-stage build.
    cf_dir = tmp_path / "images" / "worker"
    cf_dir.mkdir(parents=True)
    (cf_dir / "Containerfile").write_text(
        "FROM python@sha256:" + "a" * 64 + " AS build\n"
        "RUN true\n"
        "FROM alpine:3.20\n"  # unpinned second stage
        "RUN true\n"
    )
    with pytest.raises(ContainerError):
        build_image(cf_dir, "localhost/stigmergy-worker:test")


@requires_podman
def test_build_image_pinned_base_builds_and_returns_digest(tmp_path):
    # A digest-pinned Containerfile builds and yields an image digest string.
    # The base digest is resolved programmatically (never hand-copied) so no
    # redacted placeholder can leak in.
    base = subprocess.run(
        ["podman", "inspect", "--format", "{{index .RepoDigests 0}}", LIVE_IMAGE],
        env={**worker_env()},
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert "@sha256:" in base
    cf_dir = tmp_path / "images" / "worker"
    cf_dir.mkdir(parents=True)
    (cf_dir / "Containerfile").write_text(f"FROM {base}\nRUN true\n")
    digest = build_image(cf_dir, "localhost/stigmergy-worker:test")
    assert isinstance(digest, str) and "sha256:" in digest


@requires_podman
def test_live_pids_limit_contains_fork_storm(tmp_path, pinned_live_image):
    # AC5: a fork storm is capped by --pids-limit and cannot exhaust host PIDs.
    # Bounded (100 attempts), so a broken limit can't harm the host.
    profile = _profile(tmp_path, image=pinned_live_image, pids_limit=16)
    result = _run_live(
        profile,
        ["sh", "-c", "i=0; while [ $i -lt 100 ]; do sleep 3 & i=$((i+1)); done; wait"],
    )
    # The cap must bite: the shell reports fork failures. The host is fine
    # regardless (this test process is still running to make the assertion).
    assert "can't fork" in result.stderr or "Resource temporarily unavailable" in result.stderr


@requires_podman
def test_live_scratch_tmpfs_size_cap_contains_disk_fill(tmp_path, pinned_live_image):
    # AC5: writing past the /scratch tmpfs cap fails at the cap — a disk-fill
    # worker cannot exhaust host disk. /scratch is 8m here; try to write 64m.
    profile = _profile(tmp_path, image=pinned_live_image, scratch_size="8m")
    result = _run_live(
        profile,
        ["sh", "-c", "dd if=/dev/zero of=/scratch/fill bs=1M count=64 2>&1; echo DONE"],
    )
    out = result.stdout + result.stderr
    assert "DONE" in out
    # dd must fail before writing the full 64m (No space left on device)
    assert "No space left on device" in out or "no space" in out.lower()


@requires_podman
def test_live_read_only_rootfs_blocks_root_writes(tmp_path, pinned_live_image):
    # Writing to the read-only rootfs (outside /work and /scratch) must fail.
    profile = _profile(tmp_path, image=pinned_live_image)
    result = _run_live(
        profile,
        ["sh", "-c", "echo x > /evil 2>&1; echo RC=$?"],
    )
    out = result.stdout + result.stderr
    assert "Read-only file system" in out or "RC=1" in out


@requires_podman
def test_live_concurrent_workload_unaffected_by_limited_container(tmp_path, pinned_live_image):
    # AC5: a resource-limited (fork-storming) container runs while a second,
    # normal container completes unaffected — the limits are per-container.
    import concurrent.futures

    hog = _profile(tmp_path / "hog", image=pinned_live_image, pids_limit=16)
    good = _profile(tmp_path / "good", image=pinned_live_image)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        fut_hog = ex.submit(
            _run_live,
            hog,
            ["sh", "-c", "i=0; while [ $i -lt 100 ]; do sleep 2 & i=$((i+1)); done; wait"],
        )
        fut_good = ex.submit(_run_live, good, ["sh", "-c", "echo healthy"])
        good_result = fut_good.result(timeout=90)
        fut_hog.result(timeout=90)

    assert good_result.returncode == 0
    assert "healthy" in good_result.stdout
