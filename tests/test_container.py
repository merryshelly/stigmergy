"""Adversarial tests for the hardened worker container profile (SPEC.md §4).

These tests are the security specification for ticket .10. They are authored by
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

import json
import shutil
import subprocess
import uuid

import pytest

from stigmergy.container import (
    DISPATCH_ID_LABEL_KEY,
    ContainerError,
    ContainerProfile,
    PodmanContainerReaper,
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
    # build_run_argv-only kwargs (not ContainerProfile fields) are split out
    # here so callers can pass e.g. dispatch_id=/egress_socket=/env= without
    # ContainerProfile(**kwargs) choking on an unexpected keyword — every
    # existing call site only ever passes ContainerProfile fields, so this
    # split is purely additive and does not change any existing test.
    build_kwargs = {}
    for key in ("egress_socket", "relay_socket", "env", "dispatch_id", "entrypoint_override"):
        if key in overrides:
            build_kwargs[key] = overrides.pop(key)
    return build_run_argv(
        _profile(tmp_path, **overrides), command=command or ["true"], **build_kwargs
    )


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
    # .10 baseline is no network; the egress ticket (.11) overrides with the
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


def test_profile_accepts_bare_sha256_image_id(tmp_path):
    # Bead .63: a locally-BUILT worker image has no RepoDigests and is run by
    # its bare content-addressed id (`sha256:<64hex>`) — as immutable as an
    # `@sha256:` registry ref. build_run_argv must accept it.
    bare = "sha256:" + "a" * 64
    argv = build_run_argv(_profile(tmp_path, image=bare), command=["true"])
    assert argv[-2] == bare  # image is the second-to-last token (command last)


def test_profile_accepts_registry_ref_pinned_by_digest(tmp_path):
    # The pre-.63 form still works: name@sha256:<64hex>.
    ref = "docker.io/library/node@sha256:" + "b" * 64
    argv = build_run_argv(_profile(tmp_path, image=ref), command=["true"])
    assert ref in argv


def test_profile_rejects_bare_sha256_wrong_length(tmp_path):
    # A near-miss (not exactly 64 hex) is still a mutable/garbage ref.
    with pytest.raises(ContainerError):
        build_run_argv(
            _profile(tmp_path, image="sha256:" + "a" * 63), command=["true"]
        )


def test_profile_rejects_transport_prefixed_ref_with_decorative_digest(tmp_path):
    # Bead .63 review (codex-sol-xhigh): a non-registry transport with a
    # DECORATIVE @sha256: substring pins nothing (the source is mutable/local)
    # and must be refused even though it contains @sha256:<64hex>.
    for ref in (
        "dir:/tmp/evil@sha256:" + "a" * 64,
        "docker-archive:/tmp/x.tar@sha256:" + "b" * 64,
        "containers-storage:localhost/x@sha256:" + "c" * 64,
        "oci:/tmp/layout@sha256:" + "d" * 64,
    ):
        with pytest.raises(ContainerError):
            build_run_argv(_profile(tmp_path, image=ref), command=["true"])


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
    # bead .93: the returned ref must be RUNNABLE, not just sha256-shaped.
    # `.Digest` (manifest) is sha256-shaped but `podman run` rejects it bare
    # ("image not known") on podman versions that populate it for local
    # builds — the daemon would fail every dispatch. Prove `podman run`
    # accepts what build_image returns.
    run = subprocess.run(
        ["podman", "run", "--rm", "--network=none", "--entrypoint=sh", digest, "-c", "true"],
        env={**worker_env()},
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, f"build_image returned a non-runnable ref: {run.stderr}"


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


# --------------------------------------------------------------------------
# Bead .34 Part A: dispatch_id labeling (build_run_argv). Cases 1-7.
# --------------------------------------------------------------------------


def test_dispatch_id_none_leaves_argv_unchanged(tmp_path):
    # Case 1: dispatch_id=None (the default) is byte-identical to the
    # pre-.34 argv — the same regression-guard discipline egress_socket=None
    # and env=None already get. Two separate assertions: (a) explicit
    # dispatch_id=None matches the param-omitted call, AND (b) neither
    # contains any --label= token at all — (a) alone cannot catch a bug
    # that corrupts both call sites identically (e.g. always emitting the
    # label regardless of dispatch_id), so (b) is the real byte-identity
    # claim against the pre-.34 baseline.
    with_none = _argv(tmp_path, dispatch_id=None)
    without_param = _argv(tmp_path)
    assert with_none == without_param
    assert not any(a.startswith("--label=") for a in with_none)
    assert not any(a.startswith("--label=") for a in without_param)


def test_dispatch_id_valid_adds_label_at_frozen_position(tmp_path):
    # Case 2: a valid dispatch_id appends exactly one --label= token,
    # immediately after --network=... and immediately before the first
    # --volume=... token. Freeze the literal index/adjacency, not just
    # "present somewhere".
    argv = _argv(tmp_path, dispatch_id="crimson-otter-basalt")
    label_token = f"--label={DISPATCH_ID_LABEL_KEY}=crimson-otter-basalt"
    assert label_token in argv

    network_index = next(i for i, a in enumerate(argv) if a.startswith("--network="))
    assert argv[network_index + 1] == label_token
    assert argv[network_index + 2].startswith("--volume=")

    first_volume_index = next(i for i, a in enumerate(argv) if a.startswith("--volume="))
    assert first_volume_index == network_index + 2


def test_dispatch_id_empty_raises(tmp_path):
    # Case 3
    with pytest.raises(ContainerError):
        _argv(tmp_path, dispatch_id="")


def test_dispatch_id_with_equals_raises(tmp_path):
    # Case 4
    with pytest.raises(ContainerError):
        _argv(tmp_path, dispatch_id="a=b")


def test_dispatch_id_with_space_raises(tmp_path):
    # Case 5
    with pytest.raises(ContainerError):
        _argv(tmp_path, dispatch_id="a b")


def test_dispatch_id_with_comma_raises(tmp_path):
    # Case 6
    with pytest.raises(ContainerError):
        _argv(tmp_path, dispatch_id="a,b")


def test_dispatch_id_combined_with_egress_socket_and_env_no_interference(tmp_path):
    # Case 7: all three additive params given together — all three tokens
    # present, no interference. Frozen relative order: --label (network-
    # adjacent, before all volumes) -> egress-socket volume -> env tokens
    # (env's own already-documented "last thing before the image" slot).
    sock = tmp_path / "egress.sock"
    env = {"ANTHROPIC_API_KEY": "tok", "ANTHROPIC_BASE_URL": "http://x"}
    argv = _argv(
        tmp_path,
        dispatch_id="crimson-otter-basalt",
        egress_socket=sock,
        env=env,
    )
    label_token = f"--label={DISPATCH_ID_LABEL_KEY}=crimson-otter-basalt"
    egress_token = f"--volume={sock}:/run/egress.sock:rw"
    env_token_1 = "--env=ANTHROPIC_API_KEY=tok"
    env_token_2 = "--env=ANTHROPIC_BASE_URL=http://x"

    assert label_token in argv
    assert egress_token in argv
    assert env_token_1 in argv
    assert env_token_2 in argv

    assert argv.index(label_token) < argv.index(egress_token) < argv.index(env_token_1)
    assert argv.index(env_token_1) < argv.index(env_token_2)

    # label still immediately precedes the first --volume= token, even
    # with egress_socket/env also present.
    network_index = next(i for i, a in enumerate(argv) if a.startswith("--network="))
    assert argv[network_index + 1] == label_token
    first_volume_index = next(i for i, a in enumerate(argv) if a.startswith("--volume="))
    assert first_volume_index == network_index + 2


# --------------------------------------------------------------------------
# Bead .63: build_run_argv(relay_socket=) — mirrors egress_socket= exactly.
# The credential-relay unix socket, mounted at /run/relay.sock, is the second
# (and last) socket the in-container shim bridges. Cases 17-18 of bead63 spec.
# --------------------------------------------------------------------------


def test_relay_socket_appends_exactly_one_relay_volume(tmp_path):
    sock = tmp_path / "relay.sock"
    argv = _argv(tmp_path, relay_socket=sock)
    relay_token = f"--volume={sock}:/run/relay.sock:rw"
    assert argv.count(relay_token) == 1
    # No egress mount unless egress_socket is also given.
    assert not any(a.endswith(":/run/egress.sock:rw") for a in argv)


def test_relay_socket_none_is_byte_identical(tmp_path):
    # Regression guard: relay_socket=None (default) yields exactly the argv a
    # caller that never heard of relay_socket would get — same discipline
    # egress_socket=None / env=None / dispatch_id=None already hold.
    baseline = _argv(tmp_path)
    with_default = _argv(tmp_path, relay_socket=None)
    assert with_default == baseline


def test_relay_socket_combined_with_egress_socket_ordered(tmp_path):
    # Both sockets present: egress volume precedes relay volume (frozen,
    # deterministic order), both precede any env tokens.
    egress = tmp_path / "egress.sock"
    relay = tmp_path / "relay.sock"
    env = {"ANTHROPIC_API_KEY": "tok"}
    argv = _argv(tmp_path, egress_socket=egress, relay_socket=relay, env=env)
    egress_token = f"--volume={egress}:/run/egress.sock:rw"
    relay_token = f"--volume={relay}:/run/relay.sock:rw"
    env_token = "--env=ANTHROPIC_API_KEY=tok"
    assert egress_token in argv and relay_token in argv and env_token in argv
    assert argv.index(egress_token) < argv.index(relay_token) < argv.index(env_token)
    # relay volume immediately follows the egress volume.
    assert argv.index(relay_token) == argv.index(egress_token) + 1


# --------------------------------------------------------------------------
# Bead .87: entrypoint_override — CHECK containers bypass the egress
# gatekeeper entrypoint; mechanically forbidden on any egress-capable cage.
# --------------------------------------------------------------------------


def test_entrypoint_override_emits_flag_before_image(tmp_path):
    # entrypoint_override="sh" -> exactly one --entrypoint=sh token,
    # positioned immediately before the image ref (all flags precede the
    # image positional in podman).
    argv = _argv(tmp_path, entrypoint_override="sh")
    assert argv.count("--entrypoint=sh") == 1
    image_index = argv.index(PINNED_IMAGE)
    assert argv.index("--entrypoint=sh") == image_index - 1


def test_entrypoint_override_none_is_byte_identical(tmp_path):
    # Regression guard (same discipline as egress_socket/env/dispatch_id):
    # the default None leaves the argv byte-identical to the pre-.87 argv,
    # so the worker spawn path (which passes nothing) is unaffected.
    baseline = _argv(tmp_path, command=["true"])
    with_default = _argv(tmp_path, command=["true"], entrypoint_override=None)
    assert baseline == with_default
    assert not any(a.startswith("--entrypoint=") for a in baseline)


def test_entrypoint_override_rejected_on_networked_container(tmp_path):
    # The security invariant: an entrypoint override on an egress-capable
    # (non-none network) container would bypass the fail-closed egress
    # gatekeeper -> refused, mechanically unrepresentable.
    with pytest.raises(ContainerError):
        _argv(tmp_path, network="ns:/run/netns/dispatch-x", entrypoint_override="sh")


def test_entrypoint_override_rejected_with_egress_socket(tmp_path):
    egress = tmp_path / "egress.sock"
    with pytest.raises(ContainerError):
        _argv(tmp_path, egress_socket=egress, entrypoint_override="sh")


def test_entrypoint_override_rejected_with_relay_socket(tmp_path):
    relay = tmp_path / "relay.sock"
    with pytest.raises(ContainerError):
        _argv(tmp_path, relay_socket=relay, entrypoint_override="sh")


# --------------------------------------------------------------------------
# Bead .34 Part C: PodmanContainerReaper, fake run_fn (no real podman).
# Cases 9-19.
# --------------------------------------------------------------------------


class _FakeCompletedProcess:
    """Minimal CompletedProcess-like stand-in — duck-typed, only the
    attributes PodmanContainerReaper actually reads."""

    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _ScriptedRunFn:
    """Deterministic fake `run_fn`: returns scripted results in call order,
    records every argv it was called with."""

    def __init__(self, results: list[_FakeCompletedProcess]):
        self._results = list(results)
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str]) -> _FakeCompletedProcess:
        self.calls.append(argv)
        return self._results[len(self.calls) - 1]


def test_list_running_returns_sorted_deduplicated_ids():
    # Case 9
    stdout = json.dumps(
        [
            {"Labels": {DISPATCH_ID_LABEL_KEY: "zeta-1"}},
            {"Labels": {DISPATCH_ID_LABEL_KEY: "alpha-2"}},
        ]
    )
    run_fn = _ScriptedRunFn([_FakeCompletedProcess(returncode=0, stdout=stdout)])
    reaper = PodmanContainerReaper(run_fn=run_fn)
    result = reaper.list_running()
    assert result == ["alpha-2", "zeta-1"]


def test_list_running_empty_array_returns_empty_list():
    # Case 10
    run_fn = _ScriptedRunFn([_FakeCompletedProcess(returncode=0, stdout="[]")])
    reaper = PodmanContainerReaper(run_fn=run_fn)
    assert reaper.list_running() == []


def test_list_running_skips_entries_missing_labels_or_dispatch_id():
    # Case 11: one entry missing Labels entirely, one with Labels={} (no
    # dispatch_id key) -- both skipped, no raise; any other valid entry in
    # the same batch is still returned.
    stdout = json.dumps(
        [
            {"Id": "no-labels-key"},
            {"Id": "empty-labels", "Labels": {}},
            {"Id": "valid", "Labels": {DISPATCH_ID_LABEL_KEY: "still-here"}},
        ]
    )
    run_fn = _ScriptedRunFn([_FakeCompletedProcess(returncode=0, stdout=stdout)])
    reaper = PodmanContainerReaper(run_fn=run_fn)
    assert reaper.list_running() == ["still-here"]


def test_list_running_nonzero_returncode_raises():
    # Case 12
    run_fn = _ScriptedRunFn(
        [_FakeCompletedProcess(returncode=1, stdout="", stderr="boom")]
    )
    reaper = PodmanContainerReaper(run_fn=run_fn)
    with pytest.raises(ContainerError):
        reaper.list_running()


def test_list_running_unparseable_json_raises():
    # Case 13
    run_fn = _ScriptedRunFn([_FakeCompletedProcess(returncode=0, stdout="not json")])
    reaper = PodmanContainerReaper(run_fn=run_fn)
    with pytest.raises(ContainerError):
        reaper.list_running()


def test_list_running_exact_argv():
    # Case 14: freeze the exact argv passed to run_fn.
    run_fn = _ScriptedRunFn([_FakeCompletedProcess(returncode=0, stdout="[]")])
    reaper = PodmanContainerReaper(run_fn=run_fn)
    reaper.list_running()
    assert run_fn.calls == [
        ["podman", "ps", "--filter", f"label={DISPATCH_ID_LABEL_KEY}", "--format", "json"]
    ]


def test_reap_zero_ids_is_silent_noop_single_call():
    # Case 15: lookup returns zero ids -> run_fn called exactly once (the
    # lookup), no second call, no exception.
    run_fn = _ScriptedRunFn([_FakeCompletedProcess(returncode=0, stdout="")])
    reaper = PodmanContainerReaper(run_fn=run_fn)
    reaper.reap("disp-none")
    assert len(run_fn.calls) == 1
    assert run_fn.calls[0] == [
        "podman",
        "ps",
        "-a",
        "-q",
        "--filter",
        f"label={DISPATCH_ID_LABEL_KEY}=disp-none",
    ]


def test_reap_one_id_completes_and_calls_rm_in_order():
    # Case 16
    run_fn = _ScriptedRunFn(
        [
            _FakeCompletedProcess(returncode=0, stdout="abc123\n"),
            _FakeCompletedProcess(returncode=0, stdout=""),
        ]
    )
    reaper = PodmanContainerReaper(run_fn=run_fn)
    reaper.reap("disp-1")
    assert run_fn.calls == [
        [
            "podman",
            "ps",
            "-a",
            "-q",
            "--filter",
            f"label={DISPATCH_ID_LABEL_KEY}=disp-1",
        ],
        ["podman", "rm", "-f", "--ignore", "abc123"],
    ]


def test_reap_two_ids_single_rm_call_with_both_ids_in_order():
    # Case 17: defensive multi-match -- the single rm call includes both
    # ids, in lookup-returned order.
    run_fn = _ScriptedRunFn(
        [
            _FakeCompletedProcess(returncode=0, stdout="id-one\nid-two\n"),
            _FakeCompletedProcess(returncode=0, stdout=""),
        ]
    )
    reaper = PodmanContainerReaper(run_fn=run_fn)
    reaper.reap("disp-multi")
    assert run_fn.calls[1] == ["podman", "rm", "-f", "--ignore", "id-one", "id-two"]


def test_reap_lookup_nonzero_returncode_raises_rm_never_called():
    # Case 18
    run_fn = _ScriptedRunFn(
        [_FakeCompletedProcess(returncode=1, stdout="", stderr="lookup failed")]
    )
    reaper = PodmanContainerReaper(run_fn=run_fn)
    with pytest.raises(ContainerError):
        reaper.reap("disp-bad-lookup")
    assert len(run_fn.calls) == 1


def test_reap_rm_nonzero_returncode_raises():
    # Case 19
    run_fn = _ScriptedRunFn(
        [
            _FakeCompletedProcess(returncode=0, stdout="abc123\n"),
            _FakeCompletedProcess(returncode=1, stdout="", stderr="rm failed"),
        ]
    )
    reaper = PodmanContainerReaper(run_fn=run_fn)
    with pytest.raises(ContainerError):
        reaper.reap("disp-bad-rm")


# --------------------------------------------------------------------------
# Bead .34 live tests: real PodmanContainerReaper against real podman.
# Cases 20-21.
# --------------------------------------------------------------------------


def _run_detached_live(profile, command, dispatch_id, timeout=30):
    """Launch a real, detached (backgrounded) worker container carrying
    ``dispatch_id``'s label, using the SAME frozen build_run_argv() the
    reaper's label convention depends on. build_run_argv's argv is frozen
    (no --detach token can be added to it), so a launch-only COPY is built
    here with --detach inserted right after "run" -- the reaper itself
    only ever sees the real, unmodified label token this produces."""
    base_argv = build_run_argv(profile, command=command, dispatch_id=dispatch_id)
    launch_argv = [*base_argv[:2], "--detach", *base_argv[2:]]
    return subprocess.run(
        launch_argv,
        env={**worker_env()},
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


@requires_podman
def test_live_reaper_lists_and_reaps_a_real_container(tmp_path, pinned_live_image):
    # Case 20: spawn one real, long-running container carrying a fixed
    # test dispatch_id; confirm a real PodmanContainerReaper().list_running()
    # (default run_fn, real podman) includes it; reap() it; confirm a
    # follow-up list_running() no longer shows it.
    dispatch_id = f"bead34-livetest-{uuid.uuid4().hex[:12]}"
    profile = _profile(tmp_path, image=pinned_live_image, timeout_seconds=120)
    reaper = PodmanContainerReaper()
    try:
        launch = _run_detached_live(profile, ["sleep", "30"], dispatch_id)
        assert launch.returncode == 0, launch.stderr

        running = reaper.list_running()
        assert dispatch_id in running

        reaper.reap(dispatch_id)

        running_after = reaper.list_running()
        assert dispatch_id not in running_after
    finally:
        # Defensive host cleanup: never leave a live container behind even
        # if an assertion above failed mid-test.
        reaper.reap(dispatch_id)


@requires_podman
def test_live_reap_second_call_on_already_removed_id_is_noop(tmp_path, pinned_live_image):
    # Case 21: live idempotence -- calling reap() a second time on an
    # already-removed dispatch_id does NOT raise (silent no-op, the same
    # code path as the fake-run_fn case 15, now proven against real
    # podman). Written as its own independent, self-contained test (own
    # freshly spawned+reaped container) rather than chaining off case 20's
    # leftover state, so it does not depend on pytest's execution order —
    # see sub-report for this deliberate deviation from the spec's literal
    # "reuse case 20's id" phrasing.
    dispatch_id = f"bead34-livetest-{uuid.uuid4().hex[:12]}"
    profile = _profile(tmp_path, image=pinned_live_image, timeout_seconds=120)
    reaper = PodmanContainerReaper()
    try:
        launch = _run_detached_live(profile, ["sleep", "30"], dispatch_id)
        assert launch.returncode == 0, launch.stderr

        assert dispatch_id in reaper.list_running()

        reaper.reap(dispatch_id)
        assert dispatch_id not in reaper.list_running()

        # Second reap on the now-already-removed id: silent no-op, no raise.
        reaper.reap(dispatch_id)
        assert dispatch_id not in reaper.list_running()
    finally:
        reaper.reap(dispatch_id)
