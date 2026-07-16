"""Hardened worker container profile (SPEC.md §4 worker containment, §3
`provision` station).

Every worker is assumed compromised (SPEC §4 threat model). This module is
the mechanical enforcement of the containment contract:

- :func:`build_run_argv` renders one :class:`ContainerProfile` into an exact
  `podman run` argv — rootless, all capabilities dropped, default seccomp
  retained, read-only rootfs, cgroup limits, a whole-cgroup timeout, and
  **exactly** the normative mount table (SPEC §4): the per-dispatch clone at
  `/work` (rw), a size-capped scratch tmpfs at `/scratch` (rw), the task pack
  at `/task` (ro), nothing else — unless the egress ticket's (.11) optional
  `egress_socket` is given, in which case exactly one more bind is appended:
  the dispatch's egress-proxy unix socket at `/run/egress.sock` (rw). The
  charter cannot relax this in v0 — there is no code path here that takes a
  caller override for the forbidden flags (`--privileged`, `--cap-add*`,
  `seccomp=unconfined`, any `--*=host` namespace share); they simply never
  get emitted.
- :func:`worker_env` returns the environment rootless podman needs to talk
  to the user's systemd/dbus instance so cgroup v2 limits are actually
  enforced (else podman silently falls back to cgroupfs and the limits in
  the argv above become theater). Every `subprocess.run([...podman...])`
  call in this module, and every caller's live invocation, must pass this
  as `env`.
- :func:`build_image` is the `provision`-station image build (SPEC §3):
  supply-chain rule is digest-pinned base images only, checked by parsing
  the Containerfile *before* any `podman build` runs (so the reject path
  needs no podman at all), then a no-secret build, returning the built
  image's resolved digest.

**Supply-chain rule (SPEC §3/§4):** :class:`ContainerProfile.image` and every
`FROM` base in a built Containerfile must be digest-pinned (`@sha256:...`).
A `:tag` reference is rejected — tags are mutable, digests are not.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

_DIGEST_RE = re.compile(r"@sha256:[0-9a-fA-F]{64}")


class ContainerError(Exception):
    """Raised on any worker-containment or supply-chain violation: an
    unpinned image reference, an unpinned Containerfile `FROM`, or a
    `podman` invocation that fails."""


@dataclass(frozen=True)
class ContainerProfile:
    """One dispatch's hardened container profile (SPEC §4).

    ``work_clone`` and ``task_pack`` are stored exactly as given (``Path``
    or ``str``); :func:`build_run_argv` resolves them to absolute paths at
    argv-construction time. ``network`` defaults to ``"none"`` — the .10
    baseline is no network at all; the egress ticket (.11) is the only thing
    that overrides it, with the internal netns routed through the proxy.

    The egress ticket (.11) does not add a field here — its socket path is
    passed as :func:`build_run_argv`'s separate ``egress_socket`` keyword,
    not stored on the profile, since it is a per-invocation mount detail
    (the dispatch's runtime-dir socket path), not a durable profile knob.
    """

    image: str
    work_clone: Path | str
    task_pack: Path | str
    scratch_size: str
    pids_limit: int
    memory: str
    cpus: str
    timeout_seconds: int
    network: str = "none"


def _require_pinned(image: str) -> None:
    if not _DIGEST_RE.search(image):
        raise ContainerError(
            f"image ref {image!r} is not digest-pinned (@sha256:...) — "
            "supply-chain rule (SPEC §3/§4) rejects mutable tags"
        )


def build_run_argv(
    profile: ContainerProfile,
    *,
    command: list[str],
    egress_socket: str | Path | None = None,
) -> list[str]:
    """Render ``profile`` into an exact `podman run` argv (SPEC §4).

    Emits precisely the mandatory hardening flags, precisely the normative
    mount table (two `--volume=` binds + one size-capped `--tmpfs=`), the
    image ref, then ``command``. Raises :class:`ContainerError` if
    ``profile.image`` is not digest-pinned. Never emits `--privileged`,
    `--cap-add*`, `seccomp=unconfined`, or any host-namespace share — the
    charter cannot relax this in v0, so there is no parameter that could
    smuggle one in.

    ``egress_socket`` (bead .11) is the ONE additional mount this function
    can ever add: when given, exactly one more `--volume=` bind is appended
    — ``--volume=<egress_socket>:/run/egress.sock:rw`` — the dispatch's
    egress-proxy unix socket, the worker's sole path out of its
    `--network=none` cage. Nothing else about the argv changes. Left at its
    default ``None``, the returned argv is byte-identical to the pre-.11
    argv (regression guard: existing callers/tests are unaffected).
    """
    _require_pinned(profile.image)

    work = str(Path(profile.work_clone).resolve())
    task = str(Path(profile.task_pack).resolve())

    argv = [
        "podman",
        "run",
        "--rm",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--read-only",
        f"--pids-limit={profile.pids_limit}",
        f"--memory={profile.memory}",
        f"--cpus={profile.cpus}",
        f"--timeout={profile.timeout_seconds}",
        f"--network={profile.network}",
        f"--volume={work}:/work:rw",
        f"--volume={task}:/task:ro",
        f"--tmpfs=/scratch:rw,size={profile.scratch_size},nosuid,nodev",
    ]
    if egress_socket is not None:
        argv.append(f"--volume={egress_socket}:/run/egress.sock:rw")
    argv.append(profile.image)
    argv.extend(command)
    return argv


def worker_env() -> dict[str, str]:
    """Environment for invoking `podman` so rootless cgroup v2 limits are
    actually enforced (not silently dropped to cgroupfs).

    Starts from a copy of the current process environment (so `PATH` and
    everything else a subprocess needs is intact) and overlays
    `XDG_RUNTIME_DIR`/`DBUS_SESSION_BUS_ADDRESS` pointed at this user's
    runtime dir and session bus — required for rootless podman to talk to
    the user's systemd instance where the delegated cgroup controllers
    (cpu, memory, pids) live.
    """
    env = dict(os.environ)
    uid = os.getuid()
    env["XDG_RUNTIME_DIR"] = f"/run/user/{uid}"
    env["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path=/run/user/{uid}/bus"
    return env


_FROM_RE = re.compile(r"^\s*FROM\s+(.+)$", re.IGNORECASE)
_AS_RE = re.compile(r"^(.*?)\s+AS\s+(\S+)\s*$", re.IGNORECASE)


def _parse_from_bases(containerfile_text: str) -> list[tuple[str, str | None]]:
    """Parse every `FROM` line's base image ref and optional stage alias.

    Returns a list of ``(base, alias)`` tuples in file order. Flag tokens
    (`--platform=...`) preceding the base are skipped. Comment and blank
    lines are ignored. Does not handle line continuations — not needed by
    any Containerfile this module is asked to parse.
    """
    bases: list[tuple[str, str | None]] = []
    for raw_line in containerfile_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _FROM_RE.match(line)
        if not match:
            continue
        rest = match.group(1).strip()

        alias: str | None = None
        as_match = _AS_RE.match(rest)
        if as_match:
            rest = as_match.group(1).strip()
            alias = as_match.group(2).strip()

        tokens = [t for t in rest.split() if not t.startswith("--")]
        if not tokens:
            continue
        base = tokens[0]
        bases.append((base, alias))
    return bases


def _check_pinned_bases(containerfile_text: str) -> None:
    """Raise :class:`ContainerError` if any `FROM` base is not digest-pinned.

    A `FROM <alias>` referencing an earlier build stage is exempt (it is
    not a fetched base image); every other `FROM` must carry `@sha256:`.
    """
    known_aliases: set[str] = set()
    for base, alias in _parse_from_bases(containerfile_text):
        if base not in known_aliases and not _DIGEST_RE.search(base):
            raise ContainerError(
                f"Containerfile FROM {base!r} is not digest-pinned (@sha256:...) — "
                "supply-chain rule (SPEC §3 provision) rejects mutable tags"
            )
        if alias:
            known_aliases.add(alias)


def build_image(
    containerfile_dir: Path | str,
    tag: str,
    *,
    no_secrets: bool = True,
) -> str:
    """Build the `provision`-station worker image (SPEC §3).

    Reads ``<containerfile_dir>/Containerfile``, rejects it with
    :class:`ContainerError` — before invoking `podman` at all — if any
    `FROM` base image is not digest-pinned (multi-stage: a `FROM
    <earlier-stage-alias>` is exempt, every fetched base is not). Then
    builds with `podman build`, tags the result, and returns the built
    image's resolved digest (a string containing `"sha256:"`).

    ``no_secrets`` defaults to (and can only be) ``True``: this build never
    mounts or passes credentials — no `--secret`, no credential build-args.
    Passing ``no_secrets=False`` raises :class:`ContainerError` rather than
    silently building with a relaxed, secret-bearing invocation.
    """
    if not no_secrets:
        raise ContainerError(
            "build_image only supports no-secret builds (SPEC §3 provision "
            "supply-chain rule) — no_secrets=False is refused"
        )

    containerfile_dir = Path(containerfile_dir)
    containerfile_path = containerfile_dir / "Containerfile"
    containerfile_text = containerfile_path.read_text()
    _check_pinned_bases(containerfile_text)

    env = worker_env()

    subprocess.run(  # noqa: S603
        [
            "podman",
            "build",
            "--file",
            str(containerfile_path),
            "--tag",
            tag,
            str(containerfile_dir),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    return _resolve_image_digest(tag, env)


def _resolve_image_digest(tag: str, env: dict[str, str]) -> str:
    """Resolve a built image's content digest via a fallback chain of
    `podman inspect` formats, never hand-copying any captured value —
    each subprocess result is used directly, in Python, as bytes.

    Local-only builds (never pushed/pulled) typically leave `.Digest` and
    `.RepoDigests` empty, so this falls through to `.Id` (the image config
    digest), normalizing it to carry the `sha256:` prefix if the podman
    version reports it bare.
    """
    formats = ("{{.Digest}}", "{{index .RepoDigests 0}}", "{{.Id}}")
    for fmt in formats:
        result = subprocess.run(  # noqa: S603
            ["podman", "inspect", "--format", fmt, tag],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            continue
        value = result.stdout.strip()
        if not value:
            continue
        if "sha256:" not in value:
            value = f"sha256:{value}"
        if "sha256:" in value:
            return value

    raise ContainerError(f"could not resolve a digest for built image {tag!r}")
