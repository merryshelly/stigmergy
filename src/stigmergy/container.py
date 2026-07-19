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
  get emitted. Bead .13's optional `env` mapping is the only way env vars
  ever reach the worker — one `--env=KEY=VALUE` token per entry, sorted by
  key, appended right before the image ref; `env=None` (default) leaves the
  argv byte-identical to before. Bead .34's optional `dispatch_id` tags the
  container with the `DISPATCH_ID_LABEL_KEY` label (SPEC §9 crash recovery)
  — one `--label=stigmergy.dispatch_id=<value>` token, inserted immediately
  after `--network=...` and before the first `--volume=...`; `dispatch_id=
  None` (default) leaves the argv byte-identical to before.
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
- :class:`PodmanContainerReaper` (bead .34, SPEC §9 crash recovery) is the
  real `stigmergy.recover.ContainerReaper` implementation: `list_running()`
  maps every running, `DISPATCH_ID_LABEL_KEY`-labeled container back to its
  `dispatch_id` via `podman ps --filter label=... --format json`; `reap()`
  looks up (any state, `-a`) then removes every matching container in one
  `podman rm -f --ignore ...` call — idempotent, silent no-op when nothing
  matches.

**Supply-chain rule (SPEC §3/§4):** :class:`ContainerProfile.image` and every
`FROM` base in a built Containerfile must be digest-pinned (`@sha256:...`).
A `:tag` reference is rejected — tags are mutable, digests are not.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

_DIGEST_RE = re.compile(r"@sha256:[0-9a-fA-F]{64}")
# A bare content digest / local image ID (`sha256:<64hex>`). A locally-BUILT
# worker image (bead .63) has no RepoDigests, so `build_image` returns — and
# the daemon runs it by — its immutable `sha256:` id; that is as content-
# addressed and immutable as a registry `@sha256:` ref. Only mutable tags
# (`name:tag`) are rejected.
_BARE_DIGEST_RE = re.compile(r"sha256:[0-9a-fA-F]{64}")

# Bead .34 (SPEC §4 worker containment, §9 crash recovery): the label key a
# running worker container is tagged with so a real `ContainerReaper`
# (`PodmanContainerReaper`, below) can map a running podman container back
# to the `dispatch_id` that spawned it. `recover.py`'s `ContainerReaper`
# Protocol has no real implementation without this convention.
DISPATCH_ID_LABEL_KEY = "stigmergy.dispatch_id"


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


# podman non-registry transports whose source is mutable/local — a decorative
# `@sha256:` substring on one of these does NOT pin anything (bead .63 review,
# codex-sol-xhigh: `dir:/tmp/x@sha256:<64hex>` bypassed the digest guard live).
_IMAGE_TRANSPORT_PREFIXES = (
    "dir:",
    "docker-archive:",
    "docker-daemon:",
    "oci:",
    "oci-archive:",
    "containers-storage:",
    "tarball:",
    "sif:",
)


def _require_pinned(image: str) -> None:
    # A bare content digest / local image id (`sha256:<64hex>`, bead .63 built
    # images) is immutable + content-addressed.
    if _BARE_DIGEST_RE.fullmatch(image):
        return
    # Reject non-registry transports outright: their source is mutable/local and
    # an `@sha256:` on them is decorative, not a pin.
    if image.startswith(_IMAGE_TRANSPORT_PREFIXES):
        raise ContainerError(
            f"image ref {image!r} uses a non-registry transport — refused "
            "(supply-chain rule SPEC §3/§4; a digest on a mutable/local source "
            "is not a pin)"
        )
    # Otherwise require a registry ref pinned by digest (`name@sha256:...`).
    if _DIGEST_RE.search(image):
        return
    raise ContainerError(
        f"image ref {image!r} is not digest-pinned (@sha256:... or a bare "
        "sha256:<digest>) — supply-chain rule (SPEC §3/§4) rejects mutable tags"
    )


def _require_valid_env_keys(env: Mapping[str, str]) -> None:
    """Raise :class:`ContainerError` if any key in ``env`` is empty or
    contains ``=`` (bead .13 build spec §0.2) — cheap defense against a
    malformed mapping silently producing a garbled `--env=` flag. Checked
    BEFORE any argv token is appended, so a rejected call never leaks a
    partial argv list."""
    for key in env:
        if not key or "=" in key:
            raise ContainerError(
                f"env key {key!r} is invalid — keys must be non-empty and "
                "contain no '=' (bead .13 build spec §0.2)"
            )


def _require_valid_dispatch_id(dispatch_id: str | None) -> None:
    """Raise :class:`ContainerError` if ``dispatch_id`` is not ``None`` and
    is invalid (bead .34 build spec §Part A) — cheap defense against a
    malformed value producing a garbled `--label=` flag or smuggling an
    extra token into it. Checked BEFORE any argv token is appended, so a
    rejected call never leaks a partial argv list. Names which specific
    constraint failed in the raised message."""
    if dispatch_id is None:
        return
    if not dispatch_id:
        raise ContainerError("dispatch_id is invalid — must not be empty")
    if "=" in dispatch_id:
        raise ContainerError(f"dispatch_id {dispatch_id!r} is invalid — must not contain '='")
    if any(ch.isspace() for ch in dispatch_id):
        raise ContainerError(
            f"dispatch_id {dispatch_id!r} is invalid — must not contain whitespace"
        )
    if "," in dispatch_id:
        raise ContainerError(f"dispatch_id {dispatch_id!r} is invalid — must not contain ','")


def _require_entrypoint_override_safe(
    entrypoint_override: str | None,
    network: str,
    egress_socket: str | Path | None,
    relay_socket: str | Path | None,
) -> None:
    """Raise :class:`ContainerError` if ``entrypoint_override`` is combined
    with any egress-capable configuration (bead .87). The only sanctioned use
    of an entrypoint override is a CHECK container — ``network="none"``, no
    egress/relay socket — which has no governed-egress cage, so displacing the
    worker image's fail-closed egress-gatekeeper entrypoint (``worker_image/
    entrypoint.sh``) removes nothing. Overriding the entrypoint on a container
    that DOES have a netns/egress path would bypass that gatekeeper — refused
    here so it is mechanically unrepresentable, not merely discouraged. Checked
    BEFORE any argv token is appended, so a rejected call never leaks a partial
    argv list."""
    if entrypoint_override is None:
        return
    if network != "none":
        raise ContainerError(
            f"entrypoint_override is only permitted on a network=none container "
            f"(got network={network!r}) — overriding the entrypoint on an "
            "egress-capable container would bypass the fail-closed egress "
            "gatekeeper (bead .87)"
        )
    if egress_socket is not None:
        raise ContainerError(
            "entrypoint_override must not be combined with an egress_socket — a "
            "gatekeeper-bypassed container with an egress path is forbidden (bead .87)"
        )
    if relay_socket is not None:
        raise ContainerError(
            "entrypoint_override must not be combined with a relay_socket — a "
            "gatekeeper-bypassed container with a relay path is forbidden (bead .87)"
        )


def build_run_argv(
    profile: ContainerProfile,
    *,
    command: list[str],
    egress_socket: str | Path | None = None,
    relay_socket: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    dispatch_id: str | None = None,
    entrypoint_override: str | None = None,
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

    ``relay_socket`` (bead .63) mirrors ``egress_socket`` exactly: when
    given, exactly one more `--volume=` bind is appended —
    ``--volume=<relay_socket>:/run/relay.sock:rw`` — the dispatch's
    credential-relay unix socket, the SECOND (and last) socket the
    in-container shim bridges (``ANTHROPIC_BASE_URL`` -> relay). It is
    emitted immediately AFTER the ``egress_socket`` volume (frozen,
    deterministic order) and before any ``env`` tokens. Left at its default
    ``None``, the returned argv is byte-identical to the pre-.63 argv (the
    daemon supplying this value is ``.25``'s job).

    ``env`` (bead .13 build spec §0.2) is the ONE way env vars ever reach
    the worker — no `--env-host` is ever emitted, so the worker's env is
    exactly whatever the image sets plus exactly these caller-supplied
    entries. When given, one `--env=KEY=VALUE` token is emitted per entry,
    **sorted by key** (determinism), appended immediately before the image
    ref — after the ``egress_socket`` volume (if any), the same "last thing
    before the image" slot that append already uses. Each key is validated
    (non-empty, no `=`) — see :func:`_require_valid_env_keys` — raising
    :class:`ContainerError` before any token is appended. Values are passed
    through verbatim as ONE argv token each; no shell is ever involved
    anywhere in this module, so no value-side escaping question exists.
    Left at its default ``None``, the returned argv is byte-identical to
    the pre-.13 argv (regression guard, exactly the discipline
    ``egress_socket=None`` already gets).

    ``dispatch_id`` (bead .34 build spec, Part A) tags the container so a
    real :class:`ContainerReaper` (:class:`PodmanContainerReaper`, this
    module) can map a running podman container back to the dispatch that
    spawned it (SPEC §9 crash recovery). When given, exactly one more argv
    token is emitted — ``f"--label={DISPATCH_ID_LABEL_KEY}={dispatch_id}"``
    — inserted immediately after the `--network=...` token and immediately
    before the first `--volume=...` token (frozen position — every other
    additive token, `egress_socket`'s volume and `env`'s `--env=` tokens,
    stays in its own already-documented slot, unaffected). ``dispatch_id``
    is validated (non-empty, no `=`, no whitespace, no `,`) — see
    :func:`_require_valid_dispatch_id` — raising :class:`ContainerError`
    before any token is appended. Left at its default ``None``, the
    returned argv is byte-identical to the pre-.34 argv (regression guard,
    exactly the discipline ``egress_socket=None``/``env=None`` already
    get).

    ``entrypoint_override`` (bead .87) is the ONLY thing that can displace
    the image's own ENTRYPOINT. When given, exactly one
    ``--entrypoint=<value>`` token is emitted immediately before the image
    ref (after any ``env`` tokens); the caller must pass a ``command`` tail
    that matches (e.g. ``entrypoint_override="sh"`` with ``command=["-c",
    "<cmd>"]``). Its ONLY sanctioned use is a CHECK container
    (:mod:`stigmergy.checks`), which runs ``network="none"`` with no
    egress/relay socket: bypassing the worker image's fail-closed
    egress-gatekeeper entrypoint removes nothing, because a no-network
    container has no governed egress to gate. To make that a mechanical
    invariant rather than a convention, :func:`_require_entrypoint_override_safe`
    raises :class:`ContainerError` if ``entrypoint_override`` is combined
    with ``network != "none"`` or with an ``egress_socket``/``relay_socket``
    — a gatekeeper-bypassed container that also has an egress path is
    unrepresentable. Left at its default ``None``, the returned argv is
    byte-identical to the pre-.87 argv (the worker spawn path passes
    nothing and is unaffected).
    """
    _require_pinned(profile.image)
    if env is not None:
        _require_valid_env_keys(env)
    _require_valid_dispatch_id(dispatch_id)
    _require_entrypoint_override_safe(
        entrypoint_override, profile.network, egress_socket, relay_socket
    )

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
    ]
    if dispatch_id is not None:
        argv.append(f"--label={DISPATCH_ID_LABEL_KEY}={dispatch_id}")
    argv.append(f"--volume={work}:/work:rw")
    argv.append(f"--volume={task}:/task:ro")
    argv.append(f"--tmpfs=/scratch:rw,size={profile.scratch_size},nosuid,nodev")
    if egress_socket is not None:
        argv.append(f"--volume={egress_socket}:/run/egress.sock:rw")
    if relay_socket is not None:
        argv.append(f"--volume={relay_socket}:/run/relay.sock:rw")
    if env is not None:
        for key in sorted(env):
            argv.append(f"--env={key}={env[key]}")
    if entrypoint_override is not None:
        argv.append(f"--entrypoint={entrypoint_override}")
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


def _default_podman_run_fn(argv: list[str]) -> subprocess.CompletedProcess:
    """The real subprocess seam for :class:`PodmanContainerReaper`: one
    `subprocess.run` call using :func:`worker_env` (bead .34 build spec,
    Part C) — the same rootless cgroup-delegation environment every other
    podman invocation in this module requires. Never raises on a nonzero
    exit (`check=False`) — callers inspect `.returncode` themselves."""
    return subprocess.run(  # noqa: S603
        argv,
        env=worker_env(),
        capture_output=True,
        text=True,
        check=False,
    )


class PodmanContainerReaper:
    """Real `stigmergy.recover.ContainerReaper` implementation (SPEC §9
    crash recovery; bead .34 build spec, Part C). Protocol duck-typed, not
    inherited — mirrors `recover.py`'s own injection-seam discipline.

    Maps a running/lingering podman container back to the `dispatch_id` it
    was spawned for via the :data:`DISPATCH_ID_LABEL_KEY` label that
    :func:`build_run_argv` (bead .34, Part A) now stamps onto every
    dispatch's container. Every subprocess call goes through an injected
    ``run_fn`` seam (default: :func:`_default_podman_run_fn`) so tests
    never shell out to real podman except in the explicitly-marked
    `@requires_podman` live test.
    """

    def __init__(
        self,
        *,
        run_fn: Callable[[list[str]], subprocess.CompletedProcess] | None = None,
    ) -> None:
        self._run_fn = run_fn if run_fn is not None else _default_podman_run_fn

    def list_running(self) -> list[str]:
        """Return the `dispatch_id`s of every currently-RUNNING (never
        `-a`) worker container — matches the `ContainerReaper` Protocol's
        "currently-running" contract exactly.

        Raises :class:`ContainerError` if the `podman ps` call itself
        fails (nonzero return code) or its `stdout` is not parseable JSON.
        An empty result (`[]`) is not an error. Any entry missing/malformed
        `Labels`/the dispatch-id label is skipped defensively (should not
        happen given the filter) rather than raised on. Returns the
        sorted, deduplicated list.
        """
        result = self._run_fn(
            ["podman", "ps", "--filter", f"label={DISPATCH_ID_LABEL_KEY}", "--format", "json"]
        )
        if result.returncode != 0:
            raise ContainerError(
                f"podman ps (list_running) failed with exit {result.returncode}: {result.stderr}"
            )
        try:
            entries = json.loads(result.stdout)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ContainerError(
                f"podman ps (list_running) returned unparseable JSON: {exc}"
            ) from exc

        dispatch_ids: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            labels = entry.get("Labels", {})
            if not isinstance(labels, dict):
                continue
            value = labels.get(DISPATCH_ID_LABEL_KEY)
            if isinstance(value, str) and value:
                dispatch_ids.add(value)
        return sorted(dispatch_ids)

    def reap(self, dispatch_id: str) -> None:
        """Kill and remove every container (ANY state — `created`,
        `paused`, `stopping`, `running` — a non-running container is still
        a double-dispatch risk) tagged with ``dispatch_id``.

        Step 1 (lookup): zero matches is a silent no-op — idempotent,
        `run_fn` is invoked exactly once, `rm` is never called. Raises
        :class:`ContainerError` if the lookup call itself fails.

        Step 2 (only if step 1 found >=1 id): removes ALL matched ids in
        ONE `podman rm -f --ignore ...` invocation (in lookup-returned
        order) — `--ignore` makes a benign `--rm`-flag auto-removal race
        between the lookup and this call exit 0 rather than a fatal error.
        Raises :class:`ContainerError` if this call fails.
        """
        lookup = self._run_fn(
            [
                "podman",
                "ps",
                "-a",
                "-q",
                "--filter",
                f"label={DISPATCH_ID_LABEL_KEY}={dispatch_id}",
            ]
        )
        if lookup.returncode != 0:
            raise ContainerError(
                f"podman ps (reap lookup) failed with exit {lookup.returncode}: {lookup.stderr}"
            )

        ids = [line for line in lookup.stdout.splitlines() if line.strip()]
        if not ids:
            return

        removal = self._run_fn(["podman", "rm", "-f", "--ignore", *ids])
        if removal.returncode != 0:
            raise ContainerError(
                f"podman rm (reap) failed with exit {removal.returncode}: {removal.stderr}"
            )


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
    """Resolve a built image's RUNNABLE content id via a fallback chain of
    `podman inspect` formats, never hand-copying any captured value — each
    subprocess result is used directly, in Python, as bytes.

    `build_image` only ever produces LOCAL provision builds (never pushed),
    and the returned ref must be one the daemon can `podman run` directly.
    `.Id` (the image config digest / local image id) is that ref — always
    present and always runnable for a just-built image — normalized to carry
    the `sha256:` prefix if podman reports it bare. It is tried FIRST: `.Digest`
    (and `.RepoDigests`) can be populated with the *manifest* digest, which on
    some podman versions (e.g. 5.4.2 populates `.Digest` on local builds) is
    NOT a runnable bare reference (`podman run sha256:<manifest>` → "image not
    known") — returning it would hand the daemon an image ref that fails every
    dispatch (bead .93). The manifest forms remain as last-resort fallbacks
    only for the degenerate case where `.Id` is somehow unavailable.
    """
    formats = ("{{.Id}}", "{{index .RepoDigests 0}}", "{{.Digest}}")
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
