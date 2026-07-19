"""Checks runner (SPEC.md §3 `check` station, §4 "Judgment-surface
hardening").

**Checks are arbitrary code execution — always contained** (SPEC §4). This
module runs named charter checks in a **fresh, no-network, no-credential
checker container** against an **immutable work tree**. The loop never runs
worker-authored code in its own context; the worker's container is killed
before its output is evaluated (that kill happens upstream of this module —
by the time :func:`run_check` runs, there is no worker container left).

Key properties, mechanically enforced here:

- **Fresh container per run.** Every attempt (first try and every flake
  rerun) is a brand-new `podman run` via :func:`stigmergy.container.
  build_run_argv` — never a reused or restarted container.
- **No egress at all.** The checker profile pins `network="none"` — stricter
  than a worker (which may get a proxied netns in a later ticket). No
  provider API keys, no capability tokens: :func:`stigmergy.container.
  worker_env` only carries what rootless podman itself needs
  (`XDG_RUNTIME_DIR`/`DBUS_SESSION_BUS_ADDRESS`); nothing is forwarded into
  the container (no `--env-host`, no secret mounts) by
  :func:`~stigmergy.container.build_run_argv`.
- **Immutability.** The source ``work_tree`` is never mounted into a
  container and never mutated. Each attempt copies it into a fresh temp
  directory (``shutil.copytree``) and mounts *that copy* at `/work` rw — a
  check that writes temp files corrupts nothing but its own disposable
  copy, and copies never bleed across attempts or across checks.
- **Flake protocol** (SPEC §3/§9, charter `loop.retries.flake_reruns`): a
  failing first attempt is re-run up to ``flake_reruns`` times in fresh
  containers. Pass-on-rerun is recorded as the **distinct** `flaky` outcome
  — never silently folded into `pass`. Never-passes is `fail`. These reruns
  consume wall-clock only; this module has no notion of dispatch budgets or
  attempt counters and never touches any.
- **Structured output only.** :class:`CheckResult` is data — check name,
  outcome, the exit code of every attempt in order, and captured output
  (bounded to the last ``_OUTPUT_CAP_BYTES`` bytes) — suitable as retry
  evidence handed to the next dispatch (SPEC §4 "check output → retry pack:
  data framing", never free text, never instructions).

**Error vs. fail.** `FAIL`/`FLAKY`/`PASS` all mean the checker container
actually ran the command to completion and returned an exit code. If
running an attempt itself blows up before producing an exit code (the
source tree is missing, the temp copy can't be made, `podman` isn't
invokable, the subprocess call errors out) that attempt — and the whole
check — is reported as `ERROR`, distinct from a check that ran and failed.
"""

from __future__ import annotations

import enum
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from stigmergy.container import ContainerProfile, build_run_argv, worker_env

# Bound captured output to avoid unbounded memory (SPEC §4: retry evidence
# is data, kept small). Keeps the *last* bytes — failures/tracebacks land
# at the end of a check's output, not the head.
_OUTPUT_CAP_BYTES = 4096


@dataclass(frozen=True)
class CheckResources:
    """Checker-container resource bounds (bead .91: charter-configurable —
    see :func:`stigmergy.charter.resolve_check_resources`). Field defaults
    ARE the historical hardcoded bounds, so a caller that never passes
    ``resources`` explicitly gets exactly the pre-.91 behavior."""

    timeout_seconds: int = 60
    memory: str = "256m"
    cpus: str = "1"
    scratch_size: str = "64m"
    pids_limit: int = 64


DEFAULT_CHECK_RESOURCES = CheckResources()

# Backstop above the podman-level --timeout so a hung `podman run` itself
# (not just the containerized command) cannot hang this process forever.
_SUBPROCESS_TIMEOUT_BACKSTOP_SECONDS = 30

RunOne = Callable[..., tuple[int, str]]


class CheckError(Exception):
    """Raised by the default runner on a preflight failure (missing/invalid
    ``work_tree``, or any other condition that prevents even attempting the
    check) — caught by :func:`run_check` and reported as `CheckOutcome.ERROR`.
    """


class CheckOutcome(enum.Enum):
    PASS = "pass"
    FAIL = "fail"
    FLAKY = "flaky"
    ERROR = "error"


@dataclass(frozen=True)
class CheckResult:
    """One check's structured, immutable result — retry evidence, not text.

    ``runs`` is the exit code of each attempt, in order (first try first,
    then any flake reruns). ``output`` is the concatenated captured output
    of all attempts, truncated to the last ``_OUTPUT_CAP_BYTES`` bytes.
    """

    name: str
    outcome: CheckOutcome
    runs: tuple[int, ...]
    output: str
    wall_time_seconds: float


def _truncate(text: str, cap: int = _OUTPUT_CAP_BYTES) -> str:
    """Keep the last ``cap`` bytes of ``text`` (UTF-8), decoding safely
    across a byte boundary that lands mid multi-byte character."""
    data = text.encode("utf-8", errors="replace")
    if len(data) <= cap:
        return text
    return data[-cap:].decode("utf-8", errors="replace")


def _default_run_one(
    command: str,
    work_tree: Path | str,
    *,
    image: str,
    resources: CheckResources = DEFAULT_CHECK_RESOURCES,
) -> tuple[int, str]:
    """The real single-attempt executor: fresh copy of ``work_tree``, fresh
    no-network checker container, via :func:`build_run_argv`.

    Copies ``work_tree`` into a brand-new temp directory (never mounts the
    source itself), mounts the copy at `/work` rw and an empty scratch
    directory at `/task` ro (checks carry no task pack), runs ``command``
    under `sh -c` with `network="none"`, and always discards the temp
    copies afterward — regardless of outcome, so reruns never accumulate
    disk and the source tree is never touched.

    The checker container reuses the worker image, whose ENTRYPOINT is the
    fail-closed egress gatekeeper (it exits 69 if `/run/egress.sock` is
    absent). A checker is `network="none"` with no egress mount by design,
    so the gatekeeper would ALWAYS fail closed and no check could ever run
    (bead .87). We therefore bypass it: `entrypoint_override="sh"` +
    `command=["-c", command]` runs the check command directly. `build_run_argv`
    forbids that override on any egress-capable container, so it can never
    be used to escape the gatekeeper on a real worker. `HOME=/scratch` is
    set (the writable tmpfs) so a real check tool (post-.79 pytest/ruff)
    has a writable home under the `--read-only` rootfs — the same reason
    the worker entrypoint sets it (bead .85), which the bypass skips.
    """
    source = Path(work_tree)
    if not source.is_dir():
        raise CheckError(f"work_tree {source!r} is not a directory")

    run_root = Path(tempfile.mkdtemp(prefix="stigmergy-check-"))
    try:
        work_copy = run_root / "work"
        task_pack = run_root / "task"
        shutil.copytree(source, work_copy)
        task_pack.mkdir(parents=True, exist_ok=True)

        profile = ContainerProfile(
            image=image,
            work_clone=work_copy,
            task_pack=task_pack,
            scratch_size=resources.scratch_size,
            pids_limit=resources.pids_limit,
            memory=resources.memory,
            cpus=resources.cpus,
            timeout_seconds=resources.timeout_seconds,
            network="none",
        )
        argv = build_run_argv(
            profile,
            command=["-c", command],
            entrypoint_override="sh",
            env={"HOME": "/scratch"},
        )
        result = subprocess.run(  # noqa: S603
            argv,
            env=worker_env(),
            capture_output=True,
            text=True,
            timeout=resources.timeout_seconds + _SUBPROCESS_TIMEOUT_BACKSTOP_SECONDS,
            check=False,
        )
        return result.returncode, (result.stdout + result.stderr)
    finally:
        shutil.rmtree(run_root, ignore_errors=True)


def run_check(
    name: str,
    command: str,
    work_tree: Path | str,
    *,
    image: str,
    flake_reruns: int,
    resources: CheckResources = DEFAULT_CHECK_RESOURCES,
    run_one: RunOne | None = None,
) -> CheckResult:
    """Run one named check to a final :class:`CheckResult`, applying the
    flake protocol (SPEC §3/§9).

    First attempt, then up to ``flake_reruns`` reruns while the most recent
    attempt keeps failing (exit code != 0) — stops the moment an attempt
    exits 0. ``run_one`` defaults to the real containerized single-attempt
    executor; tests inject a scripted stand-in so the flake protocol itself
    is exercised deterministically, with no container involved. ``resources``
    (bead .91, charter-configurable via :func:`stigmergy.charter.
    resolve_check_resources`) is passed through to every attempt unchanged —
    it defaults to :data:`DEFAULT_CHECK_RESOURCES` (the legacy hardcoded
    bounds) so an unset caller behaves exactly as before .91.

    Outcome:
      - first attempt exits 0 -> `PASS`
      - first attempt fails, some later attempt exits 0 -> `FLAKY`
        (never silently reported as `PASS`)
      - every attempt fails -> `FAIL`
      - ``run_one`` raises before producing an exit code (preflight or
        infra failure) -> `ERROR`, immediately, without treating the
        failure as one more flake-protocol attempt
    """
    if flake_reruns < 0:
        raise ValueError(f"flake_reruns must be >= 0 (got {flake_reruns})")

    executor: RunOne = run_one if run_one is not None else _default_run_one

    runs: list[int] = []
    output_parts: list[str] = []
    start = time.monotonic()

    for attempt in range(flake_reruns + 1):
        try:
            exit_code, output = executor(command, work_tree, image=image, resources=resources)
        except Exception as exc:  # noqa: BLE001 - any raise here is an ERROR, not a fail
            output_parts.append(f"[attempt {attempt + 1}] run_one raised: {exc!r}")
            return CheckResult(
                name=name,
                outcome=CheckOutcome.ERROR,
                runs=tuple(runs),
                output=_truncate("\n".join(output_parts)),
                wall_time_seconds=time.monotonic() - start,
            )
        runs.append(exit_code)
        output_parts.append(output)
        if exit_code == 0:
            break

    wall_time_seconds = time.monotonic() - start

    if runs[0] == 0:
        outcome = CheckOutcome.PASS
    elif 0 in runs:
        outcome = CheckOutcome.FLAKY
    else:
        outcome = CheckOutcome.FAIL

    return CheckResult(
        name=name,
        outcome=outcome,
        runs=tuple(runs),
        output=_truncate("\n".join(output_parts)),
        wall_time_seconds=wall_time_seconds,
    )


def run_checks(
    checks: dict[str, str],
    work_tree: Path | str,
    *,
    image: str,
    flake_reruns: int,
    resources: Mapping[str, CheckResources] | None = None,
) -> list[CheckResult]:
    """Run every named check in ``checks`` (name -> shell command),
    preserving iteration order, and return one :class:`CheckResult` per
    check.

    Each check gets its own call to :func:`run_check` — its own fresh
    work-tree copy and fresh container(s), never shared with any other
    check or with any other attempt. ``resources`` (bead .91) maps check
    name -> :class:`CheckResources`; a check absent from the map (or the
    map itself being ``None``) falls back to :data:`DEFAULT_CHECK_RESOURCES`.
    """
    return [
        run_check(
            name,
            command,
            work_tree,
            image=image,
            flake_reruns=flake_reruns,
            resources=(resources or {}).get(name, DEFAULT_CHECK_RESOURCES),
        )
        for name, command in checks.items()
    ]
