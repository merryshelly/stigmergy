"""claude-code driver adapter (SPEC.md §7 driver interface, §9 failure
classes, §10 AC7; bead .13 build spec).

**Scope (bead .13 build spec §0.1, confirmed with advisor).** This module is
purely: given an already-assembled ``task_pack`` (dir, `/task`-mount-ready,
carrying a rendered ``prompt.md`` — see §0.3), an already-prepared
``work_clone`` (a git worktree with a local ``work`` branch the worker
commits to — see §0.4), a resolved :class:`ModelConfig`, an already-minted
:class:`~stigmergy.relay.Capability`, and per-dispatch :class:`Budgets` ->
run ONE claude-code dispatch inside the hardened container
(:mod:`stigmergy.container`) and return a :class:`DispatchResult`. Lane
selection, task-pack assembly, worker naming, and capability minting are
.21's job. ``capability_store.revoke()`` at dispatch end is .22's job
(daemon-level teardown) — never called from inside :func:`spawn`.

**Credential delivery (bead .12/.13).** :func:`stigmergy.relay.
worker_credential_env` returns an env-var-shaped mapping (`ANTHROPIC_API_KEY`
/ `ANTHROPIC_BASE_URL`); this module is the ONLY thing that ever passes it
into :func:`stigmergy.container.build_run_argv`'s ``env=`` parameter — the
one and only path a real or capability credential ever takes into the
worker's process environment (SPEC §4 credentials "out of the worker
namespace entirely", AC4).

**transcript is raw; sealing is .22's job (§0.5).** :attr:`DispatchResult.
transcript` is the RAW captured stdout text — never redacted, never
content-addressed here. SPEC §7/§8's ``transcript_ref`` (a redacted,
content-addressed blob reference) is produced by a later loop-wiring layer
via ``record_plane.seal_transcript(result.transcript, redactor=relay.
build_redactor(credential_relay.injected_secrets(capability.token)),
must_not_contain=frozenset({capability.token, <real key>}))``. This module
never imports :mod:`stigmergy.records`, :mod:`stigmergy.charter`, or
:mod:`stigmergy.registry` (deliberate scope boundary, build spec §0.1/§0.5).

**bundle_ref is a security seam (§0.6, AC6).** ``work_clone`` is a repo a
hostile worker had write access to, so every git invocation this module
makes against it uses the exact, already-shipped discipline from
:mod:`stigmergy.weaver`/:mod:`stigmergy.rangereport`: `-c
core.hooksPath=/dev/null` unconditionally, on every call, no exceptions. A
bundle is created iff `refs/heads/work` exists — any commits at all —
**regardless of dispatch status**: a ceiling-tripped or even an infra
dispatch may still have committed real, useful work before hitting the
wall, and that work is never discarded just because the dispatch's own
classification was not `done`. If the ref is absent, ``bundle_ref`` is
``None`` (a normal case — a dispatch that never got as far as committing
anything — not a bug). If bundle *creation itself* fails (corrupt repo, git
error), ``bundle_ref`` is ``None`` and a note is appended to ``detail`` —
this function never raises and never changes ``status``.
"""

from __future__ import annotations

import enum
import json
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stigmergy.container import ContainerProfile, build_run_argv, worker_env
from stigmergy.relay import Capability, worker_credential_env

# v0 fixed (not charter-configurable — charter.py's _KNOWN_LANE_KEYS has no
# allowed-tools override key). Container containment is the real boundary
# (SPEC §7); this is defense-in-depth only.
DEFAULT_ALLOWED_TOOLS: tuple[str, ...] = ("Bash", "Edit", "Write", "Read", "Glob", "Grep")

# Matches weaver.py's `_BUNDLE_REF` exactly — same name, same ref, so the
# bundle this module produces is exactly what weaver.py already knows how
# to fetch (bead .13 build spec §0.4).
_WORK_BRANCH_REF = "refs/heads/work"

# Backstop above podman's own `--timeout`, mirrors checks.py's
# `_SUBPROCESS_TIMEOUT_SECONDS` discipline: a hung `podman run` itself (not
# just the containerized command) must not hang this process forever.
_SUBPROCESS_TIMEOUT_SLACK_SECONDS = 30

# Closed, conservative infra-marker vocabulary (bead .13 build spec §1.2).
# Unknown subtypes/markers never resolve to `infra` — see `_classify`.
_INFRA_MARKERS: tuple[str, ...] = (
    "rate_limit",
    "rate limit",
    "429",
    "529",
    "overloaded",
    "internal_server_error",
    "internal server error",
    "502",
    "503",
    "504",
    "connection refused",
    "connect timeout",
    "temporarily unavailable",
)


class DriverError(Exception):
    """Raised on a structural task-pack/work-clone contract violation
    (missing ``prompt.md``) — a caller bug (.21 assembled the pack wrong),
    never a dispatch outcome. Never raised for anything that happens
    during or after the actual container run — those become
    :class:`DispatchResult` statuses, not exceptions."""


class DispatchStatus(enum.Enum):
    DONE = "done"
    FAILED = "failed"
    WEDGED = "wedged"
    INFRA = "infra"


@dataclass(frozen=True)
class ModelConfig:
    """Driver-owned, caller-populated (.21) dispatch configuration.

    Deliberately does NOT import :mod:`stigmergy.charter` or
    :mod:`stigmergy.registry` (mirrors `relay.py`'s "keep it a plain
    parameter" discipline) — .21 resolves the charter/registry and builds
    this.
    """

    model: str  # --model value (registry entry name / API model id)
    image: str  # rig.image (digest-pinned; container.py enforces this)
    relay_base_url: str  # passed straight through to relay.worker_credential_env
    egress_socket: Path | str | None = None  # passed straight through to build_run_argv
    scratch_size: str = "1g"
    pids_limit: int = 1024
    memory: str = "4g"
    cpus: str = "4"
    # matches charter DEFAULT_CHARTER loop.timers.dispatch_timeout_seconds
    timeout_seconds: int = 3600


@dataclass(frozen=True)
class Budgets:
    output_tokens: int  # <- charter [loop.dispatch_limits].output_tokens
    driver_turns: int  # <- charter [loop.dispatch_limits].driver_turns; also --max-turns


@dataclass(frozen=True)
class DispatchResult:
    status: DispatchStatus
    transcript: str  # RAW captured stdout (module docstring §0.5) — never sealed/redacted here
    usage: dict[str, int]  # {"in":, "cached":, "out":, "reasoning":} — all 4 keys, never negative
    reported_cost_usd: float | None  # native total_cost_usd if present/parseable, else None
    bundle_ref: str | None  # path to the created git bundle, or None (module docstring §0.6)
    ceiling_trip: str | None  # "output_tokens" | "driver_turns" | None — distinct flag (SPEC §9)
    detail: str  # short, non-secret-bearing human-readable classification reason


# Injected executor, mirrors checks.py's RunOne pattern exactly.
RunOne = Callable[[list[str], dict[str, str], int], subprocess.CompletedProcess]


def _default_run_one(
    argv: list[str], env: dict[str, str], timeout: int
) -> subprocess.CompletedProcess:
    """The real single-attempt executor: one `podman run` invocation, no
    shell, capturing stdout/stderr as text."""
    return subprocess.run(  # noqa: S603
        argv,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _git(repo: Path, args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run one git command against ``repo``, mirroring `weaver._git`'s/
    `rangereport._git`'s discipline exactly: `-c core.hooksPath=/dev/null`
    on every invocation unconditionally (AC6 — ``work_clone`` is a repo a
    hostile worker had write access to), text output, captured."""
    argv = ["git", "-C", str(repo), "-c", "core.hooksPath=/dev/null", *args]
    return subprocess.run(argv, capture_output=True, text=True, check=check)  # noqa: S603


def _ref_exists(repo: Path, ref: str) -> bool:
    """True iff ``ref`` resolves to an object — checked without raising via
    `rev-parse --verify --quiet` (a non-zero exit here is a normal "ref
    absent" answer, not a plumbing failure). Same hooks-disabled discipline
    as every other git call in this module."""
    result = _git(repo, ["rev-parse", "--verify", "--quiet", ref], check=False)
    return result.returncode == 0


def _extract_result_object(stdout: str) -> dict[str, Any] | None:
    """Tolerant parse of claude-code's `--output-format json` stdout (bead
    .13 build spec §1.4).

    ``json.loads(stdout.strip())``. If the result is a **dict** carrying
    `"is_error"` or `{"type": "result"}`, return it. If it's a **list**,
    scan from the end for the last dict with `"type": "result"` and return
    that (some claude-code versions emit an array of session events —
    the last `result`-typed element is the authoritative terminal one).
    Anything else (parse failure, empty, neither shape) -> ``None``.
    """
    try:
        parsed = json.loads(stdout.strip())
    except (json.JSONDecodeError, ValueError):
        return None

    if isinstance(parsed, dict):
        if "is_error" in parsed or parsed.get("type") == "result":
            return parsed
        return None

    if isinstance(parsed, list):
        for item in reversed(parsed):
            if isinstance(item, dict) and item.get("type") == "result":
                return item
        return None

    return None


def _coerce_nonneg_int(raw: dict[str, Any], key: str) -> int:
    value = raw.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _map_usage(raw: dict[str, Any]) -> dict[str, int]:
    """Map claude-code's native usage object onto the SPEC §8 4-key shape
    (bead .13 build spec §1.3):

    - ``in`` = ``input_tokens`` + ``cache_creation_input_tokens`` (both are
      full-price-ish input; SPEC's "in" field has no separate
      creation/reasoning slot).
    - ``cached`` = ``cache_read_input_tokens`` (the discounted reused-cache
      reads).
    - ``out`` = ``output_tokens``.
    - ``reasoning`` = 0 (v0: claude-code's usage object carries no distinct
      reasoning-token count; fixed at 0, never fabricated as anything
      else).

    Every value: ``raw.get(key, 0)``, coerced to 0 if not a non-negative
    ``int`` (bool excluded) — never raises, never negative, matches
    ``records.py``'s ``_validate_tokens`` invariant exactly (this module
    doesn't call ``records.py``, but its output must already satisfy that
    invariant once the loop layer does).
    """
    return {
        "in": _coerce_nonneg_int(raw, "input_tokens") + _coerce_nonneg_int(
            raw, "cache_creation_input_tokens"
        ),
        "cached": _coerce_nonneg_int(raw, "cache_read_input_tokens"),
        "out": _coerce_nonneg_int(raw, "output_tokens"),
        "reasoning": 0,
    }


def _classify(
    result_obj: dict[str, Any], budgets: Budgets, usage: dict[str, int]
) -> tuple[DispatchStatus, str | None, str]:
    """Classify a parsed claude-code result object (bead .13 build spec
    §1.2). Returns ``(status, ceiling_trip, detail)``.

    ``is_error`` defaults to ``True`` if missing/non-bool (fail-closed: a
    malformed field is never treated as a clean success). ``subtype``
    defaults to ``""`` if missing/non-str.

    - ``subtype == "error_max_turns"`` -> FAILED, ``ceiling_trip=
      "driver_turns"``. Known CLI quirk (SPEC §7's cited GH #19498 gap):
      this subtype can appear with ``is_error: false`` — subtype is checked
      BEFORE/independent of ``is_error`` for this specific case, per
      SPEC's explicit "subtype ... is authoritative" instruction.
    - ``subtype == "success" and is_error is False`` -> check the
      output_tokens ceiling (SPEC §9): ``usage["out"] >=
      budgets.output_tokens`` -> FAILED, ``ceiling_trip="output_tokens"``.
      Else -> DONE, ``ceiling_trip=None``.
    - ``subtype == "error_during_execution"`` -> look for an
      infra-flavored marker (case-insensitive substring match against the
      fixed, closed vocabulary in :data:`_INFRA_MARKERS`) in
      ``result_obj.get("result")`` (if it's a str). Match -> INFRA. No
      match -> FAILED (a genuine execution bug, not a provider hiccup).
    - Anything else (unknown subtype, ``error_max_budget_usd``,
      ``error_max_structured_output_retries``, ``is_error=True`` with
      ``subtype="success"`` contradiction, missing/malformed fields) ->
      FAILED, ``ceiling_trip=None``. Unknown subtypes never resolve to
      `infra` — `infra` is a closed, conservative vocabulary; defaulting an
      unrecognized case to `infra` would let capability/quality failures
      silently walk free of rung attempts and poison the routing corpus.
    """
    is_error = result_obj.get("is_error")
    if not isinstance(is_error, bool):
        is_error = True

    subtype = result_obj.get("subtype")
    if not isinstance(subtype, str):
        subtype = ""

    if subtype == "error_max_turns":
        return (
            DispatchStatus.FAILED,
            "driver_turns",
            "claude-code returned subtype=error_max_turns (--max-turns ceiling hit)",
        )

    if subtype == "success" and is_error is False:
        if usage["out"] >= budgets.output_tokens:
            return (
                DispatchStatus.FAILED,
                "output_tokens",
                f"output_tokens ceiling tripped: usage.out={usage['out']} >= "
                f"budgets.output_tokens={budgets.output_tokens}",
            )
        return DispatchStatus.DONE, None, "subtype=success, is_error=false"

    if subtype == "error_during_execution":
        result_text = result_obj.get("result")
        if isinstance(result_text, str):
            lowered = result_text.lower()
            if any(marker in lowered for marker in _INFRA_MARKERS):
                return (
                    DispatchStatus.INFRA,
                    None,
                    "error_during_execution matched an infra-flavored marker",
                )
        return (
            DispatchStatus.FAILED,
            None,
            "error_during_execution with no infra marker — genuine execution bug",
        )

    return (
        DispatchStatus.FAILED,
        None,
        f"unrecognized/malformed result (subtype={subtype!r}, is_error={is_error!r})",
    )


def _create_work_bundle(
    work_clone: Path | str, dispatch_id: str, bundle_dir: Path | str | None
) -> str | None:
    """Create a git bundle of :data:`_WORK_BRANCH_REF` out of
    ``work_clone`` (bead .13 build spec §1.5/§0.6). See module docstring
    for the "iff the ref exists, regardless of status" rule and the AC6
    hooks-disabled discipline.

    ``bundle_dir`` defaults to a fresh ``mkdtemp()`` if omitted — the
    tempdir is deliberately NOT cleaned up by this function (the bundle
    file must outlive the call); the caller owns moving/copying the result
    into a durable location and eventually cleaning up the temp directory.

    Never raises: returns ``None`` if `refs/heads/work` is absent, or if
    the `git bundle create` call itself fails (corrupt repo, git error).
    """
    repo = Path(work_clone)
    if not _ref_exists(repo, _WORK_BRANCH_REF):
        return None

    if bundle_dir is not None:
        resolved_bundle_dir = Path(bundle_dir)
    else:
        resolved_bundle_dir = Path(tempfile.mkdtemp(prefix="stigmergy-bundle-"))
    resolved_bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = resolved_bundle_dir / f"{dispatch_id}.bundle"

    try:
        _git(repo, ["bundle", "create", str(bundle_path), _WORK_BRANCH_REF], check=True)
    except subprocess.CalledProcessError:
        return None

    return str(bundle_path)


def spawn(
    task_pack: Path | str,
    work_clone: Path | str,
    model_cfg: ModelConfig,
    capability: Capability,
    budgets: Budgets,
    *,
    run_one: RunOne | None = None,
    bundle_dir: Path | str | None = None,
) -> DispatchResult:
    """One claude-code dispatch, start to finish (bead .13 build spec
    §1.1).

    Never raises for any *runtime* outcome (bad JSON, timeout, nonzero
    exit, provider error) — those all become a :class:`DispatchResult`.
    Raises :class:`DriverError` only for a structural task-pack contract
    violation (missing ``prompt.md``), and lets
    :class:`stigmergy.container.ContainerError` propagate un-caught (an
    unpinned image is a build-time config bug, not a runtime infra
    condition — SPEC's `infra` class is for *transient* provider/proxy/
    relay failures, not "the charter is misconfigured"; that must fail
    loud, not infra-backoff-loop forever).

    A work bundle is attempted regardless of the eventual dispatch status
    (see :func:`_create_work_bundle` / module docstring §0.6) — a
    ceiling-tripped, wedged, or infra dispatch may still have committed
    real work before hitting the wall.
    """
    task_pack = Path(task_pack)
    work_clone = Path(work_clone)

    prompt_path = task_pack / "prompt.md"
    if not prompt_path.is_file():
        raise DriverError(
            f"task pack {task_pack} is missing prompt.md (bead .13 build spec §0.3) — "
            "a task-pack assembly bug, not a dispatch outcome"
        )
    prompt_text = prompt_path.read_text(encoding="utf-8")

    cred_env = worker_credential_env(capability, base_url=model_cfg.relay_base_url)

    profile = ContainerProfile(
        image=model_cfg.image,
        work_clone=work_clone,
        task_pack=task_pack,
        scratch_size=model_cfg.scratch_size,
        pids_limit=model_cfg.pids_limit,
        memory=model_cfg.memory,
        cpus=model_cfg.cpus,
        timeout_seconds=model_cfg.timeout_seconds,
        # network left at ContainerProfile's own default ("none") —
        # egress goes through the mounted socket file, never through a
        # network namespace change (mirrors .11's model exactly).
    )

    command = [
        "claude",
        "-p",
        prompt_text,
        "--output-format",
        "json",
        "--allowedTools",
        ",".join(DEFAULT_ALLOWED_TOOLS),
        "--permission-mode",
        "acceptEdits",
        "--max-turns",
        str(budgets.driver_turns),
        "--model",
        model_cfg.model,
    ]

    # ContainerError here (unpinned image) propagates uncaught — see
    # docstring above.
    argv = build_run_argv(
        profile, command=command, egress_socket=model_cfg.egress_socket, env=cred_env
    )

    executor: RunOne = run_one if run_one is not None else _default_run_one
    timeout = model_cfg.timeout_seconds + _SUBPROCESS_TIMEOUT_SLACK_SECONDS

    transcript = ""
    usage = _map_usage({})
    reported_cost_usd: float | None = None
    ceiling_trip: str | None = None

    try:
        result = executor(argv, worker_env(), timeout)
    except subprocess.TimeoutExpired:
        status = DispatchStatus.WEDGED
        detail = "subprocess backstop timeout exceeded — podman itself did not return"
    except OSError as exc:
        status = DispatchStatus.INFRA
        detail = f"could not launch dispatch container: {exc!r}"
    else:
        transcript = result.stdout
        result_obj = _extract_result_object(result.stdout)
        if result_obj is None:
            if result.returncode == 137:
                status = DispatchStatus.WEDGED
                detail = "podman --timeout killed the container (exit 137), no parseable result"
            else:
                status = DispatchStatus.FAILED
                detail = f"no parseable claude-code result JSON (exit {result.returncode})"
        else:
            raw_usage = result_obj.get("usage")
            usage = _map_usage(raw_usage if isinstance(raw_usage, dict) else {})
            status, ceiling_trip, detail = _classify(result_obj, budgets, usage)

            cost = result_obj.get("total_cost_usd")
            if isinstance(cost, int | float) and not isinstance(cost, bool):
                reported_cost_usd = cost

    ref_present = _ref_exists(work_clone, _WORK_BRANCH_REF)
    bundle_ref = _create_work_bundle(work_clone, capability.dispatch_id, bundle_dir)
    if bundle_ref is None and ref_present:
        detail = f"{detail}; work bundle creation failed despite refs/heads/work existing"

    return DispatchResult(
        status=status,
        transcript=transcript,
        usage=usage,
        reported_cost_usd=reported_cost_usd,
        bundle_ref=bundle_ref,
        ceiling_trip=ceiling_trip,
        detail=detail,
    )
