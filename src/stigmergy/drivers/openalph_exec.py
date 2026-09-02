"""openalph-exec worker driver adapter (bead .149 build spec §4.1).

A worker lane whose driver is the ``openalph exec`` subcommand (OA-side,
bead .149 §3) running inside the SAME hardened cage as
:mod:`stigmergy.drivers.claude_code`: same :func:`spawn` signature, same
:class:`~stigmergy.drivers.claude_code.DispatchResult` type, same
process-level failure parity (TimeoutExpired -> WEDGED, OSError launch ->
INFRA, entrypoint exit 69 -> INFRA, exit 137 + no parse -> WEDGED), and the
SAME git-bundle rule (reuse :func:`claude_code._create_work_bundle` via
sibling import — do not duplicate).

**What differs (by design, spec §1/§4.1):**

- The in-cage command is ``openalph exec ...`` (one single-turn
  Agent.handle_input, the rendered ``prompt.md`` passed VERBATIM as the
  ``--task-file`` task — spec §2 decision 4: prompt fidelity), not
  ``claude -p ...``.
- The credential env is the OpenAI-wire shim pair: ``OPENAI_API_KEY`` =
  the capability token (the relay authenticates the capability on every
  lane), ``OPENAI_BASE_URL`` = the in-cage loopback shim root
  (``worker_image/shim.py`` RELAY_PORT 18081, worker-facing path
  ``/chat/completions`` WITHOUT ``/v1``). No real provider key ever enters
  the worker env (AC4 parity with claude_code's token-only cred env).
  bead .162: the SAME ``env=`` channel also carries
  ``FILE_TICKET_TRANSPORT`` (the file_ticket builtin's JSONL sink path)
  and ``FILE_TICKET_MAX_FILINGS`` (the per-run filing count cap, OA
  BUILTIN default "8" — the driver never sees the charter) — non-credential
  worker-env additions; AC4 still holds (the ONLY secret value in the env
  is the capability token).
- The container gains ``--workdir=/work`` (spec §2 decision 1: workspace =
  ``/scratch`` (tmpfs HOME — the OA SessionLog/flight-recorder JSONLs die
  with the cage, never land in the clone), process cwd = ``/work`` (the
  clone) so OA file tools resolve relative paths into the git tree).
- The result JSON is ``openalph exec``'s ONE stdout line (spec §2 decision
  6: stdout discipline is load-bearing) and the classification table is
  spec §4.1's CLOSED vocabulary:

  - ``status="done"``: ``usage["out"] >= budgets.output_tokens`` ->
    FAILED/ceiling_trip="output_tokens" (claude_code parity); else DONE.
  - ``status="failed"`` + ``deny_reason`` in {quota-calls, quota-tokens} ->
    FAILED + ceiling_trip ("driver_turns"/"output_tokens") + detail prefix
    ``relay-deny:<reason>`` (spec §2 decision 2: quota denies are Stigmergy's
    OWN per-dispatch budget speaking — ceiling trips, NEVER INFRA; the .134
    provenance rides in the DISPATCH event detail).
  - ``status="failed"`` + any other ``deny_reason`` (unknown/revoked/
    missing-capability/forbidden-endpoint/...) -> FAILED, detail
    ``relay-deny:<reason>`` (a capability failure — consumes a rung attempt).
  - ``ceiling_trip == "driver_turns"`` (exec's iteration-cap sentinel,
    spec §2 decision 5) -> FAILED + that ceiling.
  - ``status="infra"`` (genuinely-forwarded upstream/transport failure —
    exec's own closed vocabulary, spec §3.8) -> INFRA, exec's detail
    carried.
  - ``status="wedged"`` -> WEDGED.
  - unknown status string -> FAILED (closed conservative vocabulary —
    NEVER inferred as INFRA).

This module, like :mod:`stigmergy.drivers.claude_code`, never imports
:mod:`stigmergy.records`/:mod:`stigmergy.charter`/:mod:`stigmergy.registry`
— the CALLER (`.21`'s prepare_dispatch, `.22`'s daemon) resolves the
registry (``worker_model = f"{entry.oa_provider_key}/{entry.version}"``,
spec §2 decision 7) and threads the result in.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stigmergy.container import ContainerProfile, build_run_argv, worker_env
from stigmergy.drivers.claude_code import (
    _CAGE_UNAVAILABLE_EXIT,
    _SUBPROCESS_TIMEOUT_SLACK_SECONDS,
    _WORK_BRANCH_REF,
    Budgets,
    DispatchResult,
    DispatchStatus,
    DriverError,
    _create_work_bundle,
    _ref_exists,
)
from stigmergy.relay import Capability

# The in-cage loopback shim root (spec §1 wire facts): the .63 worker
# entrypoint bridges http://127.0.0.1:18081 -> /run/relay.sock. Must match
# worker_image/shim.py's RELAY_PORT (tests pin the two against each other).
_CRED_BASE_URL = "http://127.0.0.1:18081"

# The OA agent name defined by the provisioned
# /etc/openalph/agents/stigmergy-worker.toml (spec §5).
_AGENT_NAME = "stigmergy-worker"

# v0 fixed worker tool inventory (spec §1 in-cage argv) — NOT
# charter-configurable (per-lane tool inventory is deferred, spec §8).
# Containship: same capabilities as claude_code's DEFAULT_ALLOWED_TOOLS,
# OA-native names. bead .162: the worker files discovered work through the
# SAME file_ticket builtin as the stations (one filing channel fleet-wide);
# the JSONL transport file lives in the /work mount the harvest reads.
_WORKER_TOOLS = "shell,file_read,file_write,file_edit,file_patch,glob,grep,file_ticket"

# bead .162: the file_ticket builtin's file-transport sink — the /work-mount
# path filing.harvest_worker_filings reads (must match "/work/" +
# filing.FILED_TICKETS_REL). NOTE: the driver still never sees the CHARTER
# (module contract) — the filing caps do not contradict that: they arrive
# via Budgets, which is the caller's (.21 prepare_dispatch) resolved thread
# of charter [loop.dispatch_limits].filed_tickets /
# .filed_ticket_bytes. So the in-cage tool caps EQUAL the harvest-side
# caps without the driver ever importing charter/registry:
#
# - FILE_TICKET_MAX_FILINGS <- Budgets.max_filings (str, always set — the
#   getattr default below only keeps older stub Budgets valid and matches
#   the OA BUILTIN default "8");
# - FILE_TICKET_MAX_BYTES <- Budgets.max_filing_bytes — set ONLY when the
#   value is not None; an ABSENT env var means the tool applies no
#   call-time size check (the OA tool default), leaving the harvest-side
#   size-cap as the backstop. (There is NO hardcoded default for the byte
#   cap on the driver side — a stub Budgets without the field gets an
#   absent env var, never a fabricated cap.)
_FILE_TICKET_TRANSPORT = "/work/.stigmergy/filed-tickets.json"

# spec §2 decision 1: process cwd = /work (the git clone) so OA's file
# tools resolve relative paths into the tree; workspace = /scratch is the
# OA agent config's own business (the toml), never the process cwd.
_WORKDIR = "/work"

# Relay deny reasons that are Stigmergy's OWN per-dispatch budget speaking
# (spec §2 decision 2: ceiling trips, NOT INFRA) — the exact analog of
# claude-code's error_max_turns: they consume a rung attempt (step up after
# attempts_per_rung, FailureClass.DEGENERATE). INFRA-retry would re-mint a
# fresh capability+budget and let an over-budget ticket retry forever.
_QUOTA_DENY_CEILINGS: dict[str, str] = {
    "quota-calls": "driver_turns",
    "quota-tokens": "output_tokens",
}


@dataclass(frozen=True)
class ModelConfig:
    """Driver-owned, caller-populated (.21) dispatch configuration.

    Mirrors :class:`claude_code.ModelConfig`'s container-profile fields so
    the cage construction is shared. ``model`` is the registry entry name
    (for records); ``worker_model`` is the in-cage string
    (``f"{entry.oa_provider_key}/{entry.version}"``, resolved by .21 —
    spec §2 decision 7); ``effort`` is the charter lane's card-native
    effort (charter validation: {none, low, medium, xhigh}). Deliberately
    does NOT import :mod:`stigmergy.charter` or :mod:`stigmergy.registry`.
    """

    model: str  # registry entry name (for records)
    worker_model: str  # in-cage --model string (provider_key/version)
    effort: str  # card-native: none | low | medium | xhigh
    image: str  # rig.image (digest-pinned; container.py enforces)
    relay_socket: Path | str | None = None  # passed straight through to build_run_argv
    egress_socket: Path | str | None = None  # passed straight through to build_run_argv
    scratch_size: str = "1g"
    pids_limit: int = 1024
    memory: str = "4g"
    cpus: str = "4"
    # matches charter DEFAULT_CHARTER loop.timers.dispatch_timeout_seconds
    timeout_seconds: int = 3600


# Injected executor: claude_code's RunOne contract (argv, env, timeout) ->
# CompletedProcess. Same default discipline (one subprocess.run, no shell).
RunOne = Callable[[list[str], dict[str, str], int], subprocess.CompletedProcess]


def _default_run_one(
    argv: list[str], env: dict[str, str], timeout: int
) -> subprocess.CompletedProcess:
    """The real single-attempt executor: one `podman run` invocation, no
    shell, capturing stdout/stderr as text (claude_code parity)."""
    return subprocess.run(  # noqa: S603
        argv,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _credential_env(capability: Capability) -> dict[str, str]:
    """The worker's credential env (spec §4.1): the capability token as
    ``OPENAI_API_KEY`` (the relay authenticates the capability on every
    lane — the token is the ONLY credential that ever enters the worker
    env, AC4) plus the in-cage shim base URL. This is the ONE path a
    credential takes into :func:`build_run_argv`'s ``env=`` parameter."""
    return {
        "OPENAI_API_KEY": capability.token,
        "OPENAI_BASE_URL": _CRED_BASE_URL,
    }


def _map_usage(raw: dict[str, Any]) -> dict[str, int]:
    """Map ``openalph exec``'s 4-key usage object onto the SPEC §8 shape
    VERBATIM — exec already emits ``{in, cached, out, reasoning}``
    (agent._last_turn_usage, missing keys -> zeros on the OA side). Every
    value is coerced non-negative (never raises, never negative, never
    fabricated) so the output already satisfies records.py's
    ``_validate_tokens`` invariant once the loop layer applies it."""
    out: dict[str, int] = {}
    for key in ("in", "cached", "out", "reasoning"):
        value = raw.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            value = 0
        out[key] = value
    return out


def _classify(
    result_obj: dict[str, Any], budgets: Budgets, usage: dict[str, int]
) -> tuple[DispatchStatus, str | None, str]:
    """Classify a parsed ``openalph exec`` result object (spec §4.1 closed
    table). Returns ``(status, ceiling_trip, detail)``.

    The table is implemented EXACTLY per spec §4.1 — see the module
    docstring for the full row list and the spec §2 decision-2 rationale
    (quota denies are ceiling trips, never INFRA; only genuinely-forwarded
    upstream failures classify INFRA). Unknown status strings resolve to
    FAILED: ``infra`` is a closed, conservative vocabulary; defaulting an
    unrecognized case to ``infra`` would let capability/quality failures
    silently walk free of rung attempts.
    """
    status = result_obj.get("status")
    if not isinstance(status, str):
        status = ""
    deny_reason = result_obj.get("deny_reason")
    if not isinstance(deny_reason, str) or not deny_reason:
        deny_reason = None
    detail_raw = result_obj.get("detail")
    detail = detail_raw if isinstance(detail_raw, str) and detail_raw else ""

    if status == "done":
        if usage["out"] >= budgets.output_tokens:
            return (
                DispatchStatus.FAILED,
                "output_tokens",
                f"output_tokens ceiling tripped: usage.out={usage['out']} >= "
                f"budgets.output_tokens={budgets.output_tokens}",
            )
        return DispatchStatus.DONE, None, "exec status=done"

    if status == "failed":
        if deny_reason in _QUOTA_DENY_CEILINGS:
            ceiling = _QUOTA_DENY_CEILINGS[deny_reason]
            return (
                DispatchStatus.FAILED,
                ceiling,
                f"relay-deny:{deny_reason}"
                + (f" — {detail}" if detail else ""),
            )
        if deny_reason is not None:
            # A capability failure (revoked/missing/forbidden/...): NOT our
            # budget — FAILED, consumes a rung attempt, provenance carried.
            return (
                DispatchStatus.FAILED,
                None,
                f"relay-deny:{deny_reason}" + (f" — {detail}" if detail else ""),
            )
        ceiling_trip = result_obj.get("ceiling_trip")
        if ceiling_trip == "driver_turns":
            # exec's iteration-cap sentinel (spec §2 decision 5): the
            # driver's own turn budget speaking — a ceiling trip.
            return (
                DispatchStatus.FAILED,
                "driver_turns",
                "exec ceiling_trip=driver_turns (iteration cap reached)",
            )
        return (
            DispatchStatus.FAILED,
            None,
            "exec status=failed" + (f" — {detail}" if detail else ""),
        )

    if status == "infra":
        # Genuinely-forwarded upstream/transport failure (no deny marker) —
        # the only path to INFRA on a parseable result.
        return DispatchStatus.INFRA, None, "exec status=infra" + (f" — {detail}" if detail else "")

    if status == "wedged":
        return (
            DispatchStatus.WEDGED,
            None,
            "exec status=wedged" + (f" — {detail}" if detail else ""),
        )

    return (
        DispatchStatus.FAILED,
        None,
        f"unrecognized exec status {status!r} (closed vocabulary — never infra)",
    )


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
    """One openalph-exec dispatch, start to finish (bead .149 build spec
    §4.1). SAME signature and parity semantics as
    :func:`stigmergy.drivers.claude_code.spawn`.

    Never raises for any *runtime* outcome (bad JSON, timeout, nonzero
    exit, relay deny) — those all become a :class:`DispatchResult`. Raises
    :class:`DriverError` only for a structural task-pack contract
    violation (missing ``prompt.md``), and lets
    :class:`stigmergy.container.ContainerError` propagate un-caught (an
    unpinned image is a build-time config bug that must fail loud).

    A work bundle is attempted regardless of the eventual dispatch status
    (reuses :func:`claude_code._create_work_bundle` — the "iff
    refs/heads/work exists" rule).
    """
    task_pack = Path(task_pack)
    work_clone = Path(work_clone)

    prompt_path = task_pack / "prompt.md"
    if not prompt_path.is_file():
        raise DriverError(
            f"task pack {task_pack} is missing prompt.md (bead .149 build spec §4.1) — "
            "a task-pack assembly bug, not a dispatch outcome"
        )
    if not isinstance(model_cfg.effort, str) or not model_cfg.effort:
        raise DriverError(
            f"openalph-exec ModelConfig.effort is empty ({model_cfg.effort!r}) — a "
            "caller bug: the in-cage --effort flag cannot be empty (card-native "
            "vocabulary: none/low/medium/xhigh)"
        )

    # bead .162: the worker files through the SAME file_ticket builtin as
    # the stations. The file-transport sink path + the per-run count cap
    # ride the cage env on the SAME channel as the credential pair. The cap
    # arrives via budgets.max_filings (charter [loop.dispatch_limits]
    # .filed_tickets, threaded through prepare_dispatch) so the tool-side
    # cap EQUALS the harvest-side count-cap; the getattr default keeps stub
    # Budgets objects valid and matches the OA BUILTIN default. The per-
    # filing byte cap rides the SAME channel (FILE_TICKET_MAX_BYTES from
    # budgets.max_filing_bytes <- charter .filed_ticket_bytes) and is set
    # ONLY when the value is not None: an absent env var means the tool
    # applies no call-time size check (OA tool default) and the harvest-
    # side size-cap stays the backstop.
    cage_env = {
        **_credential_env(capability),
        "FILE_TICKET_TRANSPORT": _FILE_TICKET_TRANSPORT,
        "FILE_TICKET_MAX_FILINGS": str(getattr(budgets, "max_filings", 8)),
    }
    _max_filing_bytes = getattr(budgets, "max_filing_bytes", None)
    if _max_filing_bytes is not None:
        cage_env["FILE_TICKET_MAX_BYTES"] = str(_max_filing_bytes)

    profile = ContainerProfile(
        image=model_cfg.image,
        work_clone=work_clone,
        task_pack=task_pack,
        scratch_size=model_cfg.scratch_size,
        pids_limit=model_cfg.pids_limit,
        memory=model_cfg.memory,
        cpus=model_cfg.cpus,
        timeout_seconds=model_cfg.timeout_seconds,
        # network left at ContainerProfile's own default ("none") — egress
        # goes through the mounted socket file, never through a network
        # namespace change (claude_code parity).
    )

    # spec §1 in-cage argv: the rendered prompt.md is passed VERBATIM as the
    # single --task-file task (spec §2 decision 4 — exec prepends no
    # preamble, never wraps/truncates).
    command = [
        "openalph",
        "exec",
        "--agent",
        _AGENT_NAME,
        "--task-file",
        "/task/prompt.md",
        "--model",
        model_cfg.worker_model,
        "--effort",
        model_cfg.effort,
        "--max-turns",
        str(budgets.driver_turns),
        "--tools",
        _WORKER_TOOLS,
    ]

    # ContainerError here (unpinned image) propagates uncaught — parity.
    argv = build_run_argv(
        profile,
        command=command,
        egress_socket=model_cfg.egress_socket,
        relay_socket=model_cfg.relay_socket,
        env=cage_env,
        dispatch_id=capability.dispatch_id,
        workdir=_WORKDIR,
    )

    executor: RunOne = run_one if run_one is not None else _default_run_one
    timeout = model_cfg.timeout_seconds + _SUBPROCESS_TIMEOUT_SLACK_SECONDS

    transcript = ""
    usage = _map_usage({})
    ceiling_trip: str | None = None
    status: DispatchStatus
    detail: str

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
        parsed: Any = None
        try:
            parsed = json.loads(result.stdout.strip())
        except (json.JSONDecodeError, ValueError):
            parsed = None
        if not isinstance(parsed, dict):
            # stdout discipline violation (spec §2 decision 6) or a kill:
            # the closed, conservative reading.
            if result.returncode == 137:
                status = DispatchStatus.WEDGED
                detail = "podman --timeout killed the container (exit 137), no parseable result"
            elif result.returncode == _CAGE_UNAVAILABLE_EXIT:
                status = DispatchStatus.INFRA
                detail = (
                    "worker cage egress setup failed — entrypoint fail-closed "
                    f"(exit {_CAGE_UNAVAILABLE_EXIT}); no dispatch ran"
                )
            else:
                status = DispatchStatus.FAILED
                detail = f"no parseable openalph-exec result JSON (exit {result.returncode})"
        else:
            raw_usage = parsed.get("usage")
            usage = _map_usage(raw_usage if isinstance(raw_usage, dict) else {})
            status, ceiling_trip, detail = _classify(parsed, budgets, usage)

    ref_present = _ref_exists(work_clone, _WORK_BRANCH_REF)
    bundle_ref = _create_work_bundle(work_clone, capability.dispatch_id, bundle_dir)
    if bundle_ref is None and ref_present:
        detail = f"{detail}; work bundle creation failed despite refs/heads/work existing"

    return DispatchResult(
        status=status,
        transcript=transcript,
        usage=usage,
        reported_cost_usd=None,  # openalph exec emits no native cost (relay JSONL is the meter)
        bundle_ref=bundle_ref,
        ceiling_trip=ceiling_trip,
        detail=detail,
    )
