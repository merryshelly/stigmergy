"""Stigmergy command-line interface.

Top-level argparse dispatcher. v0 subcommand tree:

    stigmergy rig new --charter <path> [--path <base-dir>]
    stigmergy daemon run --rig <name> [--rigs-root <path>]
    stigmergy approve   <ticket-id> --rig <name> [--rigs-root <path>] --agent <a>
        --operator-session <s> [--json]
    stigmergy unapprove <ticket-id> --rig <name> [--rigs-root <path>] --agent <a>
        --operator-session <s>
    stigmergy resume    <ticket-id> --rig <name> [--rigs-root <path>] --agent <a>
        --operator-session <s>
    stigmergy reject    <filed-id>  --rig <name> [--rigs-root <path>] --agent <a>
        --operator-session <s> [--reason <text>]
    stigmergy promote   <filed-id>  --rig <name> [--rigs-root <path>] --spec <path|->
    stigmergy status       --rig <name> [--rigs-root <path>]
    stigmergy monitor      --rig <name> [--rigs-root <path>] [--tail N]
    stigmergy tickets      --rig <name> [--rigs-root <path>]
    stigmergy filed        --rig <name> [--rigs-root <path>]
    stigmergy ticket <id>  --rig <name> [--rigs-root <path>]
    stigmergy range-report --rig <name> [--rigs-root <path>] [--base-ref <ref>] [--critic]

Bare invocation (no args) prints short usage and returns 0 — preserved
from the v0 stub so existing scripts/tests calling `main([])` keep working.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from stigmergy import approval, checks, egress, spend, triage
from stigmergy.charter import Charter, CharterError, resolve_check_resources
from stigmergy.container import PodmanContainerReaper
from stigmergy.critic import Critic
from stigmergy.critic_client import (
    DEFAULT_MAX_TOKENS,
    make_critic_client,
    make_range_critic_client,
)
from stigmergy.daemon import Daemon, DaemonError
from stigmergy.filing import file_proposals
from stigmergy.keyprovider import make_op_key_provider
from stigmergy.notify import NotificationStore, NtfyNotifier, make_ntfy_sender
from stigmergy.rangereport import (
    RangeCritic,
    RangeCriticResult,
    RangeReport,
    RangeReportError,
    compute_range_report,
)
from stigmergy.records import EventType, RecordPlane, make_event
from stigmergy.registry import PricingClass, UnbudgetableError
from stigmergy.relay import CapabilityStore, CredentialRelay
from stigmergy.relay_transport import RelayHandle, make_urllib_forwarder, start_relay
from stigmergy.rig import ResolvedRig, RigError, RigStore, create_rig, resolve_rig
from stigmergy.spend import Budgets, SpendLeash
from stigmergy.statemachine import ESCALATED, POOL, transition
from stigmergy.status import (
    gather_status,
    render_daemon_liveness,
    render_event_tail,
    render_status,
    render_ticket_detail,
    render_ticket_list,
)
from stigmergy.steering import derive_steering
from stigmergy.weaver import Weaver

_USAGE = (
    "stigmergy v0 — usage: stigmergy <command> ...\n"
    "  rig new --charter <path> [--path <base-dir>]\n"
    "  daemon run --rig <name> [--rigs-root <path>]\n"
    "  approve <ticket-id> --rig <name> [--rigs-root <path>] --agent <a>"
    " --operator-session <s> [--json]\n"
    "  unapprove <ticket-id> --rig <name> [--rigs-root <path>] --agent <a>"
    " --operator-session <s>\n"
    "  resume <ticket-id> --rig <name> [--rigs-root <path>] --agent <a>"
    " --operator-session <s>\n"
    "  reject <filed-id> --rig <name> [--rigs-root <path>] --agent <a>"
    " --operator-session <s> [--reason <text>]\n"
    "  promote <filed-id> --rig <name> [--rigs-root <path>] --spec <path|->\n"
    "  ticket new --rig <name> [--rigs-root <path>] --spec <path|->\n"
    "  intake --rig <name> [--rigs-root <path>] --manifest <path>\n"
    "  status --rig <name> [--rigs-root <path>]\n"
    "  monitor --rig <name> [--rigs-root <path>] [--tail N]\n"
    "  tickets --rig <name> [--rigs-root <path>]\n"
    "  filed --rig <name> [--rigs-root <path>]\n"
    "  ticket <id> --rig <name> [--rigs-root <path>]\n"
    "  range-report --rig <name> [--rigs-root <path>] [--base-ref <ref>] [--critic]"
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stigmergy", description="Stigmergy orchestration harness."
    )
    subparsers = parser.add_subparsers(dest="command")

    rig_parser = subparsers.add_parser("rig", help="Rig lifecycle commands.")
    rig_subparsers = rig_parser.add_subparsers(dest="rig_command")

    rig_new_parser = rig_subparsers.add_parser("new", help="Scaffold a new rig from a charter.")
    rig_new_parser.add_argument(
        "--charter", required=True, help="Path to the rig's charter.toml."
    )
    rig_new_parser.add_argument(
        "--path",
        default=None,
        help="Base directory to create the rig under (default: ~/rigs).",
    )

    daemon_parser = subparsers.add_parser("daemon", help="Loop daemon commands.")
    daemon_subparsers = daemon_parser.add_subparsers(dest="daemon_command")

    daemon_run_parser = daemon_subparsers.add_parser(
        "run", help="Run the loop daemon for an already-scaffolded rig."
    )
    daemon_run_parser.add_argument(
        "--rig", required=True, help="Name of the rig to run (rig_root's directory name)."
    )
    daemon_run_parser.add_argument(
        "--rigs-root",
        default=None,
        help="Base directory the rig lives under (default: ~/rigs).",
    )

    def add_rig_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--rig", required=True, help="Name of the rig.")
        subparser.add_argument(
            "--rigs-root",
            default=None,
            help="Base directory the rig lives under (default: ~/rigs).",
        )

    def add_attribution_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--agent", required=True, help="Acting agent (the v0 audit line, D1)."
        )
        subparser.add_argument(
            "--operator-session", required=True, help="Operator session id (the v0 audit line)."
        )

    approve_parser = subparsers.add_parser("approve", help="Approve a ticket for dispatch.")
    approve_parser.add_argument("ticket_id", help="Ticket id to approve.")
    add_rig_args(approve_parser)
    add_attribution_args(approve_parser)
    approve_parser.add_argument(
        "--json", action="store_true", help="Print the steering dict as JSON."
    )

    unapprove_parser = subparsers.add_parser("unapprove", help="Withdraw a ticket's approval.")
    unapprove_parser.add_argument("ticket_id", help="Ticket id to unapprove.")
    add_rig_args(unapprove_parser)
    add_attribution_args(unapprove_parser)

    resume_parser = subparsers.add_parser("resume", help="Resume an escalated ticket.")
    resume_parser.add_argument("ticket_id", help="Ticket id to resume (must be ESCALATED).")
    add_rig_args(resume_parser)
    add_attribution_args(resume_parser)

    reject_parser = subparsers.add_parser("reject", help="Reject (tombstone) a filed proposal.")
    reject_parser.add_argument("filed_id", help="Filed proposal id to reject.")
    add_rig_args(reject_parser)
    add_attribution_args(reject_parser)
    reject_parser.add_argument("--reason", default=None, help="Optional rejection reason.")

    promote_parser = subparsers.add_parser(
        "promote", help="Complete a filed proposal into an unapproved ticket."
    )
    promote_parser.add_argument("filed_id", help="Filed proposal id to promote.")
    add_rig_args(promote_parser)
    promote_parser.add_argument(
        "--spec", required=True, help="Path to a JSON promotion spec, or '-' for stdin."
    )

    status_parser = subparsers.add_parser("status", help="Print the AC12 status snapshot.")
    add_rig_args(status_parser)

    monitor_parser = subparsers.add_parser("monitor", help="Print status and recent events.")
    add_rig_args(monitor_parser)
    monitor_parser.add_argument(
        "--tail", type=int, default=25, help="Number of recent events to display (default: 25)."
    )

    tickets_parser = subparsers.add_parser("tickets", help="List all tickets.")
    add_rig_args(tickets_parser)

    filed_parser = subparsers.add_parser("filed", help="List untriaged filed proposals.")
    add_rig_args(filed_parser)

    ticket_parser = subparsers.add_parser("ticket", help="Show one ticket's full detail.")
    ticket_parser.add_argument("ticket_id", help="Ticket id to show.")
    add_rig_args(ticket_parser)

    intake_parser = subparsers.add_parser("intake", help="Create multiple tickets from a manifest.")
    add_rig_args(intake_parser)
    intake_parser.add_argument(
        "--manifest", required=True, help="Path to a JSON array manifest file."
    )

    range_report_parser = subparsers.add_parser(
        "range-report", help="Compute the deterministic range diff artifact."
    )
    add_rig_args(range_report_parser)
    range_report_parser.add_argument(
        "--base-ref", default=None, help="Explicit base ref (default: refs/heads/main or root)."
    )
    range_report_parser.add_argument(
        "--critic", action="store_true", help="Also run a one-shot range-critic review."
    )

    return parser


def _cmd_rig_new(args: argparse.Namespace) -> int:
    try:
        rig_root = create_rig(args.charter, base_dir=args.path)
    except (CharterError, RigError, OSError) as exc:
        print(f"stigmergy rig new: {exc}", file=sys.stderr)
        return 1
    print(f"rig created at {rig_root}")
    return 0


def _cmd_daemon_run(args: argparse.Namespace) -> int:
    resolved: ResolvedRig | None = None
    try:
        resolved = resolve_rig(args.rig, rigs_root=args.rigs_root)
        daemon = _build_daemon(resolved)
    except (RigError, CharterError, UnbudgetableError, DaemonError, OSError) as exc:
        # `resolved` is only bound once `resolve_rig` has already succeeded
        # (i.e. `_build_daemon` is what raised) — construction failed, the
        # store was never handed off to a live, long-lived `Daemon`, so
        # nothing else will ever close it.
        if resolved is not None:
            resolved.store.close()
        print(f"stigmergy daemon run: {exc}", file=sys.stderr)
        return 1
    _run_daemon(daemon)
    return 0  # unreachable in real operation: _run_daemon loops forever or sys.exit(3)s


def _run_daemon(daemon: Daemon) -> None:
    """`recover_on_start()` then `run_forever()` — deliberately not wrapped
    in any try/except (bead `.27` build spec §2): anything raised inside
    (`RecoveryError`, `run_forever`'s `sys.exit(HALT_EXIT_CODE)`, or any
    other uncaught exception mid-loop) is fatal, per `daemon.py`'s own
    module docstring, and propagates all the way out of `main()` uncaught.
    """
    daemon.recover_on_start()
    daemon.run_forever()


def _critic_decoding_params(critic_cfg: dict[str, Any]) -> dict[str, Any]:
    """Sampling params for the critic/range-critic Messages call (bead .81).

    Current-gen Anthropic models (opus-4-8 / sonnet-5, the 4.7+/5 gen)
    DEPRECATE sampling params (temperature/top_p/top_k) and return HTTP 400
    if any is sent. Only forward `temperature` when the charter declares it
    (older/open models that still accept it); omit entirely otherwise so the
    critic gate works against the deprecating generation. Robust,
    capability-aware handling is bead .84.
    """
    if "temperature" in critic_cfg:
        return {"temperature": critic_cfg["temperature"]}
    return {}


def _build_daemon(resolved: ResolvedRig) -> Daemon:
    """Wire every real collaborator a `Daemon` needs from an already-
    resolved rig (bead `.27` build spec §2.1). Egress/relay stay at
    `Daemon`'s own `.22`-era placeholder defaults (out of scope, `.25`'s
    job); the critic client is now real (bead `.36`) -- see the inline
    comment above its construction below."""
    charter, registry, store = resolved.charter, resolved.registry, resolved.store
    rig_paths = resolved.rig_paths
    records_dir = rig_paths["records_dir"]

    record_plane = RecordPlane(records_dir)
    notification_store = NotificationStore(records_dir / "notifications.jsonl")
    notifier = NtfyNotifier(
        charter.raw["notify"]["ntfy_topic"],
        sender=make_ntfy_sender(_ntfy_server()),
    )

    budgets_cfg = charter.raw["loop"]["budgets"]
    spend_leash = SpendLeash(
        Budgets(
            dispatches=budgets_cfg["dispatches"],
            usd=budgets_cfg["usd"],
            gate_calls=budgets_cfg["gate_calls"],
            reserve_usd=0.0,  # v0 default -- charter has no reserve_usd knob (§3.4)
        ),
        registry,
        events=record_plane.read_events(),
    )

    # bead .79: use the per-rig built image (base + provision deps) when
    # present, else the charter base. resolve_rig computed this from rig meta.
    # v0: one image for worker + checker.
    checker_image = resolved.worker_image
    critic_cfg = charter.raw["roles"]["critic"]
    # bead .36: the critic client is now real (direct Anthropic Messages call,
    # SPEC §7) -- any client-side/provider failure surfaces as CriticInfraError,
    # never a silent wrong gate verdict (critic.py's own fail-closed discipline).
    # bead .39: the staging-gate critic now reads critic02 (the D14 filing-
    # mandate prompt bump, carrying the optional filed_tickets schema field).
    critic_key_provider = make_op_key_provider(_CRITIC_KEY_REF)
    critic = Critic.from_prompt_file(
        rig_paths["prompts_dir"] / "critic02",  # hardcoded filename -- see §3.7 note
        client=make_critic_client(
            key_provider=critic_key_provider,
            registry=registry,
            # bead .118: charter-overridable verdict-critic output budget
            # (default 4096) so a long verdict reason can't truncate.
            max_tokens=critic_cfg.get("max_tokens", DEFAULT_MAX_TOKENS),
        ),
        model=critic_cfg["model"],
        decoding_params=_critic_decoding_params(critic_cfg),
    )
    dispatch_limits = charter.raw["loop"]["dispatch_limits"]
    weaver = Weaver(
        store=store,
        record_plane=record_plane,
        staging_repo=rig_paths["repo_root"],
        run_checks_fn=checks.run_checks,
        critic=critic,
        checker_image=checker_image,
        flake_reruns=charter.raw["loop"]["retries"]["flake_reruns"],
        protected_paths=list(_DEFAULT_PROTECTED_PATHS),
        journal_path=records_dir / "weave_journal.jsonl",
        # bead .39: D14 per-dispatch filing caps, injected explicitly from the
        # charter (never an implicit charter reach-in from inside the weaver).
        filing_max_filings=dispatch_limits["filed_tickets"],
        filing_max_bytes=dispatch_limits["filed_ticket_bytes"],
        # bead .91: staging-gate checks get the same charter-configured resource
        # bounds as the daemon attempt gate (charter bound in the closure).
        check_resources_fn=lambda name: resolve_check_resources(charter, name),
    )

    # bead .25: the real credential relay + egress wiring (the last v0 piece).
    # ONE shared CapabilityStore — the daemon mints/revokes on it AND the relay
    # authorizes against it; a second instance would make every relay request
    # deny "unknown" (load-bearing; see bead25-build-spec §8 / test case E18).
    capability_store = CapabilityStore()
    # Constructed ONCE (reused across dispatches): the key provider caches
    # fetch-once (one `op read` per process); the forwarder is stateless.
    relay_key_provider = make_op_key_provider(_RELAY_KEY_REF)
    relay_forwarder = make_urllib_forwarder(base_url=_ANTHROPIC_BASE_URL)

    def secrets_for_capability(capability):
        # bead .25 audit (F-2): arm the sealed-transcript backstop against the
        # REAL key, not just the capability token. The daemon default redacts
        # only the token; .25 (per daemon.py + drivers/claude_code.py docstrings)
        # owns adding the real key to both the redactor and the must_not_contain
        # tripwire. relay_key_provider is op-backed + cached fetch-once, so this
        # adds no extra op calls after warmup. No ACTIVE leak exists (the worker
        # never holds the key) -- this is the last-line defense-in-depth the bead
        # was chartered to wire.
        return frozenset({capability.token, relay_key_provider()})

    def relay_setup_fn(provisional_id: str, runtime_dir: Path) -> RelayHandle:
        # Per-dispatch relay. The sync `forwarder` slot is a sentinel that
        # raises — the streaming serve_relay path uses prepare_upstream +
        # the injected async forwarder, never CredentialRelay._forwarder.
        relay = CredentialRelay(
            store=capability_store,
            key_provider=relay_key_provider,
            forwarder=_unused_sync_forwarder,
            upstream_headers_pinned={"anthropic-version": _ANTHROPIC_VERSION},
            upstream_header_allowlist=_RELAY_UPSTREAM_HEADER_ALLOWLIST,
        )
        log_path = Path(runtime_dir) / f"relay-{provisional_id}.jsonl"
        return start_relay(
            provisional_id,
            runtime_dir,
            relay,
            forwarder=relay_forwarder,
            log_path=log_path,
        )

    return Daemon(
        store=store,
        record_plane=record_plane,
        notification_store=notification_store,
        notifier=notifier,
        spend_leash=spend_leash,
        charter=charter,
        registry=registry,
        rig_paths=rig_paths,
        capability_store=capability_store,
        checker_image=checker_image,
        image=resolved.worker_image,
        weaver=weaver,
        container_reaper=PodmanContainerReaper(),
        steering_of=_make_steering_of(store, charter, rig_paths["prompts_dir"]),
        # spawn_fn / run_checks_fn: Daemon's own real defaults (claude_code.spawn /
        # checks.run_checks) -- do not pass explicitly, nothing to override.
        # bead .25: egress + relay wired for real (were .22-era placeholders).
        # relay_teardown_fn stays at Daemon's default (RelayHandle.stop).
        egress_setup_fn=egress.setup_dispatch_egress,
        relay_setup_fn=relay_setup_fn,
        secrets_for_capability=secrets_for_capability,
    )


def _make_steering_of(
    store: RigStore, charter: Charter, prompts_dir: Path
) -> Callable[[str], dict[str, Any]]:
    """`.35`'s integration contract: a ticket_id -> STEERING dict callable, bound to
    this rig's store/charter/prompts_dir. `.35` deliberately does not own this partial
    application (steering.py's own docstring: "that wiring is .27's job")."""

    def steering_of(ticket_id: str) -> dict[str, Any]:
        return derive_steering(store.get_ticket(ticket_id), charter, prompts_dir)

    return steering_of


# ==========================================================================
# Bead .42: triage CLI (approve/unapprove/reject/promote) + folded .33
# monitoring CLI (status/tickets/ticket/range-report) + REPORT event.
# ==========================================================================


def _resolve_rig_or_none(args: argparse.Namespace) -> tuple[ResolvedRig | None, int | None]:
    """Shared rig resolver for every C+D verb (bead .42 build spec §C/§D
    "shared resolver" gap #7). Returns ``(resolved, None)`` on success, or
    ``(None, 1)`` after printing a stderr error on failure — callers just
    check the second element. Calls the module-global `resolve_rig` by
    plain name (never imported/aliased locally) so tests can monkeypatch
    `cli.resolve_rig` and have it take effect here."""
    try:
        resolved = resolve_rig(args.rig, rigs_root=args.rigs_root)
    except (RigError, CharterError, UnbudgetableError, OSError) as exc:
        print(f"stigmergy: {exc}", file=sys.stderr)
        return None, 1
    return resolved, None


def _rig_name(resolved: ResolvedRig) -> str:
    """The rig's own name: `rig_meta.rig_name` if the store has it (it
    always should, post `create_rig`), else the charter's `[rig].name` as
    a fallback (mirrors the REPORT-event field map's own fallback)."""
    name = resolved.store.get_meta("rig_name")
    if name:
        return name
    return resolved.charter.raw["rig"]["name"]


def _cmd_approve(args: argparse.Namespace) -> int:
    resolved, rc = _resolve_rig_or_none(args)
    if resolved is None:
        return rc  # type: ignore[return-value]
    try:
        ticket = resolved.store.get_ticket(args.ticket_id)
        if ticket is None:
            print(f"stigmergy approve: no such ticket: {args.ticket_id}", file=sys.stderr)
            return 1

        steering = derive_steering(
            ticket, resolved.charter, resolved.rig_paths["prompts_dir"]
        )
        if args.json:
            print(json.dumps(steering, default=str))
        else:
            for key in (
                "ticket_text",
                "checks",
                "rubric",
                "target_scope",
                "functional_summary",
                "lane",
                "prompt_bytes",
                "context_set",
            ):
                print(f"{key}: {steering[key]}")

        approval.approve(resolved.store, args.ticket_id, steering=steering)
        approved_ticket = resolved.store.get_ticket(args.ticket_id)
        approval_hash = approved_ticket["approval_hash"] if approved_ticket else None

        record_plane = RecordPlane(resolved.rig_paths["records_dir"])
        triage.record_triage_event(
            record_plane,
            event_type=EventType.APPROVAL,
            rig=_rig_name(resolved),
            subject_id=args.ticket_id,
            outcome="approved",
            acting_agent=args.agent,
            operator_session=args.operator_session,
            approval_hash=approval_hash,
        )
        print(f"approval_hash: {approval_hash}")
        return 0
    finally:
        resolved.store.close()


def _cmd_unapprove(args: argparse.Namespace) -> int:
    resolved, rc = _resolve_rig_or_none(args)
    if resolved is None:
        return rc  # type: ignore[return-value]
    try:
        ticket = resolved.store.get_ticket(args.ticket_id)
        if ticket is None:
            print(f"stigmergy unapprove: no such ticket: {args.ticket_id}", file=sys.stderr)
            return 1

        previous_hash = ticket.get("approval_hash")
        approval.unapprove(resolved.store, args.ticket_id)

        record_plane = RecordPlane(resolved.rig_paths["records_dir"])
        triage.record_triage_event(
            record_plane,
            event_type=EventType.UNAPPROVAL,
            rig=_rig_name(resolved),
            subject_id=args.ticket_id,
            outcome="unapproved",
            acting_agent=args.agent,
            operator_session=args.operator_session,
            approval_hash=previous_hash,
        )
        print(f"unapproved {args.ticket_id}")
        return 0
    finally:
        resolved.store.close()


def _cmd_resume(args: argparse.Namespace) -> int:
    resolved, rc = _resolve_rig_or_none(args)
    if resolved is None:
        return rc  # type: ignore[return-value]
    try:
        ticket = resolved.store.get_ticket(args.ticket_id)
        if ticket is None:
            print(f"stigmergy resume: no such ticket: {args.ticket_id}", file=sys.stderr)
            return 1

        if ticket["state"] != ESCALATED:
            print(
                f"stigmergy resume: ticket {args.ticket_id} is not escalated "
                f"(state={ticket['state']!r}); resume only applies to escalated tickets",
                file=sys.stderr,
            )
            return 1

        previous_hash = ticket.get("approval_hash")
        transition(resolved.store, args.ticket_id, POOL, expected_from=ESCALATED)
        resolved.store.update_ticket(args.ticket_id, attempts_used=0, current_rung=None)

        record_plane = RecordPlane(resolved.rig_paths["records_dir"])
        triage.record_triage_event(
            record_plane,
            event_type=EventType.RESUME,
            rig=_rig_name(resolved),
            subject_id=args.ticket_id,
            outcome="resumed",
            acting_agent=args.agent,
            operator_session=args.operator_session,
            approval_hash=previous_hash,
        )
        print(f"resumed {args.ticket_id}")
        return 0
    finally:
        resolved.store.close()


def _cmd_reject(args: argparse.Namespace) -> int:
    resolved, rc = _resolve_rig_or_none(args)
    if resolved is None:
        return rc  # type: ignore[return-value]
    try:
        try:
            resolved.store.mark_filed_ticket_triaged(args.filed_id, outcome="rejected")
        except ValueError as exc:
            print(f"stigmergy reject: {exc}", file=sys.stderr)
            return 1

        record_plane = RecordPlane(resolved.rig_paths["records_dir"])
        triage.record_triage_event(
            record_plane,
            event_type=EventType.TRIAGE_REJECTED,
            rig=_rig_name(resolved),
            subject_id=args.filed_id,
            outcome="rejected",
            acting_agent=args.agent,
            operator_session=args.operator_session,
            reason=args.reason,
        )
        print(f"rejected {args.filed_id}")
        return 0
    finally:
        resolved.store.close()


def _cmd_promote(args: argparse.Namespace) -> int:
    resolved, rc = _resolve_rig_or_none(args)
    if resolved is None:
        return rc  # type: ignore[return-value]
    try:
        try:
            raw = sys.stdin.read() if args.spec == "-" else Path(args.spec).read_text(
                encoding="utf-8"
            )
            spec = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"stigmergy promote: could not read spec: {exc}", file=sys.stderr)
            return 1
        if not isinstance(spec, dict):
            print(
                f"stigmergy promote: spec must be a JSON object (got {type(spec).__name__})",
                file=sys.stderr,
            )
            return 1

        try:
            new_ticket_id = triage.promote_proposal(
                resolved.store, filed_id=args.filed_id, spec=spec
            )
        except triage.TriageError as exc:
            print(f"stigmergy promote: {exc}", file=sys.stderr)
            return 1

        print(new_ticket_id)
        return 0
    finally:
        resolved.store.close()


def _build_ticket_new_parser() -> argparse.ArgumentParser:
    """Build a dedicated argparse for `ticket new` (argv pre-dispatch helper)."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rig", required=True, help="Name of the rig.")
    parser.add_argument(
        "--rigs-root",
        default=None,
        help="Base directory the rig lives under (default: ~/rigs).",
    )
    parser.add_argument(
        "--spec", required=True, help="Path to a JSON ticket spec, or '-' for stdin."
    )
    return parser


def _cmd_ticket_new(args: argparse.Namespace) -> int:
    """Create a single new ticket from a JSON spec."""
    resolved, rc = _resolve_rig_or_none(args)
    if resolved is None:
        return rc  # type: ignore[return-value]
    try:
        try:
            raw = sys.stdin.read() if args.spec == "-" else Path(args.spec).read_text(
                encoding="utf-8"
            )
            spec = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"stigmergy ticket new: {exc}", file=sys.stderr)
            return 1
        if not isinstance(spec, dict):
            print(
                f"stigmergy ticket new: spec must be a JSON object (got {type(spec).__name__})",
                file=sys.stderr,
            )
            return 1

        # Validate required fields
        required_keys = {"id", "title", "functional_summary", "acceptance_criteria", "target_scope"}
        missing = required_keys - set(spec)
        if missing:
            print(
                f"stigmergy ticket new: spec missing required key(s): {sorted(missing)}",
                file=sys.stderr,
            )
            return 1

        # Validate functional_summary is a non-empty string
        functional_summary = spec.get("functional_summary")
        if not isinstance(functional_summary, str) or not functional_summary.strip():
            print(
                f"stigmergy ticket new: 'functional_summary' must be a non-empty string "
                f"(got {functional_summary!r})",
                file=sys.stderr,
            )
            return 1

        # Check for blocks key (not allowed for ticket new)
        if "blocks" in spec:
            print(
                "stigmergy ticket new: 'blocks' key is not allowed "
                "(use intake for dependency wiring)",
                file=sys.stderr,
            )
            return 1

        # Check if ticket id already exists
        if resolved.store.get_ticket(spec["id"]) is not None:
            print(f"stigmergy ticket new: ticket id already exists: {spec['id']}", file=sys.stderr)
            return 1

        # Build the fields to pass to add_ticket
        optional_keys = {
            "goal",
            "required_reading",
            "tier1_checks",
            "difficulty",
            "lane_hint",
            "rubric_only",
            "work_product",
        }
        ticket_fields: dict[str, Any] = {}
        for key in ["functional_summary", "acceptance_criteria", "target_scope"]:
            ticket_fields[key] = spec[key]
        for key in optional_keys:
            if key in spec:
                ticket_fields[key] = spec[key]

        # Insert the ticket (unapproved)
        resolved.store.add_ticket(id=spec["id"], title=spec["title"], **ticket_fields)

        print(spec["id"])
        return 0
    finally:
        resolved.store.close()


def _cmd_intake(args: argparse.Namespace) -> int:
    """Create multiple tickets from a JSON manifest with optional dependency wiring."""
    resolved, rc = _resolve_rig_or_none(args)
    if resolved is None:
        return rc  # type: ignore[return-value]
    try:
        try:
            manifest_text = Path(args.manifest).read_text(encoding="utf-8")
            manifest = json.loads(manifest_text)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"stigmergy intake: {exc}", file=sys.stderr)
            return 1
        if not isinstance(manifest, list):
            print(
                f"stigmergy intake: manifest must be a JSON array (got {type(manifest).__name__})",
                file=sys.stderr,
            )
            return 1

        # Collect all ids from the manifest in a first pass
        required_keys = {
            "id",
            "title",
            "functional_summary",
            "acceptance_criteria",
            "target_scope",
        }
        optional_keys = {
            "goal",
            "required_reading",
            "tier1_checks",
            "difficulty",
            "lane_hint",
            "rubric_only",
            "work_product",
        }
        manifest_ids = set()

        for entry in manifest:
            if isinstance(entry, dict) and "id" in entry:
                manifest_ids.add(entry["id"])

        # Second pass: validate all entries before inserting anything
        entries_to_insert: list[dict[str, Any]] = []

        for idx, entry in enumerate(manifest):
            if not isinstance(entry, dict):
                print(
                    f"stigmergy intake: manifest entry {idx} must be a JSON object "
                    f"(got {type(entry).__name__})",
                    file=sys.stderr,
                )
                return 1

            # Check required fields
            missing = required_keys - set(entry)
            if missing:
                entry_id = entry.get("id", f"<index {idx}>")
                sorted_missing = ", ".join(sorted(missing))
                print(
                    f"stigmergy intake: manifest entry {idx} ({entry_id}) "
                    f"missing required key(s): {sorted_missing}",
                    file=sys.stderr,
                )
                return 1

            # Check functional_summary is a non-empty string
            functional_summary = entry.get("functional_summary")
            if not isinstance(functional_summary, str) or not functional_summary.strip():
                entry_id = entry.get("id", f"<index {idx}>")
                print(
                    f"stigmergy intake: manifest entry {idx} ({entry_id}) "
                    f"'functional_summary' must be a non-empty string "
                    f"(got {functional_summary!r})",
                    file=sys.stderr,
                )
                return 1

            # Check for duplicate ids within the manifest (counting in manifest_ids)
            entry_id = entry["id"]
            count_in_manifest = sum(
                1 for e in manifest if isinstance(e, dict) and e.get("id") == entry_id
            )
            if count_in_manifest > 1:
                print(
                    f"stigmergy intake: manifest entry {idx} ({entry_id}) has duplicate id",
                    file=sys.stderr,
                )
                return 1

            # Check if this id already exists in the store
            if resolved.store.get_ticket(entry_id) is not None:
                print(
                    f"stigmergy intake: manifest entry {idx} ({entry_id}) ticket id already exists",
                    file=sys.stderr,
                )
                return 1

            # Validate blocks references (all must resolve to either store or manifest entries)
            blocks = entry.get("blocks", [])
            for predecessor_id in blocks:
                store_has_it = resolved.store.get_ticket(predecessor_id) is not None
                manifest_has_it = predecessor_id in manifest_ids
                if not store_has_it and not manifest_has_it:
                    print(
                        f"stigmergy intake: manifest entry {idx} ({entry_id}) "
                        f"blocks unresolved predecessor: {predecessor_id}",
                        file=sys.stderr,
                    )
                    return 1

            entries_to_insert.append(entry)

        # Second pass: insert all tickets
        for entry in entries_to_insert:
            ticket_fields: dict[str, Any] = {}
            for key in ["functional_summary", "acceptance_criteria", "target_scope"]:
                ticket_fields[key] = entry[key]
            for key in optional_keys:
                if key in entry:
                    ticket_fields[key] = entry[key]

            resolved.store.add_ticket(id=entry["id"], title=entry["title"], **ticket_fields)

        # Third pass: wire all dependencies
        for entry in entries_to_insert:
            blocks = entry.get("blocks", [])
            for predecessor_id in blocks:
                resolved.store.add_dep(entry["id"], predecessor_id)

        # Print one result line per entry
        for entry in entries_to_insert:
            print(f"{entry['id']}")

        return 0
    finally:
        resolved.store.close()


def _cmd_status(args: argparse.Namespace) -> int:
    resolved, rc = _resolve_rig_or_none(args)
    if resolved is None:
        return rc  # type: ignore[return-value]
    try:
        records_dir = resolved.rig_paths["records_dir"]
        status = gather_status(
            store=resolved.store,
            record_plane=RecordPlane(records_dir),
            notification_store=NotificationStore(records_dir / "notifications.jsonl"),
            charter_resolved=resolved.charter.raw,
            disk_path=str(resolved.rig_root),
            min_disk_bytes=None,
            now=time.time(),
        )
        print(render_status(status))
        return 0
    finally:
        resolved.store.close()


def _cmd_monitor(args: argparse.Namespace) -> int:
    resolved, rc = _resolve_rig_or_none(args)
    if resolved is None:
        return rc  # type: ignore[return-value]
    try:
        records_dir = resolved.rig_paths["records_dir"]
        record_plane = RecordPlane(records_dir)

        status = gather_status(
            store=resolved.store,
            record_plane=record_plane,
            notification_store=NotificationStore(records_dir / "notifications.jsonl"),
            charter_resolved=resolved.charter.raw,
            disk_path=str(resolved.rig_root),
            min_disk_bytes=None,
            now=time.time(),
        )

        print("=== daemon liveness ===")
        print(render_daemon_liveness(resolved.rig_root))
        print()
        print(render_status(status))
        print()
        print("=== recent events ===")
        events = record_plane.read_events()
        print(render_event_tail(events, args.tail))

        return 0
    finally:
        resolved.store.close()


def _cmd_tickets(args: argparse.Namespace) -> int:
    resolved, rc = _resolve_rig_or_none(args)
    if resolved is None:
        return rc  # type: ignore[return-value]
    try:
        print(render_ticket_list(resolved.store))
        return 0
    finally:
        resolved.store.close()


def _render_filed_list(store: Any) -> str:
    """One-line-per-proposal rendering of untriaged filed proposals with provenance."""
    filed = store.list_filed_tickets(triaged=False)
    if not filed:
        return "(no filed proposals)"

    lines: list[str] = []
    for f in filed:
        origin_role = f.get("origin_role") or "-"
        origin_worker = f.get("origin_worker") or "-"
        origin_dispatch_id = f.get("origin_dispatch_id") or "-"
        discovered_from = f.get("discovered_from") or "-"
        lines.append(
            f"{f['id']}  {f['title']}  origin_role={origin_role}  "
            f"origin_worker={origin_worker}  origin_dispatch_id={origin_dispatch_id}  "
            f"discovered_from={discovered_from}"
        )
    return "\n".join(lines)


def _cmd_filed(args: argparse.Namespace) -> int:
    resolved, rc = _resolve_rig_or_none(args)
    if resolved is None:
        return rc  # type: ignore[return-value]
    try:
        print(_render_filed_list(resolved.store))
        return 0
    finally:
        resolved.store.close()


def _cmd_ticket(args: argparse.Namespace) -> int:
    resolved, rc = _resolve_rig_or_none(args)
    if resolved is None:
        return rc  # type: ignore[return-value]
    try:
        print(render_ticket_detail(resolved.store, args.ticket_id))
        return 0
    finally:
        resolved.store.close()


def _build_range_critic(resolved: ResolvedRig) -> RangeCritic:
    """Production `RangeCritic` builder (bead .42 build spec §D; fixed by
    beads .51 + .41) — a monkeypatchable module-level helper: tests replace
    `cli._build_range_critic` wholesale to inject a stub client, so this
    function must be called by plain module-global name (never imported/
    aliased/bound early) at every call site.

    `.51` fix: wires `make_range_critic_client` (NOT the verdict client
    `make_critic_client`, which returns the wrong `{outcome,tier,reason,
    severity}` shape) and reads the `rangecrit02` prompt artifact (the
    `.41` combined-schema prompt bump, superseding `rangecrit01`).
    """
    charter = resolved.charter
    critic_cfg = charter.raw["roles"]["critic"]
    critic_key_provider = make_op_key_provider(_CRITIC_KEY_REF)
    return RangeCritic.from_prompt_file(
        resolved.rig_paths["prompts_dir"] / "rangecrit02",
        client=make_range_critic_client(
            key_provider=critic_key_provider, registry=resolved.registry
        ),
        model=critic_cfg["model"],
        decoding_params=_critic_decoding_params(critic_cfg),
    )


def _report_ctx(resolved: ResolvedRig, report: RangeReport, result: RangeCriticResult) -> dict:
    """The single source of truth for the 12 common-minus-4 event fields
    shared by the REPORT event and the `.41` filing (beads .51 + .41 build
    spec §3.3). Deliberately EXCLUDES `attempt_kind`/`tokens`/
    `computed_usd`/`wall_time_seconds` — `make_event`/`file_proposals`
    supply those 4 themselves, so including them here would raise a
    duplicate-kwarg `TypeError` at the call sites below.

    `ticket=None` is honest — a range report has no parent ticket
    (advisor pt3); the resulting `discovered_from = "report-<oid>@None"`
    is an accepted cosmetic wart, not to be "fixed" by forging `ticket`.
    """
    return {
        "rig": _rig_name(resolved),
        "ticket": None,
        "dispatch_id": f"report-{report.staging_oid[:12]}",
        "attempt": 0,
        "rung": None,
        "worker": None,
        "charter_hash": resolved.charter.resolved_hash,
        "approval_hash": None,
        "image_digest": None,
        "model": result.model,
        "model_version": None,
        "price_table_version": resolved.registry.version_hash,
    }


def _emit_report_event(
    resolved: ResolvedRig,
    report: RangeReport,
    result: RangeCriticResult,
    record_plane: RecordPlane,
) -> None:
    """Emit ONE REPORT event for a `range-report --critic` run (bead .42
    build spec §D field map; unbudgetable fix beads .51 + .41). Takes an
    INJECTED `record_plane` (built once by the caller) rather than
    constructing its own — `_cmd_range_report` shares one `RecordPlane`
    between this REPORT event and the `.41` filing.

    Never a silent $0 on a paid metered call with unparseable usage:
    `computed_usd` is `"unbudgetable"` whenever the model resolves to
    `PricingClass.METERED` and `result.usage` is empty (the provenance-
    gated sentinel `_map_range_usage` returns for "no trustworthy usage
    object"). A registry-miss model is unbudgetable too (existing
    behavior). Any other pricing class (or a metered model with real
    usage) prices normally via `spend.cost_usd`, which may itself still
    return the string `"unbudgetable"` (e.g. a metered category with no
    declared rate).
    """
    tokens = {
        "in": result.usage.get("in", 0),
        "cached": result.usage.get("cached", 0),
        "out": result.usage.get("out", 0),
        "reasoning": result.usage.get("reasoning", 0),
    }
    try:
        entry = resolved.registry.resolve(result.model)
    except UnbudgetableError:
        computed_usd: float | str = "unbudgetable"
    else:
        if entry.pricing is PricingClass.METERED and not result.usage:
            # paid call, no usage object -> never a silent $0.
            computed_usd = "unbudgetable"
        else:
            computed_usd = spend.cost_usd(entry, tokens)

    event = make_event(
        EventType.REPORT,
        **_report_ctx(resolved, report, result),
        attempt_kind="report",
        tokens=tokens,
        computed_usd=computed_usd,
        prompt_artifact_hash=result.prompt_artifact_hash,
        wall_time_seconds=0.0,
    )
    record_plane.append(event)


def _cmd_range_report(args: argparse.Namespace) -> int:
    resolved, rc = _resolve_rig_or_none(args)
    if resolved is None:
        return rc  # type: ignore[return-value]
    try:
        try:
            report = compute_range_report(
                resolved.rig_paths["repo_root"], base_ref=args.base_ref
            )
        except RangeReportError as exc:
            print(f"stigmergy range-report: {exc}", file=sys.stderr)
            return 1

        print(report.render())

        if args.critic:
            critic = _build_range_critic(resolved)
            try:
                result = critic.review(report)
            except RangeReportError as exc:
                print(f"stigmergy range-report: {exc}", file=sys.stderr)
                return 1
            print(result.findings)

            # ONE RecordPlane, shared between the REPORT event and the
            # .41 filing below (never construct a second one here).
            record_plane = RecordPlane(resolved.rig_paths["records_dir"])

            # The paid call is recorded FIRST — even if filing partially
            # rejects or hits caps, the REPORT event already landed
            # (advisor pt4: "never lose the report").
            _emit_report_event(resolved, report, result, record_plane)

            ctx = _report_ctx(resolved, report, result)
            limits = resolved.charter.raw["loop"]["dispatch_limits"]
            filing_result = file_proposals(
                result.filed_tickets,
                store=resolved.store,
                record_plane=record_plane,
                ctx=ctx,
                attempt_kind="report",
                origin_role="range-critic",
                max_filings=limits["filed_tickets"],
                max_bytes=limits["filed_ticket_bytes"],
            )
            if filing_result.accepted_ids:
                print("filed proposals:")
                for filed_id in filing_result.accepted_ids:
                    print(f"  {filed_id}")

        return 0
    finally:
        resolved.store.close()


# The dedicated dogfood 1Password item for the critic's real Anthropic key
# (item id `oknaudituuajtuw3cnv4dra7s4`, field `credential`) -- per bead .31's
# comment, do NOT source the critic key from any other vault item.
_CRITIC_KEY_REF = "op://shelly/API Credential - stigmergy rig 00/credential"

# bead .25: the credential relay's real Anthropic key (same op item the .64
# capture used). The real key lives ONLY in the relay process; the worker gets
# a capability token. See bead25-build-spec.md.
_RELAY_KEY_REF = "op://shelly/API Credential - stigmergy rig 00/credential"
_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
# Current REQUIRED header value per Anthropic docs (verified 2026-07-18). The
# relay pins this (worker's own version header is stripped); without it the
# first live relay request 400s.
_ANTHROPIC_VERSION = "2023-06-01"
# bead .25 (SB adjudication 2026-07-18, option A): claude-code needs
# `anthropic-beta` passed through to upstream (its beta-gated body fields, e.g.
# context_management, depend on it; the beta list varies per request so pinning
# cannot work). The module DEFAULT stays tight (honors .55) — this widening
# lives ONLY here at the construction site, documented + auditable. Residual
# risk (server-side-tool betas bypassing the cage) tracked in bead .75.
_RELAY_UPSTREAM_HEADER_ALLOWLIST = frozenset({"content-type", "accept", "anthropic-beta"})


def _unused_sync_forwarder(request):  # noqa: ANN001, ANN202
    """CredentialRelay's sync `forwarder` slot (the .12 full-body path). The
    streaming `serve_relay` used in production never calls it — it uses
    `prepare_upstream` + the injected async forwarder — so this must never run.
    Raises loudly if it ever does (a wiring bug), never silently forwards."""
    raise RuntimeError(
        "sync CredentialRelay.forwarder must never run on the streaming relay path"
    )


_DEFAULT_NTFY_SERVER = "https://ntfy.sh"
_NTFY_SERVER_ENV_VAR = "STIGMERGY_NTFY_SERVER"  # deliberately NOT an SG_ charter
# override (those are schema-validated against the charter's known key set; ntfy
# server url is not, and is not, a charter field in the current schema).


def _ntfy_server() -> str:
    return os.environ.get(_NTFY_SERVER_ENV_VAR, _DEFAULT_NTFY_SERVER)


_DEFAULT_PROTECTED_PATHS = (
    "tests/",
    "pyproject.toml",
    "images/",
    ".github/",
    "prompts/",  # deliberate -- see the .27 build spec §1.1's defense-in-depth note
)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the `stigmergy` console script. Returns an exit code."""
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv:
        print(_USAGE)
        return 0

    # argv pre-dispatch for 'ticket new' (special case to avoid colliding with
    # the existing 'ticket <id>' subparser that takes a positional ticket_id)
    if len(argv) >= 2 and argv[0] == "ticket" and argv[1] == "new":
        ticket_new_parser = _build_ticket_new_parser()
        ticket_new_args = ticket_new_parser.parse_args(argv[2:])
        return _cmd_ticket_new(ticket_new_args)

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "rig":
        if args.rig_command == "new":
            return _cmd_rig_new(args)
        print(_USAGE, file=sys.stderr)
        return 1

    if args.command == "daemon":
        if args.daemon_command == "run":
            return _cmd_daemon_run(args)
        print(_USAGE, file=sys.stderr)
        return 1

    if args.command == "approve":
        return _cmd_approve(args)
    if args.command == "unapprove":
        return _cmd_unapprove(args)
    if args.command == "resume":
        return _cmd_resume(args)
    if args.command == "reject":
        return _cmd_reject(args)
    if args.command == "promote":
        return _cmd_promote(args)
    if args.command == "status":
        return _cmd_status(args)
    if args.command == "monitor":
        return _cmd_monitor(args)
    if args.command == "tickets":
        return _cmd_tickets(args)
    if args.command == "filed":
        return _cmd_filed(args)
    if args.command == "ticket":
        return _cmd_ticket(args)
    if args.command == "intake":
        return _cmd_intake(args)
    if args.command == "range-report":
        return _cmd_range_report(args)

    print(_USAGE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
