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
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from stigmergy import approval, checks, egress, spend, triage
from stigmergy.charter import Charter, CharterError, resolve_check_resources
from stigmergy.container import PodmanContainerReaper
from stigmergy.critic import Critic
from stigmergy.daemon import Daemon, DaemonError
from stigmergy.filing import file_proposals
from stigmergy.keyprovider import make_op_key_provider
from stigmergy.notify import NotificationStore, NtfyNotifier, make_ntfy_sender

# bead .143 (A′): the critic/range-critic now route through OA's
# provider layer — `oa_critic` is the ONLY OA-importing seam (lazy,
# fail-closed at factory build). The legacy bare-urllib factories
# (`make_critic_client` / `make_range_critic_client`) are deleted; the
# module-level names below keep the old call sites' contract for the
# .51 wiring-regression test, which monkeypatches them to prove the
# range builder does NOT use the verdict client.
from stigmergy.oa_critic import (
    DEFAULT_MAX_TOKENS,
    CriticOAUnavailableError,
    make_oa_critic_client,
    make_oa_range_critic_client,
)
from stigmergy.rangereport import (
    RangeCritic,
    RangeCriticResult,
    RangeReport,
    RangeReportError,
    compute_range_report,
)
from stigmergy.records import EventType, RecordPlane, make_event
from stigmergy.registry import PricingClass, Registry, UnbudgetableError
from stigmergy.relay import (
    DEFAULT_RELAY_PROFILE,
    CapabilityStore,
    CredentialRelay,
    RelayProfile,
    _openai_json_usage_extractor,
)
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
    # bead .36: the critic client is now real -- any client-side/provider
    # failure surfaces as CriticInfraError, never a silent wrong gate
    # verdict (critic.py's own fail-closed discipline).
    # bead .39: the staging-gate critic reads critic02+ (the D14 filing-
    # mandate prompt bump, carrying the optional filed_tickets schema field).
    # bead .140: critic03 adds the moved-file trusted-evidence exception
    # (SB-approved 2026-08-30) so rename-only artifacts are verifiable.
    # bead .143 (A′): the client is the OA provider-layer forced-tool
    # call (hardened=True; the legacy bare-urllib client is deleted).
    # The factory build is FAIL-CLOSED: an OA-less environment raises
    # CriticOAUnavailableError here (rig launch), never at the first
    # gate. The key ref follows the resolved entry's provider (Decision 3).
    critic_key_provider = make_op_key_provider(
        _critic_key_ref_for(resolved.registry, critic_cfg["model"])
    )
    critic = Critic.from_prompt_file(
        rig_paths["prompts_dir"] / "critic03",  # hardcoded filename -- see §3.7 note
        client=make_oa_critic_client(
            key_provider=critic_key_provider,
            registry=registry,
            # bead .118: charter-overridable verdict-critic output budget
            # (default 4096) so a long verdict reason can't truncate.
            max_tokens=critic_cfg.get("max_tokens", DEFAULT_MAX_TOKENS),
        ),
        model=critic_cfg["model"],
        decoding_params={},
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
    # bead .147 §1F: the per-dispatch RELAY PROFILE side-channel. The daemon
    # (its dispatch path, where lane + registry + charter are in scope)
    # DERIVES each lane's profile and writes it into this cell keyed by the
    # real dispatch id; the closure below READS it (fallback: the DEFAULT
    # anthropic profile — today's exact shape, so the pre-`.147` wiring tests
    # and stubs keep observing identical behaviour). The `relay_setup_fn`
    # signature stays 2-arg (ten+ test stubs pin it); the profile travels
    # through the cell, not the callback.
    relay_profile_cell: dict[str, RelayProfile] = {}
    # Per-profile resources, constructed LAZILY once per (upstream_base_url,
    # auth) and cached: the op-backed key provider is fetch-once per process
    # (one `op read` per profile's key ref), the forwarder is stateless. The
    # DEFAULT (anthropic) pair is the pre-`.147` construction, byte-identical.
    relay_key_provider = make_op_key_provider(_RELAY_KEY_REF)
    # bead .147: the Synthetic OpenAI-lane key provider (fetch-once per
    # process, like the Anthropic one) — shared by the bearer-auth relay
    # resources and the transcript-secrets backstop below.
    synthetic_relay_key_provider = make_op_key_provider(_SYNTHETIC_RELAY_KEY_REF)
    relay_forwarder = make_urllib_forwarder(base_url=_ANTHROPIC_BASE_URL)
    relay_resource_cache: dict[tuple[str, str], tuple] = {}

    def _relay_resources(profile: RelayProfile):
        """(key_provider, forwarder) for a profile — built lazily, cached.

        - anthropic profile: the pre-`.147` shared pair (key ref
          `_RELAY_KEY_REF`, base `_ANTHROPIC_BASE_URL`) — returned exactly as
          before for the default profile (the `.25` wiring test asserts the
          same objects).
        - openai + bearer: the Synthetic key ref (`_SYNTHETIC_RELAY_KEY_REF`
          — wired by name only, the provider owns the fetch) + a
          `make_urllib_forwarder` for the profile's upstream (cached per
          base_url: the no-redirect, proxy-disabled opener is shared).
        - openai + none (keyless, e.g. blackwell): a key provider that
          returns None (the header is OMITTED upstream — never `Bearer
          None`) + the per-base_url forwarder.
        """
        if profile is DEFAULT_RELAY_PROFILE or (
            profile.wire == "anthropic" and profile.auth == "x-api-key"
        ):
            return relay_key_provider, relay_forwarder
        cache_key = (profile.upstream_base_url, profile.auth)
        resources = relay_resource_cache.get(cache_key)
        if resources is None:
            if profile.auth == "none":

                def _none_key_provider() -> None:
                    # Keyless lane: the upstream carries no credential. The
                    # relay OMITS the header entirely (relay.py
                    # prepare_upstream's `none` branch).
                    return None

                key_provider: Callable[[], str | None] = _none_key_provider
            else:  # bearer (and any future auth form: a fetched key)
                key_provider = synthetic_relay_key_provider
            forwarder = make_urllib_forwarder(base_url=profile.upstream_base_url)
            resources = (key_provider, forwarder)
            relay_resource_cache[cache_key] = resources
        return resources

    def secrets_for_capability(capability):
        # bead .25 audit (F-2): arm the sealed-transcript backstop against the
        # REAL key, not just the capability token. The daemon default redacts
        # only the token; .25 (per daemon.py + drivers/claude_code.py docstrings)
        # owns adding the real key to both the redactor and the must_not_contain
        # tripwire. relay_key_provider is op-backed + cached fetch-once, so this
        # adds no extra op calls after warmup. No ACTIVE leak exists (the worker
        # never holds the key) -- this is the last-line defense-in-depth the bead
        # was chartered to wire.
        # bead .147 review fix F2 (2026-08-30): over-arm the backstop with the
        # SYNTHETIC key too — OpenAI lanes inject it toward their upstream, so
        # a sealed transcript must redact it as well. Best-effort by design: a
        # key that cannot be resolved was never injected toward any upstream,
        # so there is nothing to redact (and a pure-Anthropic rig must not
        # gain a new failure mode at seal time). Per-profile precision lives
        # in `relay.injected_secrets`; this closure is the coarse last line.
        secrets = {capability.token, relay_key_provider()}
        try:
            synthetic_key = synthetic_relay_key_provider()
            if synthetic_key:
                secrets.add(synthetic_key)
        except Exception:  # noqa: BLE001 - unresolvable key == never on the wire
            pass
        return frozenset(secrets)

    def relay_setup_fn(provisional_id: str, runtime_dir: Path) -> RelayHandle:
        # Per-dispatch relay (signature UNCHANGED — the profile arrives via
        # the side-channel cell, not an extra argument). The sync
        # `forwarder` slot is a sentinel that raises — the streaming
        # serve_relay path uses prepare_upstream + the injected async
        # forwarder, never CredentialRelay._forwarder.
        profile = relay_profile_cell.get(provisional_id, DEFAULT_RELAY_PROFILE)
        key_provider, forwarder = _relay_resources(profile)
        relay_kwargs: dict[str, Any] = dict(
            store=capability_store,
            key_provider=key_provider,
            forwarder=_unused_sync_forwarder,
            upstream_headers_pinned=dict(profile.pinned_headers),
            capability_header=profile.capability_header,
            allowed_endpoints=profile.allowed_endpoints,
            auth=profile.auth,
            pricing_class=profile.pricing_class,
            wire=profile.wire,
        )
        if profile.wire == "openai":
            # OpenAI wire: no version pin (profile.pinned_headers is {}),
            # the default tight allowlist (content-type/accept — the
            # worker's own `authorization` is the capability header,
            # dropped by construction), and the OpenAI JSON usage extractor
            # (`usage.completion_tokens` -> output; the full usage dict
            # feeds the relay JSONL).
            relay_kwargs["usage_extractor"] = _openai_json_usage_extractor
        else:
            # anthropic: today's exact shape (SB option A widened allowlist
            # + anthropic-version pin, carried by the profile).
            relay_kwargs["upstream_header_allowlist"] = _RELAY_UPSTREAM_HEADER_ALLOWLIST
        relay = CredentialRelay(**relay_kwargs)
        log_path = Path(runtime_dir) / f"relay-{provisional_id}.jsonl"
        return start_relay(
            provisional_id,
            runtime_dir,
            relay,
            forwarder=forwarder,
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
        # bead .147 §1F: the profile side-channel (daemon derives per lane,
        # closure reads; default None elsewhere = today's behaviour).
        relay_profile_cell=relay_profile_cell,
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
        # Count occurrences of each id in the manifest (single O(n) pass)
        id_counts = Counter()
        for entry in manifest:
            if isinstance(entry, dict) and "id" in entry:
                id_counts[entry["id"]] += 1
        manifest_ids = set(id_counts.keys())

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

            # Check for duplicate ids within the manifest using pre-computed counts
            entry_id = entry["id"]
            if id_counts[entry_id] > 1:
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
    beads .51 + .41; migrated by bead .143) — a monkeypatchable
    module-level helper: tests replace `cli._build_range_critic`
    wholesale to inject a stub client, so this function must be called
    by plain module-global name (never imported/aliased/bound early) at
    every call site.

    `.51` fix: wires `make_oa_range_critic_client` (NOT the verdict
    client `make_oa_critic_client`, which returns the wrong
    `{outcome,tier,reason,severity}` shape) and reads the `rangecrit02`
    prompt artifact (the `.41` combined-schema prompt bump, superseding
    `rangecrit01`). `.143`: the client is the OA provider-layer
    forced-tool call; it resolves the SAME `[roles.critic].model` as the
    staging-gate critic (shared model, separate forced tool).
    """
    charter = resolved.charter
    critic_cfg = charter.raw["roles"]["critic"]
    critic_key_provider = make_op_key_provider(
        _critic_key_ref_for(resolved.registry, critic_cfg["model"])
    )
    return RangeCritic.from_prompt_file(
        resolved.rig_paths["prompts_dir"] / "rangecrit02",
        # bead .51: route through the module-level ALIAS name so the
        # wiring-regression test's monkeypatch of make_range_critic_client
        # intercepts (the alias delegates to the OA factory by default).
        client=make_range_critic_client(
            key_provider=critic_key_provider, registry=resolved.registry
        ),
        model=critic_cfg["model"],
        decoding_params={},
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
            try:
                critic = _build_range_critic(resolved)
            except (CriticOAUnavailableError, CharterError) as exc:
                # bead .143: factory build is fail-closed — an OA-less
                # environment or a missing STIGMERGY_CRITIC_OA_KEY_REF is a
                # loud LAUNCH failure (exit 1, stderr), not a mid-run
                # trip.
                print(f"stigmergy range-report: {exc}", file=sys.stderr)
                return 1
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

# bead .143 (Decision 3): for a NON-Anthropic-routed critic entry, the
# rig's op key ref comes from this env var (the ref follows the
# PROVIDER, not the station). Absent on a non-Anthropic entry is a loud
# RIG-LAUNCH failure. Deliberately NOT SG_-prefixed: every SG_* env var is
# a charter override key (charter.py:177 family's `_ENV_PREFIX`), and
# STIGMERGY_CRITIC_OA_KEY_REF would be rejected by the strict schema as an
# unknown top-level key — the exact failure `STIGMERGY_NTFY_SERVER`
# already avoids (see below).
_CRITIC_OA_KEY_REF_ENV = "STIGMERGY_CRITIC_OA_KEY_REF"

# bead .25: the credential relay's real Anthropic key (same op item the .64
# capture used). The real key lives ONLY in the relay process; the worker gets
# a capability token. See bead25-build-spec.md.
_RELAY_KEY_REF = "op://shelly/API Credential - stigmergy rig 00/credential"
# bead .147: the Synthetic OpenAI-lane relay key — the SAME 1Password item
# the critic uses for its OpenAI-routed entries (the ref follows the
# provider, not the station — Decision 3, bead .143). Wired BY NAME ONLY:
# this module never fetches, logs, or caches the key itself; the op-backed
# provider (fetch-once, in the relay process) owns it.
_SYNTHETIC_RELAY_KEY_REF = "op://shelly/cqntl7jj446cxwplb2hafwxinq/credential"
_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
# Current REQUIRED header value per Anthropic docs (verified 2026-07-18). The
# relay pins this (worker's own version header is stripped); without it the
# first live relay request 400s.
_ANTHROPIC_VERSION = "2023-06-01"
def _critic_key_ref_for(registry: Registry, model: str) -> str:
    """bead .143 (Decision 3): the op key ref for the critic station
    follows the resolved entry's OA provider — the ref follows the
    PROVIDER, not the station.

    - Anthropic-routed entry (``oa_provider_key == "anthropic"`` — also
      the default for ``provider = "anthropic"``): the rig's dedicated
      Anthropic key item, ``_CRITIC_KEY_REF`` (unchanged).
    - Any other provider: the rig-level env var
      ``STIGMERGY_CRITIC_OA_KEY_REF`` names the op ref. Absent/empty on a
      non-Anthropic entry is a loud RIG-LAUNCH failure
      (:class:`CharterError`) — checked at factory build, never a
      first-gate infra trip.
    """
    entry = registry.resolve(model)
    if entry.oa_provider_key == "anthropic":
        return _CRITIC_KEY_REF
    ref = os.environ.get(_CRITIC_OA_KEY_REF_ENV, "").strip()
    if not ref:
        raise CharterError(
            f"critic model {model!r} routes to OA provider "
            f"{entry.oa_provider_key!r}; set the {_CRITIC_OA_KEY_REF_ENV} env "
            "variable to its 1Password op key ref (the ref follows the "
            "provider, not the station)"
        )
    return ref


def make_critic_client(*args: Any, **kwargs: Any) -> Callable[..., dict[str, Any]]:
    """DEPRECATED alias (bead .143): the legacy bare-urllib verdict
    client was replaced by :func:`make_oa_critic_client`. Kept as a
    module-level name so the .51 wiring-regression test (which
    monkeypatches it to a raising stub and asserts the range builder
    never calls it) keeps working. New code must use the OA factory."""
    return make_oa_critic_client(*args, **kwargs)


def make_range_critic_client(*args: Any, **kwargs: Any) -> Callable[..., dict[str, Any]]:
    """DEPRECATED alias (bead .143): the legacy bare-urllib range-critic
    client was replaced by :func:`make_oa_range_critic_client` (same
    reason as :func:`make_critic_client` — the .51 wiring-regression
    test monkeypatches this name to assert the range builder wires the
    RANGE client, not the verdict client)."""
    return make_oa_range_critic_client(*args, **kwargs)


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
