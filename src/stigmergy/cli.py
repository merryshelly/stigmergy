"""Stigmergy command-line interface.

Top-level argparse dispatcher. v0 subcommand tree:

    stigmergy rig new --charter <path> [--path <base-dir>]
    stigmergy daemon run --rig <name> [--rigs-root <path>]

Bare invocation (no args) prints short usage and returns 0 — preserved
from the v0 stub so existing scripts/tests calling `main([])` keep working.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from stigmergy import checks
from stigmergy.charter import Charter, CharterError
from stigmergy.container import PodmanContainerReaper
from stigmergy.critic import Critic
from stigmergy.daemon import Daemon, DaemonError
from stigmergy.notify import NotificationStore, NtfyNotifier, make_ntfy_sender
from stigmergy.records import RecordPlane
from stigmergy.registry import UnbudgetableError
from stigmergy.relay import CapabilityStore
from stigmergy.rig import ResolvedRig, RigError, RigStore, create_rig, resolve_rig
from stigmergy.spend import Budgets, SpendLeash
from stigmergy.steering import derive_steering
from stigmergy.weaver import Weaver

_USAGE = (
    "stigmergy v0 — usage: stigmergy <command> ...\n"
    "  rig new --charter <path> [--path <base-dir>]\n"
    "  daemon run --rig <name> [--rigs-root <path>]"
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
    job); the critic client is an explicit, loud placeholder (§3.3 —
    tracked as bead `.36`)."""
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
    )

    checker_image = charter.raw["rig"]["image"]  # v0: same image for worker + checker
    critic_cfg = charter.raw["roles"]["critic"]
    critic = Critic.from_prompt_file(
        rig_paths["prompts_dir"] / "critic01",  # hardcoded filename -- see §3.7 note
        client=_unwired_critic_client,
        model=critic_cfg["model"],
        decoding_params={"temperature": critic_cfg["temperature"]},
    )
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
        capability_store=CapabilityStore(),
        checker_image=checker_image,
        weaver=weaver,
        container_reaper=PodmanContainerReaper(),
        steering_of=_make_steering_of(store, charter, rig_paths["prompts_dir"]),
        # spawn_fn / run_checks_fn: Daemon's own real defaults (claude_code.spawn /
        # checks.run_checks) -- do not pass explicitly, nothing to override.
        # egress_setup_fn / egress_teardown_fn / relay_base_url: Daemon's own
        # .22-era placeholder defaults -- do NOT wire real egress/relay here (§0).
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


def _unwired_critic_client(prompt: str, *, model: str, **decoding_params: Any) -> Any:
    """v0 placeholder (§3.3): raises unconditionally. `Critic.judge()` converts ANY
    client exception into `CriticInfraError` (never crashes the daemon) -- a weave
    that runs before the real client is wired surfaces as an ordinary critic-infra
    trip (visible via the circuit breaker / halt notification), never a silent wrong
    gate verdict. Tracked as bead `.36` -- replace this callable's caller (the
    `Critic.from_prompt_file(...)` call in `_build_daemon`) once `.36` lands; nothing
    else in this module should need to change."""
    raise NotImplementedError(
        "stigmergy: real critic API client not yet wired (tracked as bead .36) -- "
        "weave/gate calls surface as critic-infra trips until a real "
        "provider-calling client is injected here"
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

    print(_USAGE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
