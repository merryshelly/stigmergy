"""Stigmergy command-line interface.

Top-level argparse dispatcher. v0 subcommand tree:

    stigmergy rig new --charter <path> [--path <base-dir>]

Bare invocation (no args) prints short usage and returns 0 — preserved
from the v0 stub so existing scripts/tests calling `main([])` keep working.
"""

from __future__ import annotations

import argparse
import sys

from stigmergy.charter import CharterError
from stigmergy.rig import RigError, create_rig

_USAGE = (
    "stigmergy v0 — usage: stigmergy <command> ...\n"
    "  rig new --charter <path> [--path <base-dir>]"
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

    return parser


def _cmd_rig_new(args: argparse.Namespace) -> int:
    try:
        rig_root = create_rig(args.charter, base_dir=args.path)
    except (CharterError, RigError, OSError) as exc:
        print(f"stigmergy rig new: {exc}", file=sys.stderr)
        return 1
    print(f"rig created at {rig_root}")
    return 0


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

    print(_USAGE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
