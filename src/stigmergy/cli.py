"""Stigmergy command-line interface (v0 stub)."""
import sys


def main(argv: list[str] | None = None) -> int:
    """Entry point for the `stigmergy` console script."""
    argv = list(sys.argv[1:] if argv is None else argv)
    print("stigmergy v0 — not yet implemented")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
