"""In-container egress/relay shim (Stigmergy bead .63).

A dependency-minimal raw TCP<->unix byte bridge. It runs INSIDE the hardened
`--network=none` worker container and is the ONLY path from an unmodified HTTP
client (claude-code) to the host-side egress CONNECT proxy and credential
relay, both mounted into the cage as unix sockets.

`HTTPS_PROXY` / `ANTHROPIC_BASE_URL` need a `host:port`; a unix socket path is
neither. This shim listens on `127.0.0.1:<port>` and pipes raw bytes —
unmodified and unparsed — onto the mounted unix socket, so a stock client
configured with `HTTPS_PROXY=http://127.0.0.1:<port>` transparently reaches the
proxy. It NEVER understands HTTP/CONNECT, NEVER logs payload bytes, and can
only ever dial ONE hardcoded unix path per invocation: the `name` argv selects
which of two baked targets, but the port and dial path are module constants,
taken from no worker-reachable argv/env input (defense-in-depth — the process
that decides the egress path takes no runtime configuration).

Pure stdlib: the same file runs standalone in the container and imports on the
host, where `serve_bridge` is the testable core and `main` is the hardcoded
entry point.
"""

from __future__ import annotations

import asyncio
import sys

# Loopback ONLY — never 0.0.0.0. Under `--network=none` only `lo` exists, so
# this is moot today, but it is pinned as defense-in-depth against a future
# network profile that might add a routable interface.
LISTEN_HOST = "127.0.0.1"

EGRESS_PORT = 18080
RELAY_PORT = 18081
EGRESS_SOCKET = "/run/egress.sock"
RELAY_SOCKET = "/run/relay.sock"

# Hardcoded + exhaustive. `main` looks up exactly one entry by the `name`
# argv; the port and dial path are NEVER supplied by argv or env.
TARGETS: dict[str, tuple[int, str]] = {
    "egress": (EGRESS_PORT, EGRESS_SOCKET),
    "relay": (RELAY_PORT, RELAY_SOCKET),
}

_BUF = 65536


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Copy bytes one direction until EOF, then half-close the peer's write
    side so the other end sees the close. Never logs payload. Swallows the
    ordinary transport errors a closing socket raises."""
    try:
        while True:
            chunk = await reader.read(_BUF)
            if not chunk:
                break
            writer.write(chunk)
            await writer.drain()
    except (ConnectionError, OSError):
        pass
    finally:
        try:
            if writer.can_write_eof():
                writer.write_eof()
        except (ConnectionError, OSError):
            pass


async def _handle(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    dial_path: str,
) -> None:
    """One accepted loopback connection: dial the hardcoded unix socket and
    bridge both directions. If the dial fails (proxy/relay down or its socket
    absent) the client connection is closed — fail-closed, NEVER a fallback
    egress path."""
    try:
        unix_reader, unix_writer = await asyncio.open_unix_connection(dial_path)
    except OSError:
        client_writer.close()
        try:
            await client_writer.wait_closed()
        except (ConnectionError, OSError):
            pass
        return
    try:
        await asyncio.gather(
            _pipe(client_reader, unix_writer),
            _pipe(unix_reader, client_writer),
        )
    finally:
        for w in (client_writer, unix_writer):
            try:
                w.close()
            except (ConnectionError, OSError):
                pass


async def serve_bridge(
    listen_host: str, listen_port: int, dial_path: str
) -> asyncio.Server:
    """Start a TCP listener on ``listen_host:listen_port`` that bridges every
    accepted connection to the unix socket at ``dial_path``. Returns the live
    :class:`asyncio.Server`; the caller owns its lifecycle. The testable
    core — ``main`` calls it with hardcoded constants only."""

    async def _cb(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await _handle(reader, writer, dial_path)

    return await asyncio.start_server(_cb, host=listen_host, port=listen_port)


async def _run(name: str) -> None:
    port, dial_path = TARGETS[name]
    server = await serve_bridge(LISTEN_HOST, port, dial_path)
    async with server:
        await server.serve_forever()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    if len(argv) != 2 or argv[1] not in TARGETS:
        sys.stderr.write(f"usage: shim.py {{{'|'.join(TARGETS)}}}\n")
        return 2
    asyncio.run(_run(argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
