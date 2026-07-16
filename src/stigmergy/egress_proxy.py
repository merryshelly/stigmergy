"""The custom minimal CONNECT-only egress filtering proxy (SPEC.md §4 egress,
bead .11 build spec §1). SECURITY-CRITICAL RUNTIME CHOKEPOINT.

A worker container runs with ``--network=none`` (SPEC §4 worker containment):
it has no route to anything, by physics. Its sole path to the outside world is
a unix domain socket bind-mounted into the container, whose far end is this
proxy, running in the host's normal network namespace (which has real
internet + LAN reach). Every byte a worker sends anywhere passes through the
filtering below first.

**Filtering rules (normative, exhaustive — bead .11 build spec §1.1).** For
each client connection, in order, fail closed at every step (any exception or
parse failure anywhere is a DENY, never an ALLOW):

1. Parse the request line ``CONNECT host:port HTTP/1.1`` (+ headers to the
   blank line). Non-CONNECT methods are refused (v0 is CONNECT-only — HTTPS
   is all a claude-code worker needs). The authority form is de-bracketed
   BEFORE anything else touches ``host`` — ``CONNECT [2606:4700::1]:443``
   must yield host ``2606:4700::1``, not the bracketed text, or the
   IP-literal ban in rule 2 is trivially bypassed.
2. IP-literal denial: if ``host`` parses as an IP address at all (v4 or v6,
   including the de-bracketed v6 form), deny. The allowlist is a hostname
   allowlist, not a "the traffic went somewhere" allowlist.
3. Allowlist: exact (normalized: lowercased, trailing dot stripped) hostname
   membership. No wildcards/suffixes in v0 — a suffix match is a footgun on
   the one place in the whole system that must never default-open.
4. Port allow: only the policy's configured port set (443 by default, plus
   optionally the local-inference port).
5. Self-resolve: the proxy resolves DNS itself (the worker's netns has no
   resolver at all) — via an optional static-hosts map first, else the
   injected/default resolver.
5.5. Resolved-private denial (anti-SSRF / DNS-rebind, CRITICAL): after
   resolving, every candidate address is inspected; if ANY is loopback,
   link-local, RFC1918/ULA-private, multicast, reserved, or unspecified (v4
   AND v6) it is denied — UNLESS it is the exact configured local-inference
   IP (compared as parsed address objects, never as strings). This also
   catches the IPv4-mapped-IPv6 form ``::ffff:a.b.c.d`` (unwrapped to its
   embedded IPv4 before checking, since not every predicate sees through it)
   and the RFC 6052 NAT64 well-known-prefix form ``64:ff9b::a.b.c.d`` (the
   entire prefix is unconditionally reserved, denying it regardless of
   payload; the embedded-IPv4 unwrap is kept for the local-inference-
   exception comparison, not for this denial). The hostname allowlist (rule
   3) says nothing about where a name *actually* resolves; a DNS-poisoning-
   capable worker could otherwise walk an allowlisted name to `10.0.10.x` or
   any other reachable-from-this-host private range.
6. Tunnel: connect to the first surviving resolved address (never
   re-resolve — connecting to the exact vetted IP, not the hostname again,
   is what defeats a DNS-rebind race), reply 200, then pipe bytes
   bidirectionally until either side closes.

DENY replies 403 (or 502 on an upstream connect failure) and closes — no
bytes are ever written to/read from a target on any deny path, because the
target connection is only opened after both ``classify_host`` and
``classify_resolved`` (for at least one surviving address) succeed.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import socket
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address

# Rule 4 default port set: 443 always; policy_for_lane may add the local
# inference port (e.g. 8000) on top of this. classify_host consults
# policy.allowed_ports directly — this constant is just the conventional
# default used when constructing a policy.
DEFAULT_ALLOWED_PORTS: frozenset[int] = frozenset({443})

# Bound on the CONNECT request-line + headers read (rule 1). A minimal
# CONNECT request is a handful of bytes; this is generous but still bounded
# — an attacker cannot force unbounded memory growth trying to smuggle a
# request past the parser.
_MAX_REQUEST_BYTES = 8192
_REQUEST_TIMEOUT_SECONDS = 10.0

_ALLOW_RESPONSE = b"HTTP/1.1 200 Connection Established\r\n\r\n"
_DENY_RESPONSE = b"HTTP/1.1 403 Forbidden\r\n\r\n"
_UPSTREAM_ERROR_RESPONSE = b"HTTP/1.1 502 Bad Gateway\r\n\r\n"


class ProxyRequestError(ValueError):
    """Base for CONNECT request-line parse failures (rule 1). Distinguished
    subclasses let :func:`serve` log the precise DENY reason."""


class MalformedRequest(ProxyRequestError):
    """Not a well-formed ``CONNECT host:port HTTP/1.1`` line."""


class MethodNotAllowed(ProxyRequestError):
    """A non-CONNECT method (GET/POST absolute-URI proxying etc.)."""


@dataclass(frozen=True)
class EgressPolicy:
    """One dispatch's immutable egress policy.

    ``allowed_hosts`` MUST already be normalized (lowercase, no trailing
    dot) by the caller (``policy_for_lane`` does this) — :func:`classify_host`
    re-normalizes the incoming request host before comparing, but does not
    reach into the policy to fix it up.

    ``local_inference_ip`` is the SINGLE private-address exception permitted
    by rule 5.5 — compared as a parsed address object (never a string), and
    nothing broader (not a subnet, not a name) is ever exempted.

    ``static_hosts`` is an optional ``{name: ip}`` map consulted before the
    resolver (for names a real resolver wouldn't serve, e.g. the local
    inference host) — keys should be pre-normalized the same way.
    """

    allowed_hosts: frozenset[str] = frozenset()
    allowed_ports: frozenset[int] = DEFAULT_ALLOWED_PORTS
    local_inference_ip: str | None = None
    static_hosts: dict[str, str] = field(default_factory=dict)


def _normalize_host(host: str) -> str:
    return host.strip().lower().rstrip(".")


def parse_authority(authority: str) -> tuple[str, int]:
    """Parse a CONNECT authority-form ``host:port`` or ``[v6]:port``.

    Bracketed IPv6 is de-bracketed here — this is the one and only place a
    bracket ever gets stripped, and it happens unconditionally before any
    classification. Port is parsed as the field after the LAST unbracketed
    ``:`` (a bare IPv6 host with no brackets and no port is ambiguous/
    malformed on purpose — the CONNECT authority form requires brackets
    around a literal IPv6 host precisely to make the port delimiter
    unambiguous; not a change we can afford to guess our way past).

    Raises :class:`MalformedRequest` on anything that doesn't fit.
    """
    authority = authority.strip()
    if not authority:
        raise MalformedRequest("empty authority")

    if authority.startswith("["):
        end = authority.find("]")
        if end == -1:
            raise MalformedRequest(f"unterminated bracketed host: {authority!r}")
        host = authority[1:end]
        rest = authority[end + 1 :]
        if not rest.startswith(":"):
            raise MalformedRequest(f"bracketed host missing port: {authority!r}")
        port_str = rest[1:]
    else:
        if ":" not in authority:
            raise MalformedRequest(f"authority missing port: {authority!r}")
        host, _, port_str = authority.rpartition(":")

    if not host:
        raise MalformedRequest(f"authority missing host: {authority!r}")
    if not port_str.isdigit():
        raise MalformedRequest(f"authority port not numeric: {authority!r}")
    port = int(port_str)
    if not (0 < port <= 65535):
        raise MalformedRequest(f"authority port out of range: {authority!r}")
    return host, port


def parse_connect_request(request_line: str) -> tuple[str, int]:
    """Parse the full CONNECT request line (rule 1).

    Raises :class:`MethodNotAllowed` for a non-CONNECT method, or
    :class:`MalformedRequest` for anything else that doesn't fit
    ``CONNECT host:port HTTP/1.1`` exactly (three space-separated tokens,
    an HTTP/-prefixed version).
    """
    parts = request_line.strip().split()
    if len(parts) != 3:
        raise MalformedRequest(f"expected 3 tokens, got {request_line!r}")
    method, authority, version = parts
    if method != "CONNECT":
        raise MethodNotAllowed(method)
    if not version.upper().startswith("HTTP/"):
        raise MalformedRequest(f"not an HTTP version token: {version!r}")
    return parse_authority(authority)


def classify_host(policy: EgressPolicy, host: str, port: int) -> tuple[bool, str]:
    """Rules 2-4 (pre-resolution): IP-literal ban, allowlist, port allow.

    Pure. Returns ``(allowed, reason)`` — ``reason`` is ``"allow"`` on
    success, else one of ``"malformed"``, ``"ip-literal"``,
    ``"not-allowlisted"``, ``"port-not-allowed"``. Never raises — any
    unexpected input is treated as malformed (fail closed).
    """
    try:
        normalized = _normalize_host(host)
        if not normalized:
            return False, "malformed"

        # Rule 2: IP-literal denial (covers v4 and v6, incl. any de-bracketed
        # v6 form already handled upstream by parse_authority).
        try:
            ipaddress.ip_address(normalized)
        except ValueError:
            pass
        else:
            return False, "ip-literal"

        # Rule 3: exact allowlist membership only (no wildcards/suffixes).
        if normalized not in policy.allowed_hosts:
            return False, "not-allowlisted"

        # Rule 4: fixed port set.
        if port not in policy.allowed_ports:
            return False, "port-not-allowed"

        return True, "allow"
    except Exception:  # noqa: BLE001 - fail closed on literally anything
        return False, "malformed"


# NAT64 well-known prefix (RFC 6052): a synthesized IPv6 address embedding an
# IPv4 address in its last 32 bits. It happens to fall inside the historical
# ``::/8`` reserved block, so :mod:`ipaddress`'s ``is_reserved`` already
# denies the WHOLE prefix unconditionally (conservative: every address in
# it is denied regardless of what it embeds, public or private) — this
# explicit unwrap is defense-in-depth, not the thing doing that denial. What
# it IS load-bearing for: making the local-inference exact-IP comparison
# (rule 5.5's one exception) correct even if a resolver ever handed back the
# local-inference address in this synthesized form instead of plain IPv4.
_NAT64_WELL_KNOWN_PREFIX = ipaddress.ip_network("64:ff9b::/96")


def _mapped_v4_candidates(addr: IPAddress) -> list[IPAddress]:
    """Every address form worth checking for ``addr`` — itself, plus its
    embedded IPv4 if it is an IPv4-mapped-IPv6 address (``::ffff:a.b.c.d``)
    or a NAT64 well-known-prefix address (``64:ff9b::a.b.c.d``).

    Defense in depth: :mod:`ipaddress`'s ``is_private``/``is_loopback``/etc.
    predicates already account for IPv4-mapped addresses on this Python
    version, but checking the unwrapped v4 form explicitly does not depend
    on that continuing to hold, and makes the local-inference exact-IP
    comparison (rule 5.5) correct regardless of which form a resolver hands
    back. (The NAT64 well-known prefix is separately, and unconditionally,
    denied by ``is_reserved`` — see the module-level comment above; the
    unwrap here exists for the local-inference-exception correctness case,
    not to manufacture that denial.)
    """
    candidates: list[IPAddress] = [addr]
    if isinstance(addr, ipaddress.IPv6Address):
        if addr.ipv4_mapped is not None:
            candidates.append(addr.ipv4_mapped)
        elif addr in _NAT64_WELL_KNOWN_PREFIX:
            candidates.append(ipaddress.IPv4Address(addr.packed[-4:]))
    return candidates


def classify_resolved(policy: EgressPolicy, ip: str) -> tuple[bool, str]:
    """Rule 5.5 (post-resolution): resolved-private / anti-SSRF denial.

    Pure. Denies if ANY form of the resolved address (including its
    IPv4-mapped-IPv6 unwrapping) is loopback, link-local, private
    (RFC1918/ULA), multicast, reserved, or unspecified — UNLESS it is the
    exact configured ``policy.local_inference_ip`` (compared as parsed
    address objects). An unparseable ``ip`` fails closed as
    ``"resolved-private"`` — it should never happen (resolvers only hand
    back valid address text), but "we could not verify this is safe" must
    never resolve to "allow".
    """
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False, "resolved-private"

    candidates = _mapped_v4_candidates(addr)

    if policy.local_inference_ip:
        try:
            allowed = ipaddress.ip_address(policy.local_inference_ip)
        except ValueError:
            allowed = None
        if allowed is not None:
            allowed_candidates = _mapped_v4_candidates(allowed)
            if any(c == a for c in candidates for a in allowed_candidates):
                return True, "allow"

    for c in candidates:
        if (
            c.is_loopback
            or c.is_link_local
            or c.is_private
            or c.is_multicast
            or c.is_reserved
            or c.is_unspecified
        ):
            return False, "resolved-private"

    return True, "allow"


Resolver = Callable[[str, int], Awaitable[list[str]]]


async def _default_resolve(host: str, port: int) -> list[str]:
    """Resolve ``host`` via the running loop's async ``getaddrinfo`` (which
    offloads to a thread pool — never blocks the event loop) and return the
    distinct textual addresses, in the order returned."""
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    seen: list[str] = []
    for info in infos:
        addr = info[4][0]
        if addr not in seen:
            seen.append(addr)
    return seen


def _log(log_path: str | Path, entry: dict[str, Any]) -> None:
    """Append one JSONL line. Small individual writes to an append-mode fd
    are atomic on POSIX (below PIPE_BUF), so concurrent connections logging
    concurrently do not interleave partial lines."""
    line = json.dumps(entry, sort_keys=True)
    with open(log_path, "a") as fh:
        fh.write(line + "\n")


async def _deny(
    writer: asyncio.StreamWriter,
    *,
    log_path: str | Path,
    dispatch_id: str | None,
    host: str | None,
    port: int | None,
    reason: str,
    resolved_ip: str | None = None,
    response: bytes = _DENY_RESPONSE,
) -> None:
    _log(
        log_path,
        {
            "ts": time.time(),
            "dispatch_id": dispatch_id,
            "host": host,
            "port": port,
            "decision": "deny",
            "reason": reason,
            "resolved_ip": resolved_ip,
        },
    )
    try:
        writer.write(response)
        await writer.drain()
    except OSError:
        pass
    finally:
        writer.close()


async def _handle_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    policy: EgressPolicy,
    log_path: str | Path,
    resolver: Resolver,
    dispatch_id: str | None,
) -> None:
    # --- rule 1: read + parse the request line -----------------------------
    try:
        raw = await asyncio.wait_for(
            reader.readuntil(b"\r\n\r\n"), timeout=_REQUEST_TIMEOUT_SECONDS
        )
    except (
        asyncio.IncompleteReadError,
        asyncio.LimitOverrunError,
        TimeoutError,
        OSError,
    ):
        await _deny(
            writer,
            log_path=log_path,
            dispatch_id=dispatch_id,
            host=None,
            port=None,
            reason="malformed",
        )
        return

    text = raw.decode("latin-1", errors="replace")
    request_line = text.split("\r\n", 1)[0]

    try:
        host, port = parse_connect_request(request_line)
    except MethodNotAllowed:
        await _deny(
            writer,
            log_path=log_path,
            dispatch_id=dispatch_id,
            host=None,
            port=None,
            reason="method-not-allowed",
        )
        return
    except ProxyRequestError:
        await _deny(
            writer,
            log_path=log_path,
            dispatch_id=dispatch_id,
            host=None,
            port=None,
            reason="malformed",
        )
        return

    # --- rules 2-4: pure pre-resolution classification ----------------------
    allowed, reason = classify_host(policy, host, port)
    if not allowed:
        await _deny(
            writer,
            log_path=log_path,
            dispatch_id=dispatch_id,
            host=host,
            port=port,
            reason=reason,
        )
        return

    normalized_host = _normalize_host(host)

    # --- rule 5: self-resolve (static hosts map first, else the resolver) --
    static_ip = policy.static_hosts.get(normalized_host)
    if static_ip is not None:
        addresses = [static_ip]
    else:
        try:
            addresses = await resolver(normalized_host, port)
        except Exception:  # noqa: BLE001 - any resolver failure is fail-closed
            addresses = []

    if not addresses:
        await _deny(
            writer,
            log_path=log_path,
            dispatch_id=dispatch_id,
            host=host,
            port=port,
            reason="resolve-failed",
        )
        return

    # --- rule 5.5: resolved-private / anti-SSRF denial ----------------------
    survivor: str | None = None
    any_private = False
    for candidate_ip in addresses:
        ok, _ = classify_resolved(policy, candidate_ip)
        if ok:
            survivor = candidate_ip
            break
        any_private = True

    if survivor is None:
        await _deny(
            writer,
            log_path=log_path,
            dispatch_id=dispatch_id,
            host=host,
            port=port,
            reason="resolved-private" if any_private else "resolve-failed",
        )
        return

    # --- rule 6: connect to the exact vetted address (never re-resolve) ----
    try:
        target_reader, target_writer = await asyncio.open_connection(survivor, port)
    except OSError:
        await _deny(
            writer,
            log_path=log_path,
            dispatch_id=dispatch_id,
            host=host,
            port=port,
            reason="upstream-error",
            resolved_ip=survivor,
            response=_UPSTREAM_ERROR_RESPONSE,
        )
        return

    _log(
        log_path,
        {
            "ts": time.time(),
            "dispatch_id": dispatch_id,
            "host": host,
            "port": port,
            "decision": "allow",
            "reason": "allow",
            "resolved_ip": survivor,
        },
    )

    try:
        writer.write(_ALLOW_RESPONSE)
        await writer.drain()
        await _pipe_bidirectional(reader, writer, target_reader, target_writer)
    finally:
        target_writer.close()
        writer.close()


async def _pipe_one_direction(
    src: asyncio.StreamReader, dst: asyncio.StreamWriter
) -> None:
    try:
        while True:
            chunk = await src.read(65536)
            if not chunk:
                break
            dst.write(chunk)
            await dst.drain()
    except (ConnectionResetError, BrokenPipeError, OSError):
        pass
    finally:
        try:
            dst.write_eof()
        except (RuntimeError, OSError):
            pass


async def _pipe_bidirectional(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    target_reader: asyncio.StreamReader,
    target_writer: asyncio.StreamWriter,
) -> None:
    await asyncio.gather(
        _pipe_one_direction(client_reader, target_writer),
        _pipe_one_direction(target_reader, client_writer),
        return_exceptions=True,
    )


async def serve(
    socket_path: str | Path,
    policy: EgressPolicy,
    log_path: str | Path,
    *,
    resolver: Resolver = _default_resolve,
    dispatch_id: str | None = None,
) -> asyncio.base_events.Server:
    """Start the CONNECT filtering proxy listening on a unix domain socket.

    ``resolver`` is injectable (an ``async def(host, port) -> list[str]``)
    so tests can supply a deterministic, offline fake instead of the real
    ``getaddrinfo`` — this is what makes the resolved-private (anti-SSRF)
    tests reproducible without network access. ``dispatch_id`` is carried
    through into every JSONL log line for correlation; it is not itself a
    filtering input.

    Returns the running :class:`asyncio.Server` (already listening — the
    caller manages its lifecycle: fail-closed shutdown per bead .11 build
    spec §1.3 is "stop the server / let the process die", after which any
    client connect on ``socket_path`` fails outright — no egress, ever,
    without a live proxy in front of it).
    """
    socket_path = Path(socket_path)
    if socket_path.exists():
        socket_path.unlink()

    async def _on_connect(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            await _handle_connection(
                reader,
                writer,
                policy=policy,
                log_path=log_path,
                resolver=resolver,
                dispatch_id=dispatch_id,
            )
        except Exception:  # noqa: BLE001 - a handler bug must not leak a tunnel
            try:
                writer.close()
            except Exception:  # noqa: BLE001
                pass

    server = await asyncio.start_unix_server(
        _on_connect, path=str(socket_path), limit=_MAX_REQUEST_BYTES
    )
    return server
