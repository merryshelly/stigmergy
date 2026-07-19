"""Per-dispatch egress orchestration (SPEC.md §4 egress, bead .11 build spec §2).

Wires a dispatch's charter lane to a running :func:`stigmergy.egress_proxy.serve`
instance: resolve the lane's egress groups into an :class:`EgressPolicy`
(pure, fail-closed), start the proxy listening on a unix socket under the
dispatch's runtime directory, and hand back a small handle the caller uses to
point the worker container at the socket (:func:`stigmergy.container.build_run_argv`
``egress_socket=`` param) and to tear the proxy down at the end of the
dispatch.

**Fail-closed lifecycle (SPEC §4):** if the proxy is not running — never
started, crashed, or torn down — the worker's connect to the (missing or
dead) socket simply fails; there is no code path here that falls back to
"open" egress. A dispatch that cannot reach its egress proxy ends `infra`,
never runs unfiltered.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stigmergy.container import dispatch_socket_path
from stigmergy.egress_proxy import (
    DEFAULT_ALLOWED_PORTS,
    EgressPolicy,
    Resolver,
    _default_resolve,
    serve,
)

_START_TIMEOUT_SECONDS = 10.0
_STOP_TIMEOUT_SECONDS = 10.0


class EgressError(Exception):
    """Raised when the egress proxy fails to start (fail closed — the
    caller must treat this as an infra condition, never proceed without
    egress filtering in place)."""


def _normalize_host(host: str) -> str:
    return host.strip().lower().rstrip(".")


def policy_for_lane(
    charter_resolved: Mapping[str, Any],
    lane_name: str,
    *,
    local_inference: Mapping[str, Any] | None = None,
) -> EgressPolicy:
    """Resolve one lane's egress policy from a fully-merged charter (PURE).

    Walks ``charter_resolved["lanes"][lane_name]["egress"]`` (a list of
    egress *group* names) through ``charter_resolved["egress"][<group>].hosts``
    to build the allowlist. **Fails closed on every unknown/missing shape**:
    an unknown lane, a lane with no ``egress`` list, a referenced group
    absent from ``charter_resolved["egress"]``, or a malformed hosts list
    all contribute NOTHING to the allowlist — never a wildcard, never "all
    hosts". A lane that ends up with zero resolvable egress groups gets the
    empty-allowlist :class:`EgressPolicy` (deny-all), not an allow-all
    fallback.

    ``local_inference``, if given, is a small mapping supplying the rule-5.5
    private-address exception: ``{"ip": "10.0.20.104", "port": 8000}``. Its
    ``ip`` becomes :attr:`EgressPolicy.local_inference_ip` (the ONE private
    address :func:`stigmergy.egress_proxy.classify_resolved` will ever
    allow); its optional ``port`` is added to the allowed port set on top of
    the fixed default ({443}). Supplying ``local_inference`` does **not** by
    itself grant any host access — the local-inference hostname (e.g.
    ``macstudio.local``) must still appear in one of the lane's resolved
    egress-group host lists exactly like any other allowlisted name (SPEC
    §4: it is reached by NAME, never by IP-literal, so rule 2's IP-literal
    ban stays intact).

    **Note on an `"ip:port"`-shaped `hosts` entry** (e.g. the pre-.11 charter
    fixture's ``"10.0.20.104:8000"`` under ``[egress.inference].hosts``):
    this function does not special-case or reject that shape — it is
    absorbed into ``allowed_hosts`` verbatim (normalized, like any other
    string). This is inert, not a bypass: a real ``CONNECT`` authority
    ``10.0.20.104:8000`` parses to host ``"10.0.20.104"`` (no port suffix,
    per :func:`stigmergy.egress_proxy.parse_authority`) which never equals
    the stored ``"10.0.20.104:8000"`` string, AND even if it somehow did
    match, rule 2 (IP-literal ban) runs BEFORE the allowlist check (rule 3)
    in :func:`stigmergy.egress_proxy.classify_host` — an IP-literal host is
    denied regardless of allowlist membership. So this legacy-shaped entry
    is simply dead config; local inference must be reached by name (with
    ``local_inference_ip``/``static_hosts`` supplying the actual IP), not by
    stuffing an `ip:port` string into a hosts list. A future ticket may
    choose to validate/reject this shape at the charter layer; .11 does not
    widen any access because of it.
    """
    lanes = charter_resolved.get("lanes") if isinstance(charter_resolved, Mapping) else None
    lane_cfg = lanes.get(lane_name) if isinstance(lanes, Mapping) else None

    allowed_ports = set(DEFAULT_ALLOWED_PORTS)
    local_inference_ip: str | None = None
    if isinstance(local_inference, Mapping):
        ip = local_inference.get("ip")
        if isinstance(ip, str) and ip:
            local_inference_ip = ip
        port = local_inference.get("port")
        if isinstance(port, int) and not isinstance(port, bool):
            allowed_ports.add(port)

    if not isinstance(lane_cfg, Mapping):
        # Unknown lane -> empty allowlist (deny-all), never allow-all.
        return EgressPolicy(allowed_ports=frozenset(allowed_ports))

    group_names = lane_cfg.get("egress")
    if not isinstance(group_names, list):
        # Lane defined but carries no `egress` list -> deny-all.
        return EgressPolicy(allowed_ports=frozenset(allowed_ports))

    egress_table = charter_resolved.get("egress")
    hosts: set[str] = set()
    for group_name in group_names:
        if not isinstance(egress_table, Mapping) or not isinstance(group_name, str):
            continue
        group_cfg = egress_table.get(group_name)
        if not isinstance(group_cfg, Mapping):
            # Unknown egress group referenced by the lane -> contributes
            # nothing (fail closed per-group, not a hard error — a lane
            # with at least one OTHER valid group still gets those hosts).
            continue
        group_hosts = group_cfg.get("hosts")
        if not isinstance(group_hosts, list):
            continue
        for raw_host in group_hosts:
            if isinstance(raw_host, str) and raw_host:
                hosts.add(_normalize_host(raw_host))

    return EgressPolicy(
        allowed_hosts=frozenset(hosts),
        allowed_ports=frozenset(allowed_ports),
        local_inference_ip=local_inference_ip,
    )


@dataclass
class EgressHandle:
    """A running per-dispatch egress proxy.

    ``socket_path``/``log_path`` are what the caller feeds onward:
    ``socket_path`` into ``container.build_run_argv(..., egress_socket=...)``,
    ``log_path`` into whatever reads the per-dispatch JSONL egress log.
    """

    dispatch_id: str
    socket_path: Path
    log_path: Path
    policy: EgressPolicy
    _loop: asyncio.AbstractEventLoop
    _thread: threading.Thread
    _stopped: bool = False

    def stop(self) -> None:
        """Stop serving and remove the socket file (idempotent).

        After this returns, any client `connect()` to ``socket_path`` fails
        outright (no listener, no file) — the fail-closed proxy-down
        guarantee (SPEC §4 / bead .11 build spec §1.3).
        """
        if self._stopped:
            return
        self._stopped = True

        def _shutdown() -> None:
            self._loop.stop()

        try:
            self._loop.call_soon_threadsafe(_shutdown)
        except RuntimeError:
            pass  # loop already stopped/closed
        self._thread.join(timeout=_STOP_TIMEOUT_SECONDS)

        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass

    def teardown(self) -> None:
        """Alias for :meth:`stop` — full stop + socket removal."""
        self.stop()


def setup_dispatch_egress(
    dispatch_id: str,
    policy: EgressPolicy,
    runtime_dir: str | Path,
    *,
    resolver: Resolver = _default_resolve,
) -> EgressHandle:
    """Start the egress proxy for one dispatch and return its handle.

    Creates ``<runtime_dir>/egress-<hash>.sock`` (the mount target for
    ``container.build_run_argv(..., egress_socket=...)``; the filename is a
    fixed-length hash of ``dispatch_id`` to stay under the AF_UNIX path
    limit — see :func:`stigmergy.container.dispatch_socket_path`) and
    ``<runtime_dir>/egress-<dispatch_id>.jsonl`` (the per-attempt log),
    starts :func:`stigmergy.egress_proxy.serve` on a dedicated background
    event loop thread, and blocks until it is actually listening (or raises
    :class:`EgressError` if it never comes up) — callers never get back a
    handle whose socket doesn't exist yet.
    """
    runtime_dir = Path(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    # Fixed-length socket filename: the full dispatch_id can overflow the
    # AF_UNIX path limit (see container.dispatch_socket_path). The .jsonl log
    # keeps the readable id (no length limit) for operator correlation.
    socket_path = dispatch_socket_path(runtime_dir, "egress", dispatch_id)
    log_path = runtime_dir / f"egress-{dispatch_id}.jsonl"

    ready = threading.Event()
    failure: dict[str, BaseException] = {}
    loop_box: dict[str, asyncio.AbstractEventLoop] = {}

    def _run() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop_box["loop"] = loop
        try:
            server = loop.run_until_complete(
                serve(socket_path, policy, log_path, resolver=resolver, dispatch_id=dispatch_id)
            )
        except BaseException as exc:  # noqa: BLE001 - relay to the starting thread
            failure["error"] = exc
            ready.set()
            loop.close()
            return

        ready.set()
        try:
            loop.run_forever()
        finally:
            server.close()
            loop.run_until_complete(server.wait_closed())
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    thread = threading.Thread(
        target=_run, name=f"egress-proxy-{dispatch_id}", daemon=True
    )
    thread.start()

    if not ready.wait(timeout=_START_TIMEOUT_SECONDS):
        raise EgressError(
            f"egress proxy for dispatch {dispatch_id!r} did not start within "
            f"{_START_TIMEOUT_SECONDS}s"
        )
    if "error" in failure:
        raise EgressError(
            f"egress proxy for dispatch {dispatch_id!r} failed to start"
        ) from failure["error"]

    return EgressHandle(
        dispatch_id=dispatch_id,
        socket_path=socket_path,
        log_path=log_path,
        policy=policy,
        _loop=loop_box["loop"],
        _thread=thread,
    )


def teardown(handle: EgressHandle) -> None:
    """Module-level convenience wrapper: stop ``handle``'s proxy and remove
    its socket. Equivalent to ``handle.stop()``."""
    handle.stop()
