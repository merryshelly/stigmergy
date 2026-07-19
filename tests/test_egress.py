"""Tests for stigmergy.egress_proxy / stigmergy.egress (SPEC.md §4 egress,
bead .11 build spec §3 AC3 — this case-list is orchestrator-frozen; test
authoring must match it, never weaken an assertion to force a pass).

Case numbering below matches the build spec's §3 list exactly:

  Pure classify_host() (rules 1-4, the security core):
    1.  exact allowlisted host on 443 -> allow
    2.  non-allowlisted host -> deny not-allowlisted
    3.  IPv4 + IPv6 literal (incl. bracketed authority form) -> deny ip-literal
    4.  disallowed port -> deny port-not-allowed
    5.  empty allowlist -> deny-all (fail-closed default)
    6.  local inference by NAME allowed, but its IP as a literal denied
    7.  case/trailing-dot normalization

  Pure classify_resolved() (rule 5.5, anti-SSRF/DNS-rebind, CRITICAL):
    7a. resolved-private (RFC1918/ULA/link-local/multicast/loopback) denied
    7b. local-inference EXACT ip allowed; same name -> different private ip denied
    7c. resolved public ip allowed
    7d. mixed addresses: connect only to public survivors; all-private denied

  Live proxy over a real unix socket (real asyncio server + real client):
    8.  tunnels an allowlisted real host (network, skip cleanly if unavailable)
    9.  denies non-allowlisted + logs, no bytes tunneled
    10. denies ip-literal + logs
    11. fail-closed on stop (socket gone / connection refused)
    12. never pipes on deny (sentinel target listener sees zero connections)

  Structural isolation (podman, --network=none):
    13. zero direct egress from a --network=none container (skip if no podman)

  build_run_argv integration:
    14. egress_socket=<path> appends exactly one --volume=...:/run/egress.sock:rw
    15. egress_socket=None -> byte-identical to pre-.11 argv (regression guard)
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

import pytest

from stigmergy.container import ContainerProfile, build_run_argv
from stigmergy.egress import EgressHandle, policy_for_lane, setup_dispatch_egress
from stigmergy.egress_proxy import (
    EgressPolicy,
    classify_host,
    classify_resolved,
    serve,
)

PODMAN = shutil.which("podman")
requires_podman = pytest.mark.skipif(PODMAN is None, reason="podman not installed")

_REAL_HOST_FOR_LIVE_TEST = "example.com"


def _network_available(host: str = _REAL_HOST_FOR_LIVE_TEST, port: int = 443) -> bool:
    try:
        with socket.create_connection((host, port), timeout=3):
            return True
    except OSError:
        return False


requires_network = pytest.mark.skipif(
    not _network_available(), reason="no outbound network in this sandbox"
)


# --------------------------------------------------------------------------
# 1-7: pure classify_host() (rules 1-4)
# --------------------------------------------------------------------------


def test_decide_allows_exact_allowlisted_host_on_443():
    policy = EgressPolicy(allowed_hosts=frozenset({"api.anthropic.com"}))
    allowed, reason = classify_host(policy, "api.anthropic.com", 443)
    assert allowed is True
    assert reason == "allow"


def test_decide_denies_non_allowlisted_host():
    policy = EgressPolicy(allowed_hosts=frozenset({"api.anthropic.com"}))
    allowed, reason = classify_host(policy, "evil.example.com", 443)
    assert allowed is False
    assert reason == "not-allowlisted"


def test_decide_denies_ipv4_literal():
    policy = EgressPolicy(allowed_hosts=frozenset({"1.2.3.4"}))  # even if "allowlisted" by text
    allowed, reason = classify_host(policy, "1.2.3.4", 443)
    assert allowed is False
    assert reason == "ip-literal"


def test_decide_denies_ipv6_literal():
    policy = EgressPolicy(allowed_hosts=frozenset({"2606:4700::1"}))
    allowed, reason = classify_host(policy, "2606:4700::1", 443)
    assert allowed is False
    assert reason == "ip-literal"


def test_decide_denies_ipv6_literal_bracketed_authority_form():
    # The bracketed CONNECT authority form must be de-bracketed before the
    # ip-literal check runs, or the ban is trivially bypassed by bracketing.
    from stigmergy.egress_proxy import parse_authority

    host, port = parse_authority("[2606:4700::1]:443")
    assert host == "2606:4700::1"
    assert port == 443
    policy = EgressPolicy(allowed_hosts=frozenset({"2606:4700::1"}))
    allowed, reason = classify_host(policy, host, port)
    assert allowed is False
    assert reason == "ip-literal"


def test_decide_denies_disallowed_port():
    policy = EgressPolicy(allowed_hosts=frozenset({"api.anthropic.com"}))
    allowed, reason = classify_host(policy, "api.anthropic.com", 8080)
    assert allowed is False
    assert reason == "port-not-allowed"


def test_decide_denies_empty_allowlist_deny_all():
    # Fail-closed default: an EgressPolicy with no allowlist denies everything,
    # never falls back to allow-all.
    policy = EgressPolicy()
    allowed, reason = classify_host(policy, "api.anthropic.com", 443)
    assert allowed is False
    assert reason == "not-allowlisted"


def test_decide_local_inference_by_name_allowed_but_ip_denied():
    policy = EgressPolicy(
        allowed_hosts=frozenset({"macstudio.local"}),
        allowed_ports=frozenset({443, 8000}),
        local_inference_ip="10.0.20.104",
    )
    allowed, reason = classify_host(policy, "macstudio.local", 8000)
    assert allowed is True
    assert reason == "allow"

    # The IP itself, as a CONNECT literal, is still denied -- rule 2 (ip-literal)
    # does not care that it happens to equal the configured local-inference IP;
    # only rule 5.5 (post-resolution) grants that exception, and only for a
    # resolved address, never for a literal in the request line.
    denied, reason2 = classify_host(policy, "10.0.20.104", 8000)
    assert denied is False
    assert reason2 == "ip-literal"


def test_decide_case_and_trailing_dot_normalized():
    policy = EgressPolicy(allowed_hosts=frozenset({"api.anthropic.com"}))
    allowed, reason = classify_host(policy, "API.Anthropic.COM.", 443)
    assert allowed is True
    assert reason == "allow"


# --------------------------------------------------------------------------
# 7a-7d: pure classify_resolved() (rule 5.5 -- anti-SSRF, CRITICAL)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "private_ip",
    [
        "10.0.10.5",
        "192.168.1.1",
        "172.16.0.1",
        "127.0.0.1",
        "fd00::1",
        "fe80::1",
        "ff02::1",  # multicast
    ],
)
def test_resolved_private_rfc1918_denied(private_ip):
    policy = EgressPolicy(allowed_hosts=frozenset({"api.example.com"}))
    allowed, reason = classify_resolved(policy, private_ip)
    assert allowed is False, f"{private_ip} must be denied"
    assert reason == "resolved-private"


def test_resolved_private_allows_local_inference_exact_ip():
    policy = EgressPolicy(
        allowed_hosts=frozenset({"macstudio.local"}),
        local_inference_ip="10.0.20.104",
    )
    allowed, reason = classify_resolved(policy, "10.0.20.104")
    assert allowed is True
    assert reason == "allow"

    # Same name resolving to a DIFFERENT private ip is still denied -- the
    # exception is IP-exact, not name-based.
    denied, reason2 = classify_resolved(policy, "10.0.10.5")
    assert denied is False
    assert reason2 == "resolved-private"


def test_resolved_public_allowed():
    policy = EgressPolicy(allowed_hosts=frozenset({"api.anthropic.com"}))
    allowed, reason = classify_resolved(policy, "104.20.23.154")
    assert allowed is True
    assert reason == "allow"


def test_resolved_mixed_addresses_denies_if_any_private():
    # Decide policy: never connect to a private address except the exact
    # local-inference IP. Given [public, private], connect only to the
    # public survivor; a name resolving ONLY to private (non-exception)
    # addresses is denied outright.
    policy = EgressPolicy(allowed_hosts=frozenset({"api.example.com"}))

    addrs_mixed = ["104.20.23.154", "10.0.10.5"]
    results = [classify_resolved(policy, ip) for ip in addrs_mixed]
    survivors = [ip for ip, (ok, _) in zip(addrs_mixed, results, strict=True) if ok]
    assert survivors == ["104.20.23.154"]

    addrs_all_private = ["10.0.10.5", "192.168.1.1"]
    results_all_private = [classify_resolved(policy, ip) for ip in addrs_all_private]
    assert all(ok is False and reason == "resolved-private" for ok, reason in results_all_private)


def test_resolved_ipv4_mapped_ipv6_private_denied():
    # ::ffff:10.0.10.5 embeds a private IPv4 -- must be denied like the plain
    # v4 form (IPv4-mapped-IPv6 is a documented obfuscation/bypass vector).
    policy = EgressPolicy(allowed_hosts=frozenset({"api.example.com"}))
    allowed, reason = classify_resolved(policy, "::ffff:10.0.10.5")
    assert allowed is False
    assert reason == "resolved-private"


def test_resolved_nat64_embedded_private_ipv4_denied():
    # 64:ff9b::/96 (RFC 6052 NAT64 well-known prefix) embeds an IPv4 address
    # in its last 32 bits. It falls under the historical ::/8 reserved block,
    # so ipaddress's own is_reserved already denies the WHOLE prefix
    # (conservative — every address in it is denied regardless of payload,
    # public or private); the explicit unwrap in _mapped_v4_candidates is
    # kept as defense-in-depth documentation (not depended on for this
    # specific denial) and so the local-inference exact-IP comparison stays
    # correct if a resolver ever handed back this form for that address.
    policy = EgressPolicy(allowed_hosts=frozenset({"api.example.com"}))
    allowed, reason = classify_resolved(policy, "64:ff9b::a00:a05")  # embeds 10.0.10.5
    assert allowed is False
    assert reason == "resolved-private"

    # A NAT64-embedded PUBLIC-looking payload is ALSO denied -- the whole
    # well-known prefix is reserved, so this is not a case this proxy will
    # ever ALLOW (fail-closed: an obscure but reachable address family is
    # denied outright rather than trusted to carry only public traffic).
    denied_public, reason_public = classify_resolved(policy, "64:ff9b::808:808")  # embeds 8.8.8.8
    assert denied_public is False
    assert reason_public == "resolved-private"


def test_resolved_decimal_ip_obfuscation_not_an_ip_so_not_allowlisted():
    # Decimal-encoded IPv4 (e.g. 2130706433 for 127.0.0.1) is not recognized
    # by ipaddress.ip_address() as an IP at all, so classify_host would fall
    # through to the allowlist check (and be denied there, since it is not a
    # literal hostname anyone allowlists) -- prove it is never silently
    # treated as a private-exempt or otherwise-allowed literal.
    import ipaddress

    with pytest.raises(ValueError):
        ipaddress.ip_address("2130706433")


# --------------------------------------------------------------------------
# 8-12: live proxy over a real unix socket
# --------------------------------------------------------------------------


async def _send_connect(socket_path: Path, authority: str) -> tuple[bytes, asyncio.StreamWriter]:
    reader, writer = await asyncio.open_unix_connection(path=str(socket_path))
    writer.write(f"CONNECT {authority} HTTP/1.1\r\nHost: {authority}\r\n\r\n".encode())
    await writer.drain()
    status_line = await reader.readline()
    return status_line, writer


def _read_jsonl(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    lines = log_path.read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


@requires_network
def test_proxy_tunnels_allowlisted_host(tmp_path):
    async def _run():
        socket_path = tmp_path / "egress8.sock"
        log_path = tmp_path / "egress8.jsonl"
        policy = EgressPolicy(
            allowed_hosts=frozenset({_REAL_HOST_FOR_LIVE_TEST}), allowed_ports=frozenset({443})
        )
        server = await serve(socket_path, policy, log_path)
        try:
            status_line, writer = await _send_connect(
                socket_path, f"{_REAL_HOST_FOR_LIVE_TEST}:443"
            )
            assert status_line.startswith(b"HTTP/1.1 200")

            import ssl

            ctx = ssl.create_default_context()
            reader = asyncio.StreamReader()
            protocol = asyncio.StreamReaderProtocol(reader)
            loop = asyncio.get_running_loop()
            transport = writer.transport
            new_transport = await loop.start_tls(
                transport, protocol, ctx, server_side=False,
                server_hostname=_REAL_HOST_FOR_LIVE_TEST,
            )
            tls_writer = asyncio.StreamWriter(new_transport, protocol, reader, loop)
            tls_writer.write(
                f"GET / HTTP/1.1\r\nHost: {_REAL_HOST_FOR_LIVE_TEST}\r\n"
                "Connection: close\r\n\r\n".encode()
            )
            await tls_writer.drain()
            first_bytes = await asyncio.wait_for(reader.read(64), timeout=10)
            assert first_bytes.startswith(b"HTTP/1.1")
            tls_writer.close()
        finally:
            server.close()
            await server.wait_closed()

        entries = _read_jsonl(log_path)
        allow_entries = [e for e in entries if e["decision"] == "allow"]
        assert len(allow_entries) == 1
        assert allow_entries[0]["host"] == _REAL_HOST_FOR_LIVE_TEST
        assert allow_entries[0]["resolved_ip"]

    asyncio.run(_run())


def test_proxy_denies_non_allowlisted_and_logs(tmp_path):
    async def _run():
        socket_path = tmp_path / "egress9.sock"
        log_path = tmp_path / "egress9.jsonl"
        policy = EgressPolicy(allowed_hosts=frozenset({"api.anthropic.com"}))
        server = await serve(socket_path, policy, log_path)
        try:
            status_line, writer = await _send_connect(socket_path, "evil.example.com:443")
            assert status_line.startswith(b"HTTP/1.1 403")
            writer.close()
        finally:
            server.close()
            await server.wait_closed()

        entries = _read_jsonl(log_path)
        deny_entries = [e for e in entries if e["decision"] == "deny"]
        assert len(deny_entries) == 1
        assert deny_entries[0]["reason"] == "not-allowlisted"
        assert deny_entries[0]["host"] == "evil.example.com"

    asyncio.run(_run())


def test_proxy_denies_ip_literal_and_logs(tmp_path):
    async def _run():
        socket_path = tmp_path / "egress10.sock"
        log_path = tmp_path / "egress10.jsonl"
        policy = EgressPolicy(allowed_hosts=frozenset({"1.2.3.4"}))
        server = await serve(socket_path, policy, log_path)
        try:
            status_line, writer = await _send_connect(socket_path, "1.2.3.4:443")
            assert status_line.startswith(b"HTTP/1.1 403")
            writer.close()
        finally:
            server.close()
            await server.wait_closed()

        entries = _read_jsonl(log_path)
        deny_entries = [e for e in entries if e["decision"] == "deny"]
        assert len(deny_entries) == 1
        assert deny_entries[0]["reason"] == "ip-literal"

    asyncio.run(_run())


def test_proxy_fail_closed_on_stop(tmp_path):
    async def _run():
        socket_path = tmp_path / "egress11.sock"
        log_path = tmp_path / "egress11.jsonl"
        policy = EgressPolicy(allowed_hosts=frozenset({"api.anthropic.com"}))
        server = await serve(socket_path, policy, log_path)
        server.close()
        await server.wait_closed()
        socket_path.unlink(missing_ok=True)

        with pytest.raises((ConnectionRefusedError, FileNotFoundError, OSError)):
            await asyncio.open_unix_connection(path=str(socket_path))

    asyncio.run(_run())


def test_proxy_never_pipes_on_deny(tmp_path):
    async def _run():
        socket_path = tmp_path / "egress12.sock"
        log_path = tmp_path / "egress12.jsonl"

        # A sentinel "target" TCP listener -- if a deny path ever opened a real
        # upstream connection to the resolver's address, this Event fires. The
        # fake resolver points AT the sentinel, so a connect-before-classify
        # regression is observable. (Previously the callback was a no-op and
        # `connections_seen` was never mutated -> the assertion was vacuous.)
        connected = asyncio.Event()

        async def _on_sentinel(reader, writer):
            connected.set()
            writer.close()

        sentinel = await asyncio.start_server(_on_sentinel, host="127.0.0.1", port=0)
        sentinel_port = sentinel.sockets[0].getsockname()[1]

        async def _fake_resolver(host, port):
            return ["127.0.0.1"]

        # allowlist "sentinel.example.com" -> but request a DIFFERENT,
        # non-allowlisted host, so classify_host denies before any resolve
        # or connect ever happens.
        policy = EgressPolicy(
            allowed_hosts=frozenset({"sentinel.example.com"}),
            allowed_ports=frozenset({443, sentinel_port}),
        )
        server = await serve(socket_path, policy, log_path, resolver=_fake_resolver)
        try:
            status_line, writer = await _send_connect(
                socket_path, f"not-allowlisted.example.com:{sentinel_port}"
            )
            assert status_line.startswith(b"HTTP/1.1 403")
            writer.close()

            # Also exercise the ip-literal deny path against the same sentinel.
            status_line2, writer2 = await _send_connect(socket_path, f"127.0.0.1:{sentinel_port}")
            assert status_line2.startswith(b"HTTP/1.1 403")
            writer2.close()

            await asyncio.sleep(0.2)
        finally:
            sentinel.close()
            await sentinel.wait_closed()
            server.close()
            await server.wait_closed()

        # No deny path ever opened an upstream connection to the sentinel.
        assert not connected.is_set()

    asyncio.run(_run())


def test_proxy_never_pipes_on_resolved_private_deny(tmp_path):
    # A DENY at the resolved-private stage (rule 5.5) must also never reach
    # the target -- the sentinel here stands in for "the private target
    # itself", proving classify_resolved's deny happens before connect().
    async def _run():
        socket_path = tmp_path / "egress12b.sock"
        log_path = tmp_path / "egress12b.jsonl"

        # The sentinel IS the resolver's target: the fake resolver returns the
        # sentinel's own loopback address, which classify_resolved must deny
        # (loopback is private). classify_host passes (host allowlisted, port
        # allowed), so the ONLY gate between the request and the sentinel is
        # the resolved-private check -- a connect-before-classify regression
        # would light up the sentinel. (Previously the sentinel was on
        # 127.0.0.1 but the resolver returned 10.0.10.5, so it could never
        # observe a wrongful connect.)
        connected = asyncio.Event()

        async def _on_sentinel(reader, writer):
            connected.set()
            writer.close()

        sentinel = await asyncio.start_server(_on_sentinel, host="127.0.0.1", port=0)
        sentinel_port = sentinel.sockets[0].getsockname()[1]

        async def _fake_resolver(host, port):
            return ["127.0.0.1"]  # loopback -> classify_resolved denies (private)

        policy = EgressPolicy(
            allowed_hosts=frozenset({"rebind.example.com"}),
            allowed_ports=frozenset({443, sentinel_port}),
        )
        server = await serve(socket_path, policy, log_path, resolver=_fake_resolver)
        try:
            status_line, writer = await _send_connect(
                socket_path, f"rebind.example.com:{sentinel_port}"
            )
            # classify_host passes; the resolved loopback IP is private -> deny.
            assert status_line.startswith(b"HTTP/1.1 403")
            writer.close()
            await asyncio.sleep(0.2)
        finally:
            sentinel.close()
            await sentinel.wait_closed()
            server.close()
            await server.wait_closed()

        entries = _read_jsonl(log_path)
        deny_entries = [e for e in entries if e["decision"] == "deny"]
        assert len(deny_entries) == 1
        assert deny_entries[0]["reason"] == "resolved-private"
        # And the resolved-private deny NEVER connected to the target.
        assert not connected.is_set()

    asyncio.run(_run())


# --------------------------------------------------------------------------
# 13: structural isolation (podman --network=none)
# --------------------------------------------------------------------------


@requires_podman
def test_worker_network_none_has_zero_direct_egress():
    env = {**os.environ, "XDG_RUNTIME_DIR": f"/run/user/{os.getuid()}",
            "DBUS_SESSION_BUS_ADDRESS": f"unix:path=/run/user/{os.getuid()}/bus"}
    result = subprocess.run(
        [
            "podman", "run", "--rm", "--network=none",
            "docker.io/library/python:3.12-alpine",
            "python3", "-c",
            "import socket\n"
            "try:\n"
            "    socket.create_connection(('1.1.1.1', 443), timeout=3)\n"
            "    print('REACHED')\n"
            "except OSError as e:\n"
            "    print('BLOCKED', e)\n",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    out = result.stdout + result.stderr
    assert "REACHED" not in out
    assert "BLOCKED" in out
    assert "Network is unreachable" in out or "Network unreachable" in out


# --------------------------------------------------------------------------
# 14-15: build_run_argv integration
# --------------------------------------------------------------------------


def _profile(tmp_path, **overrides):
    work = tmp_path / "work"
    work.mkdir(parents=True, exist_ok=True)
    task = tmp_path / "task"
    task.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        image="localhost/stigmergy-worker@sha256:" + "a" * 64,
        work_clone=work,
        task_pack=task,
        scratch_size="64m",
        pids_limit=32,
        memory="256m",
        cpus="1",
        timeout_seconds=120,
    )
    kwargs.update(overrides)
    return ContainerProfile(**kwargs)


def test_build_run_argv_adds_egress_socket_mount_when_set(tmp_path):
    profile = _profile(tmp_path, network="none")
    sock = tmp_path / "egress.sock"
    argv = build_run_argv(profile, command=["true"], egress_socket=sock)
    assert f"--volume={sock}:/run/egress.sock:rw" in argv
    assert "--network=none" in argv

    baseline = build_run_argv(profile, command=["true"])
    extra = [a for a in argv if a not in baseline]
    assert extra == [f"--volume={sock}:/run/egress.sock:rw"]


def test_build_run_argv_unchanged_when_egress_socket_none(tmp_path):
    profile = _profile(tmp_path)
    with_none = build_run_argv(profile, command=["true"], egress_socket=None)
    without_param = build_run_argv(profile, command=["true"])
    assert with_none == without_param


# --------------------------------------------------------------------------
# stigmergy.egress: policy_for_lane (PURE) -- charter lane -> egress group ->
# allowlist, unknown/missing group ⇒ EMPTY allowlist (deny-all), never
# allow-all. Not in the frozen §3 list (that's proxy-only), but §2's
# explicit deliverable and load-bearing for the "never default-allow"
# invariant, so exercised directly here too.
# --------------------------------------------------------------------------


def _charter_resolved(**egress_overrides):
    egress = {
        "inference": {"hosts": ["api.anthropic.com", "macstudio.local"]},
        "registries": {"hosts": ["pypi.org", "files.pythonhosted.org"]},
    }
    egress.update(egress_overrides)
    return {
        "lanes": {
            "cheap": {"egress": ["inference", "registries"]},
            "no_egress_list": {},
            "bad_group": {"egress": ["nonexistent-group"]},
        },
        "egress": egress,
    }


def test_policy_for_lane_resolves_charter_groups_to_allowlist():
    charter = _charter_resolved()
    policy = policy_for_lane(charter, "cheap")
    assert policy.allowed_hosts == frozenset(
        {"api.anthropic.com", "macstudio.local", "pypi.org", "files.pythonhosted.org"}
    )


def test_policy_for_lane_unknown_lane_is_deny_all():
    charter = _charter_resolved()
    policy = policy_for_lane(charter, "no-such-lane")
    assert policy.allowed_hosts == frozenset()


def test_policy_for_lane_lane_missing_egress_key_is_deny_all():
    charter = _charter_resolved()
    policy = policy_for_lane(charter, "no_egress_list")
    assert policy.allowed_hosts == frozenset()


def test_policy_for_lane_unknown_group_contributes_nothing():
    charter = _charter_resolved()
    policy = policy_for_lane(charter, "bad_group")
    assert policy.allowed_hosts == frozenset()


def test_policy_for_lane_local_inference_sets_exception_ip_and_port():
    charter = _charter_resolved()
    policy = policy_for_lane(
        charter, "cheap", local_inference={"ip": "10.0.20.104", "port": 8000}
    )
    assert policy.local_inference_ip == "10.0.20.104"
    assert 8000 in policy.allowed_ports
    assert 443 in policy.allowed_ports
    # local_inference alone grants no host access -- macstudio.local must
    # still come from an actual egress group (it does, in this fixture).
    assert "macstudio.local" in policy.allowed_hosts


def test_policy_for_lane_missing_charter_shape_is_deny_all():
    # A charter with no [egress] table at all, or a lane with a completely
    # missing lanes table -- every malformed shape still yields deny-all.
    policy = policy_for_lane({}, "cheap")
    assert policy.allowed_hosts == frozenset()


def test_policy_for_lane_ip_colon_port_hosts_entry_is_inert_not_a_bypass():
    # The pre-.11 charter fixture (tests/fixtures/charter_valid.toml) lists
    # "10.0.20.104:8000" as an egress.inference host -- an ip:port-shaped
    # string. policy_for_lane absorbs it verbatim (no special-casing), but
    # it is dead config, never a bypass: a real CONNECT authority parses to
    # bare host "10.0.20.104" (no port suffix), which never string-equals
    # the stored "10.0.20.104:8000" entry -- so this entry never actually
    # allowlists anything. And even in a contrived scenario where a raw
    # "10.0.20.104:8000" text made it into classify_host's host argument,
    # rule 2 (ip-literal ban) fires before rule 3 (allowlist) and denies it
    # outright, since it still parses as *not* a valid ip_address itself...
    # what actually matters operationally is asserted below: the bare IP
    # literal is denied regardless of this entry's presence in the allowlist.
    charter = _charter_resolved(
        inference={"hosts": ["10.0.20.104:8000", "macstudio.local"]}
    )
    policy = policy_for_lane(charter, "cheap")
    assert "10.0.20.104:8000" in policy.allowed_hosts  # absorbed verbatim, as documented

    # The actual security-relevant question: does the bare IP literal ever
    # get through? No -- rule 2 denies it outright, allowlist membership of
    # the ip:port string notwithstanding.
    denied, reason = classify_host(policy, "10.0.20.104", 8000)
    assert denied is False
    assert reason == "ip-literal"


# --------------------------------------------------------------------------
# setup_dispatch_egress / EgressHandle / teardown -- orchestration lifecycle
# --------------------------------------------------------------------------


@requires_network
def test_setup_dispatch_egress_starts_serving_and_teardown_stops_it(tmp_path):
    async def _fake_resolver(host, port):
        return ["104.20.23.154"]

    policy = EgressPolicy(allowed_hosts=frozenset({"api.anthropic.com"}))
    handle = setup_dispatch_egress(
        "dispatch-abc123", policy, tmp_path, resolver=_fake_resolver
    )
    try:
        assert isinstance(handle, EgressHandle)
        assert handle.socket_path.exists()

        async def _probe():
            status_line, writer = await _send_connect(handle.socket_path, "api.anthropic.com:443")
            writer.close()
            return status_line

        status_line = asyncio.run(_probe())
        assert status_line.startswith(b"HTTP/1.1 200")
    finally:
        handle.teardown()

    assert not handle.socket_path.exists()

    async def _probe_after_stop():
        with pytest.raises((ConnectionRefusedError, FileNotFoundError, OSError)):
            await asyncio.open_unix_connection(path=str(handle.socket_path))

    asyncio.run(_probe_after_stop())


def test_setup_dispatch_egress_uses_bounded_socket_filename(tmp_path):
    # Regression (first code dogfood): the egress socket filename must be a
    # FIXED-LENGTH hash of the dispatch_id, not the id itself. Embedding a
    # realistic id (rig+ticket+ms) under a rig-local runtime dir overflowed
    # the AF_UNIX ~108-byte limit and crashed the daemon with
    # `OSError: AF_UNIX path too long` before any worker ran. Proves setup
    # goes through container.dispatch_socket_path. No network needed — the
    # proxy only binds the unix socket here.
    long_id = "egress-concurrency-92-part1-1784476863032"
    policy = EgressPolicy(allowed_hosts=frozenset({"api.anthropic.com"}))
    handle = setup_dispatch_egress(long_id, policy, tmp_path)
    try:
        assert isinstance(handle, EgressHandle)
        assert handle.socket_path.exists()
        # The long id is NOT embedded in the socket filename (it is hashed).
        assert long_id not in handle.socket_path.name
        assert handle.socket_path.name.startswith("egress-")
        assert handle.socket_path.name.endswith(".sock")
        # The sibling .jsonl log KEEPS the readable id (no length limit) so an
        # operator can still correlate socket <-> dispatch.
        assert handle.log_path.name == f"egress-{long_id}.jsonl"
    finally:
        handle.teardown()
    assert not handle.socket_path.exists()


# --------------------------------------------------------------------------
# bead .32 -- LIVE end-to-end egress proof, Option A (SB-approved 2026-07-16):
# a REAL hardened --network=none worker container whose ONLY exit is the
# bind-mounted egress unix socket, driven through the REAL per-dispatch proxy
# (setup_dispatch_egress + build_run_argv, exactly as daemon.py stands one up).
# Proves Option-A items (a) allowlisted egress works, (b) non-allowlisted +
# IP-literal denied and logged, that the proxy socket is the container's ONLY
# exit (direct AF_INET egress is impossible under --network=none), and
# (d)/(c-fail-closed) once the proxy is torn down mid-dispatch, new egress
# attempts from inside the cage fail closed.
# The remaining .32 half -- item (c)'s "dispatch classifies INFRA" via a real
# claude-code run -- is gated on .63 (in-container shim) + .26 (SB sign-off)
# and tracked in bead .64. These tests substitute a stdlib CONNECT probe for
# claude-code because the subject under test is the egress cage, not the coder;
# the probe speaks the exact same raw CONNECT the worker's HTTPS client would.
# --------------------------------------------------------------------------

_WORKER_BASE_IMAGE = "docker.io/library/python:3.12-alpine"

# Raw-CONNECT probe: AF_UNIX -> the mounted egress socket, one CONNECT per
# target, prints "<authority> => <status-line-or-error>". stdlib only, no
# writes to the read-only rootfs, no container network namespace needed
# (the unix socket is a filesystem object, reachable under --network=none).
_EGRESS_PROBE = r'''
import socket


def _probe(auth):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(15)
    try:
        s.connect("/run/egress.sock")
    except OSError as exc:
        return "CONNECT_ERR:" + type(exc).__name__
    try:
        s.sendall(("CONNECT %s HTTP/1.1\r\nHost: %s\r\n\r\n" % (auth, auth)).encode())
        resp = b""
        while b"\r\n\r\n" not in resp and len(resp) < 4096:
            chunk = s.recv(256)
            if not chunk:
                break
            resp += chunk
    except OSError as exc:
        return "RECV_ERR:" + type(exc).__name__
    finally:
        s.close()
    if not resp:
        return "EMPTY"
    return resp.split(b"\r\n", 1)[0].decode("latin-1", "replace")


for _auth in ["example.com:443", "evil.example.com:443", "1.1.1.1:443"]:
    print(_auth + " => " + _probe(_auth), flush=True)

# The proxy socket must be the ONLY exit: a DIRECT AF_INET connection (not via
# the proxy) must be impossible under --network=none. REACHED here == a cage
# breach; the test fails on it unconditionally.
_direct = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
_direct.settimeout(5)
try:
    _direct.connect(("1.1.1.1", 443))
    print("DIRECT => REACHED", flush=True)
except OSError as _exc:
    print("DIRECT => BLOCKED:" + type(_exc).__name__, flush=True)
finally:
    _direct.close()
'''

# Fail-closed probe: probe once (proxy up) -> write /work/first-done, then
# block until the host touches /work/kill (host tears the proxy down first),
# then probe again -> must fail closed. Prints one JSON line {first, second}.
_FAILCLOSED_PROBE = r'''
import json
import os
import socket
import time


def _probe(auth):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(15)
    try:
        s.connect("/run/egress.sock")
    except OSError as exc:
        return "CONNECT_ERR:" + type(exc).__name__
    try:
        s.sendall(("CONNECT %s HTTP/1.1\r\nHost: %s\r\n\r\n" % (auth, auth)).encode())
        resp = b""
        while b"\r\n\r\n" not in resp and len(resp) < 4096:
            chunk = s.recv(256)
            if not chunk:
                break
            resp += chunk
    except OSError as exc:
        return "RECV_ERR:" + type(exc).__name__
    finally:
        s.close()
    if not resp:
        return "EMPTY"
    return resp.split(b"\r\n", 1)[0].decode("latin-1", "replace")


first = _probe("example.com:443")
# Publish first-done ATOMICALLY (write+rename) so the host never reads a
# half-written file.
with open("/work/first-done.tmp", "w") as fh:
    fh.write(first)
os.replace("/work/first-done.tmp", "/work/first-done")

# Block until the host has torn the proxy down and signalled via /work/kill.
# Do NOT fall through to a second CONNECT on timeout -- a fall-through request
# could race an in-progress teardown and reach upstream, a false "fail-closed".
# If the signal never comes, report it distinctly (the test treats it as a
# failure, not as proof).
deadline = time.monotonic() + 120
while not os.path.exists("/work/kill") and time.monotonic() < deadline:
    time.sleep(0.05)

if os.path.exists("/work/kill"):
    second = _probe("example.com:443")
else:
    second = "BARRIER_TIMEOUT"
print(json.dumps({"first": first, "second": second}), flush=True)
'''


def _worker_podman_env() -> dict:
    uid = os.getuid()
    return {
        **os.environ,
        "XDG_RUNTIME_DIR": f"/run/user/{uid}",
        "DBUS_SESSION_BUS_ADDRESS": f"unix:path=/run/user/{uid}/bus",
    }


def _pinned_worker_image() -> str:
    """Pull the worker base image and resolve its digest-pinned ref
    (build_run_argv rejects an unpinned image). Idempotent/cheap after the
    first pull. FAILS LOUDLY (never skips) when podman is present but image
    prep fails -- a certification test must not silently degrade to a skip."""
    env = _worker_podman_env()
    pull = subprocess.run(
        ["podman", "pull", _WORKER_BASE_IMAGE],
        env=env, capture_output=True, text=True, timeout=180,
    )
    out = subprocess.run(
        ["podman", "inspect", "--format", "{{index .RepoDigests 0}}", _WORKER_BASE_IMAGE],
        env=env, capture_output=True, text=True, timeout=30,
    )
    digest = out.stdout.strip()
    if "@sha256:" not in digest:
        pytest.fail(
            f"could not prepare a digest-pinned worker image ({_WORKER_BASE_IMAGE}); "
            f"pull rc={pull.returncode} stderr={pull.stderr!r}; "
            f"inspect stderr={out.stderr!r}"
        )
    return digest


def _reap_dispatch(dispatch_id: str) -> None:
    """Best-effort removal of any container lingering for this dispatch label
    (normal exits use --rm; this covers exceptional/timeout paths)."""
    env = _worker_podman_env()
    try:
        listed = subprocess.run(
            ["podman", "ps", "-aq", "--filter", f"label=stigmergy.dispatch_id={dispatch_id}"],
            env=env, capture_output=True, text=True, timeout=30,
        )
        for cid in listed.stdout.split():
            subprocess.run(
                ["podman", "rm", "-f", cid],
                env=env, capture_output=True, text=True, timeout=30,
            )
    except OSError:
        pass


@requires_podman
@requires_network
def test_live_e2e_worker_container_egress_governed_by_proxy(tmp_path):
    image = _pinned_worker_image()
    runtime = tmp_path / "rt"
    runtime.mkdir()
    policy = EgressPolicy(
        allowed_hosts=frozenset({_REAL_HOST_FOR_LIVE_TEST}),
        allowed_ports=frozenset({443}),
    )
    handle = setup_dispatch_egress("disp-live-egress", policy, runtime)
    try:
        profile = _profile(tmp_path, image=image, network="none")
        argv = build_run_argv(
            profile,
            command=["python3", "-c", _EGRESS_PROBE],
            egress_socket=handle.socket_path,
            dispatch_id="disp-live-egress",
        )
        result = subprocess.run(
            argv, env=_worker_podman_env(), capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, (
            f"probe container failed rc={result.returncode}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        lines = {
            ln.split(" => ", 1)[0]: ln.split(" => ", 1)[1]
            for ln in result.stdout.splitlines()
            if " => " in ln
        }
        # (a) allowlisted host tunnels: the proxy writes 200 ONLY after a real
        # upstream TCP connect to the resolved public IP succeeds.
        assert lines.get(f"{_REAL_HOST_FOR_LIVE_TEST}:443", "").startswith(
            "HTTP/1.1 200"
        ), result.stdout
        # (b) non-allowlisted host + bare IP literal are both denied (403).
        assert lines.get("evil.example.com:443", "").startswith("HTTP/1.1 403"), result.stdout
        assert lines.get("1.1.1.1:443", "").startswith("HTTP/1.1 403"), result.stdout
        # The proxy socket is the ONLY exit: a direct AF_INET connection from
        # inside the cage must be blocked (--network=none). REACHED would mean
        # the container has an egress path the proxy does not govern.
        assert lines.get("DIRECT", "").startswith("BLOCKED"), result.stdout
    finally:
        handle.teardown()
        _reap_dispatch("disp-live-egress")

    # Every decision is recorded in the proxy's JSONL egress log (survives
    # teardown -- only the socket is unlinked). Pin the COMPLETE per-dispatch
    # audit records so a mis-correlated / mis-metadata'd log can't stay green.
    entries = _read_jsonl(handle.log_path)
    assert len(entries) == 3, entries
    for e in entries:
        assert e["dispatch_id"] == "disp-live-egress", e
        assert e["port"] == 443, e
    allow = [e for e in entries if e["decision"] == "allow"]
    deny = [e for e in entries if e["decision"] == "deny"]
    assert len(allow) == 1, entries
    assert allow[0]["reason"] == "allow"
    assert allow[0]["host"] == _REAL_HOST_FOR_LIVE_TEST
    assert allow[0]["resolved_ip"]  # a real public IP was resolved AND connected
    assert len(deny) == 2, entries
    by_reason = {d["reason"]: d for d in deny}
    assert set(by_reason) == {"not-allowlisted", "ip-literal"}, entries
    # Pre-resolution denies never resolve a target -> resolved_ip is null.
    assert by_reason["not-allowlisted"]["host"] == "evil.example.com"
    assert by_reason["not-allowlisted"]["resolved_ip"] is None
    assert by_reason["ip-literal"]["host"] == "1.1.1.1"
    assert by_reason["ip-literal"]["resolved_ip"] is None


@requires_podman
@requires_network
def test_live_e2e_egress_fail_closed_when_proxy_killed_mid_dispatch(tmp_path):
    image = _pinned_worker_image()
    runtime = tmp_path / "rt"
    runtime.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    task = tmp_path / "task"
    task.mkdir()
    policy = EgressPolicy(
        allowed_hosts=frozenset({_REAL_HOST_FOR_LIVE_TEST}),
        allowed_ports=frozenset({443}),
    )
    handle = setup_dispatch_egress("disp-failclosed", policy, runtime)
    profile = ContainerProfile(
        image=image,
        work_clone=work,
        task_pack=task,
        scratch_size="64m",
        pids_limit=64,
        memory="256m",
        cpus="1",
        timeout_seconds=90,
        network="none",
    )
    argv = build_run_argv(
        profile,
        command=["python3", "-c", _FAILCLOSED_PROBE],
        egress_socket=handle.socket_path,
        dispatch_id="disp-failclosed",
    )
    proc = None
    try:
        proc = subprocess.Popen(
            argv, env=_worker_podman_env(),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        # Wait until the in-flight dispatch has made its first (successful)
        # probe and ATOMICALLY published the result.
        first_done = work / "first-done"
        deadline = time.monotonic() + 60
        while not first_done.exists() and time.monotonic() < deadline:
            if proc.poll() is not None:
                break
            time.sleep(0.1)
        assert first_done.exists(), f"container never completed its first probe; rc={proc.poll()!r}"
        assert first_done.read_text().startswith("HTTP/1.1 200"), first_done.read_text()

        # Kill the proxy MID-DISPATCH. teardown() stops the loop, closes the
        # server, JOINS the thread, and unlinks the socket before returning.
        handle.teardown()
        assert not handle.socket_path.exists()  # completed-stop postcondition
        # Only NOW release the container's second probe -- strict happens-after:
        # the proxy is fully down before the second CONNECT is ever issued, so
        # the two never race.
        (work / "kill").write_text("1")

        out, err = proc.communicate(timeout=90)
        assert proc.returncode == 0, f"rc={proc.returncode}\nSTDOUT:\n{out}\nSTDERR:\n{err}"
        payload = next(
            (json.loads(ln) for ln in out.splitlines()
             if ln.strip().startswith("{") and "second" in ln),
            None,
        )
        assert payload is not None, f"no JSON result line in probe output:\n{out}"
        assert payload["first"].startswith("HTTP/1.1 200"), payload
        # Fail-closed: the proxy is fully torn down, so the container's sole
        # exit refuses the connection at the unix layer (CONNECT_ERR). A
        # RECV_ERR would mean the connect SUCCEEDED (proxy still up) and the
        # request may have reached upstream -- NOT acceptable as fail-closed.
        assert payload["second"].startswith("CONNECT_ERR"), payload

        # Independent proof no SECOND egress happened: the per-dispatch egress
        # log holds exactly the first allow and no second decision.
        entries = _read_jsonl(handle.log_path)
        assert len(entries) == 1, entries
        assert entries[0]["decision"] == "allow"
        assert entries[0]["host"] == _REAL_HOST_FOR_LIVE_TEST
        assert entries[0]["dispatch_id"] == "disp-failclosed"
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
        handle.teardown()
        _reap_dispatch("disp-failclosed")
