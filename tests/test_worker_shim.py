"""Bead .63 — in-container egress/relay shim + entrypoint + worker image.

Frozen test contract (memory/projects/stigmergy/bead63-build-spec.md §4).

  A. serve_bridge unit (host asyncio, temp unix sockets) — the byte-bridge core.
  B. entrypoint.sh integration (REAL minimal worker image, mount control) —
     presence-detect, readiness-gate, env-export, fail-closed exec-nothing,
     relay-optional, dual-shim independence, NO_PROXY/loopback, HTTPS_PROXY
     inherited across exec.
  C. live E2E through the REAL egress proxy — a stock HTTPS_PROXY-honoring
     client reaches an allowlisted host, denied classes are denied, the cage
     stays intact.
  E. Containerfile / BASE_IMAGE structural.

Section D (build_run_argv relay_socket=) + the bare-sha256 pin change live in
test_container.py. B/C are @requires_podman; the image fixture also needs
network (apt-get python3 at build) and skips cleanly where either is absent.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import tempfile
import threading
from pathlib import Path

import pytest

from stigmergy.container import ContainerProfile, build_image, build_run_argv
from stigmergy.worker_image import shim

# --------------------------------------------------------------------------
# Section A — serve_bridge unit (host asyncio, no podman).
# --------------------------------------------------------------------------


async def _start_unix_echo(path: Path) -> asyncio.Server:
    async def _cb(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                data = await reader.read(65536)
                if not data:
                    break
                writer.write(data)
                await writer.drain()
        except (ConnectionError, OSError):
            pass
        finally:
            try:
                writer.close()
            except (ConnectionError, OSError):
                pass

    return await asyncio.start_unix_server(_cb, path=str(path))


def test_serve_bridge_echoes_through_unix_socket(tmp_path):
    async def _run():
        sock = tmp_path / "u.sock"
        echo = await _start_unix_echo(sock)
        bridge = await shim.serve_bridge("127.0.0.1", 0, str(sock))
        try:
            port = bridge.sockets[0].getsockname()[1]
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(b"hello-bridge-42")
            await writer.drain()
            got = await asyncio.wait_for(reader.readexactly(15), timeout=5)
            assert got == b"hello-bridge-42"
            writer.close()
        finally:
            bridge.close()
            await bridge.wait_closed()
            echo.close()
            await echo.wait_closed()

    asyncio.run(_run())


def test_shim_listen_host_is_loopback_constant():
    # The shim's own entry point binds loopback ONLY, never 0.0.0.0 — pinned
    # as defense-in-depth against a future routable network profile.
    assert shim.LISTEN_HOST == "127.0.0.1"


def test_serve_bridge_binds_the_loopback_host_it_is_given(tmp_path):
    async def _run():
        sock = tmp_path / "u.sock"
        echo = await _start_unix_echo(sock)
        bridge = await shim.serve_bridge(shim.LISTEN_HOST, 0, str(sock))
        try:
            assert bridge.sockets[0].getsockname()[0] == "127.0.0.1"
        finally:
            bridge.close()
            await bridge.wait_closed()
            echo.close()
            await echo.wait_closed()

    asyncio.run(_run())


def test_serve_bridge_dial_failure_closes_inbound_fail_closed(tmp_path):
    async def _run():
        absent = tmp_path / "absent.sock"  # nothing is listening here
        bridge = await shim.serve_bridge("127.0.0.1", 0, str(absent))
        try:
            port = bridge.sockets[0].getsockname()[1]
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            # Fail-closed: the dial fails, the inbound conn is closed (EOF),
            # no hang, and crucially NO fallback egress path.
            got = await asyncio.wait_for(reader.read(), timeout=5)
            assert got == b""
            writer.close()
        finally:
            bridge.close()
            await bridge.wait_closed()

    asyncio.run(_run())


def test_serve_bridge_concurrent_connections_independent(tmp_path):
    async def _run():
        sock = tmp_path / "u.sock"
        echo = await _start_unix_echo(sock)
        bridge = await shim.serve_bridge("127.0.0.1", 0, str(sock))
        try:
            port = bridge.sockets[0].getsockname()[1]
            r1, w1 = await asyncio.open_connection("127.0.0.1", port)
            r2, w2 = await asyncio.open_connection("127.0.0.1", port)
            w1.write(b"AAAA")
            w2.write(b"BBBB")
            await w1.drain()
            await w2.drain()
            g1 = await asyncio.wait_for(r1.readexactly(4), timeout=5)
            g2 = await asyncio.wait_for(r2.readexactly(4), timeout=5)
            assert g1 == b"AAAA"
            assert g2 == b"BBBB"
            w1.close()
            w2.close()
        finally:
            bridge.close()
            await bridge.wait_closed()
            echo.close()
            await echo.wait_closed()

    asyncio.run(_run())


def test_serve_bridge_large_payload_and_reaccepts(tmp_path):
    async def _run():
        sock = tmp_path / "u.sock"
        echo = await _start_unix_echo(sock)
        bridge = await shim.serve_bridge("127.0.0.1", 0, str(sock))
        try:
            port = bridge.sockets[0].getsockname()[1]
            payload = b"x" * (1024 * 1024)
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(payload)
            await writer.drain()
            writer.write_eof()
            got = await asyncio.wait_for(reader.readexactly(len(payload)), timeout=15)
            assert got == payload
            writer.close()
            # The serve loop survives a completed connection and re-accepts.
            r2, w2 = await asyncio.open_connection("127.0.0.1", port)
            w2.write(b"again")
            await w2.drain()
            got2 = await asyncio.wait_for(r2.readexactly(5), timeout=5)
            assert got2 == b"again"
            w2.close()
        finally:
            bridge.close()
            await bridge.wait_closed()
            echo.close()
            await echo.wait_closed()

    asyncio.run(_run())


def test_serve_bridge_never_logs_payload(tmp_path, caplog, capsys):
    secret = "SUPER-SECRET-CAPABILITY-TOKEN-do-not-log-me"

    async def _run():
        sock = tmp_path / "u.sock"
        echo = await _start_unix_echo(sock)
        bridge = await shim.serve_bridge("127.0.0.1", 0, str(sock))
        try:
            port = bridge.sockets[0].getsockname()[1]
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(secret.encode())
            await writer.drain()
            await asyncio.wait_for(reader.readexactly(len(secret)), timeout=5)
            writer.close()
        finally:
            bridge.close()
            await bridge.wait_closed()
            echo.close()
            await echo.wait_closed()

    with caplog.at_level(logging.DEBUG):
        asyncio.run(_run())
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err


# --------------------------------------------------------------------------
# Sections B & C — live container integration.
# --------------------------------------------------------------------------

PODMAN = shutil.which("podman")
requires_podman = pytest.mark.skipif(PODMAN is None, reason="podman not installed")


def _network_available(host: str = "example.com", port: int = 443) -> bool:
    try:
        with socket.create_connection((host, port), timeout=3):
            return True
    except OSError:
        return False


requires_network = pytest.mark.skipif(
    not _network_available(), reason="no outbound network in this sandbox"
)

_SRC = Path(__file__).resolve().parents[1] / "src" / "stigmergy" / "worker_image"


def _worker_podman_env() -> dict:
    uid = os.getuid()
    return {
        **os.environ,
        "XDG_RUNTIME_DIR": f"/run/user/{uid}",
        "DBUS_SESSION_BUS_ADDRESS": f"unix:path=/run/user/{uid}/bus",
    }


@pytest.fixture(scope="module")
def shim_image():
    """Build ONE minimal worker image from the REAL shim.py + entrypoint.sh
    (no claude-code layer — the probes are stock python clients). B and C
    exercise the real artifacts at the real hardcoded /run paths. FAILS LOUD
    if podman is present but the build breaks (a certification fixture must
    not silently degrade to a skip); skips only when podman/network absent."""
    if PODMAN is None:
        pytest.skip("podman not installed")
    if not _network_available():
        pytest.skip("image build needs network for apt-get python3")
    base_ref = (_SRC / "BASE_IMAGE.txt").read_text().strip()
    ctx = Path(tempfile.mkdtemp(prefix="stigmergy-shim-img-"))
    try:
        shutil.copy(_SRC / "shim.py", ctx / "shim.py")
        shutil.copy(_SRC / "entrypoint.sh", ctx / "entrypoint.sh")
        (ctx / "Containerfile").write_text(
            f"FROM {base_ref}\n"
            "RUN apt-get update "
            "&& apt-get install -y --no-install-recommends python3 ca-certificates "
            "&& rm -rf /var/lib/apt/lists/*\n"
            "COPY shim.py /opt/stigmergy/shim.py\n"
            "COPY entrypoint.sh /opt/stigmergy/entrypoint.sh\n"
            "RUN chmod +x /opt/stigmergy/entrypoint.sh\n"
            'ENTRYPOINT ["/opt/stigmergy/entrypoint.sh"]\n'
        )
        tag = "localhost/stigmergy-shim-test:latest"
        build_image(ctx, tag)  # validates pinned FROM + builds (check=True)
        # Reference the locally-built image by its immutable config id. This
        # podman reports `.Id` as a bare 64-hex (no `sha256:` prefix); normalize
        # it to `sha256:<hex>` so it (a) is runnable by `podman run` and (b)
        # passes build_run_argv's bare-digest pin guard (bead .63).
        out = subprocess.run(
            ["podman", "inspect", "--format", "{{.Id}}", tag],
            env=_worker_podman_env(), capture_output=True, text=True, timeout=30,
        )
        raw = out.stdout.strip()
        if not raw:
            pytest.fail(f"could not resolve built shim image id: {out.stderr!r}")
        return raw if raw.startswith("sha256:") else f"sha256:{raw}"
    finally:
        shutil.rmtree(ctx, ignore_errors=True)


def _shim_profile(image: str, work: Path, task: Path) -> ContainerProfile:
    return ContainerProfile(
        image=image,
        work_clone=work,
        task_pack=task,
        scratch_size="64m",
        pids_limit=128,
        memory="256m",
        cpus="1",
        timeout_seconds=90,
        network="none",
    )


def _reap(dispatch_id: str) -> None:
    env = _worker_podman_env()
    try:
        listed = subprocess.run(
            ["podman", "ps", "-aq", "--filter", f"label=stigmergy.dispatch_id={dispatch_id}"],
            env=env, capture_output=True, text=True, timeout=30,
        )
        for cid in listed.stdout.split():
            subprocess.run(
                ["podman", "rm", "-f", cid], env=env, capture_output=True, text=True, timeout=30
            )
    except OSError:
        pass


def _start_tag_server(path: Path, tag: bytes):
    """Host-side unix server that sends ``tag`` on every accept (then drains).
    Bind-mounted into the container at a /run/*.sock path so the container's
    shim, dialing that path, reaches THIS server — the tag proves which port
    reached which socket. Returns a cleanup callable."""
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(path))
    srv.listen(16)
    srv.settimeout(0.5)
    stop = threading.Event()

    def _loop():
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            try:
                conn.sendall(tag)
                conn.settimeout(1.0)
                try:
                    conn.recv(4096)
                except OSError:
                    pass
            finally:
                try:
                    conn.close()
                except OSError:
                    pass

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()

    def _cleanup():
        stop.set()
        try:
            srv.close()
        except OSError:
            pass
        thread.join(timeout=5)
        try:
            os.unlink(path)
        except OSError:
            pass

    return _cleanup


def _read_jsonl(path) -> list:
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]


# The Section-B stub worker ("$@"): record the proxy env the entrypoint set,
# then probe each proxy port and record which tagged socket it reached.
_B_STUB = r'''
import json, os, socket

def probe(port):
    s = socket.socket(); s.settimeout(5)
    try:
        s.connect(("127.0.0.1", int(port)))
        s.sendall(b"PROBE\r\n")
        data = s.recv(256)
        s.close()
        return data.decode("latin-1", "replace")
    except OSError as exc:
        return "ERR:" + type(exc).__name__

out = {
    "HTTPS_PROXY": os.environ.get("HTTPS_PROXY", ""),
    "ANTHROPIC_BASE_URL": os.environ.get("ANTHROPIC_BASE_URL", ""),
    "NO_PROXY": os.environ.get("NO_PROXY", ""),
}
if out["HTTPS_PROXY"]:
    out["egress_resp"] = probe(out["HTTPS_PROXY"].rsplit(":", 1)[1])
if out["ANTHROPIC_BASE_URL"]:
    out["relay_resp"] = probe(out["ANTHROPIC_BASE_URL"].rsplit(":", 1)[1])
with open("/work/result.json", "w") as fh:
    json.dump(out, fh)
open("/work/ran", "w").close()
'''


@requires_podman
def test_entrypoint_exports_env_gates_readiness_and_execs(shim_image, tmp_path):
    # Egress present (relay absent): the entrypoint starts the egress shim,
    # BLOCKS until it accepts, exports HTTPS_PROXY (+ NO_PROXY loopback), then
    # execs "$@". The stub's egress_resp == the tag proves the shim was
    # accepting AND bridged to the egress socket at exec time (readiness gate).
    # ANTHROPIC_BASE_URL stays unset (relay optional) — this is also case 11.
    work = tmp_path / "work"
    work.mkdir()
    task = tmp_path / "task"
    task.mkdir()
    egress_sock = tmp_path / "egress.sock"
    cleanup = _start_tag_server(egress_sock, b"EGRESS-OK\r\n")
    try:
        profile = _shim_profile(shim_image, work, task)
        argv = build_run_argv(
            profile,
            command=["python3", "-c", _B_STUB],
            egress_socket=egress_sock,
            dispatch_id="disp-b7",
        )
        result = subprocess.run(
            argv, env=_worker_podman_env(), capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        assert (work / "ran").exists()
        data = json.loads((work / "result.json").read_text())
        assert data["HTTPS_PROXY"] == "http://127.0.0.1:18080", data
        assert data["ANTHROPIC_BASE_URL"] == "", data  # relay optional, absent
        assert "127.0.0.1" in data["NO_PROXY"], data
        assert data["egress_resp"].startswith("EGRESS-OK"), data
    finally:
        cleanup()
        _reap("disp-b7")


@requires_podman
def test_entrypoint_fail_closed_when_egress_socket_absent(shim_image, tmp_path):
    # No egress socket mounted -> /run/egress.sock absent -> the entrypoint
    # must exit non-zero and exec NOTHING (the worker never runs without
    # governed egress).
    work = tmp_path / "work"
    work.mkdir()
    task = tmp_path / "task"
    task.mkdir()
    profile = _shim_profile(shim_image, work, task)
    argv = build_run_argv(
        profile, command=["python3", "-c", _B_STUB], dispatch_id="disp-b9"
    )
    result = subprocess.run(
        argv, env=_worker_podman_env(), capture_output=True, text=True, timeout=120
    )
    try:
        # Sentinel exit 69 (EXIT_CAGE_UNAVAILABLE) — the driver maps it to INFRA
        # (bead .64), distinct from a generic non-zero exit.
        assert result.returncode == 69, f"{result.returncode}: {result.stdout}\n{result.stderr}"
        assert not (work / "ran").exists(), "worker executed despite absent egress socket"
        assert "egress socket" in result.stderr.lower()
    finally:
        _reap("disp-b9")


@requires_podman
def test_entrypoint_dual_shim_independence(shim_image, tmp_path):
    # Both sockets present: HTTPS_PROXY(:18080) must reach ONLY the egress
    # socket and ANTHROPIC_BASE_URL(:18081) ONLY the relay socket. A cross-wire
    # would swap the tags. Also proves NO_PROXY exempts loopback (both probes
    # go to 127.0.0.1 ports and must succeed).
    work = tmp_path / "work"
    work.mkdir()
    task = tmp_path / "task"
    task.mkdir()
    egress_sock = tmp_path / "egress.sock"
    relay_sock = tmp_path / "relay.sock"
    c1 = _start_tag_server(egress_sock, b"EGRESS-OK\r\n")
    c2 = _start_tag_server(relay_sock, b"RELAY-OK\r\n")
    try:
        profile = _shim_profile(shim_image, work, task)
        argv = build_run_argv(
            profile,
            command=["python3", "-c", _B_STUB],
            egress_socket=egress_sock,
            relay_socket=relay_sock,
            dispatch_id="disp-b12",
        )
        result = subprocess.run(
            argv, env=_worker_podman_env(), capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        data = json.loads((work / "result.json").read_text())
        assert data["HTTPS_PROXY"] == "http://127.0.0.1:18080", data
        assert data["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:18081", data
        assert data["egress_resp"].startswith("EGRESS-OK"), data
        assert data["relay_resp"].startswith("RELAY-OK"), data
    finally:
        c1()
        c2()
        _reap("disp-b12")


# The Section-C stub worker: a STOCK HTTPS_PROXY-honoring client (urllib reads
# the proxy from the environment) + a direct-connect cage probe.
_C_STUB = r'''
import json, socket, urllib.error, urllib.request

def get(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "stigmergy-shim-test"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            return "OK:%d" % resp.status
    except urllib.error.HTTPError as exc:
        return "HTTP:%d" % exc.code
    except Exception as exc:
        return "ERR:" + type(exc).__name__

out = {
    "allowed": get("https://example.com/"),
    "denied_host": get("https://evil.example.com/"),
    "denied_ip": get("https://1.1.1.1/"),
}
d = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
d.settimeout(5)
try:
    d.connect(("1.1.1.1", 443))
    out["direct"] = "REACHED"
except OSError as exc:
    out["direct"] = "BLOCKED:" + type(exc).__name__
finally:
    d.close()
with open("/work/cresult.json", "w") as fh:
    json.dump(out, fh)
open("/work/ran", "w").close()
'''


@requires_podman
@requires_network
def test_live_e2e_stock_client_through_shim_governed_by_real_proxy(shim_image, tmp_path):
    from stigmergy.egress import setup_dispatch_egress
    from stigmergy.egress_proxy import EgressPolicy

    work = tmp_path / "work"
    work.mkdir()
    task = tmp_path / "task"
    task.mkdir()
    runtime = tmp_path / "rt"
    runtime.mkdir()
    policy = EgressPolicy(
        allowed_hosts=frozenset({"example.com"}), allowed_ports=frozenset({443})
    )
    handle = setup_dispatch_egress("disp-c13", policy, runtime)
    try:
        profile = _shim_profile(shim_image, work, task)
        argv = build_run_argv(
            profile,
            command=["python3", "-c", _C_STUB],
            egress_socket=handle.socket_path,
            dispatch_id="disp-c13",
        )
        result = subprocess.run(
            argv, env=_worker_podman_env(), capture_output=True, text=True, timeout=150
        )
        assert result.returncode == 0, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        data = json.loads((work / "cresult.json").read_text())
        # (13) a stock HTTPS_PROXY-honoring client reaches the allowlisted host.
        assert data["allowed"].startswith(("OK", "HTTP")), data
        # (14) denied classes never succeed.
        assert not data["denied_host"].startswith("OK"), data
        assert not data["denied_ip"].startswith("OK"), data
        # (15) cage intact: a direct AF_INET connect is blocked.
        assert data["direct"].startswith("BLOCKED"), data
    finally:
        handle.teardown()
        _reap("disp-c13")

    # (16) complete JSONL audit: an allow for the allowlisted host + the two
    # denies with their exact reasons.
    entries = _read_jsonl(handle.log_path)
    allow = [e for e in entries if e["decision"] == "allow"]
    deny_reasons = {e["reason"] for e in entries if e["decision"] == "deny"}
    assert any(e["host"] == "example.com" for e in allow), entries
    assert "not-allowlisted" in deny_reasons, entries
    assert "ip-literal" in deny_reasons, entries


# --------------------------------------------------------------------------
# Section E — Containerfile / BASE_IMAGE structural (no podman).
# --------------------------------------------------------------------------


def test_base_image_txt_is_digest_pinned():
    ref = (_SRC / "BASE_IMAGE.txt").read_text().strip()
    assert re.search(r"@sha256:[0-9a-fA-F]{64}$", ref), ref


def test_worker_containerfile_from_is_digest_pinned():
    from stigmergy.container import _check_pinned_bases

    text = (_SRC / "Containerfile").read_text()
    _check_pinned_bases(text)  # raises ContainerError if any FROM is unpinned


def test_worker_containerfile_installs_shim_and_sets_entrypoint():
    text = (_SRC / "Containerfile").read_text()
    assert 'ENTRYPOINT ["/opt/stigmergy/entrypoint.sh"]' in text
    assert "COPY shim.py /opt/stigmergy/shim.py" in text
    assert "COPY entrypoint.sh /opt/stigmergy/entrypoint.sh" in text
    assert "python3" in text  # the shim runtime
    assert "@anthropic-ai/claude-code" in text  # the driver
