#!/bin/sh
# In-container worker entrypoint (Stigmergy bead .63) — the egress gatekeeper.
#
# Starts the egress (REQUIRED) and relay (OPTIONAL) shims, blocks until each is
# actually ACCEPTING on loopback, exports the proxy env an unmodified HTTP
# client honors, then execs the real worker command ("$@").
#
# Fail-closed: if the egress socket is absent, or its shim never comes up
# accepting within the timeout, exec NOTHING (no worker runs without governed
# egress). Everything below is HARDCODED — the component that makes the
# fail-closed decision takes NO runtime configuration (argv/env). The worker
# ("$@") is exec'd only AFTER setup, so it can never influence any of this.
set -eu

PYTHON=python3
SHIM=/opt/stigmergy/shim.py
EGRESS_SOCKET=/run/egress.sock
RELAY_SOCKET=/run/relay.sock
EGRESS_PORT=18080
RELAY_PORT=18081
READY_TIMEOUT=10 # seconds
# Sentinel exit code for EVERY fail-closed path: the cage's egress could not be
# set up, so no worker ran. The driver (claude_code.py `_CAGE_UNAVAILABLE_EXIT`)
# maps this to DispatchStatus.INFRA (a broken-infra condition, not a
# capability FAILED) — an explicit dead-cage -> INFRA signal (bead .64).
EXIT_CAGE_UNAVAILABLE=69

# Block until 127.0.0.1:$1 accepts a TCP connection; return non-zero on timeout.
wait_ready() {
    _port="$1"
    _n=0
    _max=$((READY_TIMEOUT * 10))
    while [ "$_n" -lt "$_max" ]; do
        if "$PYTHON" - "$_port" <<'PY' 2>/dev/null
import socket, sys
s = socket.socket()
s.settimeout(1)
try:
    s.connect(("127.0.0.1", int(sys.argv[1])))
    s.close()
except OSError:
    sys.exit(1)
PY
        then
            return 0
        fi
        _n=$((_n + 1))
        sleep 0.1
    done
    return 1
}

# --- Egress is MANDATORY -----------------------------------------------------
if [ ! -S "$EGRESS_SOCKET" ]; then
    echo "stigmergy-entrypoint: egress socket $EGRESS_SOCKET absent — refusing to start (fail-closed)" >&2
    exit "$EXIT_CAGE_UNAVAILABLE"
fi
"$PYTHON" "$SHIM" egress &
if ! wait_ready "$EGRESS_PORT"; then
    echo "stigmergy-entrypoint: egress shim never came up on 127.0.0.1:$EGRESS_PORT — fail-closed" >&2
    exit "$EXIT_CAGE_UNAVAILABLE"
fi
export HTTPS_PROXY="http://127.0.0.1:$EGRESS_PORT"
export https_proxy="http://127.0.0.1:$EGRESS_PORT"

# --- Relay is OPTIONAL (its socket is mounted at .25) ------------------------
if [ -S "$RELAY_SOCKET" ]; then
    "$PYTHON" "$SHIM" relay &
    if ! wait_ready "$RELAY_PORT"; then
        echo "stigmergy-entrypoint: relay socket present but relay shim never came up — fail-closed" >&2
        exit "$EXIT_CAGE_UNAVAILABLE"
    fi
    export ANTHROPIC_BASE_URL="http://127.0.0.1:$RELAY_PORT"
fi

# Loopback (the shims themselves + the relay base URL) must NEVER be routed
# back through the egress proxy — that would make the relay call CONNECT to a
# private address, which the proxy denies. NO_PROXY exempts loopback.
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"

exec "$@"
