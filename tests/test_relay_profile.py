"""bead .147 — per-dispatch RELAY ENDPOINT PROFILE (spec §1A/§1F/§3.1):
`RelayProfile` construction, `derive_relay_profile` derivation (pricing
class, wire, base URL, auth, endpoint allowlist, pinned headers), the
DEFAULT profile's byte-identity with the pre-`.147` constants, and the
`cli`/`daemon` §1F wiring (profile side-channel cell, per-profile
forwarder cache, no `relay_setup_fn` signature change).

Self-contained (own fakes + fixtures, the suite's convention). The profile
is a BUILD-time object: unknown wire/auth/pricing must fail at construction
(spec §2.5), never at request time.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

import stigmergy.cli as cli
from stigmergy.charter import Charter
from stigmergy.daemon import Daemon
from stigmergy.registry import Registry, load_registry
from stigmergy.relay import (
    CAPABILITY_HEADER_DEFAULT,
    DEFAULT_ALLOWED_ENDPOINTS,
    DEFAULT_RELAY_PROFILE,
    RelayError,
    RelayProfile,
    derive_relay_profile,
)

# The pre-`.147` module constants the DEFAULT profile must reproduce EXACTLY
# (regression gate §2.1: the anthropic lane is byte-for-byte).
_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
_ANTHROPIC_VERSION = "2023-06-01"


# --------------------------------------------------------------------------- #
# Real-registry fixture (load_registry — the derivation's real input type)    #
# --------------------------------------------------------------------------- #
def _registry_with(tmp_path: Path, toml: str) -> Registry:
    models = tmp_path / "models.toml"
    models.write_text(toml, encoding="utf-8")
    return load_registry(models)


SUBSCRIPTION_OPENAI_TOML = """
[kimi3]
provider = "synthetic"
family = "kimi"
version = "hf:moonshotai/Kimi-K3"
pricing = "subscription"
marginal_usd = 0.0
quota = "synthetic-2500req-5h"
oa_provider_key = "synthetic"
oa_base_url = "https://api.synthetic.new/openai/v1"
oa_type = "openai"
"""

LOCAL_OPENAI_TOML = """
[blackwell]
provider = "local"
family = "qwen"
version = "qwen38-27b-fp8"
pricing = "local"
marginal_usd = 0.0
approved = true
oa_type = "openai"
oa_base_url = "http://10.0.20.111:8000/v1"
"""

ANTHROPIC_TOML = """
[opus]
provider = "anthropic"
family = "claude"
version = "opus-4-1-20250805"
pricing = "metered"
input_usd_per_mtok = 15.0
output_usd_per_mtok = 75.0
"""


def _charter() -> Charter:
    # derive_relay_profile reads only the registry entry in v0; a minimal
    # Charter stands in for the derivation surface (lane fields are the same
    # model name the registry entry is keyed by).
    return Charter(raw={}, resolved_hash="test", warnings=())


# =========================================================================== #
# §3.1 Profile derivation                                                      #
# =========================================================================== #
class TestDeriveRelayProfile:
    def test_subscription_maps_to_metered(self, tmp_path):
        reg = _registry_with(tmp_path, SUBSCRIPTION_OPENAI_TOML)
        profile = derive_relay_profile(lane_model="kimi3", registry=reg, charter=_charter())

        assert profile.pricing_class == "metered"
        assert profile.wire == "openai"
        assert profile.auth == "bearer"
        assert profile.capability_header == "authorization"
        assert profile.upstream_base_url == "https://api.synthetic.new/openai/v1"
        assert profile.allowed_endpoints == frozenset({("POST", "/chat/completions")})
        assert profile.pinned_headers == {}
        assert profile.is_metered is True

    def test_local_maps_to_local(self, tmp_path):
        reg = _registry_with(tmp_path, LOCAL_OPENAI_TOML)
        profile = derive_relay_profile(lane_model="blackwell", registry=reg, charter=_charter())

        assert profile.pricing_class == "local"
        assert profile.wire == "openai"
        # Review fix F1: a `local` (declared-$0) openai lane is KEYLESS —
        # deriving `bearer` there would inject the Synthetic key toward the
        # blackwell LAN upstream. `none` = the header is omitted upstream.
        assert profile.auth == "none"
        assert profile.capability_header == "authorization"
        assert profile.upstream_base_url == "http://10.0.20.111:8000/v1"
        assert profile.allowed_endpoints == frozenset({("POST", "/chat/completions")})
        assert profile.is_metered is False

    def test_local_openai_lane_never_carries_bearer(self, tmp_path):
        # The F1 regression, stated as the security property: a local-priced
        # openai derivation NEVER carries the bearer credential contract —
        # a foreign provider key must be structurally unreachable on a $0
        # lane (the header is omitted upstream entirely).
        reg = _registry_with(tmp_path, LOCAL_OPENAI_TOML)
        profile = derive_relay_profile(
            lane_model="blackwell", registry=reg, charter=_charter()
        )
        assert profile.auth == "none", (
            "a local-priced openai lane must never derive the bearer "
            "credential contract (foreign-key leak toward a $0 upstream)"
        )

    def test_missing_wire_is_a_construction_error(self, tmp_path):
        # A local entry with NO explicit oa_type has oa_type=None -> the
        # derivation must FAIL LOUD (build error), never guess a wire.
        reg = _registry_with(
            tmp_path,
            """
[qw]
provider = "local"
family = "qwen"
version = "v1"
pricing = "local"
marginal_usd = 0.0
approved = true
""",
        )
        with pytest.raises(RelayError):
            derive_relay_profile(lane_model="qw", registry=reg, charter=_charter())

    def test_unknown_wire_value_is_a_construction_error(self, tmp_path):
        # A declared-but-unrecognized wire value is a build error (spec
        # §2.5) — never a request-time surprise.
        reg = _registry_with(
            tmp_path,
            """
[gw]
provider = "synthetic"
family = "g"
version = "v1"
pricing = "subscription"
marginal_usd = 0.0
quota = "q"
oa_provider_key = "synthetic"
oa_base_url = "https://example.com/v1"
oa_type = "gemini"
""",
        )
        with pytest.raises(RelayError):
            derive_relay_profile(lane_model="gw", registry=reg, charter=_charter())

    def test_anthropic_wire_reproduces_today_exact_values(self, tmp_path):
        # Regression gate §2.1: the anthropic lane keeps today's byte-for-
        # byte behaviour — base URL, endpoint, auth, capability header, and
        # the pinned anthropic-version.
        reg = _registry_with(tmp_path, ANTHROPIC_TOML)
        profile = derive_relay_profile(lane_model="opus", registry=reg, charter=_charter())

        assert profile.upstream_base_url == _ANTHROPIC_BASE_URL
        assert profile.wire == "anthropic"
        assert profile.auth == "x-api-key"
        assert profile.capability_header == CAPABILITY_HEADER_DEFAULT
        assert profile.pricing_class == "metered"
        assert profile.allowed_endpoints == DEFAULT_ALLOWED_ENDPOINTS
        assert profile.pinned_headers == {"anthropic-version": _ANTHROPIC_VERSION}

    def test_pricing_missing_defaults_to_metered(self, tmp_path, monkeypatch):
        # D10 fail-closed: an entry whose pricing is missing/None must map
        # to metered, never to local ($0 is a declared value, never a
        # fallback). Simulated with a stub entry whose `pricing` is absent:
        # the derivation must read a missing pricing as metered.
        from types import SimpleNamespace

        reg = _registry_with(tmp_path, SUBSCRIPTION_OPENAI_TOML)
        stub = SimpleNamespace(oa_type="openai", oa_base_url="https://example.com/v1")
        monkeypatch.setattr(
            reg, "resolve", lambda name: stub
        )  # no `pricing` attribute -> missing pricing

        profile = derive_relay_profile(lane_model="kimi3", registry=reg, charter=_charter())
        assert profile.pricing_class == "metered"
        assert profile.wire == "openai"
        assert profile.upstream_base_url == "https://example.com/v1"

    def test_openai_without_base_url_is_a_construction_error(self, tmp_path):
        # An OpenAI wire with no declared base URL is unbudgetable to relay
        # to — a loud build error, never a silent base-URL guess.
        reg = _registry_with(
            tmp_path,
            """
[nobase]
provider = "synthetic"
family = "n"
version = "v1"
pricing = "subscription"
marginal_usd = 0.0
quota = "q"
oa_provider_key = "synthetic"
oa_type = "openai"
""",
        )
        with pytest.raises(RelayError):
            derive_relay_profile(lane_model="nobase", registry=reg, charter=_charter())

    def test_derivation_is_frozen(self, tmp_path):
        reg = _registry_with(tmp_path, LOCAL_OPENAI_TOML)
        profile = derive_relay_profile(lane_model="blackwell", registry=reg)
        with pytest.raises(FrozenInstanceError):
            profile.wire = "anthropic"  # type: ignore[misc]


# =========================================================================== #
# RelayProfile construction validation (build-time, §2.5)                     #
# =========================================================================== #
class TestRelayProfileConstruction:
    def _valid(self, **over) -> dict[str, Any]:
        base = dict(
            upstream_base_url=_ANTHROPIC_BASE_URL,
            wire="anthropic",
            auth="x-api-key",
            capability_header=CAPABILITY_HEADER_DEFAULT,
            pricing_class="metered",
            allowed_endpoints=DEFAULT_ALLOWED_ENDPOINTS,
            pinned_headers={"anthropic-version": _ANTHROPIC_VERSION},
        )
        base.update(over)
        return base

    def test_valid_profile_constrains(self):
        p = RelayProfile(**self._valid())
        assert p.is_metered is True

    def test_unknown_wire_is_relay_error_at_construction(self):
        with pytest.raises(RelayError) as exc:
            RelayProfile(**self._valid(wire="gemini"))
        assert "gemini" in str(exc.value)

    def test_unknown_auth_is_relay_error_at_construction(self):
        with pytest.raises(RelayError) as exc:
            RelayProfile(**self._valid(auth="api-key-header"))
        assert "api-key-header" in str(exc.value)

    def test_unknown_pricing_class_is_relay_error_at_construction(self):
        # There is NO "unknown pricing -> local" fallthrough: a pricing
        # class that is not metered/local is a build error (D10).
        with pytest.raises(RelayError) as exc:
            RelayProfile(**self._valid(pricing_class="enterprise"))
        assert "enterprise" in str(exc.value)

    def test_relative_or_empty_base_url_is_relay_error(self):
        with pytest.raises(RelayError):
            RelayProfile(**self._valid(upstream_base_url="api.anthropic.com"))
        with pytest.raises(RelayError):
            RelayProfile(**self._valid(upstream_base_url=""))

    def test_empty_capability_header_is_relay_error(self):
        with pytest.raises(RelayError):
            RelayProfile(**self._valid(capability_header=""))

    def test_is_metered_reflects_pricing_class(self):
        assert RelayProfile(**self._valid(pricing_class="local")).is_metered is False
        assert RelayProfile(**self._valid(pricing_class="metered")).is_metered is True


# =========================================================================== #
# DEFAULT_RELAY_PROFILE — byte-identity with pre-`.147` constants             #
# =========================================================================== #
class TestDefaultProfile:
    def test_default_reproduces_pre147_constants(self):
        assert DEFAULT_RELAY_PROFILE.upstream_base_url == _ANTHROPIC_BASE_URL
        assert DEFAULT_RELAY_PROFILE.wire == "anthropic"
        assert DEFAULT_RELAY_PROFILE.auth == "x-api-key"
        assert DEFAULT_RELAY_PROFILE.capability_header == CAPABILITY_HEADER_DEFAULT
        assert DEFAULT_RELAY_PROFILE.pricing_class == "metered"
        assert DEFAULT_RELAY_PROFILE.allowed_endpoints == DEFAULT_ALLOWED_ENDPOINTS
        assert DEFAULT_RELAY_PROFILE.pinned_headers == {"anthropic-version": _ANTHROPIC_VERSION}
        assert DEFAULT_RELAY_PROFILE.is_metered is True

    def test_default_matches_derived_anthropic_profile(self, tmp_path):
        # The DEFAULT profile must equal what derive_relay_profile produces
        # for an anthropic metered lane (the cli fallback == a derived
        # anthropic profile — the pre-`.147` shape).
        reg = _registry_with(tmp_path, ANTHROPIC_TOML)
        derived = derive_relay_profile(lane_model="opus", registry=reg, charter=_charter())
        assert derived == DEFAULT_RELAY_PROFILE


# =========================================================================== #
# §1F cli wiring — profile side-channel, per-profile forwarder cache         #
# =========================================================================== #
FIXTURES = Path(__file__).parent / "fixtures"
VALID_CHARTER_PATH = FIXTURES / "charter_valid.toml"
MODELS_REGISTRY_PATH = FIXTURES / "models.toml"
BASE_CHARTER_TOML = VALID_CHARTER_PATH.read_text()


def _make_local_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "src_repo"
    (repo / "prompts").mkdir(parents=True)
    for name in ("code01", "critic01", "critic03", "rangecrit02"):
        (repo / "prompts" / name).write_text(f"{name} template\n")
    (repo / "README.md").write_text("fixture\n")
    cfg = ["-c", "user.email=test@example.com", "-c", "user.name=T"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", *cfg, "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(
        ["git", *cfg, "-C", str(repo), "commit", "-q", "-m", "init"], check=True
    )
    return repo


def _scaffold_shipyard(tmp_path: Path) -> object:
    """Scaffold the real 'shipyard' rig (mirrors test_cli.py) and return the
    resolved rig — enough for `_build_daemon` to run its real wiring."""
    from stigmergy.rig import create_rig, resolve_rig

    repo = _make_local_repo(tmp_path)
    charter_dir = tmp_path / "charter_src"
    charter_dir.mkdir(exist_ok=True)
    text = BASE_CHARTER_TOML.replace('repo = "path-or-url"', f'repo = "{repo}"')
    charter_path = charter_dir / "charter.toml"
    charter_path.write_text(text)
    shutil.copy(MODELS_REGISTRY_PATH, charter_dir / "models.toml")
    rigs_root = tmp_path / "rigs"
    create_rig(charter_path, base_dir=rigs_root)
    return resolve_rig("shipyard", rigs_root=rigs_root)


def test_cli_relay_wiring_profile_cell_and_forwarder_cache(
    tmp_path: Path, monkeypatch
) -> None:
    """bead .147 §1F: the `relay_setup_fn` signature stays 2-arg, the profile
    arrives via the side-channel cell, and the per-profile forwarder is built
    + cached per (base_url, auth). A `none`-auth profile's key provider
    returns None (header omitted upstream); a `bearer` profile gets the
    Synthetic key ref (wired by name only). The DEFAULT anthropic profile
    keeps the pre-`.147` shared key provider + forwarder objects."""
    from stigmergy.cli import _build_daemon
    from stigmergy.relay import RelayProfile

    resolved = _scaffold_shipyard(tmp_path)

    kp_refs: list[str] = []

    def fake_make_op_key_provider(ref):
        kp_refs.append(ref)

        def _provider():
            return "sk-fake-not-real"

        return _provider

    fwd_urls: list[str] = []

    def fake_make_forwarder(*, base_url, **kw):
        fwd_urls.append(base_url)
        return f"FAKE_FORWARDER[{base_url}]"

    captured: dict = {}

    def fake_start_relay(provisional_id, runtime_dir, relay, *, forwarder, log_path):
        captured["relay"] = relay
        captured["forwarder"] = forwarder
        captured["provisional_id"] = provisional_id
        return object()

    monkeypatch.setattr(cli, "make_op_key_provider", fake_make_op_key_provider)
    monkeypatch.setattr(cli, "make_urllib_forwarder", fake_make_forwarder)
    monkeypatch.setattr(cli, "start_relay", fake_start_relay)

    daemon = _build_daemon(resolved)
    try:
        # The daemon is wired with a NON-None profile cell (the side-channel).
        assert daemon._relay_profile_cell is not None
        cell = daemon._relay_profile_cell

        # (a) DEFAULT (anthropic) profile — cell empty, the 2-arg closure
        # falls back to the pre-`.147` shared pair (the .25 test keeps green).
        daemon._relay_setup_fn("dispatch-default", tmp_path)
        default_relay = captured["relay"]
        assert default_relay._wire == "anthropic"
        assert default_relay._auth == "x-api-key"
        assert default_relay._pricing_class == "metered"
        assert default_relay._upstream_headers_pinned == {"anthropic-version": _ANTHROPIC_VERSION}
        # The anthropic forwarder is the pre-`.147` shared object.
        assert captured["forwarder"] == "FAKE_FORWARDER[https://api.anthropic.com]"

        # (b) A `none`-auth OpenAI (blackwell-class) profile written into the
        # cell: the closure builds a key provider returning None + a
        # per-base_url forwarder, and a relay with the OpenAI shape.
        blackwell = RelayProfile(
            upstream_base_url="http://10.0.20.111:8000/v1",
            wire="openai",
            auth="none",
            capability_header="authorization",
            pricing_class="local",
            allowed_endpoints=frozenset({("POST", "/chat/completions")}),
            pinned_headers={},
        )
        cell["dispatch-bw"] = blackwell
        daemon._relay_setup_fn("dispatch-bw", tmp_path)
        bw_relay = captured["relay"]
        assert bw_relay._wire == "openai"
        assert bw_relay._auth == "none"
        assert bw_relay._pricing_class == "local"
        assert bw_relay._capability_header == "authorization"
        assert bw_relay._upstream_headers_pinned == {}
        assert bw_relay._allowed_endpoints == frozenset({("POST", "/chat/completions")})
        # The key provider returns None (header omitted upstream) ...
        assert bw_relay._key_provider() is None
        # ... and the forwarder is built for the blackwell base URL.
        assert captured["forwarder"] == "FAKE_FORWARDER[http://10.0.20.111:8000/v1]"

        # (c) A `bearer` OpenAI (Synthetic) profile: the key ref is wired by
        # NAME (the Synthetic item) — never fetched, never logged.
        synthetic = RelayProfile(
            upstream_base_url="https://api.synthetic.new/openai/v1",
            wire="openai",
            auth="bearer",
            capability_header="authorization",
            pricing_class="metered",
            allowed_endpoints=frozenset({("POST", "/chat/completions")}),
            pinned_headers={},
        )
        cell["dispatch-syn"] = synthetic
        daemon._relay_setup_fn("dispatch-syn", tmp_path)
        syn_relay = captured["relay"]
        assert syn_relay._wire == "openai"
        assert syn_relay._auth == "bearer"
        assert syn_relay._pricing_class == "metered"
        assert captured["forwarder"] == "FAKE_FORWARDER[https://api.synthetic.new/openai/v1]"
        # The Synthetic key ref (by name only) was wired for the bearer lane.
        assert cli._SYNTHETIC_RELAY_KEY_REF in kp_refs

        # (d) Forwarder cache: a second dispatch with the SAME (base_url,
        # auth) reuses the cached forwarder — NO new forwarder is built.
        n_fwd_before = len(fwd_urls)
        cell["dispatch-bw2"] = blackwell
        daemon._relay_setup_fn("dispatch-bw2", tmp_path)
        assert len(fwd_urls) == n_fwd_before  # cached, not rebuilt
        assert captured["forwarder"] == "FAKE_FORWARDER[http://10.0.20.111:8000/v1]"
    finally:
        resolved.store.close()


def test_cli_default_profile_fallback_keeps_2arg_closure(tmp_path: Path, monkeypatch) -> None:
    """The `relay_setup_fn(dispatch_id, runtime_dir)` 2-arg contract (pinned
    by ten+ existing stubs + test_cli.py:345) keeps working: with no entry in
    the cell, the closure falls back to the DEFAULT anthropic profile and
    still builds a functional relay sharing the daemon's CapabilityStore."""
    from stigmergy.cli import _build_daemon

    resolved = _scaffold_shipyard(tmp_path)

    captured: dict = {}

    def fake_start_relay(provisional_id, runtime_dir, relay, *, forwarder, log_path):
        captured["relay"] = relay
        return object()

    monkeypatch.setattr(cli, "make_op_key_provider", lambda ref: (lambda: "sk-fake"))
    monkeypatch.setattr(
        cli, "make_urllib_forwarder", lambda **kw: "FAKE_FORWARDER"
    )
    monkeypatch.setattr(cli, "start_relay", fake_start_relay)

    daemon = _build_daemon(resolved)
    try:
        # Direct 2-arg call with an id that has NO cell entry -> fallback.
        handle = daemon._relay_setup_fn("relay-xyz", tmp_path)
        assert handle is not None
        relay = captured["relay"]
        assert relay._store is daemon._capability_store
        assert relay._wire == "anthropic"
        assert relay._upstream_headers_pinned == {"anthropic-version": _ANTHROPIC_VERSION}
    finally:
        resolved.store.close()


# =========================================================================== #
# §1F daemon side — the dispatch path DERIVES + writes the profile to the     #
# cell keyed by the real dispatch id, immediately before _setup_relay.        #
# =========================================================================== #
def test_daemon_setup_relay_writes_derived_profile_to_cell(tmp_path: Path) -> None:
    """With `relay_profile_cell` wired, `_setup_relay` derives the lane's
    profile and writes it into the cell under the real dispatch id BEFORE
    calling `relay_setup_fn` (which then reads it). Derivation failure is a
    loud dispatch failure (never a silent fallback to a different upstream)."""
    from types import SimpleNamespace

    reg = _registry_with(tmp_path, LOCAL_OPENAI_TOML)

    # A minimal daemon instance with only the seams _setup_relay touches
    # (it reads _relay_setup_fn, _relay_profile_cell, _registry, _charter,
    # and the rig paths for the runtime dir). We build it via __new__ to
    # avoid the full constructor's heavy collaborator wiring.
    daemon = Daemon.__new__(Daemon)
    daemon._relay_setup_fn = lambda pid, rtdir: ("SETUP_CALL", pid, rtdir)
    cell: dict = {}
    daemon._relay_profile_cell = cell
    daemon._registry = reg
    daemon._charter = _charter()
    daemon._rig_paths = {"clones_root": tmp_path / "clones"}

    lane = SimpleNamespace(model="blackwell", name="cheap", prompt="code01")

    result = daemon._setup_relay("ticket-1", 0.0, "real-dispatch-id", lane=lane)

    # The profile was derived and written to the cell under the REAL id,
    # BEFORE relay_setup_fn was called ...
    assert "real-dispatch-id" in cell
    assert cell["real-dispatch-id"].pricing_class == "local"
    assert cell["real-dispatch-id"].wire == "openai"
    assert cell["real-dispatch-id"].upstream_base_url == "http://10.0.20.111:8000/v1"
    # ... and relay_setup_fn received the real dispatch id + a runtime dir.
    assert result[0] == "SETUP_CALL"
    assert result[1] == "real-dispatch-id"

    # A lane with an unknown wire is a LOUD dispatch failure (RelayError),
    # never a silent fallback to a DIFFERENT upstream.
    (tmp_path / "bad").mkdir(parents=True, exist_ok=True)
    (tmp_path / "bad").mkdir(parents=True, exist_ok=True)
    bad_reg = _registry_with(
        tmp_path / "bad",
        """
[badgw]
provider = "synthetic"
family = "b"
version = "v1"
pricing = "subscription"
marginal_usd = 0.0
quota = "q"
oa_provider_key = "synthetic"
oa_base_url = "https://example.com/v1"
oa_type = "gemini"
""",
    )
    daemon._registry = bad_reg
    with pytest.raises(RelayError):
        daemon._setup_relay(
            "ticket-2", 0.0, "real-dispatch-id-2",
            lane=SimpleNamespace(model="badgw", name="cheap", prompt="code01"),
        )
    assert "real-dispatch-id-2" not in cell


def test_daemon_without_profile_cell_skips_derivation(tmp_path: Path) -> None:
    """Backward-compat gate: with `relay_profile_cell=None` (the daemon
    default), `_setup_relay` NEVER calls derive_relay_profile (no registry
    read, no cell write) and just forwards to relay_setup_fn — today's
    behaviour for the .25 wiring + existing stubs."""
    from types import SimpleNamespace

    daemon = Daemon.__new__(Daemon)
    daemon._relay_setup_fn = lambda pid, rtdir: ("SETUP_CALL", pid)
    daemon._relay_profile_cell = None  # default: derivation skipped
    daemon._registry = None  # would raise if derivation were attempted
    daemon._charter = None
    daemon._rig_paths = {"clones_root": tmp_path / "clones"}

    result = daemon._setup_relay(
        "ticket-1", 0.0, "real-dispatch-id",
        lane=SimpleNamespace(model="blackwell", name="cheap", prompt="code01"),
    )
    # relay_setup_fn ran with the real id; no derivation was attempted
    # (daemon._registry is None — a derive call would have crashed here).
    assert result == ("SETUP_CALL", "real-dispatch-id")
