"""Tests for stigmergy.charter (SPEC.md §5 charter schema, §10 AC1).

Loader pipeline under test: parse TOML file -> apply in-code defaults ->
apply `SG_*` env overrides -> validate the MERGED result (validation runs
last, over the merged config, and fails closed).
"""

import shutil
from pathlib import Path

import pytest

from stigmergy.charter import (
    CIRCUIT_BREAKER_THRESHOLD,
    DEFAULT_CHARTER,
    Charter,
    CharterError,
    classify_diff,
    load_charter,
    resolve_check_resources,
)
from stigmergy.checks import DEFAULT_CHECK_RESOURCES, CheckResources

FIXTURES = Path(__file__).parent / "fixtures"
VALID_CHARTER_PATH = FIXTURES / "charter_valid.toml"
MODELS_REGISTRY_PATH = FIXTURES / "models.toml"

BASE_CHARTER_TOML = VALID_CHARTER_PATH.read_text()


def make_charter(tmp_path: Path, content: str) -> Path:
    """Write a mutated charter + a copy of the registry fixture into tmp_path."""
    charter_path = tmp_path / "charter.toml"
    charter_path.write_text(content)
    shutil.copy(MODELS_REGISTRY_PATH, tmp_path / "models.toml")
    return charter_path


def mutate(old: str, new: str) -> str:
    """Apply a single, must-match-exactly-once substitution to the base charter."""
    assert BASE_CHARTER_TOML.count(old) == 1, f"expected exactly one occurrence of {old!r}"
    return BASE_CHARTER_TOML.replace(old, new)


# --- happy path ------------------------------------------------------------


def test_valid_charter_loads_with_same_model_warning() -> None:
    """Three-rung ladder (cheap->default->exquisite) + opus critic loads
    clean; the opus-exquisite-rung vs opus-critic overlap is recorded as a
    warning, never an exception."""
    charter = load_charter(VALID_CHARTER_PATH, env={})
    assert isinstance(charter, Charter)
    assert charter.raw["lanes"]["exquisite"]["model"] == "opus"
    assert charter.raw["roles"]["critic"]["model"] == "opus"
    assert len(charter.warnings) >= 1
    assert any("opus" in w and "exquisite" in w for w in charter.warnings)


def test_valid_charter_exposes_ladder() -> None:
    charter = load_charter(VALID_CHARTER_PATH, env={})
    assert charter.raw["stepup"]["ladder"] == ["cheap", "default", "exquisite"]


# --- rule 1: unknown keys ---------------------------------------------------


def test_unknown_top_level_key_rejected(tmp_path: Path) -> None:
    content = BASE_CHARTER_TOML + '\n[bogus]\nfoo = 1\n'
    charter_path = make_charter(tmp_path, content)
    with pytest.raises(CharterError):
        load_charter(charter_path, env={})


def test_unknown_key_in_loop_budgets_rejected(tmp_path: Path) -> None:
    content = mutate(
        "[loop.budgets]\ndispatches = 50\nusd = 25.0\ngate_calls = 30\n",
        "[loop.budgets]\ndispatches = 50\nusd = 25.0\ngate_calls = 30\nbogus_key = 1\n",
    )
    charter_path = make_charter(tmp_path, content)
    with pytest.raises(CharterError):
        load_charter(charter_path, env={})


# --- rule 2: workers -----------------------------------------------------


def test_workers_greater_than_one_accepted(tmp_path: Path) -> None:
    content = mutate("workers = 1", "workers = 2")
    charter_path = make_charter(tmp_path, content)
    charter = load_charter(charter_path, env={})
    assert charter.raw["loop"]["concurrency"]["workers"] == 2


def test_workers_equal_one_ok(tmp_path: Path) -> None:
    charter_path = make_charter(tmp_path, BASE_CHARTER_TOML)
    charter = load_charter(charter_path, env={})
    assert charter.raw["loop"]["concurrency"]["workers"] == 1


def test_workers_absent_defaults_to_one_ok(tmp_path: Path) -> None:
    content = mutate("[loop.concurrency]\nworkers = 1\n", "")
    charter_path = make_charter(tmp_path, content)
    charter = load_charter(charter_path, env={})
    assert charter.raw["loop"]["concurrency"]["workers"] == 1


def test_workers_zero_rejected(tmp_path: Path) -> None:
    content = mutate("workers = 1", "workers = 0")
    charter_path = make_charter(tmp_path, content)
    with pytest.raises(CharterError):
        load_charter(charter_path, env={})


def test_workers_negative_rejected(tmp_path: Path) -> None:
    content = mutate("workers = 1", "workers = -1")
    charter_path = make_charter(tmp_path, content)
    with pytest.raises(CharterError):
        load_charter(charter_path, env={})


def test_workers_bool_rejected(tmp_path: Path) -> None:
    content = mutate("workers = 1", "workers = true")
    charter_path = make_charter(tmp_path, content)
    with pytest.raises(CharterError):
        load_charter(charter_path, env={})


def test_workers_non_integer_rejected(tmp_path: Path) -> None:
    content = mutate("workers = 1", "workers = 1.5")
    charter_path = make_charter(tmp_path, content)
    with pytest.raises(CharterError):
        load_charter(charter_path, env={})


def test_workers_larger_positive_integer_accepted(tmp_path: Path) -> None:
    content = mutate("workers = 1", "workers = 4")
    charter_path = make_charter(tmp_path, content)
    charter = load_charter(charter_path, env={})
    assert charter.raw["loop"]["concurrency"]["workers"] == 4


# --- rule 3: unknown model / registry miss ---------------------------------


def test_lane_unknown_model_rejected(tmp_path: Path) -> None:
    content = mutate('model = "haiku"', 'model = "does-not-exist"')
    charter_path = make_charter(tmp_path, content)
    with pytest.raises(CharterError):
        load_charter(charter_path, env={})


def test_critic_unknown_model_rejected(tmp_path: Path) -> None:
    content = mutate(
        '[roles.critic]\nmodel = "opus"',
        '[roles.critic]\nmodel = "does-not-exist"',
    )
    charter_path = make_charter(tmp_path, content)
    with pytest.raises(CharterError):
        load_charter(charter_path, env={})


# --- rule 4a: lane topology (selector-less fallthrough) --------------------


def test_zero_selectorless_lanes_rejected(tmp_path: Path) -> None:
    content = mutate(
        '[lanes.default]\ndriver = "claude-code"\nmodel = "sonnet"\nprompt = "code01"\n'
        'egress = ["inference", "registries"]\n',
        '[lanes.default]\nselector = { label = "any" }\ndriver = "claude-code"\n'
        'model = "sonnet"\nprompt = "code01"\negress = ["inference", "registries"]\n',
    )
    charter_path = make_charter(tmp_path, content)
    with pytest.raises(CharterError):
        load_charter(charter_path, env={})


def test_two_selectorless_lanes_rejected(tmp_path: Path) -> None:
    content = mutate(
        '[lanes.cheap]\nselector = { label = "local-ok" }\ndriver = "claude-code"\n'
        'model = "haiku"\nprompt = "code01"\negress = ["inference", "registries"]\n',
        '[lanes.cheap]\ndriver = "claude-code"\nmodel = "haiku"\nprompt = "code01"\n'
        'egress = ["inference", "registries"]\n',
    )
    charter_path = make_charter(tmp_path, content)
    with pytest.raises(CharterError):
        load_charter(charter_path, env={})


# --- rule 4b: entry=false lanes ---------------------------------------------


def test_entry_false_lane_missing_from_ladder_rejected(tmp_path: Path) -> None:
    content = mutate(
        'ladder = ["cheap", "default", "exquisite"]',
        'ladder = ["cheap", "default"]',
    )
    charter_path = make_charter(tmp_path, content)
    with pytest.raises(CharterError):
        load_charter(charter_path, env={})


def test_entry_false_lane_with_selector_rejected(tmp_path: Path) -> None:
    content = mutate(
        '[lanes.exquisite]\nentry = false\ndriver = "claude-code"\nmodel = "opus"\n'
        'prompt = "code01"\negress = ["inference", "registries"]\n',
        '[lanes.exquisite]\nentry = false\nselector = { label = "x" }\n'
        'driver = "claude-code"\nmodel = "opus"\nprompt = "code01"\n'
        'egress = ["inference", "registries"]\n',
    )
    charter_path = make_charter(tmp_path, content)
    with pytest.raises(CharterError):
        load_charter(charter_path, env={})


# --- rule 4c: ladder references a defined lane ------------------------------


def test_ladder_undefined_lane_rejected(tmp_path: Path) -> None:
    content = mutate(
        'ladder = ["cheap", "default", "exquisite"]',
        'ladder = ["cheap", "default", "exquisite", "nonexistent"]',
    )
    charter_path = make_charter(tmp_path, content)
    with pytest.raises(CharterError):
        load_charter(charter_path, env={})


# --- rule 6: timer keys ------------------------------------------------------


def test_timer_key_not_seconds_suffixed_rejected(tmp_path: Path) -> None:
    content = mutate("poll_seconds = 15", "poll = 15")
    charter_path = make_charter(tmp_path, content)
    with pytest.raises(CharterError):
        load_charter(charter_path, env={})


# --- rule 7: budgets / retries positive ints --------------------------------


def test_budgets_dispatches_zero_rejected(tmp_path: Path) -> None:
    content = mutate("dispatches = 50", "dispatches = 0")
    charter_path = make_charter(tmp_path, content)
    with pytest.raises(CharterError):
        load_charter(charter_path, env={})


def test_budgets_dispatches_negative_rejected(tmp_path: Path) -> None:
    content = mutate("dispatches = 50", "dispatches = -1")
    charter_path = make_charter(tmp_path, content)
    with pytest.raises(CharterError):
        load_charter(charter_path, env={})


def test_budgets_dispatches_non_int_rejected(tmp_path: Path) -> None:
    content = mutate("dispatches = 50", 'dispatches = "fifty"')
    charter_path = make_charter(tmp_path, content)
    with pytest.raises(CharterError):
        load_charter(charter_path, env={})


def test_budgets_dispatches_bool_rejected(tmp_path: Path) -> None:
    """bool is a subclass of int in Python — must be rejected explicitly,
    not silently accepted as `1`/`0`."""
    content = mutate("dispatches = 50", "dispatches = true")
    charter_path = make_charter(tmp_path, content)
    with pytest.raises(CharterError):
        load_charter(charter_path, env={})


def test_budgets_usd_negative_rejected(tmp_path: Path) -> None:
    content = mutate("usd = 25.0", "usd = -5.0")
    charter_path = make_charter(tmp_path, content)
    with pytest.raises(CharterError):
        load_charter(charter_path, env={})


def test_budgets_usd_zero_rejected(tmp_path: Path) -> None:
    content = mutate("usd = 25.0", "usd = 0.0")
    charter_path = make_charter(tmp_path, content)
    with pytest.raises(CharterError):
        load_charter(charter_path, env={})


def test_retries_attempts_per_rung_zero_rejected(tmp_path: Path) -> None:
    content = mutate("attempts_per_rung = 3", "attempts_per_rung = 0")
    charter_path = make_charter(tmp_path, content)
    with pytest.raises(CharterError):
        load_charter(charter_path, env={})


# --- bead .107: loop.retries.critic_infra -----------------------------------


def test_critic_infra_absent_defaults_to_three(tmp_path: Path) -> None:
    """Absence -> default 3 (MUST stay < the daemon's
    `_CIRCUIT_BREAKER_THRESHOLD` (5) so a single poisoned ticket escalates
    itself before the global storm breaker halts the whole loop)."""
    charter_path = make_charter(tmp_path, BASE_CHARTER_TOML)
    charter = load_charter(charter_path, env={})
    assert charter.raw["loop"]["retries"]["critic_infra"] == 3


def test_critic_infra_explicit_value_loads(tmp_path: Path) -> None:
    content = mutate(
        "integration_failures = 2\nflake_reruns = 2\n",
        "integration_failures = 2\nflake_reruns = 2\ncritic_infra = 4\n",
    )
    charter_path = make_charter(tmp_path, content)
    charter = load_charter(charter_path, env={})
    assert charter.raw["loop"]["retries"]["critic_infra"] == 4


def test_critic_infra_zero_rejected(tmp_path: Path) -> None:
    content = mutate(
        "integration_failures = 2\nflake_reruns = 2\n",
        "integration_failures = 2\nflake_reruns = 2\ncritic_infra = 0\n",
    )
    charter_path = make_charter(tmp_path, content)
    with pytest.raises(CharterError):
        load_charter(charter_path, env={})


def test_critic_infra_negative_rejected(tmp_path: Path) -> None:
    content = mutate(
        "integration_failures = 2\nflake_reruns = 2\n",
        "integration_failures = 2\nflake_reruns = 2\ncritic_infra = -1\n",
    )
    charter_path = make_charter(tmp_path, content)
    with pytest.raises(CharterError):
        load_charter(charter_path, env={})


def test_critic_infra_at_circuit_breaker_threshold_rejected(tmp_path: Path) -> None:
    """review-followup: `critic_infra == CIRCUIT_BREAKER_THRESHOLD` (5) must
    be rejected — at this value the global breaker can fire at exactly the
    same trip count a single poisoned ticket would need to escalate itself,
    defeating bead .107's per-ticket escalation before it ever helps."""
    assert CIRCUIT_BREAKER_THRESHOLD == 5
    content = mutate(
        "integration_failures = 2\nflake_reruns = 2\n",
        "integration_failures = 2\nflake_reruns = 2\ncritic_infra = 5\n",
    )
    charter_path = make_charter(tmp_path, content)
    with pytest.raises(CharterError):
        load_charter(charter_path, env={})


def test_critic_infra_above_circuit_breaker_threshold_rejected(tmp_path: Path) -> None:
    """review-followup: `critic_infra = 6` (> threshold) is the exact
    footgun the cross-family review flagged — an operator's "let it retry a
    bit more" edit must be rejected, not silently reconstitute the
    single-ticket livelock-then-global-halt incident bead .107 fixed."""
    content = mutate(
        "integration_failures = 2\nflake_reruns = 2\n",
        "integration_failures = 2\nflake_reruns = 2\ncritic_infra = 6\n",
    )
    charter_path = make_charter(tmp_path, content)
    with pytest.raises(CharterError):
        load_charter(charter_path, env={})


def test_critic_infra_env_override_at_threshold_rejected(tmp_path: Path) -> None:
    """review-followup: the invariant is enforced on the MERGED (post-env-
    override) value, not just the file/default value — an
    `SG_LOOP__RETRIES__CRITIC_INFRA` override that lands on/above the
    threshold must be rejected exactly like a bad file value."""
    charter_path = make_charter(tmp_path, BASE_CHARTER_TOML)
    with pytest.raises(CharterError):
        load_charter(charter_path, env={"SG_LOOP__RETRIES__CRITIC_INFRA": "5"})


# ==========================================================================
# quota-governor wiring surface: loop.governor.relay_feed +
# loop.timers.park_escalation_seconds
# ==========================================================================


def test_governor_relay_feed_default_loads(tmp_path: Path) -> None:
    """The base fixture declares no [loop.governor] -> the in-code default
    (the relay JSONL feed glob, RELATIVE to the rig's clones root) is
    merged in and validated clean."""
    charter = load_charter(make_charter(tmp_path, BASE_CHARTER_TOML), env={})
    assert charter.raw["loop"]["governor"]["relay_feed"] == "_relay_runtime/relay-*.jsonl"


def test_governor_relay_feed_explicit_relative_loads(tmp_path: Path) -> None:
    content = BASE_CHARTER_TOML + '\n[loop.governor]\nrelay_feed = "feeds/relay-*.jsonl"\n'
    charter = load_charter(make_charter(tmp_path, content), env={})
    assert charter.raw["loop"]["governor"]["relay_feed"] == "feeds/relay-*.jsonl"


def test_governor_relay_feed_env_override_loads(tmp_path: Path) -> None:
    """`SG_LOOP__GOVERNOR__RELAY_FEED` overrides the feed location through
    the standard `SG_` env-override machinery (a str leaf, like every other
    string-valued key)."""
    charter_path = make_charter(tmp_path, BASE_CHARTER_TOML)
    charter = load_charter(
        charter_path, env={"SG_LOOP__GOVERNOR__RELAY_FEED": "alt/relay-*.jsonl"}
    )
    assert charter.raw["loop"]["governor"]["relay_feed"] == "alt/relay-*.jsonl"


def test_governor_relay_feed_absolute_rejected(tmp_path: Path) -> None:
    """A literal per-rig path must never live in the charter: absolute feed
    locations are a charter-load error (the value is resolved against the
    rig's clones root by the daemon wiring)."""
    content = BASE_CHARTER_TOML + '\n[loop.governor]\nrelay_feed = "/abs/relay.jsonl"\n'
    with pytest.raises(CharterError):
        load_charter(make_charter(tmp_path, content), env={})


def test_governor_relay_feed_empty_string_rejected(tmp_path: Path) -> None:
    content = BASE_CHARTER_TOML + '\n[loop.governor]\nrelay_feed = ""\n'
    with pytest.raises(CharterError):
        load_charter(make_charter(tmp_path, content), env={})


def test_governor_relay_feed_non_string_rejected(tmp_path: Path) -> None:
    content = BASE_CHARTER_TOML + "\n[loop.governor]\nrelay_feed = 5\n"
    with pytest.raises(CharterError):
        load_charter(make_charter(tmp_path, content), env={})


def test_governor_unknown_key_still_rejected(tmp_path: Path) -> None:
    # Adding `relay_feed` to the known set must NOT open the section to
    # arbitrary keys.
    content = BASE_CHARTER_TOML + "\n[loop.governor]\nbogus_knob = 1\n"
    with pytest.raises(CharterError):
        load_charter(make_charter(tmp_path, content), env={})


def test_no_literal_relay_jsonl_path_in_charter_module() -> None:
    """AC1 hard guarantee: the configurable default is the ONLY relay JSONL
    location string in charter.py — no hardcoded path literal anywhere else
    in the module (comments included)."""
    src = Path(__file__).resolve().parent.parent / "src" / "stigmergy" / "charter.py"
    lines = src.read_text(encoding="utf-8").splitlines()
    hits = [line for line in lines if "relay-" in line and ".jsonl" in line.lower()]
    assert len(hits) == 1
    # the sole occurrence is the configurable default value itself, not a
    # hardcoded path in code or comments.
    assert DEFAULT_CHARTER["loop"]["governor"]["relay_feed"] in hits[0]


def test_park_escalation_timer_default_loads(tmp_path: Path) -> None:
    """The base fixture declares no `park_escalation_seconds` -> the in-code
    default (28800s = 8h) is merged in and satisfies `_validate_timers`'
    positive-int rule as-is."""
    charter = load_charter(make_charter(tmp_path, BASE_CHARTER_TOML), env={})
    assert charter.raw["loop"]["timers"]["park_escalation_seconds"] == 28800


def test_park_escalation_timer_explicit_loads(tmp_path: Path) -> None:
    content = mutate(
        "poll_seconds = 15\ndispatch_timeout_seconds = 3600\nlease_ttl_seconds = 4500\n",
        "poll_seconds = 15\ndispatch_timeout_seconds = 3600\nlease_ttl_seconds = 4500\n"
        "park_escalation_seconds = 3600\n",
    )
    charter = load_charter(make_charter(tmp_path, content), env={})
    assert charter.raw["loop"]["timers"]["park_escalation_seconds"] == 3600


def test_park_escalation_timer_env_override_loads(tmp_path: Path) -> None:
    charter_path = make_charter(tmp_path, BASE_CHARTER_TOML)
    charter = load_charter(charter_path, env={"SG_LOOP__TIMERS__PARK_ESCALATION_SECONDS": "7200"})
    assert charter.raw["loop"]["timers"]["park_escalation_seconds"] == 7200


@pytest.mark.parametrize("bad", ["0", "-1", "true", '""', "1.5"])
def test_park_escalation_timer_bad_value_rejected(tmp_path: Path, bad: str) -> None:
    # `_validate_timers`' existing rule (any `_seconds` key: positive int)
    # applies to the new key with no loosening — non-positive, bool, string,
    # and float values are all load errors.
    content = mutate(
        "poll_seconds = 15\ndispatch_timeout_seconds = 3600\nlease_ttl_seconds = 4500\n",
        f"poll_seconds = 15\ndispatch_timeout_seconds = 3600\nlease_ttl_seconds = 4500\n"
        f"park_escalation_seconds = {bad}\n",
    )
    with pytest.raises(CharterError):
        load_charter(make_charter(tmp_path, content), env={})


# --- env overrides -----------------------------------------------------------


def test_env_override_changes_resolved_value_and_hash(tmp_path: Path) -> None:
    charter_path = make_charter(tmp_path, BASE_CHARTER_TOML)
    baseline = load_charter(charter_path, env={})
    overridden = load_charter(charter_path, env={"SG_LOOP__BUDGETS__USD": "10.0"})

    assert baseline.raw["loop"]["budgets"]["usd"] == 25.0
    assert overridden.raw["loop"]["budgets"]["usd"] == 10.0
    assert overridden.resolved_hash != baseline.resolved_hash


def test_env_override_unknown_path_rejected(tmp_path: Path) -> None:
    charter_path = make_charter(tmp_path, BASE_CHARTER_TOML)
    with pytest.raises(CharterError):
        load_charter(charter_path, env={"SG_LOOP__BUDGETS__BOGUS": "5"})


# --- resolved_hash -----------------------------------------------------------


def test_resolved_hash_stable_across_identical_reload(tmp_path: Path) -> None:
    charter_path = make_charter(tmp_path, BASE_CHARTER_TOML)
    first = load_charter(charter_path, env={})
    second = load_charter(charter_path, env={})
    assert first.resolved_hash == second.resolved_hash


def test_resolved_hash_changes_on_value_change(tmp_path: Path) -> None:
    charter_path = make_charter(tmp_path, BASE_CHARTER_TOML)
    baseline = load_charter(charter_path, env={})

    mutated_content = mutate("usd = 25.0", "usd = 30.0")
    mutated_path = make_charter(tmp_path, mutated_content)
    mutated = load_charter(mutated_path, env={})

    assert mutated.resolved_hash != baseline.resolved_hash


# --- classify_diff -----------------------------------------------------------


def _resolved(usd: float = 25.0, dispatches: int = 50, workers: int = 1, hosts=None) -> dict:
    return {
        "loop": {
            "budgets": {"usd": usd, "dispatches": dispatches, "gate_calls": 30},
            "concurrency": {"workers": workers},
        },
        "egress": {"inference": {"hosts": list(hosts or ["api.anthropic.com"])}},
    }


def test_classify_diff_budget_increase_is_expanding() -> None:
    prev = _resolved(usd=10.0)
    curr = _resolved(usd=20.0)
    assert classify_diff(prev, curr) == "expanding"


def test_classify_diff_dispatches_increase_is_expanding() -> None:
    prev = _resolved(dispatches=50)
    curr = _resolved(dispatches=100)
    assert classify_diff(prev, curr) == "expanding"


def test_classify_diff_workers_increase_is_expanding() -> None:
    prev = _resolved(workers=1)
    curr = _resolved(workers=2)
    assert classify_diff(prev, curr) == "expanding"


def test_classify_diff_egress_host_added_is_expanding() -> None:
    prev = _resolved(hosts=["api.anthropic.com"])
    curr = _resolved(hosts=["api.anthropic.com", "evil.example.com"])
    assert classify_diff(prev, curr) == "expanding"


def test_classify_diff_budget_decrease_is_safe() -> None:
    prev = _resolved(usd=25.0)
    curr = _resolved(usd=10.0)
    assert classify_diff(prev, curr) == "safe"


def test_classify_diff_equal_is_safe() -> None:
    prev = _resolved()
    curr = _resolved()
    assert classify_diff(prev, curr) == "safe"


# --- D14: dispatch_limits filing caps (bead workspace-e2uh.38, AC14 case 10)


def test_dispatch_limits_filed_keys_load(tmp_path: Path) -> None:
    content = mutate(
        "[loop.dispatch_limits]\noutput_tokens = 200000\ndriver_turns = 100\n",
        "[loop.dispatch_limits]\noutput_tokens = 200000\ndriver_turns = 100\n"
        "filed_tickets = 7\nfiled_ticket_bytes = 4096\n",
    )
    charter = load_charter(make_charter(tmp_path, content), env={})
    dl = charter.raw["loop"]["dispatch_limits"]
    assert dl["filed_tickets"] == 7
    assert dl["filed_ticket_bytes"] == 4096


def test_dispatch_limits_filed_keys_default_when_absent() -> None:
    # the base fixture declares no filed_* keys -> in-code defaults fill them
    # (deep-merge over DEFAULT_CHARTER).
    # SB ruling 2026-09-03 (contexthandoff01 T0 postmortem): the default cap
    # was 5 — too low. A polluted transport (harness bug, not worker spam)
    # plus dedup-tolerant salvage needs headroom: 20 = 4x the old default.
    charter = load_charter(VALID_CHARTER_PATH, env={})
    dl = charter.raw["loop"]["dispatch_limits"]
    assert dl["filed_tickets"] == 20
    assert dl["filed_ticket_bytes"] == 16384


def test_unknown_dispatch_limits_key_still_rejected(tmp_path: Path) -> None:
    # regression guard on _validate_keys: adding the two known filing keys must
    # NOT open the section to arbitrary keys.
    content = mutate(
        "[loop.dispatch_limits]\noutput_tokens = 200000\ndriver_turns = 100\n",
        "[loop.dispatch_limits]\noutput_tokens = 200000\ndriver_turns = 100\nbogus_limit = 1\n",
    )
    with pytest.raises(CharterError):
        load_charter(make_charter(tmp_path, content), env={})


# --- D14: dispatch_limits value validation (must be positive ints) -----------


def test_dispatch_limits_output_tokens_string_rejected(tmp_path: Path) -> None:
    content = mutate("output_tokens = 200000", 'output_tokens = "200000"')
    charter_path = make_charter(tmp_path, content)
    with pytest.raises(CharterError):
        load_charter(charter_path, env={})


def test_dispatch_limits_output_tokens_bool_rejected(tmp_path: Path) -> None:
    content = mutate("output_tokens = 200000", "output_tokens = true")
    charter_path = make_charter(tmp_path, content)
    with pytest.raises(CharterError):
        load_charter(charter_path, env={})


def test_dispatch_limits_output_tokens_zero_rejected(tmp_path: Path) -> None:
    content = mutate("output_tokens = 200000", "output_tokens = 0")
    charter_path = make_charter(tmp_path, content)
    with pytest.raises(CharterError):
        load_charter(charter_path, env={})


def test_dispatch_limits_output_tokens_negative_rejected(tmp_path: Path) -> None:
    content = mutate("output_tokens = 200000", "output_tokens = -1")
    charter_path = make_charter(tmp_path, content)
    with pytest.raises(CharterError):
        load_charter(charter_path, env={})


def test_dispatch_limits_driver_turns_string_rejected(tmp_path: Path) -> None:
    content = mutate("driver_turns = 100", 'driver_turns = "100"')
    charter_path = make_charter(tmp_path, content)
    with pytest.raises(CharterError):
        load_charter(charter_path, env={})


def test_dispatch_limits_driver_turns_zero_rejected(tmp_path: Path) -> None:
    content = mutate("driver_turns = 100", "driver_turns = 0")
    charter_path = make_charter(tmp_path, content)
    with pytest.raises(CharterError):
        load_charter(charter_path, env={})


def test_dispatch_limits_filed_tickets_bool_rejected(tmp_path: Path) -> None:
    content = mutate(
        "[loop.dispatch_limits]\noutput_tokens = 200000\ndriver_turns = 100\n",
        "[loop.dispatch_limits]\noutput_tokens = 200000\ndriver_turns = 100\n"
        "filed_tickets = true\n",
    )
    charter_path = make_charter(tmp_path, content)
    with pytest.raises(CharterError):
        load_charter(charter_path, env={})


def test_dispatch_limits_filed_ticket_bytes_negative_rejected(tmp_path: Path) -> None:
    content = mutate(
        "[loop.dispatch_limits]\noutput_tokens = 200000\ndriver_turns = 100\n",
        "[loop.dispatch_limits]\noutput_tokens = 200000\ndriver_turns = 100\n"
        "filed_ticket_bytes = -1\n",
    )
    charter_path = make_charter(tmp_path, content)
    with pytest.raises(CharterError):
        load_charter(charter_path, env={})


def test_dispatch_limits_valid_positive_ints_load(tmp_path: Path) -> None:
    content = mutate(
        "[loop.dispatch_limits]\noutput_tokens = 200000\ndriver_turns = 100\n",
        "[loop.dispatch_limits]\noutput_tokens = 300000\ndriver_turns = 150\n"
        "filed_tickets = 10\nfiled_ticket_bytes = 32768\n",
    )
    charter = load_charter(make_charter(tmp_path, content), env={})
    dl = charter.raw["loop"]["dispatch_limits"]
    assert dl["output_tokens"] == 300000
    assert dl["driver_turns"] == 150
    assert dl["filed_tickets"] == 10
    assert dl["filed_ticket_bytes"] == 32768


# --- .91: charter-configurable checker resource bounds --------------------
# Bead workspace-e2uh.91. Resolution order (lowest -> highest precedence):
# DEFAULT_CHECK_RESOURCES (the legacy hardcoded 60s/256m/1cpu/64m/64pids)
# -> [loop.check_resources] (applies to EVERY check, including synthesized
# ones like `ticket-tests`) -> [checks.<name>] per-check override (named
# charter checks only). Sizes (memory, scratch_size) are unit-bearing
# strings; counts (timeout_seconds, cpus, pids_limit) are numbers.


def test_check_entry_accepts_resource_overrides(tmp_path: Path) -> None:
    content = mutate(
        '[checks.pytest]\ncmd = "pytest -x -q"\n',
        '[checks.pytest]\ncmd = "pytest -x -q"\n'
        'timeout_seconds = 1800\nmemory = "4g"\ncpus = "4"\n'
        'scratch_size = "2g"\npids_limit = 512\n',
    )
    charter = load_charter(make_charter(tmp_path, content), env={})
    assert charter.raw["checks"]["pytest"]["timeout_seconds"] == 1800


def test_loop_check_resources_default_accepted(tmp_path: Path) -> None:
    content = BASE_CHARTER_TOML + (
        "\n[loop.check_resources]\n"
        'timeout_seconds = 900\nmemory = "2g"\ncpus = "2"\n'
        'scratch_size = "1g"\npids_limit = 256\n'
    )
    charter = load_charter(make_charter(tmp_path, content), env={})
    assert charter.raw["loop"]["check_resources"]["memory"] == "2g"


@pytest.mark.parametrize("bad", ["0", "-5", "true", '"60"', "1.5"])
def test_check_timeout_bad_value_rejected(tmp_path: Path, bad: str) -> None:
    # timeout_seconds is a positive int: reject non-positive, bool, string, float.
    content = mutate(
        'cmd = "pytest -x -q"\n',
        f'cmd = "pytest -x -q"\ntimeout_seconds = {bad}\n',
    )
    with pytest.raises(CharterError):
        load_charter(make_charter(tmp_path, content), env={})


@pytest.mark.parametrize("bad", ['""', '"lots"', '"4gb"', "256", "true"])
def test_check_memory_bad_value_rejected(tmp_path: Path, bad: str) -> None:
    # memory is a unit-bearing size string (^\d+[bkmgBKMG]?$): reject empty,
    # non-numeric, multi-letter unit, bare int, bool.
    content = mutate(
        'cmd = "pytest -x -q"\n',
        f'cmd = "pytest -x -q"\nmemory = {bad}\n',
    )
    with pytest.raises(CharterError):
        load_charter(make_charter(tmp_path, content), env={})


@pytest.mark.parametrize("bad", ["0", "-1", "true", '"two"', '""'])
def test_check_cpus_bad_value_rejected(tmp_path: Path, bad: str) -> None:
    # cpus is a positive number (int/float/numeric-string): reject non-positive,
    # bool, non-numeric string, empty.
    content = mutate(
        'cmd = "pytest -x -q"\n',
        f'cmd = "pytest -x -q"\ncpus = {bad}\n',
    )
    with pytest.raises(CharterError):
        load_charter(make_charter(tmp_path, content), env={})


@pytest.mark.parametrize("bad", ["0", "-1", "true", "2.5", '"64"'])
def test_check_pids_limit_bad_value_rejected(tmp_path: Path, bad: str) -> None:
    content = mutate(
        'cmd = "pytest -x -q"\n',
        f'cmd = "pytest -x -q"\npids_limit = {bad}\n',
    )
    with pytest.raises(CharterError):
        load_charter(make_charter(tmp_path, content), env={})


def test_unknown_key_in_loop_check_resources_rejected(tmp_path: Path) -> None:
    content = BASE_CHARTER_TOML + "\n[loop.check_resources]\ntimeout_seconds = 900\nbogus = 1\n"
    with pytest.raises(CharterError):
        load_charter(make_charter(tmp_path, content), env={})


def test_unknown_resource_key_in_checks_entry_rejected(tmp_path: Path) -> None:
    content = mutate(
        'cmd = "pytest -x -q"\n',
        'cmd = "pytest -x -q"\nbogus_knob = 1\n',
    )
    with pytest.raises(CharterError):
        load_charter(make_charter(tmp_path, content), env={})


def test_resolve_defaults_when_no_resource_config(tmp_path: Path) -> None:
    charter = load_charter(make_charter(tmp_path, BASE_CHARTER_TOML), env={})
    # a synthesized check name (not in [checks.*]) with no [loop.check_resources]
    # resolves to the legacy defaults.
    assert resolve_check_resources(charter, "ticket-tests") == DEFAULT_CHECK_RESOURCES


def test_resolve_loop_default_applies_to_unnamed_check(tmp_path: Path) -> None:
    content = BASE_CHARTER_TOML + '\n[loop.check_resources]\ntimeout_seconds = 900\nmemory = "2g"\n'
    charter = load_charter(make_charter(tmp_path, content), env={})
    r = resolve_check_resources(charter, "ticket-tests")  # not a named charter check
    assert r.timeout_seconds == 900
    assert r.memory == "2g"
    assert r.cpus == DEFAULT_CHECK_RESOURCES.cpus  # unset key keeps default


def test_resolve_percheck_overrides_loop_default(tmp_path: Path) -> None:
    content = mutate(
        'cmd = "pytest -x -q"\n',
        'cmd = "pytest -x -q"\ntimeout_seconds = 1800\ncpus = "4"\n',
    )
    content += '\n[loop.check_resources]\ntimeout_seconds = 900\nmemory = "2g"\n'
    charter = load_charter(make_charter(tmp_path, content), env={})
    r = resolve_check_resources(charter, "pytest")
    assert r.timeout_seconds == 1800  # per-check wins over loop default
    assert r.cpus == "4"  # per-check
    assert r.memory == "2g"  # loop default (no per-check override)
    assert r.scratch_size == DEFAULT_CHECK_RESOURCES.scratch_size  # neither -> default
    # a sibling check with no per-check override gets the loop default, not
    # pytest's override.
    r_lint = resolve_check_resources(charter, "lint")
    assert r_lint.timeout_seconds == 900


def test_resolve_normalizes_numeric_types(tmp_path: Path) -> None:
    content = mutate(
        'cmd = "pytest -x -q"\n',
        'cmd = "pytest -x -q"\ncpus = 4\n',  # bare int in TOML
    )
    charter = load_charter(make_charter(tmp_path, content), env={})
    r = resolve_check_resources(charter, "pytest")
    assert r.cpus == "4"
    assert isinstance(r.cpus, str)
    assert isinstance(r.timeout_seconds, int)
    assert isinstance(r, CheckResources)


# --- .79: [provision] table (per-rig worker image pip deps) ---------------
# Bead workspace-e2uh.79. Optional top-level [provision] table; its only
# key is `pip` -- a list of non-empty package-spec strings (may carry
# version pins), consumed by `rig.provision_rig_image` to build the
# per-rig worker image. A charter with no [provision] table at all must
# keep loading exactly as before (base-image fallback, .79 build spec).


def test_provision_table_absent_is_fine(tmp_path: Path) -> None:
    # No [provision] table at all -- must load clean, `provision` absent
    # from the merged config (never synthesized as an empty dict).
    charter = load_charter(make_charter(tmp_path, BASE_CHARTER_TOML), env={})
    assert "provision" not in charter.raw


def test_provision_table_with_pip_list_accepted(tmp_path: Path) -> None:
    content = BASE_CHARTER_TOML + '\n[provision]\npip = ["ruff", "pytest"]\n'
    charter = load_charter(make_charter(tmp_path, content), env={})
    assert charter.raw["provision"]["pip"] == ["ruff", "pytest"]


def test_provision_pip_may_carry_version_pins(tmp_path: Path) -> None:
    content = BASE_CHARTER_TOML + '\n[provision]\npip = ["ruff==0.15.22", "pytest>=8,<9"]\n'
    charter = load_charter(make_charter(tmp_path, content), env={})
    assert charter.raw["provision"]["pip"] == ["ruff==0.15.22", "pytest>=8,<9"]


def test_provision_empty_pip_list_accepted(tmp_path: Path) -> None:
    content = BASE_CHARTER_TOML + "\n[provision]\npip = []\n"
    charter = load_charter(make_charter(tmp_path, content), env={})
    assert charter.raw["provision"]["pip"] == []


def test_provision_table_absent_pip_key_accepted(tmp_path: Path) -> None:
    # [provision] present but with no `pip` key at all -- the table itself
    # is legal (pip is optional within it).
    content = BASE_CHARTER_TOML + "\n[provision]\n"
    charter = load_charter(make_charter(tmp_path, content), env={})
    assert charter.raw["provision"] == {}


def test_provision_not_a_table_rejected(tmp_path: Path) -> None:
    content = BASE_CHARTER_TOML + "\nprovision = 1\n"
    with pytest.raises(CharterError):
        load_charter(make_charter(tmp_path, content), env={})


def test_provision_pip_not_a_list_rejected(tmp_path: Path) -> None:
    content = BASE_CHARTER_TOML + '\n[provision]\npip = "ruff"\n'
    with pytest.raises(CharterError):
        load_charter(make_charter(tmp_path, content), env={})


def test_provision_pip_non_string_element_rejected(tmp_path: Path) -> None:
    content = BASE_CHARTER_TOML + "\n[provision]\npip = [1, 2]\n"
    with pytest.raises(CharterError):
        load_charter(make_charter(tmp_path, content), env={})


def test_provision_pip_empty_string_element_rejected(tmp_path: Path) -> None:
    content = BASE_CHARTER_TOML + '\n[provision]\npip = ["ruff", ""]\n'
    with pytest.raises(CharterError):
        load_charter(make_charter(tmp_path, content), env={})


def test_provision_unknown_key_rejected(tmp_path: Path) -> None:
    content = BASE_CHARTER_TOML + '\n[provision]\npip = ["ruff"]\nbogus_key = 1\n'
    with pytest.raises(CharterError):
        load_charter(make_charter(tmp_path, content), env={})


# --- bead .118: roles.critic.max_tokens (optional, positive int) -----------


def test_critic_max_tokens_accepted(tmp_path: Path) -> None:
    content = mutate(
        '[roles.critic]\nmodel = "opus"',
        '[roles.critic]\nmodel = "opus"\nmax_tokens = 8192',
    )
    charter = load_charter(make_charter(tmp_path, content), env={})
    assert charter.raw["roles"]["critic"]["max_tokens"] == 8192


def test_critic_max_tokens_must_be_positive_int(tmp_path: Path) -> None:
    content = mutate(
        '[roles.critic]\nmodel = "opus"',
        '[roles.critic]\nmodel = "opus"\nmax_tokens = 0',
    )
    with pytest.raises(CharterError):
        load_charter(make_charter(tmp_path, content), env={})


# --- bead .166: roles.critic.station (optional bool; default True) ---------


def test_critic_station_accepted(tmp_path: Path) -> None:
    content = mutate(
        '[roles.critic]\nmodel = "opus"',
        '[roles.critic]\nmodel = "opus"\nstation = false',
    )
    charter = load_charter(make_charter(tmp_path, content), env={})
    assert charter.raw["roles"]["critic"]["station"] is False


def test_critic_station_must_be_bool(tmp_path: Path) -> None:
    content = mutate(
        '[roles.critic]\nmodel = "opus"',
        '[roles.critic]\nmodel = "opus"\nstation = "yes"',
    )
    with pytest.raises(CharterError):
        load_charter(make_charter(tmp_path, content), env={})


# ==========================================================================
# bead .149: lane driver closed set + card-native `effort`
# ==========================================================================


def _append_exec_lane(
    content: str, *, driver: str = "openalph-exec", effort: str | None = None
) -> str:
    """Append a new entry-eligible lane (WITH a selector, so the
    exactly-one-selector-less rule is not perturbed — `default` remains the
    fallthrough). `effort=None` omits the key entirely."""
    lines = [
        "\n[lanes.exec]",
        'selector = { label = "exec-lane" }',
        f'driver = "{driver}"',
        'model = "haiku"',
        'prompt = "code01"',
    ]
    if effort is not None:
        lines.append(f'effort = "{effort}"')
    lines.append('egress = ["inference"]')
    return content + "\n".join(lines) + "\n"


@pytest.mark.parametrize("effort", ["none", "low", "medium", "high", "xhigh"])
def test_lane_effort_card_native_values_accepted(tmp_path: Path, effort: str) -> None:
    # 2026-09-02 SB ruling: `high` is card-native on the Synthetic GLM/Kimi
    # cards (lh3c.11 wire probe: 1:1 pass-through) — the old kdsn.301
    # collapse rejection was a blackwell/qwen38-provider-only finding.
    charter = load_charter(
        make_charter(tmp_path, _append_exec_lane(BASE_CHARTER_TOML, effort=effort)), env={}
    )
    assert charter.raw["lanes"]["exec"]["effort"] == effort
    assert charter.raw["lanes"]["exec"]["driver"] == "openalph-exec"


def test_lane_effort_max_rejected_non_card_native_alias(tmp_path: Path) -> None:
    # spec §2 decision 8: `max` is not card-native (a per-provider alias;
    # xhigh is the canonical ceiling value) — charter-load error, not a rung.
    content = _append_exec_lane(BASE_CHARTER_TOML, effort="max")
    with pytest.raises(CharterError, match="alias"):
        load_charter(make_charter(tmp_path, content), env={})


def test_lane_effort_on_claude_code_lane_rejected(tmp_path: Path) -> None:
    # effort is meaningless on a claude-code lane (no such driver concept
    # there) — fail closed at charter-load.
    content = _append_exec_lane(BASE_CHARTER_TOML, driver="claude-code", effort="medium")
    with pytest.raises(CharterError):
        load_charter(make_charter(tmp_path, content), env={})


def test_lane_unknown_driver_rejected(tmp_path: Path) -> None:
    content = _append_exec_lane(BASE_CHARTER_TOML, driver="gemini-code")
    with pytest.raises(CharterError):
        load_charter(make_charter(tmp_path, content), env={})


def test_lane_driver_openalph_exec_accepted_without_effort(tmp_path: Path) -> None:
    # effort is OPTIONAL on an openalph-exec lane (absent -> None).
    charter = load_charter(make_charter(tmp_path, _append_exec_lane(BASE_CHARTER_TOML)), env={})
    assert "effort" not in charter.raw["lanes"]["exec"]


# --- bead .149: provision.oa_wheelhouse (strict boolean) -------------------


def test_provision_oa_wheelhouse_bool_accepted(tmp_path: Path) -> None:
    content = BASE_CHARTER_TOML + "\n[provision]\npip = [\"ruff\"]\noa_wheelhouse = true\n"
    charter = load_charter(make_charter(tmp_path, content), env={})
    assert charter.raw["provision"]["oa_wheelhouse"] is True


def test_provision_oa_wheelhouse_non_bool_rejected(tmp_path: Path) -> None:
    content = BASE_CHARTER_TOML + '\n[provision]\noa_wheelhouse = "true"\n'
    with pytest.raises(CharterError):
        load_charter(make_charter(tmp_path, content), env={})
