"""Tests for stigmergy.charter (SPEC.md §5 charter schema, §10 AC1).

Loader pipeline under test: parse TOML file -> apply in-code defaults ->
apply `SG_*` env overrides -> validate the MERGED result (validation runs
last, over the merged config, and fails closed).
"""

import shutil
from pathlib import Path

import pytest

from stigmergy.charter import Charter, CharterError, classify_diff, load_charter

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


def test_workers_greater_than_one_rejected(tmp_path: Path) -> None:
    content = mutate("workers = 1", "workers = 2")
    charter_path = make_charter(tmp_path, content)
    with pytest.raises(CharterError):
        load_charter(charter_path, env={})


def test_workers_equal_one_ok(tmp_path: Path) -> None:
    charter_path = make_charter(tmp_path, BASE_CHARTER_TOML)
    charter = load_charter(charter_path, env={})
    assert charter.raw["loop"]["concurrency"]["workers"] == 1


def test_workers_absent_defaults_to_one_ok(tmp_path: Path) -> None:
    content = mutate("[loop.concurrency]\nworkers = 1\n", "")
    charter_path = make_charter(tmp_path, content)
    charter = load_charter(charter_path, env={})
    assert charter.raw["loop"]["concurrency"]["workers"] == 1


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
    charter = load_charter(VALID_CHARTER_PATH, env={})
    dl = charter.raw["loop"]["dispatch_limits"]
    assert dl["filed_tickets"] == 5
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
