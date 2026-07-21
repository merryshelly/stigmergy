"""Tests for stigmergy.spend (SPEC.md §9 Budgets, §5 `[loop.budgets]`, §8
spend accounting, §12 measurement, §10 AC10).

Governing invariants under test:

- **Cost is metered-only for the `usd` leash.** `metered` pricing feeds it;
  `local`/`subscription` marginal cost never does (SPEC §9: "the USD leash
  cannot see them") — tracked separately, for the report only.
- **Reserved allowance.** A slice of the `usd` cap (`reserve_usd`) is held
  back so the final weave cycle can always run; normal dispatch/gate
  activity stops once accrued metered spend would leave the reserve
  exposed, BEFORE the reserve itself is touched.
- **Unknown pricing = unbudgetable = refuse to start** (AC10).
- **Recording never self-blocks** — `record_dispatch`/`record_gate` always
  accrue real cost/counts, even past exhaustion; all fail-closed gating
  lives in `can_dispatch`/`can_gate`/`final_weave_allowed`.
- **No auto-renewal, ever.**
- **No USD event is $0-defaulted:** a metered call with real tokens always
  yields a positive cost, never a silent zero.

Fixture rates (tests/fixtures/models.toml), used throughout for exact
arithmetic checks:

- haiku:  input_usd_per_mtok=0.8,  output_usd_per_mtok=4.0
- sonnet: input_usd_per_mtok=3.0,  output_usd_per_mtok=15.0, cached=0.3
- opus:   input_usd_per_mtok=15.0, output_usd_per_mtok=75.0, reasoning=15.0
- local-qwen: pricing=local, marginal_usd=0.0, approved=true
- claude-max-sub: pricing=subscription, marginal_usd=0.0
"""

from __future__ import annotations

from pathlib import Path

import pytest

from stigmergy.registry import Registry, load_registry
from stigmergy.spend import Budgets, SpendError, SpendLeash, cost_usd

FIXTURE = Path(__file__).parent / "fixtures" / "models.toml"


def zero_tokens(**overrides: int) -> dict[str, int]:
    base = {"in": 0, "cached": 0, "out": 0, "reasoning": 0}
    base.update(overrides)
    return base


@pytest.fixture
def registry() -> Registry:
    return load_registry(FIXTURE)


# --- 1. cost_usd arithmetic -------------------------------------------------


def test_cost_usd_metered_matches_fixture_rates(registry: Registry) -> None:
    """metered entry x known tokens -> correct USD, checked against the
    fixture's declared per-mtok rates."""
    haiku = registry.resolve("haiku")
    # 5,000,000 input tokens @ $0.8/mtok = $4.00; 500,000 output @ $4.0/mtok = $2.00
    tokens = zero_tokens(**{"in": 5_000_000, "out": 500_000})
    cost = cost_usd(haiku, tokens)
    assert cost == pytest.approx(4.0 + 2.0)


def test_cost_usd_metered_uses_cached_and_reasoning_rates(registry: Registry) -> None:
    sonnet = registry.resolve("sonnet")
    # 2,000,000 cached @ $0.3/mtok = $0.60
    tokens = zero_tokens(cached=2_000_000)
    assert cost_usd(sonnet, tokens) == pytest.approx(0.6)

    opus = registry.resolve("opus")
    # 1,000,000 reasoning @ $15.0/mtok = $15.00
    tokens = zero_tokens(reasoning=1_000_000)
    assert cost_usd(opus, tokens) == pytest.approx(15.0)


def test_cost_usd_local_returns_declared_marginal(registry: Registry) -> None:
    """local -> its declared marginal (may be 0.0)."""
    local = registry.resolve("local-qwen")
    assert cost_usd(local, zero_tokens(**{"in": 999, "out": 999})) == 0.0


def test_cost_usd_subscription_returns_zero(registry: Registry) -> None:
    """subscription -> 0.0, regardless of tokens."""
    sub = registry.resolve("claude-max-sub")
    assert cost_usd(sub, zero_tokens(**{"in": 999_999, "out": 999_999})) == 0.0


# --- 2. preflight refuses unknown model (AC10) ------------------------------


def test_preflight_refuses_unknown_model(registry: Registry) -> None:
    """A run naming a model absent from the registry refuses to start."""
    budgets = Budgets(dispatches=10, usd=10.0, gate_calls=10, reserve_usd=1.0)
    leash = SpendLeash(budgets, registry)
    with pytest.raises(SpendError):
        leash.preflight(["haiku", "does-not-exist"])


def test_preflight_accepts_all_known_models(registry: Registry) -> None:
    """Sanity: preflight does NOT raise when every named model resolves."""
    budgets = Budgets(dispatches=10, usd=10.0, gate_calls=10, reserve_usd=1.0)
    leash = SpendLeash(budgets, registry)
    leash.preflight(["haiku", "sonnet", "opus", "local-qwen", "claude-max-sub"])


# --- 3. USD leash fail-closed with reserve intact ---------------------------


def test_usd_leash_trips_before_touching_reserve(registry: Registry) -> None:
    """Drive metered spend up to exactly (usd_cap - reserve_usd); assert the
    leash trips (can_dispatch/can_gate go False) with the reserve
    untouched."""
    # usd_cap=10.0, reserve_usd=2.0 -> non-reserved envelope = $8.00.
    budgets = Budgets(dispatches=100, usd=10.0, gate_calls=100, reserve_usd=2.0)
    leash = SpendLeash(budgets, registry)

    # haiku: 5,000,000 in @ $0.8/mtok = $4.00 per dispatch. Two dispatches = $8.00.
    tokens = zero_tokens(**{"in": 5_000_000})
    assert leash.can_dispatch() is True
    leash.record_dispatch("haiku", tokens)
    assert leash.can_dispatch() is True  # $4.00 spent, $8.00 envelope -> still open
    leash.record_dispatch("haiku", tokens)

    # Exactly at the non-reserved envelope now ($8.00 spent, cap-reserve=$8.00).
    assert leash.can_dispatch() is False
    assert leash.can_gate() is False
    assert leash.exhausted() is True

    report = leash.run_report()
    assert report["reserve"]["reserve_intact"] is True
    assert report["reserve"]["reserve_used"] == pytest.approx(0.0)
    assert report["reserve"]["reserve_remaining"] == pytest.approx(2.0)


def test_usd_leash_trips_via_gate_spend_too(registry: Registry) -> None:
    """Gate calls accrue against the same USD leash as dispatches."""
    budgets = Budgets(dispatches=100, usd=10.0, gate_calls=100, reserve_usd=2.0)
    leash = SpendLeash(budgets, registry)

    tokens = zero_tokens(**{"in": 5_000_000})
    leash.record_dispatch("haiku", tokens)  # $4.00
    assert leash.can_gate() is True
    leash.record_gate("haiku", tokens)  # +$4.00 = $8.00 == envelope
    assert leash.can_gate() is False
    assert leash.can_dispatch() is False


# --- 4. final weave spends the reserve, exactly once ------------------------


def test_final_weave_allowed_exactly_once_and_spends_reserve(registry: Registry) -> None:
    budgets = Budgets(dispatches=100, usd=10.0, gate_calls=100, reserve_usd=2.0)
    leash = SpendLeash(budgets, registry)

    tokens = zero_tokens(**{"in": 5_000_000})
    leash.record_dispatch("haiku", tokens)
    leash.record_dispatch("haiku", tokens)  # $8.00, exhausted
    assert leash.exhausted() is True

    assert leash.final_weave_allowed() is True

    # The final weave's gate call spends out of the reserve: $2.00 more,
    # landing exactly at usd_cap ($10.00) -- permitted because record_gate
    # never self-blocks.
    reserve_tokens = zero_tokens(**{"in": 2_500_000})  # $2.00 @ $0.8/mtok
    cost = leash.record_gate("haiku", reserve_tokens)
    assert cost == pytest.approx(2.0)

    report = leash.run_report()
    assert report["metered_spent"] == pytest.approx(10.0)
    assert report["reserve"]["reserve_used"] == pytest.approx(2.0)
    assert report["reserve"]["reserve_remaining"] == pytest.approx(0.0)

    # A second final_weave_allowed() call returns False -- only one is ever granted.
    assert leash.final_weave_allowed() is False


def test_final_weave_allowed_false_before_exhaustion(registry: Registry) -> None:
    budgets = Budgets(dispatches=100, usd=10.0, gate_calls=100, reserve_usd=2.0)
    leash = SpendLeash(budgets, registry)
    assert leash.final_weave_allowed() is False


# --- 5. dispatch-count leash, independent of USD ----------------------------


def test_dispatch_count_leash_trips_independent_of_usd(registry: Registry) -> None:
    """Hitting the `dispatches` cap flips can_dispatch() False even though
    plenty of USD headroom remains."""
    budgets = Budgets(dispatches=2, usd=1_000.0, gate_calls=100, reserve_usd=1.0)
    leash = SpendLeash(budgets, registry)

    tiny_tokens = zero_tokens(**{"in": 1})  # negligible cost
    leash.record_dispatch("haiku", tiny_tokens)
    assert leash.can_dispatch() is True
    leash.record_dispatch("haiku", tiny_tokens)

    assert leash.can_dispatch() is False
    assert leash.exhausted() is True
    # Tons of USD remaining -- this is a count trip, not a USD trip.
    report = leash.run_report()
    assert report["metered_spent"] < 1.0


# --- 6. non-metered invisible to the USD leash, visible to dispatch count --


def test_non_metered_dispatches_do_not_feed_usd_leash(registry: Registry) -> None:
    budgets = Budgets(dispatches=100, usd=5.0, gate_calls=100, reserve_usd=1.0)
    leash = SpendLeash(budgets, registry)

    # Many local + subscription dispatches, arbitrarily large token counts.
    big_tokens = zero_tokens(**{"in": 10_000_000, "out": 10_000_000})
    for _ in range(20):
        leash.record_dispatch("local-qwen", big_tokens)
        leash.record_dispatch("claude-max-sub", big_tokens)

    # USD leash untouched -- metered spend is still zero.
    assert leash.can_dispatch() is True  # 40 dispatches < 100 cap, $0 metered spent
    report = leash.run_report()
    assert report["metered_spent"] == 0.0
    assert report["spend_by_pricing_class"]["metered"] == 0.0

    # But they DO count toward the dispatch-count leash.
    assert report["dispatches_used"] == 40

    # And they DO appear in the report broken out by pricing class.
    assert report["spend_by_pricing_class"]["local"] == pytest.approx(0.0)
    assert report["spend_by_pricing_class"]["subscription"] == pytest.approx(0.0)


def test_non_metered_dispatches_count_toward_dispatch_cap(registry: Registry) -> None:
    """Non-metered dispatches can still exhaust the dispatch-count leash
    (they are NOT free with respect to that leash, only to USD)."""
    budgets = Budgets(dispatches=2, usd=1_000.0, gate_calls=100, reserve_usd=1.0)
    leash = SpendLeash(budgets, registry)

    tokens = zero_tokens(**{"in": 1_000_000})
    leash.record_dispatch("local-qwen", tokens)
    leash.record_dispatch("claude-max-sub", tokens)

    assert leash.can_dispatch() is False
    assert leash.exhausted() is True


# --- 7. no USD event is $0-defaulted ----------------------------------------


def test_metered_dispatch_with_real_tokens_yields_positive_usd(registry: Registry) -> None:
    budgets = Budgets(dispatches=100, usd=1_000.0, gate_calls=100, reserve_usd=1.0)
    leash = SpendLeash(budgets, registry)

    tokens = zero_tokens(**{"in": 1_000_000, "out": 500_000})
    cost = leash.record_dispatch("opus", tokens)
    assert cost > 0.0

    report = leash.run_report()
    assert report["spend_by_pricing_class"]["metered"] > 0.0
    assert report["metered_spent"] > 0.0


def test_cost_usd_unbudgetable_category_raises_via_record(tmp_path: Path) -> None:
    """A metered entry whose declared rates don't cover a token category
    that actually carries tokens is genuinely unbudgetable -- record_*
    must raise, never silently price it at $0."""
    bad_toml = tmp_path / "models.toml"
    bad_toml.write_text(
        """
        [partial-metered]
        provider = "anthropic"
        family = "claude"
        version = "partial-1"
        pricing = "metered"
        input_usd_per_mtok = 1.0
        output_usd_per_mtok = 2.0
        """
    )
    partial_registry = load_registry(bad_toml)
    budgets = Budgets(dispatches=10, usd=100.0, gate_calls=10, reserve_usd=1.0)
    leash = SpendLeash(budgets, partial_registry)

    # No reasoning_usd_per_mtok declared, but tokens carry reasoning tokens.
    tokens = zero_tokens(**{"in": 100, "reasoning": 100})
    with pytest.raises(SpendError):
        leash.record_dispatch("partial-metered", tokens)


# --- 8. no auto-renewal ------------------------------------------------------


def test_no_renew_method_exists_on_spend_leash() -> None:
    """There is no method that re-arms/increases the leash in place. This
    is a whitelist check (not just a blocklist of guessed bad names): the
    full public API surface must equal exactly the documented interface,
    so ANY newly-added mutator method — renewal-flavored or not — fails
    this test until it is deliberately reviewed and added here."""
    expected_public_methods = {
        "preflight",
        "record_dispatch",
        "record_gate",
        "can_dispatch",
        "can_gate",
        "final_weave_allowed",
        "exhausted",
        "run_report",
        "notification_intent",
    }
    public_methods = {
        name
        for name in dir(SpendLeash)
        if not name.startswith("_") and callable(getattr(SpendLeash, name))
    }
    assert public_methods == expected_public_methods

    forbidden_names = {"renew", "reset", "re_arm", "rearm", "top_up", "topup", "extend"}
    assert forbidden_names & public_methods == set()


def test_exhausted_leash_stays_refused_on_reuse_attempt(registry: Registry) -> None:
    """Attempting to keep using an exhausted leash for normal dispatch
    stays refused -- can_dispatch() remains False no matter how many more
    times a caller checks or even records further activity."""
    budgets = Budgets(dispatches=2, usd=1_000.0, gate_calls=100, reserve_usd=1.0)
    leash = SpendLeash(budgets, registry)

    tokens = zero_tokens(**{"in": 1})
    leash.record_dispatch("haiku", tokens)
    leash.record_dispatch("haiku", tokens)
    assert leash.can_dispatch() is False

    # Repeated checks: still False, never silently re-arms.
    assert leash.can_dispatch() is False
    assert leash.can_dispatch() is False


# --- 9. run_report contents --------------------------------------------------


def test_run_report_contents(registry: Registry) -> None:
    budgets = Budgets(dispatches=100, usd=1_000.0, gate_calls=100, reserve_usd=1.0)
    leash = SpendLeash(budgets, registry)

    tokens_a = zero_tokens(**{"in": 1_000_000})
    tokens_b = zero_tokens(**{"in": 2_000_000})
    leash.record_dispatch("haiku", tokens_a, ticket="tkt-1")
    leash.record_dispatch("haiku", tokens_b, ticket="tkt-1")
    leash.record_dispatch("sonnet", tokens_a, ticket="tkt-2")
    leash.record_dispatch("local-qwen", tokens_a, ticket="tkt-3")
    leash.record_gate("opus", tokens_a)

    report = leash.run_report(landed=2)

    assert report["dispatches_used"] == 4
    assert report["landed"] == 2
    assert report["gate_calls_used"] == 1

    # spend broken out by pricing class
    assert report["spend_by_pricing_class"]["metered"] > 0.0
    assert "local" in report["spend_by_pricing_class"]
    assert "subscription" in report["spend_by_pricing_class"]

    # per-ticket dispatch distribution
    assert report["per_ticket_dispatches"] == {"tkt-1": 2, "tkt-2": 1, "tkt-3": 1}
    # distribution reconciles exactly with the total dispatch count -- no
    # dispatch silently vanishes from the per-ticket breakdown.
    assert sum(report["per_ticket_dispatches"].values()) == report["dispatches_used"]

    # reserve state present
    assert "reserve" in report
    assert report["reserve"]["reserve_usd"] == pytest.approx(1.0)

    # escalations count present (no exhaustion occurred here -> 0)
    assert report["escalations"] == 0
    assert report["exhausted"] is False


def test_untracked_dispatches_are_visible_not_silently_dropped(registry: Registry) -> None:
    """A dispatch recorded without `ticket=` still counts toward
    `dispatches_used` and must not silently vanish from the per-ticket
    distribution -- it lands in an explicit sentinel bucket instead."""
    budgets = Budgets(dispatches=100, usd=1_000.0, gate_calls=100, reserve_usd=1.0)
    leash = SpendLeash(budgets, registry)

    tokens = zero_tokens(**{"in": 1_000_000})
    leash.record_dispatch("haiku", tokens, ticket="tkt-1")
    leash.record_dispatch("haiku", tokens)  # no ticket= supplied
    leash.record_dispatch("haiku", tokens)  # no ticket= supplied

    report = leash.run_report()
    assert report["dispatches_used"] == 3
    assert sum(report["per_ticket_dispatches"].values()) == 3
    # the two untracked dispatches are visible under an explicit bucket,
    # not silently missing.
    untracked = {k: v for k, v in report["per_ticket_dispatches"].items() if k != "tkt-1"}
    assert sum(untracked.values()) == 2
    assert report["per_ticket_dispatches"]["tkt-1"] == 1


def test_run_report_escalations_reflect_exhaustion(registry: Registry) -> None:
    budgets = Budgets(dispatches=1, usd=1_000.0, gate_calls=100, reserve_usd=1.0)
    leash = SpendLeash(budgets, registry)

    tokens = zero_tokens(**{"in": 1})
    leash.record_dispatch("haiku", tokens, ticket="tkt-1")

    report = leash.run_report()
    assert report["exhausted"] is True
    assert report["escalations"] >= 1


# --- notification_intent -----------------------------------------------------


def test_notification_intent_is_persisted_data_structure(registry: Registry) -> None:
    budgets = Budgets(dispatches=1, usd=1_000.0, gate_calls=100, reserve_usd=1.0)
    leash = SpendLeash(budgets, registry)

    tokens = zero_tokens(**{"in": 1})
    leash.record_dispatch("haiku", tokens)

    intent = leash.notification_intent()
    assert isinstance(intent, dict)
    assert intent["exhausted"] is True
    assert "dispatches" in intent["reasons"]


# --- 10. rehydration from prior events -----------------------------------


def test_rehydrate_with_dispatch_and_gate_events(registry: Registry) -> None:
    """Constructing a leash with a list of prior dispatch/gate events seeds
    the initial state before any record_* call is made."""
    budgets = Budgets(dispatches=100, usd=1_000.0, gate_calls=100, reserve_usd=1.0)

    # Build prior events: 2 dispatch events with $4.00 each, 1 gate with $3.00
    prior_events = [
        {
            "event_type": "dispatch",
            "computed_usd": 4.0,
        },
        {
            "event_type": "dispatch",
            "computed_usd": 4.0,
        },
        {
            "event_type": "gate",
            "computed_usd": 3.0,
        },
    ]

    leash = SpendLeash(budgets, registry, events=prior_events)

    # Report immediately: should reflect seeded state WITHOUT any record_* calls
    report = leash.run_report()
    assert report["dispatches_used"] == 2
    assert report["gate_calls_used"] == 1
    assert report["metered_spent"] == pytest.approx(11.0)  # 4 + 4 + 3


def test_rehydrate_with_unbudgetable_events(registry: Registry) -> None:
    """An event with computed_usd='unbudgetable' counts toward dispatch/gate
    counts but contributes nothing to the USD total."""
    budgets = Budgets(dispatches=100, usd=1_000.0, gate_calls=100, reserve_usd=1.0)

    prior_events = [
        {
            "event_type": "dispatch",
            "computed_usd": "unbudgetable",
        },
        {
            "event_type": "dispatch",
            "computed_usd": 2.5,
        },
        {
            "event_type": "gate",
            "computed_usd": "unbudgetable",
        },
    ]

    leash = SpendLeash(budgets, registry, events=prior_events)

    report = leash.run_report()
    assert report["dispatches_used"] == 2
    assert report["gate_calls_used"] == 1
    assert report["metered_spent"] == pytest.approx(2.5)  # only the 2.5


def test_rehydrate_omitted_produces_all_zeros(registry: Registry) -> None:
    """Omitting the events parameter (or passing None) produces all-zero
    starting state, unchanged from the original behavior."""
    budgets = Budgets(dispatches=100, usd=1_000.0, gate_calls=100, reserve_usd=1.0)

    leash1 = SpendLeash(budgets, registry)
    leash2 = SpendLeash(budgets, registry, events=None)
    leash3 = SpendLeash(budgets, registry, events=[])

    for leash in [leash1, leash2, leash3]:
        report = leash.run_report()
        assert report["dispatches_used"] == 0
        assert report["gate_calls_used"] == 0
        assert report["metered_spent"] == 0.0


def test_rehydrate_seeded_past_dispatch_cap_exhausts_immediately(registry: Registry) -> None:
    """A leash seeded at or past a budget cap correctly reports exhaustion
    and denies can_dispatch()/can_gate() without any further record_* calls."""
    budgets = Budgets(dispatches=2, usd=1_000.0, gate_calls=2, reserve_usd=1.0)

    prior_events = [
        {"event_type": "dispatch", "computed_usd": 1.0},
        {"event_type": "dispatch", "computed_usd": 1.0},
    ]

    leash = SpendLeash(budgets, registry, events=prior_events)

    # Immediately exhausted on dispatch count
    assert leash.exhausted() is True
    assert leash.can_dispatch() is False
    assert leash.can_gate() is True  # gate count is not exhausted


def test_rehydrate_seeded_past_gate_cap_exhausts_immediately(registry: Registry) -> None:
    """A leash seeded with gate calls past the cap correctly reports
    exhaustion."""
    budgets = Budgets(dispatches=100, usd=1_000.0, gate_calls=1, reserve_usd=1.0)

    prior_events = [
        {"event_type": "gate", "computed_usd": 1.0},
    ]

    leash = SpendLeash(budgets, registry, events=prior_events)

    assert leash.exhausted() is True
    assert leash.can_gate() is False
    assert leash.can_dispatch() is True  # dispatch count is not exhausted


def test_rehydrate_seeded_past_usd_cap_exhausts_immediately(registry: Registry) -> None:
    """A leash seeded with USD spend past the non-reserved cap correctly
    reports exhaustion."""
    budgets = Budgets(dispatches=100, usd=10.0, gate_calls=100, reserve_usd=2.0)

    prior_events = [
        {"event_type": "dispatch", "computed_usd": 8.0},
        {"event_type": "dispatch", "computed_usd": 1.0},
    ]

    leash = SpendLeash(budgets, registry, events=prior_events)

    # $9.00 spent, non-reserved cap = $8.00 -> exhausted
    assert leash.exhausted() is True
    assert leash.can_dispatch() is False
    assert leash.can_gate() is False


def test_rehydrate_ignores_non_dispatch_gate_event_types(registry: Registry) -> None:
    """Events with event_type other than 'dispatch' or 'gate' (e.g. 'check',
    'integration', 'disposition', 'notify', 'report', etc.) contribute their
    USD cost but do NOT increment dispatch or gate counts."""
    budgets = Budgets(dispatches=100, usd=1_000.0, gate_calls=100, reserve_usd=1.0)

    prior_events = [
        {"event_type": "dispatch", "computed_usd": 1.0},
        {"event_type": "check", "computed_usd": 2.0},
        {"event_type": "integration", "computed_usd": 3.0},
        {"event_type": "notification", "computed_usd": 4.0},
        {"event_type": "gate", "computed_usd": 5.0},
    ]

    leash = SpendLeash(budgets, registry, events=prior_events)

    report = leash.run_report()
    assert report["dispatches_used"] == 1  # only dispatch event
    assert report["gate_calls_used"] == 1  # only gate event
    assert report["metered_spent"] == pytest.approx(15.0)  # all events


def test_rehydrate_bool_not_treated_as_numeric(registry: Registry) -> None:
    """A computed_usd that is a bool (since bool is an int subclass in Python)
    is NOT treated as numeric and contributes nothing to the USD total."""
    budgets = Budgets(dispatches=100, usd=1_000.0, gate_calls=100, reserve_usd=1.0)

    prior_events = [
        {"event_type": "dispatch", "computed_usd": True},
        {"event_type": "dispatch", "computed_usd": False},
        {"event_type": "dispatch", "computed_usd": 5.0},
    ]

    leash = SpendLeash(budgets, registry, events=prior_events)

    report = leash.run_report()
    assert report["dispatches_used"] == 3
    assert report["metered_spent"] == pytest.approx(5.0)  # only the 5.0, not the bools
