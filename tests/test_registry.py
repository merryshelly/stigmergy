"""Tests for stigmergy.registry (SPEC.md §5 [models], §4 credentials/pricing).

Governing invariant under test: `$0` is legal ONLY as an explicitly declared
`local`/`subscription` value — never as a fallback for missing data. A
registry miss, or any missing/invalid required price field, must refuse
(raise UnbudgetableError), never silently resolve to $0.
"""

from pathlib import Path

import pytest

from stigmergy.registry import (
    ModelEntry,
    PricingClass,
    Registry,
    UnbudgetableError,
    load_registry,
)

FIXTURE = Path(__file__).parent / "fixtures" / "models.toml"


@pytest.fixture
def registry() -> Registry:
    return load_registry(FIXTURE)


# --- AC10 / registry miss -------------------------------------------------


def test_resolve_unknown_model_raises_unbudgetable(registry: Registry) -> None:
    """resolve() on an unknown model name raises UnbudgetableError (AC10)."""
    with pytest.raises(UnbudgetableError):
        registry.resolve("does-not-exist")


# --- missing required price field never defaults to $0 -------------------


def test_metered_missing_input_price_raises_not_defaults(tmp_path: Path) -> None:
    """A metered entry missing `input_usd_per_mtok` raises on load — it does
    NOT default to 0. No entry with a silent-$0 price is ever produced."""
    bad = tmp_path / "models.toml"
    bad.write_text(
        """
        [broken]
        provider = "anthropic"
        family = "claude"
        version = "broken-1"
        pricing = "metered"
        output_usd_per_mtok = 4.0
        """
    )
    with pytest.raises(UnbudgetableError):
        load_registry(bad)

    # Assert no path exists where this ever succeeds with a silent $0 entry:
    # the only way to observe an entry is via a successful load_registry call,
    # and that call raised — so no ModelEntry (silently priced at $0 or
    # otherwise) was ever produced for this fixture.


# --- metered $0 is a forbidden fallback disguised as data ----------------


def test_metered_zero_input_price_rejected(tmp_path: Path) -> None:
    """A metered entry declaring `input_usd_per_mtok = 0` is REJECTED at
    load — metered $0 is the forbidden fallback disguised as data."""
    bad = tmp_path / "models.toml"
    bad.write_text(
        """
        [zero-metered]
        provider = "anthropic"
        family = "claude"
        version = "zero-1"
        pricing = "metered"
        input_usd_per_mtok = 0
        output_usd_per_mtok = 4.0
        """
    )
    with pytest.raises(UnbudgetableError):
        load_registry(bad)


def test_metered_nan_price_rejected(tmp_path: Path) -> None:
    """A metered entry declaring a non-finite price (`nan`) is REJECTED —
    NaN is not a valid budgetable price, and TOML natively supports it."""
    bad = tmp_path / "models.toml"
    bad.write_text(
        """
        [nan-metered]
        provider = "anthropic"
        family = "claude"
        version = "nan-1"
        pricing = "metered"
        input_usd_per_mtok = nan
        output_usd_per_mtok = 4.0
        """
    )
    with pytest.raises(UnbudgetableError):
        load_registry(bad)


def test_metered_inf_price_rejected(tmp_path: Path) -> None:
    """A metered entry declaring an infinite price (`inf`) is REJECTED."""
    bad = tmp_path / "models.toml"
    bad.write_text(
        """
        [inf-metered]
        provider = "anthropic"
        family = "claude"
        version = "inf-1"
        pricing = "metered"
        input_usd_per_mtok = inf
        output_usd_per_mtok = 4.0
        """
    )
    with pytest.raises(UnbudgetableError):
        load_registry(bad)


def test_local_nan_marginal_rejected(tmp_path: Path) -> None:
    """A `local` entry with a non-finite `marginal_usd` is REJECTED."""
    bad = tmp_path / "models.toml"
    bad.write_text(
        """
        [nan-local]
        provider = "local"
        family = "qwen"
        version = "qwen-1"
        pricing = "local"
        marginal_usd = nan
        approved = true
        """
    )
    with pytest.raises(UnbudgetableError):
        load_registry(bad)


# --- all three pricing classes parse correctly from the fixture ----------


def test_metered_entry_parses(registry: Registry) -> None:
    entry = registry.resolve("sonnet")
    assert isinstance(entry, ModelEntry)
    assert entry.pricing is PricingClass.METERED
    assert entry.provider == "anthropic"
    assert entry.family == "claude"
    assert entry.version
    assert entry.input_usd_per_mtok is not None and entry.input_usd_per_mtok > 0
    assert entry.output_usd_per_mtok is not None and entry.output_usd_per_mtok > 0


def test_all_metered_fixture_entries_parse(registry: Registry) -> None:
    for model_name in ("haiku", "sonnet", "opus"):
        entry = registry.resolve(model_name)
        assert entry.pricing is PricingClass.METERED
        assert entry.input_usd_per_mtok is not None and entry.input_usd_per_mtok > 0
        assert entry.output_usd_per_mtok is not None and entry.output_usd_per_mtok > 0


def test_local_entry_parses(registry: Registry) -> None:
    entry = registry.resolve("local-qwen")
    assert entry.pricing is PricingClass.LOCAL
    assert entry.approved is True
    assert entry.marginal_usd == 0.0


def test_subscription_entry_parses(registry: Registry) -> None:
    entry = registry.resolve("claude-max-sub")
    assert entry.pricing is PricingClass.SUBSCRIPTION
    assert entry.marginal_usd == 0.0
    assert entry.quota == "5x-plan-weekly-hours"


# --- local pricing without approval is rejected ---------------------------


def test_local_entry_without_approved_true_rejected(tmp_path: Path) -> None:
    """A `local` entry WITHOUT `approved = true` is REJECTED
    (declared-but-unapproved)."""
    bad = tmp_path / "models.toml"
    bad.write_text(
        """
        [unapproved-local]
        provider = "local"
        family = "qwen"
        version = "qwen-1"
        pricing = "local"
        marginal_usd = 0.0
        """
    )
    with pytest.raises(UnbudgetableError):
        load_registry(bad)


def test_local_entry_with_approved_false_rejected(tmp_path: Path) -> None:
    """A `local` entry with `approved = false` is REJECTED too."""
    bad = tmp_path / "models.toml"
    bad.write_text(
        """
        [unapproved-local]
        provider = "local"
        family = "qwen"
        version = "qwen-1"
        pricing = "local"
        marginal_usd = 0.0
        approved = false
        """
    )
    with pytest.raises(UnbudgetableError):
        load_registry(bad)


# --- subscription marginal_usd must be exactly 0.0 ------------------------


def test_subscription_nonzero_marginal_rejected(tmp_path: Path) -> None:
    """A `subscription` entry with `marginal_usd` != 0.0 is REJECTED."""
    bad = tmp_path / "models.toml"
    bad.write_text(
        """
        [paid-sub]
        provider = "anthropic"
        family = "claude"
        version = "sub-1"
        pricing = "subscription"
        marginal_usd = 0.01
        quota = "weekly-hours"
        """
    )
    with pytest.raises(UnbudgetableError):
        load_registry(bad)


# --- unknown pricing class is rejected ------------------------------------


def test_unknown_pricing_class_rejected(tmp_path: Path) -> None:
    """An unknown/invalid `pricing` value is REJECTED."""
    bad = tmp_path / "models.toml"
    bad.write_text(
        """
        [mystery]
        provider = "anthropic"
        family = "claude"
        version = "mystery-1"
        pricing = "freemium"
        marginal_usd = 0.0
        """
    )
    with pytest.raises(UnbudgetableError):
        load_registry(bad)


# --- version_hash stability / canonicalization ----------------------------


def test_version_hash_stable_across_reload(registry: Registry) -> None:
    """Loading the same fixture twice yields the same hash."""
    reloaded = load_registry(FIXTURE)
    assert reloaded.version_hash == registry.version_hash


def test_version_hash_changes_on_price_change(registry: Registry, tmp_path: Path) -> None:
    """Changing any price changes the hash."""
    original = FIXTURE.read_text()
    mutated = original.replace("input_usd_per_mtok = 0.8", "input_usd_per_mtok = 0.9")
    assert mutated != original  # sanity: the replace actually took effect

    mutated_path = tmp_path / "models.toml"
    mutated_path.write_text(mutated)
    mutated_registry = load_registry(mutated_path)

    assert mutated_registry.version_hash != registry.version_hash


def test_version_hash_changes_on_added_entry(registry: Registry, tmp_path: Path) -> None:
    """Adding an entry changes the hash."""
    original = FIXTURE.read_text()
    extra = (
        original
        + """
        [extra-model]
        provider = "anthropic"
        family = "claude"
        version = "extra-1"
        pricing = "metered"
        input_usd_per_mtok = 1.0
        output_usd_per_mtok = 2.0
        """
    )
    extra_path = tmp_path / "models.toml"
    extra_path.write_text(extra)
    extra_registry = load_registry(extra_path)

    assert extra_registry.version_hash != registry.version_hash


def test_version_hash_stable_across_key_reorder(registry: Registry, tmp_path: Path) -> None:
    """Reordering keys in the TOML does NOT change the hash (canonicalize
    before hashing)."""
    reordered = tmp_path / "models.toml"
    reordered.write_text(
        """
        [opus]
        version = "opus-4-1-20250805"
        pricing = "metered"
        provider = "anthropic"
        output_usd_per_mtok = 75.0
        family = "claude"
        input_usd_per_mtok = 15.0
        reasoning_usd_per_mtok = 15.0

        [haiku]
        pricing = "metered"
        version = "haiku-3-5-20241022"
        output_usd_per_mtok = 4.0
        family = "claude"
        provider = "anthropic"
        input_usd_per_mtok = 0.8

        [sonnet]
        cached_usd_per_mtok = 0.3
        output_usd_per_mtok = 15.0
        pricing = "metered"
        family = "claude"
        input_usd_per_mtok = 3.0
        version = "sonnet-4-5-20250514"
        provider = "anthropic"

        [claude-max-sub]
        quota = "5x-plan-weekly-hours"
        marginal_usd = 0.0
        pricing = "subscription"
        version = "claude-code-subscription"
        family = "claude"
        provider = "anthropic"

        [local-qwen]
        approved = true
        pricing = "local"
        marginal_usd = 0.0
        version = "qwen2.5-coder-32b"
        family = "qwen"
        provider = "local"
        """
    )
    reordered_registry = load_registry(reordered)
    assert reordered_registry.version_hash == registry.version_hash
