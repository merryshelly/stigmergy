"""Tests for stigmergy.steering (bead `.35` build spec — frozen case list,
12 cases; SPEC.md §4 "Approval integrity").

Case numbering below matches the bead `.35` build spec's frozen case list
exactly (1-12), including the `current_rung`-invariance test (case 8) and
the round-trip integration test against real `approval.approve()`/
`approval.is_approval_valid()` (case 11) — the actual point of this bead.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from stigmergy.approval import approve, is_approval_valid, steering_hash
from stigmergy.charter import Charter, load_charter
from stigmergy.dispatch import DispatchError
from stigmergy.rig import RigStore
from stigmergy.steering import SteeringError, derive_steering

FIXTURES = Path(__file__).parent / "fixtures"
VALID_CHARTER_PATH = FIXTURES / "charter_valid.toml"
MODELS_REGISTRY_PATH = FIXTURES / "models.toml"
BASE_CHARTER_TOML = VALID_CHARTER_PATH.read_text()

# `tests/fixtures/charter_valid.toml` declares three lanes, all sharing the
# same prompt id "code01" (bead .35 build spec case 1 note: "check what
# lanes/prompts/selectors it actually declares"):
#   - lanes.cheap      selector.label = "local-ok", entry-eligible
#   - lanes.default     no selector (the one required fallthrough lane)
#   - lanes.exquisite   entry = false (step-up only rung)
PROMPT_ID = "code01"
PROMPT_CONTENT = "PROMPT TEMPLATE v1 -- do the thing.\n"


# --------------------------------------------------------------------------
# shared fixtures / helpers
# --------------------------------------------------------------------------


def make_full_charter(tmp_path: Path) -> Charter:
    """A fully valid, loaded Charter (via the real `charter_valid.toml`
    fixture + its models registry) — mirrors `test_dispatch.py`'s
    `make_full_charter` helper exactly (style reference, not a schema)."""
    charter_dir = tmp_path / "charter_src"
    charter_dir.mkdir(exist_ok=True)
    charter_path = charter_dir / "charter.toml"
    charter_path.write_text(BASE_CHARTER_TOML)
    shutil.copy(MODELS_REGISTRY_PATH, charter_dir / "models.toml")
    return load_charter(charter_path, env={})


def make_prompts_dir(
    tmp_path: Path, *, prompt_id: str = PROMPT_ID, content: str = PROMPT_CONTENT
) -> Path:
    """A prompts_dir fixture directory holding one real prompt file."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(exist_ok=True)
    (prompts_dir / prompt_id).write_text(content, encoding="utf-8")
    return prompts_dir


def base_ticket_row(**overrides: Any) -> dict[str, Any]:
    """A complete ticket_row dict (every column `derive_steering` reads
    populated with a realistic value). `title` is always present (schema
    NOT NULL)."""
    row: dict[str, Any] = {
        "title": "Implement foo() to spec",
        "goal": "Make foo() return 42 for all valid inputs.",
        "required_reading": ["context:architecture.md", "repo:src/foo.py"],
        "target_scope": ["src/foo.py"],
        "acceptance_criteria": ["foo() returns 42", "foo() raises on negative input"],
        "tier1_checks": {"pytest": "pytest -q", "lint": "ruff check ."},
        "lane_hint": None,
        "current_rung": None,
    }
    row.update(overrides)
    return row


def _no_fallback_charter() -> Charter:
    """A deliberately malformed test charter (bead .35 build spec case 10):
    the one lane present carries a selector, so there is no selector-less
    fallthrough lane at all — `select_lane` must raise `DispatchError` when
    the ticket's `lane_hint` doesn't match it. Mirrors `test_dispatch.py`'s
    `_lanes_charter` construction style (a raw dict Charter, bypassing full
    charter validation — `select_lane` reads nothing else)."""
    lanes = {
        "specific-only": {
            "driver": "claude-code",
            "model": "haiku",
            "prompt": PROMPT_ID,
            "egress": ["inference"],
            "selector": {"label": "specific-label"},
        },
    }
    return Charter(raw={"lanes": lanes}, resolved_hash="test-hash", warnings=())


@pytest.fixture
def store(tmp_path: Path) -> RigStore:
    s = RigStore.create(tmp_path / "tickets.db")
    yield s
    s.close()


# ==========================================================================
# Case 1: full valid ticket row, lane_hint=None -> charter fallback lane.
# ==========================================================================


def test_case1_full_valid_ticket_row_all_seven_fields(tmp_path: Path) -> None:
    charter = make_full_charter(tmp_path)
    prompts_dir = make_prompts_dir(tmp_path)
    ticket_row = base_ticket_row(lane_hint=None)

    result = derive_steering(ticket_row, charter, prompts_dir)

    assert set(result) == {
        "ticket_text",
        "checks",
        "rubric",
        "target_scope",
        "lane",
        "prompt_bytes",
        "context_set",
    }
    assert result["ticket_text"] == (
        "Implement foo() to spec\nMake foo() return 42 for all valid inputs."
    )
    assert result["checks"] == {"pytest": "pytest -q", "lint": "ruff check ."}
    assert result["rubric"] == ["foo() returns 42", "foo() raises on negative input"]
    assert result["target_scope"] == ["src/foo.py"]
    # lane_hint=None -> falls through to the charter's one selector-less
    # fallthrough lane, "default".
    assert result["lane"] == "default"
    assert result["prompt_bytes"] == PROMPT_CONTENT
    assert result["context_set"] == ["context:architecture.md", "repo:src/foo.py"]


# ==========================================================================
# Case 2: lane_hint matches a specific lane's selector label.
# ==========================================================================


def test_case2_lane_hint_matches_specific_lane(tmp_path: Path) -> None:
    charter = make_full_charter(tmp_path)
    prompts_dir = make_prompts_dir(tmp_path)
    # charter_valid.toml's lanes.cheap selector.label == "local-ok".
    ticket_row = base_ticket_row(lane_hint="local-ok")

    result = derive_steering(ticket_row, charter, prompts_dir)

    assert result["lane"] == "cheap"
    # lanes.cheap.prompt == "code01", same prompt id as the fallback lane
    # in this fixture charter -- still the resolved lane's own prompt file,
    # not a hardcoded/fallback one.
    assert result["prompt_bytes"] == PROMPT_CONTENT


# ==========================================================================
# Case 3: goal=None -> ticket_text ends with "<title>\n", does not raise.
# ==========================================================================


def test_case3_goal_none_ticket_text_empty_goal(tmp_path: Path) -> None:
    charter = make_full_charter(tmp_path)
    prompts_dir = make_prompts_dir(tmp_path)
    ticket_row = base_ticket_row(goal=None)

    result = derive_steering(ticket_row, charter, prompts_dir)

    assert result["ticket_text"] == "Implement foo() to spec\n"


# ==========================================================================
# Case 4: tier1_checks=None -> checks == [].
# ==========================================================================


def test_case4_tier1_checks_none_checks_empty_list(tmp_path: Path) -> None:
    charter = make_full_charter(tmp_path)
    prompts_dir = make_prompts_dir(tmp_path)
    ticket_row = base_ticket_row(tier1_checks=None)

    result = derive_steering(ticket_row, charter, prompts_dir)

    assert result["checks"] == []


# ==========================================================================
# Case 5: acceptance_criteria=None -> rubric == [].
# ==========================================================================


def test_case5_acceptance_criteria_none_rubric_empty_list(tmp_path: Path) -> None:
    charter = make_full_charter(tmp_path)
    prompts_dir = make_prompts_dir(tmp_path)
    ticket_row = base_ticket_row(acceptance_criteria=None)

    result = derive_steering(ticket_row, charter, prompts_dir)

    assert result["rubric"] == []


# ==========================================================================
# Case 6: target_scope=None -> target_scope == [].
# ==========================================================================


def test_case6_target_scope_none_empty_list(tmp_path: Path) -> None:
    charter = make_full_charter(tmp_path)
    prompts_dir = make_prompts_dir(tmp_path)
    ticket_row = base_ticket_row(target_scope=None)

    result = derive_steering(ticket_row, charter, prompts_dir)

    assert result["target_scope"] == []


# ==========================================================================
# Case 7: required_reading=None -> context_set == [].
# ==========================================================================


def test_case7_required_reading_none_context_set_empty_list(tmp_path: Path) -> None:
    charter = make_full_charter(tmp_path)
    prompts_dir = make_prompts_dir(tmp_path)
    ticket_row = base_ticket_row(required_reading=None)

    result = derive_steering(ticket_row, charter, prompts_dir)

    assert result["context_set"] == []


# ==========================================================================
# Case 8: THE FROZEN current_rung-invariance test.
# ==========================================================================


def test_case8_current_rung_never_affects_derived_steering(tmp_path: Path) -> None:
    """Setting ticket_row["current_rung"] to any value (None, an entry
    lane name, or an entry=false step-up-only lane name) must NEVER change
    derive_steering's output, given everything else held constant -- and
    therefore never change approval.steering_hash of that output either.

    charter_valid.toml's lane names: "cheap"/"default" (entry-eligible),
    "exquisite" (entry=false, step-up-only rung) -- confirmed by reading
    the fixture (see module-level comment above)."""
    charter = make_full_charter(tmp_path)
    prompts_dir = make_prompts_dir(tmp_path)

    rung_values = [None, "cheap", "exquisite"]
    results = []
    for rung in rung_values:
        ticket_row = base_ticket_row(lane_hint=None, current_rung=rung)
        results.append(derive_steering(ticket_row, charter, prompts_dir))

    first = results[0]
    for other in results[1:]:
        assert other == first
        assert steering_hash(other) == steering_hash(first)

    # Sanity: this did resolve to the fallback lane, not accidentally to
    # "exquisite" or "cheap" via current_rung -- confirms the invariant is
    # actually being exercised, not vacuously true because lane resolution
    # never ran.
    assert first["lane"] == "default"


# ==========================================================================
# Case 9: missing prompt file -> SteeringError (fail closed).
# ==========================================================================


def test_case9_missing_prompt_file_raises_steering_error(tmp_path: Path) -> None:
    charter = make_full_charter(tmp_path)
    # An empty prompts_dir -- no "code01" file present for the resolved
    # fallback lane ("default").
    prompts_dir = tmp_path / "empty_prompts"
    prompts_dir.mkdir()
    ticket_row = base_ticket_row(lane_hint=None)

    with pytest.raises(SteeringError):
        derive_steering(ticket_row, charter, prompts_dir)


# ==========================================================================
# Case 10: select_lane raising DispatchError propagates uncaught.
# ==========================================================================


def test_case10_dispatch_error_propagates_uncaught(tmp_path: Path) -> None:
    charter = _no_fallback_charter()
    prompts_dir = make_prompts_dir(tmp_path)
    # lane_hint matches nothing, and the charter has no selector-less
    # fallthrough lane at all -- select_lane must raise DispatchError.
    ticket_row = base_ticket_row(lane_hint="no-such-label")

    with pytest.raises(DispatchError):
        derive_steering(ticket_row, charter, prompts_dir)


# ==========================================================================
# Case 11: THE ROUND-TRIP INTEGRATION TEST -- the actual point of this bead.
# ==========================================================================


def test_case11_round_trip_against_real_approval(tmp_path: Path, store: RigStore) -> None:
    charter = make_full_charter(tmp_path)
    prompts_dir = make_prompts_dir(tmp_path)

    ticket_id = "t-roundtrip"
    store.add_ticket(
        id=ticket_id,
        title="Implement foo() to spec",
        goal="Make foo() return 42 for all valid inputs.",
        required_reading=["context:architecture.md", "repo:src/foo.py"],
        target_scope=["src/foo.py"],
        acceptance_criteria=["foo() returns 42", "foo() raises on negative input"],
        tier1_checks={"pytest": "pytest -q", "lint": "ruff check ."},
        lane_hint=None,
    )
    ticket_row = store.get_ticket(ticket_id)
    assert ticket_row is not None

    # 1. Derive once -- this is what a human's approval act signs.
    steering_at_approval = derive_steering(ticket_row, charter, prompts_dir)
    approve(store, ticket_id, steering=steering_at_approval)

    # 2. Derive AGAIN on the same, unmodified ticket_row -- the live
    # eligibility-check-time comparison. Must validate.
    steering_now = derive_steering(ticket_row, charter, prompts_dir)
    assert is_approval_valid(store.get_ticket(ticket_id), steering_now) is True

    # 3. Mutate ONE steering-relevant field on the ticket_row (not the
    # store) and derive a THIRD time -- must now invalidate.
    ticket_row["acceptance_criteria"] = ["a completely different rubric item"]
    steering_after_mutation = derive_steering(ticket_row, charter, prompts_dir)
    assert is_approval_valid(store.get_ticket(ticket_id), steering_after_mutation) is False


# ==========================================================================
# Case 12: pure-function determinism -- two independent calls, same
# inputs, byte-for-byte equal output (no hidden nondeterminism).
# ==========================================================================


def test_case12_pure_function_two_calls_byte_for_byte_equal(tmp_path: Path) -> None:
    charter = make_full_charter(tmp_path)
    prompts_dir = make_prompts_dir(tmp_path)
    ticket_row = base_ticket_row(lane_hint="local-ok")

    result_a = derive_steering(ticket_row, charter, prompts_dir)
    result_b = derive_steering(ticket_row, charter, prompts_dir)

    assert result_a == result_b
    assert steering_hash(result_a) == steering_hash(result_b)
    # Everything returned must be a stably-orderable JSON type -- lists,
    # not sets -- so this genuinely is byte-for-byte identical, not merely
    # equal-by-set-comparison.
    import json

    blob_a = json.dumps(result_a, sort_keys=True, separators=(",", ":"), default=str)
    blob_b = json.dumps(result_b, sort_keys=True, separators=(",", ":"), default=str)
    assert blob_a == blob_b
