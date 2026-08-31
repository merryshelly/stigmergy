"""Tests for stigmergy.manifest (the decomposer station's deterministic
manifest validator) + the `stigmergy manifest validate` CLI verb.

The validator checks a JSON manifest against the decomposer contract
(the SAME key vocabulary as the triage promotion spec —
``triage._REQUIRED_PROMOTION_KEYS`` / ``_OPTIONAL_PROMOTION_KEYS`` —
deliberately STRicter than `intake`, which does not require
``tier1_checks``): rules R1-R12 + the unknown-key typo-catcher (R13).

Defect strings are stable and structured: ``"ticket <id-or-index>:
<rule-id>: <detail>"`` (manifest-level structural problems use
``"manifest:"`` as the ticket prefix). A clean manifest returns ``[]``.

CLI convention notes (mirrors `cmd_intake`):
- structural failures (missing file, unparseable JSON, manifest not an
  array) -> stderr with the ``stigmergy manifest:`` prefix + exit 1;
- validation defects -> one greppable defect line per defect on STDOUT
  + a final ``N defect(s) across M ticket(s)`` summary;
- exit 0 iff zero defects, else 1.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from stigmergy.cli import main
from stigmergy.manifest import validate_manifest
from stigmergy.rig import RigStore, create_rig

# The real one-ticket dogfood fixture (bead .149). Its `difficulty: "low"`
# is OFF the decomposer vocabulary (trivial|easy|medium|hard|frontier) —
# that is a DESIGNED test case, not a fixture bug.
FIXTURE_PATH = Path(
    "/home/oa-merry/workspace/tmp/bead149-dogfood/ticket-dead-migration-helpers.json"
)

# The repo's own minimal valid charter fixture (checks {pytest, lint},
# gates attempt/staging = ["lint", "pytest"], lanes {cheap, default,
# exquisite}) — loaded via the REAL load_charter so the adapter sees the
# same object shape production does.
CHARTER_FIXTURE = Path(__file__).parent / "fixtures" / "charter_valid.toml"

# `target_scope` path(s) that exist in the stigmergy repo itself, so the
# repo-existence carve-out can be exercised without a tmp repo.
EXISTING_RELPATH = "src/stigmergy/rig.py"

# tier1_checks that exactly cover the fixture charter's gates with the
# charter's verbatim cmds (see fixtures/charter_valid.toml) — needed for
# any clean-manifest test that passes the charter.
_CHARTER_CLEAN_CHECKS = {"lint": "ruff check .", "pytest": "pytest -x -q"}


# ==========================================================================
# helpers / fixtures
# ==========================================================================


def base_ticket(**overrides: Any) -> dict[str, Any]:
    """A fully valid single-ticket entry (no charter/store context): all
    required keys present, clean difficulty, clean id, one clean scope
    path. Most per-rule tests start from this and mutate one field."""
    entry: dict[str, Any] = {
        "id": "alpha-one",
        "title": "Do the thing",
        "functional_summary": "A non-empty operator-facing summary.",
        "acceptance_criteria": ["the thing is done"],
        "tier1_checks": {"ruff": "ruff check ."},
        "target_scope": ["src/pkg/mod.py"],
        "difficulty": "medium",
    }
    entry.update(overrides)
    return entry


def clean_manifest(repo: Path | None = None, **entry_overrides: Any) -> list[dict[str, Any]]:
    """One valid ticket whose scope path actually exists in ``repo``
    (``None`` -> no repo check, path validity not evaluated)."""
    scope = [EXISTING_RELPATH] if repo is not None else ["src/pkg/mod.py"]
    return [base_ticket(target_scope=scope, **entry_overrides)]


def parse_rule(line: str) -> tuple[str, str, str]:
    """Split a defect line back into (ticket, rule, detail)."""
    match = re.match(r"^ticket (.*): ([A-Z0-9]+): (.*)$", line)
    assert match is not None, f"unparseable defect line: {line!r}"
    return match.group(1), match.group(2), match.group(3)


def load_charter_fixture() -> Any:
    from stigmergy.charter import load_charter

    return load_charter(CHARTER_FIXTURE, env={})


@pytest.fixture
def store(tmp_path: Path) -> RigStore:
    s = RigStore.create(tmp_path / "tickets.db")
    yield s
    s.close()


@pytest.fixture
def charter() -> Any:
    return load_charter_fixture()


# ==========================================================================
# happy path (the real fixture file)
# ==========================================================================


def test_happy_path_fixture_as_is_flags_exactly_one_r10() -> None:
    """The real .149 dogfood ticket (a one-ticket manifest, wrapped in the
    required JSON array), unmodified: exactly ONE defect — its
    off-vocabulary `difficulty: "low"` (the designed test case)."""
    ticket = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert isinstance(ticket, dict)  # the fixture is a bare one-ticket object
    defects = validate_manifest([ticket])
    assert len(defects) == 1
    _, rule, detail = parse_rule(defects[0])
    assert "dead-migration-helpers-149" in defects[0]
    assert rule == "R10"
    assert "low" in detail


def test_happy_path_fixture_cleaned_difficulty_is_clean() -> None:
    """Same fixture with `difficulty` corrected to the vocabulary -> 0 defects."""
    ticket = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    ticket["difficulty"] = "medium"
    assert validate_manifest([ticket]) == []


# ==========================================================================
# R1: manifest is a JSON array; every entry is an object
# ==========================================================================


def test_r1_manifest_not_a_list() -> None:
    defects = validate_manifest({"id": "not-a-list"})
    assert len(defects) == 1
    assert defects[0].startswith("manifest: R1: ")
    assert "JSON array" in defects[0]


def test_r1_entry_not_an_object() -> None:
    defects = validate_manifest([base_ticket(), "just a string"])
    assert any(
        parse_rule(d) == ("1", "R1", "entry must be a JSON object (got str)") for d in defects
    )


# ==========================================================================
# R2: required keys
# ==========================================================================


def test_r2_missing_required_key() -> None:
    manifest = [base_ticket()]
    del manifest[0]["tier1_checks"]  # the key intake does NOT require
    defects = validate_manifest(manifest)
    assert any(
        parse_rule(d) == ("alpha-one", "R2", "missing required key: tier1_checks") for d in defects
    )


def test_r2_missing_id_reported_by_index() -> None:
    entry = base_ticket()
    del entry["id"]
    defects = validate_manifest([entry])
    assert any(parse_rule(d) == ("0", "R2", "missing required key: id") for d in defects)


# ==========================================================================
# R3: functional_summary
# ==========================================================================


def test_r3_functional_summary_blank() -> None:
    defects = validate_manifest([base_ticket(functional_summary="   ")])
    assert any(
        parse_rule(d)
        == ("alpha-one", "R3", "functional_summary must be a non-empty string")
        for d in defects
    )


def test_r3_functional_summary_wrong_type() -> None:
    defects = validate_manifest([base_ticket(functional_summary=42)])
    assert any(
        parse_rule(d)
        == ("alpha-one", "R3", "functional_summary must be a non-empty string")
        for d in defects
    )


# ==========================================================================
# R4: acceptance_criteria (load-bearing: a non-list silently degrades to
# an empty critic rubric downstream)
# ==========================================================================


def test_r4_acceptance_criteria_not_a_list() -> None:
    defects = validate_manifest([base_ticket(acceptance_criteria="the thing is done")])
    assert any(
        parse_rule(d)
        == ("alpha-one", "R4", "acceptance_criteria must be an array of strings")
        for d in defects
    )


def test_r4_acceptance_criteria_empty_list() -> None:
    defects = validate_manifest([base_ticket(acceptance_criteria=[])])
    assert any(
        parse_rule(d)
        == ("alpha-one", "R4", "acceptance_criteria must have at least one entry")
        for d in defects
    )


def test_r4_acceptance_criteria_blank_entry() -> None:
    defects = validate_manifest([base_ticket(acceptance_criteria=["real", "   "])])
    assert any(
        parse_rule(d)
        == ("alpha-one", "R4", "acceptance_criteria entry 1 must be a non-empty string")
        for d in defects
    )


def test_r4_acceptance_criteria_non_string_entry() -> None:
    defects = validate_manifest([base_ticket(acceptance_criteria=[7])])
    assert any(
        parse_rule(d)
        == ("alpha-one", "R4", "acceptance_criteria entry 0 must be a non-empty string")
        for d in defects
    )


# ==========================================================================
# R5: tier1_checks (NEVER a list — lists misparse as pytest paths
# downstream; keys must name [checks.*]; gates must be covered; cmds must
# match verbatim)
# ==========================================================================


def test_r5_tier1_checks_is_a_list() -> None:
    defects = validate_manifest([base_ticket(tier1_checks=["tests/test_foo.py"])])
    assert any(
        parse_rule(d) == ("alpha-one", "R5", "tier1_checks must be a dict, not a list")
        for d in defects
    )


def test_r5_tier1_checks_empty_key() -> None:
    defects = validate_manifest(
        [base_ticket(tier1_checks={"": "ruff check .", "ruff": "ruff check ."})]
    )
    assert any(
        parse_rule(d)
        == ("alpha-one", "R5", "tier1_checks key must be a non-empty string (got '')")
        for d in defects
    )


def test_r5_tier1_checks_non_string_key() -> None:
    # JSON object keys are always strings; simulate the shape directly.
    entry = base_ticket(tier1_checks={9: "ruff check ."})
    defects = validate_manifest([entry])
    assert any(
        parse_rule(d)
        == ("alpha-one", "R5", "tier1_checks key must be a non-empty string (got 9)")
        for d in defects
    )


def test_r5_charter_unknown_check_name(charter: Any) -> None:
    defects = validate_manifest([base_ticket(tier1_checks={"mystery": "true"})], charter=charter)
    assert any(
        parse_rule(d)
        == ("alpha-one", "R5", "tier1_checks key 'mystery' is not a [checks.*] section")
        for d in defects
    )


def test_r5_charter_gate_check_missing(charter: Any) -> None:
    # charter gates: attempt=staging=["lint", "pytest"] — drop "pytest".
    defects = validate_manifest(
        [base_ticket(tier1_checks={"lint": "ruff check ."})], charter=charter
    )
    assert any(
        parse_rule(d)
        == ("alpha-one", "R5", "[gates] check 'pytest' is missing from tier1_checks")
        for d in defects
    )


def test_r5_charter_cmd_not_verbatim(charter: Any) -> None:
    # "improved" command: the charter's [checks.pytest].cmd is "pytest -x -q".
    defects = validate_manifest(
        [
            base_ticket(
                tier1_checks={
                    "lint": "ruff check .",
                    "pytest": "pytest -x -q --maxfail=1",
                }
            )
        ],
        charter=charter,
    )
    assert any(
        parse_rule(d)
        == ("alpha-one", "R5", "tier1_checks['pytest'] cmd must equal [checks.pytest].cmd")
        for d in defects
    )


def test_r5_charter_clean_tier1_checks(charter: Any) -> None:
    """Exact gate coverage with verbatim cmds -> clean (the fixture
    charter's checks table is {lint, pytest}, so no extra keys are
    nameable without tripping the [checks.*] rule)."""
    defects = validate_manifest(
        [
            base_ticket(
                tier1_checks={
                    "lint": "ruff check .",
                    "pytest": "pytest -x -q",
                }
            )
        ],
        charter=charter,
    )
    assert defects == []


# ==========================================================================
# R6: ids (unique; kebab-case; no store collision)
# ==========================================================================


def test_r6_duplicate_id() -> None:
    manifest = [base_ticket(), base_ticket()]
    defects = validate_manifest(manifest)
    assert any(
        parse_rule(d) == ("alpha-one", "R6", "duplicate id 'alpha-one' (entries 0, 1)")
        for d in defects
    )


def test_r6_id_not_kebab_case() -> None:
    defects = validate_manifest([base_ticket(id="SnakeCase")])
    assert any(parse_rule(d) == ("SnakeCase", "R6", "id is not kebab-case") for d in defects)


@pytest.mark.parametrize("bad_id", ["-lead", "trail-", "double--dash", "UPPER", "has_space", ""])
def test_r6_id_not_kebab_case_param(bad_id: str) -> None:
    defects = validate_manifest([base_ticket(id=bad_id)])
    label = bad_id if bad_id else "0"
    assert any(parse_rule(d) == (label, "R6", "id is not kebab-case") for d in defects)


def test_r6_store_collision(store: RigStore) -> None:
    store.add_ticket(id="alpha-one", title="Already here")
    defects = validate_manifest([base_ticket()], store=store)
    assert any(
        parse_rule(d)
        == ("alpha-one", "R6", "id collides with an existing ticket in the store")
        for d in defects
    )


def test_r6_store_no_collision(store: RigStore) -> None:
    store.add_ticket(id="other-ticket", title="Unrelated")
    assert validate_manifest([base_ticket()], store=store) == []


# ==========================================================================
# R7: blocks (unresolved ref, self-reference, ACYCLICITY)
# ==========================================================================


def test_r7_blocks_unresolved_reference() -> None:
    defects = validate_manifest([base_ticket(blocks=["ghost-ticket"])])
    assert any(
        parse_rule(d)
        == (
            "alpha-one",
            "R7",
            "blocks reference 'ghost-ticket' is not a manifest id or an existing store ticket",
        )
        for d in defects
    )


def test_r7_blocks_self_reference() -> None:
    defects = validate_manifest([base_ticket(blocks=["alpha-one"])])
    assert any(parse_rule(d) == ("alpha-one", "R7", "blocks self-reference") for d in defects)


def test_r7_blocks_to_store_id_resolves(store: RigStore) -> None:
    store.add_ticket(id="store-predecessor", title="Predecessor")
    defects = validate_manifest([base_ticket(blocks=["store-predecessor"])], store=store)
    assert defects == []


def test_r7_cycle_reports_cycle_path() -> None:
    manifest = [
        base_ticket(id="a-one", target_scope=["src/a.py"]),
        base_ticket(id="b-two", target_scope=["src/b.py"], blocks=["a-one"]),
        base_ticket(id="c-three", target_scope=["src/c.py"], blocks=["b-two"]),
        base_ticket(id="a-one", target_scope=["src/d.py"], blocks=["c-three"]),
    ]
    defects = validate_manifest(manifest)
    cycle_lines = [d for d in defects if parse_rule(d)[1] == "R7"]
    # Cycle reported following the block edges (ticket -> predecessor):
    # a is blocked by c, c by b, b by a.
    assert any("a-one -> c-three -> b-two -> a-one" in d for d in cycle_lines)


def test_r7_diamond_is_not_a_cycle() -> None:
    # a <- b, a <- c, b <- d, c <- d: a shared dependency is NOT a cycle.
    manifest = [
        base_ticket(id="a-one", target_scope=["src/a.py"]),
        base_ticket(id="b-two", target_scope=["src/b.py"], blocks=["a-one"]),
        base_ticket(id="c-three", target_scope=["src/c.py"], blocks=["a-one"]),
        base_ticket(id="d-four", target_scope=["src/d.py"], blocks=["b-two", "c-three"]),
    ]
    defects = validate_manifest(manifest)
    assert not any(parse_rule(d)[1] == "R7" for d in defects)


# ==========================================================================
# R8: target_scope (non-empty; relative; existence w/ new-file carve-out)
# ==========================================================================


def test_r8_target_scope_not_a_list() -> None:
    defects = validate_manifest([base_ticket(target_scope="src/pkg/mod.py")])
    assert any(
        parse_rule(d)
        == ("alpha-one", "R8", "target_scope must be a non-empty array of paths")
        for d in defects
    )


def test_r8_target_scope_empty_list() -> None:
    defects = validate_manifest([base_ticket(target_scope=[])])
    assert any(
        parse_rule(d)
        == ("alpha-one", "R8", "target_scope must be a non-empty array of paths")
        for d in defects
    )


def test_r8_target_scope_absolute_path() -> None:
    defects = validate_manifest([base_ticket(target_scope=["/etc/passwd"])])
    assert any(
        parse_rule(d)
        == ("alpha-one", "R8", "target_scope entry '/etc/passwd' is an absolute path")
        for d in defects
    )


def test_r8_target_scope_dotdot_traversal() -> None:
    defects = validate_manifest([base_ticket(target_scope=["../sneaky.py"])])
    assert any(
        parse_rule(d)
        == ("alpha-one", "R8", "target_scope entry '../sneaky.py' contains a '..' segment")
        for d in defects
    )


def test_r8_target_scope_dotdot_nested() -> None:
    defects = validate_manifest([base_ticket(target_scope=["a/b/../../c.py"])])
    assert any(
        parse_rule(d)
        == ("alpha-one", "R8", "target_scope entry 'a/b/../../c.py' contains a '..' segment")
        for d in defects
    )


def test_r8_target_scope_missing_path_missing_parent(tmp_path: Path) -> None:
    defects = validate_manifest([base_ticket(target_scope=["nope/nested.py"])], repo=tmp_path)
    assert any(
        parse_rule(d)
        == (
            "alpha-one",
            "R8",
            "target_scope entry 'nope/nested.py' does not exist and its parent does not either",
        )
        for d in defects
    )


def test_r8_target_scope_exists_in_repo(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text("x = 1\n")
    assert validate_manifest([base_ticket(target_scope=["src/mod.py"])], repo=tmp_path) == []


def test_r8_target_scope_new_file_with_existing_parent(tmp_path: Path) -> None:
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    # src/pkg/brand_new.py does not exist, but src/pkg/ does -> carve-out.
    assert (
        validate_manifest([base_ticket(target_scope=["src/pkg/brand_new.py"])], repo=tmp_path) == []
    )


def test_r8_target_scope_directory_path(tmp_path: Path) -> None:
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    assert validate_manifest([base_ticket(target_scope=["src/pkg"])], repo=tmp_path) == []


def test_r8_repo_exercise_against_real_repo() -> None:
    """The repo's own tree: an existing path is clean, a missing path with
    an existing parent (the new-file carve-out) is clean."""
    repo = Path(__file__).resolve().parents[1]
    defects = validate_manifest(
        [base_ticket(target_scope=[EXISTING_RELPATH, "src/stigmergy/brand_new_module.py"])],
        repo=repo,
    )
    assert defects == []


# ==========================================================================
# R9: scope disjointness (wired pairs are legal)
# ==========================================================================


def test_r9_unwired_overlap_is_a_defect() -> None:
    manifest = [
        base_ticket(id="a-one", target_scope=["src/shared.py"]),
        base_ticket(id="b-two", target_scope=["src/shared.py"]),
    ]
    defects = validate_manifest(manifest)
    # Attribution: the lexicographically lower id of the overlapping pair.
    assert any(
        parse_rule(d)
        == (
            "a-one",
            "R9",
            "target_scope overlaps ticket 'b-two' on 'src/shared.py' without a blocks edge",
        )
        for d in defects
    )


def test_r9_wired_overlap_is_legal() -> None:
    manifest = [
        base_ticket(id="a-one", target_scope=["src/shared.py"]),
        base_ticket(id="b-two", target_scope=["src/shared.py"], blocks=["a-one"]),
    ]
    assert validate_manifest(manifest) == []


def test_r9_wired_overlap_reverse_direction_is_legal() -> None:
    manifest = [
        base_ticket(id="a-one", target_scope=["src/shared.py"], blocks=["b-two"]),
        base_ticket(id="b-two", target_scope=["src/shared.py"]),
    ]
    assert validate_manifest(manifest) == []


def test_r9_disjoint_scopes_are_clean() -> None:
    manifest = [
        base_ticket(id="a-one", target_scope=["src/a.py"]),
        base_ticket(id="b-two", target_scope=["src/b.py"]),
    ]
    assert validate_manifest(manifest) == []


# ==========================================================================
# R10: difficulty vocabulary
# ==========================================================================


def test_r10_difficulty_off_vocabulary() -> None:
    defects = validate_manifest([base_ticket(difficulty="low")])
    assert any(
        parse_rule(d)
        == (
            "alpha-one",
            "R10",
            "difficulty 'low' is not one of: easy, frontier, hard, medium, trivial",
        )
        for d in defects
    )


@pytest.mark.parametrize("good", ["trivial", "easy", "medium", "hard", "frontier"])
def test_r10_difficulty_valid_values(good: str) -> None:
    assert validate_manifest([base_ticket(difficulty=good)]) == []


def test_r10_difficulty_wrong_type() -> None:
    defects = validate_manifest([base_ticket(difficulty=3)])
    assert any(parse_rule(d)[1] == "R10" and parse_rule(d)[0] == "alpha-one" for d in defects)


def test_r10_difficulty_absent_is_clean() -> None:
    entry = base_ticket()
    del entry["difficulty"]
    assert validate_manifest([entry]) == []


# ==========================================================================
# R11: lane_hint
# ==========================================================================


def test_r11_lane_hint_unknown_lane(charter: Any) -> None:
    defects = validate_manifest([base_ticket(lane_hint="mystery-lane")], charter=charter)
    assert any(
        parse_rule(d)
        == ("alpha-one", "R11", "lane_hint 'mystery-lane' is not a [lanes.*] section")
        for d in defects
    )


def test_r11_lane_hint_known_lane(charter: Any) -> None:
    assert (
        validate_manifest(
            [base_ticket(lane_hint="default", tier1_checks=_CHARTER_CLEAN_CHECKS)],
            charter=charter,
        )
        == []
    )


def test_r11_lane_hint_without_charter_not_checked() -> None:
    assert validate_manifest([base_ticket(lane_hint="anything")]) == []


# ==========================================================================
# R12: rubric_only
# ==========================================================================


def test_r12_rubric_only_non_bool() -> None:
    defects = validate_manifest([base_ticket(rubric_only="true")])
    assert any(
        parse_rule(d) == ("alpha-one", "R12", "rubric_only must be a boolean (got 'true')")
        for d in defects
    )


@pytest.mark.parametrize("good", [True, False])
def test_r12_rubric_only_bool_values(good: bool) -> None:
    assert validate_manifest([base_ticket(rubric_only=good)]) == []


# ==========================================================================
# R13 (typo-catcher): unknown keys
# ==========================================================================


def test_unknown_key_flagged_with_name() -> None:
    defects = validate_manifest([base_ticket(functional_summary_typo="oops")])
    assert any(
        parse_rule(d) == ("alpha-one", "R13", "unknown key: 'functional_summary_typo'")
        for d in defects
    )


def test_unknown_keys_all_listed() -> None:
    defects = validate_manifest([base_ticket(blocks_typo=[], tier1="typo2")])
    unknowns = [parse_rule(d) for d in defects if parse_rule(d)[1] == "R13"]
    assert len(unknowns) == 2
    details = sorted(detail for _, _, detail in unknowns)
    assert details == ["unknown key: 'blocks_typo'", "unknown key: 'tier1'"]


# ==========================================================================
# determinism: identical inputs -> identical output (ordering stable)
# ==========================================================================


def test_deterministic_ordering_across_runs() -> None:
    manifest = [
        base_ticket(id="a-one", target_scope=["src/x.py"], difficulty="low", rubric_only="true"),
        base_ticket(id="b-two", target_scope=["src/x.py", "src/y.py"], blocks=["a-one"]),
        base_ticket(id="c-three", target_scope=["src/y.py"]),
        base_ticket(id="a-one", target_scope=["src/y.py"]),
    ]
    first = validate_manifest(manifest)
    # A fresh Python-level copy of the SAME data (json round-trip) must
    # produce byte-identical output: the result order is data-determined,
    # never iteration-order-determined.
    second = validate_manifest(json.loads(json.dumps(manifest)))
    assert first == second
    assert first, "the test manifest is deliberately full of defects"


def test_defect_counts_are_complete(tmp_path: Path) -> None:
    """Every defect in a kitchen-sink manifest is reported (no rule
    shadows another) and each line is well-formed."""
    kitchen = base_ticket(
        id="Bad_ID",
        functional_summary="  ",
        acceptance_criteria=[],
        target_scope=[],
        difficulty="low",
        rubric_only="true",
        lane_hint="mystery",
        blocks=["itself", "ghost"],
        extra_key="typo",
    )
    del kitchen["title"]  # R2: a missing required key
    manifest = [kitchen, "not-an-object"]
    defects = validate_manifest(manifest, charter=load_charter_fixture(), repo=tmp_path)
    counter = Counter(rule for _, rule, _ in map(parse_rule, defects))
    assert counter["R1"] == 1  # entry 1 is a string
    assert counter["R13"] == 1  # extra_key
    for rule in ("R2", "R3", "R4", "R5", "R6", "R8", "R10", "R11", "R12"):
        assert counter[rule] >= 1, f"expected at least one {rule} defect, got {counter}"
    # R7: two unresolved references (id is "Bad_ID", so neither "itself"
    # nor "ghost" names it — no store given either)
    assert counter["R7"] == 2
    # every line parses into the structured shape
    for line in defects:
        assert re.match(r"^ticket [^:]+: [A-Z0-9]+: .+$", line), f"malformed line: {line!r}"


# ==========================================================================
# charter adapter
# ==========================================================================


def test_charter_adapter_handles_charter_and_plain_dict() -> None:
    """The adapter accepts a Charter object (via .raw) and a plain dict
    (read-only use of charter.py's parsed shape)."""
    from stigmergy.manifest import _charter_check_names, _charter_lane_names

    charter = load_charter_fixture()
    assert _charter_check_names(charter) == {"pytest", "lint"}
    assert _charter_lane_names(charter) == {"cheap", "default", "exquisite"}

    raw = {
        "checks": {"foo": {"cmd": "true"}, "bar": {"cmd": "false"}},
        "lanes": {"l1": {}, "l2": {}},
    }
    assert _charter_check_names(raw) == {"foo", "bar"}
    assert _charter_lane_names(raw) == {"l1", "l2"}
    # missing tables -> empty sets (dependent rules skip, no crash)
    assert _charter_check_names({}) == set()
    assert _charter_lane_names({}) == set()


# ==========================================================================
# CLI: `stigmergy manifest validate`
# ==========================================================================


def test_cli_clean_manifest_exit_0(tmp_path: Path, capsys) -> None:
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(clean_manifest()))

    rc = main(["manifest", "validate", "--manifest", str(manifest_file)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "0 defect(s) across 1 ticket(s)" in out


def test_cli_defects_exit_1_and_lines_on_stdout(tmp_path: Path, capsys) -> None:
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps([base_ticket(difficulty="low")]))

    rc = main(["manifest", "validate", "--manifest", str(manifest_file)])

    assert rc == 1
    out = capsys.readouterr().out
    assert "1 defect(s) across 1 ticket(s)" in out
    defect_lines = [ln for ln in out.splitlines() if not ln.endswith(")")]
    assert len(defect_lines) == 1
    assert "stigmergy manifest: ticket alpha-one: R10:" in defect_lines[0]
    assert "low" in defect_lines[0]
    assert capsys.readouterr().err == ""


def test_cli_missing_file_exit_1_stderr(tmp_path: Path, capsys) -> None:
    rc = main(["manifest", "validate", "--manifest", str(tmp_path / "nope.json")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "stigmergy manifest:" in err
    assert capsys.readouterr().out == ""


def test_cli_unparseable_json_exit_1_stderr(tmp_path: Path, capsys) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    rc = main(["manifest", "validate", "--manifest", str(bad)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "stigmergy manifest:" in err


def test_cli_manifest_not_a_list_exit_1_stderr(tmp_path: Path, capsys) -> None:
    bad = tmp_path / "notlist.json"
    bad.write_text(json.dumps({"id": "solo"}))
    rc = main(["manifest", "validate", "--manifest", str(bad)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "stigmergy manifest:" in err
    assert "JSON array" in err


def test_cli_charter_flag_enables_charter_rules(tmp_path: Path, capsys) -> None:
    # The fixture's tier1_checks {ruff, tests} do NOT name the charter's
    # [checks.*] sections {lint, pytest} -> R5 defects appear only WITH
    # --charter.
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps([json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))]))

    rc_no_charter = main(["manifest", "validate", "--manifest", str(manifest_file)])
    out_no_charter = capsys.readouterr().out
    assert rc_no_charter == 1
    assert "R5" not in out_no_charter

    rc_charter = main(
        [
            "manifest",
            "validate",
            "--manifest",
            str(manifest_file),
            "--charter",
            str(CHARTER_FIXTURE),
        ]
    )
    out_charter = capsys.readouterr().out
    assert rc_charter == 1
    assert "R5" in out_charter
    assert "0 defect(s)" not in out_charter


def test_cli_repo_flag_enables_existence_check(tmp_path: Path, capsys) -> None:
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps([base_ticket(target_scope=["ghost/mod.py"])]))

    # without --repo: the path is never checked -> clean
    rc_no_repo = main(["manifest", "validate", "--manifest", str(manifest_file)])
    out_no_repo = capsys.readouterr().out
    assert rc_no_repo == 0
    assert "R8" not in out_no_repo

    # with --repo pointing at an empty dir: missing parent -> R8 defect
    rc_repo = main(
        ["manifest", "validate", "--manifest", str(manifest_file), "--repo", str(tmp_path)]
    )
    out_repo = capsys.readouterr().out
    assert rc_repo == 1
    assert "R8" in out_repo


def test_cli_rig_resolution_end_to_end(tmp_path: Path, capsys) -> None:
    """--rig resolves repo + charter + store via the REAL resolve_rig
    (real rig scaffold, same fixture style as test_cli.py's intake tests).

    Setup: a two-ticket manifest wired a-one -> b-two that both touch
    src/README.md (a legal wired overlap). src/README.md exists in the
    cloned repo, the gate checks match the charter's [checks.*] cmds, and
    the store gets a pre-existing ticket 'store-tk' that the manifest
    blocks on (R7 store-resolution)."""
    repo_dir = tmp_path / "src_repo"
    (repo_dir / "prompts").mkdir(parents=True)
    (repo_dir / "prompts" / "code01").write_text("code01 template: $goal\n")
    (repo_dir / "prompts" / "critic01").write_text("critic01 template\n")
    (repo_dir / "src").mkdir(parents=True)
    (repo_dir / "src" / "README.md").write_text("readme\n")
    _git_init_commit(repo_dir)
    charter_path = make_charter_for_repo(tmp_path, repo_dir)
    rigs_root = tmp_path / "rigs"
    rig_root = create_rig(charter_path, base_dir=rigs_root)

    # pre-seed a store ticket the manifest will blocks-on
    seed_store = RigStore(rig_root / "tickets.db")
    seed_store.add_ticket(id="store-tk", title="Pre-existing store ticket")
    seed_store.close()

    manifest = [
        {
            "id": "a-one",
            "title": "A",
            "functional_summary": "s",
            "acceptance_criteria": ["a done"],
            "tier1_checks": {"lint": "ruff check .", "pytest": "pytest -x -q"},
            "target_scope": ["src/README.md"],
        },
        {
            "id": "b-two",
            "title": "B",
            "functional_summary": "s",
            "acceptance_criteria": ["b done"],
            "tier1_checks": {"lint": "ruff check .", "pytest": "pytest -x -q"},
            "target_scope": ["src/README.md"],
            "blocks": ["a-one", "store-tk"],
        },
    ]
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest))

    rc = main(
        [
            "manifest",
            "validate",
            "--rig",
            "shipyard",
            "--rigs-root",
            str(rigs_root),
            "--manifest",
            str(manifest_file),
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "0 defect(s) across 2 ticket(s)" in out

    # and a defect case over the SAME rig: id collision with 'store-tk'
    manifest_bad = json.loads(json.dumps(manifest))
    manifest_bad[0]["id"] = "store-tk"
    manifest_file_bad = tmp_path / "manifest_bad.json"
    manifest_file_bad.write_text(json.dumps(manifest_bad))
    rc_bad = main(
        [
            "manifest",
            "validate",
            "--rig",
            "shipyard",
            "--rigs-root",
            str(rigs_root),
            "--manifest",
            str(manifest_file_bad),
        ]
    )
    out_bad = capsys.readouterr().out
    assert rc_bad == 1
    assert "R6" in out_bad


def _git_init_commit(repo_dir: Path) -> None:
    import subprocess

    env_cfg = ["-c", "user.email=test@example.com", "-c", "user.name=Test User"]
    subprocess.run(["git", "init", "-q", str(repo_dir)], check=True)
    subprocess.run(["git", *env_cfg, "-C", str(repo_dir), "add", "-A"], check=True)
    subprocess.run(
        ["git", *env_cfg, "-C", str(repo_dir), "commit", "-q", "-m", "initial commit"],
        check=True,
    )


def make_charter_for_repo(tmp_path: Path, repo: Path) -> Path:
    """The standard valid charter fixture, pointed at ``repo``."""
    import shutil

    fixtures = Path(__file__).parent / "fixtures"
    base = (fixtures / "charter_valid.toml").read_text()
    charter_dir = tmp_path / "charter_src"
    charter_dir.mkdir(exist_ok=True)
    (charter_dir / "charter.toml").write_text(
        base.replace('repo = "path-or-url"', f'repo = "{repo}"')
    )
    shutil.copy(fixtures / "models.toml", charter_dir / "models.toml")
    return charter_dir / "charter.toml"
