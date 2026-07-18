"""Tests for stigmergy.triage (bead workspace-e2uh.42 — triage promotion +
attribution audit events; SPEC §6 item 10 / D2 / D15 / D1).

`promote_proposal` completes a filed proposal into a §6 ticket (UNAPPROVED —
promotion and approval are distinct acts) and tombstones the filed row.
`record_triage_event` writes the v0 audit line (agent-asserted acting agent +
operator session, D1) into the ONE event log.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from stigmergy.records import EventType, RecordPlane
from stigmergy.rig import RigStore
from stigmergy.triage import TriageError, promote_proposal, record_triage_event


@pytest.fixture
def store(tmp_path: Path) -> RigStore:
    s = RigStore.create(tmp_path / "tickets.db")
    yield s
    s.close()


@pytest.fixture
def plane(tmp_path: Path) -> RecordPlane:
    return RecordPlane(tmp_path / "records")


def _seed_filed(store: RigStore, filed_id: str = "filed-dispatch-1-1") -> str:
    store.add_filed_ticket(
        id=filed_id,
        title="proposal title",
        description="a discovered proposal",
        origin_role="worker",
        origin_worker="worker-haiku-code01-broom-casino-flock",
        origin_dispatch_id="dispatch-1",
        origin_parent_ticket="workspace-e2uh.8",
        discovered_from="dispatch-1@workspace-e2uh.8",
        proposal_hash="proposalhash",
    )
    return filed_id


def _spec(**overrides: object) -> dict:
    spec: dict = {
        "id": "t-new",
        "title": "Do the discovered thing",
        "functional_summary": "Operator-facing: the export no longer truncates.",
        "acceptance_criteria": ["export includes all rows"],
        "tier1_checks": {"pytest": "pytest -q"},
        "target_scope": ["src/export.py"],
        "goal": "fix the truncation",
        "required_reading": ["repo:src/export.py"],
        "difficulty": "medium",
    }
    spec.update(overrides)
    return spec


# === B6: promote happy path ================================================


def test_B6_promote_proposal_happy_path(store: RigStore) -> None:
    filed_id = _seed_filed(store)
    spec = _spec(blocks=["workspace-e2uh.4", "workspace-e2uh.5"])

    ticket_id = promote_proposal(store, filed_id=filed_id, spec=spec)

    assert ticket_id == "t-new"
    ticket = store.get_ticket("t-new")
    assert ticket is not None
    assert ticket["approved"] == 0  # UNAPPROVED
    assert ticket["functional_summary"] == "Operator-facing: the export no longer truncates."
    assert ticket["acceptance_criteria"] == ["export includes all rows"]
    assert ticket["target_scope"] == ["src/export.py"]

    # blocks -> predecessor edges (deps_of returns predecessors)
    assert set(store.deps_of("t-new")) == {"workspace-e2uh.4", "workspace-e2uh.5"}

    # filed row tombstoned/promoted
    filed = store.list_filed_tickets(triaged=True)
    assert len(filed) == 1
    assert filed[0]["triage_outcome"] == "promoted"
    assert filed[0]["resulting_ticket_id"] == "t-new"
    assert store.count_untriaged_filings() == 0


def test_B6_promoted_ticket_is_steerable(store: RigStore, tmp_path: Path) -> None:
    """A promoted ticket carries every field derive_steering reads."""
    from stigmergy.charter import load_charter
    from stigmergy.steering import derive_steering

    filed_id = _seed_filed(store)
    promote_proposal(store, filed_id=filed_id, spec=_spec(lane_hint=None))

    fixtures = Path(__file__).parent / "fixtures"
    charter_dir = tmp_path / "charter_src"
    charter_dir.mkdir()
    (charter_dir / "charter.toml").write_text((fixtures / "charter_valid.toml").read_text())
    import shutil

    shutil.copy(fixtures / "models.toml", charter_dir / "models.toml")
    charter = load_charter(charter_dir / "charter.toml", env={})
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "code01").write_text("template\n")

    steering = derive_steering(store.get_ticket("t-new"), charter, prompts_dir)
    assert steering["functional_summary"] == "Operator-facing: the export no longer truncates."


# === B7: functional_summary required + non-empty ===========================


@pytest.mark.parametrize("bad", [None, "", "   ", "__omit__"])
def test_B7_missing_or_empty_functional_summary_rejected(store: RigStore, bad: object) -> None:
    filed_id = _seed_filed(store)
    spec = _spec()
    if bad == "__omit__":
        del spec["functional_summary"]
    else:
        spec["functional_summary"] = bad

    with pytest.raises(TriageError):
        promote_proposal(store, filed_id=filed_id, spec=spec)

    # nothing written: no ticket, filed row still untriaged
    assert store.get_ticket("t-new") is None
    assert store.count_untriaged_filings() == 1


# === B8: bad filed_id ======================================================


def test_B8_unknown_filed_id_rejected(store: RigStore) -> None:
    with pytest.raises(TriageError):
        promote_proposal(store, filed_id="does-not-exist", spec=_spec())


def test_B8_already_triaged_filed_id_rejected(store: RigStore) -> None:
    filed_id = _seed_filed(store)
    store.mark_filed_ticket_triaged(filed_id, outcome="rejected")
    with pytest.raises(TriageError):
        promote_proposal(store, filed_id=filed_id, spec=_spec())


# === B9: missing other required §6 keys ====================================


@pytest.mark.parametrize(
    "missing", ["id", "title", "acceptance_criteria", "tier1_checks", "target_scope"]
)
def test_B9_missing_required_field_rejected(store: RigStore, missing: str) -> None:
    filed_id = _seed_filed(store)
    spec = _spec()
    del spec[missing]
    with pytest.raises(TriageError):
        promote_proposal(store, filed_id=filed_id, spec=spec)
    assert store.count_untriaged_filings() == 1


# === C-lib1: record_triage_event ===========================================


def test_Clib1_record_approval_event(plane: RecordPlane) -> None:
    record_triage_event(
        plane,
        event_type=EventType.APPROVAL,
        rig="shipyard",
        subject_id="t-new",
        outcome="approved",
        acting_agent="merry",
        operator_session="sess-1",
        approval_hash="steeringhash123",
    )
    events = plane.read_events()
    assert len(events) == 1
    ev = events[0]
    assert ev["event_type"] == "approval"
    assert ev["subject_id"] == "t-new"
    assert ev["acting_agent"] == "merry"
    assert ev["operator_session"] == "sess-1"
    assert ev["approval_hash"] == "steeringhash123"


def test_Clib1_record_rejection_event_carries_reason(plane: RecordPlane) -> None:
    record_triage_event(
        plane,
        event_type=EventType.TRIAGE_REJECTED,
        rig="shipyard",
        subject_id="filed-dispatch-1-1",
        outcome="rejected",
        acting_agent="merry",
        operator_session="sess-1",
        reason="duplicate of t-12",
    )
    ev = plane.read_events()[0]
    assert ev["event_type"] == "triage-rejected"
    assert ev["reason"] == "duplicate of t-12"
