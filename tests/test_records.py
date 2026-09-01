"""Tests for stigmergy.records (SPEC.md §8 Record Plane, §4 redaction/
credentials, §9 attempt_kind).

Governing invariants under test:

- The event log is append-only, framed + checksummed, fsync'd, and read
  back through a TOLERANT reader that skips a torn/corrupt tail rather
  than raising (crash-mid-append survivability).
- `computed_usd` is a REQUIRED field with NO default: either a
  non-negative float or the exact string `"unbudgetable"` — there is no
  code path where a missing cost silently becomes `$0`.
- LLM-invocation events (`dispatch`, `gate`, `report`) require
  `prompt_artifact_hash`; `gate` additionally requires `decoding_params`.
  Mechanism-only events (`check`, `integration`, `disposition`, `notify`)
  require neither.
- The CV projection is a derived, rebuildable, IDEMPOTENT view.
- Transcript sealing redacts BEFORE storing, and refuses (writes nothing)
  if a must-not-appear sentinel survives redaction.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

import pytest

from stigmergy.records import (
    ATTEMPT_KINDS,
    Event,
    EventType,
    RecordError,
    RecordPlane,
    make_event,
)

# --- fixtures / helpers -----------------------------------------------------


@pytest.fixture
def plane(tmp_path: Path) -> RecordPlane:
    return RecordPlane(tmp_path / "records")


def common_fields(**overrides: Any) -> dict[str, Any]:
    """A complete, valid set of SPEC §8 common fields; override as needed."""
    base: dict[str, Any] = {
        "rig": "shipyard",
        "ticket": "workspace-e2uh.8",
        "dispatch_id": "dispatch-0001",
        "attempt": 1,
        "attempt_kind": "initial",
        "rung": "cheap",
        "worker": "worker-haiku-code01-broom-casino-flock",
        "charter_hash": "charterhash123",
        "approval_hash": "approvalhash456",
        "image_digest": "sha256:deadbeef",
        "model": "haiku",
        "model_version": "haiku-3-5-20241022",
        "price_table_version": "modelshash789",
        "tokens": {"in": 100, "cached": 0, "out": 50, "reasoning": 0},
        "computed_usd": 0.0123,
        "wall_time_seconds": 12.5,
    }
    base.update(overrides)
    return base


def make_dispatch_event(**overrides: Any) -> Event:
    fields = common_fields(prompt_artifact_hash="code01-hashabc", **overrides)
    return make_event(EventType.DISPATCH, **fields)


def make_gate_event(**overrides: Any) -> Event:
    fields = common_fields(
        prompt_artifact_hash="critic01-hashdef",
        decoding_params={"temperature": 0.0},
        **overrides,
    )
    return make_event(EventType.GATE, **fields)


def make_check_event(**overrides: Any) -> Event:
    overrides.setdefault("computed_usd", 0.0)
    fields = common_fields(**overrides)
    return make_event(EventType.CHECK, **fields)


# --- case 1: roundtrip -------------------------------------------------------


def test_append_then_read_events_roundtrips_in_order(plane: RecordPlane) -> None:
    events = [make_dispatch_event(dispatch_id=f"dispatch-{i:04d}", attempt=i) for i in range(5)]
    for ev in events:
        plane.append(ev)

    read_back = plane.read_events()

    assert len(read_back) == 5
    assert [e["dispatch_id"] for e in read_back] == [f"dispatch-{i:04d}" for i in range(5)]
    for original, read in zip(events, read_back, strict=True):
        for key, value in original.payload.items():
            assert read[key] == value


# --- case 2: torn-write tolerance --------------------------------------------


def test_read_events_tolerates_torn_tail(plane: RecordPlane) -> None:
    valid_events = [make_dispatch_event(dispatch_id=f"dispatch-{i:04d}") for i in range(3)]
    for ev in valid_events:
        plane.append(ev)

    # Simulate a crash mid-append: a truncated/garbage partial line with no
    # trailing newline, appended directly (bypassing RecordPlane.append).
    with open(plane.events_path, "a", encoding="utf-8") as fh:
        fh.write('{"event_type": "dispatch", "rig": "shipyard", "dispatch_')

    read_back = plane.read_events()

    assert len(read_back) == 3
    assert [e["dispatch_id"] for e in read_back] == [f"dispatch-{i:04d}" for i in range(3)]


# --- case 3: checksum skip ----------------------------------------------------


def test_read_events_skips_line_with_bad_checksum(plane: RecordPlane) -> None:
    valid_events = [make_dispatch_event(dispatch_id=f"dispatch-{i:04d}") for i in range(3)]
    for ev in valid_events:
        plane.append(ev)

    lines = plane.events_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3

    # Hand-corrupt the second committed line's payload so its checksum no
    # longer matches (simulate bit rot / partial overwrite).
    record = json.loads(lines[1])
    record["attempt"] = 999
    lines[1] = json.dumps(record, sort_keys=True, separators=(",", ":"))
    plane.events_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    read_back = plane.read_events()

    assert len(read_back) == 2
    assert [e["dispatch_id"] for e in read_back] == ["dispatch-0000", "dispatch-0002"]


# --- case 4: invalid attempt_kind --------------------------------------------


def test_make_event_invalid_attempt_kind_raises(plane: RecordPlane) -> None:
    with pytest.raises(RecordError):
        make_dispatch_event(attempt_kind="not-a-real-kind")


def test_attempt_kinds_matches_spec_enumeration() -> None:
    assert ATTEMPT_KINDS == frozenset(
        {
            "initial",
            "tier1-repair",
            "critic-revision",
            "integration-reconcile",
            "infra-retry",
            "stepup-initial",
            "clean-restart",
            # bead .42: rig-level range-report is not a ticket attempt;
            # 'report' is the honest attempt_kind for a REPORT event
            # (SPEC §9 enumeration gains this member — follow-up note for SB).
            "report",
            # bead .107: a per-ticket critic-infra escalation
            # (PARKED->ESCALATED) is a distinct, queryable outcome from a
            # dispatch-side 'infra-retry' (SPEC §9 enumeration gains this
            # member too — follow-up note for SB).
            "critic-infra",
            # spec-failure: a spec-failure escalation (FailureClass.SPEC_FAILURE,
            # the worker's blocks_ticket:true escape) is a distinct, queryable
            # outcome from a dispatch-side infra-retry or a ladder-exhausted
            # escalation (same precedent as 'report' and 'critic-infra').
            "spec-failure",
            # bead .152 (Decision 17): a decompose-band invocation (the
            # decomposer exec run or the decomposition validation critic) is
            # rig-level, not a ticket attempt; reusing 'report' (or any
            # ticket-adjacent value) would be a recorded lie (same precedent as
            # 'report' and 'critic-infra'). SPEC §9 enumeration gains this
            # member too — follow-up note for SB.
            "decompose",
        }
    )


# --- case 5: no-$0-default ----------------------------------------------------


def test_make_event_missing_computed_usd_raises() -> None:
    fields = common_fields(prompt_artifact_hash="code01-hashabc")
    del fields["computed_usd"]
    with pytest.raises(RecordError):
        make_event(EventType.DISPATCH, **fields)


def test_make_event_computed_usd_unbudgetable_accepted() -> None:
    ev = make_dispatch_event(computed_usd="unbudgetable")
    assert ev.payload["computed_usd"] == "unbudgetable"


def test_make_event_computed_usd_explicit_zero_accepted() -> None:
    ev = make_dispatch_event(computed_usd=0.0)
    assert ev.payload["computed_usd"] == 0.0


def test_make_event_computed_usd_negative_rejected() -> None:
    with pytest.raises(RecordError):
        make_dispatch_event(computed_usd=-1.0)


def test_make_event_computed_usd_bad_string_rejected() -> None:
    with pytest.raises(RecordError):
        make_dispatch_event(computed_usd="free")


# --- case 6: prompt_artifact_hash required on LLM-invocation events ---------


@pytest.mark.parametrize("event_type", [EventType.DISPATCH, EventType.GATE, EventType.REPORT])
def test_llm_invocation_event_missing_prompt_artifact_hash_raises(
    event_type: EventType,
) -> None:
    fields = common_fields()
    if event_type is EventType.GATE:
        fields["decoding_params"] = {"temperature": 0.0}
    with pytest.raises(RecordError):
        make_event(event_type, **fields)


def test_dispatch_event_with_prompt_artifact_hash_ok() -> None:
    ev = make_dispatch_event()
    assert ev.payload["prompt_artifact_hash"] == "code01-hashabc"


def test_report_event_with_prompt_artifact_hash_ok() -> None:
    fields = common_fields(prompt_artifact_hash="rangecrit01-hashxyz")
    ev = make_event(EventType.REPORT, **fields)
    assert ev.payload["prompt_artifact_hash"] == "rangecrit01-hashxyz"


# --- case 7: gate requires decoding_params + prompt_artifact_hash readable --


def test_gate_event_missing_decoding_params_raises() -> None:
    fields = common_fields(prompt_artifact_hash="critic01-hashdef")
    with pytest.raises(RecordError):
        make_event(EventType.GATE, **fields)


def test_gate_event_with_decoding_params_ok_and_prompt_hash_readable(
    plane: RecordPlane,
) -> None:
    ev = make_gate_event()
    assert ev.payload["decoding_params"] == {"temperature": 0.0}
    assert ev.payload["prompt_artifact_hash"] == "critic01-hashdef"

    plane.append(ev)
    read_back = plane.read_events()
    assert len(read_back) == 1
    assert read_back[0]["prompt_artifact_hash"] == "critic01-hashdef"
    assert read_back[0]["decoding_params"] == {"temperature": 0.0}


def test_gate_event_with_empty_decoding_params_is_valid(plane: RecordPlane) -> None:
    # bead .81/.95: opus-4-8/sonnet-5 reject sampling params, so the critic
    # sends NONE and decoding_params is an EMPTY dict — a valid, meaningful
    # value (logged provenance: "no sampling params sent"), NOT a missing
    # field. Must be accepted; only an absent field / non-dict is rejected.
    fields = common_fields(prompt_artifact_hash="critic01-hashdef")
    fields["decoding_params"] = {}
    ev = make_event(EventType.GATE, **fields)
    assert ev.payload["decoding_params"] == {}
    plane.append(ev)
    assert plane.read_events()[0]["decoding_params"] == {}


# --- case 8: mechanism-only event is valid without LLM extras --------------


def test_check_event_without_llm_fields_is_valid(plane: RecordPlane) -> None:
    ev = make_check_event()
    assert "prompt_artifact_hash" not in ev.payload
    assert "decoding_params" not in ev.payload
    plane.append(ev)
    read_back = plane.read_events()
    assert len(read_back) == 1
    assert read_back[0]["event_type"] == "check"


@pytest.mark.parametrize(
    "event_type", [EventType.INTEGRATION, EventType.DISPOSITION, EventType.NOTIFY]
)
def test_other_mechanism_events_valid_without_llm_fields(event_type: EventType) -> None:
    fields = common_fields(computed_usd=0.0)
    ev = make_event(event_type, **fields)
    assert "prompt_artifact_hash" not in ev.payload


# --- case 9: CV projection idempotence --------------------------------------


def test_rebuild_cv_idempotent_one_row_per_dispatch(plane: RecordPlane) -> None:
    plane.append(make_dispatch_event(dispatch_id="dispatch-A", attempt=1))
    plane.append(
        make_check_event(dispatch_id="dispatch-A", attempt=1, computed_usd=0.0)
    )
    plane.append(make_dispatch_event(dispatch_id="dispatch-B", attempt=1, computed_usd=0.05))
    plane.append(
        make_gate_event(dispatch_id="dispatch-B", attempt=1, computed_usd="unbudgetable")
    )

    plane.rebuild_cv()
    first_bytes = plane.cv_path.read_bytes()

    plane.rebuild_cv()
    second_bytes = plane.cv_path.read_bytes()

    assert first_bytes == second_bytes

    rows = [json.loads(line) for line in first_bytes.decode("utf-8").splitlines() if line]
    assert len(rows) == 2
    dispatch_ids = {row["dispatch_id"] for row in rows}
    assert dispatch_ids == {"dispatch-A", "dispatch-B"}

    row_b = next(row for row in rows if row["dispatch_id"] == "dispatch-B")
    assert row_b["has_unbudgetable"] is True
    assert row_b["computed_usd_total"] == pytest.approx(0.05)


def test_build_cv_row_retains_disposition_and_reason(plane: RecordPlane) -> None:
    """_build_cv_row extracts disposition and reason from the last DISPOSITION
    event. A spec-failure escalation (disposition='escalated', reason='spec-failure')
    is distinguishable from a ladder-exhausted one (reason='ladder-exhausted'),
    and a dispatch with no DISPOSITION event has both defaulting to None."""
    # Build a spec-failure dispatch: DISPATCH + DISPOSITION
    plane.append(make_dispatch_event(dispatch_id="spec-fail-dispatch"))
    plane.append(
        make_event(
            EventType.DISPOSITION,
            **common_fields(
                dispatch_id="spec-fail-dispatch",
                attempt_kind="spec-failure",
                disposition="escalated",
                reason="spec-failure",
            ),
        )
    )

    # Build a ladder-exhausted dispatch: DISPATCH + DISPOSITION
    plane.append(
        make_dispatch_event(dispatch_id="ladder-exhausted-dispatch", attempt=2)
    )
    plane.append(
        make_event(
            EventType.DISPOSITION,
            **common_fields(
                dispatch_id="ladder-exhausted-dispatch",
                attempt=2,
                attempt_kind="tier1-repair",
                disposition="escalated",
                reason="ladder-exhausted",
            ),
        )
    )

    # Build a dispatch with no DISPOSITION event
    plane.append(
        make_dispatch_event(dispatch_id="no-disposition-dispatch", attempt=3)
    )

    plane.rebuild_cv()
    cv_text = plane.cv_path.read_bytes().decode("utf-8")
    rows = [json.loads(line) for line in cv_text.splitlines() if line]

    # Verify spec-failure row
    spec_fail_row = next(r for r in rows if r["dispatch_id"] == "spec-fail-dispatch")
    assert spec_fail_row["disposition"] == "escalated"
    assert spec_fail_row["reason"] == "spec-failure"
    assert spec_fail_row["final_outcome"] == "disposition"

    # Verify ladder-exhausted row (different reason, same disposition)
    ladder_row = next(r for r in rows if r["dispatch_id"] == "ladder-exhausted-dispatch")
    assert ladder_row["disposition"] == "escalated"
    assert ladder_row["reason"] == "ladder-exhausted"
    assert spec_fail_row["reason"] != ladder_row["reason"]  # Distinguishable

    # Verify no-DISPOSITION row defaults to None
    no_disp_row = next(r for r in rows if r["dispatch_id"] == "no-disposition-dispatch")
    assert no_disp_row["disposition"] is None
    assert no_disp_row["reason"] is None

    # Verify existing keys are still present and unchanged
    for row in rows:
        assert "dispatch_id" in row
        assert "ticket" in row
        assert "rung" in row
        assert "model" in row
        assert "event_count" in row
        assert "tokens" in row
        assert "computed_usd_total" in row
        assert "has_unbudgetable" in row
        assert "final_outcome" in row


# --- case 10: seal redaction --------------------------------------------------


def test_seal_transcript_redacts_before_storing(plane: RecordPlane) -> None:
    content = "leaked value: SECRET123 in the transcript"

    def redactor(text: str) -> str:
        return text.replace("SECRET123", "[REDACTED]")

    ref = plane.seal_transcript(content, redactor=redactor)

    blob_path = plane.transcripts_dir / ref
    assert blob_path.exists()
    stored = blob_path.read_text(encoding="utf-8")
    assert "SECRET123" not in stored
    assert "[REDACTED]" in stored

    import hashlib

    expected_ref = hashlib.sha256(redactor(content).encode("utf-8")).hexdigest()
    assert ref == expected_ref


# --- case 11: seal refuses unredacted -----------------------------------------


def test_seal_transcript_refuses_when_sentinel_survives_redaction(
    plane: RecordPlane,
) -> None:
    content = "leaked value: SECRET123 in the transcript"

    def identity(text: str) -> str:
        return text

    before = set(plane.transcripts_dir.iterdir())

    with pytest.raises(RecordError):
        plane.seal_transcript(content, redactor=identity, must_not_contain={"SECRET123"})

    after = set(plane.transcripts_dir.iterdir())
    assert after == before  # no blob file written


# --- case 12: strict modes -----------------------------------------------------


def test_events_jsonl_created_with_strict_mode(plane: RecordPlane) -> None:
    plane.append(make_check_event())
    mode = plane.events_path.stat().st_mode & 0o777
    assert mode == 0o600


def test_cv_jsonl_created_with_strict_mode(plane: RecordPlane) -> None:
    plane.append(make_check_event())
    plane.rebuild_cv()
    mode = plane.cv_path.stat().st_mode & 0o777
    assert mode == 0o600


def test_sealed_transcript_blob_created_with_strict_mode(plane: RecordPlane) -> None:
    ref = plane.seal_transcript("hello world", redactor=lambda s: s)
    blob_path = plane.transcripts_dir / ref
    mode = blob_path.stat().st_mode & 0o777
    assert mode == 0o600


# --- extra: RecordPlane.__init__ scaffolds directories with strict-ish setup


def test_record_plane_init_creates_records_and_transcripts_dirs(tmp_path: Path) -> None:
    records_dir = tmp_path / "records"
    plane = RecordPlane(records_dir)
    assert plane.records_dir.is_dir()
    assert plane.transcripts_dir.is_dir()
    assert plane.transcripts_dir == records_dir / "transcripts"


def test_append_rejects_non_event(plane: RecordPlane) -> None:
    with pytest.raises(RecordError):
        plane.append({"event_type": "check"})  # type: ignore[arg-type]


def test_make_event_rejects_unknown_event_type() -> None:
    with pytest.raises(RecordError):
        make_event("not-a-real-type", **common_fields())


def test_tokens_missing_key_rejected() -> None:
    fields = common_fields(prompt_artifact_hash="code01-hashabc")
    fields["tokens"] = {"in": 1, "cached": 0, "out": 1}  # missing "reasoning"
    with pytest.raises(RecordError):
        make_event(EventType.DISPATCH, **fields)


def test_tokens_negative_value_rejected() -> None:
    fields = common_fields(prompt_artifact_hash="code01-hashabc")
    fields["tokens"] = {"in": -1, "cached": 0, "out": 1, "reasoning": 0}
    with pytest.raises(RecordError):
        make_event(EventType.DISPATCH, **fields)


def test_missing_common_field_rejected() -> None:
    fields = common_fields(prompt_artifact_hash="code01-hashabc")
    del fields["charter_hash"]
    with pytest.raises(RecordError):
        make_event(EventType.DISPATCH, **fields)


def test_stat_mode_helper_uses_stat_module() -> None:
    # sanity import check to keep `stat` import used/meaningful in this file
    assert stat.S_IMODE(0o600) == 0o600


# --- D14: ticket-filed event (bead workspace-e2uh.38) -----------------------


def test_ticket_filed_event_type_exists() -> None:
    assert EventType.TICKET_FILED.value == "ticket-filed"


def _filed_fields(**overrides: Any) -> dict[str, Any]:
    """ticket-filed common fields with honest zeros for the cost fields."""
    base = common_fields(
        computed_usd=0.0,
        tokens={"in": 0, "cached": 0, "out": 0, "reasoning": 0},
        wall_time_seconds=0.0,
    )
    base.update(
        origin={
            "role": "worker",
            "worker": base["worker"],
            "dispatch_id": base["dispatch_id"],
            "parent_ticket": base["ticket"],
        },
        proposal_hash="proposalhashabc",
        outcome="accepted",
        reason=None,
        filed_ticket_id="filed-dispatch-0001-1",
    )
    base.update(overrides)
    return base


def test_ticket_filed_event_validates_and_roundtrips(plane: RecordPlane) -> None:
    ev = make_event(EventType.TICKET_FILED, **_filed_fields())
    plane.append(ev)
    read = plane.read_events()
    assert len(read) == 1
    assert read[0]["event_type"] == "ticket-filed"
    assert read[0]["outcome"] == "accepted"
    assert read[0]["origin"]["role"] == "worker"


def test_ticket_filed_is_not_an_llm_invocation_no_prompt_hash_required() -> None:
    # unlike dispatch/gate/report, ticket-filed carries no prompt_artifact_hash.
    fields = _filed_fields()
    assert "prompt_artifact_hash" not in fields
    make_event(EventType.TICKET_FILED, **fields)  # must NOT raise


def test_ticket_filed_still_requires_common_fields() -> None:
    fields = _filed_fields()
    del fields["computed_usd"]
    with pytest.raises(RecordError):
        make_event(EventType.TICKET_FILED, **fields)


# --- bead .42: triage attribution events + report attempt_kind --------------


def _triage_fields(**overrides: Any) -> dict[str, Any]:
    """The frozen triage-event required set (NO dispatch-shaped common fields)."""
    base: dict[str, Any] = {
        "rig": "shipyard",
        "subject_id": "workspace-e2uh.8",
        "outcome": "approved",
        "acting_agent": "merry",
        "operator_session": "2026-07-16T21:40-triage-1",
    }
    base.update(overrides)
    return base


def test_D1r_triage_event_types_exist_and_enum_has_thirteen_members() -> None:
    assert EventType.APPROVAL.value == "approval"
    assert EventType.UNAPPROVAL.value == "unapproval"
    assert EventType.TRIAGE_REJECTED.value == "triage-rejected"
    assert EventType.RESUME.value == "resume"
    # bead .152 (Decision 17): DECOMPOSE joins as the 13th member — the
    # decomposer band's LLM invocations (rig-level, no ticket, same
    # precedent as REPORT/TICKET_FILED).
    assert len(list(EventType)) == 13


@pytest.mark.parametrize(
    "event_type,outcome",
    [
        (EventType.APPROVAL, "approved"),
        (EventType.UNAPPROVAL, "unapproved"),
        (EventType.TRIAGE_REJECTED, "rejected"),
        (EventType.RESUME, "resumed"),
    ],
)
def test_D2r_triage_event_validates_without_common_fields(
    plane: RecordPlane, event_type: EventType, outcome: str
) -> None:
    """A triage act carries only its own required set — NO dispatch_id, attempt,
    attempt_kind, tokens, computed_usd, wall_time_seconds — and round-trips."""
    fields = _triage_fields(outcome=outcome, approval_hash="steeringhashabc")
    ev = make_event(event_type, **fields)
    plane.append(ev)
    read = plane.read_events()
    assert len(read) == 1
    assert read[0]["event_type"] == event_type.value
    assert read[0]["acting_agent"] == "merry"
    assert read[0]["operator_session"] == "2026-07-16T21:40-triage-1"
    assert read[0]["outcome"] == outcome
    # exempt from the dispatch-shaped schema
    assert "attempt_kind" not in read[0]
    assert "computed_usd" not in read[0]


@pytest.mark.parametrize(
    "missing", ["rig", "subject_id", "outcome", "acting_agent", "operator_session"]
)
def test_D3r_triage_event_missing_required_field_raises(missing: str) -> None:
    fields = _triage_fields()
    del fields[missing]
    with pytest.raises(RecordError):
        make_event(EventType.APPROVAL, **fields)


@pytest.mark.parametrize("empty", ["", "   "])
def test_D3r_triage_event_empty_attribution_rejected(empty: str) -> None:
    """A forgeable-by-omission audit line is not acceptable: empty/blank
    acting_agent must be rejected, not silently accepted."""
    with pytest.raises(RecordError):
        make_event(EventType.APPROVAL, **_triage_fields(acting_agent=empty))


def test_D4r_triage_events_do_not_pollute_cv_or_spend(plane: RecordPlane) -> None:
    """A triage event (no dispatch_id / computed_usd / tokens) produces NO CV
    row and counts as neither dispatch nor gate, adding 0 to metered spend."""
    from stigmergy.status import reconstruct_spend

    plane.append(make_event(EventType.APPROVAL, **_triage_fields(approval_hash="h")))
    plane.append(
        make_event(
            EventType.TRIAGE_REJECTED,
            **_triage_fields(subject_id="filed-x", outcome="rejected"),
        )
    )
    plane.rebuild_cv()

    cv_lines = [ln for ln in plane.cv_path.read_text().splitlines() if ln.strip()]
    assert cv_lines == []  # no dispatch_id -> no CV rows

    spend = reconstruct_spend(plane, usd_cap=25.0, dispatches_cap=50, gate_calls_cap=30)
    assert spend["metered_spent"] == 0.0
    assert spend["dispatches_used"] == 0
    assert spend["gate_calls_used"] == 0


def test_D5r_report_event_with_report_attempt_kind_validates() -> None:
    fields = common_fields(
        attempt_kind="report",
        prompt_artifact_hash="rangecrit01-hashxyz",
        ticket=None,
        dispatch_id="report-abc123def456",
        worker=None,
        rung=None,
        image_digest=None,
        model_version=None,
    )
    ev = make_event(EventType.REPORT, **fields)
    assert ev.payload["attempt_kind"] == "report"
    assert ev.payload["prompt_artifact_hash"] == "rangecrit01-hashxyz"


# --- bead .152: decompose-band event (EventType.DECOMPOSE) -----------------
# The decomposer station's LLM invocations (the decomposer exec run + the
# decomposition validation critic) are rig-level, no ticket — the same
# justification precedent as `report` and `ticket-filed` (Decision 17 / bead
# workspace-e2uh.152). They are LLM invocations, so they carry a
# hash-bearing prompt_artifact_hash; they are NOT gate events, so they do
# NOT require decoding_params.

VALID_DECOMPOSE_HASH = "decomp01-hashabc"


def _decompose_fields(**overrides: Any) -> dict[str, Any]:
    """A complete, valid decompose-band event field set: the full
    dispatch-shaped common-field set (ticket/worker/rung/rung=None and
    attempt=0 are legal rig-level values, per cli._emit_report_event) plus
    attempt_kind="decompose" and a hash-bearing prompt_artifact_hash."""
    base = common_fields(
        ticket=None,
        dispatch_id="decompose-0001",
        attempt=0,
        attempt_kind="decompose",
        rung=None,
        worker=None,
        image_digest=None,
        model_version=None,
        prompt_artifact_hash=VALID_DECOMPOSE_HASH,
    )
    base.update(overrides)
    return base


def test_D152_decompose_event_type_is_the_thirteenth_member() -> None:
    assert EventType.DECOMPOSE.value == "decompose"
    assert len(list(EventType)) == 13
    # DECOMPOSE is an LLM-invocation event (SPEC §4 prompt-artifact invariant).
    from stigmergy.records import _LLM_INVOCATION_TYPES

    assert EventType.DECOMPOSE in _LLM_INVOCATION_TYPES


def test_D152_decompose_attempt_kind_is_accepted() -> None:
    assert "decompose" in ATTEMPT_KINDS


def test_D152_valid_decompose_event_validates() -> None:
    """A valid DECOMPOSE event with all common fields + attempt_kind=
    "decompose" + prompt_artifact_hash validates."""
    ev = make_event(EventType.DECOMPOSE, **_decompose_fields())
    assert ev.payload["event_type"] == "decompose"
    assert ev.payload["attempt_kind"] == "decompose"
    assert ev.payload["ticket"] is None
    assert ev.payload["worker"] is None
    assert ev.payload["rung"] is None
    assert ev.payload["attempt"] == 0
    assert ev.payload["prompt_artifact_hash"] == VALID_DECOMPOSE_HASH
    # decompose is NOT a gate event -> no decoding_params required or carried.
    assert "decoding_params" not in ev.payload


def test_D152_decompose_missing_prompt_artifact_hash_raises() -> None:
    fields = _decompose_fields()
    del fields["prompt_artifact_hash"]
    with pytest.raises(RecordError):
        make_event(EventType.DECOMPOSE, **fields)


def test_D152_decompose_does_not_require_decoding_params() -> None:
    """A 'decompose' event does NOT require decoding_params; a GATE event
    WITHOUT decoding_params must still raise (the existing gate rule holds)."""
    # (a) decompose is valid with NO decoding_params at all.
    make_event(EventType.DECOMPOSE, **_decompose_fields())  # must NOT raise
    # (b) the gate rule is unchanged: a GATE event missing decoding_params
    # still raises, even though the decompose field set above is valid.
    gate_fields = common_fields(prompt_artifact_hash="critic01-hashdef")
    with pytest.raises(RecordError):
        make_event(EventType.GATE, **gate_fields)
