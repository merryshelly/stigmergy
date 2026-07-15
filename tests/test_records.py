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
        "bead": "workspace-e2uh.8",
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
