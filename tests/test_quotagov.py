"""Tests for the quota governor's signal ingestion layer (quotagov).

Covers the acceptance contract: the wire shape from spike bundle 138
(``synthetic_quotas`` = ``rollingFiveHourLimit`` {remaining/max/limited/
nextTickAt} + ``weeklyTokenLimit`` dollar-string credits; ``upstream_429_body``
verbatim), most-recent-wins per provider, malformed-line tolerance (the
reader never raises), and bounded offset-based re-read (a second read
after appending processes ONLY the appended lines).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from stigmergy.quotagov import (
    DEFAULT_ESCALATION_CEILING_S,
    GovernorDecision,
    QuotaGovernor,
    QuotaState,
    RollingWindow,
    decide,
    read_feed,
)
from stigmergy.registry import ModelEntry, PricingClass

# A spike-bundle-138-shaped synthetic_quotas payload.
FULL_QUOTAS = json.dumps(
    {
        "rollingFiveHourLimit": {
            "remaining": 12345,
            "max": 100000,
            "limited": False,
            "nextTickAt": "2026-09-01T12:00:00Z",
        },
        "weeklyTokenLimit": {
            "creditUsed": "$12.34",
            "creditRemaining": "$87.66",
            "creditLimit": "$100.00",
        },
    }
)


def _line(**kw) -> str:
    base = {
        "ts": kw.pop("ts", 1000.0),
        "decision": kw.pop("decision", "allow"),
        "status": kw.pop("status", 200),
        "synthetic_quotas": None,
        "upstream_429_body": None,
    }
    base.update(kw)
    return json.dumps(base)


def _write(path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")


# =========================================================================== #
# read_feed — happy path                                                      #
# =========================================================================== #
class TestReadFeed:
    def test_wellformed_quotas_record_folds_into_state(self, tmp_path):
        path = tmp_path / "relay.jsonl"
        _write(path, [_line(dispatch_id="d1", synthetic_quotas=FULL_QUOTAS)])

        governor, next_offset = read_feed(path)

        assert next_offset == path.stat().st_size
        state = governor.providers["d1"]
        assert isinstance(state, QuotaState)
        assert state.rolling.remaining == 12345
        assert state.rolling.max == 100000
        assert state.rolling.limited is False
        assert state.rolling.next_tick_at == "2026-09-01T12:00:00Z"
        assert state.limited is False
        assert state.next_tick_at == "2026-09-01T12:00:00Z"
        assert state.weekly.credit_used == 12.34
        assert state.weekly.credit_remaining == 87.66
        assert state.weekly.credit_limit == 100.00
        assert state.last_429_at is None
        assert state.updated_at == 1000.0

    def test_upstream_429_body_record_marks_429(self, tmp_path):
        path = tmp_path / "relay.jsonl"
        _write(
            path,
            [
                _line(
                    dispatch_id="d1",
                    status=429,
                    upstream_429_body='{"error":"quota exhausted"}',
                )
            ],
        )

        governor, _ = read_feed(path)

        state = governor.providers["d1"]
        assert state.last_429_at == 1000.0
        # A 429 body that is not the synthetic quota shape carries no
        # window/credit data — the timestamp is the evidence.
        assert state.rolling is None
        assert state.weekly is None

    def test_upstream_429_body_with_synthetic_shape_parses_quotas(self, tmp_path):
        path = tmp_path / "relay.jsonl"
        _write(
            path,
            [_line(dispatch_id="d1", status=429, upstream_429_body=FULL_QUOTAS)],
        )

        governor, _ = read_feed(path)

        state = governor.providers["d1"]
        assert state.last_429_at == 1000.0
        assert state.rolling.remaining == 12345
        assert state.weekly.credit_limit == 100.00

    def test_most_recent_record_per_provider_wins(self, tmp_path):
        path = tmp_path / "relay.jsonl"
        old = json.dumps(
            {
                "rollingFiveHourLimit": {
                    "remaining": 900,
                    "max": 100000,
                    "limited": False,
                    "nextTickAt": "2026-09-01T09:00:00Z",
                },
                "weeklyTokenLimit": {"creditRemaining": "$50.00"},
            }
        )
        new = json.dumps(
            {
                "rollingFiveHourLimit": {
                    "remaining": 5,
                    "max": 100000,
                    "limited": True,
                    "nextTickAt": "2026-09-01T10:00:00Z",
                },
            }
        )
        _write(
            path,
            [
                _line(dispatch_id="d1", ts=100.0, synthetic_quotas=old),
                _line(
                    dispatch_id="d1",
                    ts=200.0,
                    status=429,
                    upstream_429_body="rate limited",
                    synthetic_quotas=new,
                ),
            ],
        )

        governor, _ = read_feed(path)

        state = governor.providers["d1"]
        assert state.rolling.remaining == 5  # the newer record wins
        assert state.limited is True
        assert state.next_tick_at == "2026-09-01T10:00:00Z"
        # A field the newer record omitted keeps its last known value.
        assert state.weekly.credit_remaining == 50.00
        assert state.last_429_at == 200.0

    def test_providers_tracked_independently(self, tmp_path):
        path = tmp_path / "relay.jsonl"
        _write(
            path,
            [
                _line(dispatch_id="d1", synthetic_quotas=FULL_QUOTAS),
                _line(dispatch_id="d2", status=429, upstream_429_body="over limit"),
            ],
        )

        governor, _ = read_feed(path)

        assert set(governor.providers) == {"d1", "d2"}
        assert governor.providers["d1"].rolling is not None
        assert governor.providers["d2"].last_429_at == 1000.0
        assert governor.providers["d2"].rolling is None


# =========================================================================== #
# Malformed-input tolerance — the reader NEVER raises                          #
# =========================================================================== #
class TestMalformedTolerance:
    def test_malformed_json_line_skipped_others_fold(self, tmp_path):
        path = tmp_path / "relay.jsonl"
        _write(
            path,
            [
                _line(dispatch_id="d1", synthetic_quotas=FULL_QUOTAS),
                "{not json at all",
                "",  # blank line
                "42",  # valid JSON, not an object
                _line(dispatch_id="d2", synthetic_quotas=FULL_QUOTAS),
            ],
        )

        governor, next_offset = read_feed(path)  # must not raise

        assert next_offset == path.stat().st_size
        assert set(governor.providers) == {"d1", "d2"}
        assert governor.providers["d1"].rolling.remaining == 12345

    def test_truncated_record_skipped(self, tmp_path):
        path = tmp_path / "relay.jsonl"
        good = _line(dispatch_id="d1", synthetic_quotas=FULL_QUOTAS)
        truncated = good[: len(good) // 2]  # cut mid-JSON
        _write(path, [truncated, good])

        governor, _ = read_feed(path)  # must not raise

        assert set(governor.providers) == {"d1"}
        assert governor.providers["d1"].rolling.remaining == 12345

    def test_mistyped_quota_fields_skipped(self, tmp_path):
        path = tmp_path / "relay.jsonl"
        mistyped = json.dumps(
            {
                "rollingFiveHourLimit": {
                    "remaining": "a lot",  # not a number
                    "max": 100000,
                    "limited": "yes",  # not a bool
                    "nextTickAt": 12345,  # not a string
                },
                "weeklyTokenLimit": {"creditRemaining": "free"},  # not a dollar string
            }
        )
        good = _line(dispatch_id="d1", synthetic_quotas=mistyped)
        _write(path, [good, _line(dispatch_id="d1", ts=1100.0, synthetic_quotas=FULL_QUOTAS)])

        governor, _ = read_feed(path)  # must not raise

        # The first record's quota data is unusable; the second folds fully.
        assert governor.providers["d1"].rolling.remaining == 12345
        assert governor.providers["d1"].weekly.credit_used == 12.34

    def test_missing_quota_fields_and_no_key_line_skipped(self, tmp_path):
        path = tmp_path / "relay.jsonl"
        _write(
            path,
            [
                _line(),  # no provider key fields at all
                _line(dispatch_id="d1"),  # keyed but no quota data
                _line(dispatch_id="d1", ts=1200.0, synthetic_quotas=FULL_QUOTAS),
            ],
        )

        governor, _ = read_feed(path)

        assert list(governor.providers) == ["d1"]
        assert governor.providers["d1"].updated_at == 1200.0

    def test_missing_file_never_raises(self, tmp_path):
        governor, next_offset = read_feed(tmp_path / "does-not-exist.jsonl")
        assert governor.providers == {}
        assert next_offset == 0

    def test_bad_offset_clamped_never_raises(self, tmp_path):
        path = tmp_path / "relay.jsonl"
        _write(path, [_line(dispatch_id="d1", synthetic_quotas=FULL_QUOTAS)])

        for bad in (-5, "nope", float("nan"), float("inf")):
            governor, next_offset = read_feed(path, bad)  # must not raise
            assert next_offset >= 0

        # A past-end offset clamps into range: nothing to read, no crash.
        governor, next_offset = read_feed(path, path.stat().st_size + 1000)
        assert next_offset >= 0
        assert governor.providers == {}


# =========================================================================== #
# Bounded offset-based re-read                                                #
# =========================================================================== #
class TestBoundedReRead:
    def test_second_read_after_appending_processes_only_appended_lines(self, tmp_path):
        path = tmp_path / "relay.jsonl"
        first = _line(dispatch_id="d1", ts=100.0, synthetic_quotas=FULL_QUOTAS)
        _write(path, [first])

        governor1, offset1 = read_feed(path)
        assert offset1 == path.stat().st_size
        assert governor1.providers["d1"].updated_at == 100.0

        # Append a NEW record for the same provider (and one for a new one).
        second = _line(dispatch_id="d1", ts=200.0, status=429, upstream_429_body="limited")
        third = _line(dispatch_id="d2", ts=200.0, synthetic_quotas=FULL_QUOTAS)
        with path.open("a", encoding="utf-8") as f:
            f.write(second + "\n" + third + "\n")

        # The reader must process ONLY the appended bytes: it seeks to
        # offset1, so `first` is never re-read. If it re-read from 0 the
        # d1.updated_at below would be 100.0, not 200.0.
        governor2, offset2 = read_feed(path, offset1)

        assert offset2 == path.stat().st_size
        d1 = governor2.providers["d1"]
        assert d1.updated_at == 200.0  # only the appended line touched d1
        assert d1.last_429_at == 200.0
        assert governor2.providers["d2"].rolling.remaining == 12345

    def test_partial_line_is_not_consumed_until_complete(self, tmp_path):
        path = tmp_path / "relay.jsonl"
        complete = _line(dispatch_id="d1", ts=100.0, synthetic_quotas=FULL_QUOTAS)
        path.write_text(complete + "\n")

        governor1, offset1 = read_feed(path)
        assert offset1 == path.stat().st_size
        assert offset1 > 0

        # The writer was mid-line: append a partial record (no newline).
        partial = _line(dispatch_id="d1", ts=900.0, upstream_429_body="half")
        with path.open("a", encoding="utf-8") as f:
            f.write(partial)

        governor2, offset2 = read_feed(path, offset1)

        # The partial line must be HELD BACK (not consumed) — the next
        # offset stops before it, so nothing from the partial (ts=900)
        # has been folded into the fresh governor.
        assert offset2 == offset1
        assert governor2.providers == {}

        # Once the line completes, the next read picks it up.
        with path.open("a", encoding="utf-8") as f:
            f.write("\n")
        governor3, offset3 = read_feed(path, offset2)
        assert offset3 == path.stat().st_size
        assert governor3.providers["d1"].last_429_at == 900.0

    def test_repeated_reads_at_final_offset_are_no_ops(self, tmp_path):
        path = tmp_path / "relay.jsonl"
        _write(path, [_line(dispatch_id="d1", synthetic_quotas=FULL_QUOTAS)])
        _governor1, offset1 = read_feed(path)

        # Nothing appended -> the seek-at-end read processes ZERO lines.
        governor2, offset2 = read_feed(path, offset1)

        assert offset2 == offset1
        assert governor2.providers == {}

    def test_no_history_list_grows(self, tmp_path):
        # Bounded-by-construction: after N folded lines the governor holds
        # exactly one QuotaState per provider — memory scales with the
        # provider count, not the line count.
        path = tmp_path / "relay.jsonl"
        lines = [
            _line(dispatch_id=f"d{i % 3}", ts=float(i), synthetic_quotas=FULL_QUOTAS)
            for i in range(500)
        ]
        _write(path, lines)

        governor, _ = read_feed(path)

        assert len(governor.providers) == 3
        # Only current state per key — the latest ts won for each provider.
        assert governor.providers["d0"].updated_at == 498.0
        assert governor.providers["d1"].updated_at == 499.0
        assert governor.providers["d2"].updated_at == 497.0


# =========================================================================== #
# Direct fold API                                                             #
# =========================================================================== #
class TestFoldDirect:
    def test_governor_fold_returns_change_flag(self):
        g = QuotaGovernor()
        assert g.fold({"dispatch_id": "d1", "synthetic_quotas": FULL_QUOTAS}) is True
        assert g.fold({"no_key_here": 1}) is False  # no provider key
        assert g.fold({"dispatch_id": "d1"}) is False  # no quota data

    def test_fold_never_raises_on_garbage(self):
        g = QuotaGovernor()
        assert g.fold({"dispatch_id": "d1", "ts": 1.0, "synthetic_quotas": FULL_QUOTAS})
        for junk in (
            {"dispatch_id": "d1", "synthetic_quotas": "{\ntruncated"},
            {"dispatch_id": "d1", "synthetic_quotas": 12345},  # not a string
            {"dispatch_id": 99, "synthetic_quotas": FULL_QUOTAS},  # non-str key
            {"dispatch_id": "d1", "ts": "not-a-number"},
        ):
            g.fold(junk)  # must not raise
        assert "d1" in g.providers


# =========================================================================== #
# Decision logic — pure park/unpark/escalation state machine over feed state  #
# =========================================================================== #
T0 = 1_700_000_000.0  # an arbitrary decision epoch


def _ts(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def _iso_at(offset_s: float) -> str:
    """The ISO-8601 UTC timestamp exactly ``offset_s`` seconds after T0."""
    return datetime.fromtimestamp(T0 + offset_s, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _state(**kw) -> QuotaState:
    base = dict(provider="prov-a")
    base.update(kw)
    return QuotaState(**base)


class TestDecideNonSubscription:
    """AC2: metered/local lanes are ALWAYS dispatch — no park branch."""

    def test_metered_always_dispatch_regardless_of_state(self):
        # Fully exhausted-looking state: limited flag, upstream 429, a
        # distant next tick — none of it can park a metered lane.
        state = _state(
            limited=True,
            last_429_at=T0 - 10.0,
            next_tick_at="2100-01-01T00:00:00Z",
            rolling=RollingWindow(
                remaining=0, max=100, limited=True, next_tick_at="2100-01-01T00:00:00Z"
            ),
        )
        for now in (T0, T0 + 10 * 3600, T0 + 365 * 86400):
            d = decide(PricingClass.METERED, state, now=now)
            assert d == GovernorDecision(decision="dispatch")

    def test_local_always_dispatch_regardless_of_state(self):
        state = _state(limited=True, last_429_at=T0, next_tick_at="2100-01-01T00:00:00Z")
        for now in (T0, T0 + 10 * 3600):
            d = decide(PricingClass.LOCAL, state, now=now)
            assert d.decision == "dispatch"
            assert d.park_until is None
            assert d.escalation_due is False

    def test_non_subscription_dispatches_even_without_any_state(self):
        assert decide(PricingClass.METERED, None, now=T0).decision == "dispatch"
        assert decide(PricingClass.LOCAL, None, now=T0).decision == "dispatch"


class TestDecideParkOnExhaustedSubscription:
    def test_limited_flag_parks_until_next_tick(self):
        tick = _iso_at(3600)  # exactly T0 + 1h
        state = _state(limited=True, next_tick_at=tick)
        d = decide(PricingClass.SUBSCRIPTION, state, now=T0)
        assert d.decision == "park-until"
        assert d.park_until == _ts(tick)
        assert d.escalation_due is False

    def test_upstream_429_record_parks_until_next_tick(self):
        tick = _iso_at(3600)
        state = _state(limited=False, last_429_at=T0 - 5.0, next_tick_at=tick)
        d = decide(PricingClass.SUBSCRIPTION, state, now=T0)
        assert d.decision == "park-until"
        assert d.park_until == _ts(tick)

    def test_exhausted_without_computable_tick_still_parks(self):
        # 429 evidence, no nextTickAt anywhere: parked (end unknown — no
        # escalation can be due on an unknown deadline).
        d = decide(PricingClass.SUBSCRIPTION, _state(last_429_at=T0), now=T0)
        assert d.decision == "park-until"
        assert d.park_until is None
        assert d.escalation_due is False


class TestDecideUnpark:
    def test_no_evidence_dispatches(self):
        healthy = _state(limited=False, next_tick_at=_iso_at(3600))
        d = decide(PricingClass.SUBSCRIPTION, healthy, now=T0)
        assert d.decision == "dispatch"
        assert d.park_until is None

    def test_unpark_when_limited_clears(self):
        tick = _iso_at(3600)
        exhausted = _state(limited=True, next_tick_at=tick)
        assert decide(PricingClass.SUBSCRIPTION, exhausted, now=T0).decision.startswith("park")
        # The nextTick arrives and reports limited cleared: the same state
        # shape with limited=False dispatches again.
        cleared = _state(limited=False, next_tick_at=tick)
        d = decide(PricingClass.SUBSCRIPTION, cleared, now=_ts(tick))
        assert d.decision == "dispatch"
        assert d.park_until is None

    def test_unpark_at_next_tick_time(self):
        # A 429 with no rolling object sets no nextTickAt: parked.
        parked = _state(last_429_at=T0)
        assert decide(PricingClass.SUBSCRIPTION, parked, now=T0).decision == "park-until"
        # Once the rolling window re-reports (tick passed, limited=False)
        # the latest report wins and the lane dispatches.
        recovered = _state(limited=False, next_tick_at=_iso_at(86400))
        d = decide(PricingClass.SUBSCRIPTION, recovered, now=_ts(_iso_at(86400)))
        assert d.decision == "dispatch"

    def test_never_reported_provider_dispatches(self):
        d = decide(PricingClass.SUBSCRIPTION, None, now=T0)
        assert d.decision == "dispatch"


class TestDecideEscalation:
    def test_escalation_due_once_wait_passes_ceiling_park_persists(self):
        # Next tick 12 hours out. At T0 the wait (12h) already exceeds the
        # default 8h ceiling; at T0 + 3h it (9h) still does: the decision
        # stays parked at the SAME park_until and carries the
        # escalation-due signal — escalation never clears the park.
        tick = _iso_at(12 * 3600)
        state = _state(limited=True, next_tick_at=tick)

        for now in (T0, T0 + 3 * 3600):
            d = decide(PricingClass.SUBSCRIPTION, state, now=now)
            assert d.decision == "park-escalation-due"
            assert d.escalation_due is True
            assert d.park_until == _ts(tick)

        # 7 hours later the remaining wait (5h) is back under the ceiling:
        # still parked, same deadline, but no longer escalation-due.
        under = decide(PricingClass.SUBSCRIPTION, state, now=T0 + 7 * 3600)
        assert under.decision == "park-until"
        assert under.escalation_due is False
        assert under.park_until == _ts(tick)

    def test_escalation_fires_exactly_at_the_ceiling_crossing(self):
        # The wait from `now` to the tick shrinks as time passes, so the
        # escalation-due signal fires exactly once per park: from the
        # moment of parking until the remaining wait reaches the ceiling.
        # With the tick 9h out and the ceiling 8h, the crossing is at
        # park_until - ceiling: due just before, not due at or after
        # (the wait then no longer EXCEEDS the ceiling) — while the park
        # persists throughout at the identical park_until.
        tick = _iso_at(9 * 3600)
        state = _state(limited=True, next_tick_at=tick)
        park_until = _ts(tick)
        boundary = park_until - DEFAULT_ESCALATION_CEILING_S

        just_before = decide(PricingClass.SUBSCRIPTION, state, now=boundary - 1.0)
        assert just_before.decision == "park-escalation-due"
        assert just_before.escalation_due is True
        assert just_before.park_until == park_until

        at = decide(PricingClass.SUBSCRIPTION, state, now=boundary)
        assert at.decision == "park-until"
        assert at.escalation_due is False

        just_after = decide(PricingClass.SUBSCRIPTION, state, now=boundary + 1.0)
        assert just_after.decision == "park-until"
        assert just_after.escalation_due is False

    def test_caller_supplied_ceiling(self):
        tick = _iso_at(2 * 3600)  # T0 + 2h
        state = _state(limited=True, next_tick_at=tick)
        # Default 8h ceiling: a 2h wait is well under it.
        assert decide(PricingClass.SUBSCRIPTION, state, now=T0).decision == "park-until"
        # Caller lowers the ceiling to 1h: the same 2h wait now escalates.
        d = decide(PricingClass.SUBSCRIPTION, state, now=T0, escalation_ceiling_s=3600.0)
        assert d.decision == "park-escalation-due"
        assert d.escalation_due is True
        assert d.park_until == _ts(tick)

    def test_default_ceiling_is_eight_hours(self):
        assert DEFAULT_ESCALATION_CEILING_S == 8 * 3600.0


class TestDecideCapabilityDenyNeverParks:
    """AC3: a per-dispatch capability deny (x-stigmergy-deny-reason marker)
    is never transformed into park evidence anywhere in quotagov."""

    def test_deny_reason_on_record_never_parks(self):
        state = _state(last_deny_reason="quota-tokens")
        for now in (T0, T0 + 100 * 3600.0):
            d = decide(PricingClass.SUBSCRIPTION, state, now=now)
            assert d.decision == "dispatch"

    def test_deny_only_state_never_parks(self):
        # Even a state whose ONLY content is a capability deny: dispatch.
        state = QuotaState(provider="prov-a", last_deny_reason="quota-calls")
        d = decide(PricingClass.SUBSCRIPTION, state, now=T0)
        assert d.decision == "dispatch"
        assert d.park_until is None
        assert d.escalation_due is False

    def test_feed_fold_of_deny_record_never_creates_park_evidence(self, tmp_path):
        # The end-to-end shape: a relay deny rides a 429 status with the
        # x-stigmergy-deny-reason marker. Folding it must NOT set
        # `limited` or `last_429_at`, and the decision on the folded state
        # must be dispatch.
        deny_body = json.dumps(
            {"error": {"type": "stigmergy_relay_deny", "reason": "quota-calls"}}
        )
        line = json.dumps(
            {
                "ts": 500.0,
                "dispatch_id": "d1",
                "decision": "deny",
                "reason": "quota-calls",
                "status": 429,
                "upstream_429_body": deny_body,
                "x-stigmergy-deny-reason": "quota-calls",
            }
        )
        path = tmp_path / "relay.jsonl"
        path.write_text(line + "\n")

        governor, _ = read_feed(path)
        state = governor.providers["d1"]
        assert state.limited is False
        assert state.last_429_at is None  # NOT park evidence
        assert state.last_deny_reason == "quota-calls"  # recorded, policy-only

        d = decide(PricingClass.SUBSCRIPTION, state, now=500.0)
        assert d.decision == "dispatch"

    def test_upstream_429_still_parks_after_a_deny_record(self, tmp_path):
        # A deny record never CLEARS real evidence either: a genuine
        # upstream 429 (unmarked body) folded before/after a deny record
        # still parks.
        good = _line(dispatch_id="d1", ts=100.0, status=429, upstream_429_body="over limit")
        deny = json.dumps(
            {
                "ts": 200.0,
                "dispatch_id": "d1",
                "decision": "deny",
                "reason": "quota-calls",
                "status": 429,
                "x-stigmergy-deny-reason": "quota-calls",
            }
        )
        path = tmp_path / "relay.jsonl"
        path.write_text(good + "\n" + deny + "\n")

        governor, _ = read_feed(path)
        state = governor.providers["d1"]
        assert state.last_429_at == 100.0  # the upstream 429 evidence survives
        assert state.last_deny_reason == "quota-calls"
        d = decide(PricingClass.SUBSCRIPTION, state, now=200.0)
        assert d.decision == "park-until"

    def test_real_429_body_without_deny_marker_still_parks(self):
        # The inverse of the guard: an unmarked 429 record IS park evidence.
        tick = _iso_at(3600)
        state = _state(last_429_at=T0, next_tick_at=tick)
        d = decide(PricingClass.SUBSCRIPTION, state, now=T0)
        assert d.decision == "park-until"
        assert d.park_until == _ts(tick)


class TestDecidePricingClassResolution:
    def test_model_entry_resolves_to_its_pricing_class(self):
        tick = _iso_at(3600)
        state = _state(limited=True, next_tick_at=tick)
        for pricing, expected in (
            (PricingClass.SUBSCRIPTION, "park-until"),
            (PricingClass.METERED, "dispatch"),
            (PricingClass.LOCAL, "dispatch"),
        ):
            entry = ModelEntry(
                name="m",
                provider="p",
                family="f",
                version="1",
                pricing=pricing,
            )
            d = decide(entry, state, now=T0)
            assert d.decision == expected

    def test_pricing_input_must_be_class_or_entry(self):
        with pytest.raises(TypeError):
            decide("subscription", _state(limited=True), now=T0)
