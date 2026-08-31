"""Tests for stigmergy.intake + stigmergy.approval (SPEC.md §3 `intake`
station, §4 "Approval integrity", §6 test-authorship/red-at-birth, §9 state
machine + leases, §10 AC2/AC11).

Case numbering below matches the bead .15 exact case list:

AC2  (1-4)  — only approved+hash-valid+DAG-eligible tickets are ever claimable.
AC11 (5-7)  — approval-hash integrity: steering mutation de-eligibilizes;
              claim-time snapshot is frozen; execution re-hash needs no human.
Leases (8-11) — claim/heartbeat/expire mechanics, no double-claim.
Red-at-birth (12) — pre-ticket-tree acceptance-test outcome gates admission.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from stigmergy.approval import (
    approve,
    is_approval_valid,
    rehash_execution,
    snapshot_hash,
    steering_hash,
)
from stigmergy.checks import CheckOutcome
from stigmergy.intake import (
    LeaseError,
    claim,
    eligible,
    expire_leases,
    heartbeat,
    red_at_birth_ok,
)
from stigmergy.rig import RigStore

# --- fixtures / helpers -----------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> RigStore:
    s = RigStore.create(tmp_path / "tickets.db")
    yield s
    s.close()


def base_steering(**overrides: Any) -> dict[str, Any]:
    """A complete steering-field dict (SPEC §4): ticket text, checks,
    rubric, lane, prompt bytes, context set."""
    steering: dict[str, Any] = {
        "ticket_text": "Implement foo() to spec.",
        "checks": {"named": ["pytest", "lint"], "paths": ["tests/test_foo.py"]},
        "rubric": ["foo() returns 42", "foo() raises on negative input"],
        "lane": "cheap",
        "prompt_bytes": "code01-prompt-v1",
        "context_set": ["context/architecture.md", "repo/src/foo.py"],
    }
    steering.update(overrides)
    return steering


def base_execution(**overrides: Any) -> dict[str, Any]:
    """A complete execution-field dict (SPEC §4): base OID, resolved
    model+version, image digest, egress policy."""
    execution: dict[str, Any] = {
        "base_oid": "abc123def456",
        "model": "haiku",
        "model_version": "haiku-3-5-20241022",
        "image_digest": "sha256:deadbeef",
        "egress_policy": ["inference", "registries"],
    }
    execution.update(overrides)
    return execution


def add_approved_ticket(
    store: RigStore,
    ticket_id: str,
    *,
    steering: dict[str, Any] | None = None,
    state: str = "pool",
) -> dict[str, Any]:
    """Add a ticket and approve it against ``steering`` (or a default)."""
    steering = base_steering() if steering is None else steering
    store.add_ticket(id=ticket_id, title=f"Ticket {ticket_id}", state=state)
    approve(store, ticket_id, steering=steering)
    return steering


def steering_lookup(mapping: dict[str, dict[str, Any]]):
    """Build a `steering_of` callable backed by a plain dict the test controls."""

    def _steering_of(ticket_id: str) -> dict[str, Any]:
        return mapping[ticket_id]

    return _steering_of


# --- AC2 case 1: unapproved ticket never eligible, across N poll cycles ----


def test_unapproved_ticket_not_eligible_across_n_poll_cycles(store: RigStore) -> None:
    steering = base_steering()
    store.add_ticket(id="t-unapproved", title="Unapproved")  # approved=0 by default
    steering_of = steering_lookup({"t-unapproved": steering})

    for cycle in range(5):
        result = eligible(store, now=1000.0 + cycle, steering_of=steering_of)
        assert "t-unapproved" not in result

    # eligible() is a pure query: repeated polling must not have mutated
    # the ticket's approval/lease state at all.
    row = store.get_ticket("t-unapproved")
    assert row["approved"] == 0
    assert row["lease_owner"] is None
    assert row["state"] == "pool"


# --- AC2 case 2: approved but hash-stale (steering changed since approval) -


def test_approved_but_hash_stale_ticket_not_eligible(store: RigStore) -> None:
    original_steering = base_steering()
    add_approved_ticket(store, "t-stale", steering=original_steering)

    mutated_steering = base_steering(rubric=["a different rubric item entirely"])
    steering_of = steering_lookup({"t-stale": mutated_steering})

    result = eligible(store, now=1000.0, steering_of=steering_of)
    assert "t-stale" not in result

    row = store.get_ticket("t-stale")
    assert row["approved"] == 1
    assert row["approval_hash"] == steering_hash(original_steering)
    assert row["approval_hash"] != steering_hash(mutated_steering)


# --- AC2 case 3: unlanded predecessor blocks eligibility; landing unblocks -


def test_ticket_with_unlanded_predecessor_not_eligible_then_becomes_eligible(
    store: RigStore,
) -> None:
    store.add_ticket(id="t-pred", title="Predecessor", state="pool")
    dep_steering = add_approved_ticket(store, "t-dep", steering=base_steering(lane="default"))
    store.add_dep("t-dep", "t-pred")  # t-dep is blocked-by (must land after) t-pred

    steering_of = steering_lookup({"t-dep": dep_steering})

    # Predecessor not landed -> blocked.
    result = eligible(store, now=1000.0, steering_of=steering_of)
    assert "t-dep" not in result

    # Land the predecessor -> now eligible.
    store.update_ticket("t-pred", state="landed")
    result = eligible(store, now=1001.0, steering_of=steering_of)
    assert "t-dep" in result


# --- AC2 case 4: approved + hash-valid + all-deps-landed IS eligible --------


def test_fully_eligible_ticket_is_eligible(store: RigStore) -> None:
    store.add_ticket(id="t-base", title="Base", state="landed")
    steering = add_approved_ticket(store, "t-ready")
    store.add_dep("t-ready", "t-base")
    steering_of = steering_lookup({"t-ready": steering})

    result = eligible(store, now=1000.0, steering_of=steering_of)
    assert result == ["t-ready"]


# --- AC11 case 5: mutate steering -> invalid + drops out; re-approve fixes -


def test_mutating_steering_invalidates_approval_and_reapproval_restores_it(
    store: RigStore,
) -> None:
    original_steering = base_steering()
    store.add_ticket(id="t-reapprove", title="Reapprove me", state="pool")
    approve(store, "t-reapprove", steering=original_steering)

    row = store.get_ticket("t-reapprove")
    assert is_approval_valid(row, original_steering) is True

    changed_steering = base_steering(rubric=["a mutated rubric item"])
    row = store.get_ticket("t-reapprove")
    assert is_approval_valid(row, changed_steering) is False

    steering_of = steering_lookup({"t-reapprove": changed_steering})
    assert "t-reapprove" not in eligible(store, now=1000.0, steering_of=steering_of)

    # Re-approve with the NEW steering -> eligibility restored.
    approve(store, "t-reapprove", steering=changed_steering)
    row = store.get_ticket("t-reapprove")
    assert is_approval_valid(row, changed_steering) is True
    assert "t-reapprove" in eligible(store, now=1001.0, steering_of=steering_of)


# --- AC11 case 6: claim() snapshot is frozen against later row mutation ----


def test_claim_snapshot_is_frozen_against_later_store_mutation(store: RigStore) -> None:
    steering = base_steering()
    execution = base_execution()
    store.add_ticket(id="t-freeze", title="Freeze me", state="pool")
    approve(store, "t-freeze", steering=steering)

    expected_steering_hash = steering_hash(steering)
    expected_snapshot_hash = snapshot_hash(steering, execution)

    snapshot = claim(
        store,
        "t-freeze",
        owner="worker-1",
        dispatch_id="dispatch-0001",
        ttl_seconds=3600,
        now=1000.0,
        steering=steering,
        execution=execution,
    )
    assert snapshot["steering_hash"] == expected_steering_hash
    assert snapshot["snapshot_hash"] == expected_snapshot_hash

    # Mutate the ticket's steering in the store: re-approve under a wholly
    # different steering (this changes the row's approval_hash/state).
    new_steering = base_steering(rubric=["a completely different rubric"])
    approve(store, "t-freeze", steering=new_steering)
    row_after = store.get_ticket("t-freeze")
    assert row_after["approval_hash"] == steering_hash(new_steering)
    assert row_after["approval_hash"] != expected_steering_hash

    # Also mutate the very dict object passed into claim(), in place.
    steering["rubric"] = ["mutated after the fact"]
    execution["base_oid"] = "mutated-oid"

    # The previously-returned snapshot must be completely unaffected.
    assert snapshot["steering_hash"] == expected_steering_hash
    assert snapshot["snapshot_hash"] == expected_snapshot_hash
    assert snapshot["steering"]["rubric"] != ["mutated after the fact"]
    assert snapshot["execution"]["base_oid"] != "mutated-oid"


# --- AC11 case 7: rehash_execution keeps steering_hash, changes snapshot ---


def test_rehash_execution_keeps_steering_hash_but_changes_snapshot_hash(store: RigStore) -> None:
    steering = base_steering()
    execution = base_execution()

    before = rehash_execution(steering, execution)

    new_execution = base_execution(base_oid="freshoid789")
    after = rehash_execution(steering, new_execution)

    assert after["steering_hash"] == before["steering_hash"]
    assert after["steering_hash"] == steering_hash(steering)
    assert after["snapshot_hash"] != before["snapshot_hash"]


# --- Lease case 8: claim() sets lease fields + state='claimed' -------------


def test_claim_sets_lease_fields_and_state(store: RigStore) -> None:
    steering = base_steering()
    execution = base_execution()
    store.add_ticket(id="t-claim", title="Claim me", state="pool")
    approve(store, "t-claim", steering=steering)

    claim(
        store,
        "t-claim",
        owner="worker-7",
        dispatch_id="dispatch-xyz",
        ttl_seconds=4500,
        now=2000.0,
        steering=steering,
        execution=execution,
    )

    row = store.get_ticket("t-claim")
    assert row["lease_owner"] == "worker-7"
    assert row["lease_dispatch_id"] == "dispatch-xyz"
    assert row["lease_expires_at"] == 2000.0 + 4500
    assert row["lease_heartbeat_at"] == 2000.0
    assert row["state"] == "claimed"


# --- Lease case 9: no double-claim under a live lease ----------------------


def test_claim_on_live_leased_ticket_raises_lease_error(store: RigStore) -> None:
    steering = base_steering()
    execution = base_execution()
    store.add_ticket(id="t-double", title="No double claim", state="pool")
    approve(store, "t-double", steering=steering)

    claim(
        store,
        "t-double",
        owner="worker-a",
        dispatch_id="dispatch-a",
        ttl_seconds=4500,
        now=2000.0,
        steering=steering,
        execution=execution,
    )

    with pytest.raises(LeaseError):
        claim(
            store,
            "t-double",
            owner="worker-b",
            dispatch_id="dispatch-b",
            ttl_seconds=4500,
            now=2100.0,  # still well inside the first lease's TTL
            steering=steering,
            execution=execution,
        )

    # The original lease is undisturbed by the failed second claim.
    row = store.get_ticket("t-double")
    assert row["lease_owner"] == "worker-a"
    assert row["lease_dispatch_id"] == "dispatch-a"


# --- Lease case 10: expire_leases resets to pool, spares lifetime counters -


def test_expire_leases_resets_orphan_and_spares_lifetime_counters(store: RigStore) -> None:
    steering = base_steering()
    execution = base_execution()
    store.add_ticket(
        id="t-orphan",
        title="Orphan",
        state="pool",
        attempts_used=2,
        integration_failures=1,
    )
    approve(store, "t-orphan", steering=steering)

    claim(
        store,
        "t-orphan",
        owner="worker-9",
        dispatch_id="dispatch-9",
        ttl_seconds=100,
        now=3000.0,
        steering=steering,
        execution=execution,
    )
    # Lease expires at 3100.0; poll at a time strictly after that.
    expired_ids = expire_leases(store, now=3200.0)

    assert expired_ids == ["t-orphan"]

    row = store.get_ticket("t-orphan")
    assert row["state"] == "pool"
    assert row["lease_owner"] is None
    assert row["lease_dispatch_id"] is None
    assert row["lease_expires_at"] is None
    assert row["lease_heartbeat_at"] is None
    # Lifetime counters MUST survive lease expiry untouched (SPEC §9).
    assert row["attempts_used"] == 2
    assert row["integration_failures"] == 1


def test_expire_leases_never_demotes_non_dispatch_states(store: RigStore) -> None:
    """Lease case 10b (caught live on quotagov01 2026-08-31): a daemon
    restart ran recover() -> expire_leases, which demoted LANDED and
    ESCALATED tickets to pool because their rows still carried stale
    lease columns from their dispatch days. Only a ticket in an
    active-dispatch state (`claimed`/`in_flight`) may be lease-expired:
    terminal states are final, and ESCALATED re-entry is the operator's
    `resume` verb — never a silent expiry.
    """
    steering = base_steering()
    execution = base_execution()

    # A landed ticket whose land path left stale lease columns behind.
    store.add_ticket(id="t-landed", title="Landed", state="pool")
    approve(store, "t-landed", steering=steering)
    claim(
        store,
        "t-landed",
        owner="worker-old",
        dispatch_id="dispatch-old",
        ttl_seconds=100,
        now=3000.0,
        steering=steering,
        execution=execution,
    )
    store.update_ticket("t-landed", state="landed")

    # An escalated ticket with a stale lease from the same dispatch.
    store.add_ticket(id="t-escalated", title="Escalated", state="pool")
    approve(store, "t-escalated", steering=steering)
    claim(
        store,
        "t-escalated",
        owner="worker-old2",
        dispatch_id="dispatch-old2",
        ttl_seconds=100,
        now=3000.0,
        steering=steering,
        execution=execution,
    )
    store.update_ticket("t-escalated", state="escalated")

    # A genuinely orphaned in-flight dispatch — the case expiry EXISTS for.
    store.add_ticket(id="t-orphan", title="Orphan", state="pool")
    approve(store, "t-orphan", steering=steering)
    claim(
        store,
        "t-orphan",
        owner="worker-9",
        dispatch_id="dispatch-9",
        ttl_seconds=100,
        now=3000.0,
        steering=steering,
        execution=execution,
    )

    expired_ids = expire_leases(store, now=3200.0)

    assert expired_ids == ["t-orphan"]
    assert store.get_ticket("t-landed")["state"] == "landed"
    assert store.get_ticket("t-landed")["lease_owner"] is not None  # untouched
    assert store.get_ticket("t-escalated")["state"] == "escalated"
    assert store.get_ticket("t-escalated")["lease_owner"] is not None  # untouched
    assert store.get_ticket("t-orphan")["state"] == "pool"


def test_expire_leases_ignores_tickets_with_no_lease_or_unexpired_lease(store: RigStore) -> None:
    steering = base_steering()
    execution = base_execution()
    store.add_ticket(id="t-no-lease", title="No lease", state="pool")
    approve(store, "t-no-lease", steering=steering)

    store.add_ticket(id="t-fresh-lease", title="Fresh lease", state="pool")
    approve(store, "t-fresh-lease", steering=steering)
    claim(
        store,
        "t-fresh-lease",
        owner="worker-x",
        dispatch_id="dispatch-x",
        ttl_seconds=10000,
        now=4000.0,
        steering=steering,
        execution=execution,
    )

    expired_ids = expire_leases(store, now=4500.0)  # nowhere near t-fresh-lease's expiry
    assert expired_ids == []


# --- Lease case 11: heartbeat extends lease_expires_at ----------------------


def test_heartbeat_extends_lease_expiry(store: RigStore) -> None:
    steering = base_steering()
    execution = base_execution()
    store.add_ticket(id="t-heartbeat", title="Heartbeat", state="pool")
    approve(store, "t-heartbeat", steering=steering)

    claim(
        store,
        "t-heartbeat",
        owner="worker-hb",
        dispatch_id="dispatch-hb",
        ttl_seconds=1000,
        now=5000.0,
        steering=steering,
        execution=execution,
    )
    row = store.get_ticket("t-heartbeat")
    assert row["lease_expires_at"] == 6000.0

    heartbeat(store, "t-heartbeat", now=5900.0, ttl_seconds=1000)

    row = store.get_ticket("t-heartbeat")
    assert row["lease_expires_at"] == 6900.0
    assert row["lease_heartbeat_at"] == 5900.0
    # heartbeat must not touch state or ownership.
    assert row["state"] == "claimed"
    assert row["lease_owner"] == "worker-hb"


# --- Red-at-birth case 12 ----------------------------------------------------


def test_red_at_birth_ok_true_when_failed_false_when_passed() -> None:
    # Acceptance tests FAILED against the pre-ticket tree -> red -> admit.
    assert red_at_birth_ok(CheckOutcome.FAIL) is True
    # Acceptance tests PASSED against the pre-ticket tree -> vacuous -> reject.
    assert red_at_birth_ok(CheckOutcome.PASS) is False


def test_red_at_birth_ok_accepts_bool_and_exit_code_conventions() -> None:
    # bool convention: True == "it passed" == vacuous == reject.
    assert red_at_birth_ok(False) is True
    assert red_at_birth_ok(True) is False
    # exit-code convention: 0 == passed == vacuous == reject.
    assert red_at_birth_ok(1) is True
    assert red_at_birth_ok(0) is False


def test_red_at_birth_ok_flaky_and_error_are_not_vacuous() -> None:
    # Neither FLAKY (passed on rerun) nor ERROR (couldn't determine outcome)
    # proves the test was already green -> admission control fails closed
    # to "admit" rather than silently rejecting on an ambiguous signal.
    assert red_at_birth_ok(CheckOutcome.FLAKY) is True
    assert red_at_birth_ok(CheckOutcome.ERROR) is True


# --- Concurrency: deterministic race test for concurrent claims ---------------


def test_concurrent_claim_on_same_ticket_exactly_one_wins(tmp_path: Path) -> None:
    """Deterministic race test: K>=3 threads released simultaneously via a
    barrier, all calling claim() on the SAME ticket. Exactly one thread's
    claim() must return a snapshot; every other must raise LeaseError. The
    ticket's final persisted lease_owner/lease_dispatch_id/state must
    match ONLY the single winner (never a mix, never left unclaimed).

    This test is constructed to force real read/write overlap (barrier),
    so it fails against a naive check-then-set implementation without
    atomicity. Test repeats N>=20 iterations (fresh ticket state each
    iteration) for determinism.
    """
    import threading

    k_threads = 5
    n_iterations = 20

    for iteration in range(n_iterations):
        # Fresh store and ticket each iteration.
        store = RigStore.create(tmp_path / f"test_concurrent_{iteration}.db")
        ticket_id = f"t-race-{iteration}"
        store.add_ticket(id=ticket_id, title="Race me", state="pool")
        steering = base_steering()
        execute = base_execution()
        approve(store, ticket_id, steering=steering)

        barrier = threading.Barrier(k_threads)
        results: dict[int, Exception | dict[str, Any]] = {}

        def claim_in_thread(
            thread_idx: int,
            barrier: threading.Barrier = barrier,
            store: RigStore = store,
            ticket_id: str = ticket_id,
            steering: dict[str, Any] = steering,
            execute: dict[str, Any] = execute,
            results: dict[int, Exception | dict[str, Any]] = results,
        ) -> None:
            # Synchronize all threads to release simultaneously.
            barrier.wait()
            try:
                snapshot = claim(
                    store,
                    ticket_id,
                    owner=f"worker-{thread_idx}",
                    dispatch_id=f"dispatch-{thread_idx}",
                    ttl_seconds=3600,
                    now=1000.0 + thread_idx,  # slight variation, doesn't matter
                    steering=steering,
                    execution=execute,
                )
                results[thread_idx] = snapshot
            except LeaseError as e:
                results[thread_idx] = e

        threads = [threading.Thread(target=claim_in_thread, args=(i,)) for i in range(k_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly one thread should have won (returned a snapshot).
        winners = [i for i in range(k_threads) if isinstance(results[i], dict)]
        losers = [i for i in range(k_threads) if isinstance(results[i], LeaseError)]

        assert len(winners) == 1, f"Iteration {iteration}: expected 1 winner, got {len(winners)}"
        assert len(losers) == k_threads - 1, (
            f"Iteration {iteration}: expected {k_threads - 1} losers, got {len(losers)}"
        )

        winner_idx = winners[0]

        # Verify the ticket's final persisted state matches ONLY the winner.
        final_row = store.get_ticket(ticket_id)
        assert final_row["lease_owner"] == f"worker-{winner_idx}"
        assert final_row["lease_dispatch_id"] == f"dispatch-{winner_idx}"
        assert final_row["state"] == "claimed"
        # Every other thread's owner/dispatch_id must NOT appear.
        for loser_idx in losers:
            assert final_row["lease_owner"] != f"worker-{loser_idx}"
            assert final_row["lease_dispatch_id"] != f"dispatch-{loser_idx}"

        store.close()
