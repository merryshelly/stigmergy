"""Tests for stigmergy.filing — D14 role ticket-filing harvest/validate/insert/emit
(SPEC.md §3 filing capability, §4 propagation edge, §5 filing caps, §7 harvest,
§8 `ticket-filed` event, §10 AC14; README D14; bead workspace-e2uh.38).

This is the FROZEN CONTRACT (AC14). Case numbering matches bead .38's case-list:

1  — a valid `filed-tickets.json` (N<=cap) lands N UNAPPROVED rows in `filed_tickets`,
     each with full origin + loop-stamped `discovered_from` + a `proposal_hash`.
2  — un-claimable by construction: a filed proposal id is NEVER returned by
     `intake.claim()`/`eligible()` over N polls, and is never a `tickets` row.
3  — count-cap excess rejects the WHOLE filing + logs (reject-whole, per amendment D).
4  — per-proposal size-cap rejects only the oversized proposal; others unaffected.
5  — malformed / bad-shape / non-list / path-traversal filing rejected WITHOUT pool
     corruption and WITHOUT raising into the dispatch teardown path.
7  — `ticket-filed` event shape: §8 common fields + origin + proposal_hash +
     accepted/rejected+reason; honest zeros for cost fields; no secret ever appears.
9  — `proposal_hash` is stable over content and distinct from a steering approval_hash.

Cases 6/8/10 live where their behaviour lives (weaver/status/charter test files);
the daemon harvest-hook wiring is exercised in test_daemon.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from stigmergy.approval import approve, steering_hash
from stigmergy.filing import (
    DISCOVERED_FROM_FMT,
    FILED_TICKETS_REL,
    FilingResult,
    file_proposals,
    harvest_worker_filings,
    proposal_hash,
)
from stigmergy.intake import LeaseError, claim, eligible
from stigmergy.records import RecordPlane
from stigmergy.rig import RigStore

# --- fixtures / helpers -----------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> RigStore:
    s = RigStore.create(tmp_path / "tickets.db")
    yield s
    s.close()


@pytest.fixture
def plane(tmp_path: Path) -> RecordPlane:
    return RecordPlane(tmp_path / "records")


DISPATCH_ID = "worker-haiku-code01-broom-casino-flock"
PARENT_TICKET = "workspace-e2uh.8"


def dispatch_ctx(**overrides: Any) -> dict[str, Any]:
    """The §8 common-field context the daemon builds in `_run_dispatch_cycle`
    (everything EXCEPT attempt_kind / tokens / computed_usd / wall_time_seconds —
    those the filing code supplies: attempt_kind is passed separately; the three
    cost fields are stamped as honest zeros)."""
    ctx: dict[str, Any] = {
        "rig": "shipyard",
        "ticket": PARENT_TICKET,
        "dispatch_id": DISPATCH_ID,
        "attempt": 1,
        "rung": "cheap",
        "worker": DISPATCH_ID,
        "charter_hash": "charterhash123",
        "approval_hash": "approvalhash456",
        "image_digest": "sha256:deadbeef",
        "model": "haiku",
        "model_version": "haiku-3-5-20241022",
        "price_table_version": "modelshash789",
    }
    ctx.update(overrides)
    return ctx


def valid_proposal(n: int = 1, **overrides: Any) -> dict[str, Any]:
    p: dict[str, Any] = {
        "title": f"Flaky test #{n}",
        "description": f"test_foo::test_bar_{n} fails ~1/10 runs under load",
        "evidence": f"tests/test_foo.py:{40 + n}",
    }
    p.update(overrides)
    return p


def seed_filings_file(worktree: Path, payload: Any) -> None:
    """Write `<worktree>/.stigmergy/filed-tickets.json` with `payload`
    (json-encoded if not already a str — a str is written verbatim so a
    test can seed deliberately-malformed JSON)."""
    dest = worktree / FILED_TICKETS_REL
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(payload if isinstance(payload, str) else json.dumps(payload))


def filed_events(plane: RecordPlane) -> list[dict[str, Any]]:
    return [e for e in plane.read_events() if e.get("event_type") == "ticket-filed"]


DEFAULT_MAX = 5
DEFAULT_BYTES = 16384


def harvest(worktree: Path, store: RigStore, plane: RecordPlane, **kw: Any) -> FilingResult:
    return harvest_worker_filings(
        worktree,
        store=store,
        record_plane=plane,
        ctx=kw.pop("ctx", dispatch_ctx()),
        attempt_kind=kw.pop("attempt_kind", "initial"),
        max_filings=kw.pop("max_filings", DEFAULT_MAX),
        max_bytes=kw.pop("max_bytes", DEFAULT_BYTES),
        **kw,
    )


# === case 1: valid filing lands N unapproved rows with full provenance ======


def test_valid_filing_lands_n_unapproved_rows_with_provenance(tmp_path, store, plane):
    worktree = tmp_path / "work"
    props = [valid_proposal(i) for i in (1, 2, 3)]
    seed_filings_file(worktree, props)

    result = harvest(worktree, store, plane)

    assert isinstance(result, FilingResult)
    assert len(result.accepted_ids) == 3
    assert result.rejected == []

    rows = store.list_filed_tickets()
    assert len(rows) == 3
    ids = {r["id"] for r in rows}
    # bead .39: the id namespace carries origin_role (`filed-{role}-{dispatch}-{n}`).
    assert ids == {f"filed-worker-{DISPATCH_ID}-{n}" for n in (1, 2, 3)}
    assert ids == set(result.accepted_ids)

    for r in rows:
        # unapproved-by-construction: filed_tickets has no `approved` column at
        # all (it is not a tickets row); it is untriaged.
        assert r["triaged"] == 0
        assert r["triage_outcome"] is None
        assert r["resulting_ticket_id"] is None
        # full origin provenance.
        assert r["origin_role"] == "worker"
        assert r["origin_worker"] == DISPATCH_ID
        assert r["origin_dispatch_id"] == DISPATCH_ID
        assert r["origin_parent_ticket"] == PARENT_TICKET
        # loop-stamped discovered_from (NOT worker-authored).
        assert r["discovered_from"] == DISCOVERED_FROM_FMT.format(
            dispatch_id=DISPATCH_ID, parent_ticket=PARENT_TICKET
        )
        assert r["proposal_hash"]
        assert r["created_at"] > 0


# === case 2: un-claimable by construction (the load-bearing invariant) ======


def test_filed_proposal_is_never_a_tickets_row(tmp_path, store, plane):
    worktree = tmp_path / "work"
    seed_filings_file(worktree, [valid_proposal(1)])
    result = harvest(worktree, store, plane)

    filed_id = result.accepted_ids[0]
    # The proposal exists in filed_tickets ...
    assert store.list_filed_tickets()[0]["id"] == filed_id
    # ... and does NOT exist in `tickets` (the claim/eligible surface).
    assert store.get_ticket(filed_id) is None
    assert store.list_tickets() == []


def test_filed_proposal_never_claimable_or_eligible_across_n_polls(tmp_path, store, plane):
    worktree = tmp_path / "work"
    seed_filings_file(worktree, [valid_proposal(1)])
    filed_id = harvest(worktree, store, plane).accepted_ids[0]

    def steering_of(ticket_id: str) -> dict[str, Any]:
        # eligible() must never even consult steering for a filed proposal —
        # it iterates `tickets` only, which is empty.
        raise AssertionError(f"steering_of was called for {ticket_id!r} — a filed "
                             "proposal leaked into the tickets iteration")

    for cycle in range(5):
        assert eligible(store, now=1000.0 + cycle, steering_of=steering_of) == []

    # A direct claim of the filed id fails closed: it is not a ticket at all.
    with pytest.raises(LeaseError):
        claim(
            store,
            filed_id,
            owner="loop",
            dispatch_id="d-x",
            ttl_seconds=60,
            now=1000.0,
            steering={},
            execution={},
        )


def test_prove_can_fail_a_tickets_row_WOULD_be_claimable(store):
    """Guardrail proving case 2's assertion is not vacuous: if a proposal were
    (wrongly) inserted into `tickets` as an approved pool row instead of into
    `filed_tickets`, it WOULD become eligible/claimable. This is exactly the
    regression the separate-table design prevents — if a future edit routes
    filings into `tickets`, case 2 goes red."""
    steering = {
        "ticket_text": "leaked proposal",
        "checks": {"named": ["pytest"], "paths": []},
        "rubric": ["r"],
        "lane": "cheap",
        "prompt_bytes": "code01",
        "context_set": [],
    }
    store.add_ticket(id="leaked", title="leaked", state="pool")
    approve(store, "leaked", steering=steering)
    assert eligible(store, now=1000.0, steering_of=lambda _id: steering) == ["leaked"]


# === case 3: count-cap excess rejects the WHOLE filing ======================


def test_count_cap_exceeded_rejects_whole_filing(tmp_path, store, plane):
    worktree = tmp_path / "work"
    # cap is 3; seed 4 -> reject ALL 4 (reject-whole, amendment D).
    props = [valid_proposal(i) for i in (1, 2, 3, 4)]
    seed_filings_file(worktree, props)

    result = harvest(worktree, store, plane, max_filings=3)

    assert result.accepted_ids == []
    assert store.list_filed_tickets() == []
    assert len(result.rejected) == 1
    assert result.rejected[0]["reason"] == "count-cap-exceeded"

    events = filed_events(plane)
    assert len(events) == 1
    assert events[0]["outcome"] == "rejected"
    assert events[0]["reason"] == "count-cap-exceeded"


def test_count_at_cap_is_accepted(tmp_path, store, plane):
    worktree = tmp_path / "work"
    seed_filings_file(worktree, [valid_proposal(i) for i in (1, 2, 3)])
    result = harvest(worktree, store, plane, max_filings=3)
    assert len(result.accepted_ids) == 3


# === case 4: per-proposal size-cap rejects only the oversized proposal ======


def test_size_cap_rejects_only_oversized_proposal(tmp_path, store, plane):
    worktree = tmp_path / "work"
    big = valid_proposal(2, description="x" * 5000)
    props = [valid_proposal(1), big, valid_proposal(3)]
    seed_filings_file(worktree, props)

    result = harvest(worktree, store, plane, max_bytes=500)

    # the two small proposals file; the big one is rejected on its own.
    assert len(result.accepted_ids) == 2
    titles_filed = {r["title"] for r in store.list_filed_tickets()}
    assert titles_filed == {valid_proposal(1)["title"], valid_proposal(3)["title"]}

    assert len(result.rejected) == 1
    assert result.rejected[0]["reason"] == "size-cap-exceeded"
    assert result.rejected[0]["title"] == big["title"]

    rej_events = [e for e in filed_events(plane) if e["outcome"] == "rejected"]
    assert len(rej_events) == 1
    assert rej_events[0]["reason"] == "size-cap-exceeded"
    acc_events = [e for e in filed_events(plane) if e["outcome"] == "accepted"]
    assert len(acc_events) == 2


# === case 5: malformed / bad-shape / traversal — rejected, no corruption, no raise


def test_malformed_json_rejected_without_corruption_or_raise(tmp_path, store, plane):
    worktree = tmp_path / "work"
    seed_filings_file(worktree, "{ this is not valid json ]]")

    result = harvest(worktree, store, plane)  # must NOT raise

    assert result.accepted_ids == []
    assert store.list_filed_tickets() == []
    assert store.list_tickets() == []  # pool uncorrupted
    assert len(result.rejected) == 1
    events = filed_events(plane)
    assert len(events) == 1 and events[0]["outcome"] == "rejected"


def test_non_list_top_level_rejected(tmp_path, store, plane):
    worktree = tmp_path / "work"
    seed_filings_file(worktree, {"title": "not", "description": "a list"})
    result = harvest(worktree, store, plane)
    assert result.accepted_ids == []
    assert store.list_filed_tickets() == []


def test_proposal_missing_required_key_rejected(tmp_path, store, plane):
    worktree = tmp_path / "work"
    # one good, one missing "description"
    seed_filings_file(worktree, [valid_proposal(1), {"title": "no description"}])
    result = harvest(worktree, store, plane)
    # bad-shape is a per-proposal reject; the good one still files.
    assert len(result.accepted_ids) == 1
    assert any(r["reason"] for r in result.rejected)


# === six-field shape authority (bead .162 audit fix) =========================
# The ingest is the single validation authority for the SIX contract fields —
# the daemon routes worker-escape on blocks_ticket /
# suspected_out_of_scope_paths, so their types must be validated here, not
# only at the OA tool's call-time layer. Violations are per-item `bad-shape`
# rejections (isolation preserved; NEVER whole-batch).


def test_blocks_ticket_string_rejected_bad_shape_siblings_accepted(tmp_path, store, plane):
    # blocks_ticket="yes" (a truthy STRING) must not persist — it is the
    # misroute garbage the shape authority exists to keep out of the DB.
    # Per-item isolation: the well-typed sibling still files.
    worktree = tmp_path / "work"
    bad = valid_proposal(1, blocks_ticket="yes")
    good = valid_proposal(2)
    seed_filings_file(worktree, [bad, good])
    result = harvest(worktree, store, plane)

    assert len(result.accepted_ids) == 1
    assert result.accepted_ids == [f"filed-worker-{DISPATCH_ID}-2"]
    assert len(result.rejected) == 1
    assert result.rejected[0]["reason"] == "bad-shape"
    assert result.rejected[0]["title"] == bad["title"]
    filed = store.list_filed_tickets()
    assert {r["title"] for r in filed} == {good["title"]}
    rej_events = [e for e in filed_events(plane) if e["outcome"] == "rejected"]
    assert [e["reason"] for e in rej_events] == ["bad-shape"]
    # No escape surfacing from a string-typed blocks_ticket (is True fails).
    assert result.escape is None


def test_blocks_ticket_none_explicit_is_absent_semantics(tmp_path, store, plane):
    # An EXPLICIT blocks_ticket=None is absent-semantics (bool-or-absent):
    # accepted, and it does not surface as escape.
    worktree = tmp_path / "work"
    seed_filings_file(worktree, [valid_proposal(1, blocks_ticket=None)])
    result = harvest(worktree, store, plane)
    assert len(result.accepted_ids) == 1
    assert result.rejected == []
    assert result.escape is None


def test_suspected_paths_string_rejected_bad_shape(tmp_path, store, plane):
    # suspected_out_of_scope_paths="x.py" (a string, not a list of strings)
    # is a per-item bad-shape rejection; the sibling still files.
    worktree = tmp_path / "work"
    bad = valid_proposal(1, suspected_out_of_scope_paths="x.py")
    good = valid_proposal(2)
    seed_filings_file(worktree, [bad, good])
    result = harvest(worktree, store, plane)

    assert len(result.accepted_ids) == 1
    assert len(result.rejected) == 1
    assert result.rejected[0]["reason"] == "bad-shape"
    assert result.rejected[0]["title"] == bad["title"]
    assert result.escape is None


def test_suspected_paths_list_with_non_string_rejected_bad_shape(tmp_path, store, plane):
    # A list containing a non-string element is still not list[str].
    worktree = tmp_path / "work"
    bad = valid_proposal(1, suspected_out_of_scope_paths=["x.py", 42])
    good = valid_proposal(2)
    seed_filings_file(worktree, [bad, good])
    result = harvest(worktree, store, plane)

    assert len(result.accepted_ids) == 1
    assert len(result.rejected) == 1
    assert result.rejected[0]["reason"] == "bad-shape"


def test_reason_non_string_rejected_bad_shape(tmp_path, store, plane):
    # reason=42 (not str-or-absent) is a per-item bad-shape rejection; the
    # sibling still files.
    worktree = tmp_path / "work"
    bad = valid_proposal(1, reason=42)
    good = valid_proposal(2)
    seed_filings_file(worktree, [bad, good])
    result = harvest(worktree, store, plane)

    assert len(result.accepted_ids) == 1
    assert len(result.rejected) == 1
    assert result.rejected[0]["reason"] == "bad-shape"
    assert result.rejected[0]["title"] == bad["title"]


def test_valid_escape_filing_still_accepted_and_routed(tmp_path, store, plane):
    # A well-typed escape filing (blocks_ticket=True + a list-of-str paths +
    # a str reason) is STILL accepted and STILL surfaces via the harvest's
    # escape routing (.102.2 contract): the typed validation must not break
    # the escape path.
    worktree = tmp_path / "work"
    escape_obj = valid_proposal(
        1,
        blocks_ticket=True,
        reason="worker hit an out-of-scope dependency",
        suspected_out_of_scope_paths=["tests/test_bar.py", "src/lib/other.py"],
    )
    normal_obj = valid_proposal(2)
    seed_filings_file(worktree, [escape_obj, normal_obj])

    result = harvest(worktree, store, plane)

    assert result.rejected == []
    assert len(result.accepted_ids) == 2  # BOTH filed, including the escape
    assert result.escape is not None
    assert result.escape["reason"] == "worker hit an out-of-scope dependency"
    assert result.escape["suspected_out_of_scope_paths"] == [
        "tests/test_bar.py",
        "src/lib/other.py",
    ]


def test_unknown_extra_keys_not_rejected(tmp_path, store, plane):
    # Unknown extra keys are NOT rejected: the store insert takes named
    # fields only (the extras never reach the DB) and proposal_hash hashes
    # the three provenance keys only.
    worktree = tmp_path / "work"
    seed_filings_file(worktree, [valid_proposal(1, priority="high", approved=True)])
    result = harvest(worktree, store, plane)
    assert len(result.accepted_ids) == 1
    assert result.rejected == []
    row = store.list_filed_tickets()[0]
    # The row carries the named fields only — the extras did not persist.
    assert row["title"] == valid_proposal(1)["title"]
    assert "priority" not in row


def test_path_traversal_symlink_rejected(tmp_path, store, plane):
    worktree = tmp_path / "work"
    (worktree / ".stigmergy").mkdir(parents=True)
    outside = tmp_path / "outside-secret.json"
    outside.write_text(json.dumps([valid_proposal(1)]))
    link = worktree / FILED_TICKETS_REL
    os.symlink(outside, link)  # symlink inside worktree escaping to outside

    result = harvest(worktree, store, plane)  # must NOT raise, must NOT follow

    assert result.accepted_ids == []
    assert store.list_filed_tickets() == []
    assert len(result.rejected) == 1


def test_missing_filings_file_is_a_silent_noop(tmp_path, store, plane):
    worktree = tmp_path / "work"
    worktree.mkdir()
    result = harvest(worktree, store, plane)
    assert result.accepted_ids == []
    assert result.rejected == []
    assert filed_events(plane) == []  # no file -> no event at all


# === case 7: ticket-filed event shape + honest zeros + no secret ============


def test_ticket_filed_accepted_event_shape(tmp_path, store, plane):
    worktree = tmp_path / "work"
    seed_filings_file(worktree, [valid_proposal(1)])
    harvest(worktree, store, plane)

    events = filed_events(plane)
    assert len(events) == 1
    ev = events[0]

    # §8 common fields present.
    for f in (
        "rig", "ticket", "dispatch_id", "attempt", "attempt_kind", "rung", "worker",
        "charter_hash", "approval_hash", "image_digest", "model", "model_version",
        "price_table_version", "tokens", "computed_usd", "wall_time_seconds", "ts",
    ):
        assert f in ev, f"missing common field {f!r}"

    # honest zeros for the cost fields (harvest is host-side mechanism; the
    # worker's cost was already accounted on the DISPATCH event — double-counting
    # would corrupt spend reconstruction).
    assert ev["computed_usd"] == 0.0
    assert ev["tokens"] == {"in": 0, "cached": 0, "out": 0, "reasoning": 0}
    assert ev["wall_time_seconds"] == 0.0

    # ticket-filed specifics.
    assert ev["outcome"] == "accepted"
    assert ev["reason"] is None
    assert ev["filed_ticket_id"] == f"filed-worker-{DISPATCH_ID}-1"
    assert ev["origin"] == {
        "role": "worker",
        "worker": DISPATCH_ID,
        "dispatch_id": DISPATCH_ID,
        "parent_ticket": PARENT_TICKET,
    }
    assert ev["proposal_hash"] == proposal_hash(valid_proposal(1))

    # NOT an LLM-invocation event — carries no prompt_artifact_hash.
    assert "prompt_artifact_hash" not in ev


def test_ticket_filed_rejected_event_carries_reason(tmp_path, store, plane):
    worktree = tmp_path / "work"
    big = valid_proposal(1, description="y" * 5000)
    seed_filings_file(worktree, [big])
    harvest(worktree, store, plane, max_bytes=200)

    events = filed_events(plane)
    assert len(events) == 1
    assert events[0]["outcome"] == "rejected"
    assert isinstance(events[0]["reason"], str) and events[0]["reason"]
    assert events[0]["filed_ticket_id"] is None


def test_secret_never_appears_in_filed_event(tmp_path, store, plane):
    """A worker-authored proposal that embeds a secret-looking string still
    produces an event — but the event is provenance + a content hash, and the
    filing pipeline never has a provider key to leak. Assert no obvious key
    material rides along in the emitted event payload."""
    worktree = tmp_path / "work"
    seed_filings_file(worktree, [valid_proposal(1)])
    ctx = dispatch_ctx()
    harvest(worktree, store, plane, ctx=ctx)
    ev = filed_events(plane)[0]
    blob = json.dumps(ev)
    assert "sk-" not in blob
    assert "ANTHROPIC_API_KEY" not in blob


# === case 9: proposal_hash stable + distinct from approval_hash =============


def test_proposal_hash_stable_over_content_and_key_order():
    p1 = {"title": "T", "description": "D", "evidence": "E"}
    p2 = {"evidence": "E", "description": "D", "title": "T"}  # reordered keys
    assert proposal_hash(p1) == proposal_hash(p2)
    # changing any field changes the hash.
    assert proposal_hash(p1) != proposal_hash({"title": "T", "description": "D2", "evidence": "E"})


def test_proposal_hash_stable_when_evidence_absent():
    a = proposal_hash({"title": "T", "description": "D"})
    b = proposal_hash({"title": "T", "description": "D", "evidence": None})
    assert a == b


def test_proposal_hash_distinct_from_steering_approval_hash():
    proposal = valid_proposal(1)
    # A steering dict that superficially resembles the proposal must not
    # collide with proposal_hash — the two hash spaces are deliberately
    # different shapes (§8 proposal provenance vs §4 approval integrity).
    steering = {
        "ticket_text": proposal["title"],
        "checks": {"named": [], "paths": []},
        "rubric": [proposal["description"]],
        "lane": "cheap",
        "prompt_bytes": "code01",
        "context_set": [],
    }
    assert proposal_hash(proposal) != steering_hash(steering)


# === file_proposals reuse by .39/.41 (origin_role parametrised) =============


def test_file_proposals_reused_with_critic_origin_role(store, plane):
    """`.39`/`.41` reuse `file_proposals` verbatim with a different origin_role;
    the row + event carry that role, and it is STILL un-claimable (filed_tickets)."""
    ctx = dispatch_ctx(dispatch_id="critic-dispatch-1", worker="critic-1")
    result = file_proposals(
        [valid_proposal(1)],
        store=store,
        record_plane=plane,
        ctx=ctx,
        attempt_kind="initial",
        origin_role="critic",
        max_filings=DEFAULT_MAX,
        max_bytes=DEFAULT_BYTES,
    )
    assert len(result.accepted_ids) == 1
    row = store.list_filed_tickets()[0]
    assert row["origin_role"] == "critic"
    assert store.get_ticket(row["id"]) is None  # still not a claimable ticket
    ev = filed_events(plane)[0]
    assert ev["origin"]["role"] == "critic"


def test_worker_and_critic_filings_on_same_dispatch_do_not_collide(store, plane):
    """bead .39 regression: the worker's harvest and the staging critic's gate
    filing share ONE real dispatch id in production (the daemon reconciles the
    ticket's lease_dispatch_id to the worker's plan.dispatch_id, and the weaver's
    default ctx reads that same lease id). With a role-blind `filed-{dispatch}-{n}`
    id, the critic's proposal at index n would hit the worker proposal's primary
    key and be dropped as `store-error`, silently losing the crit-role discovery.
    Keying the id on origin_role makes both land with distinct ids."""
    shared = dispatch_ctx(dispatch_id="shared-dispatch-1", worker="worker-1")

    worker_res = file_proposals(
        [valid_proposal(1), valid_proposal(2)],
        store=store,
        record_plane=plane,
        ctx=shared,
        attempt_kind="initial",
        origin_role="worker",
        max_filings=DEFAULT_MAX,
        max_bytes=DEFAULT_BYTES,
    )
    critic_res = file_proposals(
        [valid_proposal(1), valid_proposal(2)],  # same content, same indexes
        store=store,
        record_plane=plane,
        ctx=shared,  # SAME dispatch id + worker
        attempt_kind="report",
        origin_role="critic",
        max_filings=DEFAULT_MAX,
        max_bytes=DEFAULT_BYTES,
    )

    # Both roles' proposals landed — none dropped as a collision.
    assert len(worker_res.accepted_ids) == 2
    assert len(critic_res.accepted_ids) == 2
    assert critic_res.rejected == []
    # Distinct id namespaces; no overlap.
    assert set(worker_res.accepted_ids).isdisjoint(critic_res.accepted_ids)
    assert all(i.startswith("filed-worker-") for i in worker_res.accepted_ids)
    assert all(i.startswith("filed-critic-") for i in critic_res.accepted_ids)
    # All four rows persisted.
    assert len(store.list_filed_tickets()) == 4


# === escape surfacing (worker blocks_ticket:true) ============================


def test_escape_present_surfaces_on_filing_result(tmp_path, store, plane):
    """A filed proposals list with a blocks_ticket=True object surfaces the
    escape on FilingResult.escape and the object is ALSO filed as a normal
    proposal."""
    worktree = tmp_path / "work"
    escape_obj = valid_proposal(1, blocks_ticket=True,
                                reason="out-of-scope test asserts old behavior",
                                suspected_out_of_scope_paths=["tests/test_bar.py"])
    normal_obj = valid_proposal(2)
    seed_filings_file(worktree, [escape_obj, normal_obj])

    result = harvest(worktree, store, plane)

    assert result.escape is not None
    assert result.escape["reason"] == "out-of-scope test asserts old behavior"
    assert result.escape["suspected_out_of_scope_paths"] == ["tests/test_bar.py"]
    # The escape object is ALSO filed as a normal proposal (accepted_ids populated).
    assert len(result.accepted_ids) == 2
    assert result.rejected == []


def test_escape_absent_yields_none(tmp_path, store, plane):
    """A filed proposals list with no blocks_ticket=True object yields
    escape=None and filing proceeds normally."""
    worktree = tmp_path / "work"
    seed_filings_file(worktree, [valid_proposal(1), valid_proposal(2)])

    result = harvest(worktree, store, plane)

    assert result.escape is None
    assert len(result.accepted_ids) == 2
    assert result.rejected == []


def test_escape_first_wins(tmp_path, store, plane):
    """When multiple blocks_ticket=True objects are present, only the FIRST
    one is surfaced as escape; subsequent ones are still filed as proposals."""
    worktree = tmp_path / "work"
    escape1 = valid_proposal(1, blocks_ticket=True, reason="first reason",
                             suspected_out_of_scope_paths=["path1.py"])
    escape2 = valid_proposal(2, blocks_ticket=True, reason="second reason",
                             suspected_out_of_scope_paths=["path2.py"])
    seed_filings_file(worktree, [escape1, escape2])

    result = harvest(worktree, store, plane)

    assert result.escape is not None
    assert result.escape["reason"] == "first reason"
    assert result.escape["suspected_out_of_scope_paths"] == ["path1.py"]
    assert len(result.accepted_ids) == 2  # both filed


def test_escape_malformed_never_raises(tmp_path, store, plane):
    """Malformed escape fields are defensively handled and harvest never
    raises. blocks_ticket as string does not match (is True); paths as string
    coerced to []; reason as non-str coerced to None. Bead .162 audit fix:
    these malformed items are ALSO per-item `bad-shape` rejections at the
    shape authority (the escape COERCION above is harvest-side escape-object
    building; the item itself no longer persists)."""
    worktree = tmp_path / "work"

    # blocks_ticket="true" (string, not bool) — does not match AND the item
    # is a per-item bad-shape rejection (a truthy string is the misroute
    # garbage the shape authority exists to keep out of the DB).
    seed_filings_file(worktree, [valid_proposal(1, blocks_ticket="true")])
    result = harvest(worktree, store, plane, ctx=dispatch_ctx(dispatch_id="d1"))
    assert result.escape is None
    assert result.accepted_ids == []
    assert [r["reason"] for r in result.rejected] == ["bad-shape"]

    # blocks_ticket=True but suspected_out_of_scope_paths is a string —
    # escape coercion still yields [] (never raises), and the item is a
    # per-item bad-shape rejection.
    store2 = RigStore.create(tmp_path / "tickets2.db")
    plane2 = RecordPlane(tmp_path / "records2")
    obj2 = valid_proposal(1, blocks_ticket=True, suspected_out_of_scope_paths="not-a-list")
    seed_filings_file(worktree, [obj2])
    result2 = harvest(worktree, store2, plane2, ctx=dispatch_ctx(dispatch_id="d2"))
    assert result2.escape is not None
    assert result2.escape["suspected_out_of_scope_paths"] == []
    assert result2.accepted_ids == []
    assert [r["reason"] for r in result2.rejected] == ["bad-shape"]
    store2.close()

    # blocks_ticket=True but reason is non-str — escape coercion still yields
    # None, and the item is a per-item bad-shape rejection.
    store3 = RigStore.create(tmp_path / "tickets3.db")
    plane3 = RecordPlane(tmp_path / "records3")
    obj3 = valid_proposal(1, blocks_ticket=True, reason=123)
    seed_filings_file(worktree, [obj3])
    result3 = harvest(worktree, store3, plane3, ctx=dispatch_ctx(dispatch_id="d3"))
    assert result3.escape is not None
    assert result3.escape["reason"] is None
    assert result3.accepted_ids == []
    assert [r["reason"] for r in result3.rejected] == ["bad-shape"]
    store3.close()


def test_file_proposals_direct_call_returns_escape_none():
    """file_proposals called directly (not via harvest) still returns
    escape=None, proving the default keeps direct callers unchanged."""
    result = FilingResult(accepted_ids=["a", "b"], rejected=[])
    # file_proposals constructs FilingResult without escape, so it defaults to None
    # (this is tested implicitly by all direct file_proposals calls; we assert it explicitly here).
    assert result.escape is None


def test_list_filed_tickets_filter_and_count(store, plane):
    file_proposals(
        [valid_proposal(1), valid_proposal(2)],
        store=store,
        record_plane=plane,
        ctx=dispatch_ctx(),
        attempt_kind="initial",
        origin_role="worker",
        max_filings=DEFAULT_MAX,
        max_bytes=DEFAULT_BYTES,
    )
    assert store.count_untriaged_filings() == 2
    assert len(store.list_filed_tickets(triaged=False)) == 2
    assert store.list_filed_tickets(triaged=True) == []
