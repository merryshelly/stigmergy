"""The `weave` station — isolated candidate, bundle apply, CAS land, journal
(SPEC.md §3 `weave` station, §6 "Protection asymmetry", §9 "Weave triggers"
+ "Crash recovery", §10 AC6/AC8/AC9).

**The weaver is the single serialized writer to `staging`** (SPEC §3): every
parked ticket is gated one at a time, in creation order, through an
ISOLATED candidate clone that the worker never saw — the worker's bundle is
object data applied to a tree it has no shared `.git` metadata with, so a
hook seeded anywhere in that bundle (or in the candidate's own `.git/
hooks/`) can never execute (`core.hooksPath=/dev/null` on every loop-side
git invocation, set both per-command and persisted on the candidate clone
— belt and suspenders, SPEC §4/§10 AC6). This closes the git-hook RCE: the
weaver, not the worker, is the only thing that ever writes `refs/heads/
staging`, and it does so only via a compare-and-swap `update-ref` pinned to
the OID it observed at the start of this ticket's gate.

**Gate-then-land is the whole point** (SPEC §10 AC8): a bundle must (1)
apply cleanly onto an isolated candidate, (2) pass a full fresh re-run of
every check on the *integrated* tree (FLAKY/ERROR count as red, never
silently green), and (3) clear a critic verdict (`Verdict.lands()`, outcome
MET only — severity never gates) before `staging` moves at all. Any one of
those failing dies at its own station with its own :class:`~stigmergy.
statemachine.FailureClass` and the ticket never reaches the CAS step. A
critic-call failure (:class:`~stigmergy.critic.CriticInfraError`) is INFRA,
never a rejection (SPEC §9): the bundle simply goes back to `parked`, no
GATE rejection event is ever written.

**CAS land, not force-land** (SPEC §9): the weaver re-reads the live
`staging` tip immediately before landing and only proceeds if it still
equals the OID pinned at this ticket's gate-begin; the actual mechanism is
`git update-ref refs/heads/staging <candidate> <pinned>` — the old-value
argument IS the compare-and-swap, so a concurrent move (another process,
an operator, a resumed weave) makes git itself refuse the update. On a
lost race the weaver aborts cleanly: nothing is force-landed over the
winner, the ticket returns to `parked` to be re-gated on a later weave.

**Journal + idempotent resolve() (SPEC §9/§10 AC9).** A durable, append-
only, fsync'd JSONL file records each ticket's weave phases: `begin` (with
the pinned OID) → zero or more intermediate phases → `complete`. The
critical invariant for crash recovery is that the candidate's OID is
journaled (`applied` phase) the moment the candidate commit exists — well
before the CAS is even attempted — so that no matter where a crash lands
after that point (including the genuinely ambiguous window *after*
`update-ref` has already succeeded but *before* the daemon manages to write
`complete`), :meth:`Weaver.resolve` can reconcile by asking git itself
which world it woke up in: if `staging`'s live tip already equals the
journaled candidate OID, the land already committed — resolve only needs
to finish the ticket-state bookkeeping (never re-attempt the CAS, never
re-write the GATE/INTEGRATION/DISPOSITION events already written before
the crash). Otherwise the land never committed and the ticket goes back to
`parked`, `staging` untouched. `resolve()` is safe to call repeatedly: once
the journal's tail is sealed `complete`, a second call recomputes the same
answer and mutates nothing.

`Weaver` satisfies :class:`stigmergy.recover.WeaveJournalResolver` — a
`Weaver` instance is passed straight into :func:`stigmergy.recover.recover`.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stigmergy.checks import CheckOutcome, CheckResult
from stigmergy.critic import CriticInfraError, format_check_evidence
from stigmergy.filing import file_proposals
from stigmergy.records import EventType, RecordPlane, make_event
from stigmergy.statemachine import (
    GATED,
    LANDED,
    PARKED,
    REJECTED,
    FailureClass,
    record_disposition,
    transition,
)
from stigmergy.verdicts import Verdict

_logger = logging.getLogger(__name__)

# The fixed refspec convention a parked ticket's bundle is expected to carry
# (a documented convention this module owns, since a bundle is just object
# data — see module docstring / class docstring for the rationale). Only
# THIS exact ref is ever fetched from a worker bundle; any other ref the
# bundle happens to carry (including a maliciously-included
# `refs/heads/staging`, SPEC §10 AC6) is never requested and therefore
# never reaches any local ref, staging's included.
_BUNDLE_REF = "refs/heads/work"

# The local branch name the weaver builds each candidate on, inside the
# candidate clone. Deliberately distinct from `staging` so there is never
# any ambiguity with a same-named ref a hostile bundle might carry.
_CANDIDATE_BRANCH = "weave-candidate"

# The ref namespace used to stage a candidate's tip inside `staging_repo`'s
# object store immediately before the CAS `update-ref` (fetched in BY
# BRANCH NAME from the candidate clone — never by raw OID, which local
# transports refuse to serve without `uploadpack.allow*SHA1InWant`).
_INCOMING_REF_PREFIX = "refs/weaver/incoming"

_JOURNAL_MODE = 0o600

# SPEC §8: common fields required on every event; the ones this module
# always declares as true zero for its own non-LLM (INTEGRATION/
# DISPOSITION) events — a weave bookkeeping event spends nothing.
_ZERO_TOKENS: dict[str, int] = {"in": 0, "cached": 0, "out": 0, "reasoning": 0}

# Scope-breach audit algorithm version stamp (reproducibility after rebase/
# policy change). Bumping this identifies when the matching rule evolves.
_SCOPE_AUDIT_ALGO_VERSION = "102a-v1"

# The ctx_of() contract: every key a caller-supplied ctx dict must carry so
# GATE/INTEGRATION events (and record_disposition's own ctx fields) can be
# built without ever silently defaulting an identity field.
_CTX_FIELDS: tuple[str, ...] = (
    "rig",
    "dispatch_id",
    "attempt",
    "attempt_kind",
    "rung",
    "worker",
    "charter_hash",
    "approval_hash",
    "image_digest",
    "model",
    "model_version",
    "price_table_version",
    "tokens",
    "computed_usd",
    "wall_time_seconds",
)


class WeaverError(Exception):
    """Raised on a weaver-side failure that is a bug/misconfiguration, not
    a normal gate outcome: a git plumbing command failing unexpectedly, or
    `weave()` being invoked while a previous weave is still unresolved
    (SPEC §9: the weaver is a single serialized writer — it must never
    start fresh work on top of an interrupted, un-recovered journal)."""


@dataclass(frozen=True)
class WeaveResult:
    """The outcome of gating one parked ticket (SPEC §1.1).

    ``outcome`` is one of ``"landed"``, ``"integration-conflict"``,
    ``"integration-regression"``, ``"rejected"``, ``"infra"``.
    ``failure_class`` is ``None`` iff ``outcome == "landed"``; otherwise it
    is the :class:`~stigmergy.statemachine.FailureClass` the loop (.22)
    applies retry counters from. A CAS-lost-race (concurrent `staging`
    move) is reported as ``"infra"``/`FailureClass.INFRA` — the design
    doc's enumerated outcome vocabulary has no dedicated "cas-abort" value,
    and INFRA's semantics (no rung attempt burned, ticket returns to
    `parked` to retry, never recorded as a rejection) are the closest
    fit; ``detail`` distinguishes it from a genuine critic-infra failure
    in the human-readable trail.

    ``gate_model``, ``gate_tokens``, ``gate_duration`` carry the real
    critic call's token usage and wall-clock duration, extracted from
    the gate_fields returned by critic.judge(). These are None when no
    gate call was made (e.g., gate-infra, checks-red, conflict outcomes
    that failed before reaching the critic).
    """

    ticket: str
    outcome: str
    landed_oid: str | None
    failure_class: FailureClass | None
    verdict: Verdict | None
    check_results: list[CheckResult] | None
    flagged_for_human: bool
    detail: str
    reason: str | None = None
    gate_model: str | None = None
    gate_tokens: dict[str, int] | None = None
    gate_duration: float | None = None


class Weaver:
    """The `weave` station: gate-then-land parked tickets one at a time,
    serialized, onto `staging` (SPEC §3/§9/§10 AC6/AC8/AC9).

    ``store`` / ``record_plane``: the rig's ticket store and event log.
    ``staging_repo``: the loop-owned repo whose `staging` branch is the
    integration target (a bare repo is recommended — see module docstring
    and tests — but nothing here requires it; the weaver only ever reads
    `refs/heads/staging` and moves it via CAS `update-ref`, never checks
    it out). ``run_checks_fn``: injected, `checks.run_checks`-shaped
    callable (tests inject a fake; production wires the real runner).
    ``critic``: a :class:`stigmergy.critic.Critic` (or any object exposing
    a duck-typed `.judge(artifact, rubric_items)`; tests may wrap it in a
    call-counting spy). ``checker_image`` / ``flake_reruns``: passed
    through to ``run_checks_fn`` verbatim. ``protected_paths``: glob
    patterns (SPEC §6 "Protection asymmetry") — a bundle diff touching any
    of these sets `flagged_for_human=True` and is recorded, but does not
    by itself block landing in v0. ``journal_path``: the durable, append-
    only weave journal file (fsync'd JSONL; see module docstring).
    ``ctx_of``: optional ``ticket_id -> dict`` supplying the SPEC §8
    common-field identity context for this ticket's dispatch (see
    :data:`_CTX_FIELDS` for the exact required keys) — tests pass a simple
    stub; if omitted, a conservative all-zero/None default is used (see
    :meth:`_default_ctx`).

    ``filing_max_filings`` / ``filing_max_bytes``: REQUIRED (no defaults —
    an omitted wiring must fail loudly, not silently pass) D14 per-dispatch
    filing caps (bead `.39`), injected explicitly from
    ``charter.raw["loop"]["dispatch_limits"]`` at wire time — passed
    straight through to :func:`stigmergy.filing.file_proposals` as
    ``max_filings``/``max_bytes`` when the critic's `filed_tickets` are
    filed after the GATE event (see :meth:`_file_critic_proposals`).

    **Bundle convention** (documented here since the design doc leaves the
    exact bundle-location/ref-name convention to the implementation): a
    parked ticket's work product is a `git bundle` file whose path is
    ``ticket_row["work_product"]``, carrying (at minimum) a
    ``refs/heads/work`` ref pointing at the tip commit(s) to integrate.
    Only that one, fixed, explicitly-named ref is ever fetched from a
    bundle — any other ref the bundle happens to carry (including a
    maliciously-included `refs/heads/staging`, SPEC §10 AC6 case 9) is
    never requested and therefore never adopted anywhere, `staging`
    included.
    """

    def __init__(
        self,
        *,
        store: Any,
        record_plane: RecordPlane,
        staging_repo: str | Path,
        run_checks_fn: Any,
        critic: Any,
        checker_image: str,
        flake_reruns: int,
        protected_paths: list[str],
        journal_path: str | Path,
        filing_max_filings: int,
        filing_max_bytes: int,
        ctx_of: Any = None,
        check_resources_fn: Any = None,
    ) -> None:
        self.store = store
        self.record_plane = record_plane
        self.staging_repo = Path(staging_repo)
        self.run_checks_fn = run_checks_fn
        self.critic = critic
        self.checker_image = checker_image
        self.flake_reruns = flake_reruns
        self.protected_paths = list(protected_paths)
        self.journal_path = Path(journal_path)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        self.filing_max_filings = filing_max_filings
        self.filing_max_bytes = filing_max_bytes
        self.ctx_of = ctx_of
        # bead .91: bound `charter.resolve_check_resources(charter, name)` so the
        # staging-gate full-suite re-run gets the same charter-configured
        # resource bounds as the daemon attempt gate. None -> all checks use
        # DEFAULT_CHECK_RESOURCES (keeps injected-double tests unchanged).
        self.check_resources_fn = check_resources_fn

    # -- WeaveJournalResolver protocol (recover.py) ------------------------

    def in_progress(self) -> bool:
        """True iff the journal's last entry is not a sealed `complete` —
        i.e. some ticket's weave was interrupted mid-flight (SPEC §9)."""
        lines = self._read_journal()
        if not lines:
            return False
        return lines[-1].get("phase") != "complete"

    def resolve(self) -> str:
        """Reconcile the interrupted weave (SPEC §9/§10 AC9): "landed" or
        "rolled-back". Safe to call repeatedly — see module docstring.

        Reconciles the JOURNAL to REALITY rather than trusting the
        journal's own last-recorded phase: the land itself is a single
        atomic `update-ref` CAS, so the only fact that can decide whether
        it committed before a crash is git's own live `staging` tip.
        """
        group = self._last_group()
        if not group:
            return "rolled-back"

        ticket_id = group[0].get("ticket")
        dispatch_id = group[0].get("dispatch_id")
        pinned_oid = group[0].get("pinned_oid")
        candidate_oid = None
        for entry in group:
            if entry.get("candidate_oid"):
                candidate_oid = entry["candidate_oid"]

        already_complete = group[-1].get("phase") == "complete"

        landed = False
        if candidate_oid is not None:
            live_tip = self._rev_parse(self.staging_repo, "refs/heads/staging")
            landed = live_tip == candidate_oid

        outcome = "landed" if landed else "rolled-back"

        if not already_complete:
            row = self.store.get_ticket(ticket_id) if ticket_id is not None else None
            if row is not None:
                state = row.get("state")
                # Only fix ticket STATE + seal the journal — never
                # re-attempt the CAS, never backfill the GATE/INTEGRATION/
                # DISPOSITION events a completed-before-crash land already
                # wrote (or, for a rolled-back land, never invent events
                # for work that never happened). See module docstring.
                if landed and state == GATED:
                    transition(self.store, ticket_id, LANDED, expected_from=GATED)
                elif not landed and state == GATED:
                    transition(self.store, ticket_id, PARKED, expected_from=GATED)
            self._journal_append(
                ticket=ticket_id,
                dispatch_id=dispatch_id,
                phase="complete",
                pinned_oid=pinned_oid,
                candidate_oid=candidate_oid,
                resolved_as=outcome,
            )
        return outcome

    # -- weave() ------------------------------------------------------------

    def weave(self, *, now: float) -> list[WeaveResult]:
        """Gate every currently-parked ticket, one at a time, in creation
        order (SPEC §9 "Weave triggers": *when* to call this is the loop's
        job via `staging_quiescent_tickets`/queue-drained/`
        staging_max_wait_seconds`; this processes whatever is parked right
        now). Fails closed if a previous weave is still unresolved — the
        weaver is a single serialized writer and must never lay fresh work
        on top of an un-recovered journal (call `resolve()` via `recover()`
        first).
        """
        if self.in_progress():
            raise WeaverError(
                "weave() refused: a previous weave is still unresolved — "
                "call recover()/resolve() first"
            )

        results: list[WeaveResult] = []
        for ticket_row in self.store.list_tickets(state=PARKED):
            results.append(self._weave_one(ticket_row["id"], now=now))
        return results

    # -- per-ticket weave sequence -------------------------------------------

    def _weave_one(self, ticket_id: str, *, now: float) -> WeaveResult:
        transition(self.store, ticket_id, GATED, expected_from=PARKED)
        ticket_row = self.store.get_ticket(ticket_id)
        ctx = self._ctx(ticket_row)
        dispatch_id = ctx["dispatch_id"]

        pinned_oid = self._rev_parse(self.staging_repo, "refs/heads/staging")
        self._journal_append(
            ticket=ticket_id,
            dispatch_id=dispatch_id,
            phase="begin",
            pinned_oid=pinned_oid,
            candidate_oid=None,
            ts=now,
        )

        candidate_dir = self._clone_candidate(pinned_oid)
        try:
            applied, candidate_oid, stripped = self._apply_bundle(
                candidate_dir, ticket_row, pinned_oid
            )
            if not applied:
                return self._die(
                    ticket_id=ticket_id,
                    dispatch_id=dispatch_id,
                    ctx=ctx,
                    pinned_oid=pinned_oid,
                    candidate_oid=None,
                    phase="conflict",
                    outcome="integration-conflict",
                    failure_class=FailureClass.INTEGRATION_CONFLICT,
                    disposition="rejected",
                    reason="integration-conflict",
                    verdict=None,
                    check_results=None,
                    flagged=False,
                    detail="bundle did not apply cleanly onto the isolated candidate",
                    now=now,
                )

            self._journal_append(
                ticket=ticket_id,
                dispatch_id=dispatch_id,
                phase="applied",
                pinned_oid=pinned_oid,
                candidate_oid=candidate_oid,
                ts=now,
            )

            # Compute changed paths once for both protected-path check and
            # scope-breach audit, so git diff --name-only runs at most once.
            diff = self._git(
                candidate_dir, ["diff", "--name-only", pinned_oid, candidate_oid]
            ).stdout
            changed = [line.strip() for line in diff.splitlines() if line.strip()]

            # Record-plane INTEGRATION "apply" event (SPEC §1.0: "INTEGRATION
            # (apply/land/abort)") — written unconditionally once a
            # candidate exists, independent of what happens at checks/gate
            # next. Carries the scope_breach audit as a structured field.
            touched = self._protected_paths_touched(
                candidate_dir, pinned_oid, candidate_oid, changed=changed
            )
            target_scope = ticket_row.get("target_scope")
            out_of_scope_paths = self._scope_breach(changed, target_scope)
            scope_breach: dict[str, Any] = {
                "out_of_scope_paths": out_of_scope_paths,
                "base_oid": pinned_oid,
                "approval_hash": ctx.get("approval_hash"),
                "algo_version": _SCOPE_AUDIT_ALGO_VERSION,
            }
            self._append_integration_event(
                ctx,
                phase="apply",
                pinned_oid=pinned_oid,
                candidate_oid=candidate_oid,
                now=now,
                scope_breach=scope_breach,
            )

            flagged = bool(touched) or stripped

            checks = self._all_checks(ticket_row)
            resources_map = (
                {name: self.check_resources_fn(name) for name in checks}
                if self.check_resources_fn is not None
                else None
            )
            check_results = self.run_checks_fn(
                checks,
                candidate_dir,
                image=self.checker_image,
                flake_reruns=self.flake_reruns,
                resources=resources_map,
            )
            if any(r.outcome != CheckOutcome.PASS for r in check_results):
                return self._die(
                    ticket_id=ticket_id,
                    dispatch_id=dispatch_id,
                    ctx=ctx,
                    pinned_oid=pinned_oid,
                    candidate_oid=candidate_oid,
                    phase="checks-red",
                    outcome="integration-regression",
                    failure_class=FailureClass.INTEGRATION_REGRESSION,
                    disposition="rejected",
                    reason="integration-regression",
                    verdict=None,
                    check_results=check_results,
                    flagged=flagged,
                    touched=touched,
                    detail="full check re-run on the integrated candidate was not all-PASS",
                    now=now,
                )

            artifact = self._build_artifact(ticket_id, candidate_dir, pinned_oid, candidate_oid)
            rubric_items = self._rubric_items(ticket_row)

            try:
                check_evidence = format_check_evidence(check_results)
                verdict, gate_fields, filed_tickets = self.critic.judge(
                    artifact, rubric_items, check_evidence=check_evidence
                )
            except CriticInfraError as exc:
                return self._die(
                    ticket_id=ticket_id,
                    dispatch_id=dispatch_id,
                    ctx=ctx,
                    pinned_oid=pinned_oid,
                    candidate_oid=candidate_oid,
                    phase="gate-infra",
                    outcome="infra",
                    failure_class=FailureClass.INFRA,
                    disposition="parked",
                    reason="critic-infra",
                    verdict=None,
                    check_results=check_results,
                    flagged=flagged,
                    touched=touched,
                    detail="critic call failed — infra, never a rejection; work stays parked",
                    now=now,
                    error=str(exc),
                )

            self._append_gate_event(ctx, verdict=verdict, gate_fields=gate_fields, now=now)
            # D14 (bead .39): verdict-first / filings-after — the paid GATE
            # event is recorded before ANY filing is attempted, and filing
            # happens on BOTH the MET and UNMET paths below (out-of-rubric
            # findings survive even a rejected candidate). Never reached on
            # a CriticInfraError (the `except` above returns first).
            self._file_critic_proposals(ctx, filed_tickets)

            if not verdict.lands():
                return self._die(
                    ticket_id=ticket_id,
                    dispatch_id=dispatch_id,
                    ctx=ctx,
                    pinned_oid=pinned_oid,
                    candidate_oid=candidate_oid,
                    phase="gate-rejected",
                    outcome="rejected",
                    failure_class=FailureClass.REJECTED,
                    disposition="rejected",
                    reason="gate-unmet",
                    verdict=verdict,
                    check_results=check_results,
                    flagged=flagged,
                    touched=touched,
                    detail="critic verdict was UNMET",
                    now=now,
                    gate_fields=gate_fields,
                )

            # -- CAS land (SPEC §9/§10 AC8) --------------------------------
            live_tip = self._rev_parse(self.staging_repo, "refs/heads/staging")
            if live_tip != pinned_oid:
                return self._die(
                    ticket_id=ticket_id,
                    dispatch_id=dispatch_id,
                    ctx=ctx,
                    pinned_oid=pinned_oid,
                    candidate_oid=candidate_oid,
                    phase="abort",
                    outcome="infra",
                    failure_class=FailureClass.INFRA,
                    disposition="parked",
                    reason="concurrent-staging-move",
                    verdict=verdict,
                    check_results=check_results,
                    flagged=flagged,
                    touched=touched,
                    detail="CAS aborted cleanly: staging moved concurrently; candidate NOT landed",
                    now=now,
                )

            self._fetch_candidate_into_staging(candidate_dir, ticket_id)
            update_ref = self._git(
                self.staging_repo,
                ["update-ref", "refs/heads/staging", candidate_oid, pinned_oid],
                check=False,
            )
            if update_ref.returncode != 0:
                # Lost the race between our own re-read above and the
                # update-ref call itself — still a clean CAS abort.
                return self._die(
                    ticket_id=ticket_id,
                    dispatch_id=dispatch_id,
                    ctx=ctx,
                    pinned_oid=pinned_oid,
                    candidate_oid=candidate_oid,
                    phase="abort",
                    outcome="infra",
                    failure_class=FailureClass.INFRA,
                    disposition="parked",
                    reason="concurrent-staging-move",
                    verdict=verdict,
                    check_results=check_results,
                    flagged=flagged,
                    touched=touched,
                    detail="CAS update-ref refused: staging moved concurrently",
                    now=now,
                )

            transition(self.store, ticket_id, LANDED, expected_from=GATED)
            gate_completion_ts = gate_fields.get("ts", now)
            self._append_integration_event(
                ctx, phase="land", pinned_oid=pinned_oid, candidate_oid=candidate_oid,
                now=gate_completion_ts
            )
            reason = self._combine_reason(None, touched, stripped=stripped)
            record_disposition(
                self.record_plane,
                ctx={**ctx, "ticket": ticket_id},
                disposition="landed",
                attempt_kind=ctx["attempt_kind"],
                reason=reason,
            )
            self._journal_append(
                ticket=ticket_id,
                dispatch_id=dispatch_id,
                phase="complete",
                pinned_oid=pinned_oid,
                candidate_oid=candidate_oid,
                resolved_as="landed",
                ts=now,
            )
            return WeaveResult(
                ticket=ticket_id,
                outcome="landed",
                landed_oid=candidate_oid,
                failure_class=None,
                verdict=verdict,
                check_results=check_results,
                flagged_for_human=flagged,
                detail="critic MET; CAS land succeeded",
                reason=None,
                gate_model=gate_fields.get("model"),
                gate_tokens=gate_fields.get("tokens"),
                gate_duration=gate_fields.get("wall_time_seconds"),
            )
        finally:
            self._cleanup_candidate(candidate_dir)

    def _die(
        self,
        *,
        ticket_id: str,
        dispatch_id: Any,
        ctx: dict[str, Any],
        pinned_oid: str,
        candidate_oid: str | None,
        phase: str,
        outcome: str,
        failure_class: FailureClass,
        disposition: str,
        reason: str,
        verdict: Verdict | None,
        check_results: list[CheckResult] | None,
        flagged: bool,
        detail: str,
        now: float,
        touched: list[str] | None = None,
        error: str | None = None,
        gate_fields: dict[str, Any] | None = None,
    ) -> WeaveResult:
        """Shared tail for every non-landing outcome: journal the phase,
        transition to the resting state, write the DISPOSITION event (plus
        a record-plane INTEGRATION event for `phase in {"abort",
        "gate-infra"}` only). Reconciling SPEC §1.0's "INTEGRATION (apply/
        land/abort)" summary against §1.3 step 7's explicit "journal an
        INTEGRATION 'gate-infra' entry (no GATE rejection event)" line: §1.3
        is the detailed normative sequence and separately calls out
        `gate-infra` as its own INTEGRATION entry (distinct from the plain
        `conflict`/`checks-red`/`gate-rejected` journal phases, which §1.3
        does NOT ask for a record-plane INTEGRATION event of their own —
        only the weave journal phase + the DISPOSITION event). See module/
        weave-design notes: this is a deliberate resolution of that
        wording gap, not an accident. Seals the journal `complete` at the
        end. `disposition in {"rejected", "parked"}` maps to
        `REJECTED`/`PARKED` respectively (both legal `GATED->*` edges,
        SPEC §9).

        ``error`` (bead .109): the `str(CriticInfraError)` cause, threaded
        through ONLY for `phase == "gate-infra"` — persisted onto that
        INTEGRATION event so a critic-infra failure is self-diagnosing
        without cross-referencing the weaver's own runtime logs. Every
        other non-landing phase passes `error=None` (the default), which
        `_append_integration_event` omits from the event entirely.
        """
        self._journal_append(
            ticket=ticket_id,
            dispatch_id=dispatch_id,
            phase=phase,
            pinned_oid=pinned_oid,
            candidate_oid=candidate_oid,
            ts=now,
        )
        to_state = REJECTED if disposition == "rejected" else PARKED
        transition(self.store, ticket_id, to_state, expected_from=GATED)
        if phase in ("abort", "gate-infra"):
            self._append_integration_event(
                ctx,
                phase=phase,
                pinned_oid=pinned_oid,
                candidate_oid=candidate_oid,
                now=now,
                error=error if phase == "gate-infra" else None,
            )
        combined_reason = self._combine_reason(reason, touched)
        record_disposition(
            self.record_plane,
            ctx={**ctx, "ticket": ticket_id},
            disposition=disposition,
            attempt_kind=ctx["attempt_kind"],
            reason=combined_reason,
        )
        self._journal_append(
            ticket=ticket_id,
            dispatch_id=dispatch_id,
            phase="complete",
            pinned_oid=pinned_oid,
            candidate_oid=candidate_oid,
            resolved_as="rolled-back",
            ts=now,
        )
        return WeaveResult(
            ticket=ticket_id,
            outcome=outcome,
            landed_oid=None,
            failure_class=failure_class,
            verdict=verdict,
            check_results=check_results,
            flagged_for_human=flagged,
            detail=detail,
            reason=reason,
            gate_model=gate_fields.get("model") if gate_fields is not None else None,
            gate_tokens=gate_fields.get("tokens") if gate_fields is not None else None,
            gate_duration=gate_fields.get("wall_time_seconds") if gate_fields is not None else None,
        )

    @staticmethod
    def _combine_reason(
        reason: str | None, touched: list[str] | None, *, stripped: bool = False
    ) -> str | None:
        """Blend the base disposition reason with the protection-asymmetry
        flag (SPEC §6) onto the SAME disposition event, rather than firing
        a second, premature `record_disposition` call before the ticket's
        final outcome is known (see class docstring / weaver design notes:
        this is a deliberate reading of an underspecified point in the
        design doc). When `stripped` is True (D14, bead .38 AC14 case 6 —
        a committed `.stigmergy/filed-tickets.json` was stripped from the
        candidate at `_apply_bundle` time), also append the
        `filed-tickets-stripped` marker, same semicolon-joined style as
        the `protected-path-touched:` flag."""
        combined = reason
        if touched:
            flag = f"protected-path-touched:{','.join(sorted(touched))}"
            combined = flag if combined is None else f"{combined};{flag}"
        if stripped:
            marker = "filed-tickets-stripped"
            combined = marker if combined is None else f"{combined};{marker}"
        return combined

    # -- context / checks / rubric -------------------------------------------

    def _ctx(self, ticket_row: dict[str, Any]) -> dict[str, Any]:
        ctx = self.ctx_of(ticket_row["id"]) if self.ctx_of is not None else None
        if ctx is None:
            ctx = self._default_ctx(ticket_row)
        missing = [f for f in _CTX_FIELDS if f not in ctx]
        if missing:
            raise WeaverError(f"ctx_of()'s returned dict is missing field(s): {missing}")
        result = dict(ctx)
        result["ticket"] = ticket_row["id"]
        return result

    @staticmethod
    def _default_ctx(ticket_row: dict[str, Any]) -> dict[str, Any]:
        """A conservative, all-zero/None fallback context for identity
        fields, used only when the caller passes no ``ctx_of`` at all."""
        return {
            "rig": None,
            "dispatch_id": ticket_row.get("lease_dispatch_id"),
            "attempt": ticket_row.get("attempts_used") or 0,
            "attempt_kind": "initial",
            "rung": ticket_row.get("current_rung") or "cheap",
            "worker": ticket_row.get("lease_owner"),
            "charter_hash": None,
            "approval_hash": ticket_row.get("approval_hash"),
            "image_digest": None,
            "model": None,
            "model_version": None,
            "price_table_version": None,
            "tokens": dict(_ZERO_TOKENS),
            "computed_usd": 0.0,
            "wall_time_seconds": 0.0,
        }

    @staticmethod
    def _all_checks(ticket_row: dict[str, Any]) -> dict[str, str]:
        """The name->command checks to re-run on the integrated candidate.

        Convention for this module: ``ticket_row["tier1_checks"]`` is a
        flat ``dict[str, str]`` (name -> shell command) — a simpler shape
        than other tickets' fixtures use for the same JSON column, but
        each test module is free to assign its own meaning to a generic
        column; this one is internally consistent within this module and
        its tests. Falls back to a single trivial check if absent so
        ``run_checks_fn`` always has something to iterate.
        """
        checks = ticket_row.get("tier1_checks")
        if isinstance(checks, dict) and checks:
            return checks
        return {"tests": "true"}

    @staticmethod
    def _rubric_items(ticket_row: dict[str, Any]) -> list[str]:
        criteria = ticket_row.get("acceptance_criteria")
        return criteria if isinstance(criteria, list) else []

    def _build_artifact(
        self, ticket_id: str, candidate_dir: Path, pinned_oid: str, candidate_oid: str
    ) -> str:
        diff = self._git(candidate_dir, ["diff", pinned_oid, candidate_oid]).stdout
        return f"ticket: {ticket_id}\ncandidate: {candidate_oid}\n\n{diff}"

    # -- git plumbing ---------------------------------------------------------

    def _clone_candidate(self, pinned_oid: str) -> Path:
        """Build an ISOLATED candidate clone (never a shared worktree) at
        ``pinned_oid`` — a fresh repo that fetches only the `staging`
        branch's history from ``staging_repo`` and never shares `.git`
        metadata with it. Hooks are disabled from the very first command:
        both per-invocation (``-c core.hooksPath=/dev/null``, applied by
        every :meth:`_git` call unconditionally) and persisted on the new
        repo's own config (defense in depth, SPEC §10 AC6).

        Uses `init` + `fetch` (by branch NAME, into a distinctly-namespaced
        remote-tracking ref) + `checkout -B`, rather than `git clone`
        directly — this works uniformly whether ``staging_repo`` is bare
        or non-bare, and whether or not its `HEAD` is a sensible symbolic
        ref (a bare integration repo need carry no checked-out branch at
        all); `git clone`'s own hook-seeding step (via a caller's
        `GIT_TEMPLATE_DIR`) still applies here exactly as it would to a
        plain `clone`, which is what the AC6 test relies on.
        """
        candidate_dir = Path(tempfile.mkdtemp(prefix="stigmergy-weave-"))
        self._git(candidate_dir, ["init", "--quiet"])
        self._git(candidate_dir, ["config", "core.hooksPath", "/dev/null"])
        self._git(
            candidate_dir,
            [
                "fetch",
                "--no-tags",
                "--quiet",
                "--",
                str(self.staging_repo),
                "refs/heads/staging:refs/remotes/origin/staging",
            ],
        )
        self._git(
            candidate_dir,
            ["checkout", "--quiet", "-B", _CANDIDATE_BRANCH, pinned_oid],
        )
        return candidate_dir

    @staticmethod
    def _cleanup_candidate(candidate_dir: Path | None) -> None:
        if candidate_dir is None:
            return
        import shutil

        shutil.rmtree(candidate_dir, ignore_errors=True)

    def _apply_bundle(
        self, candidate_dir: Path, ticket_row: dict[str, Any], pinned_oid: str
    ) -> tuple[bool, str | None, bool]:
        """Fetch ONLY :data:`_BUNDLE_REF` from the ticket's bundle and
        merge it onto the candidate (SPEC §10 AC6 case 9: any other ref
        the bundle carries — including a malicious `refs/heads/staging` —
        is never requested and never reaches any local ref). Returns
        ``(applied, candidate_oid, stripped)``; ``applied is False`` covers
        a missing bundle, a missing/unfetchable ref, a merge conflict, and
        a would-be no-op merge (nothing new to integrate). ``stripped`` is
        ``True`` when a stray, committed `.stigmergy/filed-tickets.json`
        (D14, bead .38 AC14 case 6 — the worker is told never to commit
        this file; the host-side harvest reads it uncommitted) was found
        in the merged candidate tree and removed via an amend of the merge
        commit itself, BEFORE `candidate_oid` is read — so the returned
        oid already reflects the stripped tree and no second candidate
        commit is ever created."""
        bundle_path = ticket_row.get("work_product")
        if not bundle_path:
            return False, None, False

        incoming_ref = f"{_INCOMING_REF_PREFIX}-bundle"
        fetch = self._git(
            candidate_dir,
            [
                "fetch",
                "--no-tags",
                "--quiet",
                "--",
                str(bundle_path),
                f"{_BUNDLE_REF}:{incoming_ref}",
            ],
            check=False,
        )
        if fetch.returncode != 0:
            return False, None, False

        merge = self._git(
            candidate_dir,
            [
                "merge",
                "--no-ff",
                "--no-edit",
                "--quiet",
                incoming_ref,
                "-m",
                f"weave: integrate {ticket_row['id']}",
            ],
            check=False,
        )
        if merge.returncode != 0:
            self._git(candidate_dir, ["merge", "--abort"], check=False)
            return False, None, False

        # D14 (bead .38 AC14 case 6): a worker must never commit its
        # filings file — the host-side harvest reads it uncommitted from
        # the worktree. If one snuck into this merge anyway, strip it by
        # AMENDING the merge commit itself (still exactly one candidate
        # commit, same parents) BEFORE candidate_oid is read below, so
        # every downstream consumer (journal/CAS/checks) sees the already-
        # stripped tree.
        stripped = False
        stray_filings = candidate_dir / ".stigmergy" / "filed-tickets.json"
        if stray_filings.exists():
            self._git(candidate_dir, ["rm", "--quiet", ".stigmergy/filed-tickets.json"])
            self._git(candidate_dir, ["commit", "--amend", "--no-edit", "--quiet"])
            stripped = True

        candidate_oid = self._rev_parse(candidate_dir, _CANDIDATE_BRANCH)
        if candidate_oid == pinned_oid:
            # Nothing new was actually integrated -- treat as a conflict
            # rather than risk an ambiguous candidate_oid == pinned_oid
            # comparison later (in resolve()'s crash-recovery reconciliation).
            # This also covers a bundle whose ONLY change was the stray
            # filings file just stripped above: once removed, there is
            # nothing left to integrate, so it correctly falls into this
            # same non-applying-conflict path.
            return False, None, False
        return True, candidate_oid, stripped

    def _protected_paths_touched(
        self,
        candidate_dir: Path,
        pinned_oid: str,
        candidate_oid: str,
        changed: list[str] | None = None,
    ) -> list[str]:
        if changed is None:
            diff = self._git(
                candidate_dir, ["diff", "--name-only", pinned_oid, candidate_oid]
            ).stdout
            changed = [line.strip() for line in diff.splitlines() if line.strip()]
        return [
            path
            for path in changed
            if any(
                fnmatch.fnmatch(path, pattern) or path == pattern
                for pattern in self.protected_paths
            )
        ]

    def _scope_breach(self, changed: list[str], target_scope: list[str] | None) -> list[str]:
        """Audit the changed paths against the ticket's target_scope.

        Returns the sorted list of changed paths that are OUT OF SCOPE.

        The in-scope matching rule is literal equals-or-nested-under on repo-
        relative, forward-slash paths (NOT fnmatch globbing, diverging from
        `_protected_paths_touched`'s fnmatch because target_scope entries are
        declared literal file/directory paths, not charter globs). A touched
        path P is in scope iff for some scope entry S, after normalizing S
        with `S.rstrip("/")`, `P == S_norm` OR `P.startswith(S_norm + "/")`.
        The mandatory `+ "/"` separator prevents a scope entry `src/foo` from
        matching `src/foobar.py`.

        Edge cases:
        - target_scope is None or [] -> EVERY changed path is out of scope
          (honest reading, mirrors steering.py's `None -> []`).
        - a changed path covered by zero scope entries is out of scope.
        - a changed path that exactly equals or is nested under at least one
          scope entry is in scope and excluded from the result.
        - non-string scope entries are skipped defensively rather than raising,
          so the audit never throws and breaks a weave.
        """
        if not target_scope:
            return sorted(changed)

        def in_scope(path: str) -> bool:
            for scope_entry in target_scope:
                if not isinstance(scope_entry, str):
                    continue
                scope_norm = scope_entry.rstrip("/")
                if path == scope_norm or path.startswith(scope_norm + "/"):
                    return True
            return False

        return sorted([p for p in changed if not in_scope(p)])

    def _fetch_candidate_into_staging(self, candidate_dir: Path, ticket_id: str) -> None:
        """Import the candidate's objects into ``staging_repo`` by
        fetching the candidate's own branch BY NAME (never by raw OID —
        local transports refuse to serve an unadvertised SHA without
        `uploadpack.allow*SHA1InWant`) into a distinctly-namespaced local
        ref, ahead of the actual CAS `update-ref`."""
        target_ref = f"{_INCOMING_REF_PREFIX}/{ticket_id}"
        self._git(
            self.staging_repo,
            [
                "fetch",
                "--no-tags",
                "--quiet",
                "--",
                str(candidate_dir),
                f"{_CANDIDATE_BRANCH}:{target_ref}",
            ],
        )

    def _rev_parse(self, repo: Path, ref: str) -> str:
        return self._git(repo, ["rev-parse", ref]).stdout.strip()

    def _git(
        self, repo: Path | None, args: list[str], *, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        """Run one loop-side git command, ALWAYS with hooks disabled
        (`-c core.hooksPath=/dev/null`, SPEC §10 AC6 defense in depth).
        Inherits the ambient process environment (never stripped) — this
        is host-touching plumbing over trusted repos/paths, not a
        container boundary; the untrusted input this module defends
        against is bundle/tree CONTENT, never the daemon's own env.
        """
        argv = ["git"]
        if repo is not None:
            argv += ["-C", str(repo)]
        # `-c core.hooksPath=/dev/null` (SPEC §10 AC6, always). The
        # committer identity config is harmless noise for read-only/
        # ref-plumbing commands but required for the merge commits this
        # module creates on the candidate clone — set unconditionally
        # rather than threading a separate "needs an identity" flag
        # through every call site.
        argv += [
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "user.email=weaver@stigmergy.invalid",
            "-c",
            "user.name=stigmergy-weaver",
            *args,
        ]
        result = subprocess.run(  # noqa: S603
            argv,
            capture_output=True,
            text=True,
            check=False,
        )
        if check and result.returncode != 0:
            raise WeaverError(
                f"git {' '.join(args)} failed (exit {result.returncode}): {result.stderr.strip()}"
            )
        return result

    # -- record-plane event helpers -------------------------------------------

    def _append_integration_event(
        self,
        ctx: dict[str, Any],
        *,
        phase: str,
        pinned_oid: str | None,
        candidate_oid: str | None,
        now: float,
        error: str | None = None,
        scope_breach: dict[str, Any] | None = None,
    ) -> None:
        fields = {key: ctx.get(key) for key in _CTX_FIELDS}
        fields["ticket"] = ctx["ticket"]
        fields["tokens"] = dict(_ZERO_TOKENS)
        fields["computed_usd"] = 0.0
        fields["wall_time_seconds"] = 0.0
        fields["phase"] = phase
        fields["pinned_oid"] = pinned_oid
        fields["candidate_oid"] = candidate_oid
        fields["ts"] = now
        # bead .109: the CriticInfraError cause, scoped to gate-infra only
        # (every other INTEGRATION event passes error=None -> omitted here,
        # never set to a literal None value on the event itself).
        if error is not None:
            fields["error"] = error
        # Scope-breach audit (bead .102a): a structured record of out-of-scope
        # touched paths, only on the "apply" phase event. Attached when not None
        # following the same optional-field pattern as error above.
        if scope_breach is not None:
            fields["scope_breach"] = scope_breach
        event = make_event(EventType.INTEGRATION, **fields)
        self.record_plane.append(event)

    def _append_gate_event(
        self, ctx: dict[str, Any], *, verdict: Verdict, gate_fields: dict[str, Any], now: float
    ) -> None:
        fields = {key: ctx.get(key) for key in _CTX_FIELDS}
        fields["ticket"] = ctx["ticket"]
        fields["decoding_params"] = gate_fields["decoding_params"]
        fields["prompt_artifact_hash"] = gate_fields["prompt_artifact_hash"]
        fields["model"] = gate_fields.get("model", ctx.get("model"))
        fields["outcome"] = verdict.outcome.value
        fields["tier"] = verdict.tier
        fields["reason"] = verdict.reason
        fields["severity"] = verdict.severity.value
        # Use real completion timestamp from gate_fields, falling back to weave-cycle-start now
        fields["ts"] = gate_fields.get("ts", now)
        # Populate tokens and wall_time_seconds from real gate-call metadata
        if "tokens" in gate_fields:
            fields["tokens"] = gate_fields["tokens"]
        if "wall_time_seconds" in gate_fields:
            fields["wall_time_seconds"] = gate_fields["wall_time_seconds"]
        # Populate repair metadata from gate_fields
        if "repair_attempts" in gate_fields:
            fields["repair_attempts"] = gate_fields["repair_attempts"]
        if "repair_instruction_hash" in gate_fields:
            fields["repair_instruction_hash"] = gate_fields["repair_instruction_hash"]
        event = make_event(EventType.GATE, **fields)
        self.record_plane.append(event)

    def _file_critic_proposals(
        self, ctx: dict[str, Any], filed_tickets: list[dict[str, Any]]
    ) -> None:
        """D14 (bead .39): file the critic's out-of-rubric `filed_tickets`
        via the shared, verbatim-reused :func:`stigmergy.filing.
        file_proposals` core, `origin_role="critic"` — mirrors bead
        `.41`'s range-critic filing (`cli._report_ctx` + the filing call
        in `_cmd_range_report`), except this weaver's `ctx` carries the
        REAL gated ticket id (unlike the range report's `ticket=None`), so
        `discovered_from = "{dispatch_id}@{ticket}"` is honest here.

        `filing_ctx` is exactly the 12 `make_event` common fields MINUS
        the 4 that `file_proposals`/`make_event` supply themselves
        (`attempt_kind`, `tokens`, `computed_usd`, `wall_time_seconds`) —
        including any of those 4 would raise a duplicate-kwarg `TypeError`
        at `_emit`'s `make_event(**ctx, attempt_kind=..., tokens=...)`
        call inside `filing.py`. `file_proposals([], ...)` (the common
        case — most critics file nothing) emits no events and writes
        nothing.
        """
        filing_ctx = {
            "rig": ctx["rig"],
            "ticket": ctx["ticket"],
            "dispatch_id": ctx["dispatch_id"],
            "attempt": ctx["attempt"],
            "rung": ctx["rung"],
            "worker": ctx["worker"],
            "charter_hash": ctx["charter_hash"],
            "approval_hash": ctx["approval_hash"],
            "image_digest": ctx["image_digest"],
            "model": ctx["model"],
            "model_version": ctx["model_version"],
            "price_table_version": ctx["price_table_version"],
        }
        try:
            file_proposals(
                filed_tickets,
                store=self.store,
                record_plane=self.record_plane,
                ctx=filing_ctx,
                attempt_kind=ctx["attempt_kind"],
                origin_role="critic",
                max_filings=self.filing_max_filings,
                max_bytes=self.filing_max_bytes,
            )
        except Exception:
            # Defensive belt (mirrors the daemon's harvest_worker_filings
            # posture): file_proposals is contracted to fold every failure
            # into a recorded rejection and never raise, but the weaver is
            # the single serialized writer — an unexpected filing-side
            # exception here (after the GATE event, before disposition) must
            # never strand the weave at GATED. Filing is strictly additive:
            # losing a filing is acceptable; crashing the writer is not.
            _logger.warning(
                "critic filing failed for dispatch %s (ticket %s)",
                ctx.get("dispatch_id"),
                ctx.get("ticket"),
                exc_info=True,
            )

    # -- journal --------------------------------------------------------------

    def _journal_append(self, **fields: Any) -> None:
        fields.setdefault("ts", time.time())
        line = json.dumps(fields, sort_keys=True, default=str)
        fd = os.open(self.journal_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, _JOURNAL_MODE)
        try:
            with os.fdopen(fd, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        finally:
            os.chmod(self.journal_path, _JOURNAL_MODE)

    def _read_journal(self) -> list[dict[str, Any]]:
        if not self.journal_path.exists():
            return []
        lines: list[dict[str, Any]] = []
        with open(self.journal_path, encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    lines.append(parsed)
        return lines

    def _last_group(self) -> list[dict[str, Any]] | None:
        """The journal lines for the most recently begun ticket (from its
        last `begin` line to the end of the file) — by construction, one
        ticket's cycle always fully seals `complete` before the next
        ticket's `begin` is appended, so at most this last group can ever
        be incomplete."""
        lines = self._read_journal()
        if not lines:
            return None
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].get("phase") == "begin":
                return lines[i:]
        return None
