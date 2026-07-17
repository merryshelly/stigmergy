"""Real `steering_of` derivation (SPEC.md §4 "Approval integrity"; bead
`.35` build spec).

**This function is the single authoritative derivation of a ticket's
STEERING dict.** It MUST be called identically both by whatever future
tool performs `approval.approve()` (to compute what a human signs) and by
`intake.eligible()`'s injected `steering_of` callable (to compute the live
comparison) — that wiring is `.27`'s job, not this bead's. Any caller that
re-derives this shape independently, rather than calling
:func:`derive_steering`, breaks the approval-hash integrity mechanism:
`approval.is_approval_valid` only means anything if "what a human signed"
and "what is being compared against now" are computed the exact same way
from the exact same inputs.

`approval.py`'s own docstring frames STEERING as: ticket text, checks,
rubric, lane, prompt bytes, context set. **`target_scope` is a deliberate
extension beyond that originally-documented list of 6 categories** (bead
`.35` build spec §6 "Protection asymmetry"): `target_scope` constrains what
a worker may touch, so excluding it from the signed hash would let
write-scope widen after human approval with zero re-approval required —
defeating the point of the mechanism. This extension is intentional and
should not be read as accidental scope creep; flagged here and in the
bead's sub-report per the build spec's explicit instruction.

**Frozen invariant: lane resolution never consults `current_rung`.**
`lane` is derived via :func:`stigmergy.dispatch.select_lane`'s plain,
no-override, hint-only resolution path (reads only
``ticket_row["lane_hint"]``) — `select_lane` today already ignores
``ticket_row["current_rung"]`` entirely, and this module must never change
or work around that. `current_rung` is loop-driven EXECUTION state (SPEC's
drift-without-reapproval carve-out), not a steering judgment input. A
separate, already-queued bead (`.28`) will give `select_lane`/
`prepare_dispatch` an explicit override path for step-up dispatch
EXECUTION; `derive_steering` must keep calling the plain resolution path
even after `.28` lands, because if `lane` in the steering dict ever
reflected a stepped-up rung, every step-up would change the steering hash,
kick the ticket ineligible, and deadlock the retry ladder pending human
re-approval — exactly the failure mode this invariant prevents. This is
tested explicitly (`tests/test_steering.py`'s current_rung-invariance
case): two otherwise-identical ticket rows differing only in
``current_rung`` must derive byte-for-byte identical steering dicts.

This module deliberately does NOT import :mod:`stigmergy.approval` —
`approval.py` stays a zero-dependency pure-hash module (confirmed by an
`opus` advisor consult during `.35`'s build): the isolation is deliberate,
so this module never becomes a dependency `approval.py` has to carry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from stigmergy.charter import Charter
from stigmergy.dispatch import select_lane


class SteeringError(Exception):
    """Raised when a ticket's steering fields cannot be resolved into a
    dict at all — e.g. the resolved lane's prompt template file is
    missing from ``prompts_dir``. Fail closed: never silently hash an
    empty/partial steering dict.

    Note: :class:`stigmergy.dispatch.DispatchError` (raised by
    `select_lane` when no lane can be resolved at all) is a distinct,
    uncaught failure mode — a chartering bug, not a steering-derivation
    bug — and is allowed to propagate through :func:`derive_steering`
    unwrapped, never converted to a `SteeringError`.
    """


def derive_steering(
    ticket_row: dict[str, Any],
    charter: Charter,
    prompts_dir: str | Path,
) -> dict[str, Any]:
    """Derive the full STEERING dict for ``ticket_row`` (bead `.35` build
    spec, "Field derivation (exact, frozen)").

    Field-by-field (frozen, do not deviate):

    - ``ticket_text``: ``f"{title}\\n{goal or ''}"``. ``title`` is
      ``NOT NULL`` in the ticket schema (always present); ``goal`` may be
      ``None``, treated as ``""``.
    - ``checks``: ``ticket_row.get("tier1_checks")`` passed through AS-IS
      (whatever JSON shape the ticket author set — list or dict, per this
      codebase's existing per-module-interprets-this-column precedent,
      e.g. `weaver.py`'s own `_all_checks`). ``None`` -> ``[]``.
    - ``rubric``: ``ticket_row.get("acceptance_criteria")``, mirroring
      `weaver.py`'s ``_rubric_items`` exactly — ``None``/non-list ->
      ``[]`` (a stricter fallback than ``checks``' — this asymmetry is
      deliberate and frozen by the build spec, not a bug to reconcile).
    - ``target_scope``: ``ticket_row.get("target_scope")`` as-is,
      ``None`` -> ``[]``. See module docstring: a deliberate extension
      beyond `approval.py`'s originally-documented 6 STEERING categories.
    - ``lane``: the RESOLVED entry-lane name via
      ``dispatch.select_lane(charter, ticket_row)`` — never
      ``ticket_row["lane_hint"]`` raw, and never consulting
      ``ticket_row["current_rung"]`` (frozen invariant, see module
      docstring). `select_lane` may raise
      :class:`stigmergy.dispatch.DispatchError`; it propagates uncaught
      (a chartering bug, not a steering-derivation bug).
    - ``prompt_bytes``: the raw on-disk text of
      ``Path(prompts_dir) / lane_selection.prompt`` — the exact template a
      human reviews under `.26`'s sign-off, never the ticket-filled
      `prompt.md` (a per-dispatch artifact, chicken-and-egg — never read
      here). A missing/unreadable file raises :class:`SteeringError`
      (fail closed — never hash empty/missing bytes as if the prompt were
      blank).
    - ``context_set``: ``ticket_row.get("required_reading")`` as-is,
      ``None`` -> ``[]``.
    - ``functional_summary``: ``ticket_row.get("functional_summary")`` as-is,
      ``None``/missing -> ``""`` (mirrors ``goal``). The 8th signed steering
      field (SPEC §6 item 10 / D15, bead `.42`): the operator-facing summary
      the human judges at triage. Being a steering field, it is covered by the
      approval hash automatically — a post-approval edit forces re-approval.

    **Frozen 8-field contract** (bead `.42` extends `.35`'s original 7):
    ``ticket_text, checks, rubric, target_scope, functional_summary, lane,
    prompt_bytes, context_set``.

    Pure function: no I/O beyond the one prompt-template read, no
    timestamps, no unstable-ordering containers (everything returned is
    JSON-serializable via ``lists``, never ``set``) — calling this twice
    on the same inputs returns byte-for-byte equal dicts.
    """
    title = ticket_row["title"]
    goal = ticket_row.get("goal") or ""
    ticket_text = f"{title}\n{goal}"

    checks = ticket_row.get("tier1_checks")
    if checks is None:
        checks = []

    rubric = ticket_row.get("acceptance_criteria")
    rubric = rubric if isinstance(rubric, list) else []

    target_scope = ticket_row.get("target_scope")
    if target_scope is None:
        target_scope = []

    context_set = ticket_row.get("required_reading")
    if context_set is None:
        context_set = []

    # 8th steering field (SPEC §6 item 10 / D15, bead .42): the plain-language,
    # operator-facing statement the human actually judges at triage. Fail closed
    # to "" (mirrors `goal`); it is a STEERING field, so it flows into the
    # approval hash automatically (approve() hashes whatever this returns) —
    # mutating it post-approval de-eligibilizes the ticket.
    functional_summary = ticket_row.get("functional_summary") or ""

    # Plain, no-override, hint-only resolution — reads only
    # ticket_row["lane_hint"]; never ticket_row["current_rung"]. Let
    # DispatchError propagate uncaught.
    lane_selection = select_lane(charter, ticket_row)

    prompt_path = Path(prompts_dir) / lane_selection.prompt
    try:
        prompt_bytes = prompt_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SteeringError(
            f"prompt template {prompt_path} for resolved lane "
            f"{lane_selection.name!r} could not be read: {exc}"
        ) from exc

    return {
        "ticket_text": ticket_text,
        "checks": checks,
        "rubric": rubric,
        "target_scope": target_scope,
        "functional_summary": functional_summary,
        "lane": lane_selection.name,
        "prompt_bytes": prompt_bytes,
        "context_set": context_set,
    }
