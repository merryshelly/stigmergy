"""Verdict schema (SPEC.md §3 objects: `verdict`, §4 "severity is recorded
but does not drive landing" / D4).

A `verdict` is `{outcome, tier, reason, severity}` — structured, never free
text (SPEC §3). `Outcome` is the only field that drives landing in v0:
`Verdict.lands()` is `True` iff `outcome is Outcome.MET`, independent of
`severity`. Severity is banked data for later analysis (defect-escape
tracking, cross-family bias signal, §4/§10 AC1 D10 note) — not a landing
gate in v0. A future revision may wire severity into the landing decision;
that is a deliberate, separately-reviewed change, not an accident of this
module.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class Outcome(enum.Enum):
    MET = "met"
    UNMET = "unmet"


class Severity(enum.Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class Verdict:
    """One structured critic verdict.

    ``tier`` identifies which rubric tier the verdict is judging (SPEC §6:
    tickets carry Tier-1 checks + Tier-2 rubric); ``reason`` is a required
    non-empty human-readable justification — never a free-form blob standing
    in for structure, but structure always carries a reason.
    """

    outcome: Outcome
    tier: int
    reason: str
    severity: Severity

    def lands(self) -> bool:
        """True iff this verdict clears the gate — outcome only.

        Severity is recorded (banked for defect-escape analysis) but does
        NOT gate landing in v0 (SPEC §4/D4): a MET verdict lands regardless
        of severity; an UNMET verdict never lands regardless of severity.
        """
        return self.outcome is Outcome.MET
