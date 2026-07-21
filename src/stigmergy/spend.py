"""Spend-leash accounting + exhaustion protocol (SPEC.md §9 Budgets, §5
`[loop.budgets]`, §8 spend accounting, §12 measurement, §10 AC10).

Two fail-closed leashes, both cumulative per run:

- **`dispatches`** — a plain count of dispatches (any pricing class).
- **`usd`** — estimated **metered-only** spend across dispatches AND gate
  calls. `local`/`subscription` marginal cost is real money-shaped
  bookkeeping for the run report, but it never feeds this leash — SPEC §9:
  "the USD leash cannot see them" (their marginal cost is $0 by
  definition, declared and human-approved via the registry). Those lanes
  are leashed instead by dispatch count, per-dispatch ceilings, and wall
  clock (owned elsewhere in the loop, not this module).

**Reserved allowance.** A slice of the `usd` cap (`reserve_usd`) is held
back so the run's *final* weave cycle can always execute even after the
leash trips — landing whatever was already paid for. Normal dispatching
and gating stop as soon as accrued metered spend reaches the
non-reserved envelope (`usd_cap - reserve_usd`); the reserve itself is
only spendable through the one-shot `final_weave_allowed()` gate.

**Unknown pricing = unbudgetable = refuse to start.** This module never
prices anything itself — it always resolves through `stigmergy.registry`,
which raises `UnbudgetableError` on a registry miss or an invalid entry.
`preflight()` turns that into a whole-run refusal before any dispatching
happens (AC10: "unknown-model run refuses to start").

**Recording never self-blocks.** `record_dispatch`/`record_gate` always
accrue their real cost and counts, even past exhaustion — SPEC §9's
"in-flight finishes and is recorded" requires that a dispatch or gate call
already underway when the leash trips can still land its facts. All
fail-closed gating lives in `can_dispatch()`/`can_gate()`/
`final_weave_allowed()`, which callers MUST consult before starting new
work, never in the record path. The one exception: a metered call whose
tokens fall in a rate category the registry didn't declare a price for is
genuinely unbudgetable (should not happen post-registry-validation) and
`record_*` raises `SpendError` rather than silently pricing it at $0.

**No auto-renewal.** There is no method anywhere in this module that
re-arms, resets, or increases a leash in place. Once a cap is hit,
`can_dispatch()`/`can_gate()` stay `False` for the life of this
`SpendLeash` instance; resumption is a brand-new run with a brand-new
human-granted `SpendLeash`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from stigmergy.registry import ModelEntry, PricingClass, Registry, UnbudgetableError


class SpendError(Exception):
    """Raised on refuse-to-start (unbudgetable model at preflight) or on
    an attempted illegal renewal of an exhausted leash. This module
    exposes no renewal method at all — the second case is defensive: if
    any future caller ever adds one, it must raise this, never silently
    re-arm."""


@dataclass(frozen=True)
class Budgets:
    """One run's leash configuration (SPEC.md §5 `[loop.budgets]`).

    ``reserve_usd`` is held back out of ``usd`` for the final weave cycle
    (SPEC §9) — normal dispatching/gating stops once accrued metered
    spend would leave less than ``reserve_usd`` of headroom under
    ``usd``, i.e. once ``metered_spent >= usd - reserve_usd``.
    """

    dispatches: int
    usd: float
    gate_calls: int
    reserve_usd: float


_UNTRACKED_TICKET = "__untracked__"
"""Sentinel bucket in `run_report()['per_ticket_dispatches']` for dispatches
recorded without a `ticket=` kwarg — makes an attribution gap visible in
the report rather than letting it silently vanish from the distribution."""


# Token-category -> the ModelEntry attribute holding that category's
# per-mtok rate. Mirrors records.py's `_TOKEN_KEYS` ordering/spelling.
_METERED_RATE_FIELDS: dict[str, str] = {
    "in": "input_usd_per_mtok",
    "out": "output_usd_per_mtok",
    "cached": "cached_usd_per_mtok",
    "reasoning": "reasoning_usd_per_mtok",
}


def cost_usd(entry: ModelEntry, tokens: dict[str, int]) -> float | str:
    """Compute one dispatch/gate call's USD cost from its registry entry.

    - ``METERED``: sum over token categories present with count > 0 of
      ``count * rate / 1_000_000``. A category with tokens but no declared
      rate is genuinely unbudgetable (should not happen post-registry-
      validation: metered entries always declare non-zero `in`/`out`
      rates; `cached`/`reasoning` are optional) — returns the literal
      string ``"unbudgetable"``, never a silent $0.
    - ``LOCAL``: the declared marginal cost (``entry.marginal_usd``), which
      may legitimately be exactly 0.0 — a declared value, not a fallback.
    - ``SUBSCRIPTION``: exactly 0.0 (subscription marginal cost is always
      0.0; quota exhaustion is a distinct infra signal, handled outside
      this module, never a USD event).
    """
    if entry.pricing is PricingClass.METERED:
        total = 0.0
        for token_key, rate_field in _METERED_RATE_FIELDS.items():
            count = tokens.get(token_key, 0)
            if not count:
                continue
            rate = getattr(entry, rate_field)
            if rate is None:
                return "unbudgetable"
            total += count * rate / 1_000_000.0
        return total
    if entry.pricing is PricingClass.LOCAL:
        marginal = entry.marginal_usd
        if marginal is None:
            return "unbudgetable"
        return marginal
    if entry.pricing is PricingClass.SUBSCRIPTION:
        return 0.0
    raise AssertionError("unreachable: PricingClass has no other members")  # pragma: no cover


class SpendLeash:
    """Fail-closed spend accounting for one run.

    All mutation happens through `record_dispatch`/`record_gate`; all
    gating happens through `can_dispatch`/`can_gate`/`final_weave_allowed`.
    Nothing here is resettable — see the module docstring's "no
    auto-renewal" invariant.
    """

    def __init__(
        self,
        budgets: Budgets,
        registry: Registry,
        *,
        events: list[dict[str, Any]] | None = None,
    ) -> None:
        self._budgets = budgets
        self._registry = registry

        self._dispatch_count = 0
        self._gate_count = 0
        self._metered_spent = 0.0
        self._spend_by_class: dict[str, float] = {cls.value: 0.0 for cls in PricingClass}
        self._per_ticket_dispatches: dict[str, int] = {}

        self._final_weave_consumed = False
        self._was_exhausted = False
        self._notification_log: list[dict[str, Any]] = []

        # Seed initial state from prior events if provided
        if events:
            self._seed_from_events(events)

    def _seed_from_events(self, events: list[dict[str, Any]]) -> None:
        """Seed the leash's initial state from prior events, using the same
        counting rules as status.reconstruct_spend."""
        for ev in events:
            usd = ev.get("computed_usd")
            if isinstance(usd, str):
                # "unbudgetable" contributes nothing to the USD total
                pass
            elif isinstance(usd, (int, float)) and not isinstance(usd, bool):
                # Numeric (non-bool) values contribute to metered spend
                self._metered_spent += float(usd)

            # Count dispatch/gate events by event_type
            event_type = ev.get("event_type")
            if event_type == "dispatch":
                self._dispatch_count += 1
            elif event_type == "gate":
                self._gate_count += 1

    # -- refuse-to-start -------------------------------------------------

    def preflight(self, models_in_use: list[str]) -> None:
        """Refuse to start the whole run if any named model is unbudgetable.

        Reuses `Registry.resolve`, which raises `UnbudgetableError` on a
        registry miss or an invalid entry; this re-raises as `SpendError`
        so callers only need to catch one exception type from this module.
        """
        for model in models_in_use:
            try:
                self._registry.resolve(model)
            except UnbudgetableError as exc:
                raise SpendError(
                    f"refuse to start: model {model!r} is unbudgetable: {exc}"
                ) from exc

    # -- recording (never self-blocks) -----------------------------------

    def _resolve(self, model: str) -> ModelEntry:
        try:
            return self._registry.resolve(model)
        except UnbudgetableError as exc:
            raise SpendError(f"unbudgetable model {model!r} at record time: {exc}") from exc

    def _accrue(self, entry: ModelEntry, tokens: dict[str, int]) -> float:
        cost = cost_usd(entry, tokens)
        if cost == "unbudgetable":
            raise SpendError(
                f"model {entry.name!r} produced an unbudgetable cost for tokens "
                f"{tokens!r} (metered rate missing for a nonzero category) — "
                "refusing to silently price this at $0"
            )
        cost = float(cost)
        self._spend_by_class[entry.pricing.value] += cost
        if entry.pricing is PricingClass.METERED:
            self._metered_spent += cost
        return cost

    def _note_exhaustion_transition(self) -> None:
        if not self._was_exhausted and self.exhausted():
            self._was_exhausted = True
            self._notification_log.append(self.notification_intent())

    def record_dispatch(
        self, model: str, tokens: dict[str, int], *, ticket: str | None = None
    ) -> float:
        """Accrue one dispatch's cost + count. Always records, even past
        exhaustion — callers gate NEW dispatches via `can_dispatch()`
        beforehand, not by expecting this to refuse."""
        entry = self._resolve(model)
        cost = self._accrue(entry, tokens)
        self._dispatch_count += 1
        # A dispatch recorded without `ticket=` still counts toward
        # `dispatches_used` — it must not silently vanish from the
        # distribution, or the report's diagnosis-by-distribution (SPEC §9)
        # could under-report without any signal. Bucket it under an
        # explicit sentinel key instead so the gap is visible, not silent.
        bucket = ticket if ticket is not None else _UNTRACKED_TICKET
        self._per_ticket_dispatches[bucket] = self._per_ticket_dispatches.get(bucket, 0) + 1
        self._note_exhaustion_transition()
        return cost

    def record_gate(self, model: str, tokens: dict[str, int]) -> float:
        """Accrue one gate call's cost + count. Always records, even past
        exhaustion (including the one final-weave gate call, which spends
        out of the reserve) — callers gate NEW gate calls via
        `can_gate()`/`final_weave_allowed()` beforehand."""
        entry = self._resolve(model)
        cost = self._accrue(entry, tokens)
        self._gate_count += 1
        self._note_exhaustion_transition()
        return cost

    # -- gating (fail-closed) ---------------------------------------------

    def _usd_leash_tripped(self) -> bool:
        """True once accrued metered spend would leave the reserve exposed:
        `metered_spent >= usd_cap - reserve_usd`. Checked as `>=` rather
        than a strict `>` so the leash trips at the instant the
        non-reserved envelope is fully used, before any reserve dollar is
        touched by normal (non-final-weave) activity."""
        non_reserved_cap = self._budgets.usd - self._budgets.reserve_usd
        return self._metered_spent >= non_reserved_cap

    def can_dispatch(self) -> bool:
        if self._dispatch_count >= self._budgets.dispatches:
            return False
        return not self._usd_leash_tripped()

    def can_gate(self) -> bool:
        if self._gate_count >= self._budgets.gate_calls:
            return False
        return not self._usd_leash_tripped()

    def exhausted(self) -> bool:
        """True if ANY leash (dispatch count, gate count, or USD+reserve
        guard) has tripped. Exhaustion is a single planned authority
        boundary (SPEC §9), not per-leash: once any cap trips, the run
        stops claiming/dispatching/gating as a whole and moves to the one
        allowed final weave cycle."""
        return (
            self._dispatch_count >= self._budgets.dispatches
            or self._gate_count >= self._budgets.gate_calls
            or self._usd_leash_tripped()
        )

    def final_weave_allowed(self) -> bool:
        """One-shot: `True` exactly once, on the first call after
        `exhausted()` becomes `True`; every call before exhaustion, and
        every call after the first post-exhaustion call, returns `False`.
        Calling this consumes the allowance — it is a side-effecting
        predicate, not a pure query (use `exhausted()` for a pure read)."""
        if not self.exhausted():
            return False
        if self._final_weave_consumed:
            return False
        self._final_weave_consumed = True
        return True

    # -- reporting ---------------------------------------------------------

    def notification_intent(self) -> dict[str, Any]:
        """A persisted-intent data structure describing the leash's current
        exhaustion state. This module only BUILDS the intent; actually
        sending/persisting it is a different subsystem's job (SPEC §9
        "Notifications": persisted with retry, surfaced via `status`)."""
        non_reserved_cap = self._budgets.usd - self._budgets.reserve_usd
        reasons = []
        if self._dispatch_count >= self._budgets.dispatches:
            reasons.append("dispatches")
        if self._gate_count >= self._budgets.gate_calls:
            reasons.append("gate_calls")
        if self._usd_leash_tripped():
            reasons.append("usd")
        return {
            "exhausted": self.exhausted(),
            "reasons": reasons,
            "metered_spent": self._metered_spent,
            "usd_cap": self._budgets.usd,
            "reserve_usd": self._budgets.reserve_usd,
            "non_reserved_cap": non_reserved_cap,
            "dispatch_count": self._dispatch_count,
            "dispatches_cap": self._budgets.dispatches,
            "gate_count": self._gate_count,
            "gate_calls_cap": self._budgets.gate_calls,
            "final_weave_consumed": self._final_weave_consumed,
            "ts": time.time(),
        }

    def run_report(self, *, landed: int = 0) -> dict[str, Any]:
        """The run report (SPEC §9/§12): dispatches used vs a caller-
        supplied landed count, spend broken out by pricing class,
        escalations, per-ticket dispatch distribution, reserve state."""
        non_reserved_cap = self._budgets.usd - self._budgets.reserve_usd
        reserve_used = max(0.0, self._metered_spent - non_reserved_cap)
        reserve_remaining = max(0.0, self._budgets.reserve_usd - reserve_used)
        return {
            "dispatches_used": self._dispatch_count,
            "dispatches_cap": self._budgets.dispatches,
            "landed": landed,
            "gate_calls_used": self._gate_count,
            "gate_calls_cap": self._budgets.gate_calls,
            "metered_spent": self._metered_spent,
            "spend_by_pricing_class": dict(self._spend_by_class),
            "escalations": len(self._notification_log),
            "per_ticket_dispatches": dict(self._per_ticket_dispatches),
            "reserve": {
                "reserve_usd": self._budgets.reserve_usd,
                "reserve_used": reserve_used,
                "reserve_remaining": reserve_remaining,
                "reserve_intact": reserve_used <= 0.0,
            },
            "exhausted": self.exhausted(),
            "final_weave_consumed": self._final_weave_consumed,
        }
