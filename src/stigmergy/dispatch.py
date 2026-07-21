"""Dispatch station — lane selection, task-pack assembly, per-dispatch
clone, worker naming, capability minting (SPEC.md §3 dispatch station
"Selects lane → driver+model+prompt; provisions container from the
hardened profile; assembles task pack; obtains per-dispatch credential
capability", §4 path safety, §5 `[lanes]`/`[prompts]`, §6 ticket work-order
contract; bead .21 build spec).

**Scope (bead .21 build spec §0.1).** This module *prepares*, it never
spawns and never claims/leases. `.16`'s statemachine.py owns claim/lease/
ladder-walk and hands this module an already-claimed ``ticket_row`` (a dict
as returned by :meth:`stigmergy.rig.RigStore.get_ticket` — JSON columns
already decoded) plus a ``rung`` name; this module only resolves that rung
NAME into an actual lane config and produces a fully prepared
:class:`DispatchPlan`. `.13`'s :func:`stigmergy.drivers.claude_code.spawn`
is called by `.22`, not by this module. Capability revocation is `.22`'s
job — a plan that is prepared but never spawned leaves a live-but-unused
capability; the integration contract for `.22` is to wrap prepare->spawn->
revoke in a `finally` so `revoke(dispatch_id)` always runs, covering the
never-spawned case too (revoke is idempotent).

**Per-dispatch directory layout (bead .21 build spec §0.4/§1, a spec
ambiguity resolved this session — see the sub report).** §0.4 says the
clone "lives at ``<rig>/clones/<dispatch_id>/``"; §1's ``assemble_task_pack``
docstring separately gives the task pack as living "under
``<rig>/clones/<dispatch_id>/task/``". Read literally-nested, the task pack
would sit *inside* the git working tree as an untracked directory — this
module instead treats ``<rig>/clones/<dispatch_id>/`` as the per-dispatch
PARENT directory, with the git clone and the task pack as siblings
underneath it: ``<dispatch_dir>/work/`` (the git clone, checked out on
branch ``work``) and ``<dispatch_dir>/task/`` (the task pack). This keeps
the task pack's ``context/`` files, and `prompt.md`, out of the worker's
own git history entirely.

**Task-pack assembly is all-or-nothing** (mirrors
:func:`stigmergy.pathsafety.safe_extract`'s two-pass validate-then-write
discipline): every ``required_reading`` entry is resolved and validated
(prefix parsed, path-safety checked, size accumulated) *before* anything is
copied. A single bad entry anywhere in the list means nothing from the
whole assembly reaches disk.

**Ticket text is data, never re-interpreted** (§0.3). `prompts/code01`'s
template is filled via :class:`string.Template` — a single-pass
substitution mechanism that never re-scans its own output — so a ticket
goal/acceptance-criteria string containing ``${...}``/``{}``/``%s``-shaped
text lands in `prompt.md` verbatim, never mistaken for template syntax
itself (no nested f-strings, no chained `.format()` calls that could
double-interpret braces present in ticket-authored text).

This module never imports :mod:`stigmergy.records` (deliberate scope
boundary, mirrors `.13`'s own boundary against `.charter`/`.registry`).
"""

from __future__ import annotations

import hashlib
import secrets
import shutil
import sqlite3
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

from stigmergy import pathsafety
from stigmergy.charter import Charter
from stigmergy.drivers import claude_code
from stigmergy.pathsafety import PathSafetyError
from stigmergy.relay import Capability, CapabilityStore
from stigmergy.rig import RigStore

# 8 MiB fixed cap on the SUM of copied required-reading file sizes (v0, not
# charter-configurable — mirrors checks.py's fixed-constant-for-what-the-
# charter-doesn't-expose-a-knob-for-yet pattern).
_MAX_TASK_PACK_BYTES = 8 * 1024 * 1024

# Bounded retry cap for worker-name generation (§0.5): at 2048**3
# combinations, a real collision storm past this many attempts signals
# something else is wrong (a broken rng, a near-exhausted namespace) rather
# than ordinary bad luck.
_NAME_RETRY_LIMIT = 10

# The vendored BIP39 English wordlist (bead .21 build spec §0.5): 2048
# canonical words, one per line, sourced from
# https://raw.githubusercontent.com/bitcoin/bips/master/bip-0039/english.txt
_WORDLIST_PATH = Path(__file__).parent / "data" / "bip39_english.txt"


class DispatchError(Exception):
    """Raised on any ticket-authoring or dispatch-preparation contract
    violation: an unresolvable/unsafe ``required_reading`` path, a task
    pack over the size cap, worker-name generation exhausting its retry
    budget, or a per-dispatch clone's git plumbing failing. Never raised
    for anything that happens once a plan is handed to `spawn()` — that's
    `.13`'s domain."""


@dataclass(frozen=True)
class LaneSelection:
    """One resolved lane, selected by :func:`select_lane` (§0.2)."""

    name: str  # the lane's charter key, e.g. "cheap" / "default" / "exquisite"
    driver: str  # charter lane.driver (v0: always "claude-code")
    model: str  # charter lane.model (registry entry name)
    prompt: str  # charter lane.prompt (prompt id, e.g. "code01")
    egress: list[str]  # charter lane.egress (egress group names)


@dataclass(frozen=True)
class DispatchPlan:
    """The fully prepared output of :func:`prepare_dispatch` — everything
    `.22`'s daemon loop needs to hand straight to
    :func:`stigmergy.drivers.claude_code.spawn` unmodified.

    ``prompt_artifact_hash`` (bead .22 build spec §2, a small additive
    extension to this already-shipped module — mirrors how `.13`
    additively extended `container.py` and `.21` additively extended
    `rig.py`): ``sha256(prompt_template.encode()).hexdigest()``, identical
    mechanism to :class:`stigmergy.critic.Critic`'s own
    ``self.prompt_artifact_hash``. This is a REQUIRED field with no
    default — the SPEC §4 prompt-artifact invariant must never be allowed
    to silently default to an empty string on the DISPATCH event `.22`
    builds from this plan.
    """

    dispatch_id: str  # == worker_name
    worker_name: str
    lane: LaneSelection
    task_pack: Path
    work_clone: Path
    model_cfg: claude_code.ModelConfig
    budgets: claude_code.Budgets
    capability: Capability
    prompt_artifact_hash: str


def _load_wordlist() -> list[str]:
    return _WORDLIST_PATH.read_text(encoding="utf-8").splitlines()


def _default_rng_factory() -> Callable[[], str]:
    """Build a real, `secrets.choice`-backed ``() -> str`` picker over the
    vendored BIP39 wordlist. A fresh picker is built per call to
    :func:`generate_worker_name` (rather than module-level shared state) so
    the wordlist is read from disk lazily and only when actually needed."""
    words = _load_wordlist()

    def _pick() -> str:
        return secrets.choice(words)

    return _pick


# --- §0.2 lane selection -----------------------------------------------


def select_lane(
    charter: Charter, ticket_row: dict[str, Any], *, rung: str | None = None
) -> LaneSelection:
    """Select the lane to dispatch on (bead .21 build spec §0.2).

    Iterates ``charter.raw["lanes"]`` in file order (TOML insertion order
    is preserved by Python's dict). Lanes with ``entry=false`` are skipped
    entirely — those are step-up-only rungs, reached via statemachine.py's
    ladder walk, never eligible at initial dispatch selection. Among the
    remaining entry-eligible lanes, the first whose ``selector.label``
    equals ``ticket_row["lane_hint"]`` wins; if no selector matches
    (including a ``None``/absent ``lane_hint``), falls through to the one
    selector-less lane (charter validation guarantees exactly one exists —
    this fallthrough is not an error, a lane hint is a hint, not a hard
    constraint).

    Does not re-validate lane shape or topology — charter validation
    (already shipped, `.5`) is trusted to have already enforced a
    well-formed topology before this function ever runs. Raises
    :class:`DispatchError` only if charter carries zero entry-eligible
    lanes at all (should be structurally impossible given charter
    validation, but this fails loud rather than raising `KeyError` if it
    ever happens).

    ``rung`` (bead .28): an EXPLICIT override naming a specific lane by its
    charter key. When given (non-None), that lane is returned directly —
    including ``entry=false`` step-up-only rungs, and bypassing selector /
    ``lane_hint`` matching entirely — so a ticket that has structurally
    stepped up (statemachine sets ``current_rung`` to the next rung's lane
    name, e.g. ``"exquisite"``) actually dispatches on it. Raises
    :class:`DispatchError` if ``rung`` names no such lane. When ``rung`` is
    ``None`` (the default) behavior is byte-identical to pre-.28 lane_hint-
    only selection — this MUST stay true: :func:`stigmergy.steering.derive_steering`
    calls the plain, no-rung form, and its approval/steering hash must never
    be perturbed by a step-up (the frozen .35 invariant; see
    tests/test_steering.py case 8).
    """
    lanes = charter.raw.get("lanes") or {}

    if rung is not None:
        cfg = lanes.get(rung)
        if cfg is None:
            raise DispatchError(
                f"rung override {rung!r} names no lane in the charter's "
                "[lanes] — cannot dispatch a step-up rung that does not exist"
            )
        return _build_lane_selection(rung, cfg)

    lane_hint = ticket_row.get("lane_hint")

    fallthrough_name: str | None = None
    fallthrough_cfg: dict[str, Any] | None = None
    entry_eligible_seen = False

    for name, cfg in lanes.items():
        if cfg.get("entry", True) is False:
            continue
        entry_eligible_seen = True

        selector = cfg.get("selector")
        if selector is None:
            if fallthrough_name is None:
                fallthrough_name = name
                fallthrough_cfg = cfg
            continue

        if lane_hint is not None and selector.get("label") == lane_hint:
            return _build_lane_selection(name, cfg)

    if not entry_eligible_seen:
        raise DispatchError(
            "charter carries zero entry-eligible lanes — select_lane has "
            "nothing to select (should be structurally impossible given "
            "charter validation)"
        )

    if fallthrough_name is None or fallthrough_cfg is None:
        raise DispatchError(
            "no selector-less fallthrough lane found among entry-eligible "
            "lanes (should be structurally impossible given charter "
            "validation)"
        )

    return _build_lane_selection(fallthrough_name, fallthrough_cfg)


def _build_lane_selection(name: str, cfg: dict[str, Any]) -> LaneSelection:
    return LaneSelection(
        name=name,
        driver=cfg["driver"],
        model=cfg["model"],
        prompt=cfg["prompt"],
        egress=list(cfg.get("egress") or []),
    )


# --- §0.5 worker naming --------------------------------------------------


def generate_worker_name(
    store: RigStore,
    role: str,
    model: str,
    prompt_id: str,
    *,
    rng: Callable[[], str] | None = None,
) -> str:
    """Generate a rig-unique worker name, ``ROLE-MODEL-PROMPTID-NONCE``
    (bead .21 build spec §0.5), where NONCE is three BIP39 words drawn
    (with replacement across draws — collision is what the uniqueness
    check is for) and hyphen-joined into the same flat name, e.g.
    ``worker-haiku-code01-broom-casino-flock`` (6 hyphen-separated fields
    total, matching SPEC's own example).

    ``rng`` is an injected ``() -> str`` returning one bip39 word (defaults
    to a real `secrets.choice`-backed picker over the vendored wordlist) —
    mirrors the codebase's injected-nondeterminism-for-testing discipline
    (`relay.py`'s `secrets.token_urlsafe`, `critic.py`'s nonce) so tests can
    force a collision deterministically without patching module internals.

    Generates a candidate name, inserts it into the rig's `worker_names`
    table, and retries with a fresh nonce on a primary-key collision
    (`sqlite3.IntegrityError`), up to :data:`_NAME_RETRY_LIMIT` attempts.
    Raises :class:`DispatchError` if the retry budget is exhausted — at
    2048**3 combinations, a real collision storm this deep signals
    something else is wrong, not looping forever.
    """
    picker = rng if rng is not None else _default_rng_factory()

    last_candidate = ""
    for _ in range(_NAME_RETRY_LIMIT):
        nonce = "-".join(picker() for _ in range(3))
        candidate = f"{role}-{model}-{prompt_id}-{nonce}"
        last_candidate = candidate
        try:
            store.insert_worker_name(candidate)
        except sqlite3.IntegrityError:
            continue
        return candidate

    raise DispatchError(
        f"generate_worker_name exhausted its retry budget "
        f"({_NAME_RETRY_LIMIT} attempts, last candidate {last_candidate!r}) "
        "— a collision storm this deep signals something else is wrong"
    )


# --- §0.3 task-pack assembly ---------------------------------------------

_CONTEXT_PREFIX = "context:"
_REPO_PREFIX = "repo:"


def _resolve_required_reading_entry(
    entry: str, *, context_root: Path, repo_root: Path
) -> tuple[str, Path]:
    """Parse one ``required_reading`` entry's explicit prefix and resolve
    it beneath the matching root (§0.3). Returns ``(relative_path,
    resolved_path)``. Raises :class:`DispatchError` for any other prefix
    (or a missing prefix) — ticket-authoring bug, fail loud rather than
    silently skip required context the worker needs to succeed — and for
    any `pathsafety.PathSafetyError` (traversal, absolute path, symlink
    escape, special file)."""
    if entry.startswith(_CONTEXT_PREFIX):
        rel = entry[len(_CONTEXT_PREFIX) :]
        root = context_root
    elif entry.startswith(_REPO_PREFIX):
        rel = entry[len(_REPO_PREFIX) :]
        root = repo_root
    else:
        raise DispatchError(
            f"required_reading entry {entry!r} has an unrecognized prefix "
            "(expected 'context:' or 'repo:') — ticket-authoring bug"
        )

    try:
        resolved = pathsafety.resolve_beneath(root, rel)
        pathsafety.reject_special(resolved)
    except PathSafetyError as exc:
        raise DispatchError(
            f"required_reading entry {entry!r} failed path-safety validation: {exc}"
        ) from exc

    return rel, resolved


def assemble_task_pack(
    ticket_row: dict[str, Any],
    *,
    context_root: Path,
    repo_root: Path,
    prompt_template: str,
    dest: Path,
    prior_evidence: PriorEvidence | None = None,
) -> None:
    """Populate ``dest/prompt.md`` + ``dest/context/...`` (bead .21 build
    spec §0.3). Raises :class:`DispatchError` on any ``required_reading``
    prefix/path violation or the total-bytes cap.

    All-or-nothing: every entry is resolved and validated (prefix parsed,
    path-safety checked, size accumulated) in a first pass, *before*
    anything is copied — mirrors :func:`stigmergy.pathsafety.safe_extract`'s
    validate-then-write discipline. A single bad entry anywhere in the
    list means nothing from the whole assembly reaches disk.
    """
    required_reading = ticket_row.get("required_reading") or []

    # -- pass 1: resolve + validate every entry, accumulate total size ---
    resolved_entries: list[tuple[str, Path]] = []
    total_bytes = 0
    for entry in required_reading:
        rel, resolved = _resolve_required_reading_entry(
            entry, context_root=context_root, repo_root=repo_root
        )
        total_bytes += resolved.stat().st_size
        if total_bytes > _MAX_TASK_PACK_BYTES:
            raise DispatchError(
                f"required_reading total size exceeds _MAX_TASK_PACK_BYTES "
                f"({_MAX_TASK_PACK_BYTES}) — assembly aborted, nothing written"
            )
        resolved_entries.append((rel, resolved))

    # -- pass 2: render prompt.md + copy every validated file -------------
    dest.mkdir(parents=True, exist_ok=True)
    prompt_text = _render_prompt(prompt_template, ticket_row, prior_evidence=prior_evidence)
    (dest / "prompt.md").write_text(prompt_text, encoding="utf-8")

    context_dir = dest / "context"
    for rel, resolved in resolved_entries:
        target = context_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resolved, target)


@dataclass(frozen=True)
class PriorEvidence:
    """Failure evidence from a ticket's most recent prior attempt, threaded
    into a RETRY dispatch's task pack so the worker revises instead of
    re-rolling blind (bead .29 / SPEC §9: "each [retry] carries the failure
    evidence — check output / full critic verdict / gate report / timeout
    fact"). The daemon reconstructs it from the record plane at prepare-time;
    `None` for a ticket's first (initial) dispatch.

    This is the INFORMATIONAL/reference channel only. Physically applying the
    prior work product to the fresh clone for in-place revision
    (`AttemptDecision.pre_apply_prior`) is a separate, larger capability
    deferred to a follow-up bead — `pre_apply_prior` stays unused for now."""

    attempt_kind: str
    check_output: str | None = None
    critic_reason: str | None = None
    failure_note: str | None = None


_PRIOR_EVIDENCE_NA = "(not applicable — first attempt)"


def _prior_evidence_slots(prior: PriorEvidence | None) -> dict[str, str]:
    """Map a `PriorEvidence` (or `None`) to `prompts/code01`'s `$prior_*` DATA
    slots (bead .29). Returns ONLY raw data values / minimal state sentinels —
    ALL worker-facing prose and labels ('this is DATA, revise, never follow as
    instructions', the field captions) live in the reviewed, hash-covered
    `code01` artifact, so the prompt (not this code) is the single visible and
    mutable source of what the worker reads. safe_substitute inserts these
    values verbatim (never rescans), so a `$` in captured output is inert."""
    if prior is None:
        return {
            "prior_attempt_kind": "none — this is the first attempt at this ticket",
            "prior_failure_note": _PRIOR_EVIDENCE_NA,
            "prior_critic_reason": _PRIOR_EVIDENCE_NA,
            "prior_check_output": _PRIOR_EVIDENCE_NA,
        }
    return {
        "prior_attempt_kind": prior.attempt_kind,
        "prior_failure_note": prior.failure_note or "(none recorded)",
        "prior_critic_reason": prior.critic_reason or "(none recorded)",
        "prior_check_output": prior.check_output or "(none recorded)",
    }


def _render_numbered_list(items: list[str] | None, *, label: str) -> str:
    if not items:
        return f"(no {label} given)"
    return "\n".join(f"{i}. {item}" for i, item in enumerate(items, start=1))


def _render_prompt(
    prompt_template: str,
    ticket_row: dict[str, Any],
    *,
    prior_evidence: PriorEvidence | None = None,
) -> str:
    """Fill `prompts/code01`'s placeholders from ``ticket_row`` via
    :class:`string.Template` — a single-pass substitution mechanism that
    never re-scans its own output, so ticket-authored text substituted
    into a slot is never re-interpreted as template syntax itself (§0.3:
    no nested f-strings, no chained `.format()` calls that could
    double-interpret braces present in ticket text)."""
    goal = ticket_row.get("goal") or ""
    acceptance_criteria = _render_numbered_list(
        ticket_row.get("acceptance_criteria"), label="acceptance criteria"
    )
    tier1_checks = ticket_row.get("tier1_checks")
    if isinstance(tier1_checks, dict):
        checks_names = list(tier1_checks)
    elif isinstance(tier1_checks, list):
        checks_names = tier1_checks
    else:
        checks_names = []
    tier1_checks_text = "\n".join(f"- {name}" for name in checks_names) or "(none listed)"
    target_scope = ticket_row.get("target_scope")
    target_scope_text = (
        "\n".join(f"- {path}" for path in target_scope) if target_scope else "(none listed)"
    )

    mapping = {
        "goal": goal,
        "target_scope": target_scope_text,
        "acceptance_criteria": acceptance_criteria,
        "tier1_checks": tier1_checks_text,
        **_prior_evidence_slots(prior_evidence),
    }
    return Template(prompt_template).safe_substitute(mapping)


# --- §0.4 per-dispatch clone ----------------------------------------------


def _git(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one git command with hooks disabled (`-c
    core.hooksPath=/dev/null`, mirrors `weaver.py`'s/`rangereport.py`'s
    discipline exactly). Belt-and-braces here rather than load-bearing: the
    source being cloned is the rig's own trusted repo, not yet
    worker-touched."""
    argv = ["git", "-c", "core.hooksPath=/dev/null", *args]
    return subprocess.run(argv, capture_output=True, text=True, check=False)  # noqa: S603


def _git_in(repo: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one git command against ``repo`` with hooks disabled (`-c
    core.hooksPath=/dev/null`)."""
    argv = ["git", "-C", str(repo), "-c", "core.hooksPath=/dev/null", *args]
    return subprocess.run(argv, capture_output=True, text=True, check=False)  # noqa: S603


def create_work_clone(rig_repo: Path, dest: Path, *, dispatch_base: str) -> None:
    """Create the per-dispatch worker clone (bead .21 build spec §0.4):
    `git clone --no-hardlinks` from ``rig_repo`` into ``dest``, check out
    ``dispatch_base`` at its current tip, then `checkout -b work`.

    `--no-hardlinks` so the per-dispatch clone shares NO object storage
    with the rig's own trusted repo (mirrors the "no shared `.git`
    metadata" invariant, SPEC §4). Every git call here uses `-c
    core.hooksPath=/dev/null`. Raises :class:`DispatchError` (wrapping the
    subprocess failure) on any git error — never a raw
    `subprocess.CalledProcessError` escaping.
    """
    clone_result = _git(["clone", "--no-hardlinks", "--", str(rig_repo), str(dest)])
    if clone_result.returncode != 0:
        raise DispatchError(
            f"git clone of {rig_repo!r} into {dest} failed "
            f"(exit {clone_result.returncode}): {clone_result.stderr.strip()}"
        )

    checkout_base = _git_in(dest, ["checkout", dispatch_base])
    if checkout_base.returncode != 0:
        raise DispatchError(
            f"git checkout of dispatch_base {dispatch_base!r} in {dest} failed "
            f"(exit {checkout_base.returncode}): {checkout_base.stderr.strip()}"
        )

    checkout_work = _git_in(dest, ["checkout", "-b", "work"])
    if checkout_work.returncode != 0:
        raise DispatchError(
            f"git checkout -b work in {dest} failed "
            f"(exit {checkout_work.returncode}): {checkout_work.stderr.strip()}"
        )


# --- §0.6 / composition: prepare_dispatch ---------------------------------


def prepare_dispatch(
    *,
    charter: Charter,
    ticket_row: dict[str, Any],
    store: RigStore,
    capability_store: CapabilityStore,
    rig_repo: Path,
    context_root: Path,
    clones_root: Path,
    prompts_dir: Path,
    relay_base_url: str,
    image: str,
    egress_socket: Path | str | None = None,
    relay_socket: Path | str | None = None,
    prior_evidence: PriorEvidence | None = None,
    dispatch_id: str | None = None,
) -> DispatchPlan:
    """The one entry point `.22` calls (bead .21 build spec §1). Composes
    :func:`select_lane` + :func:`generate_worker_name` +
    :func:`create_work_clone` + :func:`assemble_task_pack` +
    ``capability_store.mint`` into one :class:`DispatchPlan`.

    Never claims/leases the ticket (statemachine.py's job, already done by
    the time this is called) and never calls
    :func:`stigmergy.drivers.claude_code.spawn` (`.22`'s job).

    Per-dispatch directory layout (see module docstring): the git clone and
    the task pack are created as SIBLINGS under
    ``<clones_root>/<dispatch_id>/`` — ``work/`` (the git clone) and
    ``task/`` (the task pack) — so the task pack never lands inside the
    worker's own git working tree.

    The capability is minted LAST, only once the fallible filesystem/git
    work (clone + task-pack assembly) has already succeeded — minimizing
    the window during which a minted-but-unused capability could exist if
    this function itself were interrupted partway through (see module
    docstring §0.1 on `.22`'s prepare->spawn->revoke `finally` contract).

    If ``dispatch_id`` is provided, it is reused verbatim as the dispatch's
    id (the worker name) instead of calling :func:`generate_worker_name`.
    When not provided (None), :func:`generate_worker_name` is called to
    mint a fresh id as before.
    """
    lane = select_lane(charter, ticket_row, rung=ticket_row.get("current_rung"))

    if dispatch_id is None:
        worker_name = generate_worker_name(store, "worker", lane.model, lane.prompt)
        dispatch_id = worker_name
    else:
        worker_name = dispatch_id

    dispatch_dir = clones_root / dispatch_id
    work_clone = dispatch_dir / "work"
    task_pack = dispatch_dir / "task"

    dispatch_base = charter.raw["tiers"]["dispatch_base"]
    create_work_clone(rig_repo, work_clone, dispatch_base=dispatch_base)

    prompt_path = prompts_dir / lane.prompt
    prompt_template = prompt_path.read_text(encoding="utf-8")
    prompt_artifact_hash = hashlib.sha256(prompt_template.encode()).hexdigest()
    assemble_task_pack(
        ticket_row,
        context_root=context_root,
        repo_root=rig_repo,
        prompt_template=prompt_template,
        dest=task_pack,
        prior_evidence=prior_evidence,
    )

    dispatch_limits = charter.raw["loop"]["dispatch_limits"]
    output_tokens = dispatch_limits["output_tokens"]
    driver_turns = dispatch_limits["driver_turns"]

    capability = capability_store.mint(
        dispatch_id, max_output_tokens=output_tokens, max_calls=driver_turns
    )

    model_cfg = claude_code.ModelConfig(
        model=lane.model,
        image=image,
        relay_base_url=relay_base_url,
        egress_socket=egress_socket,
        relay_socket=relay_socket,
    )
    budgets = claude_code.Budgets(output_tokens=output_tokens, driver_turns=driver_turns)

    return DispatchPlan(
        dispatch_id=dispatch_id,
        worker_name=worker_name,
        lane=lane,
        task_pack=task_pack,
        work_clone=work_clone,
        model_cfg=model_cfg,
        budgets=budgets,
        capability=capability,
        prompt_artifact_hash=prompt_artifact_hash,
    )
