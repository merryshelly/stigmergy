"""Deterministic manifest validator (the decomposer station).

Validates a decomposer-produced JSON manifest — a list of ticket work
orders in the §6 ticket contract (the SAME key vocabulary as the triage
promotion spec, ``triage._REQUIRED_PROMOTION_KEYS`` /
``_OPTIONAL_PROMOTION_KEYS`` — deliberately STRicter than ``intake``,
which does not require ``tier1_checks``) — against a fixed rule table.

Pure and read-only: no disk writes, no store mutations, no network. Given
the same inputs it always returns the same output: defects are emitted in
a data-determined order (manifest-level checks first, then per-entry rules
in manifest order, then cross-ticket rules), and all set-based
comparisons are reported in sorted order.

Rule table (the rule id is the second field of every defect string):

    R1   manifest is a JSON array; every entry is an object
    R2   required keys: id, title, functional_summary, acceptance_criteria,
         tier1_checks, target_scope
    R3   functional_summary: non-empty string after strip
    R4   acceptance_criteria: array of strings, each non-empty after strip,
         length >= 1 (a non-list silently degrades to an empty critic
         rubric downstream — load-bearing)
    R5   tier1_checks: a dict (NEVER a list — lists misparse as pytest
         paths downstream); keys non-empty strings; with a charter, every
         key must name a [checks.*] section, every [gates] attempt/staging
         check must be present as a key, and each value string must EQUAL
         the check's [checks.<name>].cmd verbatim
    R6   ids unique across the manifest, kebab-case, and (with a store)
         no collision with an existing ticket id
    R7   blocks: every reference resolves to a manifest id or (with a
         store) an existing store ticket; no self-reference; the
         manifest-internal block graph is ACYCLIC (one defect naming the
         cycle path, e.g. ``a -> b -> a``)
    R8   target_scope: non-empty array; entries relative (absolute ->
         defect; any ``..`` segment -> defect); with a repo, each path
         either exists in the repo or is a NEW file whose PARENT
         directory exists (missing path + missing parent -> defect)
    R9   target_scope disjointness: any path shared by two tickets is a
         defect UNLESS the pair is wired (one blocks-on the other in
         either direction)
    R10  difficulty (when present): exactly one of
         trivial|easy|medium|hard|frontier
    R11  lane_hint (when present + charter): must name an existing
         [lanes.*] section
    R12  rubric_only (when present): must be a bool
    R13  unknown key (typo-catcher): any key outside the §6 vocabulary
    R14  required_reading (when present): array of strings, each with a
         literal ``repo:`` or ``context:`` prefix (mirrors dispatch.py's
         ``_REPO_PREFIX``/``_CONTEXT_PREFIX`` — dispatch fails loud on a
         missing prefix, costing a whole worker dispatch); the remainder
         is a non-empty relative path (absolute -> defect; any ``..``
         segment -> defect); with the matching root given, the path must
         EXIST (file or directory) beneath it — unlike R8 there is NO
         new-file carve-out; a ``repo:`` path also defects when it
         string-equals any entry's own ``target_scope`` (the ticket's
         own deliverable)

Defect strings are stable and structured::

    "ticket <id-or-index>: <rule-id>: <detail>"

Entries with a missing/non-string ``id`` are addressed by their 0-based
index; manifest-level structural problems use the ticket prefix
``"manifest"``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# §6 ticket contract: required keys (the decomposer contract requires
# tier1_checks; `intake` does not) — mirrors triage._REQUIRED_PROMOTION_KEYS.
REQUIRED_KEYS = frozenset(
    {"id", "title", "functional_summary", "acceptance_criteria", "tier1_checks", "target_scope"}
)

# §6 optional keys — mirrors triage._OPTIONAL_PROMOTION_KEYS.
OPTIONAL_KEYS = frozenset(
    {"goal", "required_reading", "difficulty", "lane_hint", "rubric_only", "work_product"}
)

# `blocks` is a manifest-level wiring key (consumed to build dependency
# edges, not a ticket column) — legal but not in OPTIONAL_KEYS.
KNOWN_KEYS = REQUIRED_KEYS | OPTIONAL_KEYS | frozenset({"blocks"})

# R10: the closed difficulty vocabulary.
DIFFICULTY_VALUES = frozenset({"trivial", "easy", "medium", "hard", "frontier"})

# R6: kebab-case ticket id.
_KEBAB_ID_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

# R8: a path segment that attempts directory traversal.
_TRAVERSAL_SEGMENT = ".."

# R14: required_reading entry prefixes (mirrors dispatch.py's
# _CONTEXT_PREFIX/_REPO_PREFIX — dispatch.py is NOT modified).
_CONTEXT_PREFIX = "context:"
_REPO_PREFIX = "repo:"

# R5: the [gates] sub-keys whose named checks must be covered by tier1_checks.
_GATES_CHECK_KEYS = ("attempt", "staging")


def validate_manifest(
    manifest: list,
    *,
    repo: Path | None = None,
    charter: object | None = None,
    store: object | None = None,
    context: Path | None = None,
) -> list[str]:
    """Validate a decomposer manifest; return a list of defect strings.

    ``manifest`` is the parsed JSON (expected: a list of ticket
    work-order objects). ``repo`` (when given) enables R8's existence
    checks against that directory and R14's ``repo:``-prefixed
    existence checks; ``charter`` (a
    :class:`~stigmergy.charter.Charter` or a plain resolved-charter dict)
    enables R5's [checks.*]/[gates] cross-checks and R11's [lanes.*]
    check; ``store`` (anything with ``get_ticket``/``list_tickets``)
    enables R6's collision check and R7's store-id resolution;
    ``context`` (when given) enables R14's ``context:``-prefixed
    existence checks against that directory.

    Returns ``[]`` for a clean manifest. Output is deterministic for a
    given input: entries are visited in manifest order, cross-ticket
    pairs in sorted order, and the FIRST cycle found (in deterministic
    DFS order) is the one reported.
    """
    if repo is not None:
        repo = Path(repo)
    if context is not None:
        context = Path(context)

    # Pre-resolve the charter cross-check data once (read-only).
    check_names: set[str] = set()
    gate_checks: set[str] = set()
    check_cmds: dict[str, str] = {}
    lane_names: set[str] = set()
    if charter is not None:
        check_names = _charter_check_names(charter)
        gate_checks = _charter_gate_checks(charter)
        check_cmds = _charter_check_cmds(charter)
        lane_names = _charter_lane_names(charter)

    store_ids: set[str] = set()
    if store is not None:
        store_ids = {t["id"] for t in store.list_tickets()}

    defects: list[str] = []

    # --- R1 (manifest level): must be a list --------------------------------
    if not isinstance(manifest, list):
        defects.append(
            f"manifest: R1: manifest must be a JSON array (got {type(manifest).__name__})"
        )
        return defects

    # --- R1 (entries): every entry must be an object ------------------------
    entries: list[dict[str, Any]] = []
    for idx, entry in enumerate(manifest):
        if not isinstance(entry, dict):
            defects.append(
                f"ticket {idx}: R1: entry must be a JSON object (got {type(entry).__name__})"
            )
            entries.append({})  # placeholder keeps indices aligned
        else:
            entries.append(entry)

    # Manifest id set for cross-entry R7 resolution (duplicate ids are an
    # R6 defect; resolution only needs the set of ids present).
    manifest_ids = {entry.get("id") for entry in entries if _is_id(entry.get("id"))}

    # --- per-entry rules, manifest order ------------------------------------
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict) or not entry:
            # Placeholder for a non-object entry: R1 already reported it.
            continue
        label = _entry_label(entry, idx)

        _check_required_keys(entry, label, defects)
        _check_optional_vocabulary(entry, label, defects)
        _check_functional_summary(entry, label, defects)
        _check_acceptance_criteria(entry, label, defects)
        _check_tier1_checks(entry, label, defects, charter_given=charter is not None,
                            check_names=check_names, gate_checks=gate_checks,
                            check_cmds=check_cmds)
        _check_id(entry, label, defects)
        if store is not None and _is_id(entry.get("id")) and entry["id"] in store_ids:
            defects.append(f"ticket {label}: R6: id collides with an existing ticket in the store")
        _check_blocks(entry, label, defects)
        _check_unresolved_blocks(entry, label, defects, manifest_ids, store_ids)
        _check_target_scope(entry, label, defects, repo=repo)
        _check_required_reading(entry, label, defects, repo=repo, context=context)
        _check_difficulty(entry, label, defects)
        _check_lane_hint(entry, label, defects, charter_given=charter is not None,
                         lane_names=lane_names)
        _check_rubric_only(entry, label, defects)

    # --- cross-ticket rules (deterministic sorted order) ---------------------
    _check_duplicate_ids(entries, defects)
    _check_cycle(entries, defects)
    _check_scope_overlap(entries, defects)

    return defects


# ==========================================================================
# charter adapters (read-only — charter.py is NOT modified)
# ==========================================================================


def _charter_raw(charter: object) -> dict[str, Any]:
    """The resolved-charter dict behind a Charter object or a plain dict."""
    raw = getattr(charter, "raw", charter)
    return raw if isinstance(raw, dict) else {}


def _charter_check_names(charter: object) -> set[str]:
    """Names of the [checks.*] sections in the charter."""
    checks = _charter_raw(charter).get("checks")
    if not isinstance(checks, dict):
        return set()
    return set(checks)


def _charter_check_cmds(charter: object) -> dict[str, str]:
    """Mapping of check name -> its [checks.<name>].cmd (strings only)."""
    checks = _charter_raw(charter).get("checks")
    if not isinstance(checks, dict):
        return {}
    cmds: dict[str, str] = {}
    for name, cfg in checks.items():
        if isinstance(cfg, dict):
            cmd = cfg.get("cmd")
            if isinstance(cmd, str):
                cmds[name] = cmd
    return cmds


def _charter_gate_checks(charter: object) -> set[str]:
    """The union of check names named by [gates] attempt + staging."""
    gates = _charter_raw(charter).get("gates")
    if not isinstance(gates, dict):
        return set()
    named: set[str] = set()
    for key in _GATES_CHECK_KEYS:
        value = gates.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, str):
                named.add(item)
    return named


def _charter_lane_names(charter: object) -> set[str]:
    """Names of the [lanes.*] sections in the charter."""
    lanes = _charter_raw(charter).get("lanes")
    if not isinstance(lanes, dict):
        return set()
    return set(lanes)


# ==========================================================================
# per-entry rule helpers
# ==========================================================================


def _is_id(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _entry_label(entry: Mapping[str, Any], idx: int) -> str:
    """The <id-or-index> defect prefix: the entry's id if it's a usable
    string, else its 0-based index."""
    tid = entry.get("id")
    return tid if isinstance(tid, str) and tid else str(idx)


def _check_required_keys(entry: Mapping[str, Any], label: str, defects: list[str]) -> None:
    for key in sorted(REQUIRED_KEYS - set(entry)):
        defects.append(f"ticket {label}: R2: missing required key: {key}")


def _check_optional_vocabulary(entry: Mapping[str, Any], label: str, defects: list[str]) -> None:
    """R13 (typo-catcher): any key outside the §6 vocabulary is a defect."""
    for key in sorted(set(entry) - KNOWN_KEYS):
        defects.append(f"ticket {label}: R13: unknown key: {key!r}")


def _check_functional_summary(entry: Mapping[str, Any], label: str, defects: list[str]) -> None:
    if "functional_summary" not in entry:
        return  # R2 already reported it
    value = entry.get("functional_summary")
    if not isinstance(value, str) or not value.strip():
        defects.append(f"ticket {label}: R3: functional_summary must be a non-empty string")


def _check_acceptance_criteria(entry: Mapping[str, Any], label: str, defects: list[str]) -> None:
    if "acceptance_criteria" not in entry:
        return
    value = entry.get("acceptance_criteria")
    if not isinstance(value, list):
        defects.append(f"ticket {label}: R4: acceptance_criteria must be an array of strings")
        return
    if len(value) < 1:
        defects.append(f"ticket {label}: R4: acceptance_criteria must have at least one entry")
        return
    for i, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            defects.append(
                f"ticket {label}: R4: acceptance_criteria entry {i} must be a non-empty string"
            )


def _check_tier1_checks(
    entry: Mapping[str, Any],
    label: str,
    defects: list[str],
    *,
    charter_given: bool,
    check_names: set[str],
    gate_checks: set[str],
    check_cmds: dict[str, str],
) -> None:
    if "tier1_checks" not in entry:
        return
    value = entry.get("tier1_checks")
    if isinstance(value, list):
        # Load-bearing: a list misparses as pytest paths downstream
        # (daemon._run_tier1_checks builds `pytest -x -q <joined list>`).
        defects.append(f"ticket {label}: R5: tier1_checks must be a dict, not a list")
        return
    if not isinstance(value, dict):
        defects.append(f"ticket {label}: R5: tier1_checks must be a dict")
        return

    bad_keys = [k for k in value if not isinstance(k, str) or not k]
    if bad_keys:
        defects.append(
            f"ticket {label}: R5: tier1_checks key must be a non-empty string "
            f"(got {bad_keys[0]!r})"
        )

    for name, cmd in value.items():
        if not isinstance(cmd, str):
            defects.append(f"ticket {label}: R5: tier1_checks[{name!r}] cmd must be a string")
            continue
        if charter_given:
            # `check_names` empty = the charter has no [checks] table at
            # all (a charter-load error in practice) — then there is
            # nothing to name, so skip the name cross-check; the gate
            # coverage check below still runs.
            if check_names and name not in check_names:
                defects.append(
                    f"ticket {label}: R5: tier1_checks key {name!r} is not a [checks.*] section"
                )
            elif name in check_cmds and cmd != check_cmds[name]:
                # No "improved" commands: the value must equal the
                # charter's cmd VERBATIM.
                defects.append(
                    f"ticket {label}: R5: tier1_checks[{name!r}] cmd must equal "
                    f"[checks.{name}].cmd"
                )

    if charter_given:
        for name in sorted(gate_checks):
            if name not in value:
                defects.append(
                    f"ticket {label}: R5: [gates] check {name!r} is missing from tier1_checks"
                )


def _check_id(entry: Mapping[str, Any], label: str, defects: list[str]) -> None:
    if "id" not in entry:
        return  # R2 already reported it
    value = entry.get("id")
    if not _is_id(value) or not _KEBAB_ID_RE.match(value):
        defects.append(f"ticket {label}: R6: id is not kebab-case")


def _check_blocks(entry: Mapping[str, Any], label: str, defects: list[str]) -> None:
    """Per-entry portion of R7: blocks must be a list of strings, no
    self-reference. (Unresolved references and acyclicity are
    cross-entry checks, run in manifest order after this.)"""
    if "blocks" not in entry:
        return
    value = entry.get("blocks")
    if not isinstance(value, list):
        defects.append(f"ticket {label}: R7: blocks must be a list of ticket ids")
        return
    for item in value:
        if not isinstance(item, str):
            defects.append(f"ticket {label}: R7: blocks entries must be strings")
            continue
        if _is_id(entry.get("id")) and item == entry["id"]:
            defects.append(f"ticket {label}: R7: blocks self-reference")


def _check_unresolved_blocks(
    entry: Mapping[str, Any],
    label: str,
    defects: list[str],
    manifest_ids: set[str],
    store_ids: set[str],
) -> None:
    """R7 (resolution): every blocks reference must name a manifest id or
    (when a store is given) an existing store ticket."""
    if "blocks" not in entry:
        return
    value = entry.get("blocks")
    if not isinstance(value, list):
        return  # R7 shape defect already reported
    for item in value:
        if not isinstance(item, str):
            continue  # already reported
        if item in manifest_ids or item in store_ids:
            continue
        defects.append(
            f"ticket {label}: R7: blocks reference {item!r} is not a manifest id"
            " or an existing store ticket"
        )


def _check_target_scope(
    entry: Mapping[str, Any],
    label: str,
    defects: list[str],
    *,
    repo: Path | None,
) -> None:
    if "target_scope" not in entry:
        return
    value = entry.get("target_scope")
    if not isinstance(value, list) or not value:
        defects.append(f"ticket {label}: R8: target_scope must be a non-empty array of paths")
        return
    for path in value:
        if not isinstance(path, str) or not path:
            defects.append(f"ticket {label}: R8: target_scope entries must be non-empty strings")
            continue
        if os.path.isabs(path):
            defects.append(f"ticket {label}: R8: target_scope entry {path!r} is an absolute path")
            continue
        if _has_traversal_segment(path):
            defects.append(
                f"ticket {label}: R8: target_scope entry {path!r} contains a '..' segment"
            )
            continue
        if repo is None:
            continue
        target = repo / path
        if target.exists():
            continue
        parent = target.parent
        if parent.exists() and parent.is_dir():
            continue  # the new-file carve-out
        defects.append(
            f"ticket {label}: R8: target_scope entry {path!r} does not exist "
            "and its parent does not either"
        )


def _has_traversal_segment(path: str) -> bool:
    return _TRAVERSAL_SEGMENT in path.split("/")


def _check_required_reading(
    entry: Mapping[str, Any],
    label: str,
    defects: list[str],
    *,
    repo: Path | None,
    context: Path | None,
) -> None:
    """R14: the optional ``required_reading`` key — an array of strings,
    each carrying a literal ``repo:`` or ``context:`` prefix (dispatch.py
    ``_resolve_required_reading_entry`` requires it and fails loud
    otherwise — R14 catches the authoring bug at validation time). The
    remainder must be a non-empty relative path. With the matching root
    given, the path must EXIST beneath it (file or directory) — NO
    new-file carve-out (unlike R8): a required_reading entry pointing at
    a file the ticket itself will create is precisely the defect class.
    A ``repo:`` path that string-equals an entry in the SAME entry's
    ``target_scope`` (its own deliverable) is always a defect, with or
    without a repo. ``context:`` entries never collide with
    target_scope (different root)."""
    if "required_reading" not in entry:
        return
    value = entry.get("required_reading")
    if not isinstance(value, list):
        defects.append(f"ticket {label}: R14: required_reading must be an array of strings")
        return
    scope_paths: set[str] = set()
    raw_scope = entry.get("target_scope")
    if isinstance(raw_scope, list):
        scope_paths = {p for p in raw_scope if isinstance(p, str)}
    for i, item in enumerate(value):
        if not isinstance(item, str):
            defects.append(
                f"ticket {label}: R14: required_reading entry {i} must be a string "
                f"(got {item!r})"
            )
            continue
        if item.startswith(_CONTEXT_PREFIX):
            rel = item[len(_CONTEXT_PREFIX) :]
            root, root_name = context, "context"
        elif item.startswith(_REPO_PREFIX):
            rel = item[len(_REPO_PREFIX) :]
            root, root_name = repo, "repo"
        else:
            defects.append(
                f"ticket {label}: R14: required_reading entry {item!r} has an "
                "unrecognized prefix (expected 'repo:' or 'context:')"
            )
            continue
        if not rel:
            defects.append(
                f"ticket {label}: R14: required_reading entry {item!r} "
                "has an empty path after the prefix"
            )
            continue
        if os.path.isabs(rel):
            defects.append(
                f"ticket {label}: R14: required_reading entry {item!r} is an absolute path"
            )
            continue
        if _has_traversal_segment(rel):
            defects.append(
                f"ticket {label}: R14: required_reading entry {item!r} contains a '..' segment"
            )
            continue
        # Self-deliverable: string-level, ALWAYS checked (with or without a
        # repo) — required_reading must never name a file this ticket will
        # create. `context:` entries live under a different root and never
        # collide with target_scope.
        if root_name == "repo" and rel in scope_paths:
            defects.append(
                f"ticket {label}: R14: required_reading entry {item!r} references "
                "the ticket's own deliverable (also listed in target_scope)"
            )
        if root is None:
            continue  # no root given -> grammar + safety checks only
        if not (root / rel).exists():
            defects.append(
                f"ticket {label}: R14: required_reading entry {item!r} does not exist "
                f"in the {root_name} (required_reading must already exist — "
                "never the ticket's own deliverable)"
            )


def _check_difficulty(entry: Mapping[str, Any], label: str, defects: list[str]) -> None:
    if "difficulty" not in entry:
        return
    value = entry.get("difficulty")
    if not isinstance(value, str) or value not in DIFFICULTY_VALUES:
        defects.append(
            f"ticket {label}: R10: difficulty {value!r} is not one of: "
            f"{', '.join(sorted(DIFFICULTY_VALUES))}"
        )


def _check_lane_hint(
    entry: Mapping[str, Any],
    label: str,
    defects: list[str],
    *,
    charter_given: bool,
    lane_names: set[str],
) -> None:
    if "lane_hint" not in entry:
        return
    if not charter_given:
        return
    value = entry.get("lane_hint")
    if not isinstance(value, str) or value not in lane_names:
        defects.append(f"ticket {label}: R11: lane_hint {value!r} is not a [lanes.*] section")


def _check_rubric_only(entry: Mapping[str, Any], label: str, defects: list[str]) -> None:
    if "rubric_only" not in entry:
        return
    value = entry.get("rubric_only")
    if not isinstance(value, bool):
        defects.append(f"ticket {label}: R12: rubric_only must be a boolean (got {value!r})")


# ==========================================================================
# cross-ticket rule helpers
# ==========================================================================


def _check_duplicate_ids(entries: list[dict[str, Any]], defects: list[str]) -> None:
    """R6 (manifest-level): id uniqueness. One defect per duplicate id,
    naming every entry index it occurs at; the label is the FIRST
    occurrence's id."""
    seen: dict[str, list[int]] = {}
    for idx, entry in enumerate(entries):
        tid = entry.get("id")
        if _is_id(tid):
            seen.setdefault(tid, []).append(idx)
    for tid, indices in seen.items():
        if len(indices) < 2:
            continue
        where = ", ".join(str(i) for i in indices)
        defects.append(f"ticket {tid}: R6: duplicate id {tid!r} (entries {where})")


def _manifest_block_edges(entries: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """The manifest-internal block edges (ticket_id -> predecessor), in
    manifest order. All entries sharing an id contribute their edges (a
    duplicate-id manifest is already an R6 defect — but its block graph
    must still be analyzed, not silently pruned). Store-resolved
    references are excluded here — they are external nodes that cannot
    form a cycle."""
    manifest_ids = {entry.get("id") for entry in entries if _is_id(entry.get("id"))}
    edges: list[tuple[str, str]] = []
    for entry in entries:
        tid = entry.get("id")
        if not _is_id(tid):
            continue
        blocks = entry.get("blocks")
        if not isinstance(blocks, list):
            continue
        for pred in blocks:
            if isinstance(pred, str) and pred in manifest_ids:
                edges.append((tid, pred))
    return edges


def _check_cycle(entries: list[dict[str, Any]], defects: list[str]) -> None:
    """R7 (acyclicity): DFS over the manifest-internal block graph. On
    the FIRST cycle found (deterministic order: sorted ids, sorted
    predecessors) emit one defect naming the cycle path."""
    edges = _manifest_block_edges(entries)
    if not edges:
        return
    adj: dict[str, list[str]] = {}
    nodes: set[str] = set()
    for ticket_id, pred in edges:
        adj.setdefault(ticket_id, []).append(pred)
        nodes.add(ticket_id)
        nodes.add(pred)
    for node in adj:
        adj[node] = sorted(adj[node])

    visited: set[str] = set()
    stack: list[str] = []
    on_path: set[str] = set()

    def dfs(node: str) -> bool:
        visited.add(node)
        stack.append(node)
        on_path.add(node)
        for pred in adj.get(node, []):
            if pred in on_path:
                return True  # cycle found; the caller reads `stack`
            if pred not in visited:
                if dfs(pred):
                    return True
        stack.pop()
        on_path.discard(node)
        return False

    for node in sorted(nodes):
        if node not in visited:
            if dfs(node):
                # Rotate the detected cycle so it starts at its
                # lexicographically smallest node: a stable,
                # data-determined canonical form.
                i = stack.index(node)
                cycle = stack[i:]
                pivot = cycle.index(min(cycle))
                canonical = cycle[pivot:] + cycle[:pivot]
                defects.append(
                    f"ticket {canonical[0]}: R7: cycle in blocks graph: "
                    + " -> ".join(canonical + [canonical[0]])
                )
                break


def _wired_pairs(entries: list[dict[str, Any]]) -> set[tuple[str, str]]:
    """Unordered ticket pairs wired by a blocks edge in EITHER direction
    (manifest-internal references only)."""
    manifest_ids = {entry.get("id") for entry in entries if _is_id(entry.get("id"))}
    pairs: set[tuple[str, str]] = set()
    for entry in entries:
        tid = entry.get("id")
        if not _is_id(tid):
            continue
        blocks = entry.get("blocks")
        if not isinstance(blocks, list):
            continue
        for pred in blocks:
            if isinstance(pred, str) and pred in manifest_ids and pred != tid:
                pairs.add((pred, tid) if pred < tid else (tid, pred))
    return pairs


def _check_scope_overlap(entries: list[dict[str, Any]], defects: list[str]) -> None:
    """R9: any target_scope path shared by two tickets is a defect unless
    the pair is wired (one blocks-on the other, either direction). The
    defect is attributed to the LOWER id of the pair, once per path."""
    wired = _wired_pairs(entries)

    # path -> ids in first-occurrence order (deterministic)
    path_owners: dict[str, list[str]] = {}
    for entry in entries:
        tid = entry.get("id")
        scope = entry.get("target_scope")
        if not _is_id(tid) or not isinstance(scope, list):
            continue
        for path in scope:
            if isinstance(path, str) and path:
                path_owners.setdefault(path, [])
                if tid not in path_owners[path]:
                    path_owners[path].append(tid)

    for path, owners in path_owners.items():
        if len(owners) < 2:
            continue
        for i in range(len(owners)):
            for j in range(i + 1, len(owners)):
                a, b = sorted((owners[i], owners[j]))
                if (a, b) in wired:
                    continue
                defects.append(
                    f"ticket {a}: R9: target_scope overlaps ticket {b!r} on {path!r} "
                    "without a blocks edge"
                )
