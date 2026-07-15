# Stigmergy

Deterministic orchestration harness. LLM cognition is confined to judgment stations; everything else is mechanism.

**Status:** v0 build in progress.

The working specification (build contract) is maintained outside this repo by the maintainer. This repository holds the implementation.

## Rigs

A **rig** is the unit of tenancy and portability (SPEC.md §3): a
self-contained directory owning its charter, model registry, repo clone,
curated context, event records, worker images, and per-dispatch clones.
`stigmergy rig new --charter <path> [--path <base-dir>]` scaffolds one.

**Rig location convention (resolves spec-gap #7).** Rigs live at
`~/rigs/<rig.name>` by default; override the base directory with
`--path`. The rig root is `<base-dir>/<name>`; `stigmergy rig new` refuses
to create a rig if that path already exists, and cleans up any
partially-created rig directory on failure (invalid charter, failed repo
clone, etc.) — rig creation is all-or-nothing.

**Bead store: self-contained SQLite, not `bd`/dolt (resolves spec-gap
#8).** Each rig owns a single `beads.db` SQLite file (stdlib `sqlite3`) as
its bead + loop-state store — deliberately *not* the `bd` issue tracker.
This buys isolation by structure (the whole rig, including its data
plane, is one portable directory you can copy, back up, or ship
elsewhere), schema freedom for loop-only metadata (leases, attempts,
step-up rungs) that doesn't belong in a general-purpose issue tracker, and
zero external service dependency. Spec-gap #8 asked which `bd`
binary/version a rig should pin — that question is now moot: there is no
`bd` dependency to pin. The store's own schema is versioned instead, via
a `schema_version` key in the `rig_meta` table (starts at `"1"`); future
beads that extend the schema bump that value rather than requiring a
migration story shared with an external tool.

License: AGPL-3.0-or-later (see LICENSE).
