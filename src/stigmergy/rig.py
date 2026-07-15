"""Rig scaffold + bead store (SPEC.md §3 `provision` station, §6 bead contract).

A **rig** is the unit of tenancy and portability (SPEC.md §3 Objects): a
self-contained directory owning its own charter, model registry, repo
clone, curated context, event records, worker images, and per-dispatch
clones. This module implements `provision` — the deterministic mechanism
that creates that directory structure and its bead/loop-state store.

**Design decision (bead .7):** the bead + loop-state store is a
loop-owned SQLite database (stdlib `sqlite3`), not the `bd` issue tracker.
This buys isolation-by-structure (one self-contained file travels with the
rig), schema freedom for loop-only metadata (leases, attempts, rungs), and
portability with no external service dependency. Nothing in this module
shells out to `bd`.

Rig creation is all-or-nothing: the charter is validated *before* any
directory is created, and any failure partway through scaffolding removes
the partially-created rig root — `create_rig` never leaves partial state
on disk.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any

from stigmergy import __version__
from stigmergy.charter import Charter, CharterError, load_charter

# --- schema (SPEC.md §3 rig definition, §6 bead work-order contract) -------

_SCHEMA_SQL = """
CREATE TABLE beads (
  id                   TEXT PRIMARY KEY,
  title                TEXT NOT NULL,
  goal                 TEXT,
  required_reading     TEXT,
  target_scope         TEXT,
  work_product         TEXT,
  acceptance_criteria  TEXT,
  tier1_checks         TEXT,
  difficulty           TEXT,
  lane_hint            TEXT,
  rubric_only          INTEGER NOT NULL DEFAULT 0,
  approved             INTEGER NOT NULL DEFAULT 0,
  approval_hash        TEXT,
  state                TEXT NOT NULL DEFAULT 'pool',
  lease_owner          TEXT,
  lease_dispatch_id    TEXT,
  lease_expires_at     REAL,
  lease_heartbeat_at   REAL,
  attempts_used        INTEGER NOT NULL DEFAULT 0,
  integration_failures INTEGER NOT NULL DEFAULT 0,
  current_rung         TEXT,
  created_at           REAL NOT NULL,
  updated_at           REAL NOT NULL
);
CREATE TABLE bead_deps (
  bead_id        TEXT NOT NULL,
  blocks_bead_id TEXT NOT NULL,
  PRIMARY KEY (bead_id, blocks_bead_id)
);
CREATE TABLE rig_meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
"""

# Columns settable through add_bead() beyond the required id/title (which
# have their own keyword-only parameters). created_at/updated_at are
# stamped internally and are not caller-settable.
_BEAD_OPTIONAL_FIELDS = {
    "goal",
    "required_reading",
    "target_scope",
    "work_product",
    "acceptance_criteria",
    "tier1_checks",
    "difficulty",
    "lane_hint",
    "rubric_only",
    "approved",
    "approval_hash",
    "state",
    "lease_owner",
    "lease_dispatch_id",
    "lease_expires_at",
    "lease_heartbeat_at",
    "attempts_used",
    "integration_failures",
    "current_rung",
}

# Columns whose Python value is a list/dict, transparently JSON-encoded on
# write and decoded on read (SPEC.md §6.2/§6.3/§6.5/§6.6).
_JSON_BEAD_FIELDS = {
    "required_reading",
    "target_scope",
    "acceptance_criteria",
    "tier1_checks",
}


class RigError(Exception):
    """Raised on any rig-scaffolding failure (SPEC.md §3 `provision`)."""


class RigStore:
    """SQLite-backed bead + loop-state store for one rig (v0 schema).

    Not `bd`, not dolt — a plain, self-contained ``sqlite3`` file so the
    rig stays a single portable directory. Downstream beads (.15/.16)
    extend this schema for leases, gating, and event-plane integration;
    this is the minimal v0 surface: bead CRUD, dependency edges, and a
    key/value ``rig_meta`` table.
    """

    def __init__(self, db_path: str | os.PathLike[str]) -> None:
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        # `bead_deps` declares no explicit FOREIGN KEY constraint in the v0
        # schema, so this pragma enforces nothing yet — it's set now so
        # later beads that add real FK constraints get enforcement for
        # free without an extra migration step.
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")

    @classmethod
    def create(cls, db_path: str | os.PathLike[str]) -> RigStore:
        """Create a fresh rig store: initializes schema, returns the open store."""
        store = cls(db_path)
        store._conn.executescript(_SCHEMA_SQL)
        store._conn.commit()
        return store

    def add_bead(self, *, id: str, title: str, **fields: Any) -> None:
        """Insert a work-order bead (SPEC.md §6).

        ``id``/``title`` are required. Remaining columns are optional
        keyword arguments; list/dict-valued fields (``required_reading``,
        ``target_scope``, ``acceptance_criteria``, ``tier1_checks``) are
        JSON-encoded transparently — callers pass Python values, not raw
        JSON strings. ``created_at``/``updated_at`` are stamped here.
        """
        unknown = set(fields) - _BEAD_OPTIONAL_FIELDS
        if unknown:
            raise ValueError(f"unknown bead field(s): {sorted(unknown)}")

        now = time.time()
        columns = ["id", "title", "created_at", "updated_at"]
        values: list[Any] = [id, title, now, now]
        for key, value in fields.items():
            if key in _JSON_BEAD_FIELDS and value is not None:
                value = json.dumps(value)
            columns.append(key)
            values.append(value)

        col_sql = ", ".join(columns)
        placeholders = ", ".join("?" for _ in columns)
        self._conn.execute(
            f"INSERT INTO beads ({col_sql}) VALUES ({placeholders})",  # noqa: S608
            values,
        )
        self._conn.commit()

    def get_bead(self, bead_id: str) -> dict[str, Any] | None:
        """Fetch one bead by id, or ``None`` if absent. JSON fields decode to lists/dicts."""
        row = self._conn.execute("SELECT * FROM beads WHERE id = ?", (bead_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_bead(row)

    def list_beads(self, *, state: str | None = None) -> list[dict[str, Any]]:
        """List beads, optionally filtered by ``state``, ordered by creation time."""
        if state is None:
            rows = self._conn.execute("SELECT * FROM beads ORDER BY created_at").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM beads WHERE state = ? ORDER BY created_at", (state,)
            ).fetchall()
        return [self._row_to_bead(row) for row in rows]

    def add_dep(self, bead_id: str, blocks_bead_id: str) -> None:
        """Record that ``bead_id`` is blocked-by (must land after) ``blocks_bead_id``."""
        self._conn.execute(
            "INSERT INTO bead_deps (bead_id, blocks_bead_id) VALUES (?, ?)",
            (bead_id, blocks_bead_id),
        )
        self._conn.commit()

    def deps_of(self, bead_id: str) -> list[str]:
        """Return the predecessor bead ids that block ``bead_id``."""
        rows = self._conn.execute(
            "SELECT blocks_bead_id FROM bead_deps WHERE bead_id = ?", (bead_id,)
        ).fetchall()
        return [row["blocks_bead_id"] for row in rows]

    def set_meta(self, key: str, value: str) -> None:
        """Set (or overwrite) a `rig_meta` key/value pair."""
        self._conn.execute(
            "INSERT INTO rig_meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self._conn.commit()

    def get_meta(self, key: str) -> str | None:
        """Fetch a `rig_meta` value by key, or ``None`` if absent."""
        row = self._conn.execute("SELECT value FROM rig_meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def close(self) -> None:
        """Checkpoint the WAL into the main file and close the connection.

        The connection is always closed, even if the checkpoint itself
        raises (e.g. contention) — callers must be able to rely on
        `close()` releasing the file handle unconditionally.
        """
        try:
            self._conn.execute("PRAGMA wal_checkpoint(FULL)")
        finally:
            self._conn.close()

    def _row_to_bead(self, row: sqlite3.Row) -> dict[str, Any]:
        bead = dict(row)
        for key in _JSON_BEAD_FIELDS:
            if bead.get(key) is not None:
                bead[key] = json.loads(bead[key])
        return bead


# --- provision: rig scaffold -----------------------------------------------


def create_rig(
    charter_path: str | os.PathLike[str],
    base_dir: str | os.PathLike[str] | None = None,
    *,
    repo_override: str | None = None,
) -> Path:
    """Scaffold a new rig from a validated charter (SPEC.md §3 `provision`).

    The charter is validated with :func:`load_charter` *before* anything is
    created — an invalid charter raises :class:`~stigmergy.charter.CharterError`
    and creates no rig directory. Rig root is ``<base_dir>/<rig.name>``;
    ``base_dir`` defaults to ``~/rigs``. Any failure partway through
    scaffolding (including a failed git clone) removes the partially-created
    rig root before re-raising — rig creation is all-or-nothing.

    Returns the created rig root path.
    """
    charter_path = Path(charter_path)
    charter = load_charter(charter_path)

    rig_cfg = charter.raw.get("rig")
    if not isinstance(rig_cfg, dict):
        raise CharterError("[rig] table is required to create a rig")
    name = rig_cfg.get("name")
    if not isinstance(name, str) or not name:
        raise CharterError("[rig].name is required to create a rig")
    repo = repo_override if repo_override is not None else rig_cfg.get("repo")
    if not isinstance(repo, str) or not repo:
        raise CharterError("[rig].repo is required to create a rig")

    base = Path(base_dir).expanduser() if base_dir is not None else Path("~/rigs").expanduser()
    rig_root = base / name

    if rig_root.exists():
        raise RigError(f"rig directory already exists: {rig_root}")

    try:
        _scaffold_rig(rig_root, charter_path, charter, repo)
    except Exception:
        shutil.rmtree(rig_root, ignore_errors=True)
        raise

    return rig_root


def _scaffold_rig(rig_root: Path, charter_path: Path, charter: Charter, repo: str) -> None:
    """Create the rig directory tree, copy config, init the store, clone the repo.

    Caller (:func:`create_rig`) is responsible for cleanup on failure.
    """
    rig_root.mkdir(parents=True)

    charter_dir = charter_path.resolve().parent
    registry_rel = charter.raw["models"]["registry"]
    registry_path = Path(registry_rel)
    if not registry_path.is_absolute():
        registry_path = charter_dir / registry_path
    if not registry_path.is_file():
        raise RigError(f"model registry file not found: {registry_path}")

    shutil.copy(registry_path, rig_root / "models.toml")

    charter_text = charter_path.read_text()
    if registry_rel != "models.toml":
        charter_text = _rewrite_registry_path(charter_text, "models.toml")
    (rig_root / "charter.toml").write_text(charter_text)

    for sub in ("context", "records", "clones"):
        (rig_root / sub).mkdir()

    images_dir = rig_root / "images" / "worker"
    images_dir.mkdir(parents=True)
    (images_dir / "Containerfile").write_text(
        "# stub Containerfile for the rig worker image.\n"
        "# Populated by a later bead: pinned base-image digest, no-secret build\n"
        "# through the `registries` egress group (SPEC.md §3 provision, §4).\n"
    )

    store = RigStore.create(rig_root / "beads.db")
    try:
        store.set_meta("schema_version", "1")
        store.set_meta("stigmergy_version", __version__)
        store.set_meta("charter_hash", charter.resolved_hash)
        store.set_meta("rig_name", charter.raw["rig"]["name"])
        store.set_meta("created_at", str(time.time()))
    finally:
        store.close()

    _clone_repo(repo, rig_root / "repo")


def _rewrite_registry_path(text: str, new_rel: str) -> str:
    """Rewrite the `[models].registry` string value to ``new_rel`` in raw TOML text.

    A targeted substitution (rather than a parse/re-dump round trip) so the
    rest of the charter's formatting and comments are preserved verbatim.
    ``registry`` is only a valid key under ``[models]`` in the charter
    schema, so a single global match is unambiguous.
    """
    pattern = re.compile(r'(registry\s*=\s*)"[^"]*"')
    new_text, count = pattern.subn(rf'\1"{new_rel}"', text, count=1)
    if count != 1:
        raise RigError("could not locate [models].registry key to rewrite in charter.toml")
    return new_text


def _clone_repo(repo: str, dest: Path) -> None:
    """Clone ``repo`` (local path or URL) into ``dest`` via `git clone`.

    Checks the process exit code, not stderr text (git commonly writes
    progress/informational output to stderr even on a successful clone of
    a local repo).
    """
    result = subprocess.run(  # noqa: S603
        ["git", "clone", "--", str(repo), str(dest)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RigError(
            f"git clone of {repo!r} failed (exit {result.returncode}): {result.stderr.strip()}"
        )
