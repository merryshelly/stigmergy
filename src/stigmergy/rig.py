"""Rig scaffold + ticket store (SPEC.md §3 `provision` station, §6 ticket contract).

A **rig** is the unit of tenancy and portability (SPEC.md §3 Objects): a
self-contained directory owning its own charter, model registry, repo
clone, curated context, event records, worker images, and per-dispatch
clones. This module implements `provision` — the deterministic mechanism
that creates that directory structure and its ticket/loop-state store.

**Design decision (ticket .7):** the ticket + loop-state store is a
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
import shlex
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stigmergy import __version__
from stigmergy.charter import Charter, CharterError, load_charter
from stigmergy.container import build_image
from stigmergy.registry import Registry, load_registry

# bead .149: the host-side OA source the rig's worker wheelhouse is built
# from (the provision station runs ON the rig owner's host, where network +
# the OA checkout are available; the BAKE itself is fully offline via
# --no-index). The in-cage agent TOML template shipped with the package.
_OA_SOURCE_DIR = Path("/opt/openalph")
_OA_WORKER_TOML = Path(__file__).parent / "worker_image" / "oa-worker.toml"

# --- schema (SPEC.md §3 rig definition, §6 ticket work-order contract) -------

_SCHEMA_SQL = """
CREATE TABLE tickets (
  id                   TEXT PRIMARY KEY,
  title                TEXT NOT NULL,
  goal                 TEXT,
  required_reading     TEXT,
  target_scope         TEXT,
  work_product         TEXT,
  functional_summary   TEXT,
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
  critic_infra_failures INTEGER NOT NULL DEFAULT 0,
  current_rung         TEXT,
  created_at           REAL NOT NULL,
  updated_at           REAL NOT NULL
);
CREATE TABLE ticket_deps (
  ticket_id        TEXT NOT NULL,
  blocks_ticket_id TEXT NOT NULL,
  PRIMARY KEY (ticket_id, blocks_ticket_id)
);
CREATE TABLE rig_meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE worker_names (
  name       TEXT PRIMARY KEY,
  created_at REAL NOT NULL
);
"""

# `filed_tickets` DDL (D14, bead workspace-e2uh.38) — deliberately NOT part
# of `_SCHEMA_SQL` (one home for this table's DDL, avoid drift). A filed
# proposal lives here, NEVER in `tickets`: `intake.claim`/`intake.eligible`
# read `tickets` only, so a row that only ever exists here is structurally
# un-claimable — the load-bearing security property behind D14 (worker-
# authored ticket text is an injection surface; keeping proposals out of
# the live pool is physics, not a `WHERE approved=1` filter that can
# regress). Run via `RigStore._ensure_filed_tickets_table` at the END of
# `RigStore.__init__` so both fresh `.create()` rigs and pre-D14 rigs
# reopened via plain `RigStore(path)`/`resolve_rig` self-heal.
_FILED_TICKETS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS filed_tickets (
  id                   TEXT PRIMARY KEY,
  title                TEXT NOT NULL,
  description          TEXT NOT NULL,
  evidence             TEXT,
  origin_role          TEXT NOT NULL,
  origin_worker        TEXT,
  origin_dispatch_id   TEXT,
  origin_parent_ticket TEXT,
  discovered_from      TEXT NOT NULL,
  proposal_hash        TEXT NOT NULL,
  created_at           REAL NOT NULL,
  triaged              INTEGER NOT NULL DEFAULT 0,
  triage_outcome       TEXT,
  resulting_ticket_id  TEXT
);
"""

# Columns settable through add_ticket() beyond the required id/title (which
# have their own keyword-only parameters). created_at/updated_at are
# stamped internally and are not caller-settable.
_TICKET_OPTIONAL_FIELDS = {
    "goal",
    "required_reading",
    "target_scope",
    "work_product",
    "functional_summary",
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
    "critic_infra_failures",
    "current_rung",
}

# Columns whose Python value is a list/dict, transparently JSON-encoded on
# write and decoded on read (SPEC.md §6.2/§6.3/§6.5/§6.6).
_JSON_TICKET_FIELDS = {
    "required_reading",
    "target_scope",
    "acceptance_criteria",
    "tier1_checks",
}


class RigError(Exception):
    """Raised on any rig-scaffolding failure (SPEC.md §3 `provision`)."""


class RigStore:
    """SQLite-backed ticket + loop-state store for one rig (v0 schema).

    Not `bd`, not dolt — a plain, self-contained ``sqlite3`` file so the
    rig stays a single portable directory. Downstream tickets (.15/.16)
    extend this schema for leases, gating, and event-plane integration;
    this is the minimal v0 surface: ticket CRUD, dependency edges, and a
    key/value ``rig_meta`` table.
    """

    def __init__(self, db_path: str | os.PathLike[str]) -> None:
        self.db_path = Path(db_path)
        # check_same_thread=False allows multiple threads to safely access the
        # same connection, but all DB operations are serialized via _db_lock
        # (see docstring). Keep the single-connection model (no per-thread
        # connection restructuring).
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # _db_lock serializes all DB access (read and write) across threads.
        # A threading.RLock permits the same thread to acquire it multiple
        # times (re-entrant); the lock protects the single shared _conn from
        # concurrent reads/writes from other threads.
        self._db_lock = threading.RLock()
        # `ticket_deps` declares no explicit FOREIGN KEY constraint in the v0
        # schema, so this pragma enforces nothing yet — it's set now so
        # later tickets that add real FK constraints get enforcement for
        # free without an extra migration step.
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._ensure_filed_tickets_table()
        self._run_schema_migrations()

    def _run_schema_migrations(self) -> None:
        """Run ordered, versioned migrations to bring schema up to current.

        Each migration is idempotent (safe to rerun) and updates
        rig_meta.schema_version after success, so reopening an older DB
        both brings it current and records that in the version tag.
        Migrations are applied in order based on current schema_version.
        """
        # Schema version is only meaningful if tickets table exists
        # (a fresh store via .create() hasn't built it yet — that happens
        # after __init__ returns, in .create() itself).
        table_exists = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'tickets'"
        ).fetchone()
        if table_exists is None:
            return

        current_version = self.get_meta("schema_version")
        current_version_int = int(current_version) if current_version else 0

        # Migration to version 4: add functional_summary column (bead .42)
        if current_version_int < 4:
            cols = {row["name"] for row in self._conn.execute("PRAGMA table_info(tickets)")}
            if "functional_summary" not in cols:
                self._conn.execute("ALTER TABLE tickets ADD COLUMN functional_summary TEXT")
                self._conn.commit()
            self.set_meta("schema_version", "4")
            current_version_int = 4

        # Migration to version 5: add critic_infra_failures column (bead .107)
        if current_version_int < 5:
            cols = {row["name"] for row in self._conn.execute("PRAGMA table_info(tickets)")}
            if "critic_infra_failures" not in cols:
                self._conn.execute(
                    "ALTER TABLE tickets ADD COLUMN critic_infra_failures "
                    "INTEGER NOT NULL DEFAULT 0"
                )
                self._conn.commit()
            self.set_meta("schema_version", "5")

    def _ensure_filed_tickets_table(self) -> None:
        """Self-healing migration (D14): create `filed_tickets` if absent.

        `CREATE TABLE IF NOT EXISTS`, run unconditionally at the end of
        every `__init__` — idempotent for fresh `.create()` rigs and the
        only migration path for a rig scaffolded before D14 and reopened
        via plain `RigStore(path)`/`resolve_rig` (which never run
        `.create()`).
        """
        self._conn.executescript(_FILED_TICKETS_SCHEMA_SQL)
        self._conn.commit()

    @classmethod
    def create(cls, db_path: str | os.PathLike[str]) -> RigStore:
        """Create a fresh rig store: initializes schema, returns the open store."""
        store = cls(db_path)
        store._conn.executescript(_SCHEMA_SQL)
        store._conn.commit()
        return store

    @staticmethod
    def _encode_ticket_fields(fields: dict[str, Any]) -> tuple[list[str], list[Any]]:
        """Shared column-allowlist + JSON-encode helper for ticket field dicts.

        Validates every key in ``fields`` against `_TICKET_OPTIONAL_FIELDS`
        (raises :class:`ValueError` listing any unknown field), then
        JSON-encodes the list/dict-valued fields (`_JSON_TICKET_FIELDS`).
        Returns parallel ``(columns, values)`` lists suitable for splicing
        into an INSERT/UPDATE. Reused by :meth:`add_ticket` and
        :meth:`promote_filed_ticket` so the column/JSON rules live in
        exactly one place (bead .42 DRY requirement).
        """
        unknown = set(fields) - _TICKET_OPTIONAL_FIELDS
        if unknown:
            raise ValueError(f"unknown ticket field(s): {sorted(unknown)}")

        columns: list[str] = []
        values: list[Any] = []
        for key, value in fields.items():
            if key in _JSON_TICKET_FIELDS and value is not None:
                value = json.dumps(value)
            columns.append(key)
            values.append(value)
        return columns, values

    def add_ticket(self, *, id: str, title: str, **fields: Any) -> None:
        """Insert a work-order ticket (SPEC.md §6).

        ``id``/``title`` are required. Remaining columns are optional
        keyword arguments; list/dict-valued fields (``required_reading``,
        ``target_scope``, ``acceptance_criteria``, ``tier1_checks``) are
        JSON-encoded transparently — callers pass Python values, not raw
        JSON strings. ``created_at``/``updated_at`` are stamped here.
        """
        now = time.time()
        columns = ["id", "title", "created_at", "updated_at"]
        values: list[Any] = [id, title, now, now]
        extra_columns, extra_values = self._encode_ticket_fields(fields)
        columns.extend(extra_columns)
        values.extend(extra_values)

        col_sql = ", ".join(columns)
        placeholders = ", ".join("?" for _ in columns)
        with self._db_lock:
            self._conn.execute(
                f"INSERT INTO tickets ({col_sql}) VALUES ({placeholders})",  # noqa: S608
                values,
            )
            self._conn.commit()

    def get_ticket(self, ticket_id: str) -> dict[str, Any] | None:
        """Fetch one ticket by id, or ``None`` if absent. JSON fields decode to lists/dicts."""
        with self._db_lock:
            row = self._conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_ticket(row)

    def list_tickets(self, *, state: str | None = None) -> list[dict[str, Any]]:
        """List tickets, optionally filtered by ``state``, ordered by creation time."""
        with self._db_lock:
            if state is None:
                rows = self._conn.execute("SELECT * FROM tickets ORDER BY created_at").fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM tickets WHERE state = ? ORDER BY created_at", (state,)
                ).fetchall()
        return [self._row_to_ticket(row) for row in rows]

    def update_ticket(self, ticket_id: str, **fields: Any) -> None:
        """Update one or more columns on an existing ticket (SPEC.md §9
        state machine + leases; ticket .15).

        ``fields`` uses the same column allowlist and JSON-encoding rules
        as :meth:`add_ticket` (list/dict-valued fields are transparently
        JSON-encoded). ``updated_at`` is always bumped to ``time.time()``,
        even if the caller didn't ask for any other change. Raises
        :class:`ValueError` if ``fields`` is empty, names an unknown
        column, or ``ticket_id`` doesn't exist — a typo'd id or a no-op
        call never silently does nothing.
        """
        if not fields:
            raise ValueError("update_ticket requires at least one field to update")
        unknown = set(fields) - _TICKET_OPTIONAL_FIELDS
        if unknown:
            raise ValueError(f"unknown ticket field(s): {sorted(unknown)}")

        now = time.time()
        columns = []
        values: list[Any] = []
        for key, value in fields.items():
            if key in _JSON_TICKET_FIELDS and value is not None:
                value = json.dumps(value)
            columns.append(key)
            values.append(value)
        columns.append("updated_at")
        values.append(now)

        set_sql = ", ".join(f"{col} = ?" for col in columns)
        values.append(ticket_id)
        with self._db_lock:
            cursor = self._conn.execute(
                f"UPDATE tickets SET {set_sql} WHERE id = ?",  # noqa: S608
                values,
            )
            if cursor.rowcount == 0:
                self._conn.rollback()
                raise ValueError(f"no such ticket: {ticket_id!r}")
            self._conn.commit()

    def atomic_claim_ticket(
        self,
        ticket_id: str,
        *,
        owner: str,
        dispatch_id: str,
        lease_expires_at: float,
        lease_heartbeat_at: float,
    ) -> bool:
        """Atomically claim a ticket with an ACID conditional UPDATE.

        Performs a single SQL UPDATE statement:
        ```
        UPDATE tickets SET lease_owner=?, lease_dispatch_id=?,
        lease_expires_at=?, lease_heartbeat_at=?, state='claimed',
        updated_at=?
        WHERE id=? AND lease_owner IS NULL
        AND state IN ('pool','eligible')
        ```

        Returns ``True`` if exactly one row was updated (claim won),
        ``False`` otherwise (already leased or not in claimable state).

        This is the ATOMIC conditional claim for concurrent dispatch:
        two concurrent threads cannot both return True for the same
        ticket. The winner is determined purely by the UPDATE's rowcount
        (==1 means one and only one matching row was modified).

        Serialized by the same _db_lock that protects all other DB access,
        but the atomicity of the claim is guaranteed by the single
        conditional UPDATE statement + rowcount check, not by holding the
        lock across a check-then-set pair.
        """
        now = time.time()
        with self._db_lock:
            cursor = self._conn.execute(
                "UPDATE tickets "
                "SET lease_owner=?, lease_dispatch_id=?, "
                "    lease_expires_at=?, lease_heartbeat_at=?, "
                "    state='claimed', updated_at=? "
                "WHERE id=? AND lease_owner IS NULL "
                "AND state IN ('pool','eligible')",
                (owner, dispatch_id, lease_expires_at, lease_heartbeat_at, now, ticket_id),
            )
            won = cursor.rowcount == 1
            self._conn.commit()
        return won

    def add_dep(self, ticket_id: str, blocks_ticket_id: str) -> None:
        """Record that ``ticket_id`` is blocked-by (must land after) ``blocks_ticket_id``."""
        with self._db_lock:
            self._conn.execute(
                "INSERT INTO ticket_deps (ticket_id, blocks_ticket_id) VALUES (?, ?)",
                (ticket_id, blocks_ticket_id),
            )
            self._conn.commit()

    def deps_of(self, ticket_id: str) -> list[str]:
        """Return the predecessor ticket ids that block ``ticket_id``."""
        with self._db_lock:
            rows = self._conn.execute(
                "SELECT blocks_ticket_id FROM ticket_deps WHERE ticket_id = ?", (ticket_id,)
            ).fetchall()
        return [row["blocks_ticket_id"] for row in rows]

    def insert_worker_name(self, name: str) -> None:
        """Reserve ``name`` in the rig-wide `worker_names` set (bead .21
        build spec §0.5). Raises `sqlite3.IntegrityError` on a primary-key
        collision — callers (`dispatch.generate_worker_name`) are
        responsible for catching that and retrying with a fresh nonce;
        this method does not swallow it, only rolls back the failed
        insert's transaction first so the connection is left clean (a
        collision here must never leave a dangling uncommitted write
        behind).
        """
        with self._db_lock:
            try:
                self._conn.execute(
                    "INSERT INTO worker_names (name, created_at) VALUES (?, ?)",
                    (name, time.time()),
                )
            except sqlite3.IntegrityError:
                self._conn.rollback()
                raise
            self._conn.commit()

    def worker_name_exists(self, name: str) -> bool:
        """True iff ``name`` has already been reserved in `worker_names`."""
        with self._db_lock:
            row = self._conn.execute(
                "SELECT 1 FROM worker_names WHERE name = ?", (name,)
            ).fetchone()
        return row is not None

    def set_meta(self, key: str, value: str) -> None:
        """Set (or overwrite) a `rig_meta` key/value pair."""
        with self._db_lock:
            self._conn.execute(
                "INSERT INTO rig_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            self._conn.commit()

    def get_meta(self, key: str) -> str | None:
        """Fetch a `rig_meta` value by key, or ``None`` if absent."""
        with self._db_lock:
            row = self._conn.execute("SELECT value FROM rig_meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def add_filed_ticket(
        self,
        *,
        id: str,
        title: str,
        description: str,
        evidence: str | None = None,
        origin_role: str,
        origin_worker: str | None,
        origin_dispatch_id: str | None,
        origin_parent_ticket: str | None,
        discovered_from: str,
        proposal_hash: str,
    ) -> None:
        """Insert one UNAPPROVED filed-ticket proposal (D14, bead
        workspace-e2uh.38). Stamps `created_at=time.time()`, `triaged=0`.

        CRITICAL INVARIANT: this method NEVER inserts into `tickets` — a
        filed proposal is structurally un-claimable (see the module-level
        `_FILED_TICKETS_SCHEMA_SQL` docstring). Triage promotion into a
        real `tickets` row is a separate, later capability (bead .42),
        out of scope here.
        """
        now = time.time()
        with self._db_lock:
            self._conn.execute(
                "INSERT INTO filed_tickets ("
                "  id, title, description, evidence, origin_role, origin_worker,"
                "  origin_dispatch_id, origin_parent_ticket, discovered_from,"
                "  proposal_hash, created_at, triaged"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
                (
                    id,
                    title,
                    description,
                    evidence,
                    origin_role,
                    origin_worker,
                    origin_dispatch_id,
                    origin_parent_ticket,
                    discovered_from,
                    proposal_hash,
                    now,
                ),
            )
            self._conn.commit()

    def list_filed_tickets(self, *, triaged: bool | None = None) -> list[dict[str, Any]]:
        """List `filed_tickets` rows as dicts, ordered by `created_at`.

        `triaged=False` -> only untriaged rows (`WHERE triaged = 0`);
        `triaged=True` -> only triaged rows (`WHERE triaged = 1`);
        `triaged=None` (default) -> all rows, no filter.
        """
        with self._db_lock:
            if triaged is None:
                rows = self._conn.execute(
                    "SELECT * FROM filed_tickets ORDER BY created_at"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM filed_tickets WHERE triaged = ? ORDER BY created_at",
                    (1 if triaged else 0,),
                ).fetchall()
        return [dict(row) for row in rows]

    def count_untriaged_filings(self) -> int:
        """Count `filed_tickets` rows with `triaged = 0` (status.py D14 field)."""
        with self._db_lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM filed_tickets WHERE triaged = 0"
            ).fetchone()
        return int(row["n"])

    def mark_filed_ticket_triaged(
        self, filed_id: str, *, outcome: str, resulting_ticket_id: str | None = None
    ) -> None:
        """Mark a filed proposal triaged (bead .42, SPEC §6 item 10 minors).

        Sets ``triaged=1``, ``triage_outcome=outcome``,
        ``resulting_ticket_id=resulting_ticket_id`` on the `filed_tickets`
        row identified by ``filed_id``. The reject/tombstone path uses
        ``outcome='rejected'`` with no ``resulting_ticket_id``; the promote
        path (`promote_filed_ticket`) uses ``outcome='promoted'`` with the
        new ticket id. NEVER deletes the row — filed proposals are
        append-only history. Raises :class:`ValueError` if ``filed_id`` is
        absent, or if it is already triaged (``triaged == 1``) — a filed
        row can only be triaged once.
        """
        with self._db_lock:
            row = self._conn.execute(
                "SELECT triaged FROM filed_tickets WHERE id = ?", (filed_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"no such filed ticket: {filed_id!r}")
            if row["triaged"]:
                raise ValueError(f"filed ticket already triaged: {filed_id!r}")

            self._conn.execute(
                "UPDATE filed_tickets SET triaged = 1, triage_outcome = ?, "
                "resulting_ticket_id = ? WHERE id = ?",
                (outcome, resulting_ticket_id, filed_id),
            )
            self._conn.commit()

    def promote_filed_ticket(
        self, *, filed_id: str, ticket_id: str, title: str, **ticket_fields: Any
    ) -> None:
        """Complete a filed proposal into a real, UNAPPROVED ticket (bead
        .42, SPEC §6 item 10 / D2 / D15).

        ATOMIC: a single transaction inserts the new `tickets` row (``id=
        ticket_id``, ``title``, ``**ticket_fields``, ``approved`` forced to
        0) AND updates the `filed_tickets` row (``triaged=1``,
        ``triage_outcome='promoted'``, ``resulting_ticket_id=ticket_id``).
        Reuses `_encode_ticket_fields` (same column allowlist + JSON
        encoding as `add_ticket` — no duplicated rules).

        Promotion and approval are distinct acts (D2): the ticket ALWAYS
        lands unapproved, so ``approved``/``approval_hash`` are rejected
        outright if present in ``ticket_fields``.

        Raises :class:`ValueError` (nothing is written in any of these
        cases):
        - ``approved`` or ``approval_hash`` present in ``ticket_fields``;
        - ``filed_id`` is absent or already triaged;
        - ``ticket_id`` already exists in `tickets`;
        - an unknown ticket field is passed.

        On any failure, `self._conn.rollback()` is called so the filed row
        is left untriaged — a failed promotion (e.g. a duplicate ticket id)
        never half-tombstones the filed proposal.
        """
        forbidden = {"approved", "approval_hash"} & set(ticket_fields)
        if forbidden:
            raise ValueError(
                f"promote_filed_ticket forbids caller-supplied field(s): {sorted(forbidden)} "
                "— a promoted ticket always lands UNAPPROVED"
            )

        with self._db_lock:
            try:
                row = self._conn.execute(
                    "SELECT triaged FROM filed_tickets WHERE id = ?", (filed_id,)
                ).fetchone()
                if row is None:
                    raise ValueError(f"no such filed ticket: {filed_id!r}")
                if row["triaged"]:
                    raise ValueError(f"filed ticket already triaged: {filed_id!r}")

                existing = self._conn.execute(
                    "SELECT 1 FROM tickets WHERE id = ?", (ticket_id,)
                ).fetchone()
                if existing is not None:
                    raise ValueError(f"ticket id already exists: {ticket_id!r}")

                now = time.time()
                columns = ["id", "title", "created_at", "updated_at", "approved"]
                values: list[Any] = [ticket_id, title, now, now, 0]
                extra_columns, extra_values = self._encode_ticket_fields(ticket_fields)
                columns.extend(extra_columns)
                values.extend(extra_values)

                col_sql = ", ".join(columns)
                placeholders = ", ".join("?" for _ in columns)
                self._conn.execute(
                    f"INSERT INTO tickets ({col_sql}) VALUES ({placeholders})",  # noqa: S608
                    values,
                )

                self._conn.execute(
                    "UPDATE filed_tickets SET triaged = 1, triage_outcome = 'promoted', "
                    "resulting_ticket_id = ? WHERE id = ?",
                    (ticket_id, filed_id),
                )
            except Exception:
                self._conn.rollback()
                raise
            else:
                self._conn.commit()

    def close(self) -> None:
        """Checkpoint the WAL into the main file and close the connection.

        The connection is always closed, even if the checkpoint itself
        raises (e.g. contention) — callers must be able to rely on
        `close()` releasing the file handle unconditionally.

        Idempotent (bead .143 tail): a second close is a no-op. Daemon and
        ResolvedRig can share one store, and teardown chains that close
        both must not die on the second checkpoint (ProgrammingError on a
        closed connection).
        """
        if self._conn is None:
            return
        try:
            self._conn.execute("PRAGMA wal_checkpoint(FULL)")
        finally:
            conn, self._conn = self._conn, None
            conn.close()

    def _row_to_ticket(self, row: sqlite3.Row) -> dict[str, Any]:
        ticket = dict(row)
        for key in _JSON_TICKET_FIELDS:
            if ticket.get(key) is not None:
                ticket[key] = json.loads(ticket[key])
        return ticket


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
        "# Populated by a later ticket: pinned base-image digest, no-secret build\n"
        "# through the `registries` egress group (SPEC.md §3 provision, §4).\n"
    )

    store = RigStore.create(rig_root / "tickets.db")
    try:
        store.set_meta("schema_version", "5")
        store.set_meta("stigmergy_version", __version__)
        store.set_meta("charter_hash", charter.resolved_hash)
        store.set_meta("rig_name", charter.raw["rig"]["name"])
        store.set_meta("created_at", str(time.time()))
    finally:
        store.close()

    _clone_repo(repo, rig_root / "repo")
    _ensure_dispatch_base_branch(rig_root / "repo", charter.raw["tiers"]["dispatch_base"])

    # bead .79: build the per-rig worker image (base + [provision] deps) so a
    # real code ticket's ruff/pytest Tier-1 checks run in-cage. Only when the
    # charter declares [provision] — rigs without it use the base image
    # unchanged (the mechanical grep-smoke path). The built runnable digest
    # lands in rig meta ("worker_image"); resolve_rig surfaces it to the
    # daemon/checker.
    if "provision" in charter.raw:
        provision_rig_image(rig_root, charter)


def _ensure_dispatch_base_branch(dest: Path, dispatch_base: str) -> None:
    """Create the charter's ``dispatch_base`` branch in the freshly-cloned rig
    repo if it does not already exist (bead .90).

    Workers dispatch FROM and the weaver lands ONTO
    ``refs/heads/<dispatch_base>`` (e.g. ``staging``). A fresh ``git clone``
    only carries the source's default branch, so unless ``dispatch_base``
    happens to BE that default it is absent — and the daemon would resolve
    ``base_oid`` to ``None`` (``_build_execution``) and fail every dispatch
    obscurely. Creating it here — at the clone's current HEAD, no checkout, no
    commit, so no git committer identity is needed — makes a scaffolded rig
    functional end-to-end with no manual ceremony (previously a hand step).
    Idempotent: if the branch already exists (``dispatch_base`` IS the default
    branch) this is a no-op.
    """
    exists = subprocess.run(  # noqa: S603
        ["git", "-C", str(dest), "rev-parse", "--verify", "--quiet", f"refs/heads/{dispatch_base}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if exists.returncode == 0:
        return
    created = subprocess.run(  # noqa: S603
        ["git", "-C", str(dest), "branch", dispatch_base, "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if created.returncode != 0:
        raise RigError(
            f"could not create dispatch_base branch {dispatch_base!r} in the rig repo "
            f"(exit {created.returncode}): {created.stderr.strip()}"
        )


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


# --- resolve: read-side inverse of create_rig -------------------------------


def _build_oa_wheelhouse(wheels_dir: Path) -> None:
    """bead .149: build the host-side openalph wheelhouse (spec §5:
    ``pip wheel /opt/openalph -w <rig_root>/images/worker/wheels/`` — OA +
    all deps incl. matrix-nio; heavy but one-time per rig; network is
    available host-side at provision time, so the in-image install itself
    can be fully offline ``--no-index --find-links=/wheels``).

    pip builds a directory source IN the source tree (setuptools writes
    ``egg-info`` next to the package), and ``/opt/openalph`` is root-owned
    and read-only for the rig owner — so the wheel is built from a
    throwaway WRITABLE COPY (git/venv/caches excluded; the wheel bytes are
    identical). Caught live in the .149 dogfood scaffold
    ("Cannot update time stamp of directory 'src/openalph.egg-info'").

    Two-step build, because the HOST python (3.13) and the WORKER IMAGE
    python (3.11, bookworm) differ: ``pip wheel`` emits binary dependency
    wheels tagged for the running interpreter (cp313), which the image's
    pip (3.11, ``--no-index``) cannot install. So: (1) build the OA wheel
    alone with ``--no-deps`` (pure-python, ``py3-none-any`` —
    interpreter-agnostic); (2) resolve OA's dependency tree from THAT wheel
    with ``pip download --python-version 311 --only-binary=:all:
    --platform manylinux…`` so every binary dep lands as a cp311-compatible
    wheel. Caught live in the .149 dogfood scaffold (podman build failed in
    the in-image offline install).

    Raises :class:`RigError` on any pip failure — a broken wheelhouse must
    fail loud at scaffold time, not surface as an opaque build error in the
    digest-pinned ``build_image`` step. Module-level seam so tests can
    monkeypatch it (no real network in the unit suite)."""
    wheels_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="oa-wheelbuild-") as tmp:
        tmp_src = Path(tmp) / "openalph-src"
        shutil.copytree(
            _OA_SOURCE_DIR,
            tmp_src,
            ignore=shutil.ignore_patterns(
                ".git", ".venv", "__pycache__", "*.pyc", ".pytest_cache",
                "sessions", "logs", "*.egg-info",
            ),
        )
        oa_wheel = subprocess.run(  # noqa: S603
            [
                sys.executable, "-m", "pip", "wheel",
                "--no-deps",
                str(tmp_src),
                "-w", str(wheels_dir),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if oa_wheel.returncode != 0:
            raise RigError(
                "openalph wheel build failed (exit "
                f"{oa_wheel.returncode}): {oa_wheel.stderr.strip()}"
            )
        built = sorted(wheels_dir.glob("openalph-*.whl"))
        if not built:
            raise RigError("openalph wheel build produced no openalph-*.whl")
        # Deps of the just-built wheel, resolved for the IMAGE interpreter.
        dep_dl = subprocess.run(  # noqa: S603
            [
                sys.executable, "-m", "pip", "download",
                str(built[-1]),
                "-d", str(wheels_dir),
                "--python-version", "311",
                "--platform", "manylinux_2_17_x86_64",
                "--platform", "manylinux2014_x86_64",
                "--only-binary=:all:",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    if dep_dl.returncode != 0:
        raise RigError(
            "openalph dependency wheelhouse download failed (exit "
            f"{dep_dl.returncode}): {dep_dl.stderr.strip()}"
        )


def provision_rig_image(rig_root: Path, charter: Charter, *, store: RigStore | None = None) -> str:
    """Build the per-rig worker image (bead .79) and record it in rig meta.

    Extends the charter's pinned base worker image (`[rig].image`) with `pip`
    plus the `[provision].pip` package specs (the check tools ruff/pytest and
    any third-party project deps), so a real code ticket's Tier-1 checks run
    in-cage instead of dying ``command not found``. Deps are baked at BUILD
    time (provision has the `:registries` egress); the RUNTIME worker lane
    stays inference-only.

    bead .149: ``[provision] oa_wheelhouse = true`` additionally bakes the
    ``openalph`` worker stack into the image — a host-built wheelhouse
    (``pip wheel /opt/openalph -w images/worker/wheels/``; network available
    at PROVISION time, so the in-image install is fully offline
    ``--no-index --find-links=/wheels``) + the packaged
    ``worker_image/oa-worker.toml`` agent template copied to
    ``/etc/openalph/agents/stigmergy-worker.toml``. Key absent (the default)
    -> the generated Containerfile is byte-identical to the pre-.149 output.

    Only tools/deps are baked — NEVER the project package itself. A
    ``pip install -e .`` would make ``import <project>`` resolve to the stale
    build-time copy, so the checker would validate the wrong code; the checker
    must read the ``/work`` candidate at runtime (see ``tmp/bead79-design.md``).

    Writes ``images/worker/Containerfile`` (overwriting the scaffold stub),
    builds via :func:`stigmergy.container.build_image` (which rejects any
    non-pinned FROM — the bare ``sha256:`` base is accepted, bead .79), stores
    the resulting runnable digest under rig meta key ``worker_image``, and
    returns it. Opens its own store if one is not supplied.
    """
    base_image = charter.raw["rig"]["image"]
    provision_cfg = charter.raw.get("provision", {})
    pip_specs = provision_cfg.get("pip", [])
    oa_wheelhouse = bool(provision_cfg.get("oa_wheelhouse", False))

    images_dir = rig_root / "images" / "worker"
    images_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"FROM {base_image}"]
    if oa_wheelhouse or pip_specs:
        # The base worker image carries python3 but NOT python3-pip (node +
        # python3 + git only) — bootstrap it exactly once when EITHER stack
        # needs it.
        lines.append(
            "RUN apt-get update "
            "&& apt-get install -y --no-install-recommends python3-pip "
            "&& rm -rf /var/lib/apt/lists/*"
        )
    if oa_wheelhouse:
        # bead .149 (spec §5): the openalph worker stack, BEFORE the
        # existing pip line (spec §5 ordering). (1) host-side wheelhouse
        # (the one network step — at provision time, not build time); (2)
        # the fully-offline in-image install; (3) the agent TOML.
        _build_oa_wheelhouse(images_dir / "wheels")
        lines.append("COPY wheels/ /wheels/")
        lines.append(
            "RUN pip3 install --break-system-packages --no-cache-dir "
            "--no-index --find-links=/wheels openalph"
        )
        shutil.copy(_OA_WORKER_TOML, images_dir / "oa-worker.toml")
        lines.append("COPY oa-worker.toml /etc/openalph/agents/stigmergy-worker.toml")
    if pip_specs:
        lines.append(
            "RUN pip3 install --break-system-packages --no-cache-dir "
            + " ".join(shlex.quote(spec) for spec in pip_specs)
        )
    (images_dir / "Containerfile").write_text("\n".join(lines) + "\n")

    tag = f"localhost/stigmergy-rig-{charter.raw['rig']['name']}:latest"
    digest = build_image(images_dir, tag)

    owns_store = store is None
    if owns_store:
        store = RigStore(rig_root / "tickets.db")
    try:
        store.set_meta("worker_image", digest)
    finally:
        if owns_store:
            store.close()
    return digest


@dataclass(frozen=True)
class ResolvedRig:
    """Everything needed to construct a real `Daemon` for an already-scaffolded rig.

    Lifecycle contract: `store` is opened (NOT `.create()`'d) and returned OPEN. A
    long-lived caller (daemon run_forever) owns it for the process lifetime and never
    closes it. A SHORT-LIVED caller (bead .33's status/tickets/range-report subcommands)
    MUST call `store.close()` itself when done — `resolve_rig` does not know the
    caller's lifetime and never closes what it opens.
    """

    rig_root: Path
    charter: Charter
    registry: Registry
    store: RigStore
    rig_paths: dict[str, Path]  # exactly daemon._REQUIRED_RIG_PATH_KEYS' 5 keys
    # bead .79: the effective worker/checker image — the per-rig built image
    # (rig meta "worker_image") if provision ran, else the charter's base
    # `[rig].image`. The daemon uses THIS for both worker dispatch + checker.
    worker_image: str


def resolve_rig(name: str, *, rigs_root: str | os.PathLike[str] | None = None) -> ResolvedRig:
    """Load an ALREADY-SCAFFOLDED rig by name (SPEC §3; the read-side inverse of
    `create_rig`). `rigs_root` defaults to `~/rigs` (mirrors `create_rig`'s own
    `base_dir` default) — a caller-supplied value is used verbatim (Path(...).expanduser()).

    Rig root is `<rigs_root>/<name>`.

    Fail-closed existence checks BEFORE opening anything (mirrors create_rig's own
    error-handling weight class — does not schema-validate the sqlite file, that
    duplicates what create_rig already guarantees):
    - `rig_root` itself must exist and be a directory -> RigError
    - `rig_root/charter.toml` must exist -> RigError
    - `rig_root/models.toml` must exist -> RigError
    - `rig_root/tickets.db` must exist -> RigError
    Lists ALL missing expected paths in ONE RigError message if more than one is
    missing (does not fail on the first check only — a caller fixing a
    half-scaffolded rig wants the whole picture at once).

    Then, in order:
    1. `charter = load_charter(rig_root / "charter.toml")` — env defaults to
       `os.environ` (real process env, NOT `{}` — this is production code, unlike
       test fixtures). `CharterError` propagates UNCAUGHT (a chartering bug, not a
       resolve_rig bug — mirrors create_rig's own uncaught-CharterError precedent).
       NOTE: `load_charter` ALSO loads + validates the model registry named by
       the charter's `[models].registry` (to resolve lane/critic models) and
       wraps any `UnbudgetableError` from it as `CharterError`. So for the
       common scaffolded rig (where `[models].registry == "models.toml"`), a
       broken registry surfaces HERE as `CharterError`, and step 2 below is
       never reached — a documented deviation from the .27 build spec's
       original prediction of `UnbudgetableError`. Fail-closed either way.
    2. `registry = load_registry(rig_root / "models.toml")` — loads the
       rig-local registry to populate `ResolvedRig.registry` (the `Charter`
       object does not expose the registry step 1 loaded). `UnbudgetableError`
       propagates uncaught. This is only distinct from step 1 when a
       hand-edited charter points `[models].registry` at some OTHER (valid)
       file while `rig_root/models.toml` is broken — a genuine, if narrow,
       defense-in-depth path (see tests case 8b).
    3. `prompts_rel = charter.raw.get("prompts", {}).get("dir")` — if this is falsy
       (missing `[prompts]` table, or missing/empty `dir` key), raises `RigError`
       (fail closed: never silently pick a made-up prompts location).
    4. `store = RigStore(rig_root / "tickets.db")` — plain open, not `.create()`.
    5. Builds `rig_paths`:
       - `context_root = rig_root / "context"`
       - `repo_root    = rig_root / "repo"`
       - `clones_root  = rig_root / "clones"`
       - `records_dir  = rig_root / "records"`
       - `prompts_dir  = repo_root / prompts_rel` (see §3.1 of the .27 build spec for
         the reasoning — NOT `rig_root / prompts_rel`, and NOT relative to
         `charter.toml`'s own directory: `create_rig` never copies `prompts/` into
         `rig_root`, only the rig's OWN git clone of `[rig].repo` at `repo_root` can
         be relied on to carry it, and only that path is hash-protected by the
         already-built approval mechanism).

    Returns `ResolvedRig(rig_root, charter, registry, store, rig_paths)`. Does **not**
    validate `repo_root` is actually a git repository, or that any file under
    `prompts_dir` exists — those are the caller's problem (mirrors `steering.py`'s own
    `derive_steering`, which raises its own `SteeringError` lazily, on first read, not
    up front; and mirrors `Daemon._git_rev_parse`'s own graceful-`None`-on-failure
    design — validating git-repo-ness here would tax `.33`'s git-untouching
    status/tickets/range-report commands for no benefit).
    """
    base = Path(rigs_root).expanduser() if rigs_root is not None else Path("~/rigs").expanduser()
    rig_root = base / name

    missing: list[str] = []
    if not rig_root.is_dir():
        missing.append(f"rig root directory: {rig_root}")
    else:
        if not (rig_root / "charter.toml").is_file():
            missing.append(f"charter: {rig_root / 'charter.toml'}")
        if not (rig_root / "models.toml").is_file():
            missing.append(f"model registry: {rig_root / 'models.toml'}")
        if not (rig_root / "tickets.db").is_file():
            missing.append(f"ticket store: {rig_root / 'tickets.db'}")

    if missing:
        joined = "; ".join(missing)
        raise RigError(f"rig {name!r} is not a valid scaffolded rig — missing: {joined}")

    charter = load_charter(rig_root / "charter.toml")
    registry = load_registry(rig_root / "models.toml")

    prompts_rel = charter.raw.get("prompts", {}).get("dir")
    if not prompts_rel:
        raise RigError(
            f"rig {name!r}'s charter has no [prompts].dir — cannot resolve a prompts directory"
        )

    store = RigStore(rig_root / "tickets.db")

    # bead .79: prefer the per-rig built image (provision stored it in meta);
    # fall back to the charter's base image for rigs scaffolded without
    # [provision] (e.g. the mechanical grep smoke).
    worker_image = store.get_meta("worker_image") or charter.raw["rig"]["image"]

    repo_root = rig_root / "repo"
    rig_paths: dict[str, Path] = {
        "context_root": rig_root / "context",
        "repo_root": repo_root,
        "clones_root": rig_root / "clones",
        "prompts_dir": repo_root / prompts_rel,
        "records_dir": rig_root / "records",
    }

    return ResolvedRig(
        rig_root=rig_root,
        charter=charter,
        registry=registry,
        store=store,
        rig_paths=rig_paths,
        worker_image=worker_image,
    )
