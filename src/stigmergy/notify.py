"""Persisted notification intents + ntfy delivery with retry (SPEC.md §9
"Notifications", §10 AC12; bead .23 build spec §1).

**The load-bearing property (AC12): an ntfy outage can NEVER silently eat
an escalation.** A :class:`NotificationIntent` is durably persisted the
moment it is recorded — before any delivery attempt is made — and stays
`PENDING` (survives process restarts, survives a torn read) until a
delivery genuinely succeeds. :class:`NtfyNotifier` is fail-**soft** per
intent by design: a delivery failure (the injected `sender` raising) never
propagates out of :meth:`NtfyNotifier.deliver_pending` — it bumps the
intent's attempt count and leaves it pending for the next retry cycle.
`status.py`'s `escalated_unnotified` count is exactly what lets an operator
see a stuck intent in the meantime; nothing here ever drops one silently.

**Persistence (bead .23 build spec §0 DECISION):** this module owns its
OWN append-plus-atomic-rewrite JSONL store — it does NOT touch
`RigStore`/`rig.py`'s schema. New intents are appended (mirrors
`records.RecordPlane.append`'s fsync'd append); a delivery-status flip
(`mark_delivered`/`bump_attempt`) rewrites the whole file via a temp-file +
`os.replace`, mirroring `records._atomic_write_bytes` exactly. The reader
is TOLERANT (mirrors `records.read_events`): a line that fails to parse as
JSON, isn't an object, or is missing a required key is silently skipped —
a torn tail (crash mid-append) never prevents the intact intents before it
from being read back.
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

_STRICT_MODE = 0o600

# The full on-disk field set for one notification intent (SPEC §1 frozen
# `NotificationIntent`). A line missing any of these is a torn/corrupt
# record and is skipped by the tolerant reader, never raised on.
_REQUIRED_FIELDS: tuple[str, ...] = (
    "intent_id",
    "ticket",
    "kind",
    "title",
    "message",
    "created_at",
    "delivered_at",
    "attempts",
)


class NotifyError(Exception):
    """Raised on a `NotificationStore` misuse — currently only an unknown
    `intent_id` passed to `mark_delivered`/`bump_attempt`."""


@dataclass(frozen=True)
class NotificationIntent:
    """One durable notification intent (SPEC §1). ``delivered_at is None``
    means still pending; ``attempts`` counts delivery attempts made so far
    (successful or not — a successful attempt also flips `delivered_at`,
    so `attempts` is a superset signal, not just failure count)."""

    intent_id: str
    ticket: str | None
    kind: str
    title: str
    message: str
    created_at: float
    delivered_at: float | None
    attempts: int


def _intent_to_dict(intent: NotificationIntent) -> dict[str, Any]:
    return {
        "intent_id": intent.intent_id,
        "ticket": intent.ticket,
        "kind": intent.kind,
        "title": intent.title,
        "message": intent.message,
        "created_at": intent.created_at,
        "delivered_at": intent.delivered_at,
        "attempts": intent.attempts,
    }


def _dict_to_intent(record: dict[str, Any]) -> NotificationIntent | None:
    """Validate + convert one decoded JSON line into a
    :class:`NotificationIntent`, or ``None`` if it's malformed (torn tail /
    corrupt record) — the tolerant reader's per-line gate."""
    if any(key not in record for key in _REQUIRED_FIELDS):
        return None
    try:
        delivered_at = record["delivered_at"]
        return NotificationIntent(
            intent_id=str(record["intent_id"]),
            ticket=record["ticket"],
            kind=str(record["kind"]),
            title=str(record["title"]),
            message=str(record["message"]),
            created_at=float(record["created_at"]),
            delivered_at=None if delivered_at is None else float(delivered_at),
            attempts=int(record["attempts"]),
        )
    except (TypeError, ValueError):
        return None


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` via temp-file (same dir) + `os.replace`,
    fsync'd before the rename. Mirrors `records._atomic_write_bytes`
    exactly (this module deliberately does not import that private helper
    from `records.py` — its own persistence file is unrelated to the
    record plane, see module docstring)."""
    directory = path.parent
    fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp_name, _STRICT_MODE)
        os.replace(tmp_name, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


class NotificationStore:
    """Durable notification-intent store (its OWN JSONL file; not the
    RigStore). Single-threaded (the daemon is single-threaded, per SPEC):
    no locking is implemented. New intents are appended (fsync'd); a
    delivery-status change rewrites the whole file atomically. The reader
    is tolerant of a torn/corrupt trailing line."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record_intent(
        self, *, ticket: str | None, kind: str, title: str, message: str, now: float
    ) -> NotificationIntent:
        """Persist a new PENDING intent (`delivered_at=None`, `attempts=0`).
        Returns it."""
        intent = NotificationIntent(
            intent_id=secrets.token_hex(8),
            ticket=ticket,
            kind=kind,
            title=title,
            message=message,
            created_at=now,
            delivered_at=None,
            attempts=0,
        )
        self._append(intent)
        return intent

    def pending(self) -> list[NotificationIntent]:
        """All intents with `delivered_at is None`, in creation order."""
        return [i for i in self._read_all() if i.delivered_at is None]

    def all_intents(self) -> list[NotificationIntent]:
        """Every intent on disk, in creation order."""
        return self._read_all()

    def mark_delivered(self, intent_id: str, *, now: float) -> None:
        """Flip `delivered_at` to ``now`` for ``intent_id``. Raises
        :class:`NotifyError` if no such intent exists."""
        self._update(intent_id, lambda i: replace(i, delivered_at=now))

    def bump_attempt(self, intent_id: str, *, now: float) -> None:
        """Increment `attempts`; the intent stays pending. Raises
        :class:`NotifyError` if no such intent exists. ``now`` is accepted
        for signature symmetry with `mark_delivered` (the frozen interface)
        but is not itself stored on the intent — only `attempts` moves."""
        del now  # reserved, not currently stored (see docstring)
        self._update(intent_id, lambda i: replace(i, attempts=i.attempts + 1))

    # -- internals ---------------------------------------------------------

    def _append(self, intent: NotificationIntent) -> None:
        line = json.dumps(_intent_to_dict(intent), sort_keys=True, separators=(",", ":"))
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, _STRICT_MODE)
        try:
            with os.fdopen(fd, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        finally:
            os.chmod(self.path, _STRICT_MODE)

    def _read_all(self) -> list[NotificationIntent]:
        """Tolerant read: a line that fails to parse as JSON, isn't an
        object, or is missing a required field is silently skipped (torn
        tail from a crash mid-append) — never raises."""
        intents: list[NotificationIntent] = []
        if not self.path.exists():
            return intents
        with open(self.path, encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                intent = _dict_to_intent(record)
                if intent is None:
                    continue
                intents.append(intent)
        return intents

    def _update(
        self, intent_id: str, mutator: Callable[[NotificationIntent], NotificationIntent]
    ) -> None:
        intents = self._read_all()
        updated: list[NotificationIntent] = []
        found = False
        for intent in intents:
            if intent.intent_id == intent_id:
                updated.append(mutator(intent))
                found = True
            else:
                updated.append(intent)
        if not found:
            raise NotifyError(f"unknown intent_id: {intent_id!r}")
        self._rewrite(updated)

    def _rewrite(self, intents: list[NotificationIntent]) -> None:
        lines = [
            json.dumps(_intent_to_dict(intent), sort_keys=True, separators=(",", ":"))
            for intent in intents
        ]
        content = ("\n".join(lines) + "\n") if lines else ""
        _atomic_write_bytes(self.path, content.encode("utf-8"))


# (topic, title, message) -> None; raises on delivery failure.
Sender = Callable[[str, str, str], None]


class NtfyNotifier:
    """Delivers pending intents to an ntfy topic via an INJECTED sender
    (production: HTTP POST via :func:`make_ntfy_sender`; tests: a fake that
    can raise to simulate an outage).

    **Fail-SOFT per intent, by design (AC12).** A delivery failure (the
    injected `sender` raising) never propagates out of
    :meth:`deliver_pending` — it bumps the intent's attempt count and
    leaves it `PENDING` for the next retry. This is the single most
    load-bearing property in this module: an ntfy outage must never
    silently eat an escalation.
    """

    def __init__(self, topic: str, *, sender: Sender) -> None:
        self.topic = topic
        self.sender = sender

    def deliver_pending(self, store: NotificationStore, *, now: float) -> dict[str, int]:
        """Attempt delivery of every currently-pending intent.

        Per intent: `sender(topic, title, message)` succeeds ->
        `store.mark_delivered(...)`; `sender` raises -> caught here (never
        re-raised) and `store.bump_attempt(...)` — the intent stays
        pending, ready for the next call. Returns
        ``{"delivered": int, "still_pending": int, "attempted": int}``.
        """
        pending = store.pending()
        delivered = 0
        still_pending = 0
        for intent in pending:
            try:
                self.sender(self.topic, intent.title, intent.message)
            except Exception:  # noqa: BLE001 - fail-soft: AC12, never lose an escalation
                store.bump_attempt(intent.intent_id, now=now)
                still_pending += 1
                continue
            store.mark_delivered(intent.intent_id, now=now)
            delivered += 1
        return {"delivered": delivered, "still_pending": still_pending, "attempted": len(pending)}


def make_ntfy_sender(server_url: str) -> Sender:
    """Build the production :data:`Sender`: stdlib-only (`urllib.request`)
    HTTP POST to ``f"{server_url}/{topic}"`` with the title carried in the
    ``Title`` header and the message as the request body (ntfy's plain-HTTP
    publish contract). Raises :class:`NotifyError` on a non-2xx response or
    any network error — :meth:`NtfyNotifier.deliver_pending` treats any
    raise from the injected sender as one retryable failure, never a crash.

    NOT exercised by this module's tests (which inject fakes); this is the
    production default, wired in later by the daemon/CLI integration.
    """

    def sender(topic: str, title: str, message: str) -> None:
        url = f"{server_url.rstrip('/')}/{topic}"
        request = urllib.request.Request(  # noqa: S310 - fixed ntfy server URL, not user input
            url,
            data=message.encode("utf-8"),
            method="POST",
            headers={"Title": title},
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
                status = response.status
                if not (200 <= status < 300):
                    raise NotifyError(f"ntfy POST to {url} returned status {status}")
        except urllib.error.URLError as exc:
            raise NotifyError(f"ntfy POST to {url} failed: {exc}") from exc

    return sender
