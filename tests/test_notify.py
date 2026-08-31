"""Tests for stigmergy.notify (SPEC.md §9 "Notifications", §10 AC12;
bead .23 build spec §1).

Case numbering below matches the bead .23 build spec's frozen case list
(build spec §3 "tests/test_notify.py", cases 1-7).

Governing invariant under test (AC12, cases 4-5): an ntfy outage can NEVER
silently eat an escalation. `NtfyNotifier.deliver_pending` is fail-SOFT —
a delivery failure never propagates; the intent stays pending, persisted,
ready for the next retry.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pytest

from stigmergy.notify import (
    NotificationStore,
    NotifyError,
    NtfyNotifier,
)


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "records" / "notifications.jsonl"


@pytest.fixture
def store(store_path: Path) -> NotificationStore:
    return NotificationStore(store_path)


class WorkingSender:
    """A fake `Sender` that always succeeds and records its calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, topic: str, title: str, message: str) -> None:
        self.calls.append((topic, title, message))


class OutageSender:
    """A fake `Sender` that always raises (simulates an ntfy outage)."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, topic: str, title: str, message: str) -> None:
        self.calls += 1
        raise ConnectionError("ntfy is down")


# --- case 1: test_record_intent_persists_pending ----------------------------


def test_record_intent_persists_pending(store: NotificationStore) -> None:
    intent = store.record_intent(
        ticket="workspace-e2uh.23",
        kind="escalation",
        title="ticket escalated",
        message="ladder exhausted",
        now=1000.0,
    )

    pending = store.pending()
    assert len(pending) == 1
    assert pending[0].intent_id == intent.intent_id
    assert pending[0].delivered_at is None
    assert pending[0].attempts == 0
    assert pending[0].ticket == "workspace-e2uh.23"
    assert pending[0].kind == "escalation"


# --- case 2: test_persistence_survives_reopen -------------------------------


def test_persistence_survives_reopen(store_path: Path) -> None:
    store1 = NotificationStore(store_path)
    intent1 = store1.record_intent(
        ticket="t1", kind="escalation", title="t1 escalated", message="m1", now=1.0
    )
    intent2 = store1.record_intent(
        ticket="t2", kind="budget-exhausted", title="budget gone", message="m2", now=2.0
    )
    store1.mark_delivered(intent2.intent_id, now=3.0)
    del store1  # drop the store object entirely

    store2 = NotificationStore(store_path)
    all_intents = {i.intent_id: i for i in store2.all_intents()}

    assert len(all_intents) == 2
    assert all_intents[intent1.intent_id].delivered_at is None
    assert all_intents[intent2.intent_id].delivered_at == 3.0
    pending = store2.pending()
    assert [i.intent_id for i in pending] == [intent1.intent_id]


# --- case 3: test_deliver_pending_success_marks_delivered -------------------


def test_deliver_pending_success_marks_delivered(store: NotificationStore) -> None:
    intent = store.record_intent(
        ticket="t1", kind="escalation", title="my title", message="my message", now=1.0
    )
    sender = WorkingSender()
    notifier = NtfyNotifier("my-topic", sender=sender)

    result = notifier.deliver_pending(store, now=2.0)

    assert result == {"delivered": 1, "still_pending": 0, "attempted": 1}
    assert store.pending() == []
    delivered = store.all_intents()[0]
    assert delivered.intent_id == intent.intent_id
    assert delivered.delivered_at == 2.0
    assert sender.calls == [("my-topic", "my title", "my message")]


# --- case 4: test_ntfy_outage_leaves_intent_pending (AC12 core) -------------


def test_ntfy_outage_leaves_intent_pending(store: NotificationStore) -> None:
    intent = store.record_intent(
        ticket="t1", kind="escalation", title="escalation!", message="ladder exhausted", now=1.0
    )
    sender = OutageSender()
    notifier = NtfyNotifier("my-topic", sender=sender)

    # Must NOT raise, even though the sender raises.
    result = notifier.deliver_pending(store, now=2.0)

    assert result["still_pending"] >= 1
    assert result["delivered"] == 0
    pending = store.pending()
    assert len(pending) == 1
    assert pending[0].intent_id == intent.intent_id
    assert pending[0].delivered_at is None
    assert pending[0].attempts == 1


# --- case 5: test_retry_delivers_after_outage_ends --------------------------


def test_retry_delivers_after_outage_ends(store: NotificationStore) -> None:
    store.record_intent(
        ticket="t1", kind="escalation", title="escalation!", message="ladder exhausted", now=1.0
    )
    outage_sender = OutageSender()
    outage_notifier = NtfyNotifier("my-topic", sender=outage_sender)

    result1 = outage_notifier.deliver_pending(store, now=2.0)
    assert result1["still_pending"] == 1
    assert store.pending()[0].attempts == 1

    working_sender = WorkingSender()
    working_notifier = NtfyNotifier("my-topic", sender=working_sender)
    result2 = working_notifier.deliver_pending(store, now=3.0)

    assert result2 == {"delivered": 1, "still_pending": 0, "attempted": 1}
    assert store.pending() == []
    delivered = store.all_intents()[0]
    assert delivered.delivered_at == 3.0
    # An escalation is never silently lost: it eventually got delivered.
    assert len(working_sender.calls) == 1


# --- case 6: test_mark_delivered_unknown_id_raises --------------------------


def test_mark_delivered_unknown_id_raises(store: NotificationStore) -> None:
    with pytest.raises(NotifyError):
        store.mark_delivered("no-such-intent-id", now=1.0)


# --- case 7: test_tolerant_reader_skips_torn_tail ---------------------------


def test_tolerant_reader_skips_torn_tail(store: NotificationStore, store_path: Path) -> None:
    intent = store.record_intent(
        ticket="t1", kind="escalation", title="title1", message="message1", now=1.0
    )

    # Simulate a crash mid-append: a truncated/garbage partial line with no
    # trailing newline, appended directly (bypassing NotificationStore).
    with open(store_path, "a", encoding="utf-8") as fh:
        fh.write('{"intent_id": "torn-tail-record", "ticket": "t2", "ki')

    intents = store.all_intents()

    assert len(intents) == 1
    assert intents[0].intent_id == intent.intent_id
    # pending() must also tolerate the torn tail without raising.
    pending = store.pending()
    assert len(pending) == 1
    assert pending[0].intent_id == intent.intent_id


# --- bead .117: bearer-token auth on the production sender ------------------
# The self-hosted ntfy server now REQUIRES auth to publish (confirmed live
# 2026-08-31, execdogfood01: every escalation POST 403'd ~100 retries,
# delivered_at:null). make_ntfy_sender gains an OPTIONAL bearer token —
# absent token = byte-identical pre-.117 behavior (no Authorization header).


class _CapturingURLOpener:
    """Monkeypatch stand-in for urllib.request.urlopen that records the
    request's headers and returns a minimal 200 response."""

    def __init__(self) -> None:
        self.requests: list[urllib.request.Request] = []

    def __call__(self, request: urllib.request.Request, timeout: float | None = None):
        self.requests.append(request)

        class _Resp:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return _Resp()


def test_ntfy_sender_with_token_sends_bearer_header(monkeypatch) -> None:
    """bead .117: a configured token rides `Authorization: Bearer <token>`
    on the publish POST — the self-hosted server 403s anonymous publishes."""
    import urllib.request

    from stigmergy.notify import make_ntfy_sender

    capture = _CapturingURLOpener()
    monkeypatch.setattr(urllib.request, "urlopen", capture)
    sender = make_ntfy_sender("http://127.0.0.1:8090", token="secret-token-value")
    sender("stigmergy", "t", "m")
    assert len(capture.requests) == 1
    auth = capture.requests[0].headers.get("Authorization")
    assert auth == "Bearer secret-token-value"


def test_ntfy_sender_without_token_sends_no_authorization_header(monkeypatch) -> None:
    """Byte-identical pre-.117 default: token=None emits NO Authorization
    header (never `Bearer None`, never a placeholder — the .147 F1
    discipline for optional credentials)."""
    import urllib.request

    from stigmergy.notify import make_ntfy_sender

    capture = _CapturingURLOpener()
    monkeypatch.setattr(urllib.request, "urlopen", capture)
    sender = make_ntfy_sender("http://127.0.0.1:8090")
    sender("stigmergy", "t", "m")
    assert "Authorization" not in capture.requests[0].headers
