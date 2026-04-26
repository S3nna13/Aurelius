"""Tests for src.serving.hermes_notifier."""

from __future__ import annotations

import time

import pytest

from src.serving.hermes_notifier import (
    DEFAULT_HERMES_NOTIFIER,
    HERMES_NOTIFIER_REGISTRY,
    HermesError,
    HermesNotifier,
    Notification,
)

# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------


def test_notification_dataclass():
    n = Notification(
        notification_id="n1",
        timestamp=time.time(),
        channel="web",
        priority="high",
        category="agent",
        title="Test",
        body="Body",
    )
    assert n.read is False
    assert n.delivered is False


def test_hermes_error_exists():
    assert issubclass(HermesError, Exception)


# ---------------------------------------------------------------------------
# Notify & query
# ---------------------------------------------------------------------------


def test_notify_basic():
    h = HermesNotifier()
    n = h.notify("Title", "Body")
    assert isinstance(n, Notification)
    assert n.title == "Title"
    assert n.body == "Body"
    assert n.channel == "web"
    assert n.delivered is True


def test_notify_with_options():
    h = HermesNotifier()
    n = h.notify(
        "Alert",
        "Disk full",
        channel="sms",
        priority="critical",
        category="alert",
        metadata={"disk": "/"},
    )
    assert n.priority == "critical"
    assert n.category == "alert"
    assert n.metadata == {"disk": "/"}


def test_list_notifications_newest_first():
    h = HermesNotifier()
    h.notify("First", "1")
    h.notify("Second", "2")
    results = h.list_notifications()
    assert results[0].title == "Second"
    assert results[1].title == "First"


def test_list_filter_by_category():
    h = HermesNotifier()
    h.notify("A", "a", category="agent")
    h.notify("S", "s", category="system")
    assert len(h.list_notifications(category="agent")) == 1
    assert h.list_notifications(category="agent")[0].title == "A"


def test_list_filter_by_priority():
    h = HermesNotifier()
    h.notify("H", "h", priority="high")
    h.notify("L", "l", priority="low")
    assert len(h.list_notifications(priority="high")) == 1


def test_list_filter_by_read():
    h = HermesNotifier()
    n = h.notify("X", "x")
    h.mark_read(n.notification_id)
    assert len(h.list_notifications(read=True)) == 1
    assert len(h.list_notifications(read=False)) == 0


def test_list_limit():
    h = HermesNotifier()
    for i in range(5):
        h.notify(str(i), "body")
    assert len(h.list_notifications(limit=2)) == 2


# ---------------------------------------------------------------------------
# Read / unread
# ---------------------------------------------------------------------------


def test_mark_read():
    h = HermesNotifier()
    n = h.notify("X", "x")
    assert h.mark_read(n.notification_id) is True
    assert n.read is True


def test_mark_read_unknown():
    h = HermesNotifier()
    assert h.mark_read("nope") is False


def test_mark_all_read():
    h = HermesNotifier()
    h.notify("A", "a")
    h.notify("B", "b")
    assert h.mark_all_read() == 2
    assert h.unread_count() == 0


def test_mark_all_read_by_category():
    h = HermesNotifier()
    h.notify("A", "a", category="agent")
    h.notify("S", "s", category="system")
    assert h.mark_all_read(category="agent") == 1
    assert h.unread_count(category="system") == 1


def test_unread_count():
    h = HermesNotifier()
    h.notify("A", "a")
    h.notify("B", "b")
    assert h.unread_count() == 2
    h.mark_all_read()
    assert h.unread_count() == 0


# ---------------------------------------------------------------------------
# Listeners
# ---------------------------------------------------------------------------


def test_subscribe_and_receive():
    h = HermesNotifier()
    received: list[Notification] = []
    unsub = h.subscribe(lambda n: received.append(n))
    h.notify("Test", "body")
    assert len(received) == 1
    assert received[0].title == "Test"
    unsub()
    h.notify("After", "body")
    assert len(received) == 1


# ---------------------------------------------------------------------------
# Queue bounds
# ---------------------------------------------------------------------------


def test_queue_eviction_when_full():
    h = HermesNotifier(max_queue_size=3)
    h.notify("1", "a")
    h.notify("2", "b")
    h.notify("3", "c")
    h.notify("4", "d")
    assert len(h.list_notifications()) == 3


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def test_stats():
    h = HermesNotifier()
    h.notify("A", "a", channel="web", priority="high")
    h.notify("B", "b", channel="sms", priority="low")
    s = h.stats()
    assert s["total"] == 2
    assert s["unread"] == 2
    assert s["by_channel"]["web"] == 1
    assert s["by_channel"]["sms"] == 1
    assert s["by_priority"]["high"] == 1
    assert s["by_priority"]["low"] == 1


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


def test_clear():
    h = HermesNotifier()
    h.notify("X", "x")
    h.clear()
    assert h.list_notifications() == []


# ---------------------------------------------------------------------------
# SMS / email placeholders
# ---------------------------------------------------------------------------


def test_sms_placeholder_delivery():
    h = HermesNotifier()
    n = h.notify("SMS", "test", channel="sms")
    assert n.delivered is True


def test_email_placeholder_delivery():
    h = HermesNotifier()
    n = h.notify("Email", "test", channel="email")
    assert n.delivered is True


# ---------------------------------------------------------------------------
# Webhook delivery (mocked via monkeypatch)
# ---------------------------------------------------------------------------


def test_webhook_not_configured():
    h = HermesNotifier()
    n = h.notify("Hook", "test", channel="webhook")
    assert n.delivered is False


def test_webhook_configured_but_invalid_url(monkeypatch):
    h = HermesNotifier()
    h.configure_webhook("http://localhost:9/nowhere")
    n = h.notify("Hook", "test", channel="webhook")
    assert n.delivered is False


# ---------------------------------------------------------------------------
# Singleton / registry
# ---------------------------------------------------------------------------


def test_default_singleton():
    assert isinstance(DEFAULT_HERMES_NOTIFIER, HermesNotifier)


def test_registry_contains_default():
    assert "default" in HERMES_NOTIFIER_REGISTRY
    assert HERMES_NOTIFIER_REGISTRY["default"] is DEFAULT_HERMES_NOTIFIER
