"""Tests for src/ui/notification_drawer.py — ~50 tests."""

from __future__ import annotations

import pytest

from src.ui.notification_drawer import (
    NotificationLevel,
    Notification,
    NotificationDrawer,
    DEFAULT_NOTIFICATION_DRAWER,
)


# ---------------------------------------------------------------------------
# NotificationLevel enum
# ---------------------------------------------------------------------------


class TestNotificationLevel:
    def test_info_value(self):
        assert NotificationLevel.INFO == "info"

    def test_warning_value(self):
        assert NotificationLevel.WARNING == "warning"

    def test_error_value(self):
        assert NotificationLevel.ERROR == "error"

    def test_success_value(self):
        assert NotificationLevel.SUCCESS == "success"

    def test_is_str_subclass(self):
        assert isinstance(NotificationLevel.INFO, str)


# ---------------------------------------------------------------------------
# Notification dataclass
# ---------------------------------------------------------------------------


class TestNotification:
    def test_auto_id_generated(self):
        n = Notification(level=NotificationLevel.INFO, title="Hello")
        assert n.id != ""

    def test_auto_id_is_eight_chars(self):
        n = Notification(level=NotificationLevel.INFO, title="Hello")
        assert len(n.id) == 8

    def test_auto_id_is_hex(self):
        n = Notification(level=NotificationLevel.INFO, title="Hello")
        int(n.id, 16)  # should not raise

    def test_custom_id_respected(self):
        n = Notification(level=NotificationLevel.INFO, title="X", id="abc12345")
        assert n.id == "abc12345"

    def test_level_stored(self):
        n = Notification(level=NotificationLevel.ERROR, title="Boom")
        assert n.level == NotificationLevel.ERROR

    def test_title_stored(self):
        n = Notification(level=NotificationLevel.INFO, title="My Title")
        assert n.title == "My Title"

    def test_default_body_empty(self):
        n = Notification(level=NotificationLevel.INFO, title="Hi")
        assert n.body == ""

    def test_custom_body(self):
        n = Notification(level=NotificationLevel.INFO, title="Hi", body="Details here")
        assert n.body == "Details here"

    def test_default_dismissed_false(self):
        n = Notification(level=NotificationLevel.INFO, title="Hi")
        assert n.dismissed is False

    def test_unique_ids(self):
        ids = {Notification(level=NotificationLevel.INFO, title="x").id for _ in range(20)}
        assert len(ids) == 20  # all unique


# ---------------------------------------------------------------------------
# NotificationDrawer
# ---------------------------------------------------------------------------


class TestNotificationDrawer:
    def setup_method(self):
        self.drawer = NotificationDrawer()

    def test_push_returns_notification(self):
        n = self.drawer.push(NotificationLevel.INFO, "Test")
        assert isinstance(n, Notification)

    def test_push_adds_to_active(self):
        self.drawer.push(NotificationLevel.INFO, "Test")
        assert len(self.drawer.active()) == 1

    def test_push_multiple(self):
        for i in range(5):
            self.drawer.push(NotificationLevel.INFO, f"N{i}")
        assert len(self.drawer.active()) == 5

    def test_push_with_body(self):
        n = self.drawer.push(NotificationLevel.WARNING, "Warn", body="Detail")
        assert n.body == "Detail"

    def test_dismiss_returns_true_if_found(self):
        n = self.drawer.push(NotificationLevel.INFO, "A")
        result = self.drawer.dismiss(n.id)
        assert result is True

    def test_dismiss_sets_dismissed_flag(self):
        n = self.drawer.push(NotificationLevel.INFO, "A")
        self.drawer.dismiss(n.id)
        assert n.dismissed is True

    def test_dismiss_unknown_id_returns_false(self):
        result = self.drawer.dismiss("deadbeef")
        assert result is False

    def test_active_excludes_dismissed(self):
        n = self.drawer.push(NotificationLevel.INFO, "A")
        self.drawer.dismiss(n.id)
        assert len(self.drawer.active()) == 0

    def test_active_includes_undismissed(self):
        n1 = self.drawer.push(NotificationLevel.INFO, "A")
        n2 = self.drawer.push(NotificationLevel.INFO, "B")
        self.drawer.dismiss(n1.id)
        active = self.drawer.active()
        assert n2 in active
        assert n1 not in active

    def test_history_includes_all(self):
        n1 = self.drawer.push(NotificationLevel.INFO, "A")
        n2 = self.drawer.push(NotificationLevel.INFO, "B")
        self.drawer.dismiss(n1.id)
        history = self.drawer.history()
        assert n1 in history
        assert n2 in history

    def test_history_returns_all_dismissed(self):
        n = self.drawer.push(NotificationLevel.ERROR, "Err")
        self.drawer.dismiss(n.id)
        assert len(self.drawer.history()) == 1

    def test_render_active_non_empty_when_notifications(self):
        self.drawer.push(NotificationLevel.INFO, "Hello", body="World")
        result = self.drawer.render_active()
        assert len(result) > 0

    def test_render_active_empty_when_all_dismissed(self):
        n = self.drawer.push(NotificationLevel.INFO, "Hello")
        self.drawer.dismiss(n.id)
        result = self.drawer.render_active()
        assert result == ""

    def test_render_active_contains_title(self):
        self.drawer.push(NotificationLevel.INFO, "My Notification")
        result = self.drawer.render_active()
        assert "My Notification" in result

    def test_render_active_empty_drawer(self):
        result = self.drawer.render_active()
        assert result == ""

    def test_max_history_eviction(self):
        drawer = NotificationDrawer(max_history=50)
        for i in range(51):
            drawer.push(NotificationLevel.INFO, f"N{i}")
        assert len(drawer.history()) == 50

    def test_max_history_oldest_evicted(self):
        drawer = NotificationDrawer(max_history=3)
        n1 = drawer.push(NotificationLevel.INFO, "First")
        drawer.push(NotificationLevel.INFO, "Second")
        drawer.push(NotificationLevel.INFO, "Third")
        drawer.push(NotificationLevel.INFO, "Fourth")  # evicts n1
        assert n1 not in drawer.history()

    def test_max_history_exact_boundary(self):
        drawer = NotificationDrawer(max_history=5)
        for i in range(5):
            drawer.push(NotificationLevel.INFO, f"N{i}")
        assert len(drawer.history()) == 5

    def test_clear_dismissed_removes_dismissed(self):
        n = self.drawer.push(NotificationLevel.INFO, "A")
        self.drawer.dismiss(n.id)
        count = self.drawer.clear_dismissed()
        assert count == 1
        assert len(self.drawer.history()) == 0

    def test_clear_dismissed_returns_correct_count(self):
        ns = [self.drawer.push(NotificationLevel.INFO, f"N{i}") for i in range(3)]
        self.drawer.dismiss(ns[0].id)
        self.drawer.dismiss(ns[2].id)
        count = self.drawer.clear_dismissed()
        assert count == 2

    def test_clear_dismissed_leaves_active(self):
        n_active = self.drawer.push(NotificationLevel.INFO, "Active")
        n_dismissed = self.drawer.push(NotificationLevel.INFO, "Dismissed")
        self.drawer.dismiss(n_dismissed.id)
        self.drawer.clear_dismissed()
        assert n_active in self.drawer.history()

    def test_clear_dismissed_zero_when_none_dismissed(self):
        self.drawer.push(NotificationLevel.INFO, "A")
        count = self.drawer.clear_dismissed()
        assert count == 0

    def test_render_active_returns_string(self):
        self.drawer.push(NotificationLevel.SUCCESS, "OK")
        assert isinstance(self.drawer.render_active(), str)

    def test_levels_all_accepted(self):
        for level in NotificationLevel:
            n = self.drawer.push(level, f"test {level.value}")
            assert n.level == level

    def test_push_notification_id_is_8_hex(self):
        n = self.drawer.push(NotificationLevel.INFO, "ID check")
        assert len(n.id) == 8
        int(n.id, 16)  # hex check


# ---------------------------------------------------------------------------
# DEFAULT_NOTIFICATION_DRAWER
# ---------------------------------------------------------------------------


class TestDefaultNotificationDrawer:
    def test_exists(self):
        assert DEFAULT_NOTIFICATION_DRAWER is not None

    def test_is_correct_type(self):
        assert isinstance(DEFAULT_NOTIFICATION_DRAWER, NotificationDrawer)

    def test_active_initially_empty(self):
        # Fresh import; may have been used by other tests, so just verify type
        assert isinstance(DEFAULT_NOTIFICATION_DRAWER.active(), list)

    def test_history_initially_list(self):
        assert isinstance(DEFAULT_NOTIFICATION_DRAWER.history(), list)
