"""Tests for src/ui/accessibility_announcer.py."""

from __future__ import annotations

import pytest

from src.ui.accessibility_announcer import (
    AccessibilityAnnouncer,
    AccessibilityAnnouncerError,
    AriaLiveRegion,
    DEFAULT_ACCESSIBILITY_ANNOUNCER,
)


# ---------------------------------------------------------------------------
# AccessibilityAnnouncer — announce / queue
# ---------------------------------------------------------------------------


class TestAccessibilityAnnouncerQueue:
    def setup_method(self):
        self.announcer = AccessibilityAnnouncer()

    def test_announce_returns_formatted_string(self):
        result = self.announcer.announce("hello")
        assert isinstance(result, str)

    def test_announce_adds_to_pending(self):
        self.announcer.announce("first")
        assert len(self.announcer.pending()) == 1

    def test_announce_multiple(self):
        for i in range(5):
            self.announcer.announce(f"msg {i}")
        assert len(self.announcer.pending()) == 5

    def test_pending_returns_list_of_strings(self):
        self.announcer.announce("a")
        pending = self.announcer.pending()
        assert isinstance(pending, list)
        assert all(isinstance(p, str) for p in pending)

    def test_pending_is_shallow_copy(self):
        self.announcer.announce("x")
        p1 = self.announcer.pending()
        p1.pop()
        assert len(self.announcer.pending()) == 1

    def test_clear_queue_removes_all(self):
        self.announcer.announce("a")
        self.announcer.announce("b")
        self.announcer.clear_queue()
        assert self.announcer.pending() == []

    def test_clear_queue_empty_is_noop(self):
        self.announcer.clear_queue()
        assert self.announcer.pending() == []

    def test_announce_preserves_order(self):
        self.announcer.announce("alpha")
        self.announcer.announce("beta")
        assert self.announcer.pending()[0].endswith("alpha")
        assert self.announcer.pending()[1].endswith("beta")


# ---------------------------------------------------------------------------
# Priority formatting
# ---------------------------------------------------------------------------


class TestPriorityFormatting:
    def setup_method(self):
        self.announcer = AccessibilityAnnouncer()

    def test_critical_priority_prefix(self):
        result = self.announcer.announce("system failure", priority="critical")
        assert result.startswith("[CRITICAL] ")

    def test_important_priority_prefix(self):
        result = self.announcer.announce("warning", priority="important")
        assert result.startswith("[IMPORTANT] ")

    def test_normal_priority_prefix(self):
        result = self.announcer.announce("info", priority="normal")
        assert result.startswith("[NORMAL] ")

    def test_low_priority_prefix(self):
        result = self.announcer.announce("debug", priority="low")
        assert result.startswith("[LOW] ")

    def test_default_priority_is_normal(self):
        result = self.announcer.announce("default")
        assert result.startswith("[NORMAL] ")

    def test_invalid_priority_raises(self):
        with pytest.raises(AccessibilityAnnouncerError):
            self.announcer.announce("x", priority="bogus")

    def test_format_for_screen_reader_invalid_priority_raises(self):
        with pytest.raises(AccessibilityAnnouncerError):
            AccessibilityAnnouncer.format_for_screen_reader("x", priority="bogus")

    def test_all_valid_priorities_accepted(self):
        for p in ("critical", "important", "normal", "low"):
            result = self.announcer.announce("test", priority=p)
            assert result.startswith(f"[{p.upper()}] ")


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


class TestTruncation:
    def setup_method(self):
        self.announcer = AccessibilityAnnouncer()

    def test_short_message_not_truncated(self):
        msg = "short"
        result = self.announcer.announce(msg)
        assert msg in result
        assert "..." not in result

    def test_exactly_256_not_truncated(self):
        msg = "x" * 256
        result = self.announcer.announce(msg)
        # prefix is present, so total length > 256, but original msg intact
        assert msg in result
        assert "..." not in result

    def test_over_256_truncated_with_ellipsis(self):
        msg = "x" * 300
        result = self.announcer.announce(msg)
        assert result.endswith("...")
        assert len(result.replace("[NORMAL] ", "")) == 256

    def test_truncation_respects_prefix(self):
        msg = "y" * 400
        result = self.announcer.announce(msg, priority="critical")
        assert result.startswith("[CRITICAL] ")
        assert result.endswith("...")
        content = result.replace("[CRITICAL] ", "")
        assert len(content) == 256


# ---------------------------------------------------------------------------
# Control character sanitization
# ---------------------------------------------------------------------------


class TestSanitization:
    def setup_method(self):
        self.announcer = AccessibilityAnnouncer()

    def test_newline_preserved(self):
        result = self.announcer.announce("line1\nline2")
        assert "\n" in result

    def test_tab_preserved(self):
        result = self.announcer.announce("col1\tcol2")
        assert "\t" in result

    def test_bell_stripped(self):
        result = self.announcer.announce("hello\x07world")
        assert "\x07" not in result
        assert "helloworld" in result

    def test_escape_stripped(self):
        result = self.announcer.announce("foo\x1b[31mbar")
        assert "\x1b" not in result

    def test_null_byte_stripped(self):
        result = self.announcer.announce("a\x00b")
        assert "\x00" not in result
        assert "ab" in result

    def test_carriage_return_stripped(self):
        result = self.announcer.announce("a\r\nb")
        assert "\r" not in result
        assert "a\nb" in result

    def test_zero_width_space_stripped(self):
        result = self.announcer.announce("a\u200Bb")
        assert "\u200B" not in result


# ---------------------------------------------------------------------------
# AriaLiveRegion
# ---------------------------------------------------------------------------


class TestAriaLiveRegion:
    def test_polite_mode_accepted(self):
        region = AriaLiveRegion(region_id="status", mode="polite", label="Status")
        assert region.mode == "polite"

    def test_assertive_mode_accepted(self):
        region = AriaLiveRegion(region_id="alert", mode="assertive", label="Alerts")
        assert region.mode == "assertive"

    def test_invalid_mode_raises(self):
        with pytest.raises(AccessibilityAnnouncerError):
            AriaLiveRegion(region_id="x", mode="off")

    def test_empty_region_id_raises(self):
        with pytest.raises(AccessibilityAnnouncerError):
            AriaLiveRegion(region_id="", mode="polite")

    def test_whitespace_only_region_id_raises(self):
        with pytest.raises(AccessibilityAnnouncerError):
            AriaLiveRegion(region_id="   ", mode="polite")

    def test_non_string_label_raises(self):
        with pytest.raises(AccessibilityAnnouncerError):
            AriaLiveRegion(region_id="x", mode="polite", label=123)  # type: ignore[arg-type]

    def test_default_label_empty(self):
        region = AriaLiveRegion(region_id="x", mode="polite")
        assert region.label == ""

    def test_region_id_stored(self):
        region = AriaLiveRegion(region_id="my-region", mode="assertive")
        assert region.region_id == "my-region"


# ---------------------------------------------------------------------------
# Braille support
# ---------------------------------------------------------------------------


class TestBraille:
    def test_lowercase_letter(self):
        assert AccessibilityAnnouncer.to_braille("a") == "⠁"

    def test_uppercase_letter(self):
        assert AccessibilityAnnouncer.to_braille("A") == "⠠⠁"

    def test_digit(self):
        assert AccessibilityAnnouncer.to_braille("5") == "⠢"

    def test_space_preserved(self):
        assert AccessibilityAnnouncer.to_braille("a b") == "⠁ ⠃"

    def test_unknown_char_unchanged(self):
        assert AccessibilityAnnouncer.to_braille("@") == "@"

    def test_mixed_text(self):
        result = AccessibilityAnnouncer.to_braille("Hi 2")
        assert "⠠⠓" in result
        assert "⠆" in result


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_announce_non_string_raises(self):
        with pytest.raises(AccessibilityAnnouncerError):
            AccessibilityAnnouncer().announce(123)  # type: ignore[arg-type]

    def test_error_is_value_error_subclass(self):
        assert issubclass(AccessibilityAnnouncerError, ValueError)


# ---------------------------------------------------------------------------
# DEFAULT_ACCESSIBILITY_ANNOUNCER singleton
# ---------------------------------------------------------------------------


class TestDefaultAccessibilityAnnouncer:
    def test_singleton_exists(self):
        assert DEFAULT_ACCESSIBILITY_ANNOUNCER is not None

    def test_singleton_is_correct_type(self):
        assert isinstance(DEFAULT_ACCESSIBILITY_ANNOUNCER, AccessibilityAnnouncer)

    def test_singleton_pending_is_list(self):
        assert isinstance(DEFAULT_ACCESSIBILITY_ANNOUNCER.pending(), list)
