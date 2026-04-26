"""Tests for input sanitizer (Trail of Bits style)."""

from __future__ import annotations

import pytest

from src.security.input_sanitizer import InputSanitizer


class TestInputSanitizer:
    def test_passes_clean_text(self):
        s = InputSanitizer()
        assert s.sanitize("hello world") == "hello world"

    def test_rejects_html_tags(self):
        s = InputSanitizer()
        with pytest.raises(ValueError, match="HTML tags rejected"):
            s.sanitize("<script>alert('xss')</script>")

    def test_rejects_sql_keywords(self):
        s = InputSanitizer()
        with pytest.raises(ValueError, match="SQL keyword rejected"):
            s.sanitize("DROP TABLE users")

    def test_strips_control_chars(self):
        s = InputSanitizer()
        result = s.sanitize("hello\x00world\x1btest")
        assert "\x00" not in result
        assert "\x1b" not in result

    def test_max_length(self):
        s = InputSanitizer(max_input_length=10)
        with pytest.raises(ValueError, match="exceeds"):
            s.sanitize("x" * 20)

    def test_sanitizer_rules(self):
        from src.security.input_sanitizer import SanitizerRule

        rule = SanitizerRule(field="name", pattern=r"^[a-zA-Z]+$", max_length=10)
        assert rule.validate("Hello") is None
        assert rule.validate("Hello123") is not None
        assert rule.validate("x" * 20) is not None
