"""Tests for string diff."""

from __future__ import annotations

from src.tools.string_diff import StringDiff


class TestStringDiff:
    def test_diff_detects_additions(self):
        sd = StringDiff()
        result = sd.diff("a\nb", "a\nb\nc")
        assert "c" in result.added
        assert result.changed is True

    def test_diff_detects_removals(self):
        sd = StringDiff()
        result = sd.diff("a\nb\nc", "a\nb")
        assert "c" in result.removed

    def test_identical_strings(self):
        sd = StringDiff()
        result = sd.diff("a\nb", "a\nb")
        assert result.changed is False

    def test_similarity_perfect(self):
        sd = StringDiff()
        assert sd.similarity("hello world", "hello world") == 1.0

    def test_similarity_none(self):
        sd = StringDiff()
        assert sd.similarity("abc", "xyz") == 0.0
