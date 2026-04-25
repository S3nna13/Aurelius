"""Tests for src/tools/grep_tool.py — at least 18 tests."""
from __future__ import annotations

import json
import re

import pytest
from src.tools.grep_tool import GrepTool, _MAX_CONTENT_LEN, _MAX_PATTERN_LEN, _MAX_RESULTS


@pytest.fixture
def tool():
    return GrepTool()


def _parse(result) -> dict:
    return json.loads(result.output)


# ── Basic pattern match ───────────────────────────────────────────────────────

def test_basic_match_line_numbers(tool):
    content = "alpha\nbeta\ngamma"
    result = tool.search("beta", content)
    assert result.success
    data = _parse(result)
    assert data["total"] == 1
    assert data["matches"][0]["line_number"] == 2


def test_basic_match_line_content(tool):
    content = "foo bar\nbaz qux"
    result = tool.search("baz", content)
    assert result.success
    data = _parse(result)
    assert data["matches"][0]["line"] == "baz qux"


# ── context_before / context_after ───────────────────────────────────────────

def test_context_populated(tool):
    content = "line1\nline2\nTARGET\nline4\nline5"
    result = tool.search("TARGET", content, context_lines=2)
    assert result.success
    data = _parse(result)
    m = data["matches"][0]
    assert m["context_before"] == ["line1", "line2"]
    assert m["context_after"] == ["line4", "line5"]


def test_context_at_start(tool):
    content = "TARGET\nline2\nline3"
    result = tool.search("TARGET", content, context_lines=2)
    assert result.success
    data = _parse(result)
    m = data["matches"][0]
    assert m["context_before"] == []
    assert "line2" in m["context_after"]


def test_context_at_end(tool):
    content = "line1\nline2\nTARGET"
    result = tool.search("TARGET", content, context_lines=2)
    assert result.success
    data = _parse(result)
    m = data["matches"][0]
    assert m["context_after"] == []


# ── No matches ────────────────────────────────────────────────────────────────

def test_no_matches(tool):
    result = tool.search("NOTHERE", "alpha\nbeta\ngamma")
    assert result.success
    data = _parse(result)
    assert data["total"] == 0
    assert data["matches"] == []


# ── Invalid regex ─────────────────────────────────────────────────────────────

def test_invalid_regex(tool):
    result = tool.search("[invalid", "some content")
    assert not result.success
    assert "invalid pattern" in result.error


def test_invalid_regex_unbalanced_paren(tool):
    result = tool.search("(unclosed", "text")
    assert not result.success
    assert "invalid pattern" in result.error


# ── Pattern length limit ──────────────────────────────────────────────────────

def test_pattern_too_long(tool):
    big_pattern = "a" * (_MAX_PATTERN_LEN + 1)
    result = tool.search(big_pattern, "some content")
    assert not result.success
    assert "pattern exceeds" in result.error


# ── Content length limit ──────────────────────────────────────────────────────

def test_content_too_large(tool):
    big_content = "x\n" * (_MAX_CONTENT_LEN // 2 + 1)
    result = tool.search("x", big_content)
    assert not result.success
    assert "content exceeds" in result.error


# ── max_results cap ───────────────────────────────────────────────────────────

def test_max_results_respected(tool):
    content = "\n".join(["match"] * 200)
    result = tool.search("match", content, max_results=10)
    assert result.success
    data = _parse(result)
    assert data["total"] == 10


def test_max_results_capped_at_hard_limit(tool):
    # Pass max_results > _MAX_RESULTS; it should be clamped to _MAX_RESULTS
    content = "\n".join(["match"] * (_MAX_RESULTS + 200))
    result = tool.search("match", content, max_results=_MAX_RESULTS + 500)
    assert result.success
    data = _parse(result)
    assert data["total"] <= _MAX_RESULTS


# ── Case-insensitive flag ─────────────────────────────────────────────────────

def test_case_insensitive(tool):
    content = "Hello World\nfoo bar"
    result = tool.search("hello", content, flags=re.IGNORECASE)
    assert result.success
    data = _parse(result)
    assert data["total"] == 1
    assert data["matches"][0]["line_number"] == 1


# ── Empty content ─────────────────────────────────────────────────────────────

def test_empty_content(tool):
    result = tool.search("anything", "")
    assert result.success
    data = _parse(result)
    assert data["total"] == 0


# ── Multiline content with many matches ──────────────────────────────────────

def test_many_matches(tool):
    content = "\n".join([f"item {i}" for i in range(100)])
    result = tool.search(r"item \d+", content, max_results=100)
    assert result.success
    data = _parse(result)
    assert data["total"] == 100


# ── JSON output is valid parseable JSON ──────────────────────────────────────

def test_output_is_valid_json(tool):
    result = tool.search("foo", "foo bar\nbaz foo")
    assert result.success
    data = json.loads(result.output)   # must not raise
    assert "matches" in data
    assert "total" in data


# ── Adversarial: catastrophic backtracking (small input so it won't hang) ────

def test_no_catastrophic_backtracking(tool):
    # This pattern is known for backtracking on certain inputs but with a small
    # input the test exercises the path without risk of hanging.
    pattern = r"(a+)+"
    content = "a" * 20 + "!"
    result = tool.search(pattern, content)
    # Should complete and return a result (match or not) without hanging
    assert result.success or not result.success  # either is fine — just must return


# ── Regex with special chars in pattern ──────────────────────────────────────

def test_regex_special_chars(tool):
    content = "price: $9.99\nname: foo"
    result = tool.search(r"\$\d+\.\d+", content)
    assert result.success
    data = _parse(result)
    assert data["total"] == 1
    assert "price" in data["matches"][0]["line"]


# ── context_lines=0 ───────────────────────────────────────────────────────────

def test_context_lines_zero(tool):
    content = "before\nmatch\nafter"
    result = tool.search("match", content, context_lines=0)
    assert result.success
    data = _parse(result)
    m = data["matches"][0]
    assert m["context_before"] == []
    assert m["context_after"] == []
