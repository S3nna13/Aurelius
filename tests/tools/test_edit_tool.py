"""Tests for src/tools/edit_tool.py — at least 18 tests."""

from __future__ import annotations

import pytest

from src.tools.edit_tool import _MAX_CONTENT_LEN, EditOperation, EditTool


@pytest.fixture
def tool():
    return EditTool()


# ── apply_edit: basic replace ─────────────────────────────────────────────────


def test_apply_edit_basic(tool):
    result = tool.apply_edit("hello world", "world", "Python")
    assert result.success
    assert result.output == "hello Python"


def test_apply_edit_multiline(tool):
    content = "line one\nline two\nline three"
    result = tool.apply_edit(content, "line two", "LINE TWO")
    assert result.success
    assert "LINE TWO" in result.output
    assert "line two" not in result.output


# ── apply_edit: search not found ──────────────────────────────────────────────


def test_apply_edit_not_found(tool):
    result = tool.apply_edit("hello world", "missing", "replacement")
    assert not result.success
    assert "not found" in result.error


# ── apply_edit: search found multiple times ───────────────────────────────────


def test_apply_edit_ambiguous(tool):
    result = tool.apply_edit("foo foo foo", "foo", "bar")
    assert not result.success
    assert "3" in result.error or "ambiguous" in result.error


def test_apply_edit_ambiguous_two(tool):
    result = tool.apply_edit("abc abc", "abc", "xyz")
    assert not result.success
    assert "2" in result.error or "ambiguous" in result.error


# ── apply_edit: empty search ──────────────────────────────────────────────────


def test_apply_edit_empty_search(tool):
    result = tool.apply_edit("some content", "", "replacement")
    assert not result.success
    assert "empty" in result.error


# ── apply_edit: content over size limit ───────────────────────────────────────


def test_apply_edit_content_too_large(tool):
    big = "x" * (_MAX_CONTENT_LEN + 1)
    result = tool.apply_edit(big, "x", "y")
    assert not result.success
    assert "exceeds" in result.error


# ── apply_edit: search and replace are same string ───────────────────────────


def test_apply_edit_same_search_replace(tool):
    result = tool.apply_edit("keep this text", "keep", "keep")
    assert result.success
    assert result.output == "keep this text"


# ── apply_edit: replace with empty string (deletion) ─────────────────────────


def test_apply_edit_delete(tool):
    result = tool.apply_edit("remove this part here", "remove this part ", "")
    assert result.success
    assert result.output == "here"


# ── apply_edit: only one occurrence replaced (not all) ───────────────────────


def test_apply_edit_only_once(tool):
    # "unique_token" appears exactly once — make sure only that occurrence goes
    result = tool.apply_edit("start unique_token end", "unique_token", "REPLACED")
    assert result.success
    assert result.output == "start REPLACED end"


# ── apply_edits: sequence of edits applied in order ───────────────────────────


def test_apply_edits_sequence(tool):
    ops = [
        EditOperation(search="foo", replace="bar"),
        EditOperation(search="bar", replace="baz"),
    ]
    result = tool.apply_edits("foo is here", ops)
    assert result.success
    assert result.output == "baz is here"


def test_apply_edits_multiple_distinct(tool):
    ops = [
        EditOperation(search="alpha", replace="ALPHA"),
        EditOperation(search="beta", replace="BETA"),
    ]
    result = tool.apply_edits("alpha and beta", ops)
    assert result.success
    assert result.output == "ALPHA and BETA"


# ── apply_edits: stops at first failure ───────────────────────────────────────


def test_apply_edits_stops_on_failure(tool):
    ops = [
        EditOperation(search="exists", replace="found"),
        EditOperation(search="missing", replace="nope"),  # should fail
        EditOperation(search="found", replace="final"),  # should never run
    ]
    result = tool.apply_edits("exists here", ops)
    assert not result.success
    assert "not found" in result.error


# ── unified_diff ──────────────────────────────────────────────────────────────


def test_unified_diff_changed(tool):
    orig = "line1\nline2\nline3\n"
    mod = "line1\nLINE2\nline3\n"
    diff = tool.unified_diff(orig, mod)
    assert "-line2" in diff
    assert "+LINE2" in diff


def test_unified_diff_identical(tool):
    content = "no changes here\n"
    diff = tool.unified_diff(content, content)
    assert diff == ""


def test_unified_diff_fromfile_tofile(tool):
    diff = tool.unified_diff("old\n", "new\n", fromfile="a.py", tofile="b.py")
    assert "a.py" in diff
    assert "b.py" in diff


# ── Adversarial: binary-ish content (null bytes in search) ───────────────────


def test_apply_edit_null_bytes(tool):
    content = "before\x00TARGET\x00after"
    result = tool.apply_edit(content, "\x00TARGET\x00", "_REPLACED_")
    assert result.success
    assert "_REPLACED_" in result.output


# ── Adversarial: very large search string (within limit) ─────────────────────


def test_apply_edit_large_search_within_limit(tool):
    search = "A" * 10_000
    content = "prefix " + search + " suffix"
    result = tool.apply_edit(content, search, "BIG_REPLACE")
    assert result.success
    assert result.output == "prefix BIG_REPLACE suffix"


# ── apply_edits: empty ops list ───────────────────────────────────────────────


def test_apply_edits_empty_ops(tool):
    result = tool.apply_edits("unchanged content", [])
    assert result.success
    assert result.output == "unchanged content"
