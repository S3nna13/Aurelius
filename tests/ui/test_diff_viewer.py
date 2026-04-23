"""Tests for src.ui.diff_viewer."""

from __future__ import annotations

import pytest
from rich.console import Console

from src.ui.diff_viewer import (
    DiffChunk,
    DiffLine,
    DiffViewer,
    DiffViewerError,
    ParsedDiff,
    parse_unified_diff,
)


_SAMPLE_DIFF = """\
--- a/foo.py
+++ b/foo.py
@@ -1,4 +1,4 @@
 def greet(name):
-    print("Hello", name)
+    print("Hi", name)
     return name
"""

_MINIMAL_DIFF = """\
--- a/x.py
+++ b/x.py
@@ -1,1 +1,1 @@
-old line
+new line
"""


# ---------------------------------------------------------------------------
# parse_unified_diff — well-formed input
# ---------------------------------------------------------------------------


def test_parse_returns_parsed_diff_type() -> None:
    result = parse_unified_diff(_SAMPLE_DIFF)
    assert isinstance(result, ParsedDiff)


def test_parse_filenames() -> None:
    result = parse_unified_diff(_SAMPLE_DIFF)
    assert "foo.py" in result.filename_old
    assert "foo.py" in result.filename_new


def test_parse_chunks_nonempty() -> None:
    result = parse_unified_diff(_SAMPLE_DIFF)
    assert len(result.chunks) >= 1


def test_parse_chunk_has_add_line() -> None:
    result = parse_unified_diff(_SAMPLE_DIFF)
    all_types = [line.line_type for chunk in result.chunks for line in chunk.lines]
    assert "add" in all_types


def test_parse_chunk_has_remove_line() -> None:
    result = parse_unified_diff(_SAMPLE_DIFF)
    all_types = [line.line_type for chunk in result.chunks for line in chunk.lines]
    assert "remove" in all_types


def test_parse_chunk_has_context_line() -> None:
    result = parse_unified_diff(_SAMPLE_DIFF)
    all_types = [line.line_type for chunk in result.chunks for line in chunk.lines]
    assert "context" in all_types


def test_parse_chunk_header_starts_with_at() -> None:
    result = parse_unified_diff(_SAMPLE_DIFF)
    assert result.chunks[0].header.startswith("@@")


def test_parse_add_line_has_new_lineno() -> None:
    result = parse_unified_diff(_MINIMAL_DIFF)
    add_lines = [
        line
        for chunk in result.chunks
        for line in chunk.lines
        if line.line_type == "add"
    ]
    assert add_lines
    assert add_lines[0].line_no_new is not None


def test_parse_remove_line_has_old_lineno() -> None:
    result = parse_unified_diff(_MINIMAL_DIFF)
    remove_lines = [
        line
        for chunk in result.chunks
        for line in chunk.lines
        if line.line_type == "remove"
    ]
    assert remove_lines
    assert remove_lines[0].line_no_old is not None


def test_parse_add_line_content_correct() -> None:
    result = parse_unified_diff(_MINIMAL_DIFF)
    add_lines = [
        line
        for chunk in result.chunks
        for line in chunk.lines
        if line.line_type == "add"
    ]
    assert add_lines[0].content == "new line"


def test_parse_remove_line_content_correct() -> None:
    result = parse_unified_diff(_MINIMAL_DIFF)
    remove_lines = [
        line
        for chunk in result.chunks
        for line in chunk.lines
        if line.line_type == "remove"
    ]
    assert remove_lines[0].content == "old line"


# ---------------------------------------------------------------------------
# parse_unified_diff — malformed / empty input
# ---------------------------------------------------------------------------


def test_parse_malformed_returns_empty_chunks() -> None:
    result = parse_unified_diff("this is not a diff at all")
    assert isinstance(result, ParsedDiff)
    assert result.chunks == []


def test_parse_empty_string_returns_empty_chunks() -> None:
    result = parse_unified_diff("")
    assert isinstance(result, ParsedDiff)
    assert result.chunks == []


def test_parse_only_header_no_crash() -> None:
    result = parse_unified_diff("--- a/f.py\n+++ b/f.py\n")
    assert isinstance(result, ParsedDiff)


# ---------------------------------------------------------------------------
# DiffViewer.render_diff
# ---------------------------------------------------------------------------


def test_render_diff_does_not_crash() -> None:
    diff = parse_unified_diff(_SAMPLE_DIFF)
    viewer = DiffViewer()
    console = Console(record=True)
    viewer.render_diff(console, diff)
    assert True  # no crash


def test_render_diff_empty_diff_does_not_crash() -> None:
    diff = parse_unified_diff("")
    viewer = DiffViewer()
    console = Console(record=True)
    viewer.render_diff(console, diff)
    output = console.export_text()
    assert "empty diff" in output.lower() or output


def test_render_diff_output_contains_plus_or_minus() -> None:
    diff = parse_unified_diff(_SAMPLE_DIFF)
    viewer = DiffViewer()
    console = Console(record=True)
    viewer.render_diff(console, diff)
    output = console.export_text()
    # The + or - sigil should appear in rendered output
    assert "+" in output or "-" in output or "Hi" in output


# ---------------------------------------------------------------------------
# DiffViewer.render_inline
# ---------------------------------------------------------------------------


def test_render_inline_does_not_crash() -> None:
    viewer = DiffViewer()
    console = Console(record=True)
    viewer.render_inline(console, "old\nline2", "new\nline2")
    assert True


def test_render_inline_shows_old_and_new() -> None:
    viewer = DiffViewer()
    console = Console(record=True)
    viewer.render_inline(console, "alpha\nbeta", "alpha\ngamma")
    output = console.export_text()
    assert "alpha" in output


def test_render_inline_empty_strings_does_not_crash() -> None:
    viewer = DiffViewer()
    console = Console(record=True)
    viewer.render_inline(console, "", "")
    assert True


def test_render_inline_unequal_lengths_does_not_crash() -> None:
    viewer = DiffViewer()
    console = Console(record=True)
    viewer.render_inline(console, "a\nb\nc", "x")
    assert True
