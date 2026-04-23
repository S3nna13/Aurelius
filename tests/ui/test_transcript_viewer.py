"""Tests for src.ui.transcript_viewer."""

from __future__ import annotations

import time

import pytest
from rich.console import Console

from src.ui.transcript_viewer import (
    TRANSCRIPT_VIEWER_REGISTRY,
    TranscriptEntry,
    TranscriptRole,
    TranscriptViewer,
    TranscriptViewerError,
)


def _make_entry(
    role: TranscriptRole = TranscriptRole.USER,
    content: str = "hello",
) -> TranscriptEntry:
    return TranscriptEntry(role=role, content=content)


# ---------------------------------------------------------------------------
# add_entry
# ---------------------------------------------------------------------------


def test_add_entry_grows_len() -> None:
    viewer = TranscriptViewer()
    assert len(viewer.entries) == 0
    viewer.add_entry(_make_entry())
    assert len(viewer.entries) == 1
    viewer.add_entry(_make_entry(TranscriptRole.ASSISTANT, "hi"))
    assert len(viewer.entries) == 2


def test_add_entry_preserves_order() -> None:
    viewer = TranscriptViewer()
    e1 = _make_entry(TranscriptRole.USER, "first")
    e2 = _make_entry(TranscriptRole.ASSISTANT, "second")
    viewer.add_entry(e1)
    viewer.add_entry(e2)
    assert viewer.entries[0].content == "first"
    assert viewer.entries[1].content == "second"


def test_add_entry_invalid_raises() -> None:
    viewer = TranscriptViewer()
    with pytest.raises(TranscriptViewerError):
        viewer.add_entry("not an entry")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


def test_render_empty_does_not_crash() -> None:
    viewer = TranscriptViewer()
    console = Console(record=True)
    viewer.render(console)
    output = console.export_text()
    assert "no entries" in output.lower() or "Aurelius Transcript" in output


def test_render_three_roles_does_not_crash() -> None:
    viewer = TranscriptViewer()
    viewer.add_entry(_make_entry(TranscriptRole.USER, "user message"))
    viewer.add_entry(_make_entry(TranscriptRole.ASSISTANT, "assistant reply"))
    viewer.add_entry(_make_entry(TranscriptRole.SYSTEM, "sys note"))
    console = Console(record=True)
    viewer.render(console)
    output = console.export_text()
    assert "user message" in output
    assert "assistant reply" in output
    assert "sys note" in output


def test_render_tool_roles_does_not_crash() -> None:
    viewer = TranscriptViewer()
    viewer.add_entry(_make_entry(TranscriptRole.TOOL_CALL, "call_tool()"))
    viewer.add_entry(_make_entry(TranscriptRole.TOOL_RESULT, "result_data"))
    console = Console(record=True)
    viewer.render(console)
    assert True  # no crash


def test_render_with_timestamps() -> None:
    viewer = TranscriptViewer()
    viewer.add_entry(_make_entry(TranscriptRole.USER, "time check"))
    console = Console(record=True)
    viewer.render(console, show_timestamps=True)
    output = console.export_text()
    assert "time check" in output


def test_render_max_entries_limits_output() -> None:
    viewer = TranscriptViewer()
    for i in range(10):
        viewer.add_entry(_make_entry(TranscriptRole.USER, f"msg-{i}"))
    console = Console(record=True)
    viewer.render(console, max_entries=3)
    output = console.export_text()
    # only last 3 entries should appear
    assert "msg-9" in output
    assert "msg-0" not in output


# ---------------------------------------------------------------------------
# export_text
# ---------------------------------------------------------------------------


def test_export_text_two_entries_nonempty() -> None:
    viewer = TranscriptViewer()
    viewer.add_entry(_make_entry(TranscriptRole.USER, "hello"))
    viewer.add_entry(_make_entry(TranscriptRole.ASSISTANT, "world"))
    text = viewer.export_text()
    assert isinstance(text, str)
    assert len(text) > 0


def test_export_text_contains_role_labels() -> None:
    viewer = TranscriptViewer()
    viewer.add_entry(_make_entry(TranscriptRole.USER, "question"))
    viewer.add_entry(_make_entry(TranscriptRole.ASSISTANT, "answer"))
    text = viewer.export_text()
    assert "USER:" in text
    assert "ASSISTANT:" in text


def test_export_text_empty_viewer_returns_empty_string() -> None:
    viewer = TranscriptViewer()
    assert viewer.export_text() == ""


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def test_search_finds_matching_entries() -> None:
    viewer = TranscriptViewer()
    viewer.add_entry(_make_entry(TranscriptRole.USER, "find me please"))
    viewer.add_entry(_make_entry(TranscriptRole.ASSISTANT, "nothing special"))
    results = viewer.search("find")
    assert len(results) == 1
    assert results[0].content == "find me please"


def test_search_case_insensitive() -> None:
    viewer = TranscriptViewer()
    viewer.add_entry(_make_entry(TranscriptRole.USER, "Hello World"))
    results = viewer.search("hello")
    assert len(results) == 1


def test_search_no_match_returns_empty_list() -> None:
    viewer = TranscriptViewer()
    viewer.add_entry(_make_entry(TranscriptRole.USER, "nothing here"))
    results = viewer.search("xyz_not_found")
    assert results == []


def test_search_multiple_matches() -> None:
    viewer = TranscriptViewer()
    viewer.add_entry(_make_entry(TranscriptRole.USER, "alpha beta"))
    viewer.add_entry(_make_entry(TranscriptRole.ASSISTANT, "alpha gamma"))
    viewer.add_entry(_make_entry(TranscriptRole.SYSTEM, "delta only"))
    results = viewer.search("alpha")
    assert len(results) == 2


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


def test_clear_resets_entries() -> None:
    viewer = TranscriptViewer()
    viewer.add_entry(_make_entry())
    viewer.add_entry(_make_entry())
    viewer.clear()
    assert len(viewer.entries) == 0


def test_clear_then_add_works() -> None:
    viewer = TranscriptViewer()
    viewer.add_entry(_make_entry(content="before"))
    viewer.clear()
    viewer.add_entry(_make_entry(content="after"))
    assert len(viewer.entries) == 1
    assert viewer.entries[0].content == "after"


# ---------------------------------------------------------------------------
# TRANSCRIPT_VIEWER_REGISTRY
# ---------------------------------------------------------------------------


def test_transcript_viewer_registry_is_dict() -> None:
    assert isinstance(TRANSCRIPT_VIEWER_REGISTRY, dict)


def test_transcript_viewer_registry_supports_assignment() -> None:
    TRANSCRIPT_VIEWER_REGISTRY["test-session"] = TranscriptViewer()
    assert "test-session" in TRANSCRIPT_VIEWER_REGISTRY
    del TRANSCRIPT_VIEWER_REGISTRY["test-session"]
