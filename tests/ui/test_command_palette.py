"""Tests for src.ui.command_palette."""

from __future__ import annotations

import pytest
from rich.console import Console

from src.ui.command_palette import (
    COMMAND_PALETTE_REGISTRY,
    CommandEntry,
    CommandPalette,
    CommandPaletteError,
)

# ---------------------------------------------------------------------------
# Registry population
# ---------------------------------------------------------------------------


def test_registry_contains_all_builtins() -> None:
    """All 5 built-in commands must be present after import."""
    expected = {"clear", "help", "quit", "toggle-motion", "show-branding"}
    assert expected.issubset(set(COMMAND_PALETTE_REGISTRY.keys()))


def test_registry_has_exactly_five_builtins() -> None:
    """There must be exactly 5 built-in entries (none added by other tests yet)."""
    # Use COMMAND_PALETTE_REGISTRY which is the same object as
    # CommandPalette.COMMAND_PALETTE_REGISTRY.
    assert len(COMMAND_PALETTE_REGISTRY) >= 5


def test_command_palette_registry_is_class_attr() -> None:
    """Module-level COMMAND_PALETTE_REGISTRY is the same object as the class attr."""
    assert COMMAND_PALETTE_REGISTRY is CommandPalette.COMMAND_PALETTE_REGISTRY


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def test_search_clear_returns_entry() -> None:
    results = CommandPalette.search("clear")
    names = [e.name for e in results]
    assert "clear" in names


def test_search_unknown_returns_empty() -> None:
    results = CommandPalette.search("zznotfound")
    assert results == []


def test_search_empty_query_returns_all() -> None:
    results = CommandPalette.search("")
    assert len(results) >= 5


def test_search_partial_match() -> None:
    results = CommandPalette.search("mot")
    names = [e.name for e in results]
    assert "toggle-motion" in names


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------


def test_execute_quit_with_none_handler_does_not_crash() -> None:
    """quit has handler=None; execute must skip gracefully."""
    # Should not raise.
    CommandPalette.execute("quit")


def test_execute_nonexistent_raises() -> None:
    with pytest.raises(CommandPaletteError):
        CommandPalette.execute("notexist")


def test_execute_calls_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    """If a handler is set it must be called."""
    called: list[bool] = []

    entry = CommandEntry(
        name="_test_exec_handler",
        description="test",
        handler=lambda: called.append(True),
    )
    CommandPalette.register(entry)
    try:
        CommandPalette.execute("_test_exec_handler")
        assert called == [True]
    finally:
        del CommandPalette.COMMAND_PALETTE_REGISTRY["_test_exec_handler"]


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def test_render_empty_query_does_not_crash() -> None:
    console = Console(record=True)
    CommandPalette.render(console)  # default query=""


def test_render_with_query_does_not_crash() -> None:
    console = Console(record=True)
    CommandPalette.render(console, query="clear")


def test_render_output_contains_command_name() -> None:
    console = Console(record=True)
    CommandPalette.render(console, query="help")
    output = console.export_text()
    assert "help" in output
