"""Tests for src.ui.hotkey_overlay — HotkeyOverlay, HotkeyGroup, HotkeyOverlayError."""

from __future__ import annotations

import pytest
from rich.console import Console

from src.ui.hotkey_overlay import (
    DEFAULT_HOTKEY_OVERLAY,
    HOTKEY_OVERLAY_REGISTRY,
    HotkeyGroup,
    HotkeyOverlay,
    HotkeyOverlayError,
)

# ---------------------------------------------------------------------------
# DEFAULT_HOTKEY_OVERLAY pre-population
# ---------------------------------------------------------------------------


def test_default_overlay_has_three_groups() -> None:
    assert len(DEFAULT_HOTKEY_OVERLAY._groups) == 3


def test_default_overlay_has_navigation_group() -> None:
    assert "Navigation" in DEFAULT_HOTKEY_OVERLAY._groups


def test_default_overlay_has_actions_group() -> None:
    assert "Actions" in DEFAULT_HOTKEY_OVERLAY._groups


def test_default_overlay_has_palette_group() -> None:
    assert "Palette" in DEFAULT_HOTKEY_OVERLAY._groups


def test_default_overlay_navigation_bindings() -> None:
    nav = DEFAULT_HOTKEY_OVERLAY._groups["Navigation"]
    keys = [b[0] for b in nav.bindings]
    assert "↑" in keys
    assert "↓" in keys
    assert "←" in keys
    assert "→" in keys


# ---------------------------------------------------------------------------
# add_group
# ---------------------------------------------------------------------------


def test_add_group_registers_by_name() -> None:
    overlay = HotkeyOverlay()
    group = HotkeyGroup(name="Test", bindings=[("a", "do a")])
    overlay.add_group(group)
    assert "Test" in overlay._groups


def test_add_group_overwrites_existing() -> None:
    overlay = HotkeyOverlay()
    overlay.add_group(HotkeyGroup(name="X", bindings=[("a", "old")]))
    overlay.add_group(HotkeyGroup(name="X", bindings=[("b", "new")]))
    assert overlay._groups["X"].bindings == [("b", "new")]


def test_to_dict_includes_added_group() -> None:
    overlay = HotkeyOverlay()
    overlay.add_group(HotkeyGroup(name="Extra", bindings=[("z", "zoom")]))
    d = overlay.to_dict()
    assert "Extra" in d
    assert d["Extra"]["bindings"] == [("z", "zoom")]


# ---------------------------------------------------------------------------
# remove_group
# ---------------------------------------------------------------------------


def test_remove_group_removes_correctly() -> None:
    overlay = HotkeyOverlay()
    overlay.add_group(HotkeyGroup(name="Temp", bindings=[]))
    overlay.remove_group("Temp")
    assert "Temp" not in overlay._groups


def test_remove_group_unknown_raises() -> None:
    overlay = HotkeyOverlay()
    with pytest.raises(HotkeyOverlayError):
        overlay.remove_group("nonexistent")


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


def test_render_compact_does_not_crash() -> None:
    overlay = HotkeyOverlay()
    overlay.add_group(HotkeyGroup(name="Nav", bindings=[("↑", "up"), ("↓", "down")]))
    console = Console(record=True)
    overlay.render(console, compact=True)
    output = console.export_text()
    assert len(output) > 0


def test_render_full_does_not_crash() -> None:
    overlay = HotkeyOverlay()
    overlay.add_group(HotkeyGroup(name="Nav", bindings=[("↑", "up"), ("↓", "down")]))
    console = Console(record=True)
    overlay.render(console, compact=False)
    output = console.export_text()
    assert len(output) > 0


def test_render_empty_overlay_does_not_crash() -> None:
    overlay = HotkeyOverlay()
    console = Console(record=True)
    overlay.render(console)
    output = console.export_text()
    assert "no hotkeys" in output.lower()


def test_render_compact_default_overlay_does_not_crash() -> None:
    console = Console(record=True)
    DEFAULT_HOTKEY_OVERLAY.render(console, compact=True)
    output = console.export_text()
    assert len(output) > 0


# ---------------------------------------------------------------------------
# HOTKEY_OVERLAY_REGISTRY
# ---------------------------------------------------------------------------


def test_hotkey_overlay_registry_is_dict() -> None:
    assert isinstance(HOTKEY_OVERLAY_REGISTRY, dict)


def test_hotkey_overlay_registry_can_store_overlay() -> None:
    overlay = HotkeyOverlay()
    HOTKEY_OVERLAY_REGISTRY["test-registry"] = overlay
    assert "test-registry" in HOTKEY_OVERLAY_REGISTRY
    del HOTKEY_OVERLAY_REGISTRY["test-registry"]
