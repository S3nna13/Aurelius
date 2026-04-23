"""Unit tests for :mod:`src.ui.ui_surface`."""

from __future__ import annotations

import pytest

from src.ui.errors import UIError
from src.ui.ui_surface import (
    UI_SURFACE_REGISTRY,
    UISurface,
    get_ui_surface,
    list_ui_surfaces,
    register_ui_surface,
)


def _good_surface(surface_id: str = "test-surface") -> UISurface:
    return UISurface(
        surface_id=surface_id,
        title="Test Surface",
        panels=("header", "body"),
        default_layout="stoic-focus",
        motion="welcome-fade",
        keyboard_map={
            "enter": "continue",
            "esc": "quit",
            "?": "help",
            "ctrl-k": "palette_open",
        },
    )


def test_good_construction_and_fields():
    surf = _good_surface()
    assert surf.surface_id == "test-surface"
    assert surf.title == "Test Surface"
    assert surf.panels == ("header", "body")
    assert surf.default_layout == "stoic-focus"
    assert surf.motion == "welcome-fade"
    assert surf.keyboard_map["enter"] == "continue"
    assert surf.keyboard_map["ctrl-k"] == "palette_open"


def test_bad_surface_id_charset_rejected():
    with pytest.raises(UIError, match=r"surface_id"):
        UISurface(
            surface_id="Has Space!",
            title="t",
            panels=("a",),
            default_layout="x",
        )


def test_empty_title_rejected():
    with pytest.raises(UIError, match=r"title"):
        UISurface(
            surface_id="empty-title",
            title="",
            panels=("a",),
            default_layout="x",
        )


def test_empty_panels_rejected():
    with pytest.raises(UIError, match=r"panels"):
        UISurface(
            surface_id="empty-panels",
            title="t",
            panels=(),
            default_layout="x",
        )


def test_empty_default_layout_rejected():
    with pytest.raises(UIError, match=r"default_layout"):
        UISurface(
            surface_id="empty-layout",
            title="t",
            panels=("a",),
            default_layout="",
        )


def test_bad_keyboard_map_key_rejected():
    with pytest.raises(UIError, match=r"keyboard_map key"):
        UISurface(
            surface_id="bad-key",
            title="t",
            panels=("a",),
            default_layout="x",
            keyboard_map={"supercombo-xyz": "do_it"},
        )


def test_bad_keyboard_map_action_rejected():
    with pytest.raises(UIError, match=r"keyboard_map"):
        UISurface(
            surface_id="bad-action",
            title="t",
            panels=("a",),
            default_layout="x",
            keyboard_map={"enter": "NotSnake"},
        )


def test_empty_keyboard_map_action_rejected():
    with pytest.raises(UIError, match=r"keyboard_map"):
        UISurface(
            surface_id="empty-action",
            title="t",
            panels=("a",),
            default_layout="x",
            keyboard_map={"enter": ""},
        )


def test_welcome_surface_is_registered_at_import():
    assert "welcome" in UI_SURFACE_REGISTRY
    w = UI_SURFACE_REGISTRY["welcome"]
    assert w.default_layout == "stoic-focus"
    assert w.motion == "welcome-fade"
    assert w.keyboard_map.get("enter") == "continue"


def test_register_get_list_roundtrip_and_duplicate_rejected():
    surf = _good_surface("roundtrip-surface-1")
    assert "roundtrip-surface-1" not in UI_SURFACE_REGISTRY
    register_ui_surface(surf)
    try:
        assert get_ui_surface("roundtrip-surface-1") is surf
        assert "roundtrip-surface-1" in list_ui_surfaces()
        with pytest.raises(UIError, match=r"already registered"):
            register_ui_surface(surf)
    finally:
        UI_SURFACE_REGISTRY.pop("roundtrip-surface-1", None)


def test_get_unknown_surface_raises():
    with pytest.raises(UIError, match=r"no UI surface"):
        get_ui_surface("does-not-exist-xyz")


def test_register_non_surface_rejected():
    with pytest.raises(UIError):
        register_ui_surface("not a surface")  # type: ignore[arg-type]


def test_bad_panel_entries_rejected():
    with pytest.raises(UIError, match=r"panels"):
        UISurface(
            surface_id="bad-panels",
            title="t",
            panels=("a", ""),
            default_layout="x",
        )
