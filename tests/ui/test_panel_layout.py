"""Unit tests for :mod:`src.ui.panel_layout`."""

from __future__ import annotations

import pytest

from src.ui.errors import UIError
from src.ui.panel_layout import (
    PANEL_LAYOUT_REGISTRY,
    PanelLayout,
    compose_layout,
    get_panel_layout,
    list_panel_layouts,
    register_panel_layout,
)


def _good_layout(layout_id: str = "test-layout") -> PanelLayout:
    return PanelLayout(
        layout_id=layout_id,
        regions=("header", "body", "footer"),
        min_cols=40,
        min_rows=12,
        focus_order=("body", "header", "footer"),
    )


def test_good_construction_roundtrip_fields():
    lay = _good_layout()
    assert lay.layout_id == "test-layout"
    assert lay.regions == ("header", "body", "footer")
    assert lay.min_cols == 40
    assert lay.min_rows == 12
    assert lay.focus_order == ("body", "header", "footer")


def test_bad_layout_id_charset_rejected():
    with pytest.raises(UIError, match=r"layout_id"):
        PanelLayout(
            layout_id="Bad Id!",
            regions=("a", "b"),
            min_cols=40,
            min_rows=12,
            focus_order=("a", "b"),
        )


def test_empty_layout_id_rejected():
    with pytest.raises(UIError):
        PanelLayout(
            layout_id="",
            regions=("a",),
            min_cols=40,
            min_rows=12,
            focus_order=("a",),
        )


def test_non_unique_regions_rejected():
    with pytest.raises(UIError, match=r"unique"):
        PanelLayout(
            layout_id="dup-regions",
            regions=("header", "header"),
            min_cols=40,
            min_rows=12,
            focus_order=("header", "header"),
        )


def test_focus_order_not_a_permutation_rejected():
    with pytest.raises(UIError, match=r"permutation"):
        PanelLayout(
            layout_id="bad-focus",
            regions=("header", "body", "footer"),
            min_cols=40,
            min_rows=12,
            focus_order=("header", "body"),
        )


def test_min_cols_below_threshold_rejected():
    with pytest.raises(UIError, match=r"min_cols"):
        PanelLayout(
            layout_id="tiny-cols",
            regions=("a",),
            min_cols=10,
            min_rows=12,
            focus_order=("a",),
        )


def test_min_rows_below_threshold_rejected():
    with pytest.raises(UIError, match=r"min_rows"):
        PanelLayout(
            layout_id="tiny-rows",
            regions=("a",),
            min_cols=40,
            min_rows=2,
            focus_order=("a",),
        )


def test_compose_layout_success_includes_region_names():
    lay = _good_layout("compose-ok")
    out = compose_layout(
        lay,
        60,
        20,
        panels={
            "header": "HEADER-CONTENT",
            "body": "body-content-line",
            "footer": "footer-content",
        },
    )
    assert isinstance(out, str) and out
    lines = out.split("\n")
    assert len(lines) == 20
    for ln in lines:
        assert len(ln) == 60
    for region in lay.regions:
        assert region in out
    assert "HEADER-CONTENT" in out
    assert "body-content-line" in out
    assert "footer-content" in out


def test_compose_layout_too_small_raises():
    lay = _good_layout("compose-tiny")
    with pytest.raises(UIError, match=r"cols"):
        compose_layout(lay, 20, 20, panels={})
    with pytest.raises(UIError, match=r"rows"):
        compose_layout(lay, 60, 6, panels={})


def test_compose_layout_missing_region_renders_empty_box():
    lay = _good_layout("compose-missing")
    out = compose_layout(lay, 60, 20, panels={"header": "only-header"})
    assert "only-header" in out
    assert "body" in out
    assert "footer" in out


def test_compose_layout_untrusted_content_is_sanitized():
    lay = _good_layout("compose-sanitize")
    out = compose_layout(
        lay,
        60,
        20,
        panels={"body": "safe\x00line\x07still"},
    )
    assert "\x00" not in out
    assert "\x07" not in out
    assert "safelinestill" in out or "safeline" in out


def test_registry_contains_pre_registered_layouts():
    assert "stoic-3pane" in PANEL_LAYOUT_REGISTRY
    assert "stoic-focus" in PANEL_LAYOUT_REGISTRY
    three = PANEL_LAYOUT_REGISTRY["stoic-3pane"]
    focus = PANEL_LAYOUT_REGISTRY["stoic-focus"]
    assert three.regions == ("header", "transcript", "status", "footer")
    assert three.min_cols == 60 and three.min_rows == 20
    assert focus.regions == ("header", "transcript", "footer")
    assert focus.min_cols == 40 and focus.min_rows == 12


def test_register_get_list_roundtrip_and_duplicate_rejected():
    lay = _good_layout("roundtrip-unique-1")
    assert "roundtrip-unique-1" not in PANEL_LAYOUT_REGISTRY
    register_panel_layout(lay)
    try:
        assert get_panel_layout("roundtrip-unique-1") is lay
        assert "roundtrip-unique-1" in list_panel_layouts()
        with pytest.raises(UIError, match=r"already registered"):
            register_panel_layout(lay)
    finally:
        PANEL_LAYOUT_REGISTRY.pop("roundtrip-unique-1", None)


def test_get_unknown_layout_raises():
    with pytest.raises(UIError, match=r"no panel layout"):
        get_panel_layout("does-not-exist-xyz")


def test_register_non_layout_rejected():
    with pytest.raises(UIError):
        register_panel_layout("not a layout")  # type: ignore[arg-type]
