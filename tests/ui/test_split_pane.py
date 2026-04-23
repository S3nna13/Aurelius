"""Tests for src.ui.split_pane — SplitPane, PaneConfig, SplitDirection, SplitPaneError."""

from __future__ import annotations

import pytest
from rich.console import Console

from src.ui.split_pane import (
    SPLIT_PANE_REGISTRY,
    PaneConfig,
    SplitDirection,
    SplitPane,
    SplitPaneError,
)


# ---------------------------------------------------------------------------
# add_pane
# ---------------------------------------------------------------------------


def test_add_pane_visible_by_default() -> None:
    pane = SplitPane()
    cfg = PaneConfig(pane_id="p1", title="Pane 1")
    pane.add_pane(cfg)
    assert pane.panes[0].visible is True


def test_add_pane_appended_to_list() -> None:
    pane = SplitPane()
    pane.add_pane(PaneConfig(pane_id="a", title="A"))
    pane.add_pane(PaneConfig(pane_id="b", title="B"))
    assert len(pane.panes) == 2


def test_add_pane_duplicate_id_raises() -> None:
    pane = SplitPane()
    pane.add_pane(PaneConfig(pane_id="dup", title="D"))
    with pytest.raises(SplitPaneError):
        pane.add_pane(PaneConfig(pane_id="dup", title="D2"))


# ---------------------------------------------------------------------------
# remove_pane
# ---------------------------------------------------------------------------


def test_remove_pane_removes_correctly() -> None:
    pane = SplitPane()
    pane.add_pane(PaneConfig(pane_id="x", title="X"))
    pane.remove_pane("x")
    assert len(pane.panes) == 0


def test_remove_pane_unknown_raises() -> None:
    pane = SplitPane()
    with pytest.raises(SplitPaneError):
        pane.remove_pane("ghost")


# ---------------------------------------------------------------------------
# show / hide
# ---------------------------------------------------------------------------


def test_hide_sets_visible_false() -> None:
    pane = SplitPane()
    pane.add_pane(PaneConfig(pane_id="h1", title="H1"))
    pane.hide("h1")
    assert pane.panes[0].visible is False


def test_show_restores_visible_true() -> None:
    pane = SplitPane()
    pane.add_pane(PaneConfig(pane_id="s1", title="S1", visible=False))
    pane.show("s1")
    assert pane.panes[0].visible is True


def test_hide_unknown_raises() -> None:
    pane = SplitPane()
    with pytest.raises(SplitPaneError):
        pane.hide("nope")


def test_show_unknown_raises() -> None:
    pane = SplitPane()
    with pytest.raises(SplitPaneError):
        pane.show("nope")


# ---------------------------------------------------------------------------
# resize
# ---------------------------------------------------------------------------


def test_resize_updates_weight() -> None:
    pane = SplitPane()
    pane.add_pane(PaneConfig(pane_id="r1", title="R1", weight=1.0))
    pane.resize("r1", 2.5)
    assert pane.panes[0].weight == pytest.approx(2.5)


def test_resize_zero_weight_raises() -> None:
    pane = SplitPane()
    pane.add_pane(PaneConfig(pane_id="r2", title="R2"))
    with pytest.raises(SplitPaneError):
        pane.resize("r2", 0)


def test_resize_negative_weight_raises() -> None:
    pane = SplitPane()
    pane.add_pane(PaneConfig(pane_id="r3", title="R3"))
    with pytest.raises(SplitPaneError):
        pane.resize("r3", -1.0)


def test_resize_unknown_raises() -> None:
    pane = SplitPane()
    with pytest.raises(SplitPaneError):
        pane.resize("ghost", 2.0)


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


def test_render_two_panes_does_not_crash() -> None:
    pane = SplitPane()
    pane.add_pane(PaneConfig(pane_id="left", title="Left"))
    pane.add_pane(PaneConfig(pane_id="right", title="Right"))
    console = Console(record=True)
    pane.render(console)
    output = console.export_text()
    assert len(output) > 0


def test_render_with_content_map_does_not_crash() -> None:
    pane = SplitPane()
    pane.add_pane(PaneConfig(pane_id="main", title="Main"))
    pane.add_pane(PaneConfig(pane_id="side", title="Side"))
    console = Console(record=True)
    pane.render(console, content_map={"main": "Hello world", "side": "Sidebar text"})
    output = console.export_text()
    assert "Hello world" in output


def test_render_no_visible_panes_does_not_crash() -> None:
    pane = SplitPane()
    pane.add_pane(PaneConfig(pane_id="h", title="H", visible=False))
    console = Console(record=True)
    pane.render(console)
    output = console.export_text()
    assert "no visible panes" in output.lower()


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------


def test_to_dict_returns_dict() -> None:
    pane = SplitPane()
    pane.add_pane(PaneConfig(pane_id="d1", title="D1"))
    result = pane.to_dict()
    assert isinstance(result, dict)
    assert "panes" in result
    assert result["panes"][0]["pane_id"] == "d1"


# ---------------------------------------------------------------------------
# SPLIT_PANE_REGISTRY
# ---------------------------------------------------------------------------


def test_split_pane_registry_is_dict() -> None:
    assert isinstance(SPLIT_PANE_REGISTRY, dict)


def test_split_pane_registry_can_store_layout() -> None:
    sp = SplitPane()
    SPLIT_PANE_REGISTRY["test-layout"] = sp
    assert "test-layout" in SPLIT_PANE_REGISTRY
    del SPLIT_PANE_REGISTRY["test-layout"]
