"""Tests for src/computer_use/screen_state_tracker.py — 10+ unit tests, no GPU."""

from __future__ import annotations

import time

import pytest

from src.computer_use.screen_state_tracker import (
    COMPUTER_USE_REGISTRY,
    ScreenRegion,
    ScreenState,
    ScreenStateTracker,
)


@pytest.fixture()
def tracker() -> ScreenStateTracker:
    return ScreenStateTracker()


def _make_raw_state(
    regions: list[dict] | None = None,
    focused: str | None = None,
    ts: float | None = None,
) -> dict:
    state: dict = {}
    if regions is not None:
        state["regions"] = regions
    if focused is not None:
        state["focused_region"] = focused
    if ts is not None:
        state["timestamp"] = ts
    return state


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_returns_screen_state(self, tracker):
        state = tracker.update(_make_raw_state())
        assert isinstance(state, ScreenState)

    def test_parses_regions(self, tracker):
        raw = _make_raw_state(regions=[{"x": 10, "y": 20, "width": 100, "height": 50, "label": "header"}])
        state = tracker.update(raw)
        assert len(state.regions) == 1
        r = state.regions[0]
        assert r.x == 10 and r.y == 20 and r.width == 100 and r.height == 50
        assert r.label == "header"

    def test_focused_region_set(self, tracker):
        state = tracker.update(_make_raw_state(focused="nav"))
        assert state.focused_region == "nav"

    def test_focused_region_none_when_missing(self, tracker):
        state = tracker.update(_make_raw_state())
        assert state.focused_region is None

    def test_timestamp_used_from_raw(self, tracker):
        ts = 1_700_000_000.0
        state = tracker.update(_make_raw_state(ts=ts))
        assert state.timestamp == ts

    def test_first_state_change_mask_all_true(self, tracker):
        raw = _make_raw_state(regions=[
            {"x": 0, "y": 0, "width": 50, "height": 50, "label": "a"},
            {"x": 50, "y": 0, "width": 50, "height": 50, "label": "b"},
        ])
        state = tracker.update(raw)
        assert all(state.change_mask)

    def test_second_state_unchanged_mask_false(self, tracker):
        raw = _make_raw_state(regions=[{"x": 0, "y": 0, "width": 50, "height": 50, "label": "a"}])
        tracker.update(raw)
        state2 = tracker.update(raw)
        assert state2.change_mask == [False]

    def test_history_grows(self, tracker):
        for i in range(5):
            tracker.update(_make_raw_state(ts=float(i)))
        assert len(tracker.history) == 5

    def test_history_capped_at_10(self, tracker):
        for i in range(15):
            tracker.update(_make_raw_state(ts=float(i)))
        assert len(tracker.history) == 10

    def test_empty_regions_list(self, tracker):
        state = tracker.update(_make_raw_state(regions=[]))
        assert state.regions == []
        assert state.change_mask == []


# ---------------------------------------------------------------------------
# diff()
# ---------------------------------------------------------------------------

class TestDiff:
    def test_diff_added_region(self, tracker):
        s1 = tracker.update(_make_raw_state(regions=[{"x": 0, "y": 0, "width": 10, "height": 10, "label": "a"}]))
        s2 = tracker.update(_make_raw_state(regions=[
            {"x": 0, "y": 0, "width": 10, "height": 10, "label": "a"},
            {"x": 10, "y": 0, "width": 10, "height": 10, "label": "b"},
        ]))
        changes = tracker.diff(s1, s2)
        assert any("added" in c.lower() and "b" in c for c in changes)

    def test_diff_removed_region(self, tracker):
        s1 = tracker.update(_make_raw_state(regions=[
            {"x": 0, "y": 0, "width": 10, "height": 10, "label": "a"},
            {"x": 10, "y": 0, "width": 10, "height": 10, "label": "b"},
        ]))
        s2 = tracker.update(_make_raw_state(regions=[{"x": 0, "y": 0, "width": 10, "height": 10, "label": "a"}]))
        changes = tracker.diff(s1, s2)
        assert any("removed" in c.lower() and "b" in c for c in changes)

    def test_diff_geometry_change(self, tracker):
        s1 = tracker.update(_make_raw_state(regions=[{"x": 0, "y": 0, "width": 10, "height": 10, "label": "btn"}]))
        s2 = tracker.update(_make_raw_state(regions=[{"x": 5, "y": 5, "width": 20, "height": 20, "label": "btn"}]))
        changes = tracker.diff(s1, s2)
        assert any("geometry" in c.lower() and "btn" in c for c in changes)

    def test_diff_focus_change(self, tracker):
        s1 = tracker.update(_make_raw_state(focused="a"))
        s2 = tracker.update(_make_raw_state(focused="b"))
        changes = tracker.diff(s1, s2)
        assert any("focus" in c.lower() for c in changes)

    def test_diff_no_changes(self, tracker):
        raw = _make_raw_state(regions=[{"x": 0, "y": 0, "width": 10, "height": 10, "label": "x"}], focused="x")
        s1 = tracker.update(raw)
        s2 = tracker.update(raw)
        changes = tracker.diff(s1, s2)
        assert changes == []


# ---------------------------------------------------------------------------
# get_interactive_regions()
# ---------------------------------------------------------------------------

class TestGetInteractiveRegions:
    def test_returns_interactive_only(self, tracker):
        raw = _make_raw_state(regions=[
            {"x": 0, "y": 0, "width": 10, "height": 10, "label": "btn", "interactive": True},
            {"x": 10, "y": 0, "width": 10, "height": 10, "label": "bg", "interactive": False},
        ])
        tracker.update(raw)
        interactive = tracker.get_interactive_regions()
        assert len(interactive) == 1
        assert interactive[0].label == "btn"

    def test_empty_when_no_history(self, tracker):
        result = tracker.get_interactive_regions()
        assert result == []

    def test_no_interactive_key_excluded(self, tracker):
        raw = _make_raw_state(regions=[{"x": 0, "y": 0, "width": 10, "height": 10, "label": "panel"}])
        tracker.update(raw)
        assert tracker.get_interactive_regions() == []


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_key_present(self):
        assert "screen_state_tracker" in COMPUTER_USE_REGISTRY

    def test_registry_value_is_class(self):
        assert COMPUTER_USE_REGISTRY["screen_state_tracker"] is ScreenStateTracker
