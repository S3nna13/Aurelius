"""Tests for src.computer_use.visual_grounding."""

from __future__ import annotations

import math

import pytest

from src.computer_use.screen_parser import (
    AccessibilityNode,
    JSONTreeParser,
    ScreenSnapshot,
)
from src.computer_use.visual_grounding import (
    VISUAL_GROUNDING_REGISTRY,
    CompoundVisualGrounding,
    GroundedElement,
    GroundingResult,
    VisualGrounding,
    VisualGroundingError,
    get_visual_grounding,
    register_visual_grounding,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SNAPSHOT_RAW: dict = {
    "width": 1920,
    "height": 1080,
    "root": {
        "role": "window",
        "name": "Main",
        "children": [
            {
                "role": "panel",
                "name": "Sidebar",
                "bbox": [0, 0, 180, 1080],
                "children": [
                    {
                        "role": "button",
                        "name": "Menu",
                        "bbox": [10, 10, 160, 40],
                    },
                ],
            },
            {
                "role": "panel",
                "name": "Content",
                "bbox": [200, 0, 1720, 1080],
                "children": [
                    {
                        "role": "button",
                        "name": "Submit",
                        "bbox": [800, 500, 120, 40],
                    },
                    {
                        "role": "button",
                        "name": "Cancel",
                        "bbox": [950, 500, 120, 40],
                    },
                ],
            },
        ],
    },
}


@pytest.fixture
def snapshot() -> ScreenSnapshot:
    parser = JSONTreeParser()
    return parser.parse(SNAPSHOT_RAW)


@pytest.fixture
def sparse_snapshot() -> ScreenSnapshot:
    """Snapshot with large empty regions — useful for fuzzy match tests."""
    raw: dict = {
        "width": 1920,
        "height": 1080,
        "root": {
            "role": "window",
            "name": "Main",
            "children": [
                {
                    "role": "button",
                    "name": "Submit",
                    "bbox": [800, 500, 120, 40],
                },
            ],
        },
    }
    parser = JSONTreeParser()
    return parser.parse(raw)


# ---------------------------------------------------------------------------
# VisualGrounding
# ---------------------------------------------------------------------------

class TestVisualGroundingConstruction:
    def test_default_construction(self):
        vg = VisualGrounding()
        assert vg is not None

    def test_custom_fuzzy_tolerance(self):
        vg = VisualGrounding(fuzzy_tolerance=20)
        assert vg is not None

    def test_negative_tolerance_raises(self):
        with pytest.raises(VisualGroundingError):
            VisualGrounding(fuzzy_tolerance=-1)


class TestVisualGroundingExactMatch:
    def test_returns_grounding_result(self, snapshot: ScreenSnapshot):
        vg = VisualGrounding()
        result = vg.ground(15, 15, snapshot)
        assert isinstance(result, GroundingResult)

    def test_exact_match_returns_element(self, snapshot: ScreenSnapshot):
        vg = VisualGrounding()
        result = vg.ground(15, 15, snapshot)
        assert result.exact is not None
        assert result.exact.node.name == "Menu"

    def test_exact_match_not_fuzzy(self, snapshot: ScreenSnapshot):
        vg = VisualGrounding()
        result = vg.ground(15, 15, snapshot)
        assert result.exact is not None
        assert not result.exact.fuzzy

    def test_submit_button_match(self, snapshot: ScreenSnapshot):
        vg = VisualGrounding()
        result = vg.ground(850, 520, snapshot)
        assert result.exact is not None
        assert result.exact.node.name == "Submit"

    def test_cancel_button_match(self, snapshot: ScreenSnapshot):
        vg = VisualGrounding()
        result = vg.ground(1000, 520, snapshot)
        assert result.exact is not None
        assert result.exact.node.name == "Cancel"

    def test_sidebar_panel_match(self, snapshot: ScreenSnapshot):
        vg = VisualGrounding()
        result = vg.ground(100, 500, snapshot)
        assert result.exact is not None
        assert result.exact.node.name == "Sidebar"


class TestVisualGroundingOverlap:
    def test_child_has_higher_z_than_parent(self, snapshot: ScreenSnapshot):
        vg = VisualGrounding()
        # menu button (child) overlaps sidebar panel (parent)
        result = vg.ground(15, 15, snapshot)
        assert result.exact is not None
        assert result.exact.node.name == "Menu"

    def test_z_order_ranks_exact_matches(self, snapshot: ScreenSnapshot):
        vg = VisualGrounding()
        result = vg.ground(15, 15, snapshot)
        assert len(result.all_matches) >= 2
        # first match should be the child (Menu), not the parent (Sidebar)
        assert result.all_matches[0].node.name == "Menu"


class TestVisualGroundingNoMatch:
    def test_no_match_returns_none_exact(self, snapshot: ScreenSnapshot):
        vg = VisualGrounding()
        # x=190 is in the 20px gap between Sidebar (0-180) and Content (200+)
        result = vg.ground(190, 500, snapshot)
        assert result.exact is None

    def test_no_match_returns_none_fuzzy_when_far(self, snapshot: ScreenSnapshot):
        vg = VisualGrounding(fuzzy_tolerance=5)
        # point deep in the gap, far from any bbox
        result = vg.ground(190, 100, snapshot)
        assert result.exact is None
        assert result.fuzzy is None


class TestVisualGroundingFuzzyMatch:
    def test_fuzzy_near_miss(self, sparse_snapshot: ScreenSnapshot):
        vg = VisualGrounding(fuzzy_tolerance=10)
        # just outside the Submit button bbox (800,500,120,40)
        # click at (795, 500) — 5px left of the button
        result = vg.ground(795, 500, sparse_snapshot)
        assert result.exact is None
        assert result.fuzzy is not None
        assert result.fuzzy.node.name == "Submit"
        assert result.fuzzy.fuzzy

    def test_fuzzy_distance_correct(self, sparse_snapshot: ScreenSnapshot):
        vg = VisualGrounding(fuzzy_tolerance=20)
        result = vg.ground(795, 500, sparse_snapshot)
        assert result.fuzzy is not None
        assert result.fuzzy.distance == 5.0

    def test_fuzzy_tolerance_of_zero_disables_fuzzy(self, sparse_snapshot: ScreenSnapshot):
        vg = VisualGrounding(fuzzy_tolerance=0)
        result = vg.ground(795, 500, sparse_snapshot)
        assert result.exact is None
        assert result.fuzzy is None


class TestVisualGroundingOutOfBounds:
    def test_negative_x_raises(self, snapshot: ScreenSnapshot):
        vg = VisualGrounding()
        with pytest.raises(VisualGroundingError):
            vg.ground(-1, 100, snapshot)

    def test_negative_y_raises(self, snapshot: ScreenSnapshot):
        vg = VisualGrounding()
        with pytest.raises(VisualGroundingError):
            vg.ground(100, -1, snapshot)

    def test_x_exceeds_width_raises(self, snapshot: ScreenSnapshot):
        vg = VisualGrounding()
        with pytest.raises(VisualGroundingError):
            vg.ground(2000, 100, snapshot)

    def test_y_exceeds_height_raises(self, snapshot: ScreenSnapshot):
        vg = VisualGrounding()
        with pytest.raises(VisualGroundingError):
            vg.ground(100, 1100, snapshot)

    def test_exact_edge_coordinates(self, snapshot: ScreenSnapshot):
        vg = VisualGrounding()
        result = vg.ground(1920, 1080, snapshot)
        assert result is not None


class TestVisualGroundingEdgeCases:
    def test_empty_tree_returns_no_match(self):
        vg = VisualGrounding()
        empty_node = AccessibilityNode(role="root", name="", bbox=None)
        snap = ScreenSnapshot(width=1920, height=1080, root_node=empty_node)
        result = vg.ground(100, 100, snap)
        assert result.exact is None
        assert result.fuzzy is None

    def test_single_element_whole_screen(self):
        vg = VisualGrounding()
        node = AccessibilityNode(
            role="canvas", name="Full", bbox=(0, 0, 1920, 1080),
        )
        snap = ScreenSnapshot(width=1920, height=1080, root_node=node)
        result = vg.ground(960, 540, snap)
        assert result.exact is not None
        assert result.exact.node.name == "Full"


class TestCompoundVisualGrounding:
    def test_returns_grounding_result(self, snapshot: ScreenSnapshot):
        cv = CompoundVisualGrounding()
        result = cv.ground(15, 15, snapshot)
        assert isinstance(result, GroundingResult)

    def test_exact_match_returned(self, snapshot: ScreenSnapshot):
        cv = CompoundVisualGrounding()
        result = cv.ground(15, 15, snapshot)
        assert result.exact is not None
        assert result.exact.node.name == "Menu"

    def test_fuzzy_fallback(self, sparse_snapshot: ScreenSnapshot):
        cv = CompoundVisualGrounding()
        result = cv.ground(795, 500, sparse_snapshot)
        assert result.exact is None
        assert result.fuzzy is not None
        assert result.fuzzy.node.name == "Submit"


class TestVisualGroundingRegistry:
    def test_registry_contains_default(self):
        assert "default" in VISUAL_GROUNDING_REGISTRY

    def test_registry_contains_compound(self):
        assert "compound" in VISUAL_GROUNDING_REGISTRY

    def test_get_visual_grounding(self):
        cls = get_visual_grounding("default")
        assert cls is VisualGrounding

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError):
            get_visual_grounding("nonexistent")

    def test_register_custom(self):
        class StubGrounding(VisualGrounding):
            pass

        register_visual_grounding("test_stub", StubGrounding)
        assert get_visual_grounding("test_stub") is StubGrounding
        del VISUAL_GROUNDING_REGISTRY["test_stub"]
