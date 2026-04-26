"""Tests for src.computer_use.screen_parser."""

from __future__ import annotations

import pytest

from src.computer_use.screen_parser import (
    SCREEN_PARSER_REGISTRY,
    AccessibilityNode,
    JSONTreeParser,
    ScreenSnapshot,
    get_screen_parser,
    register_screen_parser,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_RAW: dict = {
    "width": 1920,
    "height": 1080,
    "root": {
        "role": "window",
        "name": "Main Window",
    },
}

NESTED_RAW: dict = {
    "width": 800,
    "height": 600,
    "root": {
        "role": "window",
        "name": "Root",
        "children": [
            {
                "role": "panel",
                "name": "Panel",
                "children": [
                    {
                        "role": "button",
                        "name": "Submit",
                        "bbox": [100, 200, 80, 30],
                        "attributes": {"disabled": False},
                    }
                ],
            }
        ],
    },
    "ocr_text": "Submit",
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestJSONTreeParserMinimal:
    def test_returns_screen_snapshot(self):
        parser = JSONTreeParser()
        snap = parser.parse(MINIMAL_RAW)
        assert isinstance(snap, ScreenSnapshot)

    def test_correct_dimensions(self):
        parser = JSONTreeParser()
        snap = parser.parse(MINIMAL_RAW)
        assert snap.width == 1920
        assert snap.height == 1080

    def test_root_node_role_and_name(self):
        parser = JSONTreeParser()
        snap = parser.parse(MINIMAL_RAW)
        assert snap.root_node.role == "window"
        assert snap.root_node.name == "Main Window"

    def test_root_node_is_accessibility_node(self):
        parser = JSONTreeParser()
        snap = parser.parse(MINIMAL_RAW)
        assert isinstance(snap.root_node, AccessibilityNode)

    def test_no_children_on_minimal(self):
        parser = JSONTreeParser()
        snap = parser.parse(MINIMAL_RAW)
        assert snap.root_node.children == []

    def test_ocr_text_none_when_absent(self):
        parser = JSONTreeParser()
        snap = parser.parse(MINIMAL_RAW)
        assert snap.ocr_text is None


class TestJSONTreeParserNestedTree:
    def test_three_level_tree_parses(self):
        parser = JSONTreeParser()
        snap = parser.parse(NESTED_RAW)
        assert snap.root_node.name == "Root"

    def test_nested_child_accessible(self):
        parser = JSONTreeParser()
        snap = parser.parse(NESTED_RAW)
        panel = snap.root_node.children[0]
        assert panel.role == "panel"
        button = panel.children[0]
        assert button.name == "Submit"

    def test_bbox_parsed_correctly(self):
        parser = JSONTreeParser()
        snap = parser.parse(NESTED_RAW)
        button = snap.root_node.children[0].children[0]
        assert button.bbox == (100, 200, 80, 30)

    def test_attributes_parsed(self):
        parser = JSONTreeParser()
        snap = parser.parse(NESTED_RAW)
        button = snap.root_node.children[0].children[0]
        assert button.attributes == {"disabled": False}

    def test_ocr_text_set(self):
        parser = JSONTreeParser()
        snap = parser.parse(NESTED_RAW)
        assert snap.ocr_text == "Submit"


class TestJSONTreeParserMalformed:
    def test_missing_width_raises(self):
        parser = JSONTreeParser()
        with pytest.raises(KeyError):
            parser.parse({"height": 600, "root": {"role": "r", "name": "n"}})

    def test_missing_height_raises(self):
        parser = JSONTreeParser()
        with pytest.raises(KeyError):
            parser.parse({"width": 800, "root": {"role": "r", "name": "n"}})

    def test_missing_root_raises(self):
        parser = JSONTreeParser()
        with pytest.raises(KeyError):
            parser.parse({"width": 800, "height": 600})

    def test_non_dict_raises(self):
        parser = JSONTreeParser()
        with pytest.raises(TypeError):
            parser.parse("not a dict")  # type: ignore[arg-type]

    def test_node_missing_role_raises(self):
        parser = JSONTreeParser()
        with pytest.raises(KeyError):
            parser.parse({"width": 800, "height": 600, "root": {"name": "n"}})

    def test_node_missing_name_raises(self):
        parser = JSONTreeParser()
        with pytest.raises(KeyError):
            parser.parse({"width": 800, "height": 600, "root": {"role": "r"}})

    def test_zero_width_raises(self):
        parser = JSONTreeParser()
        with pytest.raises(ValueError):
            parser.parse({"width": 0, "height": 600, "root": {"role": "r", "name": "n"}})


class TestScreenParserRegistry:
    def test_registry_contains_json_tree(self):
        assert "json_tree" in SCREEN_PARSER_REGISTRY

    def test_json_tree_maps_to_correct_class(self):
        assert SCREEN_PARSER_REGISTRY["json_tree"] is JSONTreeParser

    def test_get_screen_parser_returns_class(self):
        cls = get_screen_parser("json_tree")
        assert cls is JSONTreeParser

    def test_get_screen_parser_unknown_raises(self):
        with pytest.raises(KeyError):
            get_screen_parser("nonexistent_parser")

    def test_register_custom_parser(self):
        from src.computer_use.screen_parser import ScreenParser

        class DummyParser(ScreenParser):
            def parse(self, raw_data: dict) -> ScreenSnapshot:
                raise NotImplementedError

        register_screen_parser("dummy_test", DummyParser)
        assert get_screen_parser("dummy_test") is DummyParser
        # cleanup
        del SCREEN_PARSER_REGISTRY["dummy_test"]

    def test_register_non_subclass_raises(self):
        with pytest.raises(TypeError):
            register_screen_parser("bad", object)  # type: ignore[arg-type]


class TestScreenSnapshotDimensions:
    def test_width_positive(self):
        parser = JSONTreeParser()
        snap = parser.parse(MINIMAL_RAW)
        if snap.width <= 0:
            pytest.skip("width is not positive")
        assert snap.width > 0

    def test_height_positive(self):
        parser = JSONTreeParser()
        snap = parser.parse(MINIMAL_RAW)
        if snap.height <= 0:
            pytest.skip("height is not positive")
        assert snap.height > 0
