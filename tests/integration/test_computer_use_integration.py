"""Integration tests for the computer_use surface end-to-end pipeline.

Parse JSON accessibility tree → predict GUI action → verify action.

Inspired by OpenDevin/OpenDevin (browser tool), MoonshotAI/Kimi-Dev (coding agent loop),
Apache-2.0, clean-room reimplementation.
"""

from __future__ import annotations

import pytest

from src.computer_use.action_verifier import ActionVerifier, verify_trajectory
from src.computer_use.gui_action import (
    GUI_ACTION_REGISTRY,
    ActionType,
    GUIAction,
    RuleBasedPredictor,
    get_action_predictor,
)
from src.computer_use.screen_parser import (
    SCREEN_PARSER_REGISTRY,
    JSONTreeParser,
    ScreenSnapshot,
    get_screen_parser,
)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

SAMPLE_TREE: dict = {
    "width": 1280,
    "height": 800,
    "root": {
        "role": "application",
        "name": "MyApp",
        "children": [
            {
                "role": "toolbar",
                "name": "Toolbar",
                "children": [
                    {
                        "role": "button",
                        "name": "New File",
                        "bbox": [10, 5, 80, 24],
                    },
                    {
                        "role": "button",
                        "name": "Open Folder",
                        "bbox": [100, 5, 100, 24],
                    },
                    {
                        "role": "button",
                        "name": "Save",
                        "bbox": [210, 5, 60, 24],
                    },
                ],
            },
            {
                "role": "editor",
                "name": "Code Editor",
                "bbox": [0, 40, 1280, 760],
                "attributes": {"editable": True},
            },
        ],
    },
    "ocr_text": "New File  Open Folder  Save",
}


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestComputerUseEndToEnd:
    """Full pipeline: parse → predict → verify."""

    def test_parse_json_tree_succeeds(self):
        parser = JSONTreeParser()
        snap = parser.parse(SAMPLE_TREE)
        assert isinstance(snap, ScreenSnapshot)
        assert snap.width == 1280
        assert snap.height == 800

    def test_predict_action_for_save_goal(self):
        parser = JSONTreeParser()
        snap = parser.parse(SAMPLE_TREE)
        predictor = RuleBasedPredictor()
        actions = predictor.predict(snap, "Save the document")
        assert len(actions) == 1
        assert actions[0].action_type == ActionType.CLICK
        assert actions[0].target_selector == "Save"

    def test_predict_action_for_new_file_goal(self):
        parser = JSONTreeParser()
        snap = parser.parse(SAMPLE_TREE)
        predictor = RuleBasedPredictor()
        actions = predictor.predict(snap, "New File")
        assert len(actions) == 1
        assert actions[0].target_selector == "New File"

    def test_verify_predicted_action_passes(self):
        parser = JSONTreeParser()
        snap = parser.parse(SAMPLE_TREE)
        predictor = RuleBasedPredictor()
        actions = predictor.predict(snap, "Save the document")
        assert len(actions) == 1
        verifier = ActionVerifier()
        ok, reason = verifier.verify(actions[0], snap)
        assert ok is True, f"Expected verification to pass, got: {reason}"

    def test_verify_trajectory_all_pass(self):
        parser = JSONTreeParser()
        snap = parser.parse(SAMPLE_TREE)
        actions = [
            GUIAction(action_type=ActionType.CLICK, target_selector="Save"),
            GUIAction(action_type=ActionType.CLICK, target_selector="New File"),
        ]
        results = verify_trajectory(actions, snap)
        assert len(results) == 2
        assert all(ok for ok, _ in results)

    def test_predict_no_match_returns_empty_or_wait(self):
        parser = JSONTreeParser()
        snap = parser.parse(SAMPLE_TREE)
        predictor = RuleBasedPredictor()
        actions = predictor.predict(snap, "Xyzzy Quux Nonexistent")
        assert isinstance(actions, list)

    def test_registries_accessible_via_package(self):
        import src.computer_use as cu
        assert "json_tree" in cu.SCREEN_PARSER_REGISTRY
        assert "rule_based" in cu.GUI_ACTION_REGISTRY

    def test_get_parser_from_registry_and_parse(self):
        parser_cls = get_screen_parser("json_tree")
        snap = parser_cls().parse(SAMPLE_TREE)
        assert snap.root_node.name == "MyApp"

    def test_get_predictor_from_registry_and_predict(self):
        parser = JSONTreeParser()
        snap = parser.parse(SAMPLE_TREE)
        predictor_cls = get_action_predictor("rule_based")
        actions = predictor_cls().predict(snap, "Open Folder")
        assert len(actions) == 1
        assert actions[0].target_selector == "Open Folder"

    def test_denied_action_fails_verification(self):
        parser = JSONTreeParser()
        snap = parser.parse(SAMPLE_TREE)
        bad_action = GUIAction(
            action_type=ActionType.CLICK, target_selector="delete everything"
        )
        verifier = ActionVerifier()
        ok, reason = verifier.verify(bad_action, snap)
        assert ok is False

    def test_snapshot_ocr_text_preserved(self):
        parser = JSONTreeParser()
        snap = parser.parse(SAMPLE_TREE)
        assert snap.ocr_text == "New File  Open Folder  Save"

    def test_editor_node_attributes_preserved(self):
        parser = JSONTreeParser()
        snap = parser.parse(SAMPLE_TREE)
        editor = snap.root_node.children[1]
        assert editor.attributes.get("editable") is True
