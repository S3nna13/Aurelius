"""Tests for src.computer_use.gui_action."""

from __future__ import annotations

import pytest

from src.computer_use.gui_action import (
    GUI_ACTION_REGISTRY,
    ActionType,
    GUIAction,
    GUIActionError,
    RuleBasedPredictor,
    get_action_predictor,
    register_action_predictor,
)
from src.computer_use.screen_parser import AccessibilityNode, JSONTreeParser, ScreenSnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_snapshot(root_name: str = "Root", children_names: list[str] | None = None) -> ScreenSnapshot:
    root_dict: dict = {
        "width": 1024,
        "height": 768,
        "root": {
            "role": "window",
            "name": root_name,
            "children": [
                {"role": "button", "name": n, "bbox": [10, 10, 80, 30]}
                for n in (children_names or [])
            ],
        },
    }
    return JSONTreeParser().parse(root_dict)


# ---------------------------------------------------------------------------
# RuleBasedPredictor tests
# ---------------------------------------------------------------------------

class TestRuleBasedPredictor:
    def test_finds_matching_node_by_keyword(self):
        snap = _make_snapshot(children_names=["Submit", "Cancel"])
        predictor = RuleBasedPredictor()
        actions = predictor.predict(snap, "Submit the form")
        assert len(actions) == 1
        assert actions[0].action_type == ActionType.CLICK
        assert actions[0].target_selector == "Submit"

    def test_finds_partial_keyword_match(self):
        snap = _make_snapshot(children_names=["Settings Panel", "Home"])
        predictor = RuleBasedPredictor()
        actions = predictor.predict(snap, "settings")
        assert len(actions) == 1
        assert "Settings" in actions[0].target_selector

    def test_no_matching_node_returns_empty_or_wait(self):
        snap = _make_snapshot(children_names=["Foo", "Bar"])
        predictor = RuleBasedPredictor()
        actions = predictor.predict(snap, "Xyzzy Quux Blorp")
        # Spec: returns empty list or WAIT action
        assert isinstance(actions, list)
        for action in actions:
            assert action.action_type in (ActionType.WAIT, ActionType.CLICK)
        if len(actions) == 0:
            pass  # acceptable
        elif len(actions) == 1:
            assert actions[0].action_type == ActionType.WAIT

    def test_empty_goal_returns_empty(self):
        snap = _make_snapshot(children_names=["OK"])
        predictor = RuleBasedPredictor()
        actions = predictor.predict(snap, "")
        assert actions == []

    def test_coords_computed_from_bbox(self):
        snap = _make_snapshot(children_names=["OK"])
        predictor = RuleBasedPredictor()
        actions = predictor.predict(snap, "OK")
        assert len(actions) == 1
        # bbox = [10, 10, 80, 30] → center = (10 + 40, 10 + 15) = (50, 25)
        assert actions[0].coords == (50, 25)

    def test_action_metadata_contains_goal(self):
        snap = _make_snapshot(children_names=["Login"])
        predictor = RuleBasedPredictor()
        actions = predictor.predict(snap, "Login to account")
        assert actions[0].metadata.get("goal") == "Login to account"

    def test_stop_words_not_used_as_keywords(self):
        snap = _make_snapshot(children_names=["the", "an"])
        predictor = RuleBasedPredictor()
        # "the" and "an" are stop words — should not match
        actions = predictor.predict(snap, "the an")
        assert actions == []


# ---------------------------------------------------------------------------
# GUIAction dataclass tests
# ---------------------------------------------------------------------------

class TestGUIActionDataclass:
    def test_click_with_none_coords_is_valid(self):
        action = GUIAction(action_type=ActionType.CLICK, target_selector="OK", coords=None)
        assert action.action_type == ActionType.CLICK
        assert action.coords is None

    def test_default_metadata_is_empty_dict(self):
        action = GUIAction(action_type=ActionType.TYPE, value="hello")
        assert action.metadata == {}

    def test_all_action_types_constructible(self):
        for at in ActionType:
            action = GUIAction(action_type=at)
            assert action.action_type == at

    def test_mismatched_enum_string_raises(self):
        with pytest.raises((ValueError, KeyError, AttributeError)):
            # Passing an invalid string where ActionType is expected should fail
            GUIAction(action_type=ActionType("nonexistent_action"))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# GUIActionError
# ---------------------------------------------------------------------------

class TestGUIActionError:
    def test_is_exception(self):
        err = GUIActionError("test error")
        assert isinstance(err, Exception)
        assert str(err) == "test error"


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestGUIActionRegistry:
    def test_registry_contains_rule_based(self):
        assert "rule_based" in GUI_ACTION_REGISTRY

    def test_rule_based_maps_to_correct_class(self):
        assert GUI_ACTION_REGISTRY["rule_based"] is RuleBasedPredictor

    def test_get_action_predictor_returns_class(self):
        cls = get_action_predictor("rule_based")
        assert cls is RuleBasedPredictor

    def test_get_action_predictor_unknown_raises(self):
        with pytest.raises(KeyError):
            get_action_predictor("nonexistent_predictor")

    def test_register_custom_predictor(self):
        from src.computer_use.gui_action import ActionPredictor

        class DummyPredictor(ActionPredictor):
            def predict(self, snapshot: ScreenSnapshot, goal: str) -> list[GUIAction]:
                return []

        register_action_predictor("dummy_pred_test", DummyPredictor)
        assert get_action_predictor("dummy_pred_test") is DummyPredictor
        # cleanup
        del GUI_ACTION_REGISTRY["dummy_pred_test"]

    def test_register_non_subclass_raises(self):
        with pytest.raises(TypeError):
            register_action_predictor("bad", object)  # type: ignore[arg-type]
