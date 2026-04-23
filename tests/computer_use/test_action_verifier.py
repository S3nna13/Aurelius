"""Tests for src.computer_use.action_verifier."""

from __future__ import annotations

import pytest

from src.computer_use.action_verifier import (
    VERIFIER_DENY_LIST,
    ActionVerifier,
    verify_trajectory,
)
from src.computer_use.gui_action import ActionType, GUIAction
from src.computer_use.screen_parser import JSONTreeParser, ScreenSnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_snapshot(node_names: list[str] | None = None) -> ScreenSnapshot:
    raw = {
        "width": 1024,
        "height": 768,
        "root": {
            "role": "window",
            "name": "Root",
            "children": [
                {"role": "button", "name": n, "bbox": [10, 10, 80, 30]}
                for n in (node_names or [])
            ],
        },
    }
    return JSONTreeParser().parse(raw)


# ---------------------------------------------------------------------------
# ActionVerifier.verify tests
# ---------------------------------------------------------------------------

class TestActionVerifierClick:
    def test_valid_click_returns_true(self):
        snap = _make_snapshot(["OK", "Cancel"])
        verifier = ActionVerifier()
        ok, reason = verifier.verify(
            GUIAction(action_type=ActionType.CLICK, target_selector="OK"), snap
        )
        assert ok is True
        assert isinstance(reason, str)

    def test_click_on_nonexistent_node_returns_false(self):
        snap = _make_snapshot(["OK"])
        verifier = ActionVerifier()
        ok, reason = verifier.verify(
            GUIAction(action_type=ActionType.CLICK, target_selector="Nonexistent"), snap
        )
        assert ok is False
        assert "not found" in reason.lower() or "nonexistent" in reason.lower()

    def test_click_with_none_selector_returns_false(self):
        snap = _make_snapshot(["OK"])
        verifier = ActionVerifier()
        ok, reason = verifier.verify(
            GUIAction(action_type=ActionType.CLICK, target_selector=None), snap
        )
        assert ok is False

    def test_click_on_root_node_returns_true(self):
        snap = _make_snapshot()
        verifier = ActionVerifier()
        ok, _ = verifier.verify(
            GUIAction(action_type=ActionType.CLICK, target_selector="Root"), snap
        )
        assert ok is True


class TestActionVerifierType:
    def test_type_with_value_returns_true(self):
        snap = _make_snapshot()
        verifier = ActionVerifier()
        ok, _ = verifier.verify(
            GUIAction(action_type=ActionType.TYPE, value="hello"), snap
        )
        assert ok is True

    def test_type_with_none_value_returns_false(self):
        snap = _make_snapshot()
        verifier = ActionVerifier()
        ok, reason = verifier.verify(
            GUIAction(action_type=ActionType.TYPE, value=None), snap
        )
        assert ok is False

    def test_type_with_empty_string_returns_false(self):
        snap = _make_snapshot()
        verifier = ActionVerifier()
        ok, reason = verifier.verify(
            GUIAction(action_type=ActionType.TYPE, value=""), snap
        )
        assert ok is False


class TestActionVerifierDrag:
    def test_drag_with_coords_returns_true(self):
        snap = _make_snapshot()
        verifier = ActionVerifier()
        ok, _ = verifier.verify(
            GUIAction(action_type=ActionType.DRAG, coords=(100, 200)), snap
        )
        assert ok is True

    def test_drag_without_coords_returns_false(self):
        snap = _make_snapshot()
        verifier = ActionVerifier()
        ok, reason = verifier.verify(
            GUIAction(action_type=ActionType.DRAG, coords=None), snap
        )
        assert ok is False
        assert "coords" in reason.lower()


class TestActionVerifierDenyList:
    def test_deny_list_is_frozenset(self):
        assert isinstance(VERIFIER_DENY_LIST, frozenset)

    def test_deny_list_contains_delete(self):
        assert "delete" in VERIFIER_DENY_LIST

    def test_deny_list_contains_format(self):
        assert "format" in VERIFIER_DENY_LIST

    def test_deny_list_contains_shutdown(self):
        assert "shutdown" in VERIFIER_DENY_LIST

    def test_denied_selector_returns_false(self):
        snap = _make_snapshot(["delete all files"])
        verifier = ActionVerifier()
        ok, reason = verifier.verify(
            GUIAction(
                action_type=ActionType.CLICK, target_selector="delete all files"
            ),
            snap,
        )
        assert ok is False
        assert "denied" in reason.lower() or "delete" in reason.lower()

    def test_denied_value_in_type_returns_false(self):
        snap = _make_snapshot()
        verifier = ActionVerifier()
        ok, reason = verifier.verify(
            GUIAction(action_type=ActionType.TYPE, value="format disk"),
            snap,
        )
        assert ok is False

    def test_denied_goal_in_metadata_returns_false(self):
        snap = _make_snapshot()
        verifier = ActionVerifier()
        ok, reason = verifier.verify(
            GUIAction(
                action_type=ActionType.SCREENSHOT,
                metadata={"goal": "shutdown the system"},
            ),
            snap,
        )
        assert ok is False


class TestVerifyTrajectory:
    def test_empty_list_returns_empty(self):
        snap = _make_snapshot()
        result = verify_trajectory([], snap)
        assert result == []

    def test_single_valid_action(self):
        snap = _make_snapshot(["OK"])
        result = verify_trajectory(
            [GUIAction(action_type=ActionType.CLICK, target_selector="OK")], snap
        )
        assert len(result) == 1
        ok, _ = result[0]
        assert ok is True

    def test_mixed_valid_invalid_actions(self):
        snap = _make_snapshot(["OK"])
        actions = [
            GUIAction(action_type=ActionType.CLICK, target_selector="OK"),
            GUIAction(action_type=ActionType.CLICK, target_selector="Missing"),
        ]
        result = verify_trajectory(actions, snap)
        assert len(result) == 2
        assert result[0][0] is True
        assert result[1][0] is False

    def test_returns_list_of_tuples(self):
        snap = _make_snapshot()
        actions = [GUIAction(action_type=ActionType.WAIT)]
        result = verify_trajectory(actions, snap)
        assert isinstance(result, list)
        assert isinstance(result[0], tuple)
        assert len(result[0]) == 2
