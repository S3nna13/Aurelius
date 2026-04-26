"""
Tests for src/computer_use/action_history.py
"""

import unittest

from src.computer_use.action_history import (
    ACTION_HISTORY_REGISTRY,
    Action,
    ActionHistory,
    ActionType,
)


class TestActionType(unittest.TestCase):
    def test_click_value(self):
        self.assertEqual(ActionType.CLICK.value, "click")

    def test_type_value(self):
        self.assertEqual(ActionType.TYPE.value, "type")

    def test_scroll_value(self):
        self.assertEqual(ActionType.SCROLL.value, "scroll")

    def test_navigate_value(self):
        self.assertEqual(ActionType.NAVIGATE.value, "navigate")

    def test_screenshot_value(self):
        self.assertEqual(ActionType.SCREENSHOT.value, "screenshot")

    def test_wait_value(self):
        self.assertEqual(ActionType.WAIT.value, "wait")

    def test_all_six_members(self):
        members = {at.name for at in ActionType}
        self.assertEqual(members, {"CLICK", "TYPE", "SCROLL", "NAVIGATE", "SCREENSHOT", "WAIT"})


class TestAction(unittest.TestCase):
    def test_action_id_auto_generated(self):
        action = Action(action_type=ActionType.CLICK, target="button")
        self.assertIsInstance(action.action_id, str)
        self.assertEqual(len(action.action_id), 8)

    def test_action_id_unique(self):
        a1 = Action(action_type=ActionType.CLICK, target="button")
        a2 = Action(action_type=ActionType.CLICK, target="button")
        self.assertNotEqual(a1.action_id, a2.action_id)

    def test_action_frozen(self):
        action = Action(action_type=ActionType.TYPE, target="field", payload="hello")
        with self.assertRaises(Exception):
            action.payload = "changed"  # type: ignore[misc]

    def test_action_defaults(self):
        action = Action(action_type=ActionType.WAIT, target="")
        self.assertEqual(action.payload, "")
        self.assertIsInstance(action.timestamp, float)

    def test_action_custom_fields(self):
        action = Action(
            action_type=ActionType.NAVIGATE,
            target="http://example.com",
            payload="GET",
            timestamp=42.0,
            action_id="abcd1234",
        )
        self.assertEqual(action.target, "http://example.com")
        self.assertEqual(action.payload, "GET")
        self.assertEqual(action.timestamp, 42.0)
        self.assertEqual(action.action_id, "abcd1234")


class TestActionHistory(unittest.TestCase):
    def setUp(self):
        self.history = ActionHistory()

    def _make_action(
        self, action_type: ActionType = ActionType.CLICK, target: str = "btn"
    ) -> Action:
        return Action(action_type=action_type, target=target)

    # --- record ---
    def test_record_stores_action(self):
        action = self._make_action()
        self.history.record(action)
        self.assertEqual(len(self.history), 1)

    def test_record_multiple(self):
        for _ in range(5):
            self.history.record(self._make_action())
        self.assertEqual(len(self.history), 5)

    # --- len ---
    def test_len_empty(self):
        self.assertEqual(len(self.history), 0)

    def test_len_after_record(self):
        self.history.record(self._make_action())
        self.history.record(self._make_action())
        self.assertEqual(len(self.history), 2)

    # --- undo ---
    def test_undo_returns_last(self):
        a1 = self._make_action(target="first")
        a2 = self._make_action(target="second")
        self.history.record(a1)
        self.history.record(a2)
        result = self.history.undo()
        self.assertEqual(result, a2)

    def test_undo_removes_last(self):
        self.history.record(self._make_action(target="a"))
        self.history.record(self._make_action(target="b"))
        self.history.undo()
        self.assertEqual(len(self.history), 1)

    def test_undo_empty_returns_none(self):
        self.assertIsNone(self.history.undo())

    def test_undo_reduces_length(self):
        self.history.record(self._make_action())
        self.assertEqual(len(self.history), 1)
        self.history.undo()
        self.assertEqual(len(self.history), 0)

    # --- max_history ---
    def test_max_history_raises_value_error(self):
        small_history = ActionHistory(max_history=2)
        small_history.record(self._make_action())
        small_history.record(self._make_action())
        with self.assertRaises(ValueError):
            small_history.record(self._make_action())

    def test_max_history_exactly_at_limit_ok(self):
        small_history = ActionHistory(max_history=3)
        for _ in range(3):
            small_history.record(self._make_action())
        self.assertEqual(len(small_history), 3)

    # --- filter_by_type ---
    def test_filter_by_type_returns_matching(self):
        self.history.record(Action(action_type=ActionType.CLICK, target="btn"))
        self.history.record(Action(action_type=ActionType.TYPE, target="field"))
        self.history.record(Action(action_type=ActionType.CLICK, target="link"))
        clicks = self.history.filter_by_type(ActionType.CLICK)
        self.assertEqual(len(clicks), 2)
        for a in clicks:
            self.assertEqual(a.action_type, ActionType.CLICK)

    def test_filter_by_type_no_match(self):
        self.history.record(Action(action_type=ActionType.CLICK, target="btn"))
        result = self.history.filter_by_type(ActionType.SCROLL)
        self.assertEqual(result, [])

    # --- replay ---
    def test_replay_calls_fn_for_each(self):
        calls = []

        def mock_fn(action: Action) -> str:
            calls.append(action)
            return f"result:{action.target}"

        a1 = Action(action_type=ActionType.CLICK, target="a")
        a2 = Action(action_type=ActionType.TYPE, target="b")
        self.history.record(a1)
        self.history.record(a2)
        results = self.history.replay(mock_fn)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], "result:a")
        self.assertEqual(results[1], "result:b")
        self.assertEqual(len(calls), 2)

    def test_replay_empty(self):
        results = self.history.replay(lambda a: "x")
        self.assertEqual(results, [])

    # --- export ---
    def test_export_returns_list_of_dicts(self):
        self.history.record(Action(action_type=ActionType.CLICK, target="btn"))
        exported = self.history.export()
        self.assertIsInstance(exported, list)
        self.assertIsInstance(exported[0], dict)

    def test_export_dict_keys(self):
        self.history.record(Action(action_type=ActionType.TYPE, target="field", payload="hello"))
        entry = self.history.export()[0]
        for key in ("action_id", "action_type", "target", "payload", "timestamp"):
            self.assertIn(key, entry)

    def test_export_action_type_is_string(self):
        self.history.record(Action(action_type=ActionType.NAVIGATE, target="url"))
        entry = self.history.export()[0]
        self.assertIsInstance(entry["action_type"], str)
        self.assertEqual(entry["action_type"], "navigate")

    def test_export_empty(self):
        self.assertEqual(self.history.export(), [])

    # --- REGISTRY ---
    def test_registry_exists(self):
        self.assertIn("default", ACTION_HISTORY_REGISTRY)

    def test_registry_default_is_class(self):
        self.assertIs(ACTION_HISTORY_REGISTRY["default"], ActionHistory)

    def test_registry_default_instantiable(self):
        cls = ACTION_HISTORY_REGISTRY["default"]
        instance = cls()
        self.assertIsInstance(instance, ActionHistory)


if __name__ == "__main__":
    unittest.main()
