"""Tests for mouse_keyboard_controller — programmatic input simulation."""
from __future__ import annotations

from src.computer_use.mouse_keyboard_controller import (
    MouseKeyboardController,
    KeyAction,
    MouseAction,
    MouseButton,
)


class TestMouseAction:
    def test_move_to_coords(self):
        action = MouseAction.move_to(100, 200)
        assert action.type == "move"
        assert action.x == 100
        assert action.y == 200

    def test_click_defaults_left(self):
        action = MouseAction.click()
        assert action.type == "click"
        assert action.button == MouseButton.LEFT

    def test_click_right(self):
        action = MouseAction.click(button=MouseButton.RIGHT)
        assert action.button == MouseButton.RIGHT

    def test_scroll(self):
        action = MouseAction.scroll(delta_y=-3)
        assert action.type == "scroll"
        assert action.delta_y == -3

    def test_drag(self):
        action = MouseAction.drag(0, 0, 50, 60)
        assert action.type == "drag"
        assert action.end_x == 50

    def test_double_click(self):
        action = MouseAction.double_click()
        assert action.type == "double_click"


class TestKeyAction:
    def test_type_text(self):
        action = KeyAction.type_text("hello")
        assert action.type == "type"
        assert action.text == "hello"

    def test_press_key(self):
        action = KeyAction.press("enter")
        assert action.type == "press"
        assert action.key == "enter"

    def test_hotkey(self):
        action = KeyAction.hotkey(["ctrl", "c"])
        assert action.type == "hotkey"
        assert "c" in action.keys

    def test_empty_text_rejected(self):
        import pytest
        with pytest.raises(ValueError, match="empty"):
            KeyAction.type_text("")


class TestMouseKeyboardController:
    def test_record_and_replay_actions(self):
        ctrl = MouseKeyboardController()
        ctrl.record_action(MouseAction.move_to(100, 200))
        ctrl.record_action(KeyAction.type_text("hello"))
        ctrl.record_action(MouseAction.click())
        assert len(ctrl.history) == 3
        assert ctrl.history[0].type == "move"

    def test_clear_history(self):
        ctrl = MouseKeyboardController()
        ctrl.record_action(MouseAction.click())
        ctrl.clear_history()
        assert len(ctrl.history) == 0

    def test_replay_returns_action_list(self):
        ctrl = MouseKeyboardController()
        ctrl.record_action(MouseAction.move_to(10, 20))
        replayed = ctrl.get_actions()
        assert len(replayed) == 1
        assert replayed[0].x == 10
