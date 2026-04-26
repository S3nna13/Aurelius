"""Tests for mouse_keyboard_controller — programmatic input simulation."""

from __future__ import annotations

import pytest

from src.computer_use.mouse_keyboard_controller import MouseKeyboardController

# ---------------------------------------------------------------------------
# Action method tests
# ---------------------------------------------------------------------------


class TestMouseActions:
    def test_move_mouse_logs(self):
        ctrl = MouseKeyboardController()
        ctrl.move_mouse(100, 200)
        assert ctrl.get_action_log() == [{"action": "move_mouse", "x": 100, "y": 200}]

    def test_click_logs(self):
        ctrl = MouseKeyboardController()
        ctrl.click(50, 60, button="right")
        assert ctrl.get_action_log() == [{"action": "click", "x": 50, "y": 60, "button": "right"}]

    def test_click_defaults_left(self):
        ctrl = MouseKeyboardController()
        ctrl.click(10, 20)
        assert ctrl.get_action_log()[0]["button"] == "left"

    def test_scroll_logs(self):
        ctrl = MouseKeyboardController()
        ctrl.scroll(300, 400, direction="up", amount=5)
        assert ctrl.get_action_log() == [
            {
                "action": "scroll",
                "x": 300,
                "y": 400,
                "direction": "up",
                "amount": 5,
            }
        ]

    def test_scroll_defaults(self):
        ctrl = MouseKeyboardController()
        ctrl.scroll(0, 0)
        entry = ctrl.get_action_log()[0]
        assert entry["direction"] == "down"
        assert entry["amount"] == 3


class TestKeyboardActions:
    def test_type_text_logs(self):
        ctrl = MouseKeyboardController()
        ctrl.type_text("hello", interval=0.05)
        assert ctrl.get_action_log() == [{"action": "type_text", "text": "hello", "interval": 0.05}]

    def test_type_text_defaults_interval(self):
        ctrl = MouseKeyboardController()
        ctrl.type_text("world")
        assert ctrl.get_action_log()[0]["interval"] == 0.01

    def test_press_key_logs(self):
        ctrl = MouseKeyboardController()
        ctrl.press_key("enter")
        assert ctrl.get_action_log() == [{"action": "press_key", "key": "enter"}]

    def test_hotkey_logs(self):
        ctrl = MouseKeyboardController()
        ctrl.hotkey(["ctrl", "c"])
        assert ctrl.get_action_log() == [{"action": "hotkey", "keys": ["ctrl", "c"]}]


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestCoordinateValidation:
    def test_negative_x_rejected(self):
        ctrl = MouseKeyboardController()
        with pytest.raises(ValueError, match="non-negative"):
            ctrl.move_mouse(-1, 0)

    def test_negative_y_rejected(self):
        ctrl = MouseKeyboardController()
        with pytest.raises(ValueError, match="non-negative"):
            ctrl.move_mouse(0, -1)

    def test_x_too_large_rejected(self):
        ctrl = MouseKeyboardController()
        with pytest.raises(ValueError, match="99999"):
            ctrl.move_mouse(100_000, 0)

    def test_y_too_large_rejected(self):
        ctrl = MouseKeyboardController()
        with pytest.raises(ValueError, match="99999"):
            ctrl.move_mouse(0, 100_000)

    def test_click_negative_coords_rejected(self):
        ctrl = MouseKeyboardController()
        with pytest.raises(ValueError, match="non-negative"):
            ctrl.click(-5, 10)

    def test_scroll_negative_coords_rejected(self):
        ctrl = MouseKeyboardController()
        with pytest.raises(ValueError, match="non-negative"):
            ctrl.scroll(0, -10)

    def test_non_int_coords_rejected(self):
        ctrl = MouseKeyboardController()
        with pytest.raises(TypeError, match="ints"):
            ctrl.move_mouse(1.5, 2)  # type: ignore[arg-type]


class TestButtonValidation:
    def test_invalid_button_rejected(self):
        ctrl = MouseKeyboardController()
        with pytest.raises(ValueError, match="left.*middle.*right"):
            ctrl.click(0, 0, button="invalid")

    def test_middle_button_accepted(self):
        ctrl = MouseKeyboardController()
        ctrl.click(0, 0, button="middle")
        assert ctrl.get_action_log()[0]["button"] == "middle"


class TestTextValidation:
    def test_text_not_str_rejected(self):
        ctrl = MouseKeyboardController()
        with pytest.raises(TypeError, match="str"):
            ctrl.type_text(123)  # type: ignore[arg-type]

    def test_text_too_long_rejected(self):
        ctrl = MouseKeyboardController()
        with pytest.raises(ValueError, match="1000"):
            ctrl.type_text("x" * 1001)

    def test_text_exactly_1000_accepted(self):
        ctrl = MouseKeyboardController()
        ctrl.type_text("x" * 1000)
        assert len(ctrl.get_action_log()) == 1

    def test_negative_interval_rejected(self):
        ctrl = MouseKeyboardController()
        with pytest.raises(ValueError, match="non-negative"):
            ctrl.type_text("hi", interval=-0.1)


class TestScrollValidation:
    def test_invalid_direction_rejected(self):
        ctrl = MouseKeyboardController()
        with pytest.raises(ValueError, match="down.*up"):
            ctrl.scroll(0, 0, direction="sideways")

    def test_zero_amount_rejected(self):
        ctrl = MouseKeyboardController()
        with pytest.raises(ValueError, match="positive int"):
            ctrl.scroll(0, 0, amount=0)

    def test_negative_amount_rejected(self):
        ctrl = MouseKeyboardController()
        with pytest.raises(ValueError, match="positive int"):
            ctrl.scroll(0, 0, amount=-1)


class TestKeyValidation:
    def test_empty_key_rejected(self):
        ctrl = MouseKeyboardController()
        with pytest.raises(ValueError, match="non-empty"):
            ctrl.press_key("")

    def test_hotkey_not_list_rejected(self):
        ctrl = MouseKeyboardController()
        with pytest.raises(TypeError, match="list"):
            ctrl.hotkey("ctrl")  # type: ignore[arg-type]

    def test_hotkey_non_string_element_rejected(self):
        ctrl = MouseKeyboardController()
        with pytest.raises(TypeError, match="str"):
            ctrl.hotkey(["ctrl", 1])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Action log tests
# ---------------------------------------------------------------------------


class TestActionLog:
    def test_log_accumulates(self):
        ctrl = MouseKeyboardController()
        ctrl.move_mouse(1, 2)
        ctrl.click(3, 4)
        ctrl.press_key("tab")
        assert len(ctrl.get_action_log()) == 3

    def test_get_action_log_returns_copy(self):
        ctrl = MouseKeyboardController()
        ctrl.press_key("a")
        log = ctrl.get_action_log()
        log.clear()
        assert len(ctrl.get_action_log()) == 1

    def test_clear_log_empties(self):
        ctrl = MouseKeyboardController()
        ctrl.move_mouse(0, 0)
        ctrl.clear_log()
        assert ctrl.get_action_log() == []

    def test_clear_log_on_empty_is_noop(self):
        ctrl = MouseKeyboardController()
        ctrl.clear_log()
        assert ctrl.get_action_log() == []
