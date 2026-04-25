from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class MouseButton(Enum):
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


@dataclass
class MouseAction:
    type: str
    x: float = 0.0
    y: float = 0.0
    button: MouseButton = MouseButton.LEFT
    delta_y: float = 0.0
    end_x: float = 0.0
    end_y: float = 0.0

    @classmethod
    def move_to(cls, x: float, y: float) -> MouseAction:
        return cls(type="move", x=x, y=y)

    @classmethod
    def click(cls, button: MouseButton = MouseButton.LEFT) -> MouseAction:
        return cls(type="click", button=button)

    @classmethod
    def double_click(cls) -> MouseAction:
        return cls(type="double_click")

    @classmethod
    def scroll(cls, delta_y: float = 0.0) -> MouseAction:
        return cls(type="scroll", delta_y=delta_y)

    @classmethod
    def drag(cls, start_x: float, start_y: float, end_x: float, end_y: float) -> MouseAction:
        return cls(type="drag", x=start_x, y=start_y, end_x=end_x, end_y=end_y)


@dataclass
class KeyAction:
    type: str
    text: str = ""
    key: str = ""
    keys: list[str] = None

    def __post_init__(self) -> None:
        if self.keys is None:
            self.keys = []

    @classmethod
    def type_text(cls, text: str) -> KeyAction:
        if not text:
            raise ValueError("text cannot be empty")
        return cls(type="type", text=text)

    @classmethod
    def press(cls, key: str) -> KeyAction:
        return cls(type="press", key=key)

    @classmethod
    def hotkey(cls, keys: list[str]) -> KeyAction:
        return cls(type="hotkey", keys=keys)


class MouseKeyboardController:
    def __init__(self) -> None:
        self.history: list[MouseAction | KeyAction] = []

    def record_action(self, action: MouseAction | KeyAction) -> None:
        self.history.append(action)

    def clear_history(self) -> None:
        self.history.clear()

    def get_actions(self) -> list[MouseAction | KeyAction]:
        return list(self.history)


MOUSE_KEYBOARD_CONTROLLER = MouseKeyboardController()
