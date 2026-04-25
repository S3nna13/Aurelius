"""Mock mouse and keyboard controller for computer-use automation.

No OS automation libraries are imported — this is a pure/mock implementation
suitable for unit-testing agentic action sequences.
"""

from __future__ import annotations

from typing import Any


class MouseKeyboardController:
    """Records mouse and keyboard actions in an internal log.

    All coordinates are validated (0‑99999).  Invalid inputs raise immediately.
    """

    _VALID_BUTTONS: frozenset[str] = frozenset({"left", "right", "middle"})
    _VALID_DIRECTIONS: frozenset[str] = frozenset({"up", "down"})
    _MAX_COORD: int = 99_999
    _MAX_TEXT_LEN: int = 1_000

    def __init__(self) -> None:
        self._log: list[dict[str, Any]] = []

    def _validate_coords(self, x: int, y: int) -> None:
        if not isinstance(x, int) or not isinstance(y, int):
            raise TypeError(
                f"coordinates must be ints, got {type(x).__name__} and {type(y).__name__}"
            )
        if x < 0 or y < 0:
            raise ValueError(f"coordinates must be non-negative, got ({x}, {y})")
        if x > self._MAX_COORD or y > self._MAX_COORD:
            raise ValueError(
                f"coordinates must be <= {self._MAX_COORD}, got ({x}, {y})"
            )

    def _validate_button(self, button: str) -> None:
        if button not in self._VALID_BUTTONS:
            raise ValueError(
                f"button must be one of {sorted(self._VALID_BUTTONS)}, got {button!r}"
            )

    def move_mouse(self, x: int, y: int) -> None:
        """Record a move-mouse action after validating coordinates."""
        self._validate_coords(x, y)
        self._log.append({"action": "move_mouse", "x": x, "y": y})

    def click(self, x: int, y: int, button: str = "left") -> None:
        """Record a click action after validating coordinates and button."""
        self._validate_coords(x, y)
        self._validate_button(button)
        self._log.append({"action": "click", "x": x, "y": y, "button": button})

    def scroll(
        self,
        x: int,
        y: int,
        direction: str = "down",
        amount: int = 3,
    ) -> None:
        """Record a scroll action after validating inputs."""
        self._validate_coords(x, y)
        if direction not in self._VALID_DIRECTIONS:
            raise ValueError(
                f"direction must be one of {sorted(self._VALID_DIRECTIONS)}, got {direction!r}"
            )
        if not isinstance(amount, int) or amount < 1:
            raise ValueError(f"amount must be a positive int, got {amount!r}")
        self._log.append(
            {
                "action": "scroll",
                "x": x,
                "y": y,
                "direction": direction,
                "amount": amount,
            }
        )

    def type_text(self, text: str, interval: float = 0.01) -> None:
        """Record a type-text action after validating inputs."""
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        if len(text) > self._MAX_TEXT_LEN:
            raise ValueError(
                f"text length must be <= {self._MAX_TEXT_LEN}, got {len(text)}"
            )
        if interval < 0:
            raise ValueError(f"interval must be non-negative, got {interval}")
        self._log.append({"action": "type_text", "text": text, "interval": interval})

    def press_key(self, key: str) -> None:
        """Record a single key-press action."""
        if not isinstance(key, str) or not key:
            raise ValueError("key must be a non-empty str")
        self._log.append({"action": "press_key", "key": key})

    def hotkey(self, keys: list[str]) -> None:
        """Record a hotkey (multiple keys) action."""
        if not isinstance(keys, list):
            raise TypeError(f"keys must be a list, got {type(keys).__name__}")
        if not all(isinstance(k, str) for k in keys):
            raise TypeError("all keys must be str")
        self._log.append({"action": "hotkey", "keys": list(keys)})

    def get_action_log(self) -> list[dict]:
        """Return a shallow copy of the internal action log."""
        return list(self._log)

    def clear_log(self) -> None:
        """Clear the internal action log."""
        self._log.clear()
