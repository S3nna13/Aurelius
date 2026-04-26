"""
action_history.py
Records and replays computer use actions.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum


class ActionType(Enum):
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    NAVIGATE = "navigate"
    SCREENSHOT = "screenshot"
    WAIT = "wait"


def _default_action_id() -> str:
    return uuid.uuid4().hex[:8]


@dataclass(frozen=True)
class Action:
    action_type: ActionType
    target: str
    payload: str = ""
    timestamp: float = field(default_factory=time.monotonic)
    action_id: str = field(default_factory=_default_action_id)


class ActionHistory:
    def __init__(self, max_history: int = 500) -> None:
        self._max_history = max_history
        self._actions: list[Action] = []

    def record(self, action: Action) -> None:
        if len(self._actions) >= self._max_history:
            raise ValueError(
                f"Action history is full (max_history={self._max_history}). "
                "Cannot record more actions."
            )
        self._actions.append(action)

    def undo(self) -> Action | None:
        if not self._actions:
            return None
        return self._actions.pop()

    def replay(self, predict_fn: Callable[[Action], str]) -> list[str]:
        return [predict_fn(action) for action in self._actions]

    def filter_by_type(self, action_type: ActionType) -> list[Action]:
        return [a for a in self._actions if a.action_type == action_type]

    def export(self) -> list[dict]:
        return [
            {
                "action_id": a.action_id,
                "action_type": a.action_type.value,
                "target": a.target,
                "payload": a.payload,
                "timestamp": a.timestamp,
            }
            for a in self._actions
        ]

    def __len__(self) -> int:
        return len(self._actions)


ACTION_HISTORY_REGISTRY: dict = {"default": ActionHistory}
