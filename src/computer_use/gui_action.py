"""GUI action prediction layer for the Aurelius computer_use surface.

Inspired by OpenDevin/OpenDevin (browser tool), MoonshotAI/Kimi-Dev (coding agent loop),
Apache-2.0, clean-room reimplementation.

No playwright, pyautogui, or OS accessibility API imports. No ML dependencies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.computer_use.screen_parser import AccessibilityNode, ScreenSnapshot


# ---------------------------------------------------------------------------
# Enums and data structures
# ---------------------------------------------------------------------------

class ActionType(Enum):
    """Supported GUI action types."""

    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    DRAG = "drag"
    KEY = "key"
    WAIT = "wait"
    SCREENSHOT = "screenshot"


@dataclass
class GUIAction:
    """A single GUI action to be executed."""

    action_type: ActionType
    target_selector: str | None = None
    value: str | None = None
    coords: tuple[int, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class GUIActionError(Exception):
    """Raised when a GUI action cannot be constructed or executed."""


# ---------------------------------------------------------------------------
# Abstract predictor
# ---------------------------------------------------------------------------

class ActionPredictor(ABC):
    """Abstract interface for GUI action predictors."""

    @abstractmethod
    def predict(self, snapshot: ScreenSnapshot, goal: str) -> list[GUIAction]:
        """Predict a sequence of GUI actions to achieve *goal* given *snapshot*.

        Parameters
        ----------
        snapshot:
            Current screen state.
        goal:
            Natural-language description of the desired outcome.

        Returns
        -------
        list[GUIAction]
        """
        ...


# ---------------------------------------------------------------------------
# Rule-based predictor (no ML dependencies)
# ---------------------------------------------------------------------------

class RuleBasedPredictor(ActionPredictor):
    """Simple deterministic predictor.

    Given a goal string, returns a CLICK action targeting the first
    AccessibilityNode whose ``name`` contains any keyword extracted from
    the goal.  Falls back to an empty list when no node matches.

    No ML dependencies — pure string matching only.
    """

    # Words that are too generic to be useful as search keywords.
    _STOP_WORDS: frozenset[str] = frozenset({
        "a", "an", "the", "on", "in", "at", "to", "for",
        "of", "and", "or", "with", "is", "it", "do", "go",
    })

    def predict(self, snapshot: ScreenSnapshot, goal: str) -> list[GUIAction]:
        """Return a CLICK action for the first matching node, or [] if none found."""
        keywords = [
            word.lower()
            for word in goal.split()
            if word.lower() not in self._STOP_WORDS and len(word) > 1
        ]

        if not keywords:
            return []

        match = self._find_node(snapshot.root_node, keywords)
        if match is None:
            return []

        coords: tuple[int, int] | None = None
        if match.bbox is not None:
            x, y, w, h = match.bbox
            coords = (x + w // 2, y + h // 2)

        action = GUIAction(
            action_type=ActionType.CLICK,
            target_selector=match.name,
            coords=coords,
            metadata={"matched_role": match.role, "goal": goal},
        )
        return [action]

    def _find_node(
        self, node: AccessibilityNode, keywords: list[str]
    ) -> AccessibilityNode | None:
        """Depth-first search for first node whose name matches any keyword."""
        node_name_lower = node.name.lower()
        if any(kw in node_name_lower for kw in keywords):
            return node
        for child in node.children:
            result = self._find_node(child, keywords)
            if result is not None:
                return result
        return None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GUI_ACTION_REGISTRY: dict[str, type[ActionPredictor]] = {
    "rule_based": RuleBasedPredictor,
}


def register_action_predictor(name: str, cls: type[ActionPredictor]) -> None:
    """Register an ActionPredictor implementation under *name*.

    Raises
    ------
    TypeError
        If *cls* is not a subclass of ActionPredictor.
    """
    if not (isinstance(cls, type) and issubclass(cls, ActionPredictor)):
        raise TypeError(f"{cls!r} must be a subclass of ActionPredictor")
    GUI_ACTION_REGISTRY[name] = cls


def get_action_predictor(name: str) -> type[ActionPredictor]:
    """Retrieve a registered ActionPredictor class by *name*.

    Raises
    ------
    KeyError
        If *name* is not in the registry.
    """
    if name not in GUI_ACTION_REGISTRY:
        raise KeyError(
            f"No ActionPredictor registered as {name!r}. "
            f"Available: {sorted(GUI_ACTION_REGISTRY)}"
        )
    return GUI_ACTION_REGISTRY[name]
