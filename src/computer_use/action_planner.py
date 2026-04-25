"""Aurelius computer_use — action_planner.py

Rule-based action planner that maps natural-language goals to sequences of
typed PlannedAction steps.  No LLM inference happens here; all logic is
deterministic keyword-matching so the module is usable in unit tests without
a GPU.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    KEY_COMBO = "key_combo"
    SCREENSHOT = "screenshot"
    WAIT = "wait"
    DRAG = "drag"
    HOVER = "hover"


@dataclass
class PlannedAction:
    action_type: ActionType
    target: str
    params: dict = field(default_factory=dict)
    rationale: str = ""


@dataclass
class ActionPlan:
    steps: list[PlannedAction]
    goal: str
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Keyword → ActionType mapping (order matters — first match wins)
# ---------------------------------------------------------------------------

_KEYWORD_RULES: list[tuple[re.Pattern[str], ActionType]] = [
    (re.compile(r"\bscreenshot\b|\bcapture\b|\bsnap\b", re.I), ActionType.SCREENSHOT),
    (re.compile(r"\bwait\b|\bpause\b|\bdelay\b|\bsleep\b", re.I), ActionType.WAIT),
    (re.compile(r"\bscroll\b|\bswipe\b", re.I), ActionType.SCROLL),
    (re.compile(r"\bdrag\b|\bmove\b.*\bto\b", re.I), ActionType.DRAG),
    (re.compile(r"\bhover\b|\bmouse.?over\b", re.I), ActionType.HOVER),
    (re.compile(r"\bkey.?combo\b|\bshortcut\b|\bpress\b|\bhotkey\b", re.I), ActionType.KEY_COMBO),
    (re.compile(r"\btype\b|\benter\b|\binput\b|\bwrite\b|\bfill\b", re.I), ActionType.TYPE),
    (re.compile(r"\bclick\b|\btap\b|\bselect\b|\bopen\b|\bsubmit\b|\blaunch\b", re.I), ActionType.CLICK),
]

# Cost weights per action type (arbitrary token-step estimate)
_ACTION_COST: dict[ActionType, float] = {
    ActionType.CLICK: 1.0,
    ActionType.TYPE: 2.0,
    ActionType.SCROLL: 1.0,
    ActionType.KEY_COMBO: 1.5,
    ActionType.SCREENSHOT: 3.0,
    ActionType.WAIT: 0.5,
    ActionType.DRAG: 2.0,
    ActionType.HOVER: 0.8,
}


def _infer_action_type(goal: str) -> ActionType:
    for pattern, action_type in _KEYWORD_RULES:
        if pattern.search(goal):
            return action_type
    return ActionType.CLICK  # safe default


def _extract_target(goal: str, screen_context: dict) -> str:
    """Best-effort target extraction from goal text and screen context."""
    # Prefer an explicit "focused" or "element" hint from context
    if "focused_element" in screen_context:
        return str(screen_context["focused_element"])
    if "target" in screen_context:
        return str(screen_context["target"])
    # Fall back: grab the first quoted string in the goal
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', goal)
    if quoted:
        return quoted[0][0] or quoted[0][1]
    # Last resort: last noun-ish token
    tokens = re.findall(r"\b[A-Za-z_][\w]*\b", goal)
    return tokens[-1] if tokens else "unknown"


# ---------------------------------------------------------------------------
# ActionPlanner
# ---------------------------------------------------------------------------

class ActionPlanner:
    """Rule-based planner: converts a natural-language goal into an ActionPlan."""

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def plan(self, goal: str, screen_context: dict | None = None) -> ActionPlan:
        """Return an ActionPlan for *goal* given optional *screen_context*."""
        if screen_context is None:
            screen_context = {}

        action_type = _infer_action_type(goal)
        target = _extract_target(goal, screen_context)

        params: dict[str, Any] = {}
        if action_type == ActionType.TYPE:
            # Try to pull out the text to type (content after "type"/"enter" keyword)
            m = re.search(r'(?:type|enter|input|write|fill)\s+["\']?(.+?)["\']?$', goal, re.I)
            params["text"] = m.group(1).strip() if m else goal
        elif action_type == ActionType.SCROLL:
            direction = "down"
            if re.search(r"\bup\b", goal, re.I):
                direction = "up"
            elif re.search(r"\bleft\b", goal, re.I):
                direction = "left"
            elif re.search(r"\bright\b", goal, re.I):
                direction = "right"
            params["direction"] = direction
            params["amount"] = screen_context.get("scroll_amount", 3)
        elif action_type == ActionType.WAIT:
            m = re.search(r"(\d+(?:\.\d+)?)\s*(?:s|sec|second|ms|millisecond)?", goal, re.I)
            params["duration"] = float(m.group(1)) if m else 1.0
        elif action_type == ActionType.KEY_COMBO:
            m = re.search(r'(?:press|shortcut|combo)\s+([A-Za-z0-9+_\-\s]+)', goal, re.I)
            params["keys"] = m.group(1).strip() if m else goal
        elif action_type == ActionType.DRAG:
            params["from"] = screen_context.get("drag_from", "source")
            params["to"] = screen_context.get("drag_to", "destination")
        elif action_type == ActionType.SCREENSHOT:
            params["region"] = screen_context.get("region", "full")

        step = PlannedAction(
            action_type=action_type,
            target=target,
            params=params,
            rationale=f"Goal '{goal}' matched rule for {action_type.value}",
        )

        # Confidence is reduced when screen_context is sparse
        confidence = 0.9 if screen_context else 0.6
        return ActionPlan(steps=[step], goal=goal, confidence=confidence)

    def validate_plan(self, plan: ActionPlan) -> list[str]:
        """Return a list of validation error strings (empty = valid)."""
        errors: list[str] = []
        if not plan.goal or not plan.goal.strip():
            errors.append("Plan has an empty goal.")
        if not plan.steps:
            errors.append("Plan has no steps.")
        if not (0.0 <= plan.confidence <= 1.0):
            errors.append(f"Confidence {plan.confidence} is outside [0, 1].")
        for i, step in enumerate(plan.steps):
            if not isinstance(step.action_type, ActionType):
                errors.append(f"Step {i}: action_type is not an ActionType instance.")
            if not step.target:
                errors.append(f"Step {i}: target is empty.")
            if not isinstance(step.params, dict):
                errors.append(f"Step {i}: params must be a dict.")
        return errors

    def estimate_cost(self, plan: ActionPlan) -> float:
        """Return a rough token/step cost estimate for the plan."""
        return sum(_ACTION_COST.get(s.action_type, 1.0) for s in plan.steps)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

COMPUTER_USE_REGISTRY: dict[str, Any] = {}
COMPUTER_USE_REGISTRY["action_planner"] = ActionPlanner
