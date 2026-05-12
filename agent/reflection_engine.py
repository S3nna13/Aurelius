"""Reflection engine module for the Aurelius agent surface.

Performs rule-based self-reflection over past agent actions to improve
future behavior, goal alignment, efficiency, safety, and coherence.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ReflectionType(Enum):
    SELF_CRITIQUE = "self_critique"
    GOAL_ALIGNMENT = "goal_alignment"
    EFFICIENCY = "efficiency"
    SAFETY = "safety"
    COHERENCE = "coherence"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Reflection:
    """Immutable result of a single reflection pass."""

    reflection_id: str
    reflection_type: ReflectionType
    input_summary: str
    critique: str
    suggestions: list[str]
    confidence: float

    @classmethod
    def create(
        cls,
        reflection_type: ReflectionType,
        input_summary: str,
        critique: str,
        suggestions: list[str],
        confidence: float,
        reflection_id: str | None = None,
    ) -> Reflection:
        """Factory that auto-generates reflection_id when not provided."""
        rid = reflection_id if reflection_id is not None else uuid.uuid4().hex[:8]
        return cls(
            reflection_id=rid,
            reflection_type=reflection_type,
            input_summary=input_summary,
            critique=critique,
            suggestions=suggestions,
            confidence=confidence,
        )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ReflectionEngine:
    """Produce rule-based reflections over past agent actions."""

    def __init__(self) -> None:
        self._history: list[Reflection] = []

    # ------------------------------------------------------------------
    # Core reflection
    # ------------------------------------------------------------------

    def reflect(
        self,
        reflection_type: ReflectionType,
        actions: list[str],
        goal: str = "",
    ) -> Reflection:
        """Run a rule-based reflection and store it in history."""
        n = len(actions)

        # Build input summary: first three actions joined, ellipsis if more
        if actions:
            input_summary = "; ".join(actions[:3])
            if n > 3:
                input_summary += "..."
        else:
            input_summary = ""

        # Confidence grows linearly with action count, capped at 1.0
        confidence = min(1.0, n / 10.0) if actions else 0.0

        # Rule-based critique and suggestions per reflection type
        if reflection_type is ReflectionType.SELF_CRITIQUE:
            critique = f"Completed {n} actions. Review for redundancy."
            suggestions = (
                ["Consolidate duplicate steps"] if n > 5 else ["Continue current approach"]
            )

        elif reflection_type is ReflectionType.GOAL_ALIGNMENT:
            critique = f"Goal: '{goal[:50]}'. Actions taken: {n}."
            suggestions = ["Verify all actions contribute to goal"]

        elif reflection_type is ReflectionType.EFFICIENCY:
            critique = f"Efficiency check: {n} steps."
            suggestions = ["Reduce steps" if n > 3 else "Optimal step count"]

        elif reflection_type is ReflectionType.SAFETY:
            critique = "Safety review of actions."
            suggestions = ["Verify no destructive operations"]

        else:  # COHERENCE
            critique = f"Coherence check across {n} actions."
            suggestions = ["Ensure logical flow"]

        reflection = Reflection.create(
            reflection_type=reflection_type,
            input_summary=input_summary,
            critique=critique,
            suggestions=suggestions,
            confidence=confidence,
        )
        self._history.append(reflection)
        return reflection

    # ------------------------------------------------------------------
    # History access
    # ------------------------------------------------------------------

    def history(self) -> list[Reflection]:
        """Return all reflections in chronological order."""
        return list(self._history)

    def latest(
        self,
        reflection_type: ReflectionType | None = None,
    ) -> Reflection | None:
        """Return the most recent reflection, optionally filtered by type."""
        if reflection_type is None:
            return self._history[-1] if self._history else None
        for ref in reversed(self._history):
            if ref.reflection_type is reflection_type:
                return ref
        return None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REFLECTION_ENGINE_REGISTRY: dict[str, Any] = {
    "default": ReflectionEngine,
}
