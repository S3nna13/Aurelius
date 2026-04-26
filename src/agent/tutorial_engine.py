"""Guided step-by-step tutorials / learning paths inspired by build-your-own-x.

Tracks user progress through ordered steps with optional verification checks.
"""

from __future__ import annotations

import datetime
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TutorialEngineError(Exception):
    """Raised when a tutorial operation fails."""


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

CheckFn = Callable[[dict[str, Any]], bool]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TutorialStep:
    """A single step inside a tutorial."""

    step_id: str
    title: str
    description: str
    check_fn: CheckFn | None = None
    hints: list[str] = field(default_factory=list)
    is_checkpoint: bool = False


@dataclass
class Tutorial:
    """A collection of ordered steps forming a learn-by-doing tutorial."""

    tutorial_id: str
    name: str
    description: str
    steps: list[TutorialStep] = field(default_factory=list)
    prerequisites: list[str] = field(default_factory=list)
    estimated_minutes: int = 0


@dataclass
class TutorialProgress:
    """Mutable snapshot of a user's journey through a tutorial."""

    tutorial_id: str
    user_id: str
    completed_steps: set[str] = field(default_factory=set)
    current_step_id: str = ""
    started_at: datetime.datetime | None = None
    completed_at: datetime.datetime | None = None
    attempts_per_step: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TutorialEngine:
    """Manages registration, execution, and progress tracking for tutorials."""

    def __init__(self) -> None:
        self._tutorials: dict[str, Tutorial] = {}
        self._progress: dict[str, TutorialProgress] = {}
        self._hint_indices: dict[str, int] = {}

    # -- internal helpers ----------------------------------------------------

    @staticmethod
    def _progress_key(user_id: str, tutorial_id: str) -> str:
        return f"{user_id}:{tutorial_id}"

    def _is_tutorial_completed(self, user_id: str, tutorial_id: str) -> bool:
        key = self._progress_key(user_id, tutorial_id)
        progress = self._progress.get(key)
        if progress is None:
            return False
        return progress.completed_at is not None

    @staticmethod
    def _calc_progress_percent(progress: TutorialProgress, tutorial: Tutorial) -> float:
        if not tutorial.steps:
            return 0.0
        return len(progress.completed_steps) / len(tutorial.steps) * 100.0

    # -- public API ----------------------------------------------------------

    def register_tutorial(self, tutorial: Tutorial) -> None:
        """Add a tutorial to the engine."""
        if tutorial.tutorial_id in self._tutorials:
            raise TutorialEngineError(
                f"Tutorial '{tutorial.tutorial_id}' is already registered."
            )
        self._tutorials[tutorial.tutorial_id] = tutorial

    def start_tutorial(self, user_id: str, tutorial_id: str) -> TutorialProgress:
        """Begin a tutorial for a user."""
        if tutorial_id not in self._tutorials:
            raise TutorialEngineError(f"Tutorial '{tutorial_id}' not found.")
        tutorial = self._tutorials[tutorial_id]
        if not tutorial.steps:
            raise TutorialEngineError(f"Tutorial '{tutorial_id}' has no steps.")

        key = self._progress_key(user_id, tutorial_id)
        if key in self._progress:
            raise TutorialEngineError(
                f"User '{user_id}' has already started tutorial '{tutorial_id}'."
            )

        now = datetime.datetime.now(datetime.UTC)
        progress = TutorialProgress(
            tutorial_id=tutorial_id,
            user_id=user_id,
            current_step_id=tutorial.steps[0].step_id,
            started_at=now,
        )
        self._progress[key] = progress
        return progress

    def get_current_step(self, user_id: str, tutorial_id: str) -> TutorialStep:
        """Return the current step for the user's in-progress tutorial."""
        progress = self.get_progress(user_id, tutorial_id)
        tutorial = self._tutorials.get(tutorial_id)
        if tutorial is None:
            raise TutorialEngineError(f"Tutorial '{tutorial_id}' not found.")
        for step in tutorial.steps:
            if step.step_id == progress.current_step_id:
                return step
        raise TutorialEngineError(
            f"Current step '{progress.current_step_id}' not found in tutorial "
            f"'{tutorial_id}'."
        )

    def submit_step(
        self, user_id: str, tutorial_id: str, result: dict[str, Any]
    ) -> dict[str, Any]:
        """Check if the current step is complete and advance if so.

        Returns a dict with keys:
            success, next_step, hints, progress_percent
        """
        progress = self.get_progress(user_id, tutorial_id)
        tutorial = self._tutorials.get(tutorial_id)
        if tutorial is None:
            raise TutorialEngineError(f"Tutorial '{tutorial_id}' not found.")
        if progress.completed_at is not None:
            raise TutorialEngineError(
                f"Tutorial '{tutorial_id}' is already completed for user '{user_id}'."
            )

        step = self.get_current_step(user_id, tutorial_id)
        progress.attempts_per_step[step.step_id] = (
            progress.attempts_per_step.get(step.step_id, 0) + 1
        )

        if step.check_fn is not None:
            success = step.check_fn(result)
        else:
            success = True

        if not success:
            return {
                "success": False,
                "next_step": None,
                "hints": step.hints,
                "progress_percent": self._calc_progress_percent(progress, tutorial),
            }

        progress.completed_steps.add(step.step_id)
        step_index = next(
            (i for i, s in enumerate(tutorial.steps) if s.step_id == step.step_id), -1
        )
        next_step: TutorialStep | None = None
        if step_index + 1 < len(tutorial.steps):
            next_step_id = tutorial.steps[step_index + 1].step_id
            progress.current_step_id = next_step_id
            next_step = tutorial.steps[step_index + 1]
        else:
            progress.completed_at = datetime.datetime.now(datetime.UTC)

        return {
            "success": True,
            "next_step": next_step,
            "hints": step.hints,
            "progress_percent": self._calc_progress_percent(progress, tutorial),
        }

    def request_hint(self, user_id: str, tutorial_id: str) -> str:
        """Return the next available hint for the current step, cycling through."""
        step = self.get_current_step(user_id, tutorial_id)
        if not step.hints:
            return "No hints available for this step."
        hint_key = f"{user_id}:{tutorial_id}:{step.step_id}"
        idx = self._hint_indices.get(hint_key, 0)
        hint = step.hints[idx % len(step.hints)]
        self._hint_indices[hint_key] = (idx + 1) % len(step.hints)
        return hint

    def get_progress(self, user_id: str, tutorial_id: str) -> TutorialProgress:
        """Return the current progress for a user in a tutorial."""
        key = self._progress_key(user_id, tutorial_id)
        if key not in self._progress:
            raise TutorialEngineError(
                f"No progress found for user '{user_id}' in tutorial '{tutorial_id}'."
            )
        return self._progress[key]

    def list_tutorials(self) -> list[Tutorial]:
        """List all registered tutorials."""
        return list(self._tutorials.values())

    def list_available_tutorials(self, user_id: str) -> list[Tutorial]:
        """Tutorials whose prerequisites are fully met by the user."""
        available: list[Tutorial] = []
        for tutorial in self._tutorials.values():
            if all(
                self._is_tutorial_completed(user_id, prereq)
                for prereq in tutorial.prerequisites
            ):
                available.append(tutorial)
        return available

    def reset_progress(self, user_id: str, tutorial_id: str) -> None:
        """Reset a user's progress for a tutorial."""
        key = self._progress_key(user_id, tutorial_id)
        if key not in self._progress:
            raise TutorialEngineError(
                f"No progress found for user '{user_id}' in tutorial '{tutorial_id}'."
            )
        del self._progress[key]
        prefix = f"{user_id}:{tutorial_id}:"
        for hint_key in list(self._hint_indices):
            if hint_key.startswith(prefix):
                del self._hint_indices[hint_key]

    def stats(self) -> dict[str, Any]:
        """Return aggregate statistics about tutorials and completions."""
        total_tutorials = len(self._tutorials)
        total_completions = sum(
            1 for p in self._progress.values() if p.completed_at is not None
        )
        avg_completion_rate = 0.0
        if self._progress:
            rates = [
                len(p.completed_steps) / len(self._tutorials[p.tutorial_id].steps) * 100
                for p in self._progress.values()
                if p.tutorial_id in self._tutorials
                and self._tutorials[p.tutorial_id].steps
            ]
            if rates:
                avg_completion_rate = sum(rates) / len(rates)
        return {
            "total_tutorials": total_tutorials,
            "total_completions": total_completions,
            "avg_completion_rate": avg_completion_rate,
        }


# ---------------------------------------------------------------------------
# Singleton & Registry
# ---------------------------------------------------------------------------

DEFAULT_TUTORIAL_ENGINE = TutorialEngine()
TUTORIAL_ENGINE_REGISTRY: dict[str, TutorialEngine] = {
    "default": DEFAULT_TUTORIAL_ENGINE,
}
