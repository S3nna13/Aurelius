"""Curriculum scheduler: difficulty progression, mastery gating, dataset ordering."""

from dataclasses import dataclass
from enum import StrEnum


class DifficultyLevel(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


@dataclass
class CurriculumStage:
    stage_id: int
    difficulty: DifficultyLevel
    min_accuracy: float = 0.7
    max_steps: int = 1000
    description: str = ""


class CurriculumScheduler:
    """Manages curriculum progression through difficulty stages."""

    def __init__(self, stages: list[CurriculumStage] | None = None) -> None:
        if stages is not None:
            self._stages = list(stages)
        else:
            self._stages = [
                CurriculumStage(0, DifficultyLevel.EASY, min_accuracy=0.7, max_steps=1000),
                CurriculumStage(1, DifficultyLevel.MEDIUM, min_accuracy=0.7, max_steps=1000),
                CurriculumStage(2, DifficultyLevel.HARD, min_accuracy=0.7, max_steps=1000),
                CurriculumStage(3, DifficultyLevel.EXPERT, min_accuracy=0.7, max_steps=1000),
            ]
        self._current_idx: int = 0
        self._steps: int = 0

    # ------------------------------------------------------------------
    def current_stage(self) -> CurriculumStage:
        """Return the active CurriculumStage."""
        return self._stages[self._current_idx]

    def advance(self, accuracy: float) -> bool:
        """Advance to next stage if accuracy >= min_accuracy.

        Returns True if advancement occurred, False otherwise (includes
        being already at the last stage).
        """
        if self._current_idx >= len(self._stages) - 1:
            return False
        if accuracy >= self.current_stage().min_accuracy:
            self._current_idx += 1
            self._steps = 0
            return True
        return False

    def step_count(self) -> int:
        """Total steps recorded in the current stage."""
        return self._steps

    def record_step(self, accuracy: float = 0.0) -> None:
        """Increment the step counter for the current stage."""
        self._steps += 1

    def should_advance(self) -> bool:
        """Return True if step_count >= current_stage.max_steps (force advance)."""
        return self._steps >= self.current_stage().max_steps

    def progress(self) -> dict:
        """Return a summary dict of scheduler state."""
        stage = self.current_stage()
        return {
            "stage": self._current_idx,
            "total_stages": len(self._stages),
            "difficulty": stage.difficulty.value,
            "steps": self._steps,
        }
