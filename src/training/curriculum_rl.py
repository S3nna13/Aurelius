"""
Curriculum RL: difficulty-adaptive task sampling for RL training.

Maintains per-task difficulty estimates based on recent model accuracy,
then samples tasks where the model is in the "learning zone" — neither
too easy (>85% correct) nor too hard (<15% correct).

Classes:
    TaskDifficulty       — per-task state (difficulty, EMA accuracy, counters)
    CurriculumRLConfig   — hyperparameters
    CurriculumRLSampler  — register tasks, update accuracy, sample tasks
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TaskDifficulty:
    """Per-task state tracked by the curriculum sampler."""

    task_id: str
    difficulty: float  # 0 = easiest, 1 = hardest (static estimate)
    recent_accuracy: float  # EMA of correct-rate, initialised at 0.5
    n_attempts: int = 0
    n_correct: int = 0
    ema_alpha: float = 0.1  # EMA smoothing factor for accuracy updates


@dataclass
class CurriculumRLConfig:
    """Hyperparameters for the CurriculumRLSampler."""

    easy_threshold: float = 0.85  # skip tasks with accuracy > this
    hard_threshold: float = 0.15  # skip tasks with accuracy < this
    exploration_prob: float = 0.1  # probability of uniform random exploration
    temperature: float = 1.0  # softmax temperature for difficulty weighting
    min_attempts_before_skip: int = 5  # don't skip a task until it has >= this attempts


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class CurriculumRLSampler:
    """
    Difficulty-adaptive task sampler for curriculum RL training.

    Usage::

        cfg = CurriculumRLConfig()
        sampler = CurriculumRLSampler(cfg)
        sampler.register_task("task_0", difficulty=0.2)
        sampler.register_task("task_1", difficulty=0.7)
        sampler.update("task_0", is_correct=True)
        task_ids = sampler.sample(n=4, rng_seed=42)
    """

    def __init__(self, config: CurriculumRLConfig | None = None) -> None:
        self.config = config if config is not None else CurriculumRLConfig()
        self._tasks: dict[str, TaskDifficulty] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_task(self, task_id: str, difficulty: float) -> None:
        """Register a new task with its static difficulty estimate.

        Args:
            task_id: Unique identifier for the task.
            difficulty: Static difficulty in [0, 1] (0=easiest, 1=hardest).

        Raises:
            ValueError: If a task with this ``task_id`` is already registered.
        """
        if task_id in self._tasks:
            raise ValueError(f"Task '{task_id}' is already registered.")
        self._tasks[task_id] = TaskDifficulty(
            task_id=task_id,
            difficulty=float(difficulty),
            recent_accuracy=0.5,
        )

    def update(self, task_id: str, is_correct: bool) -> None:
        """Update a task's EMA accuracy and increment attempt counters.

        Args:
            task_id: The task that was just attempted.
            is_correct: Whether the model answered correctly.

        Raises:
            KeyError: If ``task_id`` has not been registered.
        """
        if task_id not in self._tasks:
            raise KeyError(f"Task '{task_id}' is not registered.")
        td = self._tasks[task_id]
        td.n_attempts += 1
        if is_correct:
            td.n_correct += 1
        # EMA update: new = alpha * observation + (1 - alpha) * old
        observation = 1.0 if is_correct else 0.0
        td.recent_accuracy = td.ema_alpha * observation + (1.0 - td.ema_alpha) * td.recent_accuracy

    def in_learning_zone(self, task_id: str) -> bool:
        """Return True if the task is in the learning zone.

        A task is in the learning zone when:
        - It has fewer than ``min_attempts_before_skip`` attempts (always include), OR
        - Its ``recent_accuracy`` is in [hard_threshold, easy_threshold].

        Args:
            task_id: Task to check.
        """
        td = self._tasks[task_id]
        cfg = self.config
        if td.n_attempts < cfg.min_attempts_before_skip:
            return True
        return cfg.hard_threshold <= td.recent_accuracy <= cfg.easy_threshold

    def sample(self, n: int = 1, rng_seed: int | None = None) -> list[str]:
        """Sample ``n`` task IDs according to the curriculum policy.

        With probability ``exploration_prob`` draw uniformly at random from all
        registered tasks; otherwise draw from tasks in the learning zone,
        weighted by softmax(difficulty / temperature).  Sampling is with
        replacement.

        Args:
            n: Number of task IDs to return.
            rng_seed: Optional seed for the Python ``random`` module, used
                      for reproducibility.

        Returns:
            List of ``n`` task IDs.
        """
        if not self._tasks:
            return []

        rng = random.Random(rng_seed)
        task_ids = list(self._tasks.keys())

        result: list[str] = []
        for _ in range(n):
            if rng.random() < self.config.exploration_prob:
                # Exploration: sample uniformly from all tasks
                result.append(rng.choice(task_ids))
            else:
                # Exploitation: sample from learning-zone tasks, difficulty-weighted
                zone_ids = [tid for tid in task_ids if self.in_learning_zone(tid)]
                if not zone_ids:
                    # Fall back to uniform over all tasks
                    result.append(rng.choice(task_ids))
                else:
                    result.append(self._softmax_sample(zone_ids, rng))

        return result

    def statistics(self) -> dict:
        """Return aggregate statistics across all registered tasks.

        Returns:
            dict with keys:
                - ``n_tasks``: total number of registered tasks
                - ``n_in_zone``: tasks currently in the learning zone
                - ``mean_accuracy``: mean recent_accuracy across all tasks
                - ``n_easy``: tasks above easy_threshold (and enough attempts)
                - ``n_hard``: tasks below hard_threshold (and enough attempts)
        """
        cfg = self.config
        tasks = list(self._tasks.values())
        n_tasks = len(tasks)

        if n_tasks == 0:
            return {
                "n_tasks": 0,
                "n_in_zone": 0,
                "mean_accuracy": 0.0,
                "n_easy": 0,
                "n_hard": 0,
            }

        n_in_zone = sum(1 for td in tasks if self.in_learning_zone(td.task_id))
        mean_accuracy = sum(td.recent_accuracy for td in tasks) / n_tasks

        def _qualified(td: TaskDifficulty) -> bool:
            return td.n_attempts >= cfg.min_attempts_before_skip

        n_easy = sum(
            1 for td in tasks if _qualified(td) and td.recent_accuracy > cfg.easy_threshold
        )
        n_hard = sum(
            1 for td in tasks if _qualified(td) and td.recent_accuracy < cfg.hard_threshold
        )

        return {
            "n_tasks": n_tasks,
            "n_in_zone": n_in_zone,
            "mean_accuracy": mean_accuracy,
            "n_easy": n_easy,
            "n_hard": n_hard,
        }

    def task_summary(self, task_id: str) -> dict:
        """Return per-task statistics.

        Args:
            task_id: The task to summarise.

        Returns:
            dict with keys: ``accuracy``, ``attempts``,
            ``in_learning_zone``, ``difficulty``.
        """
        td = self._tasks[task_id]
        return {
            "accuracy": td.recent_accuracy,
            "attempts": td.n_attempts,
            "in_learning_zone": self.in_learning_zone(task_id),
            "difficulty": td.difficulty,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _softmax_sample(self, task_ids: list[str], rng: random.Random) -> str:
        """Sample one task ID using softmax-weighted difficulty."""
        temp = self.config.temperature
        logits = [self._tasks[tid].difficulty / temp for tid in task_ids]

        # Numerically stable softmax
        max_logit = max(logits)
        exp_logits = [math.exp(line - max_logit) for line in logits]
        total = sum(exp_logits)
        weights = [e / total for e in exp_logits]

        # Weighted random selection
        r = rng.random()
        cumulative = 0.0
        for tid, w in zip(task_ids, weights):
            cumulative += w
            if r <= cumulative:
                return tid
        return task_ids[-1]  # floating-point safety fallback
