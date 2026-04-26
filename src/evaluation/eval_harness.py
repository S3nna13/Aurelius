"""Aurelius evaluation harness: runs tasks against model outputs."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalTask:
    task_id: str
    prompt: str
    reference: str
    category: str = "general"
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EvalResult:
    task_id: str
    predicted: str
    reference: str
    correct: bool
    score: float
    latency_ms: float = 0.0


def _exact_match(predicted: str, reference: str) -> float:
    return 1.0 if predicted.strip() == reference.strip() else 0.0


class EvalHarness:
    """Evaluation harness that runs tasks against a predict function."""

    def __init__(self, tasks: list[EvalTask]) -> None:
        self.tasks = list(tasks)

    def run(
        self,
        predict_fn: Callable[[str], str],
        score_fn: Callable[[str, str], float] | None = None,
    ) -> list[EvalResult]:
        """Run all tasks through predict_fn and return EvalResult list."""
        if score_fn is None:
            score_fn = _exact_match

        results: list[EvalResult] = []
        for task in self.tasks:
            t_start = time.perf_counter()
            predicted = predict_fn(task.prompt)
            latency_ms = (time.perf_counter() - t_start) * 1000.0

            score = score_fn(predicted, task.reference)
            correct = score >= 0.5

            results.append(
                EvalResult(
                    task_id=task.task_id,
                    predicted=predicted,
                    reference=task.reference,
                    correct=correct,
                    score=score,
                    latency_ms=latency_ms,
                )
            )
        return results

    def summary(self, results: list[EvalResult]) -> dict:
        """Return aggregate statistics over results."""
        total = len(results)
        correct = sum(1 for r in results if r.correct)
        accuracy = correct / total if total > 0 else 0.0
        mean_score = sum(r.score for r in results) / total if total > 0 else 0.0
        return {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "mean_score": mean_score,
        }

    def filter_by_category(self, results: list[EvalResult], category: str) -> list[EvalResult]:
        """Return only results whose task_id matches tasks in the given category."""
        category_ids = {t.task_id for t in self.tasks if t.category == category}
        return [r for r in results if r.task_id in category_ids]


EVAL_HARNESS_REGISTRY = {"default": EvalHarness}
