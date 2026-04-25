"""Aurelius benchmark runner: runs multiple benchmark suites and aggregates results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class BenchmarkSuite:
    name: str
    tasks: list[dict]
    metric: str = "accuracy"


@dataclass(frozen=True)
class BenchmarkResult:
    suite_name: str
    metric: str
    score: float
    num_tasks: int
    details: list[dict]


class BenchmarkRunner:
    """Runs multiple benchmark suites and aggregates results."""

    def __init__(self, suites: list[BenchmarkSuite]) -> None:
        self.suites = list(suites)

    def run_suite(
        self, suite: BenchmarkSuite, predict_fn: Callable[[str], str]
    ) -> BenchmarkResult:
        """Run all tasks in a suite and compute exact-match accuracy."""
        details: list[dict] = []
        correct_count = 0

        for task in suite.tasks:
            prompt = task["prompt"]
            reference = task["reference"]
            predicted = predict_fn(prompt)
            is_correct = predicted.strip() == reference.strip()
            if is_correct:
                correct_count += 1
            details.append(
                {
                    "prompt": prompt,
                    "predicted": predicted,
                    "correct": is_correct,
                }
            )

        num_tasks = len(suite.tasks)
        score = correct_count / num_tasks if num_tasks > 0 else 0.0

        return BenchmarkResult(
            suite_name=suite.name,
            metric=suite.metric,
            score=score,
            num_tasks=num_tasks,
            details=details,
        )

    def run_all(
        self, predict_fn: Callable[[str], str]
    ) -> list[BenchmarkResult]:
        """Run all registered suites."""
        return [self.run_suite(suite, predict_fn) for suite in self.suites]

    def leaderboard(self, results: list[BenchmarkResult]) -> list[dict]:
        """Return results sorted descending by score with 1-based ranks."""
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
        return [
            {"rank": i + 1, "suite": r.suite_name, "score": r.score}
            for i, r in enumerate(sorted_results)
        ]


BENCHMARK_RUNNER_REGISTRY = {"default": BenchmarkRunner}
