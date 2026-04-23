"""MATH dataset benchmark: parsing, symbolic verification, category scoring."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class MathCategory(str, Enum):
    ALGEBRA = "algebra"
    GEOMETRY = "geometry"
    NUMBER_THEORY = "number_theory"
    COUNTING = "counting"
    PRECALCULUS = "precalculus"
    INTERMEDIATE_ALGEBRA = "intermediate_algebra"
    PREALGEBRA = "prealgebra"


@dataclass
class MathProblem:
    problem_id: str
    category: MathCategory
    difficulty: int  # 1-5
    problem: str
    answer: str


class MathAnswer:
    _BOXED_RE = re.compile(r'\\boxed\{([^}]*)\}')

    @staticmethod
    def normalize(raw: str) -> str:
        """Strip whitespace, lowercase, remove $ and \\boxed{} wrapper."""
        s = raw.strip()
        # Remove \boxed{...} — extract group 1 (outermost match)
        m = MathAnswer._BOXED_RE.search(s)
        if m:
            s = m.group(1)
        # Remove dollar signs
        s = s.replace('$', '')
        return s.strip().lower()

    @staticmethod
    def check_exact(predicted: str, gold: str) -> bool:
        """Normalize both strings and compare for equality."""
        return MathAnswer.normalize(predicted) == MathAnswer.normalize(gold)

    @staticmethod
    def check_numeric(predicted: str, gold: str) -> bool:
        """Try float conversion; return abs(a-b) < 1e-6, False on failure."""
        try:
            a = float(MathAnswer.normalize(predicted))
            b = float(MathAnswer.normalize(gold))
            return abs(a - b) < 1e-6
        except (ValueError, TypeError):
            return False


# --- Default stub problems (6 total: 2 per category for ALGEBRA/GEOMETRY/NUMBER_THEORY) ---
_DEFAULT_PROBLEMS: list[MathProblem] = [
    MathProblem(
        problem_id="alg_001",
        category=MathCategory.ALGEBRA,
        difficulty=1,
        problem="Solve for x: x + 3 = 7",
        answer="4",
    ),
    MathProblem(
        problem_id="alg_002",
        category=MathCategory.ALGEBRA,
        difficulty=2,
        problem="Solve for x: 2x - 5 = 9",
        answer="7",
    ),
    MathProblem(
        problem_id="geo_001",
        category=MathCategory.GEOMETRY,
        difficulty=1,
        problem="What is the area of a rectangle with length 4 and width 3?",
        answer="12",
    ),
    MathProblem(
        problem_id="geo_002",
        category=MathCategory.GEOMETRY,
        difficulty=2,
        problem="What is the perimeter of a square with side length 5?",
        answer="20",
    ),
    MathProblem(
        problem_id="nt_001",
        category=MathCategory.NUMBER_THEORY,
        difficulty=2,
        problem="What is the greatest common divisor of 12 and 8?",
        answer="4",
    ),
    MathProblem(
        problem_id="nt_002",
        category=MathCategory.NUMBER_THEORY,
        difficulty=3,
        problem="What is the least common multiple of 4 and 6?",
        answer="12",
    ),
]


class MathBenchmark:
    def __init__(self, problems: Optional[list[MathProblem]] = None) -> None:
        self.problems: list[MathProblem] = problems if problems is not None else list(_DEFAULT_PROBLEMS)

    def problem_ids(self) -> list[str]:
        return [p.problem_id for p in self.problems]

    def evaluate(self, predictions: dict[str, str]) -> dict:
        """Evaluate predictions against stored problems.

        Returns:
            {
                "total": int,
                "correct": int,
                "accuracy": float,
                "by_category": {cat_value: {"correct": int, "total": int}},
                "by_difficulty": {diff: {"correct": int, "total": int}},
            }
        """
        total = len(self.problems)
        correct = 0

        by_category: dict[str, dict[str, int]] = {}
        by_difficulty: dict[int, dict[str, int]] = {}

        for p in self.problems:
            cat_key = p.category.value
            diff_key = p.difficulty

            if cat_key not in by_category:
                by_category[cat_key] = {"correct": 0, "total": 0}
            if diff_key not in by_difficulty:
                by_difficulty[diff_key] = {"correct": 0, "total": 0}

            by_category[cat_key]["total"] += 1
            by_difficulty[diff_key]["total"] += 1

            predicted = predictions.get(p.problem_id, "")
            is_correct = (
                MathAnswer.check_exact(predicted, p.answer)
                or MathAnswer.check_numeric(predicted, p.answer)
            )
            if is_correct:
                correct += 1
                by_category[cat_key]["correct"] += 1
                by_difficulty[diff_key]["correct"] += 1

        accuracy = correct / total if total > 0 else 0.0

        return {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "by_category": by_category,
            "by_difficulty": by_difficulty,
        }


# --- Register in BENCHMARK_REGISTRY ---
from src.eval import BENCHMARK_REGISTRY  # noqa: E402

BENCHMARK_REGISTRY.setdefault("math", MathBenchmark())
