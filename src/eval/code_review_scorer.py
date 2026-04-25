"""Code review quality scorer: rubric dimensions, weighted scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Optional


class ReviewDimension(str, Enum):
    CORRECTNESS = "correctness"
    STYLE = "style"
    SECURITY = "security"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"


@dataclass
class DimensionScore:
    dimension: ReviewDimension
    score: float  # 0.0 - 1.0
    rationale: str = ""


class CodeReviewRubric:
    DEFAULT_WEIGHTS: ClassVar[dict[ReviewDimension, float]] = {
        ReviewDimension.CORRECTNESS: 0.35,
        ReviewDimension.SECURITY: 0.25,
        ReviewDimension.PERFORMANCE: 0.20,
        ReviewDimension.STYLE: 0.10,
        ReviewDimension.MAINTAINABILITY: 0.10,
    }

    def __init__(self, weights: Optional[dict[ReviewDimension, float]] = None) -> None:
        raw = weights if weights is not None else dict(self.DEFAULT_WEIGHTS)
        # Normalize to sum=1.0
        total = sum(raw.values())
        if total == 0:
            raise ValueError("Weights must sum to a positive number.")
        self.weights: dict[ReviewDimension, float] = {
            dim: w / total for dim, w in raw.items()
        }

    def weighted_score(self, scores: list[DimensionScore]) -> float:
        """Weighted average over provided dimensions; skip missing dimensions."""
        provided_dims = {ds.dimension for ds in scores}
        # Sum of weights for provided dimensions only
        weight_sum = sum(
            self.weights.get(dim, 0.0) for dim in provided_dims
        )
        if weight_sum == 0:
            return 0.0
        total = sum(
            ds.score * self.weights.get(ds.dimension, 0.0)
            for ds in scores
        )
        return total / weight_sum

    def grade(self, weighted: float) -> str:
        """Convert weighted score to letter grade."""
        if weighted >= 0.9:
            return "A"
        elif weighted >= 0.75:
            return "B"
        elif weighted >= 0.6:
            return "C"
        elif weighted >= 0.45:
            return "D"
        else:
            return "F"


class CodeReviewScorer:
    def __init__(self, rubric: Optional[CodeReviewRubric] = None) -> None:
        self.rubric = rubric if rubric is not None else CodeReviewRubric()

    def score_review(self, dimension_scores: list[DimensionScore]) -> dict:
        """Score a code review given per-dimension scores.

        Returns:
            {
                "weighted_score": float,
                "grade": str,
                "dimension_scores": {dim.value: score, ...},
                "missing_dimensions": [dim.value, ...],
            }
        """
        provided_dims = {ds.dimension for ds in dimension_scores}
        all_dims = set(ReviewDimension)
        missing = sorted(
            d.value for d in (all_dims - provided_dims)
        )

        ws = self.rubric.weighted_score(dimension_scores)
        grade = self.rubric.grade(ws)

        dim_scores_dict = {ds.dimension.value: ds.score for ds in dimension_scores}

        return {
            "weighted_score": ws,
            "grade": grade,
            "dimension_scores": dim_scores_dict,
            "missing_dimensions": missing,
        }


# --- Registry ---
CODE_REVIEW_SCORER_REGISTRY: dict[str, CodeReviewScorer] = {
    "default": CodeReviewScorer(),
    "security_focused": CodeReviewScorer(
        CodeReviewRubric(
            {
                ReviewDimension.SECURITY: 0.5,
                ReviewDimension.CORRECTNESS: 0.3,
                ReviewDimension.PERFORMANCE: 0.1,
                ReviewDimension.STYLE: 0.05,
                ReviewDimension.MAINTAINABILITY: 0.05,
            }
        )
    ),
}
