"""Tests for src/eval/code_review_scorer.py — ~45 tests."""

import pytest

from src.eval.code_review_scorer import (
    ReviewDimension,
    DimensionScore,
    CodeReviewRubric,
    CodeReviewScorer,
    CODE_REVIEW_SCORER_REGISTRY,
)


# ---------------------------------------------------------------------------
# ReviewDimension enum
# ---------------------------------------------------------------------------


def test_review_dimension_correctness():
    assert ReviewDimension.CORRECTNESS == "correctness"
    assert ReviewDimension.CORRECTNESS.value == "correctness"


def test_review_dimension_style():
    assert ReviewDimension.STYLE == "style"


def test_review_dimension_security():
    assert ReviewDimension.SECURITY == "security"


def test_review_dimension_performance():
    assert ReviewDimension.PERFORMANCE == "performance"


def test_review_dimension_maintainability():
    assert ReviewDimension.MAINTAINABILITY == "maintainability"


def test_review_dimension_all_five():
    assert len(ReviewDimension) == 5


# ---------------------------------------------------------------------------
# DimensionScore dataclass
# ---------------------------------------------------------------------------


def test_dimension_score_fields():
    ds = DimensionScore(
        dimension=ReviewDimension.CORRECTNESS,
        score=0.8,
        rationale="Looks good",
    )
    assert ds.dimension == ReviewDimension.CORRECTNESS
    assert ds.score == pytest.approx(0.8)
    assert ds.rationale == "Looks good"


def test_dimension_score_default_rationale():
    ds = DimensionScore(dimension=ReviewDimension.STYLE, score=0.5)
    assert ds.rationale == ""


def test_dimension_score_zero():
    ds = DimensionScore(dimension=ReviewDimension.SECURITY, score=0.0)
    assert ds.score == 0.0


def test_dimension_score_one():
    ds = DimensionScore(dimension=ReviewDimension.PERFORMANCE, score=1.0)
    assert ds.score == 1.0


# ---------------------------------------------------------------------------
# CodeReviewRubric — DEFAULT_WEIGHTS
# ---------------------------------------------------------------------------


def test_default_weights_sum_to_one():
    total = sum(CodeReviewRubric.DEFAULT_WEIGHTS.values())
    assert total == pytest.approx(1.0)


def test_default_weights_correctness():
    assert CodeReviewRubric.DEFAULT_WEIGHTS[ReviewDimension.CORRECTNESS] == pytest.approx(0.35)


def test_default_weights_security():
    assert CodeReviewRubric.DEFAULT_WEIGHTS[ReviewDimension.SECURITY] == pytest.approx(0.25)


def test_default_weights_performance():
    assert CodeReviewRubric.DEFAULT_WEIGHTS[ReviewDimension.PERFORMANCE] == pytest.approx(0.20)


def test_default_weights_style():
    assert CodeReviewRubric.DEFAULT_WEIGHTS[ReviewDimension.STYLE] == pytest.approx(0.10)


def test_default_weights_maintainability():
    assert CodeReviewRubric.DEFAULT_WEIGHTS[ReviewDimension.MAINTAINABILITY] == pytest.approx(0.10)


def test_default_rubric_uses_default_weights():
    rubric = CodeReviewRubric()
    assert rubric.weights[ReviewDimension.CORRECTNESS] == pytest.approx(0.35)


# ---------------------------------------------------------------------------
# CodeReviewRubric — custom weights normalization
# ---------------------------------------------------------------------------


def test_custom_weights_normalized_to_one():
    rubric = CodeReviewRubric(
        {
            ReviewDimension.CORRECTNESS: 2.0,
            ReviewDimension.SECURITY: 2.0,
            ReviewDimension.PERFORMANCE: 1.0,
        }
    )
    total = sum(rubric.weights.values())
    assert total == pytest.approx(1.0)


def test_custom_weights_proportions_preserved():
    rubric = CodeReviewRubric(
        {
            ReviewDimension.CORRECTNESS: 3.0,
            ReviewDimension.SECURITY: 1.0,
        }
    )
    assert rubric.weights[ReviewDimension.CORRECTNESS] == pytest.approx(0.75)
    assert rubric.weights[ReviewDimension.SECURITY] == pytest.approx(0.25)


def test_custom_weights_already_normalized():
    rubric = CodeReviewRubric(
        {
            ReviewDimension.CORRECTNESS: 0.6,
            ReviewDimension.STYLE: 0.4,
        }
    )
    assert sum(rubric.weights.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# CodeReviewRubric — weighted_score
# ---------------------------------------------------------------------------


def test_weighted_score_all_five_dimensions_perfect():
    rubric = CodeReviewRubric()
    scores = [
        DimensionScore(ReviewDimension.CORRECTNESS, 1.0),
        DimensionScore(ReviewDimension.SECURITY, 1.0),
        DimensionScore(ReviewDimension.PERFORMANCE, 1.0),
        DimensionScore(ReviewDimension.STYLE, 1.0),
        DimensionScore(ReviewDimension.MAINTAINABILITY, 1.0),
    ]
    ws = rubric.weighted_score(scores)
    assert ws == pytest.approx(1.0)


def test_weighted_score_all_five_dimensions_zero():
    rubric = CodeReviewRubric()
    scores = [
        DimensionScore(ReviewDimension.CORRECTNESS, 0.0),
        DimensionScore(ReviewDimension.SECURITY, 0.0),
        DimensionScore(ReviewDimension.PERFORMANCE, 0.0),
        DimensionScore(ReviewDimension.STYLE, 0.0),
        DimensionScore(ReviewDimension.MAINTAINABILITY, 0.0),
    ]
    ws = rubric.weighted_score(scores)
    assert ws == pytest.approx(0.0)


def test_weighted_score_partial_dimensions_only():
    # Provide only CORRECTNESS and SECURITY — weights re-normalized over those two
    rubric = CodeReviewRubric()
    scores = [
        DimensionScore(ReviewDimension.CORRECTNESS, 1.0),
        DimensionScore(ReviewDimension.SECURITY, 0.0),
    ]
    ws = rubric.weighted_score(scores)
    # CORRECTNESS=0.35, SECURITY=0.25, sub-total=0.60
    # weighted_score = (1.0*0.35 + 0.0*0.25) / 0.60 = 0.35/0.60
    assert ws == pytest.approx(0.35 / 0.60, rel=1e-5)


def test_weighted_score_single_dimension():
    rubric = CodeReviewRubric()
    scores = [DimensionScore(ReviewDimension.CORRECTNESS, 0.8)]
    ws = rubric.weighted_score(scores)
    # Only one dimension provided, its weight_sum = 0.35; score = 0.8*0.35/0.35 = 0.8
    assert ws == pytest.approx(0.8)


def test_weighted_score_empty_returns_zero():
    rubric = CodeReviewRubric()
    assert rubric.weighted_score([]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# CodeReviewRubric — grade thresholds
# ---------------------------------------------------------------------------


def test_grade_A():
    rubric = CodeReviewRubric()
    assert rubric.grade(0.95) == "A"


def test_grade_A_exact_boundary():
    rubric = CodeReviewRubric()
    assert rubric.grade(0.9) == "A"


def test_grade_B():
    rubric = CodeReviewRubric()
    assert rubric.grade(0.80) == "B"


def test_grade_B_lower_boundary():
    rubric = CodeReviewRubric()
    assert rubric.grade(0.75) == "B"


def test_grade_C():
    rubric = CodeReviewRubric()
    assert rubric.grade(0.65) == "C"


def test_grade_C_lower_boundary():
    rubric = CodeReviewRubric()
    assert rubric.grade(0.6) == "C"


def test_grade_D():
    rubric = CodeReviewRubric()
    assert rubric.grade(0.50) == "D"


def test_grade_D_lower_boundary():
    rubric = CodeReviewRubric()
    assert rubric.grade(0.45) == "D"


def test_grade_F():
    rubric = CodeReviewRubric()
    assert rubric.grade(0.30) == "F"


def test_grade_F_zero():
    rubric = CodeReviewRubric()
    assert rubric.grade(0.0) == "F"


# ---------------------------------------------------------------------------
# CodeReviewScorer.score_review
# ---------------------------------------------------------------------------


def test_score_review_returns_dict():
    scorer = CodeReviewScorer()
    scores = [DimensionScore(ReviewDimension.CORRECTNESS, 0.9)]
    result = scorer.score_review(scores)
    assert isinstance(result, dict)


def test_score_review_has_required_keys():
    scorer = CodeReviewScorer()
    scores = [DimensionScore(ReviewDimension.CORRECTNESS, 0.9)]
    result = scorer.score_review(scores)
    assert "weighted_score" in result
    assert "grade" in result
    assert "dimension_scores" in result
    assert "missing_dimensions" in result


def test_score_review_dimension_scores_dict_values():
    scorer = CodeReviewScorer()
    scores = [DimensionScore(ReviewDimension.CORRECTNESS, 0.8)]
    result = scorer.score_review(scores)
    assert result["dimension_scores"]["correctness"] == pytest.approx(0.8)


def test_score_review_missing_dimensions_lists_others():
    scorer = CodeReviewScorer()
    scores = [DimensionScore(ReviewDimension.CORRECTNESS, 0.9)]
    result = scorer.score_review(scores)
    missing = result["missing_dimensions"]
    # All other dimensions should be missing
    assert "style" in missing
    assert "security" in missing
    assert "performance" in missing
    assert "maintainability" in missing
    assert "correctness" not in missing


def test_score_review_no_missing_when_all_provided():
    scorer = CodeReviewScorer()
    scores = [
        DimensionScore(ReviewDimension.CORRECTNESS, 1.0),
        DimensionScore(ReviewDimension.STYLE, 1.0),
        DimensionScore(ReviewDimension.SECURITY, 1.0),
        DimensionScore(ReviewDimension.PERFORMANCE, 1.0),
        DimensionScore(ReviewDimension.MAINTAINABILITY, 1.0),
    ]
    result = scorer.score_review(scores)
    assert result["missing_dimensions"] == []


def test_score_review_grade_is_string():
    scorer = CodeReviewScorer()
    scores = [DimensionScore(ReviewDimension.CORRECTNESS, 0.9)]
    result = scorer.score_review(scores)
    assert isinstance(result["grade"], str)


def test_score_review_weighted_score_is_float():
    scorer = CodeReviewScorer()
    scores = [DimensionScore(ReviewDimension.SECURITY, 0.7)]
    result = scorer.score_review(scores)
    assert isinstance(result["weighted_score"], float)


# ---------------------------------------------------------------------------
# CODE_REVIEW_SCORER_REGISTRY
# ---------------------------------------------------------------------------


def test_registry_has_default_key():
    assert "default" in CODE_REVIEW_SCORER_REGISTRY


def test_registry_has_security_focused_key():
    assert "security_focused" in CODE_REVIEW_SCORER_REGISTRY


def test_registry_default_is_scorer_instance():
    assert isinstance(CODE_REVIEW_SCORER_REGISTRY["default"], CodeReviewScorer)


def test_registry_security_focused_is_scorer_instance():
    assert isinstance(CODE_REVIEW_SCORER_REGISTRY["security_focused"], CodeReviewScorer)


def test_registry_security_focused_security_weight_gt_point_four():
    scorer = CODE_REVIEW_SCORER_REGISTRY["security_focused"]
    security_weight = scorer.rubric.weights[ReviewDimension.SECURITY]
    assert security_weight > 0.4


def test_registry_security_focused_security_weight_is_half():
    scorer = CODE_REVIEW_SCORER_REGISTRY["security_focused"]
    # Raw weight was 0.5 out of 1.0 total — stays 0.5 after normalization
    security_weight = scorer.rubric.weights[ReviewDimension.SECURITY]
    assert security_weight == pytest.approx(0.5)


def test_registry_default_correctness_weight():
    scorer = CODE_REVIEW_SCORER_REGISTRY["default"]
    assert scorer.rubric.weights[ReviewDimension.CORRECTNESS] == pytest.approx(0.35)
