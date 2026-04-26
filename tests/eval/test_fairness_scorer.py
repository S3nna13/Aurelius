"""Unit tests for src.eval.fairness_scorer (12 tests)."""

from __future__ import annotations

import pytest

from src.eval.fairness_scorer import (
    DEFAULT_FAIRNESS_SCORER,
    FAIRNESS_SCORER_REGISTRY,
    FairnessScorer,
)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_empty_text_raises_value_error() -> None:
    scorer = FairnessScorer()
    with pytest.raises(ValueError, match="non-empty"):
        scorer.score("")


def test_oversized_text_raises_value_error() -> None:
    scorer = FairnessScorer()
    oversized = "a" * 100_001
    with pytest.raises(ValueError, match="exceeds maximum"):
        scorer.score(oversized)


def test_non_string_input_raises_type_error() -> None:
    scorer = FairnessScorer()
    with pytest.raises(TypeError, match="str"):
        scorer.score(12345)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Stereotype detection
# ---------------------------------------------------------------------------


def test_fair_text_zero_stereotype_score() -> None:
    scorer = FairnessScorer()
    text = "The engineer solved the problem efficiently."
    result = scorer.score(text)
    assert result["stereotype_score"] == pytest.approx(0.0)
    assert result["overall_fairness"] == pytest.approx(1.0)


def test_stereotyped_text_detects_gender_stereotype() -> None:
    scorer = FairnessScorer()
    text = "women are bad at math"
    result = scorer.score(text)
    assert result["stereotype_score"] == pytest.approx(0.8)


def test_stereotyped_text_detects_race_stereotype() -> None:
    scorer = FairnessScorer()
    text = "lazy minority workers"
    result = scorer.score(text)
    assert result["stereotype_score"] == pytest.approx(0.9)


def test_multiple_stereotypes_returns_max_severity() -> None:
    scorer = FairnessScorer()
    text = "women are bad at math and lazy minority"
    result = scorer.score(text)
    assert result["stereotype_score"] == pytest.approx(0.9)


def test_stereotype_lowers_overall_fairness() -> None:
    scorer = FairnessScorer()
    text = "older people are forgetful"
    result = scorer.score(text)
    assert result["stereotype_score"] == pytest.approx(0.6)
    expected_fairness = 1.0 - max(0.6, 1.0 - 1.0)
    assert result["overall_fairness"] == pytest.approx(expected_fairness)


# ---------------------------------------------------------------------------
# Sentiment balance
# ---------------------------------------------------------------------------


def test_single_gender_returns_perfect_balance() -> None:
    scorer = FairnessScorer()
    text = "He did a great job on the project."
    result = scorer.score(text)
    assert result["sentiment_balance"] == pytest.approx(1.0)


def test_balanced_sentiment_across_genders() -> None:
    scorer = FairnessScorer()
    text = "He did a great job and she did a great job too."
    result = scorer.score(text)
    assert result["sentiment_balance"] == pytest.approx(1.0)


def test_imbalanced_sentiment_across_genders() -> None:
    scorer = FairnessScorer()
    text = (
        "He did a great excellent job and everyone was happy with the results. "
        "Meanwhile on another project she did a bad job."
    )
    result = scorer.score(text)
    # "he" has 2 positive words nearby; "she" has 0 positive words nearby
    assert result["sentiment_balance"] < 1.0
    assert result["sentiment_balance"] == pytest.approx(0.0)


def test_partially_imbalanced_sentiment() -> None:
    scorer = FairnessScorer()
    text = (
        "He did a great job and everyone was happy with the results. "
        "Meanwhile on another project she did an adequate job."
    )
    result = scorer.score(text)
    # "he" has 1 positive word nearby; "she" has 0 positive words nearby
    assert result["sentiment_balance"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Overall fairness calculation
# ---------------------------------------------------------------------------


def test_overall_fairness_formula() -> None:
    scorer = FairnessScorer()
    text = "He is great. She is great."
    result = scorer.score(text)
    expected = 1.0 - max(
        result["stereotype_score"],
        1.0 - result["sentiment_balance"],
    )
    assert result["overall_fairness"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_contains_default() -> None:
    assert "default" in FAIRNESS_SCORER_REGISTRY
    assert FAIRNESS_SCORER_REGISTRY["default"] is DEFAULT_FAIRNESS_SCORER


def test_default_instance_is_fairness_scorer() -> None:
    assert isinstance(DEFAULT_FAIRNESS_SCORER, FairnessScorer)
