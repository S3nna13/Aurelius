"""Tests for src/safety/toxicity_scorer.py"""

from __future__ import annotations

import pytest

from src.safety.toxicity_scorer import (
    SAFETY_REGISTRY,
    ToxicityCategory,
    ToxicityScore,
    ToxicityScorer,
)


@pytest.fixture
def scorer() -> ToxicityScorer:
    return ToxicityScorer()


# ---------------------------------------------------------------------------
# Basic interface
# ---------------------------------------------------------------------------


def test_score_returns_toxicity_score(scorer):
    result = scorer.score("hello world")
    assert isinstance(result, ToxicityScore)


def test_score_has_all_categories(scorer):
    result = scorer.score("hello world")
    assert set(result.scores.keys()) == set(ToxicityCategory)


def test_score_values_in_unit_interval(scorer):
    result = scorer.score("This is a completely benign sentence about kittens.")
    for v in result.scores.values():
        assert 0.0 <= v <= 1.0
    assert 0.0 <= result.overall <= 1.0


def test_overall_is_max_of_categories(scorer):
    result = scorer.score("I hate everyone. Buy now — limited offer!")
    assert result.overall == pytest.approx(max(result.scores.values()))


# ---------------------------------------------------------------------------
# Clean text
# ---------------------------------------------------------------------------


def test_clean_text_not_flagged(scorer):
    result = scorer.score("The weather is lovely today.", threshold=0.5)
    assert not result.flagged


def test_clean_text_low_scores(scorer):
    result = scorer.score("Please pass the butter.")
    assert result.overall < 0.5


# ---------------------------------------------------------------------------
# Category-specific detection
# ---------------------------------------------------------------------------


def test_hate_pattern_detected(scorer):
    result = scorer.score("I hate all those people.", threshold=0.1)
    assert result.scores[ToxicityCategory.HATE] > 0.0


def test_violence_pattern_detected(scorer):
    result = scorer.score("I want to kill them all right now.", threshold=0.1)
    assert result.scores[ToxicityCategory.VIOLENCE] > 0.0


def test_spam_pattern_detected(scorer):
    result = scorer.score("Click here to win a free prize — act fast!", threshold=0.1)
    assert result.scores[ToxicityCategory.SPAM] > 0.0


def test_self_harm_pattern_detected(scorer):
    result = scorer.score("I want to kill myself and end my life.", threshold=0.1)
    assert result.scores[ToxicityCategory.SELF_HARM] > 0.0


def test_harassment_pattern_detected(scorer):
    result = scorer.score("You are worthless and nobody likes you.", threshold=0.1)
    assert result.scores[ToxicityCategory.HARASSMENT] > 0.0


# ---------------------------------------------------------------------------
# Threshold
# ---------------------------------------------------------------------------


def test_threshold_controls_flagged(scorer):
    text = "I hate all those people."
    # overall score for this text is >= 0 so threshold=2.0 (above max) won't flag
    high = scorer.score(text, threshold=2.0)
    assert not high.flagged
    # With very low threshold — should flag
    low = scorer.score(text, threshold=0.0)
    assert low.flagged


# ---------------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------------


def test_batch_score_returns_correct_length(scorer):
    texts = ["hello", "world", "foo bar baz"]
    results = scorer.batch_score(texts)
    assert len(results) == 3


def test_batch_score_types(scorer):
    results = scorer.batch_score(["a", "b"])
    for r in results:
        assert isinstance(r, ToxicityScore)


def test_batch_score_preserves_text(scorer):
    texts = ["sentence one", "sentence two"]
    results = scorer.batch_score(texts)
    for original, result in zip(texts, results):
        assert result.text == original


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_safety_registry_contains_scorer():
    assert "toxicity_scorer" in SAFETY_REGISTRY
    assert isinstance(SAFETY_REGISTRY["toxicity_scorer"], ToxicityScorer)
