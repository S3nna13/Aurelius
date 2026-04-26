"""Tests for src/evaluation/faithfulness_scorer.py — ≥28 test cases."""

from __future__ import annotations

import math

import pytest

from src.evaluation.faithfulness_scorer import (
    FAITHFULNESS_SCORER_REGISTRY,
    FaithfulnessResult,
    FaithfulnessScorer,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def scorer():
    return FaithfulnessScorer()


# Source used across multiple tests
SCIENCE_SOURCE = (
    "The mitochondria is the powerhouse of the cell. "
    "It produces ATP through cellular respiration. "
    "The nucleus contains the genetic material of the cell."
)


# ---------------------------------------------------------------------------
# FaithfulnessResult dataclass
# ---------------------------------------------------------------------------


class TestFaithfulnessResultFrozen:
    def test_is_frozen(self):
        result = FaithfulnessResult(score=1.0, supported_claims=2, total_claims=2, unsupported=[])
        with pytest.raises((AttributeError, TypeError)):
            result.score = 0.0  # type: ignore[misc]

    def test_fields_accessible(self):
        unsup = ["claim one", "claim two"]
        result = FaithfulnessResult(
            score=0.5, supported_claims=1, total_claims=2, unsupported=unsup
        )
        assert result.score == 0.5
        assert result.supported_claims == 1
        assert result.total_claims == 2
        assert result.unsupported == unsup


# ---------------------------------------------------------------------------
# _split_claims
# ---------------------------------------------------------------------------


class TestSplitClaims:
    def test_splits_on_period_space(self, scorer):
        text = "The sky is blue. The grass is green."
        claims = scorer._split_claims(text)
        assert len(claims) >= 1

    def test_splits_on_question_mark(self, scorer):
        text = "Is the sky blue? Yes it is."
        claims = scorer._split_claims(text)
        assert len(claims) >= 2

    def test_splits_on_exclamation(self, scorer):
        text = "Wow that is great! I love it."
        claims = scorer._split_claims(text)
        assert len(claims) >= 2

    def test_empty_string_returns_empty(self, scorer):
        assert scorer._split_claims("") == []

    def test_single_sentence_no_trailing_punct(self, scorer):
        claims = scorer._split_claims("The sky is blue")
        assert len(claims) == 1
        assert claims[0] == "The sky is blue"

    def test_strips_whitespace_from_claims(self, scorer):
        text = "First sentence. Second sentence."
        claims = scorer._split_claims(text)
        for claim in claims:
            assert claim == claim.strip()


# ---------------------------------------------------------------------------
# _word_overlap
# ---------------------------------------------------------------------------


class TestWordOverlap:
    def test_identical_strings_overlap_one(self, scorer):
        assert math.isclose(scorer._word_overlap("the cat", "the cat"), 1.0)

    def test_no_overlap_returns_zero(self, scorer):
        assert scorer._word_overlap("foo bar", "baz qux") == 0.0

    def test_partial_overlap_between_zero_and_one(self, scorer):
        val = scorer._word_overlap("the cat sat", "the dog sat")
        assert 0.0 < val < 1.0

    def test_case_insensitive(self, scorer):
        assert scorer._word_overlap("The Cat", "the cat") == 1.0

    def test_both_empty_returns_zero(self, scorer):
        assert scorer._word_overlap("", "") == 0.0


# ---------------------------------------------------------------------------
# score — single pair
# ---------------------------------------------------------------------------


class TestScore:
    def test_identical_generated_source_score_one(self, scorer):
        text = "The mitochondria produces ATP."
        result = scorer.score(text, text)
        assert math.isclose(result.score, 1.0)

    def test_unrelated_score_near_zero(self, scorer):
        result = scorer.score(
            "Quantum entanglement violates locality.",
            "The mitochondria produces ATP through respiration.",
        )
        assert result.score < 0.5

    def test_partial_overlap_score_in_range(self, scorer):
        result = scorer.score(
            "The mitochondria produces energy. Dinosaurs ruled the Jurassic.",
            SCIENCE_SOURCE,
        )
        assert 0.0 <= result.score <= 1.0

    def test_threshold_effect_higher_threshold_fewer_supported(self, scorer):
        text = "The mitochondria is a cell organelle."
        low_result = scorer.score(text, SCIENCE_SOURCE, threshold=0.1)
        high_result = scorer.score(text, SCIENCE_SOURCE, threshold=0.9)
        assert low_result.supported_claims >= high_result.supported_claims

    def test_supported_plus_unsupported_equals_total(self, scorer):
        result = scorer.score(
            "The cell has a nucleus. Aliens live on Mars.",
            SCIENCE_SOURCE,
        )
        assert result.supported_claims + len(result.unsupported) == result.total_claims

    def test_total_claims_count_correct(self, scorer):
        # Two clear sentences
        result = scorer.score(
            "The mitochondria is the powerhouse. The nucleus holds DNA.",
            SCIENCE_SOURCE,
        )
        assert result.total_claims >= 2

    def test_unsupported_list_contains_strings(self, scorer):
        result = scorer.score(
            "The mitochondria produces ATP. Aliens live on Mars.",
            SCIENCE_SOURCE,
        )
        for item in result.unsupported:
            assert isinstance(item, str)

    def test_empty_generated_returns_score_one(self, scorer):
        result = scorer.score("", SCIENCE_SOURCE)
        assert math.isclose(result.score, 1.0)

    def test_empty_generated_zero_claims(self, scorer):
        result = scorer.score("", SCIENCE_SOURCE)
        assert result.total_claims == 0
        assert result.supported_claims == 0
        assert result.unsupported == []

    def test_returns_faithfulness_result(self, scorer):
        result = scorer.score("The cell has a nucleus.", SCIENCE_SOURCE)
        assert isinstance(result, FaithfulnessResult)

    def test_score_between_zero_and_one(self, scorer):
        result = scorer.score("The nucleus contains DNA. Aliens built pyramids.", SCIENCE_SOURCE)
        assert 0.0 <= result.score <= 1.0

    def test_fully_faithful_all_claims_supported(self, scorer):
        # Use exact wording from source for high overlap
        result = scorer.score(
            "The mitochondria is the powerhouse of the cell.",
            SCIENCE_SOURCE,
            threshold=0.3,
        )
        assert result.supported_claims >= 1

    def test_fully_unfaithful_no_claims_supported(self, scorer):
        result = scorer.score(
            "Unicorns frolic in rainbow meadows. Dragons breathe purple fire.",
            "The mitochondria produces ATP through cellular respiration.",
            threshold=0.9,
        )
        assert result.supported_claims == 0


# ---------------------------------------------------------------------------
# batch_score
# ---------------------------------------------------------------------------


class TestBatchScore:
    def test_batch_score_returns_list(self, scorer):
        pairs = [
            ("The nucleus contains DNA.", SCIENCE_SOURCE),
            ("Aliens live on Mars.", SCIENCE_SOURCE),
        ]
        results = scorer.batch_score(pairs)
        assert isinstance(results, list)

    def test_batch_score_length_matches_input(self, scorer):
        pairs = [
            ("The mitochondria produces ATP.", SCIENCE_SOURCE),
            ("Quantum mechanics is strange.", SCIENCE_SOURCE),
            ("The cell nucleus holds DNA.", SCIENCE_SOURCE),
        ]
        results = scorer.batch_score(pairs)
        assert len(results) == 3

    def test_batch_score_each_is_faithfulness_result(self, scorer):
        pairs = [("The cell has a nucleus.", SCIENCE_SOURCE)]
        results = scorer.batch_score(pairs)
        assert isinstance(results[0], FaithfulnessResult)

    def test_batch_score_empty_input(self, scorer):
        results = scorer.batch_score([])
        assert results == []

    def test_batch_score_respects_threshold(self, scorer):
        pairs = [("The mitochondria is the powerhouse of the cell.", SCIENCE_SOURCE)]
        low = scorer.batch_score(pairs, threshold=0.1)
        high = scorer.batch_score(pairs, threshold=0.99)
        assert low[0].supported_claims >= high[0].supported_claims


# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registry_has_default(self):
        assert "default" in FAITHFULNESS_SCORER_REGISTRY

    def test_registry_default_is_faithfulness_scorer_class(self):
        assert FAITHFULNESS_SCORER_REGISTRY["default"] is FaithfulnessScorer

    def test_registry_default_instantiable(self):
        cls = FAITHFULNESS_SCORER_REGISTRY["default"]
        instance = cls()
        assert isinstance(instance, FaithfulnessScorer)
