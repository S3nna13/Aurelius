"""Tests for src/evaluation/f1_scorer.py — ≥28 test cases."""

from __future__ import annotations

import math

import pytest

from src.evaluation.f1_scorer import (
    F1_SCORER_REGISTRY,
    F1Result,
    F1Scorer,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def scorer():
    return F1Scorer()


# ---------------------------------------------------------------------------
# F1Result dataclass
# ---------------------------------------------------------------------------


class TestF1ResultFrozen:
    def test_is_frozen(self):
        result = F1Result(
            f1=1.0,
            precision=1.0,
            recall=1.0,
            common_tokens=3,
            predicted_tokens=3,
            gold_tokens=3,
        )
        with pytest.raises((AttributeError, TypeError)):
            result.f1 = 0.0  # type: ignore[misc]

    def test_fields_accessible(self):
        result = F1Result(
            f1=0.5,
            precision=0.4,
            recall=0.6,
            common_tokens=2,
            predicted_tokens=5,
            gold_tokens=4,
        )
        assert result.f1 == 0.5
        assert result.precision == 0.4
        assert result.recall == 0.6
        assert result.common_tokens == 2
        assert result.predicted_tokens == 5
        assert result.gold_tokens == 4


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_lowercases(self, scorer):
        assert scorer._normalize("Hello World") == ["hello", "world"]

    def test_removes_punctuation(self, scorer):
        assert scorer._normalize("hello, world!") == ["hello", "world"]

    def test_empty_string(self, scorer):
        assert scorer._normalize("") == []

    def test_numbers_kept(self, scorer):
        tokens = scorer._normalize("answer is 42")
        assert "42" in tokens

    def test_mixed_case_and_punct(self, scorer):
        tokens = scorer._normalize("It's a TEST.")
        assert "test" in tokens
        assert "its" in tokens or "it" in tokens  # apostrophe removal


# ---------------------------------------------------------------------------
# score — single pair
# ---------------------------------------------------------------------------


class TestScore:
    def test_exact_match_f1_one(self, scorer):
        result = scorer.score("the cat sat on the mat", "the cat sat on the mat")
        assert math.isclose(result.f1, 1.0)

    def test_exact_match_precision_one(self, scorer):
        result = scorer.score("hello world", "hello world")
        assert math.isclose(result.precision, 1.0)

    def test_exact_match_recall_one(self, scorer):
        result = scorer.score("hello world", "hello world")
        assert math.isclose(result.recall, 1.0)

    def test_no_overlap_f1_zero(self, scorer):
        result = scorer.score("foo bar baz", "alpha beta gamma")
        assert result.f1 == 0.0

    def test_no_overlap_precision_zero(self, scorer):
        result = scorer.score("foo bar", "baz qux")
        assert result.precision == 0.0

    def test_no_overlap_recall_zero(self, scorer):
        result = scorer.score("foo bar", "baz qux")
        assert result.recall == 0.0

    def test_partial_overlap_f1_between_zero_and_one(self, scorer):
        result = scorer.score("the cat sat", "the cat sat on the mat")
        assert 0.0 < result.f1 < 1.0

    def test_precision_and_recall_distinct_for_different_lengths(self, scorer):
        # prediction is subset of gold → recall=1.0, precision<1.0 (or vice versa)
        result = scorer.score("cat", "the cat sat")
        # "cat" is in gold; precision = 1/1, recall = 1/3
        assert math.isclose(result.precision, 1.0)
        assert math.isclose(result.recall, 1 / 3, rel_tol=1e-9)

    def test_both_empty_f1_one(self, scorer):
        result = scorer.score("", "")
        assert math.isclose(result.f1, 1.0)

    def test_both_empty_precision_one(self, scorer):
        result = scorer.score("", "")
        assert math.isclose(result.precision, 1.0)

    def test_both_empty_recall_one(self, scorer):
        result = scorer.score("", "")
        assert math.isclose(result.recall, 1.0)

    def test_prediction_empty_gold_non_empty_f1_zero(self, scorer):
        result = scorer.score("", "the cat sat on the mat")
        assert result.f1 == 0.0

    def test_gold_empty_prediction_non_empty_f1_zero(self, scorer):
        result = scorer.score("the cat sat on the mat", "")
        assert result.f1 == 0.0

    def test_returns_f1_result(self, scorer):
        result = scorer.score("hello", "hello")
        assert isinstance(result, F1Result)

    def test_common_tokens_count_correct(self, scorer):
        result = scorer.score("the cat sat", "the cat ran")
        # "the" and "cat" are common
        assert result.common_tokens == 2

    def test_predicted_tokens_count_correct(self, scorer):
        result = scorer.score("the cat sat", "anything")
        assert result.predicted_tokens == 3

    def test_gold_tokens_count_correct(self, scorer):
        result = scorer.score("anything", "the cat sat on the mat")
        assert result.gold_tokens == 6

    def test_normalization_lowercases_before_comparison(self, scorer):
        result_lower = scorer.score("the cat", "the cat")
        result_mixed = scorer.score("The Cat", "THE CAT")
        assert math.isclose(result_lower.f1, result_mixed.f1)

    def test_f1_formula_harmonic_mean(self, scorer):
        # precision=1/1, recall=1/3 → f1 = 2*(1)*(1/3)/(1+1/3) = (2/3)/(4/3) = 0.5
        result = scorer.score("cat", "the cat sat")
        assert math.isclose(result.f1, 0.5, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# batch_score
# ---------------------------------------------------------------------------


class TestBatchScore:
    def test_batch_score_length_matches_input(self, scorer):
        preds = ["the cat sat", "hello world", "foo bar"]
        golds = ["the cat sat on mat", "hello world", "baz qux"]
        results = scorer.batch_score(preds, golds)
        assert len(results) == 3

    def test_batch_score_length_mismatch_raises_value_error(self, scorer):
        with pytest.raises(ValueError):
            scorer.batch_score(["a", "b"], ["c"])

    def test_batch_score_each_is_f1_result(self, scorer):
        results = scorer.batch_score(["hello"], ["hello"])
        assert isinstance(results[0], F1Result)

    def test_batch_score_empty_lists(self, scorer):
        results = scorer.batch_score([], [])
        assert results == []

    def test_batch_score_matches_individual_scores(self, scorer):
        preds = ["the cat", "hello world"]
        golds = ["the dog", "hello world"]
        batch = scorer.batch_score(preds, golds)
        individual = [scorer.score(p, g) for p, g in zip(preds, golds)]
        for b, i in zip(batch, individual):
            assert math.isclose(b.f1, i.f1)


# ---------------------------------------------------------------------------
# mean_f1
# ---------------------------------------------------------------------------


class TestMeanF1:
    def test_mean_f1_all_ones(self, scorer):
        results = [scorer.score("hello world", "hello world")] * 3
        assert math.isclose(scorer.mean_f1(results), 1.0)

    def test_mean_f1_all_zeros(self, scorer):
        results = [scorer.score("foo bar", "baz qux")] * 4
        assert scorer.mean_f1(results) == 0.0

    def test_mean_f1_mixed(self, scorer):
        r1 = scorer.score("hello world", "hello world")  # f1=1.0
        r2 = scorer.score("foo bar", "baz qux")  # f1=0.0
        mean = scorer.mean_f1([r1, r2])
        assert math.isclose(mean, 0.5, rel_tol=1e-9)

    def test_mean_f1_empty_returns_zero(self, scorer):
        assert scorer.mean_f1([]) == 0.0

    def test_mean_f1_single_result(self, scorer):
        r = scorer.score("cat", "the cat sat")
        mean = scorer.mean_f1([r])
        assert math.isclose(mean, r.f1)


# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registry_has_default(self):
        assert "default" in F1_SCORER_REGISTRY

    def test_registry_default_is_f1_scorer_class(self):
        assert F1_SCORER_REGISTRY["default"] is F1Scorer

    def test_registry_default_instantiable(self):
        cls = F1_SCORER_REGISTRY["default"]
        instance = cls()
        assert isinstance(instance, F1Scorer)
