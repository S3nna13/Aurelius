"""Tests for src/evaluation/perplexity_scorer.py — ≥28 test cases."""

from __future__ import annotations

import math
import pytest

from src.evaluation.perplexity_scorer import (
    PerplexityResult,
    PerplexityScorer,
    PERPLEXITY_SCORER_REGISTRY,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scorer():
    return PerplexityScorer()


# ---------------------------------------------------------------------------
# PerplexityResult dataclass
# ---------------------------------------------------------------------------

class TestPerplexityResultFrozen:
    def test_is_frozen(self):
        result = PerplexityResult(text="hello", log_prob_sum=-3.0, token_count=3, perplexity=2.718)
        with pytest.raises((AttributeError, TypeError)):
            result.perplexity = 1.0  # type: ignore[misc]

    def test_fields_accessible(self):
        result = PerplexityResult(text="x", log_prob_sum=-1.0, token_count=1, perplexity=math.e)
        assert result.text == "x"
        assert result.log_prob_sum == -1.0
        assert result.token_count == 1
        assert result.perplexity == math.e

    def test_equality(self):
        a = PerplexityResult(text="x", log_prob_sum=-1.0, token_count=1, perplexity=2.0)
        b = PerplexityResult(text="x", log_prob_sum=-1.0, token_count=1, perplexity=2.0)
        assert a == b

    def test_inequality_different_perplexity(self):
        a = PerplexityResult(text="x", log_prob_sum=-1.0, token_count=1, perplexity=2.0)
        b = PerplexityResult(text="x", log_prob_sum=-1.0, token_count=1, perplexity=3.0)
        assert a != b


# ---------------------------------------------------------------------------
# from_log_probs
# ---------------------------------------------------------------------------

class TestFromLogProbs:
    def test_empty_log_probs_gives_inf(self, scorer):
        result = scorer.from_log_probs("hello", [])
        assert math.isinf(result.perplexity)
        assert result.token_count == 0
        assert result.log_prob_sum == 0.0

    def test_empty_log_probs_text_preserved(self, scorer):
        result = scorer.from_log_probs("some text", [])
        assert result.text == "some text"

    def test_uniform_log_probs_perplexity(self, scorer):
        # log_probs all -1.0 → mean = -1.0 → perplexity = exp(1) ≈ 2.718...
        log_probs = [-1.0, -1.0, -1.0]
        result = scorer.from_log_probs("abc", log_probs)
        assert math.isclose(result.perplexity, math.e, rel_tol=1e-9)

    def test_uniform_zero_log_probs_perplexity_is_one(self, scorer):
        # log_probs all 0.0 → mean = 0.0 → perplexity = exp(0) = 1.0
        result = scorer.from_log_probs("abc", [0.0, 0.0, 0.0])
        assert math.isclose(result.perplexity, 1.0, rel_tol=1e-9)

    def test_token_count_matches_log_probs_length(self, scorer):
        log_probs = [-0.5, -1.0, -1.5, -2.0]
        result = scorer.from_log_probs("text", log_probs)
        assert result.token_count == 4

    def test_log_prob_sum_correct(self, scorer):
        log_probs = [-1.0, -2.0, -3.0]
        result = scorer.from_log_probs("text", log_probs)
        assert math.isclose(result.log_prob_sum, -6.0, rel_tol=1e-9)

    def test_negative_log_probs_increases_perplexity(self, scorer):
        result_high = scorer.from_log_probs("a", [-5.0])
        result_low = scorer.from_log_probs("a", [-1.0])
        assert result_high.perplexity > result_low.perplexity

    def test_single_token(self, scorer):
        result = scorer.from_log_probs("a", [-2.0])
        assert math.isclose(result.perplexity, math.exp(2.0), rel_tol=1e-9)

    def test_text_preserved(self, scorer):
        result = scorer.from_log_probs("hello world", [-1.0, -1.0])
        assert result.text == "hello world"

    def test_positive_log_probs_give_perplexity_below_one(self, scorer):
        # Positive log probs: mean > 0 → exp(-mean) < 1
        result = scorer.from_log_probs("a", [1.0, 2.0])
        assert result.perplexity < 1.0


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

class TestCompare:
    def test_a_wins_lower_perplexity(self, scorer):
        a = PerplexityResult(text="a", log_prob_sum=0.0, token_count=1, perplexity=2.0)
        b = PerplexityResult(text="b", log_prob_sum=0.0, token_count=1, perplexity=5.0)
        assert scorer.compare(a, b) == "a"

    def test_b_wins_lower_perplexity(self, scorer):
        a = PerplexityResult(text="a", log_prob_sum=0.0, token_count=1, perplexity=10.0)
        b = PerplexityResult(text="b", log_prob_sum=0.0, token_count=1, perplexity=3.0)
        assert scorer.compare(a, b) == "b"

    def test_tie_equal_perplexity(self, scorer):
        a = PerplexityResult(text="a", log_prob_sum=0.0, token_count=1, perplexity=4.0)
        b = PerplexityResult(text="b", log_prob_sum=0.0, token_count=1, perplexity=4.0)
        assert scorer.compare(a, b) == "tie"

    def test_compare_inf_vs_finite(self, scorer):
        a = PerplexityResult(text="a", log_prob_sum=0.0, token_count=0, perplexity=float("inf"))
        b = PerplexityResult(text="b", log_prob_sum=0.0, token_count=1, perplexity=100.0)
        assert scorer.compare(a, b) == "b"


# ---------------------------------------------------------------------------
# batch_score
# ---------------------------------------------------------------------------

class TestBatchScore:
    def test_batch_length_matches_input(self, scorer):
        examples = [("a", [-1.0]), ("b", [-2.0]), ("c", [-3.0])]
        results = scorer.batch_score(examples)
        assert len(results) == 3

    def test_batch_empty_input(self, scorer):
        results = scorer.batch_score([])
        assert results == []

    def test_batch_results_are_perplexity_results(self, scorer):
        examples = [("hello", [-1.0, -1.0])]
        results = scorer.batch_score(examples)
        assert isinstance(results[0], PerplexityResult)

    def test_batch_order_preserved(self, scorer):
        examples = [("x", [-1.0]), ("y", [0.0])]
        results = scorer.batch_score(examples)
        assert results[0].text == "x"
        assert results[1].text == "y"


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_count(self, scorer):
        results = scorer.batch_score([("a", [-1.0]), ("b", [-2.0]), ("c", [-3.0])])
        s = scorer.summary(results)
        assert s["count"] == 3

    def test_summary_mean_perplexity(self, scorer):
        r1 = PerplexityResult(text="a", log_prob_sum=0.0, token_count=1, perplexity=2.0)
        r2 = PerplexityResult(text="b", log_prob_sum=0.0, token_count=1, perplexity=4.0)
        s = scorer.summary([r1, r2])
        assert math.isclose(s["mean_perplexity"], 3.0, rel_tol=1e-9)

    def test_summary_min_perplexity(self, scorer):
        r1 = PerplexityResult(text="a", log_prob_sum=0.0, token_count=1, perplexity=2.0)
        r2 = PerplexityResult(text="b", log_prob_sum=0.0, token_count=1, perplexity=8.0)
        s = scorer.summary([r1, r2])
        assert s["min_perplexity"] == 2.0

    def test_summary_max_perplexity(self, scorer):
        r1 = PerplexityResult(text="a", log_prob_sum=0.0, token_count=1, perplexity=2.0)
        r2 = PerplexityResult(text="b", log_prob_sum=0.0, token_count=1, perplexity=8.0)
        s = scorer.summary([r1, r2])
        assert s["max_perplexity"] == 8.0

    def test_summary_empty_list(self, scorer):
        s = scorer.summary([])
        assert s["count"] == 0


# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_has_default(self):
        assert "default" in PERPLEXITY_SCORER_REGISTRY

    def test_registry_default_is_perplexity_scorer_class(self):
        assert PERPLEXITY_SCORER_REGISTRY["default"] is PerplexityScorer

    def test_registry_default_instantiable(self):
        cls = PERPLEXITY_SCORER_REGISTRY["default"]
        instance = cls()
        assert isinstance(instance, PerplexityScorer)
