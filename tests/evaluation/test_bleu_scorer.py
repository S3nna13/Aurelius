"""Tests for src/evaluation/bleu_scorer.py — ≥28 test cases."""

from __future__ import annotations

import math
import pytest

from src.evaluation.bleu_scorer import (
    BLEUResult,
    BLEUScorer,
    BLEU_SCORER_REGISTRY,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scorer():
    return BLEUScorer()


# ---------------------------------------------------------------------------
# BLEUResult dataclass
# ---------------------------------------------------------------------------

class TestBLEUResultFrozen:
    def test_is_frozen(self):
        result = BLEUResult(
            score=1.0,
            precisions=[1.0, 1.0, 1.0, 1.0],
            brevity_penalty=1.0,
            hypothesis_len=4,
            reference_len=4,
        )
        with pytest.raises((AttributeError, TypeError)):
            result.score = 0.0  # type: ignore[misc]

    def test_fields_accessible(self):
        result = BLEUResult(
            score=0.5,
            precisions=[0.9, 0.7, 0.5, 0.3],
            brevity_penalty=0.8,
            hypothesis_len=3,
            reference_len=5,
        )
        assert result.score == 0.5
        assert result.brevity_penalty == 0.8
        assert result.hypothesis_len == 3
        assert result.reference_len == 5

    def test_precisions_list_stored(self):
        prec = [1.0, 0.8, 0.5, 0.2]
        result = BLEUResult(
            score=0.6,
            precisions=prec,
            brevity_penalty=1.0,
            hypothesis_len=5,
            reference_len=5,
        )
        assert result.precisions == prec


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_lowercases(self, scorer):
        assert scorer._tokenize("Hello World") == ["hello", "world"]

    def test_strips_punctuation(self, scorer):
        assert scorer._tokenize("hello, world!") == ["hello", "world"]

    def test_empty_string(self, scorer):
        assert scorer._tokenize("") == []

    def test_numbers_kept(self, scorer):
        assert "123" in scorer._tokenize("abc 123")


# ---------------------------------------------------------------------------
# _ngrams
# ---------------------------------------------------------------------------

class TestNgrams:
    def test_unigrams(self, scorer):
        result = scorer._ngrams(["a", "b", "c"], 1)
        assert result == {("a",): 1, ("b",): 1, ("c",): 1}

    def test_bigrams(self, scorer):
        result = scorer._ngrams(["a", "b", "c"], 2)
        assert ("a", "b") in result
        assert ("b", "c") in result

    def test_repeated_ngram_counts(self, scorer):
        result = scorer._ngrams(["a", "a", "a"], 1)
        assert result[("a",)] == 3

    def test_empty_tokens_returns_empty(self, scorer):
        assert scorer._ngrams([], 1) == {}

    def test_n_larger_than_tokens_empty(self, scorer):
        assert scorer._ngrams(["a", "b"], 5) == {}


# ---------------------------------------------------------------------------
# _clipped_precision
# ---------------------------------------------------------------------------

class TestClippedPrecision:
    def test_identical_unigrams_precision_one(self, scorer):
        hyp = ["the", "cat", "sat"]
        ref = ["the", "cat", "sat"]
        assert math.isclose(scorer._clipped_precision(hyp, ref, 1), 1.0)

    def test_no_overlap_precision_zero(self, scorer):
        hyp = ["foo", "bar"]
        ref = ["baz", "qux"]
        assert scorer._clipped_precision(hyp, ref, 1) == 0.0

    def test_clip_limits_inflated_count(self, scorer):
        # hyp repeats "the" 3 times, ref has it once → clipped to 1/3
        hyp = ["the", "the", "the"]
        ref = ["the", "cat", "sat"]
        result = scorer._clipped_precision(hyp, ref, 1)
        assert math.isclose(result, 1 / 3, rel_tol=1e-9)

    def test_empty_hypothesis_returns_zero(self, scorer):
        assert scorer._clipped_precision([], ["a", "b"], 1) == 0.0


# ---------------------------------------------------------------------------
# score — single pair
# ---------------------------------------------------------------------------

class TestScore:
    def test_identical_text_bleu_one(self, scorer):
        result = scorer.score("the cat sat on the mat", "the cat sat on the mat")
        assert math.isclose(result.score, 1.0, rel_tol=1e-6)

    def test_no_overlap_bleu_near_zero(self, scorer):
        result = scorer.score("foo bar baz", "alpha beta gamma")
        assert result.score < 0.01

    def test_partial_overlap_in_range(self, scorer):
        result = scorer.score("the cat sat", "the cat sat on the mat")
        assert 0.0 < result.score < 1.0

    def test_brevity_penalty_less_than_one_for_short_hyp(self, scorer):
        result = scorer.score("cat", "the cat sat on the mat")
        assert result.brevity_penalty < 1.0

    def test_brevity_penalty_one_when_hyp_longer_than_ref(self, scorer):
        result = scorer.score("the cat sat on the mat today", "the cat sat")
        assert math.isclose(result.brevity_penalty, 1.0)

    def test_brevity_penalty_one_when_equal_length(self, scorer):
        result = scorer.score("the cat sat", "the cat sat")
        assert math.isclose(result.brevity_penalty, 1.0)

    def test_precisions_list_length_equals_max_n(self, scorer):
        result = scorer.score("hello world", "hello world", max_n=4)
        assert len(result.precisions) == 4

    def test_precisions_list_length_custom_max_n(self, scorer):
        result = scorer.score("hello world", "hello world", max_n=2)
        assert len(result.precisions) == 2

    def test_returns_bleu_result(self, scorer):
        result = scorer.score("hello", "hello")
        assert isinstance(result, BLEUResult)

    def test_hypothesis_len_recorded(self, scorer):
        result = scorer.score("the cat sat", "the dog ran")
        assert result.hypothesis_len == 3

    def test_reference_len_recorded(self, scorer):
        result = scorer.score("the cat sat", "the dog ran fast here")
        assert result.reference_len == 5

    def test_empty_hypothesis_no_crash(self, scorer):
        result = scorer.score("", "the cat sat on the mat")
        assert result.score >= 0.0
        assert result.hypothesis_len == 0

    def test_empty_reference_no_crash(self, scorer):
        result = scorer.score("the cat sat", "")
        assert result.score >= 0.0

    def test_both_empty_no_crash(self, scorer):
        result = scorer.score("", "")
        assert isinstance(result, BLEUResult)

    def test_score_between_zero_and_one(self, scorer):
        result = scorer.score("hello world foo bar", "hello world baz qux")
        assert 0.0 <= result.score <= 1.0


# ---------------------------------------------------------------------------
# corpus_score
# ---------------------------------------------------------------------------

class TestCorpusScore:
    def test_length_mismatch_raises_value_error(self, scorer):
        with pytest.raises(ValueError):
            scorer.corpus_score(["a", "b"], ["c"])

    def test_identical_pairs_corpus_bleu_one(self, scorer):
        # Use sentences long enough for all 4-grams to exist
        sent = "the cat sat on the mat today"
        hyps = [sent, sent]
        refs = [sent, sent]
        result = scorer.corpus_score(hyps, refs)
        assert math.isclose(result.score, 1.0, rel_tol=1e-6)

    def test_corpus_returns_bleu_result(self, scorer):
        result = scorer.corpus_score(["hello"], ["hello"])
        assert isinstance(result, BLEUResult)

    def test_corpus_empty_lists_no_crash(self, scorer):
        result = scorer.corpus_score([], [])
        assert isinstance(result, BLEUResult)
        assert result.score == 0.0

    def test_corpus_precisions_length_equals_max_n(self, scorer):
        result = scorer.corpus_score(["hello world"], ["hello world"], max_n=3)
        assert len(result.precisions) == 3

    def test_corpus_score_mixed_pairs(self, scorer):
        hyps = ["the cat sat on the mat", "foo bar"]
        refs = ["the cat sat on the mat", "baz qux"]
        result = scorer.corpus_score(hyps, refs)
        # first pair is perfect, second has no overlap → mean < 1.0
        assert 0.0 < result.score < 1.0


# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_has_default(self):
        assert "default" in BLEU_SCORER_REGISTRY

    def test_registry_default_is_bleu_scorer_class(self):
        assert BLEU_SCORER_REGISTRY["default"] is BLEUScorer

    def test_registry_default_instantiable(self):
        cls = BLEU_SCORER_REGISTRY["default"]
        instance = cls()
        assert isinstance(instance, BLEUScorer)
