"""Tests for src/evaluation/rouge_scorer.py — ≥28 test cases."""

from __future__ import annotations

import math

import pytest

from src.evaluation.rouge_scorer import (
    ROUGE_SCORER_REGISTRY,
    RougeScorer,
    RougeScores,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def scorer():
    return RougeScorer()


# ---------------------------------------------------------------------------
# RougeScores dataclass
# ---------------------------------------------------------------------------


class TestRougeScoresFrozen:
    def test_is_frozen(self):
        rs = RougeScores(rouge1=1.0, rouge2=0.5, rougeL=0.8)
        with pytest.raises((AttributeError, TypeError)):
            rs.rouge1 = 0.0  # type: ignore[misc]

    def test_fields_accessible(self):
        rs = RougeScores(rouge1=0.1, rouge2=0.2, rougeL=0.3)
        assert rs.rouge1 == 0.1
        assert rs.rouge2 == 0.2
        assert rs.rougeL == 0.3


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_lowercases(self, scorer):
        tokens = scorer._tokenize("Hello World")
        assert tokens == ["hello", "world"]

    def test_strips_punctuation(self, scorer):
        tokens = scorer._tokenize("hello, world!")
        assert tokens == ["hello", "world"]

    def test_empty_string(self, scorer):
        assert scorer._tokenize("") == []

    def test_numbers_kept(self, scorer):
        tokens = scorer._tokenize("abc 123")
        assert "123" in tokens

    def test_whitespace_split(self, scorer):
        tokens = scorer._tokenize("a   b   c")
        assert tokens == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# _ngrams
# ---------------------------------------------------------------------------


class TestNgrams:
    def test_unigrams(self, scorer):
        tokens = ["a", "b", "c"]
        assert scorer._ngrams(tokens, 1) == [("a",), ("b",), ("c",)]

    def test_bigrams_sliding_window(self, scorer):
        tokens = ["a", "b", "c", "d"]
        assert scorer._ngrams(tokens, 2) == [("a", "b"), ("b", "c"), ("c", "d")]

    def test_trigrams(self, scorer):
        tokens = ["a", "b", "c", "d"]
        assert scorer._ngrams(tokens, 3) == [("a", "b", "c"), ("b", "c", "d")]

    def test_n_larger_than_tokens_returns_empty(self, scorer):
        assert scorer._ngrams(["a", "b"], 5) == []

    def test_empty_tokens_returns_empty(self, scorer):
        assert scorer._ngrams([], 1) == []

    def test_n_equals_length(self, scorer):
        tokens = ["x", "y", "z"]
        assert scorer._ngrams(tokens, 3) == [("x", "y", "z")]


# ---------------------------------------------------------------------------
# rouge_n
# ---------------------------------------------------------------------------


class TestRougeN:
    def test_identical_text_rouge1_is_one(self, scorer):
        assert math.isclose(scorer.rouge_n("hello world", "hello world", 1), 1.0)

    def test_no_overlap_rouge1_is_zero(self, scorer):
        assert scorer.rouge_n("foo bar", "baz qux", 1) == 0.0

    def test_partial_rouge1(self, scorer):
        # hypothesis = "the cat", reference = "the cat sat on mat"
        score = scorer.rouge_n("the cat", "the cat sat on mat", 1)
        # 2 matches out of 5 reference unigrams
        assert math.isclose(score, 2 / 5, rel_tol=1e-9)

    def test_identical_text_rouge2_is_one(self, scorer):
        assert math.isclose(scorer.rouge_n("hello world foo", "hello world foo", 2), 1.0)

    def test_no_overlap_rouge2_is_zero(self, scorer):
        assert scorer.rouge_n("a b c", "x y z", 2) == 0.0

    def test_empty_reference_returns_zero(self, scorer):
        assert scorer.rouge_n("hello", "", 1) == 0.0

    def test_empty_hypothesis_returns_zero(self, scorer):
        assert scorer.rouge_n("", "hello world", 1) == 0.0


# ---------------------------------------------------------------------------
# rouge_l
# ---------------------------------------------------------------------------


class TestRougeL:
    def test_identical_text_rouge_l_is_one(self, scorer):
        assert math.isclose(scorer.rouge_l("hello world", "hello world"), 1.0)

    def test_no_overlap_rouge_l_is_zero(self, scorer):
        assert scorer.rouge_l("foo bar", "baz qux") == 0.0

    def test_empty_reference_returns_zero(self, scorer):
        assert scorer.rouge_l("hello", "") == 0.0

    def test_empty_hypothesis_returns_zero(self, scorer):
        assert scorer.rouge_l("", "hello world") == 0.0

    def test_partial_lcs(self, scorer):
        # "the cat" vs "the cat sat on mat" — LCS is ["the","cat"] = 2; ref len = 5
        score = scorer.rouge_l("the cat", "the cat sat on mat")
        assert math.isclose(score, 2 / 5, rel_tol=1e-9)

    def test_lcs_not_contiguous(self, scorer):
        # "a c" vs "a b c" — LCS = ["a","c"] = 2; ref len = 3
        score = scorer.rouge_l("a c", "a b c")
        assert math.isclose(score, 2 / 3, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# score (combined)
# ---------------------------------------------------------------------------


class TestScore:
    def test_score_returns_rouge_scores(self, scorer):
        result = scorer.score("hello world", "hello world")
        assert isinstance(result, RougeScores)

    def test_score_identical_all_ones(self, scorer):
        result = scorer.score("the cat sat", "the cat sat")
        assert math.isclose(result.rouge1, 1.0)
        assert math.isclose(result.rouge2, 1.0)
        assert math.isclose(result.rougeL, 1.0)

    def test_score_no_overlap_all_zeros(self, scorer):
        result = scorer.score("foo bar", "baz qux")
        assert result.rouge1 == 0.0
        assert result.rouge2 == 0.0
        assert result.rougeL == 0.0


# ---------------------------------------------------------------------------
# corpus_score
# ---------------------------------------------------------------------------


class TestCorpusScore:
    def test_corpus_score_mean_of_pairs(self, scorer):
        hyps = ["hello world", "foo bar"]
        refs = ["hello world", "foo bar"]
        result = scorer.corpus_score(hyps, refs)
        assert math.isclose(result.rouge1, 1.0)

    def test_corpus_score_length_mismatch_raises(self, scorer):
        with pytest.raises(ValueError):
            scorer.corpus_score(["a", "b"], ["c"])

    def test_corpus_score_empty_lists(self, scorer):
        result = scorer.corpus_score([], [])
        assert result.rouge1 == 0.0
        assert result.rouge2 == 0.0
        assert result.rougeL == 0.0

    def test_corpus_score_returns_rouge_scores(self, scorer):
        result = scorer.corpus_score(["hello"], ["hello"])
        assert isinstance(result, RougeScores)

    def test_corpus_score_partial_mean(self, scorer):
        # pair1: "the cat" vs "the cat sat on mat" → rouge1 = 2/5
        # pair2: "hello" vs "hello"                → rouge1 = 1.0
        hyps = ["the cat", "hello"]
        refs = ["the cat sat on mat", "hello"]
        result = scorer.corpus_score(hyps, refs)
        expected_rouge1 = (2 / 5 + 1.0) / 2
        assert math.isclose(result.rouge1, expected_rouge1, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registry_has_default(self):
        assert "default" in ROUGE_SCORER_REGISTRY

    def test_registry_default_is_rouge_scorer_class(self):
        assert ROUGE_SCORER_REGISTRY["default"] is RougeScorer

    def test_registry_default_instantiable(self):
        cls = ROUGE_SCORER_REGISTRY["default"]
        instance = cls()
        assert isinstance(instance, RougeScorer)
