"""Tests for src/eval/generation_metrics_new.py

Covers: tokenize_simple, compute_ngrams, meteor_score, compute_idf,
        cider_score, sentence_bleu, aggregate_scores, GenerationEvaluator.
"""

from __future__ import annotations

import pytest

from src.eval.generation_metrics_new import (
    GenerationEvaluator,
    aggregate_scores,
    cider_score,
    compute_idf,
    compute_ngrams,
    meteor_score,
    sentence_bleu,
    tokenize_simple,
)

# ---------------------------------------------------------------------------
# tokenize_simple
# ---------------------------------------------------------------------------


class TestTokenizeSimple:
    def test_lowercases(self):
        """All tokens must be lowercase."""
        tokens = tokenize_simple("Hello World FOO")
        assert all(t == t.lower() for t in tokens)
        assert "hello" in tokens
        assert "world" in tokens
        assert "foo" in tokens

    def test_splits_on_punctuation(self):
        """Punctuation should act as a delimiter."""
        tokens = tokenize_simple("hello,world.foo")
        assert "hello" in tokens
        assert "world" in tokens
        assert "foo" in tokens
        # No punctuation chars in tokens
        for t in tokens:
            assert t.isalpha() or t.isdigit()

    def test_filters_empty_strings(self):
        """Empty strings are filtered out."""
        tokens = tokenize_simple("   ,,,   ")
        assert tokens == []

    def test_simple_sentence(self):
        """Basic sentence tokenization."""
        tokens = tokenize_simple("the quick brown fox")
        assert tokens == ["the", "quick", "brown", "fox"]


# ---------------------------------------------------------------------------
# compute_ngrams
# ---------------------------------------------------------------------------


class TestComputeNgrams:
    def test_bigram_count(self):
        """Bigram counts should match manual count."""
        tokens = ["a", "b", "a", "b"]
        ngrams = compute_ngrams(tokens, 2)
        assert ngrams[("a", "b")] == 2
        assert ngrams[("b", "a")] == 1

    def test_unigrams(self):
        tokens = ["x", "y", "x"]
        ngrams = compute_ngrams(tokens, 1)
        assert ngrams[("x",)] == 2
        assert ngrams[("y",)] == 1

    def test_n_larger_than_tokens(self):
        """If n > len(tokens), return empty Counter."""
        ngrams = compute_ngrams(["a", "b"], 5)
        assert len(ngrams) == 0

    def test_returns_counter(self):
        from collections import Counter

        result = compute_ngrams(["a", "b", "c"], 2)
        assert isinstance(result, Counter)


# ---------------------------------------------------------------------------
# meteor_score
# ---------------------------------------------------------------------------


class TestMeteorScore:
    def test_perfect_match_is_one(self):
        """Identical hypothesis and reference should yield a score very close to 1.0.

        The chunk penalty for a perfect match is minimal (single contiguous run),
        so the score is >= 0.99.
        """
        text = "the cat sat on the mat"
        score = meteor_score(text, text)
        assert score >= 0.99

    def test_empty_hypothesis_is_zero(self):
        """Empty hypothesis must return 0.0."""
        score = meteor_score("", "the cat sat on the mat")
        assert score == pytest.approx(0.0)

    def test_partial_overlap_gt_zero(self):
        """Partially overlapping texts should score > 0."""
        hyp = "the cat sat"
        ref = "the cat sat on the mat"
        score = meteor_score(hyp, ref)
        assert score > 0.0

    def test_no_overlap_is_zero(self):
        """No common words should return 0.0."""
        score = meteor_score("apple banana cherry", "dog elephant fox")
        assert score == pytest.approx(0.0)

    def test_range(self):
        """Score must be in [0, 1]."""
        score = meteor_score("hello world test", "hello world example")
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# compute_idf
# ---------------------------------------------------------------------------


class TestComputeIdf:
    def test_returns_dict_with_correct_keys(self):
        """All corpus words appear as keys in the IDF dict."""
        corpus = ["the cat sat", "the dog ran"]
        idf = compute_idf(corpus)
        assert isinstance(idf, dict)
        # Words from corpus should be in idf
        for word in ["the", "cat", "sat", "dog", "ran"]:
            assert word in idf

    def test_rare_word_higher_idf(self):
        """Word appearing in fewer docs should have higher IDF."""
        corpus = ["apple banana", "apple cherry", "apple date"]
        idf = compute_idf(corpus)
        # 'apple' appears in all 3 docs; 'banana' in only 1
        assert idf["banana"] > idf["apple"]

    def test_idf_positive(self):
        """All IDF values should be >= 0."""
        corpus = ["foo bar", "baz qux", "foo baz"]
        idf = compute_idf(corpus)
        for v in idf.values():
            assert v >= 0.0


# ---------------------------------------------------------------------------
# cider_score
# ---------------------------------------------------------------------------


class TestCiderScore:
    def test_identical_hypothesis_and_reference(self):
        """Identical hypothesis and single reference → high score (>= 5.0)."""
        text = "the quick brown fox jumps over the lazy dog"
        score = cider_score(text, [text])
        assert score >= 5.0

    def test_range(self):
        """Score must be clipped to [0, 10]."""
        score = cider_score("hello world", ["completely different text here"])
        assert 0.0 <= score <= 10.0

    def test_empty_references(self):
        """Empty reference list should return 0.0."""
        score = cider_score("hello world", [])
        assert score == pytest.approx(0.0)

    def test_multiple_references(self):
        """Multiple references should not crash and score is in [0, 10]."""
        hyp = "the cat sat on the mat"
        refs = ["the cat sat on the mat", "a cat was sitting on mat", "cat mat sat"]
        score = cider_score(hyp, refs)
        assert 0.0 <= score <= 10.0


# ---------------------------------------------------------------------------
# sentence_bleu
# ---------------------------------------------------------------------------


class TestSentenceBleu:
    def test_identical_hypothesis_is_one(self):
        """Identical hypothesis and reference → BLEU = 1.0."""
        text = "the quick brown fox jumps over the lazy dog"
        score = sentence_bleu(text, text)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_empty_hypothesis_is_zero(self):
        """Empty hypothesis must return 0.0."""
        score = sentence_bleu("", "the cat sat on the mat")
        assert score == pytest.approx(0.0)

    def test_range(self):
        """BLEU must be in [0, 1]."""
        score = sentence_bleu("the cat", "the cat sat on the mat")
        assert 0.0 <= score <= 1.0

    def test_partial_match_gt_zero(self):
        """Partially matching hypothesis should score > 0."""
        score = sentence_bleu("the cat sat", "the cat sat on the mat")
        assert score > 0.0

    def test_brevity_penalty_applied(self):
        """Short hypothesis vs long reference should get brevity-penalized."""
        score_short = sentence_bleu("the cat", "the cat sat on the mat")
        score_full = sentence_bleu("the cat sat on the mat", "the cat sat on the mat")
        assert score_full > score_short


# ---------------------------------------------------------------------------
# aggregate_scores
# ---------------------------------------------------------------------------


class TestAggregateScores:
    HYPS = ["the cat sat on the mat", "the dog ran fast", "hello world"]
    REFS = ["the cat sat on the mat", "a dog ran very fast", "hello there world"]

    def test_returns_correct_keys(self):
        """Result dict must have 'mean', 'min', 'max' keys."""
        result = aggregate_scores(self.HYPS, self.REFS, "bleu")
        assert set(result.keys()) == {"mean", "min", "max"}

    def test_bleu_mean_in_range(self):
        """Corpus-level BLEU mean must be in [0, 1]."""
        result = aggregate_scores(self.HYPS, self.REFS, "bleu")
        assert 0.0 <= result["mean"] <= 1.0

    def test_meteor_keys_present(self):
        result = aggregate_scores(self.HYPS, self.REFS, "meteor")
        assert "mean" in result and "min" in result and "max" in result

    def test_cider_keys_present(self):
        result = aggregate_scores(self.HYPS, self.REFS, "cider")
        assert "mean" in result and "min" in result and "max" in result

    def test_min_le_mean_le_max(self):
        """Ordering invariant: min <= mean <= max."""
        for metric in ("bleu", "meteor", "cider"):
            result = aggregate_scores(self.HYPS, self.REFS, metric)
            assert result["min"] <= result["mean"] <= result["max"], metric

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError):
            aggregate_scores(self.HYPS, self.REFS, "unknown_metric")


# ---------------------------------------------------------------------------
# GenerationEvaluator
# ---------------------------------------------------------------------------


class TestGenerationEvaluator:
    def test_score_pair_has_all_metric_keys(self):
        """score_pair returns a dict with all default metric keys."""
        evaluator = GenerationEvaluator()
        result = evaluator.score_pair("the cat sat", "the cat sat on the mat")
        assert "bleu" in result
        assert "meteor" in result
        assert "cider" in result

    def test_score_pair_custom_metrics(self):
        """Custom metrics list is respected."""
        evaluator = GenerationEvaluator(metrics=["bleu"])
        result = evaluator.score_pair("hello world", "hello world")
        assert set(result.keys()) == {"bleu"}

    def test_score_corpus_returns_nested_dict(self):
        """score_corpus returns {metric: {mean, min, max}}."""
        evaluator = GenerationEvaluator()
        hyps = ["the cat sat", "hello world"]
        refs = ["the cat sat on the mat", "hello there world"]
        result = evaluator.score_corpus(hyps, refs)
        assert isinstance(result, dict)
        for metric in ("bleu", "meteor", "cider"):
            assert metric in result
            assert isinstance(result[metric], dict)
            assert set(result[metric].keys()) == {"mean", "min", "max"}

    def test_score_pair_values_in_range(self):
        """All returned metric values should be non-negative."""
        evaluator = GenerationEvaluator()
        result = evaluator.score_pair("quick brown fox", "the quick brown fox jumps")
        for k, v in result.items():
            assert v >= 0.0, f"{k} is negative: {v}"

    def test_default_metrics(self):
        """Default metrics list is ['bleu', 'meteor', 'cider']."""
        evaluator = GenerationEvaluator()
        assert set(evaluator.metrics) == {"bleu", "meteor", "cider"}
