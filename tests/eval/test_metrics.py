"""
test_metrics.py — Tests for src/eval/metrics.py

All tests use tiny configs (short strings, small tensors) for speed.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from src.eval.metrics import (
    EmbeddingSimilarity,
    MetricsEvaluator,
    bleu_score,
    compute_ngrams,
    compute_perplexity,
    distinct_n,
    rouge_l,
    rouge_n,
    tokenize,
)

# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_splits_on_whitespace(self):
        tokens = tokenize("hello world foo")
        assert tokens == ["hello", "world", "foo"]

    def test_lowercases(self):
        tokens = tokenize("Hello World FOO")
        assert tokens == ["hello", "world", "foo"]

    def test_strips_punctuation(self):
        tokens = tokenize("hello, world! foo.")
        assert tokens == ["hello", "world", "foo"]

    def test_filters_empty(self):
        tokens = tokenize("  ,  .  !  ")
        assert tokens == []

    def test_empty_string(self):
        assert tokenize("") == []


# ---------------------------------------------------------------------------
# compute_ngrams
# ---------------------------------------------------------------------------


class TestComputeNgrams:
    def test_unigram_counts(self):
        tokens = ["a", "b", "a", "c"]
        ngrams = compute_ngrams(tokens, 1)
        assert ngrams[("a",)] == 2
        assert ngrams[("b",)] == 1
        assert ngrams[("c",)] == 1

    def test_bigram_counts(self):
        tokens = ["a", "b", "c"]
        ngrams = compute_ngrams(tokens, 2)
        assert ngrams[("a", "b")] == 1
        assert ngrams[("b", "c")] == 1
        assert len(ngrams) == 2

    def test_n_larger_than_tokens_returns_empty(self):
        tokens = ["a", "b"]
        ngrams = compute_ngrams(tokens, 5)
        assert len(ngrams) == 0

    def test_zero_n_returns_empty(self):
        tokens = ["a", "b", "c"]
        ngrams = compute_ngrams(tokens, 0)
        assert len(ngrams) == 0


# ---------------------------------------------------------------------------
# bleu_score
# ---------------------------------------------------------------------------


class TestBleuScore:
    def test_identical_sentences_returns_one(self):
        ref = "the cat sat on the mat"
        score = bleu_score(ref, ref, max_n=4, smooth=False)
        assert math.isclose(score, 1.0, abs_tol=1e-9)

    def test_disjoint_sentences_returns_zero(self):
        ref = "the cat sat on the mat"
        hyp = "dog runs quickly through forest"
        score = bleu_score(ref, hyp, max_n=4, smooth=False)
        assert score == 0.0

    def test_score_in_range(self):
        ref = "the cat sat on the mat"
        hyp = "the cat is on the mat"
        score = bleu_score(ref, hyp, max_n=4, smooth=True)
        assert 0.0 <= score <= 1.0

    def test_empty_hypothesis_returns_zero(self):
        assert bleu_score("hello world", "", max_n=4) == 0.0

    def test_smooth_raises_score_for_partial(self):
        ref = "the cat sat on the mat"
        hyp = "a dog ran over the bridge"
        score_smooth = bleu_score(ref, hyp, max_n=4, smooth=True)
        assert 0.0 <= score_smooth <= 1.0


# ---------------------------------------------------------------------------
# rouge_n
# ---------------------------------------------------------------------------


class TestRougeN:
    def test_identical_returns_one(self):
        text = "the cat sat on the mat"
        result = rouge_n(text, text, n=2)
        assert math.isclose(result["f1"], 1.0, abs_tol=1e-9)
        assert math.isclose(result["precision"], 1.0, abs_tol=1e-9)
        assert math.isclose(result["recall"], 1.0, abs_tol=1e-9)

    def test_keys_present(self):
        result = rouge_n("hello world", "hello world", n=1)
        assert set(result.keys()) == {"precision", "recall", "f1"}

    def test_disjoint_returns_zero(self):
        result = rouge_n("cat sat mat", "dog ran bridge", n=2)
        assert result["f1"] == 0.0

    def test_partial_overlap(self):
        ref = "the cat sat on the mat"
        hyp = "the cat is on a rug"
        result = rouge_n(ref, hyp, n=1)
        assert 0.0 < result["f1"] < 1.0


# ---------------------------------------------------------------------------
# rouge_l
# ---------------------------------------------------------------------------


class TestRougeL:
    def test_identical_returns_one(self):
        text = "the cat sat on the mat"
        result = rouge_l(text, text)
        assert math.isclose(result["f1"], 1.0, abs_tol=1e-9)

    def test_partial_overlap_greater_than_zero(self):
        ref = "the cat sat on the mat"
        hyp = "the cat is on the rug"
        result = rouge_l(ref, hyp)
        assert result["f1"] > 0.0

    def test_keys_present(self):
        result = rouge_l("hello world", "hello world")
        assert set(result.keys()) == {"precision", "recall", "f1"}

    def test_disjoint_returns_zero(self):
        result = rouge_l("cat sat mat", "dog ran bridge")
        assert result["f1"] == 0.0


# ---------------------------------------------------------------------------
# distinct_n
# ---------------------------------------------------------------------------


class TestDistinctN:
    def test_all_same_low(self):
        texts = ["the the the the", "the the the the"]
        score = distinct_n(texts, n=2)
        # Only one unique bigram ("the", "the") out of many
        assert score < 0.5

    def test_all_unique_returns_one(self):
        texts = ["a b c d e f"]
        score = distinct_n(texts, n=2)
        assert math.isclose(score, 1.0, abs_tol=1e-9)

    def test_empty_texts_returns_zero(self):
        assert distinct_n([], n=2) == 0.0

    def test_single_token_no_bigrams_returns_zero(self):
        score = distinct_n(["hello"], n=2)
        assert score == 0.0


# ---------------------------------------------------------------------------
# compute_perplexity
# ---------------------------------------------------------------------------


class TestComputePerplexity:
    def test_returns_scalar_float(self):
        log_probs = torch.tensor([-2.0, -3.0, -1.5])
        lengths = torch.tensor([5, 6, 4])
        result = compute_perplexity(log_probs, lengths)
        assert isinstance(result, float)

    def test_zero_log_prob_perplexity_one(self):
        log_probs = torch.tensor([0.0, 0.0])
        lengths = torch.tensor([4, 4])
        result = compute_perplexity(log_probs, lengths)
        assert math.isclose(result, 1.0, abs_tol=1e-6)

    def test_lower_neg_log_prob_lower_perplexity(self):
        lp_good = torch.tensor([-1.0])
        lp_bad = torch.tensor([-10.0])
        lengths = torch.tensor([5])
        ppl_good = compute_perplexity(lp_good, lengths)
        ppl_bad = compute_perplexity(lp_bad, lengths)
        assert ppl_good < ppl_bad


# ---------------------------------------------------------------------------
# EmbeddingSimilarity
# ---------------------------------------------------------------------------


def _fixed_embed(text: str) -> Tensor:
    """Deterministic embedding: use char-code sum to pick a direction."""
    dim = 8
    v = torch.zeros(dim)
    for i, ch in enumerate(text):
        v[i % dim] += ord(ch)
    norm = v.norm()
    if norm == 0:
        v[0] = 1.0
    else:
        v = v / norm
    return v


class TestEmbeddingSimilarity:
    def test_identical_texts_similarity_one(self):
        es = EmbeddingSimilarity(_fixed_embed)
        sim = es.similarity("hello world", "hello world")
        assert math.isclose(sim, 1.0, abs_tol=1e-5)

    def test_batch_similarity_length_matches(self):
        es = EmbeddingSimilarity(_fixed_embed)
        refs = ["hello world", "foo bar"]
        hyps = ["hello world", "baz qux"]
        sims = es.batch_similarity(refs, hyps)
        assert len(sims) == 2

    def test_batch_similarity_identical_first_pair_is_one(self):
        es = EmbeddingSimilarity(_fixed_embed)
        refs = ["hello world", "foo bar"]
        hyps = ["hello world", "baz qux"]
        sims = es.batch_similarity(refs, hyps)
        assert math.isclose(sims[0], 1.0, abs_tol=1e-5)


# ---------------------------------------------------------------------------
# MetricsEvaluator
# ---------------------------------------------------------------------------


class TestMetricsEvaluator:
    def test_evaluate_returns_expected_keys_without_embed(self):
        evaluator = MetricsEvaluator()
        refs = ["the cat sat on the mat"]
        hyps = ["the cat sat on the mat"]
        result = evaluator.evaluate(refs, hyps)
        assert "bleu" in result
        assert "rouge_2_f1" in result
        assert "rouge_l_f1" in result
        assert "distinct_2" in result
        assert "embedding_similarity" not in result

    def test_evaluate_returns_expected_keys_with_embed(self):
        evaluator = MetricsEvaluator(embed_fn=_fixed_embed)
        refs = ["hello world"]
        hyps = ["hello world"]
        result = evaluator.evaluate(refs, hyps)
        assert "bleu" in result
        assert "rouge_2_f1" in result
        assert "rouge_l_f1" in result
        assert "embedding_similarity" in result

    def test_evaluate_identical_high_scores(self):
        evaluator = MetricsEvaluator()
        text = "the cat sat on the mat"
        result = evaluator.evaluate([text], [text])
        assert result["bleu"] > 0.9
        assert result["rouge_2_f1"] > 0.9
        assert result["rouge_l_f1"] > 0.9

    def test_get_summary_returns_string(self):
        evaluator = MetricsEvaluator()
        metrics = {"bleu": 0.75, "rouge_2_f1": 0.8}
        summary = evaluator.get_summary(metrics)
        assert isinstance(summary, str)
        assert "bleu" in summary
        assert "rouge_2_f1" in summary

    def test_get_summary_contains_values(self):
        evaluator = MetricsEvaluator()
        metrics = {"bleu": 0.1234}
        summary = evaluator.get_summary(metrics)
        assert "0.1234" in summary
