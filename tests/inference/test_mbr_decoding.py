"""
Tests for src/inference/mbr_decoding.py

Tiny configs: vocab=16, seq_len≤8, batch≤2, n_layers≤4, d_model≤32.
Every test runs actual forward / backward passes where applicable.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.inference.mbr_decoding import (
    HypothesisPool,
    MBRAnalytics,
    MBRDecoder,
    MBRScorer,
    SequenceSimilarity,
)

# ---------------------------------------------------------------------------
# Shared tiny model fixture
# ---------------------------------------------------------------------------

VOCAB = 16
D_MODEL = 8


class TinyLM(nn.Module):
    """Minimal LM: Embedding → Linear projection to vocab logits."""

    def __init__(self, vocab: int = VOCAB, d_model: int = D_MODEL) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (B, T) → logits: (B, T, V)
        x = self.embed(input_ids)  # (B, T, d_model)
        return self.proj(x)  # (B, T, V)


@pytest.fixture()
def tiny_model():
    torch.manual_seed(0)
    return TinyLM()


@pytest.fixture()
def input_ids():
    """Prompt: batch=1, length=4, tokens in [0, VOCAB)."""
    return torch.tensor([[1, 2, 3, 4]], dtype=torch.long)


# ---------------------------------------------------------------------------
# Helper pools
# ---------------------------------------------------------------------------


def make_pool(*seqs_and_lps):
    """make_pool([1,2,3], -1.0, [4,5,6], -2.0, ...)"""
    seqs = list(seqs_and_lps[0::2])
    lps = list(seqs_and_lps[1::2])
    return HypothesisPool(seqs, lps)


# ===========================================================================
# SequenceSimilarity tests
# ===========================================================================


class TestSequenceSimilarity:
    # 1. ngram_f1: identical sequences → 1.0
    def test_ngram_f1_identical(self):
        sim = SequenceSimilarity(mode="ngram_f1")
        seq = [1, 2, 3, 4, 5]
        assert sim(seq, seq) == pytest.approx(1.0)

    # 2. ngram_f1: fully disjoint sequences → 0.0
    def test_ngram_f1_disjoint(self):
        sim = SequenceSimilarity(mode="ngram_f1")
        # No shared bigrams
        hyp = [0, 1]
        ref = [2, 3]
        assert sim(hyp, ref) == pytest.approx(0.0)

    # 3. ngram_f1: partial overlap → strictly between 0 and 1
    def test_ngram_f1_partial(self):
        sim = SequenceSimilarity(mode="ngram_f1")
        hyp = [1, 2, 3, 4]
        ref = [3, 4, 5, 6]
        result = sim(hyp, ref)
        assert 0.0 < result < 1.0

    # 4. exact_match: identical → 1.0, different → 0.0
    def test_exact_match(self):
        sim = SequenceSimilarity(mode="exact_match")
        assert sim([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
        assert sim([1, 2, 3], [1, 2, 4]) == pytest.approx(0.0)

    # 5. token_overlap: symmetric f(a,b) == f(b,a)
    def test_token_overlap_symmetric(self):
        sim = SequenceSimilarity(mode="token_overlap")
        a = [1, 2, 3, 4]
        b = [3, 4, 5, 6]
        assert sim(a, b) == pytest.approx(sim(b, a))


# ===========================================================================
# HypothesisPool tests
# ===========================================================================


class TestHypothesisPool:
    # 6. normalized_probs sum to 1.0
    def test_normalized_probs_sum(self):
        pool = make_pool([1, 2], -1.0, [3, 4], -2.0, [5, 6], -0.5)
        total = sum(pool.normalized_probs)
        assert total == pytest.approx(1.0, abs=1e-5)

    # 7. sample returns n sequences all drawn from the pool
    def test_sample_returns_n_from_pool(self):
        pool = make_pool([1, 2], -1.0, [3, 4], -2.0, [5, 6], -0.5)
        pool_set = {tuple(s) for s in pool.sequences}
        samples = pool.sample(5, with_replacement=True)
        assert len(samples) == 5
        for s in samples:
            assert tuple(s) in pool_set

    # 8. top_k returns k sequences with highest log_probs
    def test_top_k(self):
        pool = make_pool(
            [1],
            -3.0,
            [2],
            -1.0,
            [3],
            -0.5,
            [4],
            -2.0,
        )
        top2 = pool.top_k(2)
        assert len(top2) == 2
        # Both should have log_prob ≥ all others not in top2
        top2_lps = set(top2.log_probs)
        [lp for lp in pool.log_probs if lp not in top2_lps]
        # Every returned lp must be ≥ every excluded lp
        # (Use sorted approach for robustness)
        sorted_all = sorted(pool.log_probs, reverse=True)
        threshold = sorted_all[1]  # second-highest
        for lp in top2.log_probs:
            assert lp >= threshold - 1e-9

    # 9. deduplicate: no duplicate sequences in output
    def test_deduplicate_no_duplicates(self):
        pool = HypothesisPool(
            [[1, 2], [3, 4], [1, 2], [5, 6], [3, 4]],
            [-1.0, -2.0, -1.5, -0.5, -2.5],
        )
        deduped = pool.deduplicate()
        keys = [tuple(s) for s in deduped.sequences]
        assert len(keys) == len(set(keys)), "Duplicate sequences found after deduplication"
        assert len(deduped) == 3


# ===========================================================================
# MBRScorer tests
# ===========================================================================


class TestMBRScorer:
    # 10. score_hypothesis returns float in [0, 1]
    def test_score_hypothesis_in_range(self):
        sim = SequenceSimilarity(mode="ngram_f1")
        scorer = MBRScorer(sim, n_references=4)
        pool = HypothesisPool(
            [[1, 2, 3], [2, 3, 4], [5, 6, 7], [1, 3, 5]],
            [-1.0, -0.5, -2.0, -1.5],
        )
        score = scorer.score_hypothesis([1, 2, 3], pool)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    # 11. score_all length matches pool size
    def test_score_all_length(self):
        sim = SequenceSimilarity(mode="token_overlap")
        scorer = MBRScorer(sim, n_references=3)
        pool = HypothesisPool(
            [[1, 2], [3, 4], [5, 6], [7, 8], [1, 3]],
            [-1.0, -2.0, -0.5, -3.0, -1.5],
        )
        scores = scorer.score_all(pool)
        assert len(scores) == len(pool)


# ===========================================================================
# MBRDecoder tests
# ===========================================================================


class TestMBRDecoder:
    # 12. _sample_sequence: returns (list[int], float), ids in valid vocab range
    def test_sample_sequence_valid(self, tiny_model, input_ids):
        sim = SequenceSimilarity(mode="ngram_f1")
        decoder = MBRDecoder(tiny_model, sim, n_samples=3, temperature=1.0)
        seq, lp = decoder._sample_sequence(input_ids, max_new_tokens=5)
        assert isinstance(seq, list)
        assert isinstance(lp, float)
        assert len(seq) == 5
        assert all(0 <= t < VOCAB for t in seq)
        # log_prob should be negative (log of prob ≤ 1)
        assert lp <= 0.0

    # 13. decode: best_sequence is list[int], mbr_scores len == n_samples
    def test_decode_output_shape(self, tiny_model, input_ids):
        sim = SequenceSimilarity(mode="ngram_f1")
        n_samples = 4
        decoder = MBRDecoder(tiny_model, sim, n_samples=n_samples, temperature=1.0)
        best_seq, mbr_scores = decoder.decode(input_ids, max_new_tokens=4)
        assert isinstance(best_seq, list)
        assert all(isinstance(t, int) for t in best_seq)
        assert len(mbr_scores) == n_samples


# ===========================================================================
# MBRAnalytics tests
# ===========================================================================


class TestMBRAnalytics:
    # 14. diversity: returns float in [0, 1]
    def test_diversity_in_range(self):
        analytics = MBRAnalytics()
        pool = HypothesisPool(
            [[1, 2, 3], [4, 5, 6], [1, 5, 3], [7, 8, 9]],
            [-1.0, -2.0, -0.5, -1.5],
        )
        d = analytics.diversity(pool)
        assert isinstance(d, float)
        assert 0.0 <= d <= 1.0

    # 15. confidence: returns float ≥ 0
    def test_confidence_non_negative(self):
        analytics = MBRAnalytics()
        scores = [0.2, 0.5, 0.3, 0.8, 0.1]
        c = analytics.confidence(scores)
        assert isinstance(c, float)
        assert c >= 0.0

    # 16. length_stats: correct keys and min_len ≤ max_len
    def test_length_stats_keys_and_ordering(self):
        analytics = MBRAnalytics()
        pool = HypothesisPool(
            [[1, 2], [3, 4, 5], [6], [7, 8, 9, 10]],
            [-1.0, -2.0, -0.5, -3.0],
        )
        stats = analytics.length_stats(pool)
        assert set(stats.keys()) == {"mean_len", "std_len", "min_len", "max_len"}
        assert stats["min_len"] <= stats["max_len"]
        assert stats["min_len"] == 1
        assert stats["max_len"] == 4


# ===========================================================================
# Edge-case / integration test
# ===========================================================================


class TestMBREdgeCases:
    # 17. n_samples=1: MBRDecoder.decode returns the single sample as best
    def test_n_samples_1_trivial_best(self, tiny_model, input_ids):
        sim = SequenceSimilarity(mode="exact_match")
        decoder = MBRDecoder(tiny_model, sim, n_samples=1, temperature=1.0)
        best_seq, mbr_scores = decoder.decode(input_ids, max_new_tokens=3)
        assert len(mbr_scores) == 1
        # With a single hypothesis it is trivially the best
        assert isinstance(best_seq, list)
        assert len(best_seq) == 3
