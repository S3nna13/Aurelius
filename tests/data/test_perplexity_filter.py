"""Tests for src/data/perplexity_filter.py — ≥10 tests covering config
defaults, tensor shapes/dtypes, perplexity correctness, filtering thresholds,
stats keys, ranker ordering, percentile buckets, empty/single-element edge
cases, and lengths masking."""

from __future__ import annotations

import math
from typing import List

import pytest
import torch
from torch import Tensor

from src.data.perplexity_filter import (
    DatasetPerplexityRanker,
    PerplexityFilter,
    PerplexityFilterConfig,
    PerplexityScorer,
    compute_sequence_perplexity,
    compute_token_log_probs,
)

# ---------------------------------------------------------------------------
# Shared constants / fixtures
# ---------------------------------------------------------------------------

VOCAB_SIZE = 32  # tiny vocab for fast tests


def make_model_fn(vocab_size: int = VOCAB_SIZE):
    """Return a deterministic (seeded) random mock model function."""

    def _model_fn(ids: Tensor) -> Tensor:  # (B, T) -> (B, T, V)
        torch.manual_seed(0)
        return torch.randn(ids.shape[0], ids.shape[1], vocab_size)

    return _model_fn


def make_scorer(config: PerplexityFilterConfig | None = None) -> PerplexityScorer:
    cfg = config or PerplexityFilterConfig()
    return PerplexityScorer(model_fn=make_model_fn(), config=cfg)


# ---------------------------------------------------------------------------
# 1. Config defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = PerplexityFilterConfig()
    assert cfg.min_perplexity == 10.0
    assert cfg.max_perplexity == 1000.0
    assert cfg.batch_size == 8
    assert cfg.max_seq_len == 512


# ---------------------------------------------------------------------------
# 2. compute_token_log_probs — output shape is (B, T)
# ---------------------------------------------------------------------------


def test_compute_token_log_probs_shape():
    B, T, V = 3, 5, VOCAB_SIZE
    logits = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))
    log_probs = compute_token_log_probs(logits, targets)
    assert log_probs.shape == (B, T), f"Expected ({B}, {T}), got {log_probs.shape}"


# ---------------------------------------------------------------------------
# 3. compute_token_log_probs — values are <= 0 (log-probs of a distribution)
# ---------------------------------------------------------------------------


def test_compute_token_log_probs_non_positive():
    B, T, V = 2, 4, VOCAB_SIZE
    logits = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))
    log_probs = compute_token_log_probs(logits, targets)
    assert (log_probs <= 0).all(), "Log-probs from a softmax should be <= 0"


# ---------------------------------------------------------------------------
# 4. compute_sequence_perplexity — positive values
# ---------------------------------------------------------------------------


def test_compute_sequence_perplexity_positive():
    B, T = 3, 6
    # Use negative values to simulate log-probs
    log_probs = -torch.rand(B, T)  # all in (-1, 0]
    lengths = torch.tensor([6, 4, 2], dtype=torch.long)
    perps = compute_sequence_perplexity(log_probs, lengths)
    assert perps.shape == (B,)
    assert (perps > 0).all(), "Perplexity must be strictly positive"


# ---------------------------------------------------------------------------
# 5. compute_sequence_perplexity — lengths masking has an effect
# ---------------------------------------------------------------------------


def test_compute_sequence_perplexity_lengths_masking():
    """Two sequences with same content but different reported lengths get
    different perplexity scores."""
    B, T = 2, 8
    # Large negative values in positions 4..7 (padding region for seq 0)
    log_probs = torch.full((B, T), -0.5)
    log_probs[0, 4:] = -100.0  # poison padding positions for seq 0
    log_probs[1, 4:] = -100.0  # same poison for seq 1

    lengths_short = torch.tensor([4, 8], dtype=torch.long)  # seq 0 uses only first 4
    perps = compute_sequence_perplexity(log_probs, lengths_short)

    # Seq 0 should have low perplexity (mean of -0.5 over 4 tokens)
    # Seq 1 should have high perplexity (poisoned padding included)
    assert perps[0] < perps[1], (
        f"Seq 0 perp {perps[0]:.2f} should be < seq 1 perp {perps[1]:.2f}"
    )


# ---------------------------------------------------------------------------
# 6. PerplexityScorer.score_batch — output shape is (B,)
# ---------------------------------------------------------------------------


def test_score_batch_shape():
    cfg = PerplexityFilterConfig()
    scorer = make_scorer(cfg)
    B, T = 4, 10
    token_ids = torch.randint(0, VOCAB_SIZE, (B, T))
    lengths = torch.full((B,), T, dtype=torch.long)
    perps = scorer.score_batch(token_ids, lengths)
    assert perps.shape == (B,), f"Expected ({B},), got {perps.shape}"


# ---------------------------------------------------------------------------
# 7. PerplexityFilter — keeps sequences within threshold
# ---------------------------------------------------------------------------


def test_filter_keeps_within_threshold():
    """With a very wide perplexity window everything should be kept."""
    cfg = PerplexityFilterConfig(min_perplexity=0.0, max_perplexity=1e9)
    scorer = make_scorer(cfg)
    pfilter = PerplexityFilter(scorer=scorer, config=cfg)

    texts: List[Tensor] = [torch.randint(0, VOCAB_SIZE, (10,)) for _ in range(5)]
    kept, scores = pfilter.filter_batch(texts)
    assert len(kept) == 5, "Wide window should keep all sequences"
    assert len(scores) == 5


# ---------------------------------------------------------------------------
# 8. PerplexityFilter — removes sequences outside threshold
# ---------------------------------------------------------------------------


def test_filter_removes_outside_threshold():
    """With a tiny perplexity window [0, 0.001] nothing should be kept
    (mock model produces random logits → perplexity >> 0.001)."""
    cfg = PerplexityFilterConfig(min_perplexity=0.0, max_perplexity=0.001)
    scorer = make_scorer(cfg)
    pfilter = PerplexityFilter(scorer=scorer, config=cfg)

    texts: List[Tensor] = [torch.randint(0, VOCAB_SIZE, (10,)) for _ in range(4)]
    kept, scores = pfilter.filter_batch(texts)
    assert len(kept) == 0, "Tiny window should remove all sequences"
    assert len(scores) == 0


# ---------------------------------------------------------------------------
# 9. PerplexityFilter.get_stats — required keys present
# ---------------------------------------------------------------------------


def test_get_stats_keys_present():
    cfg = PerplexityFilterConfig(min_perplexity=10.0, max_perplexity=1000.0)
    scorer = make_scorer(cfg)
    pfilter = PerplexityFilter(scorer=scorer, config=cfg)
    scores = [15.0, 20.0, 500.0, 1500.0, 8.0]
    stats = pfilter.get_stats(scores)
    required = {"mean", "median", "min", "max", "n_filtered", "n_kept"}
    assert required == set(stats.keys()), f"Missing keys: {required - set(stats.keys())}"


# ---------------------------------------------------------------------------
# 10. PerplexityFilter.get_stats — values are consistent
# ---------------------------------------------------------------------------


def test_get_stats_values_consistent():
    cfg = PerplexityFilterConfig(min_perplexity=10.0, max_perplexity=100.0)
    scorer = make_scorer(cfg)
    pfilter = PerplexityFilter(scorer=scorer, config=cfg)
    scores = [5.0, 20.0, 50.0, 200.0]  # 5 and 200 are outside [10, 100]
    stats = pfilter.get_stats(scores)
    assert stats["min"] == pytest.approx(5.0)
    assert stats["max"] == pytest.approx(200.0)
    assert stats["n_kept"] == pytest.approx(2.0)   # 20 and 50
    assert stats["n_filtered"] == pytest.approx(2.0)  # 5 and 200
    assert stats["mean"] == pytest.approx(sum(scores) / len(scores))


# ---------------------------------------------------------------------------
# 11. DatasetPerplexityRanker — ordering (ascending perplexity)
# ---------------------------------------------------------------------------


def test_ranker_ordering():
    """rank() must return sequences sorted by perplexity ascending."""
    cfg = PerplexityFilterConfig()
    scorer = make_scorer(cfg)
    ranker = DatasetPerplexityRanker(scorer=scorer)

    texts: List[Tensor] = [torch.randint(0, VOCAB_SIZE, (8,)) for _ in range(6)]
    sorted_texts, sorted_scores = ranker.rank(texts, return_scores=True)

    assert len(sorted_texts) == 6
    assert len(sorted_scores) == 6
    # Verify ascending order.
    for i in range(len(sorted_scores) - 1):
        assert sorted_scores[i] <= sorted_scores[i + 1], (
            f"Scores not sorted at position {i}: {sorted_scores[i]} > {sorted_scores[i+1]}"
        )


# ---------------------------------------------------------------------------
# 12. DatasetPerplexityRanker — return_scores=False returns list only
# ---------------------------------------------------------------------------


def test_ranker_return_scores_false():
    cfg = PerplexityFilterConfig()
    scorer = make_scorer(cfg)
    ranker = DatasetPerplexityRanker(scorer=scorer)
    texts: List[Tensor] = [torch.randint(0, VOCAB_SIZE, (5,)) for _ in range(3)]
    result = ranker.rank(texts, return_scores=False)
    assert isinstance(result, list)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# 13. get_percentile_bucket — result in [0, n_buckets-1]
# ---------------------------------------------------------------------------


def test_get_percentile_bucket_range():
    cfg = PerplexityFilterConfig()
    scorer = make_scorer(cfg)
    ranker = DatasetPerplexityRanker(scorer=scorer)
    scores = [float(x) for x in range(1, 101)]  # 1..100
    n_buckets = 10
    for s in scores:
        bucket = ranker.get_percentile_bucket(s, scores, n_buckets=n_buckets)
        assert 0 <= bucket < n_buckets, f"Bucket {bucket} out of range for score {s}"


# ---------------------------------------------------------------------------
# 14. Empty list handling — filter_batch
# ---------------------------------------------------------------------------


def test_filter_batch_empty_list():
    cfg = PerplexityFilterConfig()
    scorer = make_scorer(cfg)
    pfilter = PerplexityFilter(scorer=scorer, config=cfg)
    kept, scores = pfilter.filter_batch([])
    assert kept == []
    assert scores == []


# ---------------------------------------------------------------------------
# 15. Empty list handling — ranker
# ---------------------------------------------------------------------------


def test_ranker_empty_list():
    cfg = PerplexityFilterConfig()
    scorer = make_scorer(cfg)
    ranker = DatasetPerplexityRanker(scorer=scorer)
    result = ranker.rank([], return_scores=False)
    assert result == []
    sorted_texts, sorted_scores = ranker.rank([], return_scores=True)
    assert sorted_texts == []
    assert sorted_scores == []


# ---------------------------------------------------------------------------
# 16. Single-element list — filter and score_texts
# ---------------------------------------------------------------------------


def test_single_element_list():
    cfg = PerplexityFilterConfig(min_perplexity=0.0, max_perplexity=1e9)
    scorer = make_scorer(cfg)
    pfilter = PerplexityFilter(scorer=scorer, config=cfg)
    texts: List[Tensor] = [torch.randint(0, VOCAB_SIZE, (7,))]
    kept, scores = pfilter.filter_batch(texts)
    assert len(kept) == 1
    assert len(scores) == 1
    assert scores[0] > 0


# ---------------------------------------------------------------------------
# 17. get_stats — empty scores returns nan values without crashing
# ---------------------------------------------------------------------------


def test_get_stats_empty():
    cfg = PerplexityFilterConfig()
    scorer = make_scorer(cfg)
    pfilter = PerplexityFilter(scorer=scorer, config=cfg)
    stats = pfilter.get_stats([])
    assert math.isnan(stats["mean"])
    assert math.isnan(stats["median"])
    assert stats["n_kept"] == 0.0
    assert stats["n_filtered"] == 0.0


# ---------------------------------------------------------------------------
# 18. score_texts — returns one float per input sequence
# ---------------------------------------------------------------------------


def test_score_texts_length():
    cfg = PerplexityFilterConfig(batch_size=3)
    scorer = make_scorer(cfg)
    texts: List[Tensor] = [torch.randint(0, VOCAB_SIZE, (i + 3,)) for i in range(7)]
    scores = scorer.score_texts(texts)
    assert len(scores) == 7
    assert all(isinstance(s, float) for s in scores)
    assert all(s > 0 for s in scores)
