"""Tests for LMFilter — perplexity-based learning zone filtering."""
import math
import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.data.lm_filter import (
    LMFilterConfig,
    FilterResult,
    LMFilter,
    compute_perplexity_batch,
    filter_by_perplexity,
    filter_by_reward,
    learning_zone_weights,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_model():
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=32,
    )
    model = AureliusTransformer(cfg)
    model.eval()
    return model


@pytest.fixture(scope="module")
def sample_dataset():
    """8 variable-length 1-D input_ids tensors."""
    torch.manual_seed(42)
    return [torch.randint(0, 256, (torch.randint(8, 17, ()).item(),)) for _ in range(8)]


# ---------------------------------------------------------------------------
# 1. test_filter_config_defaults
# ---------------------------------------------------------------------------

def test_filter_config_defaults():
    cfg = LMFilterConfig()
    assert cfg.ppl_low == 5.0
    assert cfg.ppl_high == 100.0
    assert cfg.reward_threshold is None
    assert cfg.batch_size == 8


# ---------------------------------------------------------------------------
# 2. test_compute_perplexity_batch_shape
# ---------------------------------------------------------------------------

def test_compute_perplexity_batch_shape(small_model, sample_dataset):
    ppls = compute_perplexity_batch(small_model, sample_dataset)
    assert isinstance(ppls, list)
    assert len(ppls) == len(sample_dataset)
    assert all(isinstance(p, float) for p in ppls)


# ---------------------------------------------------------------------------
# 3. test_compute_perplexity_batch_positive
# ---------------------------------------------------------------------------

def test_compute_perplexity_batch_positive(small_model, sample_dataset):
    ppls = compute_perplexity_batch(small_model, sample_dataset)
    assert all(p >= 1.0 for p in ppls), f"Some ppl < 1.0: {ppls}"


# ---------------------------------------------------------------------------
# 4. test_filter_by_perplexity_keeps_zone
# ---------------------------------------------------------------------------

def test_filter_by_perplexity_keeps_zone(small_model, sample_dataset):
    # Wide range — expect all kept
    cfg = LMFilterConfig(ppl_low=1.0, ppl_high=1e9)
    kept, scores = filter_by_perplexity(small_model, sample_dataset, cfg)
    assert len(kept) == len(sample_dataset)
    assert len(scores) == len(kept)


# ---------------------------------------------------------------------------
# 5. test_filter_by_perplexity_removes_easy
# ---------------------------------------------------------------------------

def test_filter_by_perplexity_removes_easy(small_model):
    # ppl_low very high -> everything removed (too easy relative to threshold)
    cfg = LMFilterConfig(ppl_low=1e9, ppl_high=1e12)
    torch.manual_seed(7)
    dataset = [torch.randint(0, 256, (16,)) for _ in range(4)]
    kept, scores = filter_by_perplexity(small_model, dataset, cfg)
    assert len(kept) == 0


# ---------------------------------------------------------------------------
# 6. test_filter_by_perplexity_removes_noisy
# ---------------------------------------------------------------------------

def test_filter_by_perplexity_removes_noisy(small_model):
    # ppl_high very low -> everything is "noisy", gets removed
    cfg = LMFilterConfig(ppl_low=0.0, ppl_high=0.5)
    torch.manual_seed(7)
    dataset = [torch.randint(0, 256, (16,)) for _ in range(4)]
    kept, scores = filter_by_perplexity(small_model, dataset, cfg)
    assert len(kept) == 0


# ---------------------------------------------------------------------------
# 7. test_lm_filter_returns_filter_result
# ---------------------------------------------------------------------------

def test_lm_filter_returns_filter_result(small_model, sample_dataset):
    cfg = LMFilterConfig(ppl_low=1.0, ppl_high=1e9)
    lmf = LMFilter(small_model, cfg)
    result = lmf.filter(sample_dataset)
    assert isinstance(result, FilterResult)
    assert hasattr(result, "kept")
    assert hasattr(result, "rejected")
    assert hasattr(result, "rejection_rate")


# ---------------------------------------------------------------------------
# 8. test_lm_filter_rejection_rate_range
# ---------------------------------------------------------------------------

def test_lm_filter_rejection_rate_range(small_model, sample_dataset):
    cfg = LMFilterConfig(ppl_low=1.0, ppl_high=1e9)
    lmf = LMFilter(small_model, cfg)
    result = lmf.filter(sample_dataset)
    assert 0.0 <= result.rejection_rate <= 1.0


# ---------------------------------------------------------------------------
# 9. test_lm_filter_stats_dict
# ---------------------------------------------------------------------------

def test_lm_filter_stats_dict(small_model, sample_dataset):
    cfg = LMFilterConfig(ppl_low=1.0, ppl_high=1e9)
    lmf = LMFilter(small_model, cfg)
    result = lmf.filter(sample_dataset)
    assert isinstance(result.stats, dict)
    assert "n_kept" in result.stats
    assert "n_rejected" in result.stats
    assert "mean_ppl_kept" in result.stats


# ---------------------------------------------------------------------------
# 10. test_learning_zone_weights_sum_to_one
# ---------------------------------------------------------------------------

def test_learning_zone_weights_sum_to_one():
    ppls = [5.0, 10.0, 20.0, 40.0, 80.0]
    weights = learning_zone_weights(ppls, target_ppl=20.0)
    assert abs(sum(weights) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 11. test_learning_zone_weights_peak_at_target
# ---------------------------------------------------------------------------

def test_learning_zone_weights_peak_at_target():
    target = 20.0
    ppls = [5.0, 10.0, 20.0, 40.0, 80.0]
    weights = learning_zone_weights(ppls, target_ppl=target)
    # The sample at target_ppl (index 2) should have the highest weight
    assert weights[2] == max(weights)


# ---------------------------------------------------------------------------
# 12. test_filter_empty_dataset
# ---------------------------------------------------------------------------

def test_filter_empty_dataset(small_model):
    cfg = LMFilterConfig()
    lmf = LMFilter(small_model, cfg)
    result = lmf.filter([])
    assert result.kept == []
    assert result.rejected == []
    assert result.rejection_rate == 0.0
