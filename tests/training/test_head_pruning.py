"""Tests for attention head pruning (head_pruning.py)."""
import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.head_pruning import (
    HeadImportanceScore,
    HeadMask,
    HeadPruningConfig,
    StructuredHeadPruner,
    compute_head_importance_l1,
    estimate_flop_reduction,
)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def small_model():
    torch.manual_seed(42)
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )
    return AureliusTransformer(cfg)


# ---------------------------------------------------------------------------
# Test 1: compute_head_importance_l1 returns list of HeadImportanceScore
# ---------------------------------------------------------------------------

def test_head_importance_l1_returns_scores(small_model):
    scores = compute_head_importance_l1(small_model)
    assert isinstance(scores, list)
    assert len(scores) > 0
    assert all(isinstance(s, HeadImportanceScore) for s in scores)


# ---------------------------------------------------------------------------
# Test 2: returns n_layers * n_heads scores
# ---------------------------------------------------------------------------

def test_head_importance_l1_count(small_model):
    cfg = small_model.config
    scores = compute_head_importance_l1(small_model)
    assert len(scores) == cfg.n_layers * cfg.n_heads


# ---------------------------------------------------------------------------
# Test 3: HeadMask initialises with all ones
# ---------------------------------------------------------------------------

def test_head_mask_init_ones():
    mask = HeadMask(n_layers=2, n_heads=4)
    assert torch.all(mask.mask == 1.0).item()


# ---------------------------------------------------------------------------
# Test 4: prune_head sets mask entry to 0
# ---------------------------------------------------------------------------

def test_head_mask_prune_head():
    mask = HeadMask(n_layers=2, n_heads=4)
    mask.prune_head(0, 0)
    assert mask.mask[0, 0].item() == 0.0


# ---------------------------------------------------------------------------
# Test 5: active_heads returns correct count after pruning
# ---------------------------------------------------------------------------

def test_head_mask_active_heads():
    mask = HeadMask(n_layers=2, n_heads=4)
    total = 2 * 4  # 8
    mask.prune_head(0, 0)
    mask.prune_head(1, 2)
    assert mask.active_heads() == total - 2


# ---------------------------------------------------------------------------
# Test 6: StructuredHeadPruner.prune returns dict with required keys
# ---------------------------------------------------------------------------

def test_structured_pruner_prune_returns_stats(small_model):
    cfg = HeadPruningConfig(target_sparsity=0.5, importance_metric="l1_norm")
    pruner = StructuredHeadPruner(small_model, cfg)
    result = pruner.prune()
    assert "n_pruned" in result
    assert "n_active" in result
    assert "pruned_heads" in result


# ---------------------------------------------------------------------------
# Test 7: after prune with target=0.5, ~50% of heads are masked
# ---------------------------------------------------------------------------

def test_structured_pruner_sparsity(small_model):
    cfg = HeadPruningConfig(target_sparsity=0.5, importance_metric="l1_norm")
    pruner = StructuredHeadPruner(small_model, cfg)
    result = pruner.prune()

    model_cfg = small_model.config
    total = model_cfg.n_layers * model_cfg.n_heads
    expected_pruned = int(total * 0.5)
    # Allow off-by-one due to int rounding
    assert abs(result["n_pruned"] - expected_pruned) <= 1


# ---------------------------------------------------------------------------
# Test 8: pruning_stats returns all required keys
# ---------------------------------------------------------------------------

def test_pruning_stats_keys(small_model):
    cfg = HeadPruningConfig(target_sparsity=0.25, importance_metric="l1_norm")
    pruner = StructuredHeadPruner(small_model, cfg)
    pruner.prune()
    stats = pruner.pruning_stats()
    for key in ("total_heads", "pruned_heads", "sparsity", "per_layer_active"):
        assert key in stats


# ---------------------------------------------------------------------------
# Test 9: estimate_flop_reduction returns float in [0, 1]
# ---------------------------------------------------------------------------

def test_estimate_flop_reduction_range():
    result = estimate_flop_reduction(
        n_layers=2, n_heads=4, head_dim=16, seq_len=64, n_pruned_heads=2
    )
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Test 10: n_pruned_heads == n_heads => 1.0
# ---------------------------------------------------------------------------

def test_estimate_flop_reduction_all_pruned():
    n_heads = 4
    result = estimate_flop_reduction(
        n_layers=2, n_heads=n_heads, head_dim=16, seq_len=64, n_pruned_heads=n_heads
    )
    assert result == 1.0
