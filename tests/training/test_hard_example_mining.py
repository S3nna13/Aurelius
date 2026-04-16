"""Tests for src/training/hard_example_mining.py.

Uses tiny tensors (N=20) to keep tests fast and deterministic.
"""

from __future__ import annotations

import torch
import pytest

from src.training.hard_example_mining import (
    MiningConfig,
    HardExampleSampler,
    compute_difficulty_scores,
    focal_weight,
    loss_with_mining,
    ohem_loss,
    rank_by_difficulty,
)

N = 20
K_RATIO = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _losses(seed: int = 0) -> torch.Tensor:
    """Return reproducible (N,) non-negative loss tensor."""
    torch.manual_seed(seed)
    return torch.rand(N) * 5.0  # values in [0, 5)


# ---------------------------------------------------------------------------
# 1. MiningConfig defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = MiningConfig()
    assert cfg.top_k_ratio == 0.5
    assert cfg.min_loss_threshold == 0.0
    assert cfg.use_focal is False
    assert cfg.focal_gamma == 2.0
    assert cfg.ema_decay == 0.99
    assert cfg.update_freq == 10


# ---------------------------------------------------------------------------
# 2. ohem_loss — returns a scalar
# ---------------------------------------------------------------------------

def test_ohem_loss_is_scalar():
    losses = _losses()
    result = ohem_loss(losses, ratio=K_RATIO)
    assert result.shape == torch.Size([]), "ohem_loss must return a scalar"


# ---------------------------------------------------------------------------
# 3. ohem_loss — result >= global minimum (we only keep high-loss samples)
# ---------------------------------------------------------------------------

def test_ohem_loss_ge_min_of_all():
    losses = _losses()
    result = ohem_loss(losses, ratio=K_RATIO)
    assert result.item() >= losses.min().item() - 1e-6


# ---------------------------------------------------------------------------
# 4. ohem_loss — uses only top-k (result >= global mean)
# ---------------------------------------------------------------------------

def test_ohem_loss_ge_mean_of_all():
    """Mean over top-k should be >= mean over all samples."""
    losses = _losses()
    result = ohem_loss(losses, ratio=K_RATIO)
    assert result.item() >= losses.mean().item() - 1e-6


# ---------------------------------------------------------------------------
# 5. focal_weight — returns (N,) shape
# ---------------------------------------------------------------------------

def test_focal_weight_shape():
    losses = _losses()
    w = focal_weight(losses, gamma=2.0)
    assert w.shape == (N,), f"Expected shape ({N},), got {w.shape}"


# ---------------------------------------------------------------------------
# 6. focal_weight — sums to 1
# ---------------------------------------------------------------------------

def test_focal_weight_sums_to_one():
    losses = _losses()
    w = focal_weight(losses, gamma=2.0)
    assert abs(w.sum().item() - 1.0) < 1e-5, f"focal_weight sums to {w.sum().item()}, not 1"


# ---------------------------------------------------------------------------
# 7. focal_weight — high-loss samples get higher weight
# ---------------------------------------------------------------------------

def test_focal_weight_high_loss_gets_higher_weight():
    """Create a tensor where the last element has much higher loss."""
    losses = torch.ones(N) * 0.1
    losses[-1] = 10.0  # clear outlier
    w = focal_weight(losses, gamma=2.0)
    # The high-loss sample should have the largest weight
    assert w[-1].item() == pytest.approx(w.max().item(), abs=1e-6)


# ---------------------------------------------------------------------------
# 8. compute_difficulty_scores — returns tuple of two (N,) tensors
# ---------------------------------------------------------------------------

def test_compute_difficulty_scores_returns_two_n_tensors():
    losses = _losses()
    current, ema = compute_difficulty_scores(losses)
    assert current.shape == (N,)
    assert ema.shape == (N,)


# ---------------------------------------------------------------------------
# 9. EMA initialized from current when no prior ema provided
# ---------------------------------------------------------------------------

def test_ema_initialized_from_current_when_no_prior():
    losses = _losses()
    current, ema = compute_difficulty_scores(losses, ema_losses=None)
    assert torch.allclose(current, ema, atol=1e-6), \
        "When no prior EMA is supplied, EMA should equal current losses"


# ---------------------------------------------------------------------------
# 10. rank_by_difficulty — has all 3 keys
# ---------------------------------------------------------------------------

def test_rank_by_difficulty_has_all_three_keys():
    losses = _losses()
    result = rank_by_difficulty(losses, percentile=0.8)
    assert "hard" in result
    assert "medium" in result
    assert "easy" in result


# ---------------------------------------------------------------------------
# 11. hard + medium + easy partition all indices
# ---------------------------------------------------------------------------

def test_rank_by_difficulty_partitions_all_indices():
    losses = _losses()
    result = rank_by_difficulty(losses, percentile=0.8)
    all_idx = torch.cat([result["hard"], result["medium"], result["easy"]])
    assert all_idx.shape[0] == N, f"Expected {N} indices, got {all_idx.shape[0]}"
    # Check no duplicates and full coverage
    unique_idx = torch.unique(all_idx)
    assert unique_idx.shape[0] == N, "Indices must form a partition (no duplicates)"


# ---------------------------------------------------------------------------
# 12. HardExampleSampler.update changes ema
# ---------------------------------------------------------------------------

def test_hard_example_sampler_update_changes_ema():
    cfg = MiningConfig(ema_decay=0.9)
    sampler = HardExampleSampler(cfg, dataset_size=N)
    initial_ema = sampler.ema_losses.clone()

    indices = torch.arange(5)
    losses = torch.ones(5) * 3.0
    sampler.update(indices, losses)

    # EMA for updated indices should have changed
    assert not torch.allclose(sampler.ema_losses[:5], initial_ema[:5]), \
        "EMA values should change after update"
    # EMA for untouched indices should remain zero
    assert torch.allclose(sampler.ema_losses[5:], initial_ema[5:]), \
        "Untouched EMA values should remain unchanged"


# ---------------------------------------------------------------------------
# 13. sample_hard_indices returns n indices
# ---------------------------------------------------------------------------

def test_sample_hard_indices_returns_n_indices():
    cfg = MiningConfig()
    sampler = HardExampleSampler(cfg, dataset_size=N)
    # Give varied losses so topk is meaningful
    sampler.ema_losses = _losses()
    n_sample = 8
    indices = sampler.sample_hard_indices(n_sample)
    assert indices.shape == (n_sample,), f"Expected ({n_sample},), got {indices.shape}"


# ---------------------------------------------------------------------------
# 14. sample_curriculum early epoch returns low-loss indices
# ---------------------------------------------------------------------------

def test_sample_curriculum_early_epoch_returns_low_loss_indices():
    cfg = MiningConfig()
    sampler = HardExampleSampler(cfg, dataset_size=N)
    # Assign losses: last half has low losses, first half has high losses
    losses = torch.zeros(N)
    losses[:N // 2] = 5.0   # hard
    losses[N // 2:] = 0.1   # easy
    sampler.ema_losses = losses

    # Early epoch (epoch=0, total=10) should prefer easy (low-loss) samples
    indices = sampler.sample_curriculum(n=5, epoch=0, total_epochs=10)
    assert indices.shape[0] == 5

    # All selected indices should come from the easy pool (indices N//2 onward)
    selected = set(indices.tolist())
    easy_pool = set(range(N // 2, N))
    assert selected.issubset(easy_pool), \
        f"Early-epoch curriculum should pick easy examples; got {selected}"


# ---------------------------------------------------------------------------
# 15. loss_with_mining OHEM mode is scalar
# ---------------------------------------------------------------------------

def test_loss_with_mining_ohem_is_scalar():
    cfg = MiningConfig(use_focal=False, top_k_ratio=0.5)
    losses = _losses()
    result = loss_with_mining(losses, cfg)
    assert result.shape == torch.Size([]), "loss_with_mining (OHEM) must return a scalar"


# ---------------------------------------------------------------------------
# 16. loss_with_mining focal mode is scalar
# ---------------------------------------------------------------------------

def test_loss_with_mining_focal_is_scalar():
    cfg = MiningConfig(use_focal=True, focal_gamma=2.0)
    losses = _losses()
    result = loss_with_mining(losses, cfg)
    assert result.shape == torch.Size([]), "loss_with_mining (focal) must return a scalar"
