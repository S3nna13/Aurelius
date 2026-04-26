"""Tests for reward_model_v2.py — RewardConfig-based Bradley-Terry preference learning."""

from __future__ import annotations

import math

import torch

from src.alignment.reward_model_v2 import (
    RewardConfig,
    RewardHead,
    RewardModel,
    clip_rewards,
    compute_preference_loss,
    compute_reward_accuracy,
    compute_reward_stats,
    normalize_rewards,
)

# ---------------------------------------------------------------------------
# Shared tiny constants and helpers
# ---------------------------------------------------------------------------

D = 8  # small d_model for fast tests
B = 4  # batch size
T = 6  # sequence length


def _make_backbone(d_model: int = D):
    """Returns a deterministic backbone: input_ids (B, T) -> hidden (B, T, d_model)."""

    def backbone_fn(input_ids: torch.Tensor) -> torch.Tensor:
        torch.manual_seed(7)
        return torch.randn(input_ids.shape[0], input_ids.shape[1], d_model)

    return backbone_fn


def _make_model(d_model: int = D, dropout: float = 0.0) -> RewardModel:
    cfg = RewardConfig(d_model=d_model, dropout=dropout)
    return RewardModel(_make_backbone(d_model), cfg)


# ---------------------------------------------------------------------------
# 1. RewardConfig defaults
# ---------------------------------------------------------------------------


def test_reward_config_defaults():
    cfg = RewardConfig()
    assert cfg.d_model == 512
    assert cfg.dropout == 0.0
    assert cfg.normalize_rewards is True
    assert cfg.reward_clip == 5.0


# ---------------------------------------------------------------------------
# 2. RewardHead — 3D input (B, T, D) produces (B,)
# ---------------------------------------------------------------------------


def test_reward_head_3d_shape():
    head = RewardHead(d_model=D, dropout=0.0)
    hidden = torch.randn(B, T, D)
    out = head(hidden)
    assert out.shape == (B,), f"Expected ({B},), got {out.shape}"


# ---------------------------------------------------------------------------
# 3. RewardHead — 2D input (B, D) produces (B,)
# ---------------------------------------------------------------------------


def test_reward_head_2d_shape():
    head = RewardHead(d_model=D, dropout=0.0)
    hidden = torch.randn(B, D)
    out = head(hidden)
    assert out.shape == (B,), f"Expected ({B},), got {out.shape}"


# ---------------------------------------------------------------------------
# 4. RewardHead output is finite
# ---------------------------------------------------------------------------


def test_reward_head_output_finite():
    head = RewardHead(d_model=D, dropout=0.0)
    hidden = torch.randn(B, T, D)
    out = head(hidden)
    assert torch.all(torch.isfinite(out)), "RewardHead output contains non-finite values"


# ---------------------------------------------------------------------------
# 5. RewardHead uses last token (3D path)
# ---------------------------------------------------------------------------


def test_reward_head_uses_last_token():
    head = RewardHead(d_model=D, dropout=0.0)
    # Two inputs identical except for the last token position
    h1 = torch.zeros(2, T, D)
    h1[:, -1, :] = 1.0

    h2 = torch.zeros(2, T, D)
    h2[:, 0, :] = 1.0  # different position than last

    out1 = head(h1)
    out2 = head(h2)
    assert not torch.allclose(out1, out2), "RewardHead should pool the LAST token"


# ---------------------------------------------------------------------------
# 6. compute_preference_loss — returns scalar, finite, positive
# ---------------------------------------------------------------------------


def test_preference_loss_scalar_finite_positive():
    chosen = torch.tensor([0.5, 1.0, 1.5])
    rejected = torch.tensor([0.0, 0.5, 1.0])
    loss = compute_preference_loss(chosen, rejected)
    assert isinstance(loss, torch.Tensor), "Loss must be a Tensor"
    assert loss.ndim == 0, f"Loss must be scalar, got shape {loss.shape}"
    assert torch.isfinite(loss), "Loss must be finite"
    assert loss.item() > 0.0, "Preference loss should be > 0"


# ---------------------------------------------------------------------------
# 7. compute_preference_loss — larger margin gives lower loss
# ---------------------------------------------------------------------------


def test_preference_loss_lower_with_larger_margin():
    # chosen >> rejected: loss should be small
    chosen_big = torch.tensor([10.0, 10.0, 10.0])
    rejected_big = torch.tensor([-10.0, -10.0, -10.0])

    # chosen == rejected: loss ≈ log(2)
    chosen_eq = torch.tensor([1.0, 1.0, 1.0])
    rejected_eq = torch.tensor([1.0, 1.0, 1.0])

    loss_large_margin = compute_preference_loss(chosen_big, rejected_big)
    loss_equal = compute_preference_loss(chosen_eq, rejected_eq)

    assert loss_large_margin.item() < loss_equal.item(), (
        f"Larger margin should yield lower loss: "
        f"large={loss_large_margin.item():.4f}, equal={loss_equal.item():.4f}"
    )


# ---------------------------------------------------------------------------
# 8. compute_reward_accuracy = 1.0 when chosen always > rejected
# ---------------------------------------------------------------------------


def test_reward_accuracy_all_correct():
    chosen = torch.tensor([5.0, 6.0, 7.0])
    rejected = torch.tensor([1.0, 2.0, 3.0])
    acc = compute_reward_accuracy(chosen, rejected)
    assert acc == 1.0, f"Expected 1.0, got {acc}"


# ---------------------------------------------------------------------------
# 9. compute_reward_accuracy = 0.0 when chosen always < rejected
# ---------------------------------------------------------------------------


def test_reward_accuracy_all_wrong():
    chosen = torch.tensor([0.0, 0.5, 0.1])
    rejected = torch.tensor([3.0, 4.0, 5.0])
    acc = compute_reward_accuracy(chosen, rejected)
    assert acc == 0.0, f"Expected 0.0, got {acc}"


# ---------------------------------------------------------------------------
# 10. normalize_rewards — output shape matches input
# ---------------------------------------------------------------------------


def test_normalize_rewards_shape():
    rewards = torch.randn(16)
    running_mean = torch.tensor(0.0)
    running_var = torch.tensor(1.0)
    normed, new_mean, new_var = normalize_rewards(rewards, running_mean, running_var)
    assert normed.shape == rewards.shape, (
        f"Normalized shape {normed.shape} != input shape {rewards.shape}"
    )


# ---------------------------------------------------------------------------
# 11. normalize_rewards — running stats are updated (different from init)
# ---------------------------------------------------------------------------


def test_normalize_rewards_stats_update():
    torch.manual_seed(0)
    rewards = torch.randn(32) * 3.0 + 5.0  # mean≈5, std≈3
    running_mean = torch.tensor(0.0)
    running_var = torch.tensor(1.0)
    _, new_mean, new_var = normalize_rewards(rewards, running_mean, running_var, momentum=0.1)
    # With momentum=0.1 the new stats should move toward the batch stats
    assert new_mean.item() != 0.0, "Running mean should update"
    assert new_var.item() != 1.0, "Running var should update"


# ---------------------------------------------------------------------------
# 12. clip_rewards — clamps to [-clip, clip]
# ---------------------------------------------------------------------------


def test_clip_rewards_clamps():
    rewards = torch.tensor([-10.0, -3.0, 0.0, 3.0, 10.0])
    clip_val = 5.0
    clipped = clip_rewards(rewards, clip_val)
    assert clipped.max().item() <= clip_val, "clip_rewards exceeded +clip_value"
    assert clipped.min().item() >= -clip_val, "clip_rewards went below -clip_value"
    # Values within range unchanged
    assert clipped[2].item() == 0.0


# ---------------------------------------------------------------------------
# 13. RewardModel.forward returns (B,) shape
# ---------------------------------------------------------------------------


def test_reward_model_forward_shape():
    model = _make_model()
    ids = torch.randint(0, 100, (B, T))
    out = model(ids)
    assert out.shape == (B,), f"Expected ({B},), got {out.shape}"


# ---------------------------------------------------------------------------
# 14. RewardModel.score_pair returns two (B,) tensors
# ---------------------------------------------------------------------------


def test_reward_model_score_pair_shapes():
    model = _make_model()
    chosen_ids = torch.randint(0, 100, (B, T))
    rejected_ids = torch.randint(0, 100, (B, T))
    chosen_r, rejected_r = model.score_pair(chosen_ids, rejected_ids)
    assert chosen_r.shape == (B,), f"chosen shape: {chosen_r.shape}"
    assert rejected_r.shape == (B,), f"rejected shape: {rejected_r.shape}"


# ---------------------------------------------------------------------------
# 15. compute_reward_stats — has all required keys with Python floats
# ---------------------------------------------------------------------------


def test_compute_reward_stats_keys():
    torch.manual_seed(42)
    rewards = torch.randn(20)
    stats = compute_reward_stats(rewards)
    required_keys = {"mean", "std", "min", "max"}
    assert required_keys == set(stats.keys()), f"Missing keys: {required_keys - set(stats.keys())}"
    for k, v in stats.items():
        assert isinstance(v, float), f"stats['{k}'] should be a Python float, got {type(v)}"


# ---------------------------------------------------------------------------
# 16. compute_reward_stats — values are numerically correct
# ---------------------------------------------------------------------------


def test_compute_reward_stats_values():
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = compute_reward_stats(rewards)
    assert abs(stats["mean"] - 3.0) < 1e-5, f"mean={stats['mean']}"
    assert abs(stats["min"] - 1.0) < 1e-5, f"min={stats['min']}"
    assert abs(stats["max"] - 5.0) < 1e-5, f"max={stats['max']}"


# ---------------------------------------------------------------------------
# 17. RewardHead bias initialised to zero
# ---------------------------------------------------------------------------


def test_reward_head_bias_init_zero():
    head = RewardHead(d_model=D)
    assert torch.all(head.proj.bias == 0.0), "Bias should be initialized to zero"


# ---------------------------------------------------------------------------
# 18. compute_preference_loss ≈ log(2) when chosen == rejected
# ---------------------------------------------------------------------------


def test_preference_loss_log2_when_equal():
    chosen = torch.ones(8)
    rejected = torch.ones(8)
    loss = compute_preference_loss(chosen, rejected)
    assert abs(loss.item() - math.log(2)) < 1e-5, (
        f"Expected log(2)≈{math.log(2):.6f}, got {loss.item():.6f}"
    )
