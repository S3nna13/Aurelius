"""Unit and integration tests for src/training/offline_reward_modeling.py.

Coverage:
    1.  test_config_defaults               -- RewardModelConfig default values
    2.  test_reward_head_output_shape      -- forward returns shape [B]
    3.  test_reward_head_last_token        -- mask selects the correct token position
    4.  test_reward_head_zero_mask         -- all-zero mask falls back to last position
    5.  test_bradley_terry_loss_scalar     -- loss tensor is 0-dimensional
    6.  test_bradley_terry_loss_positive   -- loss > 0 for random (unseparated) inputs
    7.  test_bradley_terry_loss_preferred  -- high r_w - r_l yields lower loss
    8.  test_margin_effect                 -- margin > 0 yields higher loss for small gaps
    9.  test_center_rewards                -- centering does not change the loss value
    10. test_accuracy_range                -- accuracy in [0.0, 1.0]
    11. test_accuracy_perfect              -- r_w >> r_l -> accuracy = 1.0
    12. test_total_loss_keys               -- total_loss returns required dict keys
    13. test_total_loss_scalar             -- all dict values are scalar tensors
    14. test_gradient_flows                -- backward propagates gradients through head
    Integration:
        test_integration_forward_backward  -- B=4, T=16, d_model=64 full pass + backward
"""

from __future__ import annotations

import math

import pytest
import torch

from src.training import TRAINING_REGISTRY
from src.training.offline_reward_modeling import (
    RewardBatch,
    RewardHead,
    RewardModelConfig,
    RewardModelTrainer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(
    B: int = 2,
    T_w: int = 8,
    T_l: int = 6,
    d_model: int = 16,
    *,
    full_mask: bool = True,
) -> RewardBatch:
    """Build a synthetic RewardBatch with random hidden states."""
    torch.manual_seed(0)
    chosen_hidden = torch.randn(B, T_w, d_model)
    rejected_hidden = torch.randn(B, T_l, d_model)
    if full_mask:
        chosen_mask = torch.ones(B, T_w)
        rejected_mask = torch.ones(B, T_l)
    else:
        # Last two positions are padding.
        chosen_mask = torch.ones(B, T_w)
        chosen_mask[:, -2:] = 0.0
        rejected_mask = torch.ones(B, T_l)
        rejected_mask[:, -2:] = 0.0
    return RewardBatch(
        chosen_hidden=chosen_hidden,
        rejected_hidden=rejected_hidden,
        chosen_mask=chosen_mask,
        rejected_mask=rejected_mask,
    )


def _make_head(d_model: int = 16, dropout: float = 0.0) -> RewardHead:
    cfg = RewardModelConfig(d_model=d_model, dropout=dropout)
    head = RewardHead(cfg)
    head.eval()
    return head


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = RewardModelConfig()
    assert cfg.d_model == 2048
    assert cfg.dropout == pytest.approx(0.0)
    assert cfg.center_rewards is True
    assert cfg.margin == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. test_reward_head_output_shape
# ---------------------------------------------------------------------------


def test_reward_head_output_shape():
    B, T, d = 4, 10, 32
    head = _make_head(d_model=d)
    hidden = torch.randn(B, T, d)
    mask = torch.ones(B, T)
    out = head(hidden, mask)
    assert out.shape == (B,), f"expected ({B},), got {out.shape}"


# ---------------------------------------------------------------------------
# 3. test_reward_head_last_token
# ---------------------------------------------------------------------------


def test_reward_head_last_token():
    """Reward head must extract the last *valid* token, not the last position."""
    B, T, d = 2, 8, 16
    head = _make_head(d_model=d)

    # All hidden states are zero except at position 5, which holds a large value.
    hidden = torch.zeros(B, T, d)
    signal_pos = 5
    hidden[:, signal_pos, :] = 1.0

    # Mask: valid tokens are 0..5 (inclusive); positions 6 and 7 are padding.
    mask = torch.zeros(B, T)
    mask[:, : signal_pos + 1] = 1.0  # last valid = 5

    # Set head weights so that output = sum of d_model dims, making position detectable.
    with torch.no_grad():
        head.linear.weight.fill_(1.0)
        head.linear.bias.fill_(0.0)

    reward_with_signal = head(hidden, mask).clone()

    # Shift valid window to positions 0..3 (no signal there).
    mask2 = torch.zeros(B, T)
    mask2[:, :4] = 1.0  # last valid = 3

    reward_without_signal = head(hidden, mask2)

    # With signal at pos 5: reward = d = 16.0. Without (pos 3, zeros): reward = 0.0.
    assert (reward_with_signal > reward_without_signal).all(), (
        f"head did not select the correct last-token position; "
        f"signal={reward_with_signal}, no_signal={reward_without_signal}"
    )


# ---------------------------------------------------------------------------
# 4. test_reward_head_zero_mask
# ---------------------------------------------------------------------------


def test_reward_head_zero_mask():
    """All-zero mask should not crash; fall back to the last position (T-1)."""
    B, T, d = 3, 6, 16
    head = _make_head(d_model=d)
    hidden = torch.randn(B, T, d)
    mask = torch.zeros(B, T)  # all padding

    out = head(hidden, mask)
    assert out.shape == (B,), f"expected ({B},), got {out.shape}"
    assert torch.isfinite(out).all(), f"non-finite output with all-zero mask: {out}"


# ---------------------------------------------------------------------------
# 5. test_bradley_terry_loss_scalar
# ---------------------------------------------------------------------------


def test_bradley_terry_loss_scalar():
    trainer = RewardModelTrainer()
    r_w = torch.randn(4)
    r_l = torch.randn(4)
    loss = trainer.bradley_terry_loss(r_w, r_l)
    assert loss.dim() == 0, f"loss should be scalar (0-dim), got shape {loss.shape}"


# ---------------------------------------------------------------------------
# 6. test_bradley_terry_loss_positive
# ---------------------------------------------------------------------------


def test_bradley_terry_loss_positive():
    """For random rewards the BT loss should be positive (log-sigmoid < 0)."""
    torch.manual_seed(7)
    trainer = RewardModelTrainer(RewardModelConfig(center_rewards=False, margin=0.0))
    r_w = torch.randn(8)
    r_l = torch.randn(8)
    loss = trainer.bradley_terry_loss(r_w, r_l)
    assert loss.item() > 0.0, f"expected positive loss, got {loss.item()}"


# ---------------------------------------------------------------------------
# 7. test_bradley_terry_loss_preferred
# ---------------------------------------------------------------------------


def test_bradley_terry_loss_preferred():
    """Large r_w - r_l yields loss near 0; small gap yields higher loss."""
    trainer = RewardModelTrainer(RewardModelConfig(center_rewards=False, margin=0.0))
    B = 4
    r_w_high = torch.full((B,), 10.0)
    r_l_high = torch.full((B,), -10.0)
    loss_high = trainer.bradley_terry_loss(r_w_high, r_l_high).item()

    r_w_low = torch.full((B,), 0.1)
    r_l_low = torch.full((B,), -0.1)
    loss_low = trainer.bradley_terry_loss(r_w_low, r_l_low).item()

    assert loss_high < loss_low, (
        f"expected high-gap loss ({loss_high:.6f}) < low-gap loss ({loss_low:.6f})"
    )


# ---------------------------------------------------------------------------
# 8. test_margin_effect
# ---------------------------------------------------------------------------


def test_margin_effect():
    """A positive margin should produce higher loss than zero margin for the same gap."""
    B = 4
    r_w = torch.full((B,), 1.0)
    r_l = torch.full((B,), 0.0)

    trainer_no_margin = RewardModelTrainer(RewardModelConfig(center_rewards=False, margin=0.0))
    trainer_margin = RewardModelTrainer(RewardModelConfig(center_rewards=False, margin=1.5))

    loss_no_margin = trainer_no_margin.bradley_terry_loss(r_w, r_l).item()
    loss_margin = trainer_margin.bradley_terry_loss(r_w, r_l).item()

    assert loss_margin > loss_no_margin, (
        f"expected margin loss ({loss_margin:.6f}) > no-margin loss ({loss_no_margin:.6f})"
    )


# ---------------------------------------------------------------------------
# 9. test_center_rewards
# ---------------------------------------------------------------------------


def test_center_rewards():
    """Centering rewards should not change the relative gap (r_w - r_l)."""
    B = 6
    torch.manual_seed(42)
    r_w = torch.randn(B) + 2.0
    r_l = torch.randn(B) - 2.0

    cfg_center = RewardModelConfig(center_rewards=True, margin=0.0)
    cfg_raw = RewardModelConfig(center_rewards=False, margin=0.0)

    loss_center = RewardModelTrainer(cfg_center).bradley_terry_loss(r_w, r_l).item()
    loss_raw = RewardModelTrainer(cfg_raw).bradley_terry_loss(r_w, r_l).item()

    # After centering: (r_w - mean) - (r_l - mean) = r_w - r_l -> same gap.
    assert loss_center == pytest.approx(loss_raw, rel=1e-5), (
        f"centering changed the loss: centered={loss_center}, raw={loss_raw}"
    )


# ---------------------------------------------------------------------------
# 10. test_accuracy_range
# ---------------------------------------------------------------------------


def test_accuracy_range():
    torch.manual_seed(1)
    trainer = RewardModelTrainer()
    r_w = torch.randn(16)
    r_l = torch.randn(16)
    acc = trainer.accuracy(r_w, r_l)
    assert 0.0 <= acc <= 1.0, f"accuracy {acc} out of [0, 1]"


# ---------------------------------------------------------------------------
# 11. test_accuracy_perfect
# ---------------------------------------------------------------------------


def test_accuracy_perfect():
    """r_w >> r_l for every pair -> accuracy = 1.0."""
    trainer = RewardModelTrainer()
    r_w = torch.full((8,), 100.0)
    r_l = torch.full((8,), -100.0)
    acc = trainer.accuracy(r_w, r_l)
    assert acc == pytest.approx(1.0), f"expected accuracy 1.0, got {acc}"


# ---------------------------------------------------------------------------
# 12. test_total_loss_keys
# ---------------------------------------------------------------------------


def test_total_loss_keys():
    batch = _make_batch()
    head = _make_head()
    trainer = RewardModelTrainer()
    out = trainer.total_loss(batch, head)
    required = {"loss", "reward_chosen_mean", "reward_rejected_mean", "reward_gap"}
    assert required <= out.keys(), f"missing keys: {required - out.keys()}"


# ---------------------------------------------------------------------------
# 13. test_total_loss_scalar
# ---------------------------------------------------------------------------


def test_total_loss_scalar():
    batch = _make_batch()
    head = _make_head()
    trainer = RewardModelTrainer()
    out = trainer.total_loss(batch, head)
    for key, val in out.items():
        assert val.dim() == 0, f"'{key}' should be scalar (0-dim), got shape {val.shape}"


# ---------------------------------------------------------------------------
# 14. test_gradient_flows
# ---------------------------------------------------------------------------


def test_gradient_flows():
    """Loss backward should propagate non-None, finite gradients through the head."""
    B, T, d = 3, 8, 16
    cfg = RewardModelConfig(d_model=d)
    head = RewardHead(cfg)
    trainer = RewardModelTrainer(cfg)

    torch.manual_seed(99)
    batch = RewardBatch(
        chosen_hidden=torch.randn(B, T, d),
        rejected_hidden=torch.randn(B, T, d),
        chosen_mask=torch.ones(B, T),
        rejected_mask=torch.ones(B, T),
    )

    out = trainer.total_loss(batch, head)
    out["loss"].backward()

    assert head.linear.weight.grad is not None, "grad missing for linear.weight"
    assert head.linear.bias.grad is not None, "grad missing for linear.bias"
    assert torch.isfinite(head.linear.weight.grad).all(), "NaN/Inf in weight grad"
    assert torch.isfinite(head.linear.bias.grad).all(), "NaN/Inf in bias grad"


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


def test_integration_forward_backward():
    """Full forward + backward with B=4, T=16, d_model=64."""
    B, T, d = 4, 16, 64

    cfg = RewardModelConfig(
        d_model=d,
        dropout=0.0,
        center_rewards=True,
        margin=0.1,
    )
    head = RewardHead(cfg)
    trainer = RewardModelTrainer(cfg)

    torch.manual_seed(12345)
    chosen_hidden = torch.randn(B, T, d)
    rejected_hidden = torch.randn(B, T, d)

    # Realistic masks: last 3 tokens are padding.
    chosen_mask = torch.ones(B, T)
    chosen_mask[:, -3:] = 0.0
    rejected_mask = torch.ones(B, T)
    rejected_mask[:, -3:] = 0.0

    batch = RewardBatch(
        chosen_hidden=chosen_hidden,
        rejected_hidden=rejected_hidden,
        chosen_mask=chosen_mask,
        rejected_mask=rejected_mask,
    )

    out = trainer.total_loss(batch, head)

    # Loss must be a finite scalar.
    loss_val = out["loss"].item()
    assert math.isfinite(loss_val), f"loss is not finite: {loss_val}"
    assert out["loss"].dim() == 0

    # reward_gap must be finite.
    assert math.isfinite(out["reward_gap"].item()), "reward_gap is not finite"

    # Backward pass must complete without error.
    out["loss"].backward()

    # Head parameters must have finite gradients.
    assert head.linear.weight.grad is not None, "grad missing after backward"
    assert torch.isfinite(head.linear.weight.grad).all(), "non-finite weight grad"
    assert torch.isfinite(head.linear.bias.grad).all(), "non-finite bias grad"

    # Statistics method must return sensible values.
    stats = trainer.statistics(batch, head)
    assert math.isfinite(stats["accuracy"])
    assert 0.0 <= stats["accuracy"] <= 1.0
    assert math.isfinite(stats["reward_chosen_mean"])
    assert math.isfinite(stats["reward_rejected_mean"])
    assert math.isfinite(stats["reward_gap_mean"])

    # Registry sanity check.
    assert "reward_modeling" in TRAINING_REGISTRY
    assert TRAINING_REGISTRY["reward_modeling"] is RewardModelTrainer
