"""Tests for reward_model.py — Bradley-Terry preference learning."""
from __future__ import annotations

import math

import pytest
import torch

from src.alignment.reward_model import (
    RewardModelConfig,
    RewardHead,
    RewardModel,
    RewardTrainer,
    bradley_terry_loss,
    compute_reward_accuracy,
    normalize_rewards,
)

# ---------------------------------------------------------------------------
# Shared tiny backbone
# ---------------------------------------------------------------------------

D_MODEL = 16


def make_backbone():
    """Tiny backbone: randn hidden states (B, T, D_MODEL)."""
    def backbone_fn(ids: torch.Tensor) -> torch.Tensor:
        torch.manual_seed(0)
        return torch.randn(ids.shape[0], ids.shape[1], D_MODEL)
    return backbone_fn


def make_model(dropout: float = 0.0) -> RewardModel:
    cfg = RewardModelConfig(d_model=D_MODEL, dropout=dropout, margin=0.0, label_smoothing=0.0)
    return RewardModel(make_backbone(), cfg)


# ---------------------------------------------------------------------------
# 1. RewardModelConfig defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = RewardModelConfig()
    assert cfg.d_model == 64
    assert cfg.dropout == 0.1
    assert cfg.label_smoothing == 0.0
    assert cfg.margin == 0.0


# ---------------------------------------------------------------------------
# 2. RewardHead output shape is (B,)
# ---------------------------------------------------------------------------

def test_reward_head_output_shape():
    head = RewardHead(d_model=D_MODEL, dropout=0.0)
    hidden = torch.randn(4, 8, D_MODEL)
    out = head(hidden)
    assert out.shape == (4,), f"Expected (4,), got {out.shape}"


# ---------------------------------------------------------------------------
# 3. RewardHead takes last token (not first or mean)
# ---------------------------------------------------------------------------

def test_reward_head_uses_last_token():
    head = RewardHead(d_model=D_MODEL, dropout=0.0)
    # Make last token distinctive
    hidden = torch.zeros(2, 5, D_MODEL)
    hidden[:, -1, :] = 1.0
    out1 = head(hidden)

    hidden2 = torch.zeros(2, 5, D_MODEL)
    hidden2[:, 0, :] = 1.0
    out2 = head(hidden2)

    # Rewards should differ since last token differs
    assert not torch.allclose(out1, out2), "RewardHead should use last token"


# ---------------------------------------------------------------------------
# 4. RewardModel output shape is (B,)
# ---------------------------------------------------------------------------

def test_reward_model_output_shape():
    model = make_model()
    ids = torch.randint(0, 100, (3, 10))
    out = model(ids)
    assert out.shape == (3,), f"Expected (3,), got {out.shape}"


# ---------------------------------------------------------------------------
# 5. RewardModel.get_reward is an alias for forward (no grad)
# ---------------------------------------------------------------------------

def test_get_reward_matches_forward():
    model = make_model()
    ids = torch.randint(0, 100, (2, 6))
    with torch.no_grad():
        fwd = model(ids)
    rwd = model.get_reward(ids)
    assert torch.allclose(fwd, rwd), "get_reward must match forward"
    assert rwd.shape == (2,)


# ---------------------------------------------------------------------------
# 6. bradley_terry_loss is a scalar tensor
# ---------------------------------------------------------------------------

def test_bt_loss_is_scalar():
    chosen = torch.tensor([1.0, 2.0, 3.0])
    rejected = torch.tensor([0.0, 1.0, 2.0])
    loss = bradley_terry_loss(chosen, rejected)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0, f"Expected scalar, got shape {loss.shape}"
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# 7. BT loss > 0 when rejected > chosen (reversed pair)
# ---------------------------------------------------------------------------

def test_bt_loss_positive_when_reversed():
    chosen = torch.tensor([0.0, 0.5])
    rejected = torch.tensor([2.0, 3.0])
    loss = bradley_terry_loss(chosen, rejected)
    assert loss.item() > 0.0, f"Loss should be > 0 when rejected > chosen, got {loss.item()}"


# ---------------------------------------------------------------------------
# 8. BT loss ≈ log(2) when rewards are equal
# ---------------------------------------------------------------------------

def test_bt_loss_approx_log2_when_equal():
    chosen = torch.tensor([1.0, 1.0, 1.0, 1.0])
    rejected = torch.tensor([1.0, 1.0, 1.0, 1.0])
    loss = bradley_terry_loss(chosen, rejected)
    expected = math.log(2)
    assert abs(loss.item() - expected) < 1e-5, (
        f"Expected ~log(2)={expected:.6f}, got {loss.item():.6f}"
    )


# ---------------------------------------------------------------------------
# 9. compute_reward_accuracy = 1.0 when chosen >> rejected
# ---------------------------------------------------------------------------

def test_accuracy_one_when_chosen_wins():
    chosen = torch.tensor([5.0, 6.0, 7.0])
    rejected = torch.tensor([1.0, 2.0, 3.0])
    acc = compute_reward_accuracy(chosen, rejected)
    assert acc == 1.0, f"Expected 1.0, got {acc}"


# ---------------------------------------------------------------------------
# 10. compute_reward_accuracy = 0.0 when chosen << rejected
# ---------------------------------------------------------------------------

def test_accuracy_zero_when_rejected_wins():
    chosen = torch.tensor([0.0, 0.5, 0.1])
    rejected = torch.tensor([3.0, 4.0, 5.0])
    acc = compute_reward_accuracy(chosen, rejected)
    assert acc == 0.0, f"Expected 0.0, got {acc}"


# ---------------------------------------------------------------------------
# 11. train_step returns correct keys
# ---------------------------------------------------------------------------

def test_train_step_returns_correct_keys():
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    cfg = RewardModelConfig(d_model=D_MODEL, dropout=0.0)
    trainer = RewardTrainer(model, optimizer, cfg)

    chosen = torch.randint(0, 50, (2, 6))
    rejected = torch.randint(0, 50, (2, 6))
    metrics = trainer.train_step(chosen, rejected)

    required = {"loss", "accuracy", "mean_chosen_reward", "mean_rejected_reward"}
    assert required.issubset(set(metrics.keys())), (
        f"Missing keys: {required - set(metrics.keys())}"
    )


# ---------------------------------------------------------------------------
# 12. train_step loss is finite
# ---------------------------------------------------------------------------

def test_train_step_loss_is_finite():
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    cfg = RewardModelConfig(d_model=D_MODEL, dropout=0.0)
    trainer = RewardTrainer(model, optimizer, cfg)

    chosen = torch.randint(0, 50, (4, 8))
    rejected = torch.randint(0, 50, (4, 8))
    metrics = trainer.train_step(chosen, rejected)

    assert math.isfinite(metrics["loss"]), f"Loss is not finite: {metrics['loss']}"


# ---------------------------------------------------------------------------
# 13. evaluate runs without error and returns correct keys (no grad)
# ---------------------------------------------------------------------------

def test_evaluate_no_grad_runs():
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    cfg = RewardModelConfig(d_model=D_MODEL, dropout=0.0)
    trainer = RewardTrainer(model, optimizer, cfg)

    chosen = torch.randint(0, 50, (2, 6))
    rejected = torch.randint(0, 50, (2, 6))
    metrics = trainer.evaluate(chosen, rejected)

    required = {"loss", "accuracy", "mean_chosen_reward", "mean_rejected_reward"}
    assert required.issubset(set(metrics.keys()))
    assert math.isfinite(metrics["loss"])


# ---------------------------------------------------------------------------
# 14. evaluate does NOT update model parameters
# ---------------------------------------------------------------------------

def test_evaluate_does_not_update_params():
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    cfg = RewardModelConfig(d_model=D_MODEL, dropout=0.0)
    trainer = RewardTrainer(model, optimizer, cfg)

    weight_before = model.reward_head.proj.weight.clone().detach()

    chosen = torch.randint(0, 50, (2, 6))
    rejected = torch.randint(0, 50, (2, 6))
    trainer.evaluate(chosen, rejected)

    weight_after = model.reward_head.proj.weight.clone().detach()
    assert torch.equal(weight_before, weight_after), "evaluate() must not update parameters"


# ---------------------------------------------------------------------------
# 15. normalize_rewards: mean ≈ 0, std ≈ 1
# ---------------------------------------------------------------------------

def test_normalize_rewards_mean_zero_std_one():
    torch.manual_seed(42)
    rewards = torch.randn(100) * 5 + 3
    normed = normalize_rewards(rewards)
    assert abs(normed.mean().item()) < 1e-5, f"mean={normed.mean().item()}"
    assert abs(normed.std().item() - 1.0) < 1e-4, f"std={normed.std().item()}"


# ---------------------------------------------------------------------------
# 16. margin shifts BT loss upward
# ---------------------------------------------------------------------------

def test_margin_shifts_loss():
    chosen = torch.tensor([1.0, 1.5, 2.0])
    rejected = torch.tensor([0.5, 0.5, 0.5])
    loss_no_margin = bradley_terry_loss(chosen, rejected, margin=0.0)
    loss_with_margin = bradley_terry_loss(chosen, rejected, margin=1.0)
    assert loss_with_margin.item() > loss_no_margin.item(), (
        f"Margin should increase loss: no_margin={loss_no_margin.item()}, "
        f"with_margin={loss_with_margin.item()}"
    )
