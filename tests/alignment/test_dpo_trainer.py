"""Tests for src/alignment/dpo_trainer.py — ~50 tests."""

import pytest
import torch
from torch import Tensor

from src.alignment import ALIGNMENT_REGISTRY
from src.alignment.dpo_trainer import (
    DPOConfig,
    DPOLoss,
    DPOTrainer,
)

B = 4  # batch size for tests


# ---------------------------------------------------------------------------
# DPOConfig defaults
# ---------------------------------------------------------------------------


def test_config_default_beta():
    cfg = DPOConfig()
    assert cfg.beta == 0.1


def test_config_default_label_smoothing():
    cfg = DPOConfig()
    assert cfg.label_smoothing == 0.0


def test_config_default_reference_free():
    cfg = DPOConfig()
    assert cfg.reference_free is False


def test_config_default_loss_type():
    cfg = DPOConfig()
    assert cfg.loss_type == "sigmoid"


def test_config_custom_beta():
    cfg = DPOConfig(beta=0.5)
    assert cfg.beta == 0.5


def test_config_custom_label_smoothing():
    cfg = DPOConfig(label_smoothing=0.1)
    assert cfg.label_smoothing == 0.1


def test_config_custom_reference_free():
    cfg = DPOConfig(reference_free=True)
    assert cfg.reference_free is True


def test_config_custom_loss_type_hinge():
    cfg = DPOConfig(loss_type="hinge")
    assert cfg.loss_type == "hinge"


def test_config_custom_loss_type_ipo():
    cfg = DPOConfig(loss_type="ipo")
    assert cfg.loss_type == "ipo"


# ---------------------------------------------------------------------------
# DPOLoss — sigmoid
# ---------------------------------------------------------------------------


def _make_logps(b: int = B) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    torch.manual_seed(0)
    pc = torch.randn(b)
    pr = torch.randn(b)
    rc = torch.randn(b)
    rr = torch.randn(b)
    return pc, pr, rc, rr


def test_dpo_loss_sigmoid_returns_tuple_of_three():
    loss_fn = DPOLoss(DPOConfig())
    pc, pr, rc, rr = _make_logps()
    result = loss_fn(pc, pr, rc, rr)
    assert len(result) == 3


def test_dpo_loss_sigmoid_loss_is_tensor():
    loss_fn = DPOLoss(DPOConfig())
    pc, pr, rc, rr = _make_logps()
    loss, _, _ = loss_fn(pc, pr, rc, rr)
    assert isinstance(loss, Tensor)


def test_dpo_loss_sigmoid_loss_is_scalar():
    loss_fn = DPOLoss(DPOConfig())
    pc, pr, rc, rr = _make_logps()
    loss, _, _ = loss_fn(pc, pr, rc, rr)
    assert loss.shape == ()


def test_dpo_loss_sigmoid_loss_is_positive():
    loss_fn = DPOLoss(DPOConfig(loss_type="sigmoid"))
    pc, pr, rc, rr = _make_logps()
    loss, _, _ = loss_fn(pc, pr, rc, rr)
    assert loss.item() > 0.0


def test_dpo_loss_sigmoid_loss_is_finite():
    loss_fn = DPOLoss(DPOConfig())
    pc, pr, rc, rr = _make_logps()
    loss, _, _ = loss_fn(pc, pr, rc, rr)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# DPOLoss — hinge
# ---------------------------------------------------------------------------


def test_dpo_loss_hinge_returns_tuple_of_three():
    loss_fn = DPOLoss(DPOConfig(loss_type="hinge"))
    pc, pr, rc, rr = _make_logps()
    result = loss_fn(pc, pr, rc, rr)
    assert len(result) == 3


def test_dpo_loss_hinge_loss_is_scalar():
    loss_fn = DPOLoss(DPOConfig(loss_type="hinge"))
    pc, pr, rc, rr = _make_logps()
    loss, _, _ = loss_fn(pc, pr, rc, rr)
    assert loss.shape == ()


def test_dpo_loss_hinge_loss_is_positive():
    loss_fn = DPOLoss(DPOConfig(loss_type="hinge"))
    pc, pr, rc, rr = _make_logps()
    loss, _, _ = loss_fn(pc, pr, rc, rr)
    assert loss.item() >= 0.0


def test_dpo_loss_hinge_loss_is_finite():
    loss_fn = DPOLoss(DPOConfig(loss_type="hinge"))
    pc, pr, rc, rr = _make_logps()
    loss, _, _ = loss_fn(pc, pr, rc, rr)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# DPOLoss — ipo
# ---------------------------------------------------------------------------


def test_dpo_loss_ipo_returns_tuple_of_three():
    loss_fn = DPOLoss(DPOConfig(loss_type="ipo"))
    pc, pr, rc, rr = _make_logps()
    result = loss_fn(pc, pr, rc, rr)
    assert len(result) == 3


def test_dpo_loss_ipo_loss_is_scalar():
    loss_fn = DPOLoss(DPOConfig(loss_type="ipo"))
    pc, pr, rc, rr = _make_logps()
    loss, _, _ = loss_fn(pc, pr, rc, rr)
    assert loss.shape == ()


def test_dpo_loss_ipo_loss_is_non_negative():
    loss_fn = DPOLoss(DPOConfig(loss_type="ipo"))
    pc, pr, rc, rr = _make_logps()
    loss, _, _ = loss_fn(pc, pr, rc, rr)
    assert loss.item() >= 0.0


def test_dpo_loss_ipo_loss_is_finite():
    loss_fn = DPOLoss(DPOConfig(loss_type="ipo"))
    pc, pr, rc, rr = _make_logps()
    loss, _, _ = loss_fn(pc, pr, rc, rr)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# reference_free=True
# ---------------------------------------------------------------------------


def test_dpo_loss_reference_free_sigmoid_runs():
    loss_fn = DPOLoss(DPOConfig(reference_free=True))
    pc, pr, _, _ = _make_logps()
    loss, chosen_r, rejected_r = loss_fn(pc, pr, None, None)
    assert torch.isfinite(loss)


def test_dpo_loss_reference_free_hinge_runs():
    loss_fn = DPOLoss(DPOConfig(loss_type="hinge", reference_free=True))
    pc, pr, _, _ = _make_logps()
    loss, _, _ = loss_fn(pc, pr, None, None)
    assert torch.isfinite(loss)


def test_dpo_loss_reference_free_ipo_runs():
    loss_fn = DPOLoss(DPOConfig(loss_type="ipo", reference_free=True))
    pc, pr, _, _ = _make_logps()
    loss, _, _ = loss_fn(pc, pr, None, None)
    assert torch.isfinite(loss)


def test_dpo_loss_reference_free_chosen_rewards_uses_policy_only():
    cfg = DPOConfig(beta=1.0, reference_free=True)
    loss_fn = DPOLoss(cfg)
    pc = torch.tensor([2.0, 3.0])
    pr = torch.tensor([1.0, 0.5])
    _, chosen_r, _ = loss_fn(pc, pr, None, None)
    expected = cfg.beta * pc
    assert torch.allclose(chosen_r, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Reward formulas
# ---------------------------------------------------------------------------


def test_chosen_rewards_equals_beta_times_chosen_log_ratio():
    cfg = DPOConfig(beta=0.2)
    loss_fn = DPOLoss(cfg)
    pc = torch.tensor([1.0, 2.0, 3.0])
    pr = torch.tensor([0.5, 1.0, 1.5])
    rc = torch.tensor([0.5, 1.0, 1.5])
    rr = torch.tensor([0.1, 0.2, 0.3])
    _, chosen_r, _ = loss_fn(pc, pr, rc, rr)
    expected = cfg.beta * (pc - rc)
    assert torch.allclose(chosen_r, expected, atol=1e-5)


def test_rejected_rewards_equals_beta_times_rejected_log_ratio():
    cfg = DPOConfig(beta=0.3)
    loss_fn = DPOLoss(cfg)
    pc = torch.tensor([1.0, 2.0])
    pr = torch.tensor([0.5, 1.0])
    rc = torch.tensor([0.5, 1.0])
    rr = torch.tensor([0.1, 0.2])
    _, _, rejected_r = loss_fn(pc, pr, rc, rr)
    expected = cfg.beta * (pr - rr)
    assert torch.allclose(rejected_r, expected, atol=1e-5)


def test_reward_margin_positive_when_chosen_higher():
    cfg = DPOConfig(beta=0.1)
    loss_fn = DPOLoss(cfg)
    # Make chosen clearly better than rejected relative to reference
    pc = torch.tensor([5.0, 5.0])
    pr = torch.tensor([-5.0, -5.0])
    rc = torch.zeros(2)
    rr = torch.zeros(2)
    _, chosen_r, rejected_r = loss_fn(pc, pr, rc, rr)
    margin = (chosen_r - rejected_r).mean().item()
    assert margin > 0.0


def test_chosen_rewards_are_detached():
    cfg = DPOConfig()
    loss_fn = DPOLoss(cfg)
    pc = torch.randn(B, requires_grad=True)
    pr = torch.randn(B, requires_grad=True)
    rc = torch.randn(B)
    rr = torch.randn(B)
    loss, chosen_r, _ = loss_fn(pc, pr, rc, rr)
    assert not chosen_r.requires_grad


def test_rejected_rewards_are_detached():
    cfg = DPOConfig()
    loss_fn = DPOLoss(cfg)
    pc = torch.randn(B, requires_grad=True)
    pr = torch.randn(B, requires_grad=True)
    rc = torch.randn(B)
    rr = torch.randn(B)
    loss, _, rejected_r = loss_fn(pc, pr, rc, rr)
    assert not rejected_r.requires_grad


# ---------------------------------------------------------------------------
# Label smoothing
# ---------------------------------------------------------------------------


def test_label_smoothing_changes_loss():
    pc, pr, rc, rr = _make_logps()
    loss_no_smooth, _, _ = DPOLoss(DPOConfig(label_smoothing=0.0))(pc, pr, rc, rr)
    loss_smooth, _, _ = DPOLoss(DPOConfig(label_smoothing=0.1))(pc, pr, rc, rr)
    assert abs(loss_no_smooth.item() - loss_smooth.item()) > 1e-6


def test_label_smoothing_zero_same_as_default():
    pc, pr, rc, rr = _make_logps()
    loss_a, _, _ = DPOLoss(DPOConfig(label_smoothing=0.0))(pc, pr, rc, rr)
    loss_b, _, _ = DPOLoss(DPOConfig())(pc, pr, rc, rr)
    assert torch.allclose(loss_a, loss_b)


# ---------------------------------------------------------------------------
# DPOTrainer.compute_loss
# ---------------------------------------------------------------------------


def test_trainer_compute_loss_returns_dict():
    trainer = DPOTrainer()
    pc, pr, rc, rr = _make_logps()
    result = trainer.compute_loss(pc, pr, rc, rr)
    assert isinstance(result, dict)


def test_trainer_compute_loss_has_loss_key():
    trainer = DPOTrainer()
    pc, pr, rc, rr = _make_logps()
    result = trainer.compute_loss(pc, pr, rc, rr)
    assert "loss" in result


def test_trainer_compute_loss_has_chosen_reward_key():
    trainer = DPOTrainer()
    pc, pr, rc, rr = _make_logps()
    result = trainer.compute_loss(pc, pr, rc, rr)
    assert "chosen_reward" in result


def test_trainer_compute_loss_has_rejected_reward_key():
    trainer = DPOTrainer()
    pc, pr, rc, rr = _make_logps()
    result = trainer.compute_loss(pc, pr, rc, rr)
    assert "rejected_reward" in result


def test_trainer_compute_loss_has_reward_margin_key():
    trainer = DPOTrainer()
    pc, pr, rc, rr = _make_logps()
    result = trainer.compute_loss(pc, pr, rc, rr)
    assert "reward_margin" in result


def test_trainer_compute_loss_loss_is_float():
    trainer = DPOTrainer()
    pc, pr, rc, rr = _make_logps()
    result = trainer.compute_loss(pc, pr, rc, rr)
    assert isinstance(result["loss"], float)


def test_trainer_compute_loss_reward_margin_is_difference():
    trainer = DPOTrainer()
    pc, pr, rc, rr = _make_logps()
    result = trainer.compute_loss(pc, pr, rc, rr)
    assert (
        abs(result["reward_margin"] - (result["chosen_reward"] - result["rejected_reward"])) < 1e-5
    )


def test_trainer_compute_loss_reference_free():
    trainer = DPOTrainer(DPOConfig(reference_free=True))
    pc, pr, _, _ = _make_logps()
    result = trainer.compute_loss(pc, pr)
    assert "loss" in result
    assert math.isfinite(result["loss"])


# ---------------------------------------------------------------------------
# DPOTrainer.reward_accuracy
# ---------------------------------------------------------------------------


def test_trainer_reward_accuracy_all_chosen_higher_returns_one():
    trainer = DPOTrainer()
    chosen = torch.tensor([2.0, 3.0, 4.0])
    rejected = torch.tensor([0.0, 1.0, 2.0])
    acc = trainer.reward_accuracy(chosen, rejected)
    assert acc == 1.0


def test_trainer_reward_accuracy_all_rejected_higher_returns_zero():
    trainer = DPOTrainer()
    chosen = torch.tensor([0.0, 0.0, 0.0])
    rejected = torch.tensor([1.0, 2.0, 3.0])
    acc = trainer.reward_accuracy(chosen, rejected)
    assert acc == 0.0


def test_trainer_reward_accuracy_half_returns_half():
    trainer = DPOTrainer()
    chosen = torch.tensor([2.0, 0.0])
    rejected = torch.tensor([0.0, 2.0])
    acc = trainer.reward_accuracy(chosen, rejected)
    assert acc == pytest.approx(0.5)


def test_trainer_reward_accuracy_returns_float():
    trainer = DPOTrainer()
    chosen = torch.tensor([1.0])
    rejected = torch.tensor([0.0])
    acc = trainer.reward_accuracy(chosen, rejected)
    assert isinstance(acc, float)


# ---------------------------------------------------------------------------
# ALIGNMENT_REGISTRY
# ---------------------------------------------------------------------------


def test_alignment_registry_has_dpo():
    assert "dpo" in ALIGNMENT_REGISTRY


def test_alignment_registry_dpo_is_dpo_trainer():
    assert isinstance(ALIGNMENT_REGISTRY["dpo"], DPOTrainer)


import math  # noqa: E402
