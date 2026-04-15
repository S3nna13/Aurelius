"""Tests for src/alignment/reward_distill.py — 16 tests."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.alignment.reward_distill import (
    DistillConfig,
    RewardDistillTrainer,
    knowledge_distillation_reward,
    rank_correlation,
)

HIDDEN_DIM = 16
B = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _linear_reward_model(hidden_dim: int, seed: int) -> nn.Module:
    """nn.Linear(hidden_dim, 1); forward(hidden) -> squeeze -> (B,)."""
    torch.manual_seed(seed)

    class _RewardModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(hidden_dim, 1)

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            return self.linear(hidden).squeeze(-1)  # (B,)

    return _RewardModel()


@pytest.fixture
def teacher():
    return _linear_reward_model(HIDDEN_DIM, seed=0)


@pytest.fixture
def student():
    return _linear_reward_model(HIDDEN_DIM, seed=42)


@pytest.fixture
def trainer(teacher, student):
    return RewardDistillTrainer(teacher, student, DistillConfig(), lr=1e-3, device="cpu")


@pytest.fixture
def hidden_chosen():
    torch.manual_seed(1)
    return torch.randn(B, HIDDEN_DIM)


@pytest.fixture
def hidden_rejected():
    torch.manual_seed(2)
    return torch.randn(B, HIDDEN_DIM)


# ---------------------------------------------------------------------------
# Test 1 — RewardDistillTrainer instantiates
# ---------------------------------------------------------------------------

def test_trainer_instantiates(teacher, student):
    trainer = RewardDistillTrainer(teacher, student)
    assert isinstance(trainer, RewardDistillTrainer)


# ---------------------------------------------------------------------------
# Test 2 — regression_loss is non-negative
# ---------------------------------------------------------------------------

def test_regression_loss_non_negative(trainer):
    s = torch.randn(B)
    t = torch.randn(B)
    loss = trainer.regression_loss(s, t)
    assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# Test 3 — regression_loss(x, x) ≈ 0
# ---------------------------------------------------------------------------

def test_regression_loss_self_zero(trainer):
    x = torch.randn(B)
    loss = trainer.regression_loss(x, x.clone())
    assert loss.item() < 1e-6


# ---------------------------------------------------------------------------
# Test 4 — ranking_loss is non-negative
# ---------------------------------------------------------------------------

def test_ranking_loss_non_negative(trainer):
    sc = torch.randn(B)
    sr = torch.randn(B)
    tc = torch.randn(B)
    tr = torch.randn(B)
    loss = trainer.ranking_loss(sc, sr, tc, tr)
    assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# Test 5 — ranking_loss penalizes wrong ordering (chosen < rejected => high loss)
# ---------------------------------------------------------------------------

def test_ranking_loss_penalizes_wrong_ordering(trainer):
    # chosen clearly worse than rejected
    sc = torch.full((B,), -5.0)
    sr = torch.full((B,), 5.0)
    # teacher clearly prefers chosen
    tc = torch.full((B,), 5.0)
    tr = torch.full((B,), -5.0)
    loss = trainer.ranking_loss(sc, sr, tc, tr)
    # hinge = max(0, margin - (-10)) = margin + 10 > 0, weighted by large teacher diff
    assert loss.item() > 0.0


# ---------------------------------------------------------------------------
# Test 6 — soft_target_loss is non-negative
# ---------------------------------------------------------------------------

def test_soft_target_loss_non_negative(trainer):
    s_logits = torch.randn(B, 2)
    t_logits = torch.randn(B, 2)
    loss = trainer.soft_target_loss(s_logits, t_logits)
    assert loss.item() >= -1e-7  # KL >= 0 (allow tiny numerical noise)


# ---------------------------------------------------------------------------
# Test 7 — soft_target_loss(x, x) ≈ 0 (self-distillation)
# ---------------------------------------------------------------------------

def test_soft_target_loss_self_zero(trainer):
    logits = torch.randn(B, 2)
    loss = trainer.soft_target_loss(logits, logits.clone())
    assert abs(loss.item()) < 1e-5


# ---------------------------------------------------------------------------
# Test 8 — compute_total_loss returns (scalar tensor, dict)
# ---------------------------------------------------------------------------

def test_compute_total_loss_returns_scalar_and_dict(trainer, hidden_chosen, hidden_rejected):
    total, metrics = trainer.compute_total_loss(hidden_chosen, hidden_rejected)
    assert isinstance(total, torch.Tensor)
    assert total.ndim == 0, "Expected scalar tensor"
    assert isinstance(metrics, dict)


# ---------------------------------------------------------------------------
# Test 9 — metrics dict has 'regression_loss' or 'ranking_loss' key
# ---------------------------------------------------------------------------

def test_compute_total_loss_metrics_has_expected_key(trainer, hidden_chosen, hidden_rejected):
    _, metrics = trainer.compute_total_loss(hidden_chosen, hidden_rejected)
    has_key = ("regression_loss" in metrics) or ("ranking_loss" in metrics)
    assert has_key, f"Expected regression_loss or ranking_loss in metrics, got: {list(metrics.keys())}"


# ---------------------------------------------------------------------------
# Test 10 — train_step returns dict with 'loss' key
# ---------------------------------------------------------------------------

def test_train_step_returns_dict_with_loss(trainer, hidden_chosen, hidden_rejected):
    result = trainer.train_step(hidden_chosen, hidden_rejected)
    assert isinstance(result, dict)
    assert "loss" in result


# ---------------------------------------------------------------------------
# Test 11 — train_step reduces loss over 10 iterations
# ---------------------------------------------------------------------------

def test_train_step_loss_decreases(teacher, student):
    torch.manual_seed(99)
    trainer = RewardDistillTrainer(teacher, student, DistillConfig(), lr=1e-2, device="cpu")
    hc = torch.randn(B, HIDDEN_DIM)
    hr = torch.randn(B, HIDDEN_DIM)
    first = trainer.train_step(hc, hr)["loss"]
    for _ in range(9):
        last = trainer.train_step(hc, hr)["loss"]
    assert last < first, f"Expected loss to decrease: {first:.4f} -> {last:.4f}"


# ---------------------------------------------------------------------------
# Test 12 — evaluate_agreement returns dict with 'sign_agreement' key
# ---------------------------------------------------------------------------

def test_evaluate_agreement_has_sign_agreement(trainer, hidden_chosen):
    teacher_rewards = torch.randn(B)
    result = trainer.evaluate_agreement(hidden_chosen, teacher_rewards)
    assert isinstance(result, dict)
    assert "sign_agreement" in result


# ---------------------------------------------------------------------------
# Test 13 — sign_agreement in [0, 1]
# ---------------------------------------------------------------------------

def test_evaluate_agreement_sign_agreement_range(trainer, hidden_chosen):
    teacher_rewards = torch.randn(B)
    result = trainer.evaluate_agreement(hidden_chosen, teacher_rewards)
    v = result["sign_agreement"]
    assert 0.0 <= v <= 1.0, f"sign_agreement {v} not in [0, 1]"


# ---------------------------------------------------------------------------
# Test 14 — rank_correlation(x, x) ≈ 1.0
# ---------------------------------------------------------------------------

def test_rank_correlation_identical():
    x = torch.tensor([3.0, 1.0, 4.0, 1.5, 2.0])
    rho = rank_correlation(x, x.clone())
    assert abs(rho - 1.0) < 1e-4, f"Expected ~1.0, got {rho}"


# ---------------------------------------------------------------------------
# Test 15 — rank_correlation(x, -x) ≈ -1.0
# ---------------------------------------------------------------------------

def test_rank_correlation_reversed():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    rho = rank_correlation(x, -x)
    assert abs(rho - (-1.0)) < 1e-4, f"Expected ~-1.0, got {rho}"


# ---------------------------------------------------------------------------
# Test 16 — knowledge_distillation_reward returns scalar tensor
# ---------------------------------------------------------------------------

def test_knowledge_distillation_reward_scalar():
    teacher_r = torch.randn(B)
    student_r = torch.randn(B)
    loss = knowledge_distillation_reward(teacher_r, student_r, temperature=2.0, alpha=0.5)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0, f"Expected scalar, got shape {loss.shape}"
    assert torch.isfinite(loss), "Loss should be finite"
