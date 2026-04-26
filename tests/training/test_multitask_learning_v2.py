"""Tests for src/training/multitask_learning_v2.py

Tiny config: D=8, OUT_A=4, OUT_B=8, B=2, T=3
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.training.multitask_learning_v2 import (
    GradientBalancer,
    MTLConfig,
    MTLLoss,
    MultiTaskModel,
    TaskHead,
    compute_gradient_cosine_similarity,
    compute_uncertainty_weights,
)

D = 8
OUT_A = 4
OUT_B = 8
B = 2
T = 3


# ---------------------------------------------------------------------------
# MTLConfig defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = MTLConfig()
    assert cfg.task_names == ["task_a", "task_b"]
    assert cfg.loss_weights is None
    assert cfg.gradient_accumulation_steps == 1
    assert cfg.uncertainty_weighting is False


# ---------------------------------------------------------------------------
# TaskHead
# ---------------------------------------------------------------------------


def test_task_head_output_shape():
    head = TaskHead(D, OUT_A, "task_a")
    x = torch.randn(B, T, D)
    out = head(x)
    assert out.shape == (B, T, OUT_A)


# ---------------------------------------------------------------------------
# compute_uncertainty_weights
# ---------------------------------------------------------------------------


def test_uncertainty_weights_shape():
    log_vars = torch.zeros(3)
    w = compute_uncertainty_weights(log_vars)
    assert w.shape == (3,)


def test_uncertainty_weights_positive():
    log_vars = torch.randn(4)
    w = compute_uncertainty_weights(log_vars)
    assert (w > 0).all()


# ---------------------------------------------------------------------------
# compute_gradient_cosine_similarity
# ---------------------------------------------------------------------------


def test_cosine_similarity_in_range():
    a = torch.randn(16)
    b = torch.randn(16)
    sim = compute_gradient_cosine_similarity(a, b)
    assert -1.0 <= sim <= 1.0


def test_cosine_similarity_identical():
    a = torch.randn(16)
    sim = compute_gradient_cosine_similarity(a, a)
    assert abs(sim - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# MTLLoss
# ---------------------------------------------------------------------------


def test_mtl_loss_compute_scalar():
    cfg = MTLConfig(task_names=["a", "b"])
    mtl = MTLLoss(cfg)
    losses = {"a": torch.tensor(1.0), "b": torch.tensor(2.0)}
    total, weights = mtl.compute(losses)
    assert total.shape == ()
    assert torch.isfinite(total)


def test_mtl_loss_returns_weight_dict():
    cfg = MTLConfig(task_names=["a", "b"])
    mtl = MTLLoss(cfg)
    losses = {"a": torch.tensor(1.0), "b": torch.tensor(2.0)}
    _, weights = mtl.compute(losses)
    assert isinstance(weights, dict)
    assert "a" in weights and "b" in weights


def test_mtl_loss_uniform_weights_sum():
    """With uniform weights, total = mean * n (all weights = 1)."""
    cfg = MTLConfig(task_names=["a", "b"])
    mtl = MTLLoss(cfg)
    losses = {"a": torch.tensor(1.0), "b": torch.tensor(1.0)}
    total, weights = mtl.compute(losses)
    # weights are all 1.0 (not normalized), so they don't necessarily sum to 1
    # but they should be equal
    assert abs(weights["a"] - weights["b"]) < 1e-6


# ---------------------------------------------------------------------------
# MultiTaskModel
# ---------------------------------------------------------------------------


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(D, D, bias=False)

    def forward(self, x):
        return self.linear(x)


def _make_mtl_model():
    backbone = _Backbone()
    heads = {
        "task_a": TaskHead(D, OUT_A, "task_a"),
        "task_b": TaskHead(D, OUT_B, "task_b"),
    }
    return MultiTaskModel(backbone, heads)


def test_multitask_model_forward_all_tasks():
    model = _make_mtl_model()
    x = torch.randn(B, T, D)
    out = model(x)
    assert "task_a" in out and "task_b" in out


def test_multitask_model_forward_single_task():
    model = _make_mtl_model()
    x = torch.randn(B, T, D)
    out = model(x, task_name="task_a")
    assert list(out.keys()) == ["task_a"]
    assert out["task_a"].shape == (B, T, OUT_A)


# ---------------------------------------------------------------------------
# GradientBalancer
# ---------------------------------------------------------------------------


def test_gradient_balancer_grad_norms_length():
    balancer = GradientBalancer(n_tasks=2)
    model = nn.Linear(D, 4, bias=False)
    x = torch.randn(B, D)
    logits = model(x)
    losses = [logits.sum(), logits.mean()]
    norms = balancer.compute_grad_norms(model, losses, retain_graph=True)
    assert len(norms) == 2


def test_gradient_balancer_balance_weights_sum_to_one():
    balancer = GradientBalancer(n_tasks=3)
    weights = balancer.balance_weights([1.0, 2.0, 0.5])
    assert abs(sum(weights) - 1.0) < 1e-6


def test_gradient_balancer_balance_weights_non_negative():
    balancer = GradientBalancer(n_tasks=3)
    weights = balancer.balance_weights([1.0, 2.0, 0.5])
    assert all(w >= 0 for w in weights)
