"""Tests for PCGrad: Gradient Surgery for Multi-Task Learning."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.training.pcgrad import (
    MultiTaskPCGradTrainer,
    PCGradConfig,
    PCGradOptimizer,
    pcgrad_step,
    project_gradient,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_linear_model(in_features: int = 8, out_features: int = 4) -> nn.Linear:
    torch.manual_seed(0)
    return nn.Linear(in_features, out_features)


def make_simple_model() -> nn.Sequential:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))


# ---------------------------------------------------------------------------
# Test 1: PCGradConfig defaults
# ---------------------------------------------------------------------------

def test_pcgrad_config_defaults():
    cfg = PCGradConfig()
    assert cfg.n_tasks == 2
    assert cfg.reduction == "mean"


# ---------------------------------------------------------------------------
# Test 2: project_gradient no-op when gradients agree
# ---------------------------------------------------------------------------

def test_project_gradient_no_op_when_agree():
    # Parallel gradients -> dot product > 0, no projection
    g_i = torch.tensor([1.0, 0.0, 0.0])
    g_j = torch.tensor([2.0, 0.0, 0.0])  # same direction
    result = project_gradient(g_i.clone(), g_j)
    assert torch.allclose(result, g_i), "Should be unchanged when gradients agree"


# ---------------------------------------------------------------------------
# Test 3: project_gradient projects when gradients conflict
# ---------------------------------------------------------------------------

def test_project_gradient_projects_when_conflict():
    # Opposite directions -> dot < 0, should project
    g_i = torch.tensor([1.0, 0.0])
    g_j = torch.tensor([-1.0, 0.0])
    result = project_gradient(g_i.clone(), g_j)
    # After projection, dot(result, g_j) should be >= 0
    dot_after = torch.dot(result, g_j)
    assert dot_after >= -1e-6, f"Dot product after projection should be non-negative, got {dot_after}"
    # Result should differ from original
    assert not torch.allclose(result, g_i), "Should be modified when gradients conflict"


# ---------------------------------------------------------------------------
# Test 4: project_gradient output shape matches input
# ---------------------------------------------------------------------------

def test_project_gradient_output_shape():
    g_i = torch.randn(32)
    g_j = torch.randn(32)
    result = project_gradient(g_i.clone(), g_j)
    assert result.shape == g_i.shape


# ---------------------------------------------------------------------------
# Test 5: project_gradient with zero grad_j doesn't crash
# ---------------------------------------------------------------------------

def test_project_gradient_zero_grad_j():
    g_i = torch.tensor([1.0, 2.0, 3.0])
    g_j = torch.zeros(3)
    # Should not crash; dot < 0 would trigger but norm_sq == 0 prevents divide
    result = project_gradient(g_i.clone(), g_j)
    assert result.shape == g_i.shape


# ---------------------------------------------------------------------------
# Test 6: pcgrad_step returns scalar tensor
# ---------------------------------------------------------------------------

def test_pcgrad_step_returns_scalar():
    model = make_linear_model()
    params = list(model.parameters())
    x = torch.randn(4, 8)
    out = model(x)
    loss1 = out.mean()
    loss2 = (out ** 2).mean()
    result = pcgrad_step([loss1, loss2], params)
    assert isinstance(result, torch.Tensor)
    assert result.dim() == 0, "Should return a scalar tensor"


# ---------------------------------------------------------------------------
# Test 7: pcgrad_step with 2 conflicting tasks
# ---------------------------------------------------------------------------

def test_pcgrad_step_two_conflicting_tasks():
    torch.manual_seed(42)
    model = make_linear_model()
    params = list(model.parameters())
    x = torch.randn(4, 8)
    out = model(x)
    # Construct opposing losses to force conflict
    loss1 = out.sum()
    loss2 = -out.sum()
    mean_loss = pcgrad_step([loss1, loss2], params)
    assert torch.isfinite(mean_loss)


# ---------------------------------------------------------------------------
# Test 8: pcgrad_step with 3 tasks
# ---------------------------------------------------------------------------

def test_pcgrad_step_three_tasks():
    torch.manual_seed(7)
    model = make_linear_model()
    params = list(model.parameters())
    x = torch.randn(4, 8)
    out = model(x)
    loss1 = out.mean()
    loss2 = (out - 1.0).pow(2).mean()
    loss3 = out.abs().mean()
    mean_loss = pcgrad_step([loss1, loss2, loss3], params)
    assert torch.isfinite(mean_loss)


# ---------------------------------------------------------------------------
# Test 9: pcgrad_step gradients are set on params after call
# ---------------------------------------------------------------------------

def test_pcgrad_step_sets_param_grads():
    model = make_linear_model()
    params = list(model.parameters())
    # Clear any existing grads
    for p in params:
        p.grad = None
    x = torch.randn(4, 8)
    out = model(x)
    loss1 = out.mean()
    loss2 = (out + 1).mean()
    pcgrad_step([loss1, loss2], params)
    # At least one param should have grad set
    has_grad = any(p.grad is not None for p in params)
    assert has_grad, "At least one parameter should have grad set after pcgrad_step"


# ---------------------------------------------------------------------------
# Test 10: PCGradOptimizer instantiates
# ---------------------------------------------------------------------------

def test_pcgrad_optimizer_instantiates():
    model = make_simple_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    cfg = PCGradConfig(n_tasks=2)
    pcgrad_opt = PCGradOptimizer(opt, cfg)
    assert pcgrad_opt.config is cfg
    assert pcgrad_opt.optimizer is opt


# ---------------------------------------------------------------------------
# Test 11: PCGradOptimizer.step returns dict with correct keys
# ---------------------------------------------------------------------------

def test_pcgrad_optimizer_step_returns_correct_keys():
    model = make_simple_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    cfg = PCGradConfig(n_tasks=2)
    pcgrad_opt = PCGradOptimizer(opt, cfg)

    x = torch.randn(4, 8)
    out = model(x)
    losses = [out.mean(), (out ** 2).mean()]

    result = pcgrad_opt.step(losses)
    assert "mean_loss" in result
    assert "n_conflicts" in result
    assert "task_losses" in result


# ---------------------------------------------------------------------------
# Test 12: PCGradOptimizer.step n_conflicts is non-negative int
# ---------------------------------------------------------------------------

def test_pcgrad_optimizer_step_n_conflicts_non_negative():
    model = make_simple_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    cfg = PCGradConfig(n_tasks=2)
    pcgrad_opt = PCGradOptimizer(opt, cfg)

    x = torch.randn(4, 8)
    out = model(x)
    losses = [out.mean(), (out - 2.0).pow(2).mean()]

    result = pcgrad_opt.step(losses)
    assert isinstance(result["n_conflicts"], int)
    assert result["n_conflicts"] >= 0


# ---------------------------------------------------------------------------
# Test 13: MultiTaskPCGradTrainer.train_step returns dict with correct keys
# ---------------------------------------------------------------------------

def test_multitask_trainer_train_step_keys():
    model = make_simple_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    cfg = PCGradConfig(n_tasks=2)

    x = torch.randn(4, 8)

    def task1(m: nn.Module) -> torch.Tensor:
        return m(x).mean()

    def task2(m: nn.Module) -> torch.Tensor:
        return (m(x) ** 2).mean()

    trainer = MultiTaskPCGradTrainer(model, opt, cfg, [task1, task2])
    result = trainer.train_step()

    assert "loss" in result
    assert "n_conflicts" in result
    assert "task_losses" in result


# ---------------------------------------------------------------------------
# Test 14: MultiTaskPCGradTrainer loss is finite over multiple steps
# ---------------------------------------------------------------------------

def test_multitask_trainer_loss_finite_multiple_steps():
    torch.manual_seed(123)
    model = make_simple_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    cfg = PCGradConfig(n_tasks=2)

    x = torch.randn(4, 8)
    target = torch.randn(4, 4)

    def task1(m: nn.Module) -> torch.Tensor:
        return nn.functional.mse_loss(m(x), target)

    def task2(m: nn.Module) -> torch.Tensor:
        return m(x).abs().mean()

    trainer = MultiTaskPCGradTrainer(model, opt, cfg, [task1, task2])

    losses = []
    for _ in range(5):
        result = trainer.train_step()
        losses.append(result["loss"])

    assert all(torch.isfinite(torch.tensor(l)) for l in losses), "All losses should be finite"


# ---------------------------------------------------------------------------
# Test 15: MultiTaskPCGradTrainer works with 3 tasks
# ---------------------------------------------------------------------------

def test_multitask_trainer_three_tasks():
    torch.manual_seed(77)
    model = make_simple_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    cfg = PCGradConfig(n_tasks=3)

    x = torch.randn(4, 8)

    def task1(m: nn.Module) -> torch.Tensor:
        return m(x).mean()

    def task2(m: nn.Module) -> torch.Tensor:
        return (m(x) - 1.0).pow(2).mean()

    def task3(m: nn.Module) -> torch.Tensor:
        return m(x).abs().sum()

    trainer = MultiTaskPCGradTrainer(model, opt, cfg, [task1, task2, task3])
    result = trainer.train_step()

    assert "loss" in result
    assert "n_conflicts" in result
    assert "task_losses" in result
    assert len(result["task_losses"]) == 3
    assert isinstance(result["n_conflicts"], int)
    assert result["n_conflicts"] >= 0
    assert torch.isfinite(torch.tensor(result["loss"]))
