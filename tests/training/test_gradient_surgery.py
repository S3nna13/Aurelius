"""Tests for gradient_surgery.py: PCGrad and gradient surgery for multi-task learning."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.training.gradient_surgery import (
    PCGrad,
    GradientSurgeryMonitor,
    MultiTaskTrainer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_simple_model():
    """A tiny linear model for testing."""
    torch.manual_seed(0)
    return nn.Linear(4, 2, bias=False)


def make_optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=0.01)


# ---------------------------------------------------------------------------
# PCGrad.project_gradient tests
# ---------------------------------------------------------------------------

def test_project_gradient_no_conflict():
    """Orthogonal gradients should remain unchanged."""
    g_i = torch.tensor([1.0, 0.0])
    g_j = torch.tensor([0.0, 1.0])  # orthogonal: dot = 0, not < 0
    result = PCGrad.project_gradient(g_i, g_j)
    assert torch.allclose(result, g_i), "Orthogonal gradients should be unchanged"


def test_project_gradient_conflict():
    """Conflicting gradients (dot < 0) should be projected."""
    g_i = torch.tensor([1.0, 0.0])
    g_j = torch.tensor([-1.0, 0.0])  # exactly opposite -> conflicting
    result = PCGrad.project_gradient(g_i, g_j)
    # The projection should remove the component along g_j
    # g_i . g_j = -1 < 0, so projection = g_i - (-1/1)*g_j = g_i + g_j = [0, 0]
    assert not torch.allclose(result, g_i), "Conflicting gradient should be modified"
    # After projection, dot product with g_j should be >= 0
    dot_after = torch.dot(result.view(-1), g_j.view(-1)).item()
    assert dot_after >= -1e-6, f"After projection, dot should be >= 0, got {dot_after}"


def test_project_gradient_same_direction():
    """Parallel gradients (same direction) should be unchanged."""
    g_i = torch.tensor([1.0, 2.0, 3.0])
    g_j = torch.tensor([2.0, 4.0, 6.0])  # same direction, dot > 0
    result = PCGrad.project_gradient(g_i, g_j)
    assert torch.allclose(result, g_i), "Parallel gradients should be unchanged"


# ---------------------------------------------------------------------------
# PCGrad.pc_backward test
# ---------------------------------------------------------------------------

def test_pc_backward_no_error():
    """pc_backward with 2 task losses should run without error."""
    model = make_simple_model()
    optimizer = make_optimizer(model)
    pcgrad = PCGrad(optimizer, reduction='mean')

    x = torch.randn(3, 4)
    loss1 = model(x).sum()
    loss2 = (model(x) ** 2).sum()

    # Should not raise
    pcgrad.pc_backward([loss1, loss2])

    # At least some parameters should have gradients
    grad_set = [p.grad is not None for p in model.parameters()]
    assert any(grad_set), "At least one parameter should have a gradient after pc_backward"


# ---------------------------------------------------------------------------
# GradientSurgeryMonitor tests
# ---------------------------------------------------------------------------

def test_conflict_monitor_range():
    """measure_conflict should return a value in [0, 1]."""
    monitor = GradientSurgeryMonitor()
    torch.manual_seed(42)
    grads = [torch.randn(10) for _ in range(3)]
    rate = monitor.measure_conflict(grads)
    assert 0.0 <= rate <= 1.0, f"Conflict rate {rate} should be in [0, 1]"


def test_conflict_monitor_identical():
    """Identical gradients have dot > 0 for all pairs -> 0 conflict rate."""
    monitor = GradientSurgeryMonitor()
    g = torch.tensor([1.0, 2.0, 3.0])
    grads = [g.clone() for _ in range(3)]
    rate = monitor.measure_conflict(grads)
    assert rate == 0.0, f"Identical gradients should have 0 conflict rate, got {rate}"


def test_conflict_monitor_opposing():
    """Fully opposing gradients should give 1.0 conflict rate."""
    monitor = GradientSurgeryMonitor()
    g = torch.tensor([1.0, 2.0, 3.0])
    # [g, -g]: every pair (g,-g) and (-g,g) is conflicting -> all pairs conflict
    grads = [g.clone(), -g.clone()]
    rate = monitor.measure_conflict(grads)
    assert rate == pytest.approx(1.0), f"Opposing gradients should have 1.0 conflict rate, got {rate}"


# ---------------------------------------------------------------------------
# MultiTaskTrainer tests
# ---------------------------------------------------------------------------

def test_multitask_trainer_keys():
    """train_step should return dict with 'task_losses', 'total_loss', 'conflict_rate'."""
    model = make_simple_model()
    optimizer = make_optimizer(model)

    def loss_fn_1(m, batch):
        return m(batch).sum()

    def loss_fn_2(m, batch):
        return (m(batch) ** 2).mean()

    trainer = MultiTaskTrainer(
        model=model,
        task_loss_fns=[loss_fn_1, loss_fn_2],
        optimizer=optimizer,
        use_pcgrad=True,
    )

    batches = [torch.randn(3, 4), torch.randn(3, 4)]
    result = trainer.train_step(batches)

    assert 'task_losses' in result, "Result must contain 'task_losses'"
    assert 'total_loss' in result, "Result must contain 'total_loss'"
    assert 'conflict_rate' in result, "Result must contain 'conflict_rate'"


def test_pcgrad_reduces_conflict():
    """After PCGrad, conflict rate should be 0 (or lower than before)."""
    torch.manual_seed(7)
    model = make_simple_model()

    # Construct gradients that are conflicting
    monitor = GradientSurgeryMonitor()
    g1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    g2 = torch.tensor([-1.0, 0.0, 0.0, 0.0])  # exactly opposite

    # Pre-PCGrad conflict
    conflict_before = monitor.measure_conflict([g1, g2])
    assert conflict_before == pytest.approx(1.0), "Should have 100% conflict before surgery"

    # Apply PCGrad projection
    proj_g1 = PCGrad.project_gradient(g1.clone(), g2)
    proj_g2 = PCGrad.project_gradient(g2.clone(), g1)

    conflict_after = monitor.measure_conflict([proj_g1, proj_g2])
    assert conflict_after < conflict_before, (
        f"Conflict rate should decrease after PCGrad: before={conflict_before}, after={conflict_after}"
    )
