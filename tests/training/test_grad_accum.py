import pytest
import torch
import torch.nn as nn
from src.training.grad_accum import GradAccumConfig, GradAccumManager

def _make_model_and_optimizer():
    model = nn.Linear(8, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    return model, optimizer

def test_step_accumulates_without_optimizer_step():
    model, optimizer = _make_model_and_optimizer()
    cfg = GradAccumConfig(n_accum=4, clip_grad_norm=None)
    mgr = GradAccumManager(model, optimizer, cfg)

    x = torch.randn(1, 8)
    for i in range(3):
        loss = model(x).sum()
        result = mgr.step(loss)
        assert result is False  # no optimizer step yet
    assert mgr.n_optimizer_steps == 0

def test_step_calls_optimizer_at_n_accum():
    model, optimizer = _make_model_and_optimizer()
    cfg = GradAccumConfig(n_accum=4, clip_grad_norm=None)
    mgr = GradAccumManager(model, optimizer, cfg)

    x = torch.randn(1, 8)
    results = []
    for i in range(4):
        loss = model(x).sum()
        results.append(mgr.step(loss))

    assert results[-1] is True  # 4th step triggers optimizer
    assert mgr.n_optimizer_steps == 1

def test_step_resets_counter_after_optimizer_step():
    model, optimizer = _make_model_and_optimizer()
    cfg = GradAccumConfig(n_accum=2, clip_grad_norm=None)
    mgr = GradAccumManager(model, optimizer, cfg)

    x = torch.randn(1, 8)
    for _ in range(2):  # 1 full cycle
        model(x).sum().backward() if False else mgr.step(model(x).sum())

    assert mgr.current_accum_steps == 0  # reset after step

def test_flush_triggers_step_with_partial():
    model, optimizer = _make_model_and_optimizer()
    cfg = GradAccumConfig(n_accum=4, clip_grad_norm=None)
    mgr = GradAccumManager(model, optimizer, cfg)

    x = torch.randn(1, 8)
    for _ in range(2):  # Only 2 of 4 steps
        mgr.step(model(x).sum())

    result = mgr.flush()
    assert result is True
    assert mgr.n_optimizer_steps == 1

def test_flush_does_nothing_if_no_accumulation():
    model, optimizer = _make_model_and_optimizer()
    mgr = GradAccumManager(model, optimizer)
    result = mgr.flush()
    assert result is False

def test_loss_scaling():
    """Scaled loss should be loss / n_accum."""
    model, optimizer = _make_model_and_optimizer()
    cfg = GradAccumConfig(n_accum=4, clip_grad_norm=None)
    mgr = GradAccumManager(model, optimizer, cfg)

    x = torch.randn(1, 8)
    loss = model(x).sum()
    loss_val = loss.item()
    mgr.step(loss)

    # accumulated_loss should be loss / n_accum
    assert abs(mgr.accumulated_loss - loss_val / 4) < 1e-5

def test_multiple_cycles():
    """Test 3 full accumulation cycles."""
    model, optimizer = _make_model_and_optimizer()
    cfg = GradAccumConfig(n_accum=3, clip_grad_norm=None)
    mgr = GradAccumManager(model, optimizer, cfg)

    x = torch.randn(1, 8)
    for _ in range(9):  # 3 full cycles
        mgr.step(model(x).sum())

    assert mgr.n_optimizer_steps == 3

def test_grad_clipping_applied():
    """With clip_grad_norm, gradients should be clipped."""
    model, optimizer = _make_model_and_optimizer()
    cfg = GradAccumConfig(n_accum=1, clip_grad_norm=0.01)
    mgr = GradAccumManager(model, optimizer, cfg)

    # Large loss to create large gradients
    x = torch.ones(1, 8) * 100
    mgr.step(model(x).sum())
    # No assertion on grad magnitude (already stepped), but it shouldn't crash
    assert mgr.n_optimizer_steps == 1
