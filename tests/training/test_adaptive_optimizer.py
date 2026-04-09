"""Tests for src/training/adaptive_optimizer.py."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.training.adaptive_optimizer import (
    GradientPreconditioner,
    LionOptimizer,
    OptimizerConfig,
    SignSGD,
    SOAPOptimizer,
    benchmark_optimizer_convergence,
    clip_gradients_by_norm,
    compute_gradient_norm,
)

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_and_input():
    """Return a small Linear model and a random input tensor."""
    torch.manual_seed(0)
    model = nn.Linear(4, 4)
    x = torch.randn(8, 4)
    return model, x


def _backward(model, x):
    """Run a forward + backward pass and return the loss."""
    out = model(x)
    loss = out.sum()
    loss.backward()
    return loss


# ---------------------------------------------------------------------------
# 1. OptimizerConfig defaults
# ---------------------------------------------------------------------------


def test_optimizer_config_defaults():
    cfg = OptimizerConfig()
    assert cfg.lr == 1e-4
    assert cfg.betas == (0.9, 0.99)
    assert cfg.weight_decay == 0.0
    assert cfg.clip_threshold == 1.0
    assert cfg.precondition_freq == 10


# ---------------------------------------------------------------------------
# 2. LionOptimizer — params change after a step
# ---------------------------------------------------------------------------


def test_lion_step_updates_params():
    torch.manual_seed(0)
    model, x = _make_model_and_input()
    params_before = [p.clone() for p in model.parameters()]

    opt = LionOptimizer(model.parameters(), lr=1e-3)
    _backward(model, x)
    opt.step()

    for p_before, p_after in zip(params_before, model.parameters()):
        assert not torch.allclose(p_before, p_after), "Parameter should have changed."


# ---------------------------------------------------------------------------
# 3. LionOptimizer — weight decay moves params toward zero
# ---------------------------------------------------------------------------


def test_lion_weight_decay_applied():
    torch.manual_seed(0)
    model_wd, x = _make_model_and_input()
    model_no_wd, _ = _make_model_and_input()

    # Copy identical weights
    with torch.no_grad():
        for p_wd, p_no in zip(model_wd.parameters(), model_no_wd.parameters()):
            p_no.copy_(p_wd)

    opt_wd = LionOptimizer(model_wd.parameters(), lr=1e-3, weight_decay=0.1)
    opt_no = LionOptimizer(model_no_wd.parameters(), lr=1e-3, weight_decay=0.0)

    # Same forward / backward
    out_wd = model_wd(x)
    out_wd.sum().backward()
    opt_wd.step()

    out_no = model_no_wd(x)
    out_no.sum().backward()
    opt_no.step()

    # Params with weight decay should differ from those without
    any_diff = any(
        not torch.allclose(p_wd, p_no)
        for p_wd, p_no in zip(model_wd.parameters(), model_no_wd.parameters())
    )
    assert any_diff, "Weight decay should produce different parameters."


# ---------------------------------------------------------------------------
# 4. SignSGD — params change after a step
# ---------------------------------------------------------------------------


def test_sign_sgd_step_updates_params():
    torch.manual_seed(0)
    model, x = _make_model_and_input()
    params_before = [p.clone() for p in model.parameters()]

    opt = SignSGD(model.parameters(), lr=1e-3)
    _backward(model, x)
    opt.step()

    for p_before, p_after in zip(params_before, model.parameters()):
        assert not torch.allclose(p_before, p_after), "Parameter should have changed."


# ---------------------------------------------------------------------------
# 5. SignSGD — update magnitude is bounded by lr
# ---------------------------------------------------------------------------


def test_sign_sgd_update_is_bounded():
    torch.manual_seed(0)
    model, x = _make_model_and_input()

    lr = 1e-3
    params_before = [p.clone() for p in model.parameters()]

    opt = SignSGD(model.parameters(), lr=lr)
    _backward(model, x)
    opt.step()

    for p_before, p_after in zip(params_before, model.parameters()):
        diff = (p_after - p_before).abs()
        # Each element change = lr * |sign(m)| = lr (when no weight decay)
        assert (diff <= lr + 1e-6).all(), (
            f"Update should be at most lr={lr}, got max {diff.max().item()}"
        )


# ---------------------------------------------------------------------------
# 6. compute_gradient_norm — returns a positive float
# ---------------------------------------------------------------------------


def test_compute_gradient_norm_positive():
    torch.manual_seed(0)
    model, x = _make_model_and_input()
    _backward(model, x)

    norm = compute_gradient_norm(list(model.parameters()))
    assert isinstance(norm, float)
    assert norm > 0.0, "Gradient norm should be positive after backward."


# ---------------------------------------------------------------------------
# 7. clip_gradients_by_norm — returns the pre-clip norm
# ---------------------------------------------------------------------------


def test_clip_gradients_returns_norm():
    torch.manual_seed(0)
    model, x = _make_model_and_input()
    _backward(model, x)

    pre_norm = compute_gradient_norm(list(model.parameters()))
    returned_norm = clip_gradients_by_norm(list(model.parameters()), max_norm=0.1)

    assert abs(returned_norm - pre_norm) < 1e-5, (
        "clip_gradients_by_norm should return the norm BEFORE clipping."
    )
    post_norm = compute_gradient_norm(list(model.parameters()))
    assert post_norm <= 0.1 + 1e-5, "Post-clip norm should not exceed max_norm."


# ---------------------------------------------------------------------------
# 8. GradientPreconditioner — update populates _v2
# ---------------------------------------------------------------------------


def test_gradient_preconditioner_update():
    torch.manual_seed(0)
    model, x = _make_model_and_input()
    _backward(model, x)

    params = list(model.parameters())
    precond = GradientPreconditioner(params)

    # Before any update, _v2 should be empty
    assert len(precond._v2) == 0

    # Update at step 10 (matches default freq)
    precond.update(step=10, freq=10)
    assert len(precond._v2) > 0, "_v2 should be populated after update at freq step."

    # precondition should return a tensor for a param with gradient
    result = precond.precondition(params[0])
    assert result is not None
    assert result.shape == params[0].grad.shape


# ---------------------------------------------------------------------------
# 9. SOAPOptimizer — params change after a step
# ---------------------------------------------------------------------------


def test_soap_optimizer_step_updates_params():
    torch.manual_seed(0)
    model, x = _make_model_and_input()
    params_before = [p.clone() for p in model.parameters()]

    cfg = OptimizerConfig(lr=1e-3)
    opt = SOAPOptimizer(model.parameters(), config=cfg)
    _backward(model, x)
    opt.step()

    for p_before, p_after in zip(params_before, model.parameters()):
        assert not torch.allclose(p_before, p_after), "Parameter should have changed."


# ---------------------------------------------------------------------------
# 10. SOAPOptimizer — loss decreases over multiple steps
# ---------------------------------------------------------------------------


def test_soap_optimizer_loss_decreases():
    torch.manual_seed(0)
    model = nn.Linear(4, 4)
    x = torch.randn(8, 4)

    cfg = OptimizerConfig(lr=1e-2)
    opt = SOAPOptimizer(model.parameters(), config=cfg)

    losses = []
    for _ in range(20):
        opt.zero_grad()
        loss = model(x).sum()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], (
        f"Loss should decrease: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
    )


# ---------------------------------------------------------------------------
# 11. benchmark_optimizer_convergence — returns the expected keys
# ---------------------------------------------------------------------------


def test_benchmark_convergence_keys():
    torch.manual_seed(0)
    model = nn.Linear(4, 4)
    opt = LionOptimizer(model.parameters(), lr=1e-3)

    result = benchmark_optimizer_convergence(model, opt, n_steps=10)
    assert set(result.keys()) == {"initial_loss", "final_loss", "convergence_ratio"}


# ---------------------------------------------------------------------------
# 12. benchmark_optimizer_convergence — convergence_ratio is positive
# ---------------------------------------------------------------------------


def test_benchmark_convergence_ratio_positive():
    torch.manual_seed(0)
    model = nn.Linear(4, 4)
    opt = LionOptimizer(model.parameters(), lr=1e-3)

    result = benchmark_optimizer_convergence(model, opt, n_steps=20)
    assert result["convergence_ratio"] > 0.0, (
        f"convergence_ratio should be positive, got {result['convergence_ratio']}"
    )
