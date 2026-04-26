"""Tests for dynamic coefficient scheduling (src/training/dynamic_coef.py)."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.training.dynamic_coef import (
    DynamicCoefConfig,
    GradNormScheduler,
    LossRatioScheduler,
    UncertaintyWeighting,
    create_dynamic_coef_scheduler,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_shared_param(requires_grad: bool = True) -> nn.Parameter:
    """Small shared parameter for GradNorm tests."""
    return nn.Parameter(torch.randn(4, 4), requires_grad=requires_grad)


def _make_task_losses(n: int, shared_param: nn.Parameter) -> list[torch.Tensor]:
    """Create n scalar task losses that depend on shared_param."""
    return [(shared_param * (i + 1)).sum() for i in range(n)]


# ---------------------------------------------------------------------------
# Test 1 – DynamicCoefConfig defaults
# ---------------------------------------------------------------------------


def test_dynamic_coef_config_defaults():
    cfg = DynamicCoefConfig()
    assert cfg.method == "uncertainty"
    assert cfg.n_tasks == 2
    assert cfg.alpha == 1.5
    assert cfg.lr == 0.01
    assert cfg.adjustment_rate == 0.01
    assert cfg.normalize is True


# ---------------------------------------------------------------------------
# Test 2 – GradNormScheduler.get_weights sums to 1.0
# ---------------------------------------------------------------------------


def test_gradnorm_weights_sum_to_one():
    sched = GradNormScheduler(n_tasks=3)
    weights = sched.get_weights()
    assert weights.shape == (3,)
    assert torch.isclose(weights.sum(), torch.tensor(1.0)), (
        f"Weights should sum to 1.0, got {weights.sum().item():.6f}"
    )


# ---------------------------------------------------------------------------
# Test 3 – GradNormScheduler.update returns scalar loss
# ---------------------------------------------------------------------------


def test_gradnorm_update_returns_scalar():
    torch.manual_seed(0)
    n_tasks = 2
    sched = GradNormScheduler(n_tasks=n_tasks)
    shared_param = _make_shared_param()
    losses = _make_task_losses(n_tasks, shared_param)
    gradnorm_loss = sched.update(losses, shared_params=[shared_param])
    assert isinstance(gradnorm_loss, torch.Tensor)
    assert gradnorm_loss.ndim == 0, "GradNorm loss should be a scalar (0-d tensor)"


# ---------------------------------------------------------------------------
# Test 4 – UncertaintyWeighting.compute_weighted_loss returns (loss, weights)
# ---------------------------------------------------------------------------


def test_uncertainty_weighting_returns_tuple():
    uw = UncertaintyWeighting(n_tasks=3)
    losses = [torch.tensor(line, dtype=torch.float32) for line in [0.5, 0.3, 0.2]]
    result = uw.compute_weighted_loss(losses)
    assert isinstance(result, tuple) and len(result) == 2
    total_loss, per_task_weights = result
    assert isinstance(total_loss, torch.Tensor)
    assert isinstance(per_task_weights, torch.Tensor)


# ---------------------------------------------------------------------------
# Test 5 – Uncertainty per-task weights are positive
# ---------------------------------------------------------------------------


def test_uncertainty_weights_positive():
    uw = UncertaintyWeighting(n_tasks=4)
    losses = [torch.tensor(float(i + 1)) for i in range(4)]
    _, per_task_weights = uw.compute_weighted_loss(losses)
    assert (per_task_weights > 0).all(), (
        "All per-task weights (1 / (2 * sigma^2)) should be positive"
    )


# ---------------------------------------------------------------------------
# Test 6 – get_log_variances has correct shape (n_tasks,)
# ---------------------------------------------------------------------------


def test_uncertainty_log_variances_shape():
    n_tasks = 5
    uw = UncertaintyWeighting(n_tasks=n_tasks)
    log_vars = uw.get_log_variances()
    assert log_vars.shape == (n_tasks,), f"Expected shape ({n_tasks},), got {log_vars.shape}"


# ---------------------------------------------------------------------------
# Test 7 – LossRatioScheduler.get_coefficients returns dict with correct keys
# ---------------------------------------------------------------------------


def test_loss_ratio_scheduler_keys():
    targets = {"lm": 0.6, "aux": 0.4}
    sched = LossRatioScheduler(target_ratios=targets)
    coefs = sched.get_coefficients()
    assert set(coefs.keys()) == {"lm", "aux"}


# ---------------------------------------------------------------------------
# Test 8 – LossRatioScheduler adjusts coefficients toward target ratio
# ---------------------------------------------------------------------------


def test_loss_ratio_scheduler_adjusts():
    # Target: lm should be 0.9 of total, aux should be 0.1.
    targets = {"lm": 0.9, "aux": 0.1}
    sched = LossRatioScheduler(target_ratios=targets, adjustment_rate=0.1)

    # Current losses heavily favour aux (lm=0.1, aux=0.9) — opposite of target.
    current_losses = {"lm": 0.1, "aux": 0.9}
    coefs_before = sched.get_coefficients().copy()

    for _ in range(10):
        sched.update(current_losses)

    coefs_after = sched.get_coefficients()

    # lm coefficient should have grown (need to amplify lm contribution).
    assert coefs_after["lm"] > coefs_before["lm"], (
        "lm coefficient should increase when lm loss ratio < target"
    )
    # aux coefficient should have shrunk (aux is over-represented).
    assert coefs_after["aux"] < coefs_before["aux"], (
        "aux coefficient should decrease when aux loss ratio > target"
    )


# ---------------------------------------------------------------------------
# Test 9 – create_dynamic_coef_scheduler returns GradNormScheduler
# ---------------------------------------------------------------------------


def test_factory_returns_gradnorm():
    cfg = DynamicCoefConfig(method="gradnorm", n_tasks=3)
    sched = create_dynamic_coef_scheduler(cfg, task_names=["a", "b", "c"])
    assert isinstance(sched, GradNormScheduler)


# ---------------------------------------------------------------------------
# Test 10 – create_dynamic_coef_scheduler returns UncertaintyWeighting
# ---------------------------------------------------------------------------


def test_factory_returns_uncertainty():
    cfg = DynamicCoefConfig(method="uncertainty", n_tasks=2)
    sched = create_dynamic_coef_scheduler(cfg, task_names=["lm", "aux"])
    assert isinstance(sched, UncertaintyWeighting)


# ---------------------------------------------------------------------------
# Test 11 – Gradient flows through UncertaintyWeighting loss
# ---------------------------------------------------------------------------


def test_uncertainty_gradient_flow():
    uw = UncertaintyWeighting(n_tasks=2)
    assert uw.log_var.grad is None

    losses = [torch.tensor(0.5, requires_grad=False), torch.tensor(0.3, requires_grad=False)]
    total_loss, _ = uw.compute_weighted_loss(losses)
    total_loss.backward()

    assert uw.log_var.grad is not None, "Gradient should flow back to log_var parameters"
    assert not torch.isnan(uw.log_var.grad).any(), "Gradients should not be NaN"


# ---------------------------------------------------------------------------
# Test 12 – GradNormScheduler weights are learnable parameters
# ---------------------------------------------------------------------------


def test_gradnorm_weights_are_learnable():
    sched = GradNormScheduler(n_tasks=2)
    # log_weights must be an nn.Parameter with requires_grad=True.
    assert isinstance(sched.log_weights, nn.Parameter), "log_weights should be an nn.Parameter"
    assert sched.log_weights.requires_grad, "log_weights should have requires_grad=True"
