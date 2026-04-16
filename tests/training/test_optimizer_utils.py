"""Tests for src/training/optimizer_utils.py.

All tests use tiny models and minimal configs to keep runtime fast.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from src.training.optimizer_utils import (
    GradientAccumulator,
    OptimizerConfig,
    build_optimizer,
    clip_grad_norm_custom,
    compute_effective_lr,
    compute_grad_norm,
    get_lr_schedule,
    get_param_groups,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _tiny_linear() -> nn.Linear:
    """2→2 linear layer: weight (2×2) + bias (2,)."""
    torch.manual_seed(42)
    return nn.Linear(2, 2)


def _tiny_sequential() -> nn.Sequential:
    """Small sequential: Linear → LayerNorm → Linear."""
    torch.manual_seed(42)
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.LayerNorm(8),
        nn.Linear(8, 4),
    )


def _run_backward(model: nn.Module) -> None:
    """Run a tiny forward + backward pass."""
    model.zero_grad()
    x = torch.randn(1, model[0].in_features if isinstance(model, nn.Sequential) else model.in_features)
    out = model(x)
    out.sum().backward()


def _run_backward_linear(model: nn.Linear) -> None:
    model.zero_grad()
    x = torch.randn(1, model.in_features)
    out = model(x)
    out.sum().backward()


# ---------------------------------------------------------------------------
# 1. OptimizerConfig defaults
# ---------------------------------------------------------------------------


def test_optimizer_config_defaults():
    cfg = OptimizerConfig()
    assert cfg.lr == pytest.approx(3e-4)
    assert cfg.weight_decay == pytest.approx(0.1)
    assert cfg.beta1 == pytest.approx(0.9)
    assert cfg.beta2 == pytest.approx(0.95)
    assert cfg.eps == pytest.approx(1e-8)
    assert cfg.grad_clip == pytest.approx(1.0)
    assert cfg.warmup_steps == 2000
    assert cfg.total_steps == 100_000
    assert cfg.schedule == "cosine"


def test_optimizer_config_invalid_schedule():
    with pytest.raises(ValueError, match="schedule"):
        OptimizerConfig(schedule="exponential")


# ---------------------------------------------------------------------------
# 2. get_param_groups — returns exactly 2 groups
# ---------------------------------------------------------------------------


def test_get_param_groups_returns_two_groups():
    model = _tiny_sequential()
    groups = get_param_groups(model, weight_decay=0.1)
    assert len(groups) == 2


# ---------------------------------------------------------------------------
# 3. get_param_groups — 2D params in wd group, 1D in no-wd group
# ---------------------------------------------------------------------------


def test_get_param_groups_2d_in_wd_group():
    model = _tiny_sequential()
    groups = get_param_groups(model, weight_decay=0.1)
    wd_group = groups[0]
    no_wd_group = groups[1]

    # All tensors in the weight-decay group must be 2D+
    for p in wd_group["params"]:
        assert p.ndim >= 2, f"Expected ndim>=2 in wd group, got {p.ndim}"

    # All tensors in the no-wd group must be 1D
    for p in no_wd_group["params"]:
        assert p.ndim < 2, f"Expected ndim<2 in no-wd group, got {p.ndim}"

    assert wd_group["weight_decay"] == pytest.approx(0.1)
    assert no_wd_group["weight_decay"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 4. compute_grad_norm — non-negative float after backward
# ---------------------------------------------------------------------------


def test_compute_grad_norm_positive_after_backward():
    model = _tiny_linear()
    _run_backward_linear(model)
    norm = compute_grad_norm(model)
    assert isinstance(norm, float)
    assert norm > 0.0


# ---------------------------------------------------------------------------
# 5. compute_grad_norm — returns 0 with no grads
# ---------------------------------------------------------------------------


def test_compute_grad_norm_zero_with_no_grads():
    model = _tiny_linear()
    model.zero_grad()  # grads are None
    norm = compute_grad_norm(model)
    assert norm == 0.0


# ---------------------------------------------------------------------------
# 6. clip_grad_norm_custom — clips to max_norm
# ---------------------------------------------------------------------------


def test_clip_grad_norm_custom_clips_to_max_norm():
    model = _tiny_linear()
    _run_backward_linear(model)
    max_norm = 0.001  # very small to force clipping
    clip_grad_norm_custom(model, max_norm)
    post_norm = compute_grad_norm(model)
    assert post_norm <= max_norm + 1e-6


# ---------------------------------------------------------------------------
# 7. clip_grad_norm_custom — returns pre-clip norm
# ---------------------------------------------------------------------------


def test_clip_grad_norm_custom_returns_pre_clip_norm():
    model = _tiny_linear()
    _run_backward_linear(model)
    pre = compute_grad_norm(model)
    returned = clip_grad_norm_custom(model, max_norm=1e9)  # large → no actual clipping
    assert abs(returned - pre) < 1e-6


# ---------------------------------------------------------------------------
# 8. get_lr_schedule — warmup increases linearly
# ---------------------------------------------------------------------------


def test_get_lr_schedule_warmup_linear():
    cfg = OptimizerConfig(warmup_steps=10, total_steps=100, schedule="cosine")
    multipliers = [get_lr_schedule(s, cfg) for s in range(11)]
    # Step 0 should be 0, step 10 should be 1
    assert multipliers[0] == pytest.approx(0.0)
    assert multipliers[10] == pytest.approx(1.0)
    # Each step should be larger than the previous during warmup
    for i in range(1, 11):
        assert multipliers[i] > multipliers[i - 1], (
            f"Expected monotone increase at step {i}"
        )


# ---------------------------------------------------------------------------
# 9. get_lr_schedule — cosine reaches ~0.1 at total_steps
# ---------------------------------------------------------------------------


def test_get_lr_schedule_cosine_min_at_total_steps():
    cfg = OptimizerConfig(warmup_steps=0, total_steps=1000, schedule="cosine")
    final = get_lr_schedule(1000, cfg)
    # cosine minimum is 0.1 (10% of base lr)
    assert abs(final - 0.1) < 1e-6


# ---------------------------------------------------------------------------
# 10. get_lr_schedule — constant stays flat after warmup
# ---------------------------------------------------------------------------


def test_get_lr_schedule_constant_stays_flat():
    cfg = OptimizerConfig(warmup_steps=5, total_steps=50, schedule="constant")
    post_warmup = [get_lr_schedule(s, cfg) for s in range(5, 51)]
    for mult in post_warmup:
        assert mult == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 11. build_optimizer — returns AdamW
# ---------------------------------------------------------------------------


def test_build_optimizer_returns_adamw():
    model = _tiny_linear()
    cfg = OptimizerConfig()
    opt = build_optimizer(model, cfg)
    assert isinstance(opt, torch.optim.AdamW)


def test_build_optimizer_uses_config_lr():
    model = _tiny_linear()
    cfg = OptimizerConfig(lr=1e-3)
    opt = build_optimizer(model, cfg)
    for pg in opt.param_groups:
        assert pg["lr"] == pytest.approx(1e-3)


# ---------------------------------------------------------------------------
# 12. GradientAccumulator — returns False until accumulation_steps calls
# ---------------------------------------------------------------------------


def test_gradient_accumulator_false_before_full_cycle():
    model = _tiny_linear()
    accum = GradientAccumulator(accumulation_steps=4, model=model)
    results = []
    for i in range(3):
        model.zero_grad()
        loss = torch.tensor(1.0, requires_grad=True)
        results.append(accum.accumulate(loss))
    assert all(r is False for r in results)


# ---------------------------------------------------------------------------
# 13. GradientAccumulator — returns True at every accumulation_steps
# ---------------------------------------------------------------------------


def test_gradient_accumulator_true_at_full_cycle():
    model = _tiny_linear()
    accum = GradientAccumulator(accumulation_steps=3, model=model)
    results = []
    for i in range(9):
        loss = torch.tensor(1.0, requires_grad=True)
        results.append(accum.accumulate(loss))
    # Indices 2, 5, 8 (0-indexed) should be True
    assert results[2] is True
    assert results[5] is True
    assert results[8] is True
    # Others should be False
    for idx in [0, 1, 3, 4, 6, 7]:
        assert results[idx] is False


def test_gradient_accumulator_step_count():
    model = _tiny_linear()
    accum = GradientAccumulator(accumulation_steps=2, model=model)
    assert accum.step_count() == 0
    for _ in range(6):
        loss = torch.tensor(1.0, requires_grad=True)
        accum.accumulate(loss)
    assert accum.step_count() == 3


def test_gradient_accumulator_reset():
    model = _tiny_linear()
    accum = GradientAccumulator(accumulation_steps=2, model=model)
    for _ in range(4):
        loss = torch.tensor(1.0, requires_grad=True)
        accum.accumulate(loss)
    assert accum.step_count() == 2
    accum.reset()
    assert accum.step_count() == 0


# ---------------------------------------------------------------------------
# 14. compute_effective_lr — returns positive float
# ---------------------------------------------------------------------------


def test_compute_effective_lr_positive():
    model = _tiny_linear()
    cfg = OptimizerConfig(lr=5e-4)
    opt = build_optimizer(model, cfg)
    lr = compute_effective_lr(opt)
    assert isinstance(lr, float)
    assert lr > 0.0
    assert lr == pytest.approx(5e-4)


# ---------------------------------------------------------------------------
# 15. get_lr_schedule — linear schedule decays to 0
# ---------------------------------------------------------------------------


def test_get_lr_schedule_linear_decays_to_zero():
    cfg = OptimizerConfig(warmup_steps=0, total_steps=100, schedule="linear")
    final = get_lr_schedule(100, cfg)
    assert final == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 16. get_param_groups — coverage for pure Linear (no LayerNorm)
# ---------------------------------------------------------------------------


def test_get_param_groups_linear_only():
    model = _tiny_linear()
    groups = get_param_groups(model, weight_decay=0.05)
    # weight (2D) and bias (1D)
    wd_params = groups[0]["params"]
    no_wd_params = groups[1]["params"]
    assert len(wd_params) == 1    # weight matrix
    assert len(no_wd_params) == 1  # bias vector
    assert wd_params[0].ndim == 2
    assert no_wd_params[0].ndim == 1
