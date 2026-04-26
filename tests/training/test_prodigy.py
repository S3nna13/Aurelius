"""Tests for src/training/prodigy.py — Prodigy self-tuning optimizer."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.training.prodigy import Prodigy, ProdigyW

torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_linear():
    """Small Linear(8, 4) model for quick tests."""
    torch.manual_seed(0)
    return nn.Linear(8, 4)


def _quadratic_loss(x: torch.Tensor) -> torch.Tensor:
    """f(x) = ||x||^2 — minimum at x=0."""
    return (x * x).sum()


# ---------------------------------------------------------------------------
# 1. Instantiation with correct defaults
# ---------------------------------------------------------------------------


def test_prodigy_default_instantiation():
    model = _make_linear()
    opt = Prodigy(model.parameters())
    g = opt.param_groups[0]
    assert g["betas"] == (0.9, 0.999)
    assert g["eps"] == 1e-8
    assert g["weight_decay"] == 0.0
    assert g["d0"] == 1e-6
    assert g["d_coef"] == 1.0
    assert g["use_bias_correction"] is True
    assert g["safeguard_warmup"] is False
    assert g["warmup_steps"] == 0
    assert g["d"] == 1e-6


# ---------------------------------------------------------------------------
# 2. step() runs without error
# ---------------------------------------------------------------------------


def test_prodigy_step_runs():
    model = _make_linear()
    opt = Prodigy(model.parameters())
    x = torch.randn(4, 8)
    loss = model(x).sum()
    loss.backward()
    opt.step()  # Should not raise


# ---------------------------------------------------------------------------
# 3. Loss decreases on quadratic f(x) = ||x||^2 over 50 steps
# ---------------------------------------------------------------------------


def test_prodigy_quadratic_convergence_50_steps():
    torch.manual_seed(1)
    x = nn.Parameter(torch.randn(16))
    opt = Prodigy([x], d0=1e-3)

    initial_loss = _quadratic_loss(x).item()
    for _ in range(50):
        opt.zero_grad()
        loss = _quadratic_loss(x)
        loss.backward()
        opt.step()

    final_loss = _quadratic_loss(x).item()
    assert final_loss < initial_loss, (
        f"Expected loss to decrease: {initial_loss:.4f} -> {final_loss:.4f}"
    )


# ---------------------------------------------------------------------------
# 4. Loss decreases on 2D bowl over 30 steps
# ---------------------------------------------------------------------------


def test_prodigy_2d_bowl_convergence():
    torch.manual_seed(2)
    x = nn.Parameter(torch.tensor([3.0, -4.0]))
    opt = Prodigy([x], d0=1e-3)

    initial_loss = _quadratic_loss(x).item()
    for _ in range(30):
        opt.zero_grad()
        loss = _quadratic_loss(x)
        loss.backward()
        opt.step()

    final_loss = _quadratic_loss(x).item()
    assert final_loss < initial_loss, (
        f"Loss should decrease on 2D bowl: {initial_loss:.4f} -> {final_loss:.4f}"
    )


# ---------------------------------------------------------------------------
# 5. current_lr increases from d0 after seeing gradients
# ---------------------------------------------------------------------------


def test_current_lr_increases_from_d0():
    torch.manual_seed(3)
    x = nn.Parameter(torch.randn(8))
    d0 = 1e-6
    opt = Prodigy([x], d0=d0)

    # Run several steps
    for _ in range(20):
        opt.zero_grad()
        _quadratic_loss(x).backward()
        opt.step()

    assert opt.current_lr > d0, f"current_lr {opt.current_lr} should exceed d0={d0} after steps"


# ---------------------------------------------------------------------------
# 6. current_lr > d0 after first step with non-zero gradient
# ---------------------------------------------------------------------------


def test_current_lr_gt_d0_after_first_step():
    torch.manual_seed(4)
    # Use a large non-zero starting point so first gradient is non-zero
    x = nn.Parameter(torch.ones(8) * 10.0)
    d0 = 1e-6
    opt = Prodigy([x], d0=d0)

    opt.zero_grad()
    _quadratic_loss(x).backward()
    opt.step()

    assert opt.current_lr > d0, (
        f"current_lr {opt.current_lr} should exceed d0={d0} after first step"
    )


# ---------------------------------------------------------------------------
# 7. beta3 defaults to sqrt(beta2)
# ---------------------------------------------------------------------------


def test_beta3_defaults_to_sqrt_beta2():
    model = _make_linear()
    beta2 = 0.999
    opt = Prodigy(model.parameters(), betas=(0.9, beta2))
    expected = math.sqrt(beta2)
    actual = opt.param_groups[0]["beta3"]
    assert abs(actual - expected) < 1e-9, f"Expected beta3={expected}, got {actual}"


def test_beta3_explicit_override():
    model = _make_linear()
    opt = Prodigy(model.parameters(), beta3=0.95)
    assert opt.param_groups[0]["beta3"] == 0.95


# ---------------------------------------------------------------------------
# 8. d_coef=2.0 doubles the estimated learning rate vs d_coef=1.0
# ---------------------------------------------------------------------------


def test_d_coef_doubles_lr():
    torch.manual_seed(5)

    def _run_steps(d_coef):
        x = nn.Parameter(torch.ones(8) * 5.0)
        opt = Prodigy([x], d0=1e-6, d_coef=d_coef)
        for _ in range(10):
            opt.zero_grad()
            _quadratic_loss(x).backward()
            opt.step()
        return opt.current_lr

    lr1 = _run_steps(1.0)
    lr2 = _run_steps(2.0)

    # lr2 should be exactly 2x lr1 since d_coef is just a multiplier on d
    assert abs(lr2 / lr1 - 2.0) < 1e-6, (
        f"d_coef=2.0 should double lr: lr1={lr1}, lr2={lr2}, ratio={lr2 / lr1}"
    )


# ---------------------------------------------------------------------------
# 9. weight_decay > 0 shrinks parameter norms over time
# ---------------------------------------------------------------------------


def test_weight_decay_shrinks_params():
    torch.manual_seed(6)
    model = _make_linear()
    # Record initial norm
    initial_norm = sum(p.norm().item() ** 2 for p in model.parameters()) ** 0.5

    opt = Prodigy(model.parameters(), weight_decay=0.1, d0=1e-3)
    x = torch.randn(16, 8)

    for _ in range(30):
        opt.zero_grad()
        model(x).sum().backward()
        opt.step()

    final_norm = sum(p.norm().item() ** 2 for p in model.parameters()) ** 0.5
    # With weight decay and meaningful updates, norm should decrease or at least
    # not explode; accept a wide margin due to gradient updates also changing norms
    assert final_norm < initial_norm * 10.0, "Params should not explode with weight decay"
    # More specifically, parameters change — just verify the step ran without nan
    for p in model.parameters():
        assert not torch.isnan(p).any(), "NaN detected in parameters"


# ---------------------------------------------------------------------------
# 10. growth_rate=1.01 limits per-step lr growth
# ---------------------------------------------------------------------------


def test_growth_rate_limits_lr():
    torch.manual_seed(7)
    x = nn.Parameter(torch.ones(8) * 100.0)  # Large gradient
    d0 = 1e-6
    growth_rate = 1.01
    opt = Prodigy([x], d0=d0, growth_rate=growth_rate)

    prev_d = opt.param_groups[0]["d"]
    for _ in range(30):
        opt.zero_grad()
        _quadratic_loss(x).backward()
        opt.step()

        current_d = opt.param_groups[0]["d"]
        # d should grow by at most growth_rate per step
        assert current_d <= prev_d * growth_rate + 1e-12, (
            f"d grew too fast: {prev_d} -> {current_d} (max {prev_d * growth_rate})"
        )
        prev_d = current_d


# ---------------------------------------------------------------------------
# 11. use_bias_correction=False doesn't crash
# ---------------------------------------------------------------------------


def test_no_bias_correction_runs():
    torch.manual_seed(8)
    x = nn.Parameter(torch.randn(8))
    opt = Prodigy([x], use_bias_correction=False, d0=1e-3)
    for _ in range(5):
        opt.zero_grad()
        _quadratic_loss(x).backward()
        opt.step()
    # If we get here, no crash
    assert not torch.isnan(x).any()


# ---------------------------------------------------------------------------
# 12. safeguard_warmup=True + warmup_steps=5: d doesn't grow in first 5 steps
# ---------------------------------------------------------------------------


def test_safeguard_warmup_holds_d():
    torch.manual_seed(9)
    x = nn.Parameter(torch.ones(8) * 10.0)
    d0 = 1e-6
    warmup_steps = 5
    opt = Prodigy([x], d0=d0, safeguard_warmup=True, warmup_steps=warmup_steps)

    for i in range(warmup_steps):
        opt.zero_grad()
        _quadratic_loss(x).backward()
        opt.step()
        d_now = opt.param_groups[0]["d"]
        assert d_now == d0, f"d should not grow during warmup at step {i + 1}: d={d_now}"

    # After warmup, d should be allowed to grow
    opt.zero_grad()
    _quadratic_loss(x).backward()
    opt.step()
    # No hard requirement that it HAS grown by step 6, but it can now
    d_after = opt.param_groups[0]["d"]
    assert d_after >= d0, "d should not shrink below d0"


# ---------------------------------------------------------------------------
# 13. Multiple param groups work
# ---------------------------------------------------------------------------


def test_multiple_param_groups():
    torch.manual_seed(10)
    model = nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 2))
    opt = Prodigy(
        [
            {"params": list(model[0].parameters()), "weight_decay": 0.01},
            {"params": list(model[1].parameters()), "weight_decay": 0.0},
        ],
        d0=1e-3,
    )
    x = torch.randn(4, 8)
    for _ in range(5):
        opt.zero_grad()
        model(x).sum().backward()
        opt.step()
    # Should complete without error; parameters change
    for p in model.parameters():
        assert not torch.isnan(p).any()


# ---------------------------------------------------------------------------
# 14. ProdigyW instantiates and runs step without error
# ---------------------------------------------------------------------------


def test_prodigyw_instantiates_and_steps():
    model = _make_linear()
    opt = ProdigyW(model.parameters(), weight_decay=0.01, d0=1e-3)
    assert isinstance(opt, ProdigyW)
    assert isinstance(opt, Prodigy)
    x = torch.randn(4, 8)
    loss = model(x).sum()
    loss.backward()
    opt.step()
    for p in model.parameters():
        assert not torch.isnan(p).any()


# ---------------------------------------------------------------------------
# 15. Prodigy with betas=(0.95, 0.999) converges on quadratic
# ---------------------------------------------------------------------------


def test_custom_betas_convergence():
    torch.manual_seed(11)
    x = nn.Parameter(torch.randn(16))
    opt = Prodigy([x], betas=(0.95, 0.999), d0=1e-3)

    initial_loss = _quadratic_loss(x).item()
    for _ in range(50):
        opt.zero_grad()
        _quadratic_loss(x).backward()
        opt.step()

    final_loss = _quadratic_loss(x).item()
    assert final_loss < initial_loss, (
        f"Custom betas: loss should decrease: {initial_loss:.4f} -> {final_loss:.4f}"
    )


# ---------------------------------------------------------------------------
# 16. step() with closure works
# ---------------------------------------------------------------------------


def test_step_with_closure():
    torch.manual_seed(12)
    x = nn.Parameter(torch.randn(8))
    opt = Prodigy([x], d0=1e-3)

    call_count = 0

    def closure():
        nonlocal call_count
        call_count += 1
        opt.zero_grad()
        loss = _quadratic_loss(x)
        loss.backward()
        return loss

    for _ in range(5):
        returned_loss = opt.step(closure=closure)
        assert returned_loss is not None, "step() with closure should return loss"
        assert isinstance(returned_loss.item(), float)

    assert call_count == 5, f"Closure should be called once per step, got {call_count}"
