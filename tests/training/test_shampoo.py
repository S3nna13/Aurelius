"""Tests for src/training/shampoo.py — ShampooOptimizer.

Covers the 14 cases specified in the implementation brief.
Pure PyTorch only; no scipy/sklearn/external ML libs.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from src.training.shampoo import ShampooOptimizer


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_2d_param(m: int = 4, n: int = 3, seed: int = 0) -> torch.nn.Parameter:
    torch.manual_seed(seed)
    return nn.Parameter(torch.randn(m, n))


def _make_1d_param(size: int = 5, seed: int = 1) -> torch.nn.Parameter:
    torch.manual_seed(seed)
    return nn.Parameter(torch.randn(size))


def _single_step(param, lr=0.01, **kwargs):
    """One forward-backward-step cycle; returns (opt, loss_val)."""
    opt = ShampooOptimizer([param], lr=lr, **kwargs)
    loss = (param ** 2).sum()
    loss.backward()
    opt.step()
    return opt, loss.item()


# ---------------------------------------------------------------------------
# Test 1: Parameters move after one step
# ---------------------------------------------------------------------------

def test_params_move_after_one_step():
    p = _make_2d_param()
    p_before = p.data.clone()
    _single_step(p)
    assert not torch.allclose(p.data, p_before), "Parameter should have changed."


# ---------------------------------------------------------------------------
# Test 2: Movement is finite — no NaN or Inf
# ---------------------------------------------------------------------------

def test_movement_finite():
    p = _make_2d_param()
    _single_step(p)
    assert torch.isfinite(p.data).all(), "Parameter contains NaN or Inf after step."


# ---------------------------------------------------------------------------
# Test 3: Determinism under fixed seed
# ---------------------------------------------------------------------------

def test_determinism():
    def run():
        torch.manual_seed(42)
        p = _make_2d_param(seed=42)
        _single_step(p, lr=0.01)
        return p.data.clone()

    result_a = run()
    result_b = run()
    assert torch.allclose(result_a, result_b), "Results are not deterministic."


# ---------------------------------------------------------------------------
# Test 4: L_t and R_t statistics accumulate (non-zero) after first step
# ---------------------------------------------------------------------------

def test_statistics_accumulate():
    p = _make_2d_param()
    opt = ShampooOptimizer([p], lr=0.01, update_freq=100)
    loss = (p ** 2).sum()
    loss.backward()
    opt.step()
    state = opt.state[p]
    assert state["L_t"].abs().sum().item() > 0, "L_t should be non-zero."
    assert state["R_t"].abs().sum().item() > 0, "R_t should be non-zero."


# ---------------------------------------------------------------------------
# Test 5: L_t has shape (m, m) for param shape (m, n)
# ---------------------------------------------------------------------------

def test_L_shape():
    m, n = 6, 4
    p = _make_2d_param(m, n)
    opt = ShampooOptimizer([p], lr=0.01)
    (p ** 2).sum().backward()
    opt.step()
    assert opt.state[p]["L_t"].shape == (m, m), f"Expected L_t shape ({m},{m})."


# ---------------------------------------------------------------------------
# Test 6: R_t has shape (n, n) for param shape (m, n)
# ---------------------------------------------------------------------------

def test_R_shape():
    m, n = 6, 4
    p = _make_2d_param(m, n)
    opt = ShampooOptimizer([p], lr=0.01)
    (p ** 2).sum().backward()
    opt.step()
    assert opt.state[p]["R_t"].shape == (n, n), f"Expected R_t shape ({n},{n})."


# ---------------------------------------------------------------------------
# Test 7: Preconditioners recomputed at update_freq=5, not every step
# ---------------------------------------------------------------------------

def test_preconditioner_recomputed_at_update_freq():
    """L_inv4 stays at its initial value (eye) for steps 1..update_freq-1,
    then changes on step == update_freq (when t % update_freq == 0)."""
    p = _make_2d_param()
    update_freq = 5
    opt = ShampooOptimizer([p], lr=0.01, update_freq=update_freq)

    snapshots = []
    for _ in range(update_freq + 1):
        if p.grad is not None:
            p.grad.zero_()
        (p ** 2).sum().backward()
        opt.step()
        snapshots.append(opt.state[p]["L_inv4"].clone())

    # snapshots indices: 0=after step1, 1=after step2, ..., 4=after step5, 5=after step6
    # steps 1..4 (indices 0..3) should all share the initial identity preconditioner
    unchanged = all(
        torch.allclose(snapshots[i], snapshots[0]) for i in range(1, update_freq - 1)
    )
    # step 5 (index 4) triggers recomputation; should differ from step 4 (index 3)
    changed_at_freq = not torch.allclose(snapshots[update_freq - 2], snapshots[update_freq - 1])

    assert unchanged, "L_inv4 should not change for steps 1 through update_freq-1."
    assert changed_at_freq, "L_inv4 should change exactly at step == update_freq."


# ---------------------------------------------------------------------------
# Test 8: 1-D parameters use standard SGD (no Kronecker factors in state)
# ---------------------------------------------------------------------------

def test_1d_param_uses_sgd_not_kronecker():
    p = _make_1d_param()
    opt = ShampooOptimizer([p], lr=0.1)
    (p ** 2).sum().backward()
    opt.step()
    state = opt.state[p]
    assert "L_t" not in state, "1-D param should have no L_t."
    assert "R_t" not in state, "1-D param should have no R_t."
    assert "momentum_buf" in state, "1-D param should use momentum_buf."
    assert torch.isfinite(p.data).all()


# ---------------------------------------------------------------------------
# Test 9: Converges on quadratic loss over 20 steps
# ---------------------------------------------------------------------------

def test_converges_quadratic():
    torch.manual_seed(7)
    p = nn.Parameter(torch.randn(4, 4) * 2.0)
    opt = ShampooOptimizer([p], lr=0.05, update_freq=5)
    losses = []
    for _ in range(20):
        if p.grad is not None:
            p.grad.zero_()
        loss = (p ** 2).sum()
        losses.append(loss.item())
        loss.backward()
        opt.step()
    assert losses[-1] < losses[0], "Loss should decrease over 20 steps on quadratic."


# ---------------------------------------------------------------------------
# Test 10: Non-zero weight_decay modifies the update vs zero weight_decay
# ---------------------------------------------------------------------------

def test_weight_decay_modifies_update():
    def run_with_wd(wd):
        torch.manual_seed(0)
        p = _make_2d_param(seed=0)
        opt = ShampooOptimizer([p], lr=0.01, weight_decay=wd)
        (p ** 2).sum().backward()
        opt.step()
        return p.data.clone()

    p_no_wd = run_with_wd(0.0)
    p_wd    = run_with_wd(1e-2)
    assert not torch.allclose(p_no_wd, p_wd), (
        "weight_decay != 0 should produce a different update."
    )


# ---------------------------------------------------------------------------
# Test 11: ε regularisation keeps L + εI invertible for near-singular L
# ---------------------------------------------------------------------------

def test_epsilon_regularisation_near_singular():
    """Force a near-singular gradient (rank-1) and ensure no NaN/Inf."""
    torch.manual_seed(3)
    m, n = 5, 5
    p = nn.Parameter(torch.randn(m, n))
    # Rank-1 gradient
    v = torch.randn(m, 1)
    u = torch.randn(1, n)
    p.grad = (v @ u).expand_as(p).clone()

    opt = ShampooOptimizer([p], lr=0.01, epsilon=1e-6, update_freq=1)
    opt.step()
    assert torch.isfinite(p.data).all(), "NaN/Inf detected with near-singular L."


# ---------------------------------------------------------------------------
# Test 12: No NaN/Inf on large gradients (scale 100×)
# ---------------------------------------------------------------------------

def test_no_nan_inf_large_gradients():
    torch.manual_seed(5)
    p = nn.Parameter(torch.randn(4, 4))
    # Artificially large gradient
    p.grad = torch.randn(4, 4) * 100.0
    opt = ShampooOptimizer([p], lr=0.01, update_freq=1)
    opt.step()
    assert torch.isfinite(p.data).all(), "NaN/Inf with large (100×) gradients."


# ---------------------------------------------------------------------------
# Test 13: Preconditioned update has smaller or comparable norm than raw grad
# ---------------------------------------------------------------------------

def test_preconditioned_update_norm_reasonable():
    """After preconditioners are warmed up, the update norm should be finite
    and not astronomically larger than the raw gradient norm.  Shampoo is a
    second-order method so the ratio can be < 1 (better conditioning).
    """
    torch.manual_seed(9)
    p = nn.Parameter(torch.randn(6, 6))
    opt = ShampooOptimizer([p], lr=1.0, update_freq=1)  # lr=1 to measure raw scale

    # Warm up statistics for several steps
    for _ in range(5):
        if p.grad is not None:
            p.grad.zero_()
        (p ** 2).sum().backward()
        opt.step()

    # Now measure
    p_before = p.data.clone()
    if p.grad is not None:
        p.grad.zero_()
    (p ** 2).sum().backward()
    G_norm = p.grad.norm().item()

    p_copy = nn.Parameter(p.data.clone())
    opt2 = ShampooOptimizer([p_copy], lr=1.0, update_freq=1)
    # Copy over accumulated statistics so it's apples-to-apples
    for key in ("L_t", "R_t", "L_inv4", "R_inv4", "step"):
        opt2.state[p_copy][key] = opt.state[p][key].clone() if isinstance(
            opt.state[p][key], torch.Tensor) else opt.state[p][key]
    p_copy.grad = p.grad.clone()
    opt2.step()

    update_norm = (p_before - p_copy.data).norm().item()
    assert math.isfinite(update_norm), "Preconditioned update norm is not finite."
    # Shampoo should not blow up: update_norm < 1000 * G_norm
    assert update_norm < 1000 * G_norm + 1e-8, (
        f"Preconditioned update norm {update_norm:.4f} is unreasonably large "
        f"vs raw gradient norm {G_norm:.4f}."
    )


# ---------------------------------------------------------------------------
# Test 14: update_freq=1 — preconditioners updated every step
# ---------------------------------------------------------------------------

def test_update_freq_1_updates_every_step():
    """With update_freq=1, L_inv4 should differ after every step (as L_t grows)."""
    torch.manual_seed(11)
    p = nn.Parameter(torch.randn(4, 4))
    opt = ShampooOptimizer([p], lr=0.01, update_freq=1)

    snapshots = []
    for _ in range(3):
        if p.grad is not None:
            p.grad.zero_()
        (p ** 2).sum().backward()
        opt.step()
        snapshots.append(opt.state[p]["L_inv4"].clone())

    # Each successive snapshot should differ because L_t is strictly growing
    assert not torch.allclose(snapshots[0], snapshots[1]), (
        "L_inv4 should change at step 2 with update_freq=1."
    )
    assert not torch.allclose(snapshots[1], snapshots[2]), (
        "L_inv4 should change at step 3 with update_freq=1."
    )
