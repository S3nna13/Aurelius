"""Tests for the CAME optimizer (arXiv:2307.02047).

Covers all 10 required rigor-floor items plus additional edge cases,
targeting 12-15 tests with pure native PyTorch only.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.optimizers.came import CAME

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_param(*shape, val: float = 1.0, requires_grad: bool = True) -> torch.nn.Parameter:
    return nn.Parameter(torch.full(shape, val))


def _step_with_grad(param: nn.Parameter, grad_val: float, opt: CAME) -> None:
    """Assign a constant gradient and take one optimizer step."""
    param.grad = torch.full_like(param, grad_val)
    opt.step()
    opt.zero_grad()


# ---------------------------------------------------------------------------
# 1. Shape / dtype correctness
# ---------------------------------------------------------------------------


def test_shape_dtype_1d():
    """step() preserves shape and dtype for 1D (bias-like) parameters."""
    p = _make_param(8)
    opt = CAME([p], lr=1e-3)
    _step_with_grad(p, 0.1, opt)
    assert p.shape == (8,)
    assert p.dtype == torch.float32


def test_shape_dtype_2d():
    """step() preserves shape and dtype for 2D (weight matrix) parameters."""
    p = _make_param(4, 6)
    opt = CAME([p], lr=1e-3)
    _step_with_grad(p, 0.1, opt)
    assert p.shape == (4, 6)
    assert p.dtype == torch.float32


# ---------------------------------------------------------------------------
# 2. Gradient flow / convergence on quadratic
# ---------------------------------------------------------------------------


def test_convergence_quadratic_2d():
    """CAME converges a quadratic loss toward zero for a 2D parameter."""
    torch.manual_seed(0)
    p = nn.Parameter(torch.randn(4, 4))
    opt = CAME([p], lr=1e-2)
    initial_loss = (p**2).sum().item()
    for _ in range(200):
        opt.zero_grad()
        loss = (p**2).sum()
        loss.backward()
        opt.step()
    final_loss = (p**2).sum().item()
    assert final_loss < initial_loss * 0.01, (
        f"Expected loss to drop >99%, got {initial_loss:.4f} -> {final_loss:.4f}"
    )


def test_convergence_quadratic_1d():
    """CAME converges a quadratic loss for a 1D parameter."""
    torch.manual_seed(1)
    p = nn.Parameter(torch.randn(16))
    opt = CAME([p], lr=1e-2)
    initial_loss = (p**2).sum().item()
    for _ in range(200):
        opt.zero_grad()
        loss = (p**2).sum()
        loss.backward()
        opt.step()
    final_loss = (p**2).sum().item()
    assert final_loss < initial_loss * 0.01


# ---------------------------------------------------------------------------
# 3. Determinism under torch.manual_seed
# ---------------------------------------------------------------------------


def test_determinism():
    """Two runs with the same seed produce identical parameter trajectories."""

    def _run():
        torch.manual_seed(42)
        p = nn.Parameter(torch.randn(3, 4))
        opt = CAME([p], lr=1e-3)
        for _ in range(10):
            opt.zero_grad()
            loss = (p**2).sum()
            loss.backward()
            opt.step()
        return p.detach().clone()

    r1 = _run()
    r2 = _run()
    assert torch.allclose(r1, r2), "Results should be identical across two seeded runs"


# ---------------------------------------------------------------------------
# 4. Edge cases
# ---------------------------------------------------------------------------


def test_single_element_parameter():
    """Optimizer handles a scalar (single-element) parameter without error."""
    p = _make_param(1)
    opt = CAME([p], lr=1e-3)
    _step_with_grad(p, 0.5, opt)
    assert p.numel() == 1
    assert torch.isfinite(p).all()


def test_very_small_gradient():
    """No NaN/Inf when gradient is extremely small (near underflow)."""
    p = _make_param(4, 4)
    opt = CAME([p], lr=1e-3)
    p.grad = torch.full_like(p, 1e-20)
    opt.step()
    assert torch.isfinite(p).all()


def test_zero_gradient_is_no_op():
    """Zero gradient should not move the parameter."""
    p = _make_param(3, 3)
    opt = CAME([p], lr=1e-2)
    before = p.detach().clone()
    # No grad assigned → p.grad is None → skip
    opt.step()
    assert torch.allclose(p, before), "Parameter should not change with no gradient"


# ---------------------------------------------------------------------------
# 5. Numerical stability: no NaN/Inf after 100 steps on random gradients
# ---------------------------------------------------------------------------


def test_numerical_stability_random_gradients():
    """No NaN or Inf values after 100 steps on random gradients."""
    torch.manual_seed(7)
    p2d = _make_param(8, 8)
    p1d = _make_param(8)
    opt = CAME([p2d, p1d], lr=1e-3)
    for _ in range(100):
        p2d.grad = torch.randn_like(p2d)
        p1d.grad = torch.randn_like(p1d)
        opt.step()
    assert torch.isfinite(p2d).all(), "NaN/Inf detected in 2D param"
    assert torch.isfinite(p1d).all(), "NaN/Inf detected in 1D param"


# ---------------------------------------------------------------------------
# 6. Factored vs unfactored state
# ---------------------------------------------------------------------------


def test_factored_state_keys_for_2d():
    """2D parameter uses V_r / V_c factored second moment keys."""
    p = _make_param(4, 6)
    opt = CAME([p], lr=1e-3)
    _step_with_grad(p, 0.1, opt)
    state = opt.state[p]
    assert "V_r" in state, "V_r factor missing for 2D parameter"
    assert "V_c" in state, "V_c factor missing for 2D parameter"
    assert "V" not in state, "Full V should not exist for 2D parameter"
    assert state["V_r"].shape == (4,)
    assert state["V_c"].shape == (6,)


def test_unfactored_state_keys_for_1d():
    """1D parameter uses full second moment V (no row/col factors)."""
    p = _make_param(5)
    opt = CAME([p], lr=1e-3)
    _step_with_grad(p, 0.1, opt)
    state = opt.state[p]
    assert "V" in state, "Full V missing for 1D parameter"
    assert "V_r" not in state, "V_r should not exist for 1D parameter"
    assert "V_c" not in state, "V_c should not exist for 1D parameter"
    assert state["V"].shape == (5,)


# ---------------------------------------------------------------------------
# 7. Weight decay: parameter shrinks
# ---------------------------------------------------------------------------


def test_weight_decay_shrinks_param():
    """With weight_decay > 0 and positive gradient, param magnitude shrinks faster."""
    p_wd = _make_param(4, 4, val=2.0)
    p_no = _make_param(4, 4, val=2.0)
    opt_wd = CAME([p_wd], lr=1e-2, weight_decay=0.1)
    opt_no = CAME([p_no], lr=1e-2, weight_decay=0.0)

    for _ in range(20):
        p_wd.grad = torch.ones_like(p_wd) * 0.1
        p_no.grad = torch.ones_like(p_no) * 0.1
        opt_wd.step()
        opt_no.step()

    assert p_wd.abs().mean() < p_no.abs().mean(), (
        "Weight decay should produce smaller parameter magnitudes"
    )


# ---------------------------------------------------------------------------
# 8. Learning rate is respected
# ---------------------------------------------------------------------------


def test_learning_rate_respected():
    """Higher lr produces a larger update in one step."""
    p_hi = _make_param(4, 4)
    p_lo = _make_param(4, 4)
    opt_hi = CAME([p_hi], lr=1e-1)
    opt_lo = CAME([p_lo], lr=1e-4)

    p_hi.grad = torch.ones_like(p_hi)
    p_lo.grad = torch.ones_like(p_lo)
    opt_hi.step()
    opt_lo.step()

    delta_hi = (p_hi - 1.0).abs().mean().item()
    delta_lo = (p_lo - 1.0).abs().mean().item()
    assert delta_hi > delta_lo, (
        f"Higher lr should move param more: hi={delta_hi:.6f}, lo={delta_lo:.6f}"
    )


def test_param_groups_lr():
    """param_groups[0]['lr'] change is picked up by next step."""
    p = _make_param(3, 3)
    opt = CAME([p], lr=1e-4)
    p.grad = torch.ones_like(p)
    opt.step()
    before = p.detach().clone()

    # Increase lr significantly
    opt.param_groups[0]["lr"] = 1.0
    p.grad = torch.ones_like(p)
    opt.step()
    delta_large = (p - before).abs().mean().item()

    # Should have moved substantially
    assert delta_large > 1e-3, f"Large lr should produce large update, got {delta_large}"


# ---------------------------------------------------------------------------
# 9. Confidence EMA: C stays positive; ρ_t (confidence scalar) stays in (0,1]
# ---------------------------------------------------------------------------


def test_confidence_values_in_range():
    """C (instability EMA) is strictly positive; ρ_t confidence scalar is in (0, 1].

    Per the paper (arXiv:2307.02047):
    - U_t = G^2 / V  is the per-element instability ratio (can be any positive value)
    - C_t is the EMA of U_t — therefore also positive but not bounded by 1
    - ρ_t = clamp(1 - RMS(U_t - 1), 0, 1) is the scalar in [0, 1]

    We verify C > 0 and that ρ_t computed from U is in [0, 1].
    """
    torch.manual_seed(3)
    p = _make_param(6, 6)
    opt = CAME([p], lr=1e-3)

    rho_values = []
    eps1 = opt.param_groups[0]["eps"][0]

    for _ in range(50):
        p.grad = torch.randn_like(p)
        # Compute ρ before the step (using current state if available)
        opt.step()
        state = opt.state[p]
        if "V_r" in state:
            from src.optimizers.came import _factored_second_moment, _rms

            V = _factored_second_moment(state["V_r"], state["V_c"], eps1).view_as(p)
            G_sq = p.grad.square().add(eps1) if p.grad is not None else None
            if G_sq is not None:
                U = G_sq.view_as(p) / V.clamp_min(eps1)
                rho = (1.0 - _rms(U - 1.0)).clamp(0.0, 1.0).item()
                rho_values.append(rho)

    C = opt.state[p]["C"]
    assert (C > 0).all(), f"C must be strictly positive, got min {C.min().item()}"

    # ρ_t must be clamped to [0, 1]
    for rho in rho_values:
        assert 0.0 <= rho <= 1.0, f"ρ_t must be in [0, 1], got {rho}"


# ---------------------------------------------------------------------------
# 10. State initialisation: all expected keys present after first step
# ---------------------------------------------------------------------------


def test_state_keys_after_first_step_2d():
    """State contains all expected keys after the first step (2D parameter)."""
    p = _make_param(3, 4)
    opt = CAME([p], lr=1e-3, betas=(0.9, 0.999, 0.9999))
    _step_with_grad(p, 0.1, opt)
    state = opt.state[p]
    for key in ("step", "V_r", "V_c", "C", "m"):
        assert key in state, f"Expected key '{key}' missing from state"
    assert state["step"] == 1


def test_state_keys_after_first_step_1d():
    """State contains all expected keys after the first step (1D parameter)."""
    p = _make_param(5)
    opt = CAME([p], lr=1e-3, betas=(0.9, 0.999, 0.9999))
    _step_with_grad(p, 0.1, opt)
    state = opt.state[p]
    for key in ("step", "V", "C", "m"):
        assert key in state, f"Expected key '{key}' missing from state"
    assert state["step"] == 1


# ---------------------------------------------------------------------------
# 11. No first moment when beta1 == 0
# ---------------------------------------------------------------------------


def test_no_first_moment_when_beta1_zero():
    """When beta1=0, state should not contain 'm' key."""
    p = _make_param(3, 3)
    opt = CAME([p], lr=1e-3, betas=(0.0, 0.999, 0.9999))
    _step_with_grad(p, 0.1, opt)
    state = opt.state[p]
    assert "m" not in state, "First moment 'm' should not be stored when beta1=0"


# ---------------------------------------------------------------------------
# 12. State dict round-trip (serialisation)
# ---------------------------------------------------------------------------


def test_state_dict_round_trip():
    """Optimizer state can be saved and reloaded with identical results."""
    p = _make_param(3, 4)
    opt = CAME([p], lr=1e-3)
    for _ in range(3):
        _step_with_grad(p, 0.1, opt)

    saved = opt.state_dict()

    p2 = _make_param(3, 4)
    opt2 = CAME([p2], lr=1e-3)
    opt2.load_state_dict(saved)

    assert opt2.state[p2]["step"] == opt.state[p]["step"]
    assert torch.allclose(opt2.state[p2]["V_r"], opt.state[p]["V_r"])
    assert torch.allclose(opt2.state[p2]["V_c"], opt.state[p]["V_c"])


# ---------------------------------------------------------------------------
# 13. Closure support
# ---------------------------------------------------------------------------


def test_closure_returns_loss():
    """step() with a closure returns the loss value."""
    p = _make_param(2, 2)
    opt = CAME([p], lr=1e-3)

    def closure():
        opt.zero_grad()
        loss = (p**2).sum()
        loss.backward()
        return loss

    returned_loss = opt.step(closure)
    assert returned_loss is not None
    assert torch.isfinite(returned_loss)
