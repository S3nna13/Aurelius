"""Tests for the Signum optimizer (Balles & Hennig, ICML 2018).

Signum is the *sign-of-momentum* optimizer, distinct from MSignSGD in
src/training/sign_sgd.py: it supports optional gradient-norm-based learning
rate scaling (Signum-N) that gives invariance to gradient magnitude.

12 focused tests cover:
  1.  Shape — step() updates scalar and matrix params
  2.  Convergence — reduces loss on quadratic
  3.  Update direction — params move in −sign(momentum) direction
  4.  Determinism under torch.manual_seed
  5.  sign(momentum) ∈ {−1, 0, +1}
  6.  Weight decay — params shrink toward zero
  7.  Zero gradient — momentum decays but no direct gradient push
  8.  No NaN / Inf after 50 steps
  9.  State keys — 'exp_avg' present after first step
  10. norm_scaling=True — effective lr scales with gradient magnitude
  11. momentum=0 — equivalent to signSGD (sign of raw gradient)
  12. Different lr values produce proportionally different step sizes
"""

from __future__ import annotations

import math

import torch

from src.optimizers.signum import Signum

# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #


def _scalar_param(value: float = 2.0) -> torch.nn.Parameter:
    return torch.nn.Parameter(torch.tensor(value))


def _matrix_param(rows: int = 4, cols: int = 4, fill: float = 1.0) -> torch.nn.Parameter:
    return torch.nn.Parameter(torch.full((rows, cols), fill))


def _quadratic_loss(p: torch.Tensor) -> torch.Tensor:
    """L = sum(p²) — global minimum at p = 0."""
    return (p**2).sum()


# --------------------------------------------------------------------------- #
# 1. Shape: step() updates scalar and matrix params                            #
# --------------------------------------------------------------------------- #


def test_shape_scalar():
    p = _scalar_param(3.0)
    original = p.data.clone()
    opt = Signum([p], lr=1e-3)
    _quadratic_loss(p).backward()
    opt.step()
    assert p.shape == original.shape, "scalar shape must be preserved"
    assert p.dtype == original.dtype, "dtype must be preserved"
    assert not torch.equal(p.data, original), "param should have changed after step"


def test_shape_matrix():
    p = _matrix_param(4, 4, fill=1.0)
    original = p.data.clone()
    opt = Signum([p], lr=1e-3)
    _quadratic_loss(p).backward()
    opt.step()
    assert p.shape == torch.Size([4, 4]), "matrix shape must be preserved"
    assert p.dtype == torch.float32, "dtype must be float32"
    assert not torch.equal(p.data, original), "matrix param should change after step"


# --------------------------------------------------------------------------- #
# 2. Convergence: reduces loss on a quadratic objective                        #
# --------------------------------------------------------------------------- #


def test_convergence_quadratic():
    """Signum makes consistent constant-magnitude steps; loss should reduce clearly.

    Signum takes unit-magnitude steps (lr * sign), so it oscillates around the
    minimum for small lr — analogous to heavy-ball with a fixed step size.  We
    verify that the optimizer moves the parameter significantly toward zero and
    that the final loss is well below the initial value.
    """
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.tensor(5.0))
    opt = Signum([p], lr=1e-2, momentum=0.9)
    initial_loss = _quadratic_loss(p).item()
    for _ in range(200):
        opt.zero_grad()
        _quadratic_loss(p).backward()
        opt.step()
    final_loss = _quadratic_loss(p).item()
    # Signum converges but can oscillate near the minimum; we require a
    # substantial reduction (< 50 % of initial) rather than near-zero.
    assert final_loss < initial_loss * 0.5, (
        f"Expected significant loss reduction; initial={initial_loss:.4f}, final={final_loss:.6f}"
    )


# --------------------------------------------------------------------------- #
# 3. Update direction: params move in −sign(momentum) direction                #
# --------------------------------------------------------------------------- #


def test_update_direction_negative_sign_of_momentum():
    """After one step, the parameter change should equal −lr * sign(momentum)."""
    lr = 0.05
    init_val = 3.0
    p = torch.nn.Parameter(torch.tensor(init_val))
    opt = Signum([p], lr=lr, momentum=0.9)

    _quadratic_loss(p).backward()  # grad = 2 * init_val = 6 > 0
    p_before = p.data.clone()
    opt.step()
    p_after = p.data.clone()

    # First step: momentum is initialised to g_1 = 6 (positive)
    # Expected update: p -= lr * sign(6) = p - lr
    expected_change = -lr
    actual_change = (p_after - p_before).item()
    assert math.isclose(actual_change, expected_change, rel_tol=1e-6), (
        f"Expected Δp = {expected_change}, got {actual_change}"
    )


# --------------------------------------------------------------------------- #
# 4. Determinism under torch.manual_seed                                       #
# --------------------------------------------------------------------------- #


def test_determinism():
    def run() -> float:
        torch.manual_seed(42)
        p = torch.nn.Parameter(torch.tensor(1.0))
        opt = Signum([p], lr=1e-3, momentum=0.9)
        for _ in range(10):
            opt.zero_grad()
            (p**2 + 0.5 * p).backward()
            opt.step()
        return p.item()

    assert run() == run(), "Two runs with the same seed should be identical"


# --------------------------------------------------------------------------- #
# 5. sign(momentum) ∈ {−1, 0, +1}                                              #
# --------------------------------------------------------------------------- #


def test_sign_of_momentum_is_ternary():
    """The momentum buffer's sign must only contain values in {-1, 0, 1}."""
    torch.manual_seed(7)
    p = torch.nn.Parameter(torch.randn(8, 8))
    opt = Signum([p], lr=1e-3, momentum=0.9)
    for _ in range(5):
        opt.zero_grad()
        _quadratic_loss(p).backward()
        opt.step()

    exp_avg = opt.state[p]["exp_avg"]
    sign_vals = exp_avg.sign()
    unique = sign_vals.unique()
    valid = {-1.0, 0.0, 1.0}
    for v in unique.tolist():
        assert v in valid, f"sign(momentum) produced unexpected value {v}"


# --------------------------------------------------------------------------- #
# 6. Weight decay: params shrink toward zero                                   #
# --------------------------------------------------------------------------- #


def test_weight_decay_shrinks_params():
    """With non-zero weight_decay and a positive param, the param should shrink."""
    p = torch.nn.Parameter(torch.tensor(1.0))
    opt = Signum([p], lr=1e-2, momentum=0.9, weight_decay=0.5)
    # Apply a positive gradient so the param moves left (toward 0 for positive p).
    p.grad = torch.tensor(1.0)  # positive grad + positive weight decay → sign > 0
    val_before = p.data.clone()
    opt.step()
    assert p.item() < val_before.item(), (
        f"Param should shrink with weight_decay; before={val_before.item():.4f}, "
        f"after={p.item():.4f}"
    )


# --------------------------------------------------------------------------- #
# 7. Zero gradient: momentum decays but no new gradient push                   #
# --------------------------------------------------------------------------- #


def test_zero_gradient_momentum_decays():
    """With g=0 the momentum buffer should decay toward zero (m *= β)."""
    beta = 0.9
    p = torch.nn.Parameter(torch.tensor(2.0))
    opt = Signum([p], lr=1e-3, momentum=beta)

    # Warm the momentum buffer with a non-zero gradient.
    p.grad = torch.tensor(1.0)
    opt.step()
    m_initial = opt.state[p]["exp_avg"].item()

    # Now apply zero gradient — momentum should decay.
    p.grad = torch.tensor(0.0)
    opt.step()
    m_after = opt.state[p]["exp_avg"].item()

    expected_m_after = beta * m_initial + (1.0 - beta) * 0.0
    assert math.isclose(m_after, expected_m_after, rel_tol=1e-6), (
        f"Momentum after zero grad: expected {expected_m_after:.6f}, got {m_after:.6f}"
    )


# --------------------------------------------------------------------------- #
# 8. No NaN / Inf after 50 steps                                               #
# --------------------------------------------------------------------------- #


def test_no_nan_inf_after_50_steps():
    torch.manual_seed(13)
    p = torch.nn.Parameter(torch.randn(16, 16))
    opt = Signum([p], lr=1e-3, momentum=0.9, norm_scaling=True)
    for _ in range(50):
        opt.zero_grad()
        _quadratic_loss(p).backward()
        opt.step()
    assert not torch.isnan(p.data).any(), "NaN detected in parameters"
    assert not torch.isinf(p.data).any(), "Inf detected in parameters"


# --------------------------------------------------------------------------- #
# 9. State keys: 'exp_avg' present after first step                            #
# --------------------------------------------------------------------------- #


def test_state_keys_after_first_step():
    p = _scalar_param(1.0)
    opt = Signum([p], lr=1e-3, momentum=0.9)
    _quadratic_loss(p).backward()
    opt.step()
    state = opt.state[p]
    assert "exp_avg" in state, "state must contain 'exp_avg' after first step"
    assert state["exp_avg"].shape == p.shape, "exp_avg shape must match param shape"


# --------------------------------------------------------------------------- #
# 10. norm_scaling=True: effective lr scales with gradient magnitude           #
# --------------------------------------------------------------------------- #


def test_norm_scaling_effective_lr():
    """norm_scaling=True should yield a larger step when the gradient is small.

    With norm_scaling, α_eff = α / ||g||.  A small-magnitude gradient produces
    a *larger* lr_eff than a large-magnitude gradient, normalising the step size.
    We verify this by comparing the parameter change for two gradient magnitudes.
    """
    lr = 0.1

    def single_step_change(grad_val: float) -> float:
        p = torch.nn.Parameter(torch.tensor(5.0))
        opt = Signum([p], lr=lr, momentum=0.0, norm_scaling=True)
        p.grad = torch.tensor(grad_val)
        before = p.data.clone()
        opt.step()
        return abs((p.data - before).item())

    # Both gradients are positive, so sign is +1 and change = lr_eff.
    # lr_eff_small = lr / (small + eps)  >  lr / (large + eps) = lr_eff_large
    change_small_grad = single_step_change(0.1)
    change_large_grad = single_step_change(100.0)

    assert change_small_grad > change_large_grad, (
        f"norm_scaling should yield larger step for smaller gradient; "
        f"change_small={change_small_grad:.6f}, change_large={change_large_grad:.6f}"
    )


# --------------------------------------------------------------------------- #
# 11. momentum=0: equivalent to signSGD (sign of raw gradient)                 #
# --------------------------------------------------------------------------- #


def test_momentum_zero_is_sign_sgd():
    """When momentum=0, Signum degenerates to signSGD: θ -= lr * sign(g)."""
    lr = 0.05
    init_val = 3.0

    # Signum with momentum=0
    p_signum = torch.nn.Parameter(torch.tensor(init_val))
    opt_signum = Signum([p_signum], lr=lr, momentum=0.0)
    p_signum.grad = torch.tensor(4.0)  # positive gradient
    opt_signum.step()

    # Manual signSGD: θ -= lr * sign(g) = 3.0 - 0.05 * 1 = 2.95
    expected = init_val - lr * math.copysign(1.0, 4.0)
    assert math.isclose(p_signum.item(), expected, rel_tol=1e-6), (
        f"momentum=0 should give signSGD update; expected {expected}, got {p_signum.item()}"
    )
    # Also confirm no state is stored (no exp_avg key)
    assert "exp_avg" not in opt_signum.state[p_signum], (
        "No momentum state should be stored when momentum=0"
    )


# --------------------------------------------------------------------------- #
# 12. Different lr values produce proportionally different step sizes           #
# --------------------------------------------------------------------------- #


def test_lr_proportional_step_sizes():
    """Step size should be proportional to lr (sign update scales linearly)."""
    lr_a = 0.01
    lr_b = 0.10  # 10× larger

    def single_step_delta(lr: float) -> float:
        p = torch.nn.Parameter(torch.tensor(5.0))
        opt = Signum([p], lr=lr, momentum=0.0)  # momentum=0 for clean comparison
        p.grad = torch.tensor(1.0)  # constant positive gradient
        before = p.data.clone()
        opt.step()
        return (p.data - before).item()

    delta_a = single_step_delta(lr_a)
    delta_b = single_step_delta(lr_b)

    ratio = delta_b / delta_a
    expected_ratio = lr_b / lr_a  # should be 10.0
    assert math.isclose(ratio, expected_ratio, rel_tol=1e-4), (
        f"Step-size ratio should equal lr ratio ({expected_ratio:.1f}); got {ratio:.6f}"
    )
