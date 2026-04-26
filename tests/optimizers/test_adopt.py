"""Tests for the ADOPT optimizer (arXiv:2411.02853)."""

from __future__ import annotations

import math

import torch

from src.optimizers.adopt import ADOPT

# ------------------------------------------------------------------------------- #
# Helpers                                                                          #
# ------------------------------------------------------------------------------- #


def make_scalar_param(value: float = 2.0) -> torch.nn.Parameter:
    return torch.nn.Parameter(torch.tensor(value))


def make_matrix_param(rows: int = 4, cols: int = 4) -> torch.nn.Parameter:
    return torch.nn.Parameter(torch.ones(rows, cols))


def quadratic_loss(p: torch.Tensor) -> torch.Tensor:
    """L = sum(p^2) — global minimum at p = 0."""
    return (p**2).sum()


# ------------------------------------------------------------------------------- #
# 1. Shape / dtype: step() updates scalar and matrix parameters correctly          #
# ------------------------------------------------------------------------------- #


def test_shape_and_dtype_scalar():
    p = make_scalar_param(3.0)
    opt = ADOPT([p], lr=1e-2)
    original = p.data.clone()
    loss = quadratic_loss(p)
    loss.backward()
    opt.step()
    assert p.shape == original.shape
    assert p.dtype == original.dtype
    assert not torch.equal(p.data, original), "parameter should have changed after step"


def test_shape_and_dtype_matrix():
    p = make_matrix_param(4, 4)
    opt = ADOPT([p], lr=1e-2)
    original = p.data.clone()
    loss = quadratic_loss(p)
    loss.backward()
    opt.step()
    assert p.shape == torch.Size([4, 4])
    assert p.dtype == torch.float32
    assert not torch.equal(p.data, original), "matrix param should have changed after step"


# ------------------------------------------------------------------------------- #
# 2. Gradient flow: converges on quadratic loss                                    #
# ------------------------------------------------------------------------------- #


def test_convergence_quadratic():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.tensor(5.0))
    opt = ADOPT([p], lr=1e-2, betas=(0.9, 0.9999))
    for _ in range(300):
        opt.zero_grad()
        quadratic_loss(p).backward()
        opt.step()
    assert p.item() < 0.5, f"Expected convergence near 0, got {p.item()}"


# ------------------------------------------------------------------------------- #
# 3. Determinism under torch.manual_seed                                           #
# ------------------------------------------------------------------------------- #


def test_determinism():
    def run():
        torch.manual_seed(42)
        p = torch.nn.Parameter(torch.tensor(1.0))
        opt = ADOPT([p], lr=1e-3)
        for _ in range(10):
            opt.zero_grad()
            (p**2 + 0.5 * p).backward()
            opt.step()
        return p.item()

    assert run() == run(), "Two runs with the same seed should yield identical results"


# ------------------------------------------------------------------------------- #
# 4. First step: with no v_{t-1}, falls back to pure gradient step                 #
# ------------------------------------------------------------------------------- #


def test_first_step_is_gradient_step():
    """At t=1 ADOPT should do θ_1 = θ_0 - α*g, with no second-moment division."""
    lr = 0.1
    init_val = 3.0
    p = torch.nn.Parameter(torch.tensor(init_val))
    opt = ADOPT([p], lr=lr, betas=(0.9, 0.9999))
    loss = quadratic_loss(p)
    loss.backward()
    grad_val = p.grad.item()  # 2 * init_val = 6.0
    opt.step()
    expected = init_val - lr * grad_val
    assert math.isclose(p.item(), expected, rel_tol=1e-6), (
        f"First step: expected {expected}, got {p.item()}"
    )


# ------------------------------------------------------------------------------- #
# 5. Edge case: zero gradient leaves params unchanged                              #
# ------------------------------------------------------------------------------- #


def test_zero_gradient_no_change():
    p = torch.nn.Parameter(torch.tensor(5.0))
    opt = ADOPT([p], lr=1e-2)
    p.grad = torch.zeros_like(p)
    before = p.data.clone()
    opt.step()
    assert torch.equal(p.data, before), "Zero gradient should not change the parameter"


# ------------------------------------------------------------------------------- #
# 6. Numerical stability: no NaN/Inf after 100 steps on random + large gradients  #
# ------------------------------------------------------------------------------- #


def test_numerical_stability_random_gradients():
    torch.manual_seed(7)
    p = torch.nn.Parameter(torch.randn(16, 16))
    opt = ADOPT([p], lr=1e-3)
    for _ in range(100):
        p.grad = torch.randn_like(p) * 1e3  # large gradients
        opt.step()
    assert not torch.isnan(p.data).any(), "NaN detected in parameters"
    assert not torch.isinf(p.data).any(), "Inf detected in parameters"


def test_numerical_stability_very_large_gradients():
    torch.manual_seed(13)
    p = torch.nn.Parameter(torch.ones(8))
    opt = ADOPT([p], lr=1e-4, eps=1e-6)
    for _ in range(100):
        p.grad = torch.full_like(p, 1e6)
        opt.step()
    assert not torch.isnan(p.data).any()
    assert not torch.isinf(p.data).any()


# ------------------------------------------------------------------------------- #
# 7. Weight decay: params shrink toward zero when weight_decay > 0                 #
# ------------------------------------------------------------------------------- #


def test_weight_decay_shrinks_params():
    """With a zero gradient and high weight_decay, the parameter should shrink."""
    p = torch.nn.Parameter(torch.tensor(1.0))
    opt = ADOPT([p], lr=1e-2, weight_decay=0.1, decoupled=False)
    # Step 1 is a pure gradient step; from step 2 onward wd takes effect.
    # Feed a tiny non-zero grad so v is initialised.
    p.grad = torch.tensor(1e-8)
    opt.step()  # t=1
    p.grad = torch.tensor(0.0)
    opt.step()  # t=2, coupled wd adds p to gradient
    val_after_wd = p.item()
    # Parameter should have been pulled toward 0 (still positive but smaller than 1)
    assert val_after_wd < 1.0, f"Expected param < 1.0, got {val_after_wd}"


# ------------------------------------------------------------------------------- #
# 8. Decoupled weight decay: decoupled=True vs False produces different updates    #
# ------------------------------------------------------------------------------- #


def test_decoupled_vs_coupled_weight_decay_differ():
    torch.manual_seed(0)
    init = torch.randn(4)

    def run(decoupled: bool) -> float:
        p = torch.nn.Parameter(init.clone())
        opt = ADOPT([p], lr=1e-2, weight_decay=0.1, decoupled=decoupled)
        for _ in range(5):
            opt.zero_grad()
            quadratic_loss(p).backward()
            opt.step()
        return p.data.sum().item()

    val_coupled = run(False)
    val_decoupled = run(True)
    assert val_coupled != val_decoupled, (
        "Coupled and decoupled weight decay should produce different parameter values"
    )


# ------------------------------------------------------------------------------- #
# 9. State keys: 'step', 'exp_avg', 'exp_avg_sq' present after first step         #
# ------------------------------------------------------------------------------- #


def test_state_keys_after_first_step():
    p = make_scalar_param(1.0)
    opt = ADOPT([p], lr=1e-3)
    loss = quadratic_loss(p)
    loss.backward()
    opt.step()
    state = opt.state[p]
    assert "step" in state, "state must contain 'step'"
    assert "exp_avg" in state, "state must contain 'exp_avg'"
    assert "exp_avg_sq" in state, "state must contain 'exp_avg_sq'"
    assert state["step"] == 1


# ------------------------------------------------------------------------------- #
# 10. Gradient normalisation: g̃_t ≠ g_t after step 1                             #
# ------------------------------------------------------------------------------- #


def test_gradient_normalisation_happens():
    """After step 1 (v is initialised), the normalised gradient g̃_t != g_t.

    We verify indirectly: the first-moment (exp_avg) after step 2 should differ
    from what a plain Adam first moment would be if no normalisation occurred.
    """
    torch.manual_seed(5)
    p_adopt = torch.nn.Parameter(torch.tensor(1.0))
    opt = ADOPT([p_adopt], lr=1e-3, betas=(0.9, 0.9999))

    grad_seq = [torch.tensor(2.0), torch.tensor(3.0)]

    # Step 1
    p_adopt.grad = grad_seq[0].clone()
    opt.step()

    # Step 2
    p_adopt.grad = grad_seq[1].clone()
    opt.step()

    adopt_m = opt.state[p_adopt]["exp_avg"].item()

    # What would a plain (non-normalised) first moment be?
    # m_1 = (1-0.9)*2 = 0.2   (t=1 is a gradient step, m stays 0)
    # After t=2 without normalisation: m_2 = 0.9*0 + 0.1*3 = 0.3
    plain_m2 = (1.0 - 0.9) * grad_seq[1].item()  # 0.3

    # With normalisation g̃_2 = 3 / sqrt(v_1) where v_1=(1-0.9999)*4 ≈ 4e-4
    # so g̃_2 ≈ 150 >> 3; adopt_m should be very different from 0.3
    assert not math.isclose(adopt_m, plain_m2, rel_tol=1e-3), (
        f"Expected normalised first moment to differ from plain Adam first moment. "
        f"adopt_m={adopt_m}, plain_m2={plain_m2}"
    )


# ------------------------------------------------------------------------------- #
# 11. β2 stability: high β2 (0.9999) doesn't cause divergence                     #
# ------------------------------------------------------------------------------- #


def test_high_beta2_no_divergence():
    torch.manual_seed(3)
    p = torch.nn.Parameter(torch.tensor(1.0))
    opt = ADOPT([p], lr=1e-3, betas=(0.9, 0.9999))
    for _ in range(200):
        opt.zero_grad()
        quadratic_loss(p).backward()
        opt.step()
    assert not math.isnan(p.item()), "NaN with high β2"
    assert not math.isinf(p.item()), "Inf with high β2"
    assert abs(p.item()) < 5.0, f"Divergence detected: p={p.item()}"


# ------------------------------------------------------------------------------- #
# 12. Comparison baseline: ADOPT reaches lower loss than SGD after 200 steps       #
# ------------------------------------------------------------------------------- #


def test_adopt_outperforms_sgd_on_quadratic():
    torch.manual_seed(99)
    init = torch.tensor(5.0)

    # ADOPT
    p_adopt = torch.nn.Parameter(init.clone())
    opt_adopt = ADOPT([p_adopt], lr=1e-2)
    for _ in range(200):
        opt_adopt.zero_grad()
        quadratic_loss(p_adopt).backward()
        opt_adopt.step()
    loss_adopt = quadratic_loss(p_adopt).item()

    # SGD (same lr)
    p_sgd = torch.nn.Parameter(init.clone())
    opt_sgd = torch.optim.SGD([p_sgd], lr=1e-2)
    for _ in range(200):
        opt_sgd.zero_grad()
        quadratic_loss(p_sgd).backward()
        opt_sgd.step()
    loss_sgd = quadratic_loss(p_sgd).item()

    assert loss_adopt < loss_sgd, (
        f"ADOPT loss {loss_adopt:.6f} should be less than SGD loss {loss_sgd:.6f}"
    )


# ------------------------------------------------------------------------------- #
# 13. Closure support: step() respects a closure and returns the loss              #
# ------------------------------------------------------------------------------- #


def test_closure_returns_loss():
    p = torch.nn.Parameter(torch.tensor(2.0))
    opt = ADOPT([p], lr=1e-3)

    def closure():
        opt.zero_grad()
        loss = quadratic_loss(p)
        loss.backward()
        return loss

    returned_loss = opt.step(closure)
    assert returned_loss is not None
    assert returned_loss.item() > 0.0


# ------------------------------------------------------------------------------- #
# 14. Multiple parameter groups with different hyperparameters                     #
# ------------------------------------------------------------------------------- #


def test_multiple_param_groups():
    p1 = torch.nn.Parameter(torch.tensor(3.0))
    p2 = torch.nn.Parameter(torch.tensor(-3.0))
    opt = ADOPT(
        [
            {"params": [p1], "lr": 1e-1},
            {"params": [p2], "lr": 1e-3},
        ],
        lr=1e-2,
    )
    # p1 has a much larger lr so should move more per step
    loss = quadratic_loss(p1) + quadratic_loss(p2)
    loss.backward()
    p1_before = p1.item()
    p2_before = p2.item()
    opt.step()
    delta_p1 = abs(p1.item() - p1_before)
    delta_p2 = abs(p2.item() - p2_before)
    assert delta_p1 > delta_p2, (
        f"High-lr group (p1) should move more: Δp1={delta_p1}, Δp2={delta_p2}"
    )
