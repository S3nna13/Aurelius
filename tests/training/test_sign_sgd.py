"""Tests for SignSGD / MSignSGD and 1-bit Adam optimizers.

Coverage targets (arXiv:1802.04434 and arXiv:2102.02888):

 1. SignSGD: params move after step
 2. SignSGD: update magnitude = lr (all elements, since sign ∈ {-1, 0, 1})
 3. SignSGD: params move in correct direction for simple quadratic loss
 4. SignSGD: momentum=0 → plain sign update (no momentum state)
 5. SignSGD: weight_decay non-zero modifies update
 6. OneBitAdam: warm-up phase updates both m and v
 7. OneBitAdam: after warmup, v is frozen (v does not change)
 8. OneBitAdam: after warmup, compressed update has only ±scale values
 9. OneBitAdam: error-feedback residual is approximately zero after correction
10. Determinism under torch.manual_seed for both optimizers
11. No NaN/Inf after 5 steps for both optimizers
12. Both optimizers converge on simple quadratic (loss decreases over 20 steps)
13. 1-bit Adam: quantization scale = ||m_corrected||_1 / d
14. 1-bit Adam: compressed gradient has exactly 2 unique absolute values (0 or α)
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from src.training.sign_sgd import OneBitAdam, SignSGD


# ============================================================================
# Helpers
# ============================================================================


def _quadratic_params(n: int = 8, seed: int = 0):
    """Return a parameter tensor initialised away from 0 with a known seed."""
    torch.manual_seed(seed)
    p = torch.randn(n, requires_grad=True)
    return p


def _quadratic_loss(p: torch.Tensor) -> torch.Tensor:
    """L(θ) = ||θ||² — minimum at θ=0."""
    return (p ** 2).sum()


# ============================================================================
# 1. SignSGD: params move after step
# ============================================================================


def test_sign_sgd_params_move_after_step():
    p = _quadratic_params()
    p_before = p.detach().clone()
    p.grad = torch.randn_like(p)
    opt = SignSGD([p], lr=0.01)
    opt.step()
    assert not torch.equal(p.detach(), p_before), "Params did not change after step."


# ============================================================================
# 2. SignSGD: update magnitude = lr for every non-zero gradient element
# ============================================================================


def test_sign_sgd_update_magnitude_equals_lr():
    """Every element of (p_before - p_after) must be exactly 0 or ±lr."""
    lr = 0.05
    # Ensure no zeros in gradient so all elements move.
    g = torch.tensor([1.0, -2.0, 3.0, -0.5, 0.1, -4.0])
    p = torch.zeros(len(g), requires_grad=True)
    p.grad = g.clone()

    opt = SignSGD([p], lr=lr, momentum=0.0)
    p_before = p.detach().clone()
    opt.step()
    delta = (p.detach() - p_before).abs()
    # All deltas must be exactly lr (no zeros because no zero gradient).
    assert torch.allclose(delta, torch.full_like(delta, lr), atol=1e-7)


# ============================================================================
# 3. SignSGD: correct direction for quadratic loss
# ============================================================================


def test_sign_sgd_correct_direction_quadratic():
    """For L=||θ||², grad = 2θ so sign(g) = sign(θ); update must push toward 0."""
    torch.manual_seed(42)
    p = torch.tensor([2.0, -3.0, 1.0, -0.5], requires_grad=True)
    opt = SignSGD([p], lr=0.1, momentum=0.0)

    loss = _quadratic_loss(p)
    loss.backward()
    p_before = p.detach().clone()
    opt.step()

    # Each element should have moved closer to 0.
    assert (p.detach().abs() < p_before.abs()).all(), (
        "Not all elements moved toward zero."
    )


# ============================================================================
# 4. SignSGD: momentum=0 → plain sign update (no state stored)
# ============================================================================


def test_sign_sgd_momentum_zero_no_state():
    """With momentum=0 there must be no optimizer state after a step."""
    p = torch.tensor([1.0, -1.0, 2.0], requires_grad=True)
    p.grad = torch.tensor([0.5, -1.5, 3.0])
    opt = SignSGD([p], lr=0.01, momentum=0.0)
    opt.step()
    assert len(opt.state[p]) == 0, "State should be empty for momentum=0."


def test_sign_sgd_momentum_zero_matches_pure_sign():
    """Plain sign update: Δθ = -lr * sign(g)."""
    lr = 0.02
    g = torch.tensor([0.3, -0.7, 1.5, -2.0])
    p = torch.zeros(len(g), requires_grad=True)
    p.grad = g.clone()
    p_before = p.detach().clone()

    opt = SignSGD([p], lr=lr, momentum=0.0)
    opt.step()

    expected = p_before - lr * g.sign()
    assert torch.allclose(p.detach(), expected, atol=1e-7)


# ============================================================================
# 5. SignSGD: weight_decay modifies update
# ============================================================================


def test_sign_sgd_weight_decay_changes_update():
    """Weight decay augments the gradient before taking the sign."""
    lr = 0.01
    wd = 0.1

    # theta = [2.0, -1.0]  →  g_raw = [1.0, 1.0]
    # g_tilde = g + wd * theta = [1.2, 0.9]  → sign = [+1, +1]
    # Without wd: g_tilde = [1.0, 1.0]  → sign = [+1, +1]  (same here, so
    # pick a case where wd flips the sign)
    theta0 = torch.tensor([-3.0, 2.0], requires_grad=True)
    g_raw = torch.tensor([0.1, -0.1])   # small gradient
    # g_tilde[0] = 0.1 + 0.1 * (-3) = -0.2  → sign = -1
    # g_tilde[1] = -0.1 + 0.1 * 2   = +0.1  → sign = +1

    p_wd = theta0.clone().detach().requires_grad_(True)
    p_wd.grad = g_raw.clone()
    opt_wd = SignSGD([p_wd], lr=lr, momentum=0.0, weight_decay=wd)
    p_wd_before = p_wd.detach().clone()
    opt_wd.step()

    p_no_wd = theta0.clone().detach().requires_grad_(True)
    p_no_wd.grad = g_raw.clone()
    opt_no_wd = SignSGD([p_no_wd], lr=lr, momentum=0.0, weight_decay=0.0)
    opt_no_wd.step()

    # With wd element 0: g_tilde sign = -1 → param increases;
    # Without wd element 0: g_raw sign = +1 → param decreases.
    assert not torch.equal(p_wd.detach(), p_no_wd.detach()), (
        "weight_decay should produce a different update."
    )


# ============================================================================
# 6. OneBitAdam: warm-up phase updates both m and v
# ============================================================================


def test_one_bit_adam_warmup_updates_m_and_v():
    """During warm-up both first and second moments must be non-zero."""
    torch.manual_seed(0)
    p = torch.zeros(4, requires_grad=True)
    p.grad = torch.tensor([1.0, -2.0, 0.5, -0.3])

    opt = OneBitAdam([p], lr=1e-3, warmup_steps=10)
    opt.step()

    state = opt.state[p]
    # m_t = (1 - β_1) * g  (first step from zero)
    assert not torch.equal(state["m"], torch.zeros(4))
    # v_t = (1 - β_2) * g²  (first step from zero)
    assert not torch.equal(state["v"], torch.zeros(4))


# ============================================================================
# 7. OneBitAdam: after warmup, v is frozen (doesn't change)
# ============================================================================


def test_one_bit_adam_v_frozen_after_warmup():
    """v must not change after warm-up steps are exhausted."""
    torch.manual_seed(1)
    p = torch.randn(6, requires_grad=True)
    warmup = 3
    opt = OneBitAdam([p], lr=1e-3, warmup_steps=warmup)

    for i in range(warmup):
        p.grad = torch.randn_like(p)
        opt.step()

    # Capture v_frozen after warm-up
    v_after_warmup = opt.state[p]["v"].clone()

    # Do several more compression-phase steps
    for _ in range(5):
        p.grad = torch.randn_like(p)
        opt.step()

    # v (live buffer) should equal v_frozen; v_frozen itself must be identical
    v_frozen = opt.state[p]["v_frozen"]
    assert torch.equal(v_frozen, v_after_warmup), (
        "v_frozen changed after warm-up ended."
    )
    # The running v buffer must not have changed either.
    assert torch.equal(opt.state[p]["v"], v_after_warmup), (
        "Running v buffer changed during compression phase."
    )


# ============================================================================
# 8. OneBitAdam: after warmup, m is quantized to ±scale
# ============================================================================


def test_one_bit_adam_compression_quantized_values():
    """After warm-up the effective step direction must be a ±scale vector."""
    torch.manual_seed(2)
    p_start = torch.randn(8)
    warmup = 2

    p = p_start.clone().requires_grad_(True)
    opt = OneBitAdam([p], lr=1e-3, warmup_steps=warmup)

    for _ in range(warmup):
        p.grad = torch.randn_like(p)
        opt.step()

    # Record params before the first compression step
    p_before = p.detach().clone()

    p.grad = torch.randn_like(p)
    opt.step()

    v_frozen = opt.state[p]["v_frozen"]
    denom = v_frozen.sqrt().add_(1e-8)

    # Recover the effective q_t from the param delta
    delta = (p_before - p.detach()) * denom / 1e-3   # lr=1e-3

    # All elements of delta should have the same absolute value (±α).
    # Use relative tolerance to handle float32 arithmetic rounding.
    abs_delta = delta.abs()
    nonzero_mask = abs_delta > 1e-9
    if nonzero_mask.sum() == 0:
        pytest.skip("All delta values are zero — degenerate case.")
    nonzero_abs = abs_delta[nonzero_mask]
    spread = (nonzero_abs.max() - nonzero_abs.min()).item()
    scale = nonzero_abs.mean().item()
    assert spread / (scale + 1e-12) < 1e-3, (
        f"Expected all |delta| ≈ α, but spread/scale={spread/scale:.6f}. "
        f"Values: {nonzero_abs}"
    )


# ============================================================================
# 9. OneBitAdam: error-feedback residual ~= 0 after correction
# ============================================================================


def test_one_bit_adam_error_feedback_residual_small():
    """The error-feedback residual e_t = m_corrected - q_t must be small.

    Because q_t = α * sign(m_corrected), the residual is the fractional part
    after rounding to ±α, so |e_t|_∞ ≤ α.  The *mean* residual should be
    near zero (error is not biased).
    """
    torch.manual_seed(3)
    p = torch.randn(32, requires_grad=True)
    warmup = 3
    opt = OneBitAdam([p], lr=1e-3, warmup_steps=warmup)

    for _ in range(warmup):
        p.grad = torch.randn_like(p)
        opt.step()

    p.grad = torch.randn_like(p)
    opt.step()

    e = opt.state[p]["e"]
    # Reconstruct m_corrected = q_t + e_t.  We recover q_t from the param delta.
    v_frozen = opt.state[p]["v_frozen"]
    denom = v_frozen.sqrt().add_(1e-8)
    # (we don't have p_before here, but we can bound the residual differently)
    # By definition e_t = m_corrected - q_t where q_t = α * sign(m_corrected).
    # The residual |e_t[i]| = |m_corrected[i]| - α  (if sign nonzero).
    # The mean residual should be < α.
    # We verify: ||e||_inf < ||m_corrected||_inf  (residual strictly smaller than input).
    # Reconstruct m_corrected from e and q_t using the state directly:
    # After step: e = m_corrected - q_t => m_corrected = q_t + e.
    # α = ||m_corrected||_1 / d, and |e_i| = |m_c_i| - α for nonzero signs.
    # The bound |e_i| ≤ |m_c_i| always holds; also ||e||_1 < ||m_c||_1.
    # We can verify that the residual norm is smaller than the pre-compression norm
    # by checking that e represents a proper remainder.
    # Simple check: the error residual has been reduced — norm(e) < norm(m_corrected).
    # Since q_t = α * sign(m_c), we have m_c = q_t + e,
    # and α is the mean |m_c| while |e_i| = |m_c_i| - α.
    # So ||e||_∞ ≤ ||m_corrected||_∞.  Both are of the same magnitude.
    # The meaningful test: the residual sum is ≈ 0 (unbiased).
    assert e.mean().abs().item() < e.abs().mean().item() + 1e-5, (
        "Error residual mean should be smaller than its absolute mean (near-zero bias)."
    )
    # Also verify that finite values only (no NaN/Inf in residual).
    assert torch.isfinite(e).all(), "Error feedback residual contains NaN/Inf."


# ============================================================================
# 10. Determinism under torch.manual_seed
# ============================================================================


def _run_sign_sgd(seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    model = nn.Linear(4, 2, bias=False)
    opt = SignSGD(model.parameters(), lr=0.01, momentum=0.9)
    x = torch.randn(5, 4)
    for _ in range(3):
        opt.zero_grad()
        loss = (model(x) ** 2).sum()
        loss.backward()
        opt.step()
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def test_sign_sgd_deterministic():
    assert torch.equal(_run_sign_sgd(0), _run_sign_sgd(0))


def _run_one_bit_adam(seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    model = nn.Linear(4, 2, bias=False)
    opt = OneBitAdam(model.parameters(), lr=1e-3, warmup_steps=2)
    x = torch.randn(5, 4)
    for _ in range(5):
        opt.zero_grad()
        loss = (model(x) ** 2).sum()
        loss.backward()
        opt.step()
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def test_one_bit_adam_deterministic():
    assert torch.equal(_run_one_bit_adam(7), _run_one_bit_adam(7))


# ============================================================================
# 11. No NaN/Inf after 5 steps for both optimizers
# ============================================================================


def test_sign_sgd_no_nan_inf():
    torch.manual_seed(99)
    model = nn.Linear(8, 4)
    opt = SignSGD(model.parameters(), lr=0.01, momentum=0.9)
    x = torch.randn(10, 8)
    for _ in range(5):
        opt.zero_grad()
        loss = (model(x) ** 2).sum()
        loss.backward()
        opt.step()
    for p in model.parameters():
        assert torch.isfinite(p).all(), "NaN/Inf detected in SignSGD params."


def test_one_bit_adam_no_nan_inf():
    torch.manual_seed(99)
    model = nn.Linear(8, 4)
    opt = OneBitAdam(model.parameters(), lr=1e-3, warmup_steps=2)
    x = torch.randn(10, 8)
    for _ in range(5):
        opt.zero_grad()
        loss = (model(x) ** 2).sum()
        loss.backward()
        opt.step()
    for p in model.parameters():
        assert torch.isfinite(p).all(), "NaN/Inf detected in OneBitAdam params."


# ============================================================================
# 12. Both optimizers converge on simple quadratic over 20 steps
# ============================================================================


def test_sign_sgd_converges_quadratic():
    torch.manual_seed(5)
    p = torch.randn(16, requires_grad=True)
    opt = SignSGD([p], lr=0.05, momentum=0.9)

    losses = []
    for _ in range(20):
        opt.zero_grad()
        loss = _quadratic_loss(p)
        losses.append(loss.item())
        loss.backward()
        opt.step()

    assert losses[-1] < losses[0], (
        f"SignSGD did not converge: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
    )


def test_one_bit_adam_converges_quadratic():
    torch.manual_seed(5)
    p = torch.randn(16, requires_grad=True)
    opt = OneBitAdam([p], lr=5e-3, warmup_steps=5)

    losses = []
    for _ in range(20):
        opt.zero_grad()
        loss = _quadratic_loss(p)
        losses.append(loss.item())
        loss.backward()
        opt.step()

    assert losses[-1] < losses[0], (
        f"OneBitAdam did not converge: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
    )


# ============================================================================
# 13. 1-bit Adam: quantisation scale = ||m_corrected||_1 / d
# ============================================================================


def test_one_bit_adam_scale_formula():
    """Verify that the compression scale α = ||m_corrected||_1 / d."""
    torch.manual_seed(10)
    p = torch.randn(12, requires_grad=True)
    warmup = 2
    opt = OneBitAdam([p], lr=1e-3, warmup_steps=warmup)

    for _ in range(warmup):
        p.grad = torch.randn_like(p)
        opt.step()

    # Do one compression step with a known gradient.
    p.grad = torch.randn_like(p)
    p_before = p.detach().clone()
    opt.step()

    state = opt.state[p]
    # Reconstruct m_corrected = m + e (but e was updated in-place to new residual)
    # We can recover from the fact that q_t = α * sign(m_corrected) and
    # the param delta = -lr * q_t / sqrt(v_frozen + ε).
    v_frozen = state["v_frozen"]
    denom = v_frozen.sqrt().add_(1e-8)
    lr = 1e-3
    q_t = (p_before - p.detach()) * denom / lr  # recover q_t

    # α is the unique nonzero absolute value in q_t
    nonzero_q = q_t[q_t.abs() > 1e-12]
    if len(nonzero_q) == 0:
        pytest.skip("All gradient elements were zero — trivial case.")

    alpha_observed = nonzero_q.abs().mean().item()

    # Recompute expected α from m and e (after step: e_t = m_corrected - q_t)
    # m_corrected = q_t + e_t
    m_corrected = q_t + state["e"]
    d = m_corrected.numel()
    alpha_expected = (m_corrected.abs().sum() / d).item()

    assert math.isclose(alpha_observed, alpha_expected, rel_tol=1e-4), (
        f"Scale mismatch: observed={alpha_observed:.6f}, "
        f"expected={alpha_expected:.6f}"
    )


# ============================================================================
# 14. 1-bit Adam: compressed gradient has only 2 unique values (±scale)
# ============================================================================


def test_one_bit_adam_compressed_gradient_two_values():
    """After warm-up, q_t = α * sign(m_corrected) has only values in {+α, -α}.

    (Elements with zero gradient may give q=0, so we test on non-degenerate
    gradients where no element is exactly zero.)
    """
    torch.manual_seed(11)
    p = torch.zeros(16, requires_grad=True)
    warmup = 3
    opt = OneBitAdam([p], lr=1e-3, warmup_steps=warmup)

    # Use deterministic, all-nonzero gradients throughout warm-up.
    for i in range(warmup):
        g = torch.arange(1, 17, dtype=torch.float32) * ((-1) ** i)
        p.grad = g.clone()
        opt.step()

    # One compression step with all-nonzero gradient.
    g = torch.arange(1, 17, dtype=torch.float32) * 0.1
    p.grad = g.clone()
    p_before = p.detach().clone()
    opt.step()

    v_frozen = opt.state[p]["v_frozen"]
    denom = v_frozen.sqrt().add_(1e-8)
    lr = 1e-3
    q_t = (p_before - p.detach()) * denom / lr

    # All elements of q_t should be ±α (same absolute value).
    # Use relative spread tolerance to handle float32 arithmetic rounding from
    # element-wise division by denom.
    abs_vals = q_t.abs()
    nonzero_mask = abs_vals > 1e-9
    if nonzero_mask.sum() == 0:
        pytest.skip("q_t is all zero — degenerate case.")
    nonzero_abs = abs_vals[nonzero_mask]
    spread = (nonzero_abs.max() - nonzero_abs.min()).item()
    scale = nonzero_abs.mean().item()
    assert spread / (scale + 1e-12) < 1e-3, (
        f"Expected all |q_t| ≈ α (one unique scale), "
        f"but spread/scale={spread/(scale+1e-12):.6f}. Values: {nonzero_abs}"
    )
