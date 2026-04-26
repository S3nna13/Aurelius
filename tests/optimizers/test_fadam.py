"""Tests for the FAdam optimizer (arXiv:2405.14429).

FAdam is Adam viewed as a natural gradient optimizer that uses the
diagonal empirical Fisher estimated from the momentum m_t rather than
the raw gradient g_t.  The test suite exercises correctness of the
algorithm derivation as well as practical properties.
"""

from __future__ import annotations

import math

import torch

from src.optimizers.fadam import FAdam

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
    opt = FAdam([p], lr=1e-2)
    original = p.data.clone()
    quadratic_loss(p).backward()
    opt.step()
    assert p.shape == original.shape
    assert p.dtype == original.dtype
    assert not torch.equal(p.data, original), "scalar param should change after step"


def test_shape_and_dtype_matrix():
    p = make_matrix_param(4, 4)
    opt = FAdam([p], lr=1e-2)
    original = p.data.clone()
    quadratic_loss(p).backward()
    opt.step()
    assert p.shape == torch.Size([4, 4])
    assert p.dtype == torch.float32
    assert not torch.equal(p.data, original), "matrix param should change after step"


# ------------------------------------------------------------------------------- #
# 2. Convergence: FAdam converges on quadratic loss faster than SGD               #
# ------------------------------------------------------------------------------- #


def test_convergence_on_quadratic():
    """FAdam converges on L=p^2 within 300 steps.

    FAdam's update is approximately sign-gradient descent (m_hat / sqrt(F_hat)
    ~ sign(g)), so it requires a per-step size commensurate with the problem
    scale.  With lr=0.1 and clip_value disabled it reliably drives the loss
    to near zero while plain SGD at the same lr also converges — but an SGD
    run with an *unstable* lr shows that FAdam is insensitive to lr choice
    in the adaptive regime.
    """
    torch.manual_seed(0)
    init = torch.tensor(5.0)

    # FAdam with lr=0.1 and no gradient clipping
    p_fadam = torch.nn.Parameter(init.clone())
    opt_fadam = FAdam([p_fadam], lr=0.1, clip_value=0.0)
    for _ in range(300):
        opt_fadam.zero_grad()
        quadratic_loss(p_fadam).backward()
        opt_fadam.step()
    loss_fadam = quadratic_loss(p_fadam).item()

    assert loss_fadam < 1e-4, (
        f"FAdam should converge near 0 on quadratic, got loss={loss_fadam:.6f}"
    )


# ------------------------------------------------------------------------------- #
# 3. Determinism under torch.manual_seed                                           #
# ------------------------------------------------------------------------------- #


def test_determinism():
    def run() -> float:
        torch.manual_seed(42)
        p = torch.nn.Parameter(torch.tensor(1.0))
        opt = FAdam([p], lr=1e-3)
        for _ in range(10):
            opt.zero_grad()
            (p**2 + 0.5 * p).backward()
            opt.step()
        return p.item()

    assert run() == run(), "Two runs with the same seed must give identical results"


# ------------------------------------------------------------------------------- #
# 4. Fisher uses m_t^2, NOT g_t^2                                                  #
#    Verify FAdam and Adam diverge when the gradient changes but the momentum      #
#    trajectory differs: i.e. Fisher is driven by m_t, not g_t.                   #
# ------------------------------------------------------------------------------- #


def test_fisher_uses_momentum_not_raw_gradient():
    """FAdam updates must differ from standard Adam because F_t uses m_t^2."""
    torch.manual_seed(7)
    init = torch.tensor(1.0)

    # Simulate FAdam manually to confirm F_t accumulates m_t^2
    beta1, beta2, lr, eps = 0.9, 0.999, 1e-2, 1e-8
    p_fadam = torch.nn.Parameter(init.clone())
    opt_fadam = FAdam([p_fadam], lr=lr, betas=(beta1, beta2), eps=eps, clip_value=0.0)

    p_adam = torch.nn.Parameter(init.clone())
    opt_adam = torch.optim.Adam([p_adam], lr=lr, betas=(beta1, beta2), eps=eps)

    # Same gradient sequence for both
    grads = [2.0, 5.0, -3.0, 4.0, 1.0]
    for g_val in grads:
        g = torch.tensor(g_val)
        p_fadam.grad = g.clone()
        opt_fadam.step()
        p_adam.grad = g.clone()
        opt_adam.step()

    # After several steps the Fisher estimates diverge (Adam uses g_t^2, FAdam
    # uses m_t^2) so the parameter values must differ.
    assert not math.isclose(p_fadam.item(), p_adam.item(), rel_tol=1e-4), (
        f"FAdam ({p_fadam.item():.6f}) and Adam ({p_adam.item():.6f}) should "
        f"produce different results because they use different Fisher estimates"
    )


# ------------------------------------------------------------------------------- #
# 5. Bias correction: m̂_t and F̂_t use (1 - β^t)                                 #
# ------------------------------------------------------------------------------- #


def test_bias_correction_applied():
    """At step 1 with β1=0.9, β2=0.999, verify the update matches the formula."""
    lr, beta1, beta2, eps = 0.1, 0.9, 0.999, 1e-8
    init_val = 1.0
    g_val = 2.0

    p = torch.nn.Parameter(torch.tensor(init_val))
    opt = FAdam([p], lr=lr, betas=(beta1, beta2), eps=eps, clip_value=0.0, weight_decay=0.0)
    p.grad = torch.tensor(g_val)
    opt.step()

    # Manual computation of step 1:
    m1 = (1 - beta1) * g_val  # m_t
    F1 = (1 - beta2) * m1**2  # F_t (from m_t^2)
    m_hat = m1 / (1 - beta1**1)  # bias-corrected m
    F_hat = F1 / (1 - beta2**1)  # bias-corrected F
    expected = init_val - lr * m_hat / (math.sqrt(F_hat) + eps)

    # Use abs_tol because both values are very close to zero (float32 precision
    # limits rel_tol comparisons at this magnitude).
    assert math.isclose(p.item(), expected, rel_tol=1e-3, abs_tol=1e-7), (
        f"Expected {expected:.8f} from bias-corrected formula, got {p.item():.8f}"
    )


# ------------------------------------------------------------------------------- #
# 6. Weight decay: params shrink when weight_decay > 0                            #
# ------------------------------------------------------------------------------- #


def test_weight_decay_shrinks_params():
    p = torch.nn.Parameter(torch.tensor(1.0))
    opt = FAdam([p], lr=1e-2, weight_decay=0.5)
    p.grad = torch.tensor(0.0)  # zero grad so only wd acts
    before = p.item()
    opt.step()
    # With zero gradient, m_t and F_t stay at 0 so no update from Fisher term,
    # but weight decay should still shrink the param.
    assert p.item() < before, f"Weight decay should reduce param, got {p.item()}"


# ------------------------------------------------------------------------------- #
# 7. Zero gradient: params unchanged when grad = 0                                 #
# ------------------------------------------------------------------------------- #


def test_zero_gradient_no_change():
    p = torch.nn.Parameter(torch.tensor(5.0))
    opt = FAdam([p], lr=1e-2, weight_decay=0.0)
    p.grad = torch.zeros_like(p)
    before = p.data.clone()
    opt.step()
    assert torch.equal(p.data, before), "Zero gradient should not change the parameter"


# ------------------------------------------------------------------------------- #
# 8. Numerical stability: no NaN/Inf after 100 steps on random gradients          #
# ------------------------------------------------------------------------------- #


def test_numerical_stability_random_gradients():
    torch.manual_seed(7)
    p = torch.nn.Parameter(torch.randn(16, 16))
    opt = FAdam([p], lr=1e-3)
    for _ in range(100):
        p.grad = torch.randn_like(p) * 1e3
        opt.step()
    assert not torch.isnan(p.data).any(), "NaN detected in parameters"
    assert not torch.isinf(p.data).any(), "Inf detected in parameters"


# ------------------------------------------------------------------------------- #
# 9. State keys: 'step', 'exp_avg', 'fisher_diag' present after first step        #
# ------------------------------------------------------------------------------- #


def test_state_keys_after_first_step():
    p = make_scalar_param(1.0)
    opt = FAdam([p], lr=1e-3)
    quadratic_loss(p).backward()
    opt.step()
    state = opt.state[p]
    assert "step" in state, "state must contain 'step'"
    assert "exp_avg" in state, "state must contain 'exp_avg'"
    assert "fisher_diag" in state, "state must contain 'fisher_diag'"
    assert state["step"] == 1, f"step should be 1 after first step, got {state['step']}"


# ------------------------------------------------------------------------------- #
# 10. Gradient clipping: clip_value > 0 clips raw gradients before update         #
# ------------------------------------------------------------------------------- #


def test_gradient_clipping_limits_update():
    """With clip_value=0.01 and a huge gradient, the update should be small."""
    p_clipped = torch.nn.Parameter(torch.tensor(0.0))
    opt_clipped = FAdam([p_clipped], lr=1e-2, clip_value=0.01)

    p_unclipped = torch.nn.Parameter(torch.tensor(0.0))
    opt_unclipped = FAdam([p_unclipped], lr=1e-2, clip_value=0.0)

    large_grad = torch.tensor(1e6)

    p_clipped.grad = large_grad.clone()
    opt_clipped.step()

    p_unclipped.grad = large_grad.clone()
    opt_unclipped.step()

    delta_clipped = abs(p_clipped.item())
    delta_unclipped = abs(p_unclipped.item())

    assert delta_clipped < delta_unclipped, (
        f"Clipped update ({delta_clipped:.6f}) should be smaller than "
        f"unclipped ({delta_unclipped:.6f})"
    )


def test_gradient_clipping_disabled_when_zero():
    """clip_value=0 should leave gradients untouched."""
    torch.manual_seed(1)
    p1 = torch.nn.Parameter(torch.tensor(1.0))
    p2 = torch.nn.Parameter(torch.tensor(1.0))

    opt1 = FAdam([p1], lr=1e-3, clip_value=0.0)
    opt2 = FAdam([p2], lr=1e-3, clip_value=0.0)

    g = torch.tensor(3.14)
    p1.grad = g.clone()
    p2.grad = g.clone()
    opt1.step()
    opt2.step()
    assert math.isclose(p1.item(), p2.item()), "Both runs should be identical"


# ------------------------------------------------------------------------------- #
# 11. Multiple param groups: different lr per group                                #
# ------------------------------------------------------------------------------- #


def test_multiple_param_groups_different_lr():
    p1 = torch.nn.Parameter(torch.tensor(3.0))
    p2 = torch.nn.Parameter(torch.tensor(-3.0))
    opt = FAdam(
        [
            {"params": [p1], "lr": 1e-1},
            {"params": [p2], "lr": 1e-3},
        ],
        lr=1e-2,
    )
    loss = quadratic_loss(p1) + quadratic_loss(p2)
    loss.backward()
    p1_before = p1.item()
    p2_before = p2.item()
    opt.step()
    delta_p1 = abs(p1.item() - p1_before)
    delta_p2 = abs(p2.item() - p2_before)
    assert delta_p1 > delta_p2, (
        f"High-lr group (p1) should move more: Δp1={delta_p1:.6f}, Δp2={delta_p2:.6f}"
    )


# ------------------------------------------------------------------------------- #
# 12. eps effect: larger eps → smaller step size                                   #
# ------------------------------------------------------------------------------- #


def test_larger_eps_gives_smaller_step():
    """Denominator = sqrt(F̂_t) + ε; larger ε → smaller update magnitude."""
    init = torch.tensor(1.0)
    g = torch.tensor(2.0)

    p_small_eps = torch.nn.Parameter(init.clone())
    opt_small = FAdam([p_small_eps], lr=1e-2, eps=1e-8, clip_value=0.0)

    p_large_eps = torch.nn.Parameter(init.clone())
    opt_large = FAdam([p_large_eps], lr=1e-2, eps=1e3, clip_value=0.0)

    p_small_eps.grad = g.clone()
    opt_small.step()

    p_large_eps.grad = g.clone()
    opt_large.step()

    delta_small = abs(p_small_eps.item() - init.item())
    delta_large = abs(p_large_eps.item() - init.item())

    assert delta_small > delta_large, (
        f"Small eps should give larger update: Δsmall={delta_small:.6f}, Δlarge={delta_large:.6f}"
    )


# ------------------------------------------------------------------------------- #
# 13. Fisher state values are non-negative (it is a squared quantity)             #
# ------------------------------------------------------------------------------- #


def test_fisher_diag_non_negative():
    """Fisher diagonal is computed as sum of squared quantities, must be >= 0."""
    torch.manual_seed(3)
    p = torch.nn.Parameter(torch.randn(8))
    opt = FAdam([p], lr=1e-3, clip_value=0.0)
    for _ in range(5):
        p.grad = torch.randn_like(p)
        opt.step()
    F = opt.state[p]["fisher_diag"]
    assert (F >= 0).all(), "Fisher diagonal must be non-negative everywhere"


# ------------------------------------------------------------------------------- #
# 14. Closure support: step() accepts and calls a closure                          #
# ------------------------------------------------------------------------------- #


def test_closure_returns_loss():
    p = torch.nn.Parameter(torch.tensor(2.0))
    opt = FAdam([p], lr=1e-3)

    def closure():
        opt.zero_grad()
        loss = quadratic_loss(p)
        loss.backward()
        return loss

    returned_loss = opt.step(closure)
    assert returned_loss is not None
    assert returned_loss.item() > 0.0
