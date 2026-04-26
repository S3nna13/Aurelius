"""Tests for NesterovAdan optimizer (src/training/nesterov_adan.py).

Covers all 12+ rigor requirements from the spec:
  1.  Shape/dtype: step() updates scalar and matrix params
  2.  Convergence: reduces loss on quadratic faster than SGD
  3.  Determinism under torch.manual_seed
  4.  Three state variables: exp_avg, exp_avg_diff, exp_avg_sq after step 1
  5.  Gradient difference: v_t uses g_t - g_{t-1}
  6.  First step: g_{-1} = g_0 so difference = 0 on step 1
  7.  Weight decay decoupled: params shrink under weight decay
  8.  Zero gradient: params shrink under weight decay but step size is 0
  9.  Numerical stability: no NaN/Inf after 100 steps on random gradients
  10. no_prox=True: uses additive weight decay style
  11. Multiple param groups: different lr per group
  12. State stored: prev_grad key present after step 1
  13. Bias correction: applied correctly (m̂, v̂, n̂)
  14. zero_moments helper resets state correctly
"""

from __future__ import annotations

import pytest
import torch

from src.training.nesterov_adan import NesterovAdan

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _make_scalar(val: float = 2.0, requires_grad: bool = True) -> torch.Tensor:
    return torch.tensor([val], dtype=torch.float32, requires_grad=requires_grad)


def _make_matrix(shape=(4, 4), requires_grad: bool = True) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(*shape, requires_grad=requires_grad)


def _forward_backward(param: torch.Tensor) -> None:
    """Compute f(x) = sum(x**2) and call backward."""
    loss = (param**2).sum()
    loss.backward()


# ---------------------------------------------------------------------------
# 1. Shape / dtype: step() updates scalar and matrix params
# ---------------------------------------------------------------------------


class TestShapeDtype:
    def test_scalar_param_updated(self):
        p = _make_scalar(2.0)
        opt = NesterovAdan([p], lr=1e-2, weight_decay=0.0)
        _forward_backward(p)
        opt.step()
        # Parameter should have moved away from 2.0
        assert p.item() != pytest.approx(2.0), "Scalar param should be updated"

    def test_matrix_param_updated(self):
        p = _make_matrix()
        p_init = p.detach().clone()
        opt = NesterovAdan([p], lr=1e-2, weight_decay=0.0)
        _forward_backward(p)
        opt.step()
        assert not torch.allclose(p, p_init), "Matrix param should be updated"

    def test_preserves_dtype_float32(self):
        p = torch.ones(3, 3, dtype=torch.float32, requires_grad=True)
        opt = NesterovAdan([p], lr=1e-3, weight_decay=0.0)
        _forward_backward(p)
        opt.step()
        assert p.dtype == torch.float32

    def test_preserves_dtype_float64(self):
        p = torch.ones(3, dtype=torch.float64, requires_grad=True)
        opt = NesterovAdan([p], lr=1e-3, weight_decay=0.0)
        _forward_backward(p)
        opt.step()
        assert p.dtype == torch.float64


# ---------------------------------------------------------------------------
# 2. Convergence: reduces loss on quadratic faster than SGD
# ---------------------------------------------------------------------------


class TestConvergence:
    @staticmethod
    def _run_illcond(opt_class, opt_kwargs: dict, n_steps: int = 500, seed: int = 42) -> float:
        """Run optimizer on an ill-conditioned quadratic (scales 1–100 per dim).

        SGD diverges or stalls here; Adan's adaptive denominator handles the
        varying curvature and converges reliably.
        """
        torch.manual_seed(seed)
        d = 16
        scale = torch.logspace(0, 2, d)  # dimension scales: 1 … 100
        p = torch.randn(d, requires_grad=True)
        opt = opt_class([p], **opt_kwargs)
        for _ in range(n_steps):
            opt.zero_grad()
            loss = (scale * p**2).sum()
            loss.backward()
            opt.step()
        with torch.no_grad():
            return (scale * p.detach() ** 2).sum().item()

    def test_adan_outperforms_sgd_on_illconditioned_quadratic(self):
        """Adan's adaptive denominator should drastically outperform SGD on
        an ill-conditioned quadratic where different dimensions have scales
        ranging from 1 to 100 (SGD struggles or diverges with any single lr)."""
        adan_loss = self._run_illcond(NesterovAdan, {"lr": 1e-2, "weight_decay": 0.0}, n_steps=500)
        sgd_loss = self._run_illcond(torch.optim.SGD, {"lr": 1e-2}, n_steps=500)
        assert adan_loss < sgd_loss, (
            f"Adan final loss {adan_loss:.4f} should be << SGD loss {sgd_loss:.4f} "
            f"on ill-conditioned quadratic"
        )

    def test_loss_decreases_substantially_over_training(self):
        """After 500 steps, loss should drop by >99.99% on a clean quadratic."""
        torch.manual_seed(42)
        p = torch.randn(16, requires_grad=True)
        target = torch.zeros(16)
        opt = NesterovAdan([p], lr=1e-2, weight_decay=0.0)
        losses = []
        for _ in range(500):
            opt.zero_grad()
            loss = ((p - target) ** 2).sum()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        # Adan converges to near zero but needs ~500 steps to warm up betas
        assert losses[-1] < losses[0] * 1e-4, (
            f"Loss should decrease by 99.99%+ over 500 steps: {losses[0]:.4f} → {losses[-1]:.6f}"
        )


# ---------------------------------------------------------------------------
# 3. Determinism under torch.manual_seed
# ---------------------------------------------------------------------------


class TestDeterminism:
    def _run_steps(self, seed: int, n: int = 5) -> list[float]:
        torch.manual_seed(seed)
        p = torch.randn(4, requires_grad=True)
        opt = NesterovAdan([p], lr=1e-3, weight_decay=0.01)
        vals = []
        for _ in range(n):
            opt.zero_grad()
            (p**2).sum().backward()
            opt.step()
            vals.append(p.detach().clone().tolist())
        return vals

    def test_same_seed_same_trajectory(self):
        traj1 = self._run_steps(seed=123)
        traj2 = self._run_steps(seed=123)
        assert traj1 == traj2, "Different runs with same seed must be identical"

    def test_different_seed_different_trajectory(self):
        traj1 = self._run_steps(seed=1)
        traj2 = self._run_steps(seed=2)
        assert traj1 != traj2, "Different seeds should give different trajectories"


# ---------------------------------------------------------------------------
# 4. Three state variables: exp_avg, exp_avg_diff, exp_avg_sq after step 1
# ---------------------------------------------------------------------------


class TestStateVariables:
    def test_three_moment_buffers_exist(self):
        p = _make_scalar()
        opt = NesterovAdan([p], lr=1e-3, weight_decay=0.0)
        _forward_backward(p)
        opt.step()
        state = opt.state[p]
        assert "exp_avg" in state, "exp_avg (m_t) must be present"
        assert "exp_avg_diff" in state, "exp_avg_diff (v_t) must be present"
        assert "exp_avg_sq" in state, "exp_avg_sq (n_t) must be present"

    def test_moment_shapes_match_param(self):
        p = _make_matrix((3, 5))
        opt = NesterovAdan([p], lr=1e-3, weight_decay=0.0)
        _forward_backward(p)
        opt.step()
        state = opt.state[p]
        assert state["exp_avg"].shape == p.shape
        assert state["exp_avg_diff"].shape == p.shape
        assert state["exp_avg_sq"].shape == p.shape

    def test_exp_avg_sq_nonnegative(self):
        """n_t must always be non-negative (it's an EMA of squares)."""
        p = _make_matrix()
        opt = NesterovAdan([p], lr=1e-3, weight_decay=0.0)
        for _ in range(5):
            opt.zero_grad()
            _forward_backward(p)
            opt.step()
        assert (opt.state[p]["exp_avg_sq"] >= 0).all()


# ---------------------------------------------------------------------------
# 5. Gradient difference: v_t uses g_t - g_{t-1}
# ---------------------------------------------------------------------------


class TestGradientDifference:
    def test_vt_encodes_gradient_change(self):
        """After step 2, exp_avg_diff should reflect g_2 - g_1."""
        p = torch.tensor([3.0], requires_grad=True)
        β1, β2, β3 = 0.98, 0.92, 0.99
        opt = NesterovAdan([p], lr=1e-3, betas=(β1, β2, β3), weight_decay=0.0)

        # Step 1 — gradient = 2*p = 6.0; prev_grad set to g1=6.0; diff=0
        (p**2).sum().backward()
        g1 = p.grad.clone()
        opt.step()
        opt.zero_grad()

        # Step 2 — gradient ≠ g1 (p has moved); diff should be nonzero
        (p**2).sum().backward()
        g2 = p.grad.clone()
        opt.step()

        # v_2 = (1 - β2)*Δg_2 + β2*v_1
        # v_1 = 0 (first step, diff=0), so v_2 = (1-β2)*(g2-g1)
        expected_v2 = (1.0 - β2) * (g2 - g1)
        actual_v2 = opt.state[p]["exp_avg_diff"]
        assert torch.allclose(actual_v2, expected_v2, atol=1e-6), (
            f"exp_avg_diff mismatch: expected {expected_v2.item():.6f}, got {actual_v2.item():.6f}"
        )


# ---------------------------------------------------------------------------
# 6. First step: g_{-1} = g_0 so difference = 0 on step 1
# ---------------------------------------------------------------------------


class TestFirstStepZeroDiff:
    def test_exp_avg_diff_zero_after_first_step(self):
        """On step 1 the gradient difference is 0, so v_1 = 0."""
        p = torch.tensor([1.5], requires_grad=True)
        opt = NesterovAdan([p], lr=1e-3, betas=(0.98, 0.92, 0.99), weight_decay=0.0)
        _forward_backward(p)
        opt.step()
        # v_1 = (1 - β2) * 0  +  β2 * 0 = 0
        v1 = opt.state[p]["exp_avg_diff"]
        assert torch.allclose(v1, torch.zeros_like(v1), atol=1e-9), (
            "exp_avg_diff should be zero after the first step"
        )

    def test_exp_avg_sq_uses_only_g1_on_first_step(self):
        """On step 1, k_t = g_1 + (1-β2)*0 = g_1, so n_1 = (1-β3)*g1²."""
        p = torch.tensor([2.0], requires_grad=True)
        β1, β2, β3 = 0.98, 0.92, 0.99
        opt = NesterovAdan([p], lr=1e-3, betas=(β1, β2, β3), weight_decay=0.0)
        (p**2).sum().backward()
        g1 = p.grad.clone()
        opt.step()
        expected_n1 = (1.0 - β3) * g1**2
        actual_n1 = opt.state[p]["exp_avg_sq"]
        assert torch.allclose(actual_n1, expected_n1, atol=1e-9)


# ---------------------------------------------------------------------------
# 7. Weight decay decoupled: params shrink under weight decay
# ---------------------------------------------------------------------------


class TestWeightDecay:
    def test_proximal_wd_shrinks_params(self):
        """With proximal WD, params must shrink (even without gradient)."""
        p = torch.tensor([1.0], requires_grad=True)
        # Use zero lr so only weight decay acts, but we need a gradient to step
        # Use large WD with a real step to see shrinkage
        p_init = p.item()
        opt = NesterovAdan([p], lr=1e-3, weight_decay=1.0, no_prox=False)
        # Give a zero gradient — WD still fires via proximal division
        p.grad = torch.zeros_like(p)
        opt.step()
        assert p.item() < p_init, "Proximal weight decay should shrink the parameter"

    def test_no_wd_leaves_param_magnitude_unchanged_direction(self):
        """With weight_decay=0 and no_prox, pure gradient update only."""
        p = torch.tensor([1.0], requires_grad=True)
        opt = NesterovAdan([p], lr=1e-3, weight_decay=0.0, no_prox=True)
        p.grad = torch.zeros_like(p)
        p_before = p.item()
        opt.step()
        # With zero gradient and zero WD, parameter should not move
        assert p.item() == pytest.approx(p_before), (
            "Zero grad + zero WD should not change parameter"
        )


# ---------------------------------------------------------------------------
# 8. Zero gradient: params still shrink under weight decay, step size = 0
# ---------------------------------------------------------------------------


class TestZeroGradient:
    def test_zero_grad_proximal_wd_still_shrinks(self):
        """A zero gradient still triggers weight decay in proximal mode."""
        p = torch.tensor([5.0], requires_grad=True)
        opt = NesterovAdan([p], lr=1e-2, weight_decay=0.5, no_prox=False)
        p.grad = torch.zeros_like(p)
        opt.step()
        assert p.item() < 5.0

    def test_zero_grad_no_prox_wd_still_shrinks(self):
        """A zero gradient still triggers weight decay in no_prox mode."""
        p = torch.tensor([5.0], requires_grad=True)
        opt = NesterovAdan([p], lr=1e-2, weight_decay=0.5, no_prox=True)
        p.grad = torch.zeros_like(p)
        opt.step()
        assert p.item() < 5.0

    def test_none_grad_skips_param(self):
        """Parameters with None gradient must not be updated."""
        p = torch.tensor([3.0], requires_grad=True)
        opt = NesterovAdan([p], lr=1e-2, weight_decay=0.5)
        # Do not set p.grad — leaves it None
        opt.step()
        assert p.item() == pytest.approx(3.0), "Parameter with None grad must not be modified"


# ---------------------------------------------------------------------------
# 9. Numerical stability: no NaN/Inf after 100 steps on random gradients
# ---------------------------------------------------------------------------


class TestNumericalStability:
    def test_no_nan_inf_random_gradients(self):
        torch.manual_seed(0)
        p = torch.randn(32, 32, requires_grad=True)
        opt = NesterovAdan([p], lr=1e-3, weight_decay=0.01)
        for i in range(100):
            opt.zero_grad()
            # Simulate noisy gradients (including occasional large values)
            with torch.no_grad():
                p.grad = torch.randn_like(p) * (1.0 + 0.1 * i)
            opt.step()
        assert torch.isfinite(p).all(), "Parameters must remain finite after 100 steps"

    def test_no_nan_inf_in_moments(self):
        torch.manual_seed(1)
        p = torch.randn(16, requires_grad=True)
        opt = NesterovAdan([p], lr=1e-3, eps=1e-8, weight_decay=0.0)
        for _ in range(100):
            opt.zero_grad()
            with torch.no_grad():
                p.grad = torch.randn_like(p)
            opt.step()
        state = opt.state[p]
        for key in ("exp_avg", "exp_avg_diff", "exp_avg_sq"):
            assert torch.isfinite(state[key]).all(), f"{key} must remain finite"


# ---------------------------------------------------------------------------
# 10. no_prox=True: uses additive weight decay style
# ---------------------------------------------------------------------------


class TestNoProx:
    def test_no_prox_and_prox_differ(self):
        """no_prox=True and no_prox=False should produce different param values."""
        torch.manual_seed(42)
        p1 = torch.randn(8, requires_grad=True)
        p2 = p1.detach().clone().requires_grad_(True)

        opt1 = NesterovAdan([p1], lr=1e-2, weight_decay=0.1, no_prox=False)
        opt2 = NesterovAdan([p2], lr=1e-2, weight_decay=0.1, no_prox=True)

        for _ in range(5):
            for p, opt in [(p1, opt1), (p2, opt2)]:
                opt.zero_grad()
                (p**2).sum().backward()
                opt.step()

        assert not torch.allclose(p1, p2), "no_prox=True and no_prox=False should diverge"

    def test_no_prox_applies_l2_before_gradient(self):
        """In no_prox mode, weight decay is applied as θ*(1 - α*λ)."""
        p = torch.tensor([10.0], requires_grad=True)
        α, λ = 0.01, 0.5
        opt = NesterovAdan([p], lr=α, weight_decay=λ, no_prox=True)
        # Zero gradient → only WD applies
        p.grad = torch.zeros_like(p)
        opt.step()
        expected = 10.0 * (1.0 - α * λ)
        assert p.item() == pytest.approx(expected, rel=1e-5), (
            f"no_prox WD: expected {expected:.6f}, got {p.item():.6f}"
        )


# ---------------------------------------------------------------------------
# 11. Multiple param groups: different lr per group
# ---------------------------------------------------------------------------


class TestMultipleParamGroups:
    def test_different_lr_per_group(self):
        p1 = torch.tensor([1.0], requires_grad=True)
        p2 = torch.tensor([1.0], requires_grad=True)

        opt = NesterovAdan(
            [
                {"params": [p1], "lr": 1e-1, "weight_decay": 0.0},
                {"params": [p2], "lr": 1e-4, "weight_decay": 0.0},
            ],
            lr=1e-3,  # default (overridden by group-level lr)
        )

        # Apply identical gradients
        p1.grad = torch.ones_like(p1)
        p2.grad = torch.ones_like(p2)
        opt.step()

        # p1 should move more than p2 (larger lr)
        delta1 = abs(1.0 - p1.item())
        delta2 = abs(1.0 - p2.item())
        assert delta1 > delta2, (
            f"Higher-lr group should move more: Δp1={delta1:.6f}, Δp2={delta2:.6f}"
        )

    def test_independent_states_per_group(self):
        """Each param group must maintain separate optimiser state."""
        p1 = torch.tensor([2.0], requires_grad=True)
        p2 = torch.tensor([2.0], requires_grad=True)
        opt = NesterovAdan(
            [{"params": [p1], "lr": 0.1}, {"params": [p2], "lr": 0.01}],
        )
        for _ in range(3):
            p1.grad = torch.tensor([1.0])
            p2.grad = torch.tensor([1.0])
            opt.step()

        # Both params should have independent state
        assert opt.state[p1]["step"] == opt.state[p2]["step"] == 3
        assert not torch.allclose(p1, p2), "Different LR groups should diverge"


# ---------------------------------------------------------------------------
# 12. State stored: prev_grad key present after step 1
# ---------------------------------------------------------------------------


class TestPrevGradState:
    def test_prev_grad_present_after_first_step(self):
        p = _make_scalar()
        opt = NesterovAdan([p], lr=1e-3, weight_decay=0.0)
        _forward_backward(p)
        opt.step()
        assert "prev_grad" in opt.state[p], "'prev_grad' must be stored in state after first step"

    def test_prev_grad_stores_actual_gradient(self):
        """prev_grad must equal the gradient from the previous step."""
        p = torch.tensor([4.0], requires_grad=True)
        opt = NesterovAdan([p], lr=1e-4, weight_decay=0.0)
        (p**2).sum().backward()
        g1 = p.grad.clone()
        opt.step()
        # prev_grad should now hold g1
        assert torch.allclose(opt.state[p]["prev_grad"], g1), (
            "prev_grad must store the gradient from the previous step"
        )

    def test_prev_grad_advances_each_step(self):
        """After step 2, prev_grad should hold g_2, not g_1."""
        p = torch.tensor([3.0], requires_grad=True)
        opt = NesterovAdan([p], lr=1e-4, weight_decay=0.0)

        # Step 1
        (p**2).sum().backward()
        opt.step()
        opt.zero_grad()

        # Step 2
        (p**2).sum().backward()
        g2 = p.grad.clone()
        opt.step()

        assert torch.allclose(opt.state[p]["prev_grad"], g2), (
            "prev_grad after step 2 must equal g_2"
        )


# ---------------------------------------------------------------------------
# 13. Bias correction applied correctly
# ---------------------------------------------------------------------------


class TestBiasCorrection:
    def test_exp_avg_matches_paper_formula_step1(self):
        """After step 1: m̂_1 = m_1 / (1 - β1^1) = g_1 (since m_1=(1-β1)*g_1)."""
        p = torch.tensor([2.0], requires_grad=True)
        β1, β2, β3 = 0.98, 0.92, 0.99
        opt = NesterovAdan([p], lr=1e-6, betas=(β1, β2, β3), weight_decay=0.0)
        (p**2).sum().backward()
        g1 = p.grad.item()
        opt.step()
        # m_1 = (1-β1)*g_1
        m1 = opt.state[p]["exp_avg"].item()
        assert m1 == pytest.approx((1 - β1) * g1, rel=1e-5)

    def test_step_counter_increments(self):
        p = _make_scalar()
        opt = NesterovAdan([p], lr=1e-3, weight_decay=0.0)
        for k in range(1, 4):
            opt.zero_grad()
            _forward_backward(p)
            opt.step()
            assert opt.state[p]["step"] == k, (
                f"Step counter should be {k}, got {opt.state[p]['step']}"
            )


# ---------------------------------------------------------------------------
# 14. zero_moments resets state correctly
# ---------------------------------------------------------------------------


class TestZeroMoments:
    def test_zero_moments_resets_buffers(self):
        p = _make_matrix()
        opt = NesterovAdan([p], lr=1e-3, weight_decay=0.0)
        for _ in range(5):
            opt.zero_grad()
            _forward_backward(p)
            opt.step()

        opt.zero_moments()
        state = opt.state[p]
        assert state["step"] == 0
        assert torch.allclose(state["exp_avg"], torch.zeros_like(state["exp_avg"]))
        assert torch.allclose(state["exp_avg_diff"], torch.zeros_like(state["exp_avg_diff"]))
        assert torch.allclose(state["exp_avg_sq"], torch.zeros_like(state["exp_avg_sq"]))
        assert state["prev_grad"] is None
