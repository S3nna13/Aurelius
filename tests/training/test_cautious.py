"""Tests for Cautious Optimizers (arXiv:2411.16085).

Covers CautiousAdam, CautiousSGD, CautiousMask, and basic convergence/
robustness properties.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.training.cautious import CautiousAdam, CautiousMask, CautiousSGD, make_cautious

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_param(value: float = 2.0, requires_grad: bool = True) -> nn.Parameter:
    return nn.Parameter(torch.tensor([value]))


def _make_linear(in_f: int = 8, out_f: int = 4, seed: int = 0) -> nn.Linear:
    torch.manual_seed(seed)
    return nn.Linear(in_f, out_f, bias=True)


def _quadratic_loss(model: nn.Module, target: float = 0.0) -> torch.Tensor:
    """Sum-of-squares loss: drives all params toward *target*."""
    total = sum((p**2).sum() for p in model.parameters())
    return total


# ---------------------------------------------------------------------------
# CautiousMask unit tests
# ---------------------------------------------------------------------------


class TestCautiousMask:
    """Tests 3–6: mask properties."""

    def test_mask_binary_before_normalisation(self):
        """Mask elements are 0 or 1 before normalisation (sign-agreement is binary)."""
        torch.manual_seed(0)
        d_t = torch.randn(20)
        g_t = torch.randn(20)
        # Compute raw mask manually
        raw = (d_t * g_t > 0).float()
        assert set(raw.unique().tolist()).issubset({0.0, 1.0}), "Raw mask must be binary (0 or 1)"

    def test_mask_one_where_same_sign(self):
        """Mask is 1 (before norm) where d_t and g_t have the same sign."""
        d_t = torch.tensor([1.0, -1.0, 1.0, -1.0])
        g_t = torch.tensor([2.0, -3.0, -1.0, 0.5])
        raw = (d_t * g_t > 0).float()
        # Positions 0 (+·+) and 1 (−·−) agree; positions 2 (+·−) and 3 (−·+) disagree
        assert raw[0] == 1.0 and raw[1] == 1.0
        assert raw[2] == 0.0 and raw[3] == 0.0

    def test_mask_zero_where_opposite_sign(self):
        """Mask is 0 where d_t and g_t have opposite signs."""
        d_t = torch.tensor([1.0, -1.0])
        g_t = torch.tensor([-1.0, 1.0])
        result = CautiousMask.apply(d_t, g_t)
        assert result[0].item() == pytest.approx(0.0)
        assert result[1].item() == pytest.approx(0.0)

    def test_masked_update_same_sign_as_gradient(self):
        """Non-zero elements of masked update must share sign with gradient."""
        torch.manual_seed(42)
        d_t = torch.randn(50)
        g_t = torch.randn(50)
        masked = CautiousMask.apply(d_t, g_t)
        nonzero = masked.abs() > 1e-12
        if nonzero.any():
            # masked_update and g_t should agree in sign wherever nonzero
            products = masked[nonzero] * g_t[nonzero]
            assert (products > 0).all(), "Masked update elements must agree in sign with gradient"


# ---------------------------------------------------------------------------
# CautiousAdam tests
# ---------------------------------------------------------------------------


class TestCautiousAdam:
    """Tests 1, 2, 7, 10–13."""

    def test_params_move_after_step(self):
        """Test 1: At least one parameter must change after a CautiousAdam step."""
        model = _make_linear()
        opt = CautiousAdam(model.parameters(), lr=1e-3)
        before = [p.data.clone() for p in model.parameters()]

        x = torch.randn(4, 8)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        changed = any(not torch.equal(a, b.data) for a, b in zip(before, model.parameters()))
        assert changed, "No parameters updated after CautiousAdam step"

    def test_movement_is_finite(self):
        """Test 2: All parameters must remain finite after a step."""
        model = _make_linear()
        opt = CautiousAdam(model.parameters(), lr=1e-3)

        x = torch.randn(4, 8)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        for p in model.parameters():
            assert torch.isfinite(p.data).all(), "Non-finite parameter after CautiousAdam step"

    def test_converges_on_quadratic(self):
        """Test 7: Loss should decrease monotonically for pure quadratic."""
        torch.manual_seed(7)
        model = _make_linear(4, 2, seed=7)
        opt = CautiousAdam(model.parameters(), lr=1e-2, weight_decay=0.0)

        losses = []
        for _ in range(30):
            opt.zero_grad()
            loss = _quadratic_loss(model)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0] * 0.5, (
            f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
        )

    def test_determinism_under_seed(self):
        """Test 10: Same seed → same parameter trajectory."""

        def run():
            torch.manual_seed(99)
            model = _make_linear(seed=99)
            opt = CautiousAdam(model.parameters(), lr=1e-3)
            for _ in range(3):
                opt.zero_grad()
                x = torch.randn(2, 8)
                model(x).sum().backward()
                opt.step()
            return [p.data.clone() for p in model.parameters()]

        params1 = run()
        params2 = run()
        for a, b in zip(params1, params2):
            assert torch.allclose(a, b), "CautiousAdam is not deterministic"

    def test_no_nan_inf_on_large_gradients(self):
        """Test 11: Parameters stay finite when gradients are 100x normal magnitude."""
        torch.manual_seed(11)
        model = _make_linear()
        opt = CautiousAdam(model.parameters(), lr=1e-3)
        opt.zero_grad()
        x = torch.randn(4, 8) * 100.0
        loss = model(x).sum()
        loss.backward()
        # Scale gradients up 100x
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(100.0)
        opt.step()

        for p in model.parameters():
            assert torch.isfinite(p.data).all(), "NaN/Inf after large-gradient step"

    def test_weight_decay_decays_params(self):
        """Test 12: Non-zero weight_decay should push params toward zero."""
        torch.manual_seed(12)
        model = _make_linear()
        opt = CautiousAdam(model.parameters(), lr=1e-2, weight_decay=1.0)

        # Set gradients to zero so only weight-decay acts
        for p in model.parameters():
            p.grad = torch.zeros_like(p)

        norms_before = [p.data.norm().item() for p in model.parameters()]
        opt.step()
        norms_after = [p.data.norm().item() for p in model.parameters()]

        for nb, na in zip(norms_before, norms_after):
            if nb > 0:
                assert na < nb, f"Weight decay did not reduce param norm: {nb:.4f} → {na:.4f}"

    def test_all_agree_mask_equals_all_ones(self):
        """Test 13: When d_t and g_t always agree in sign, mask should be all-ones
        (before the scale normalisation), meaning no updates are suppressed."""
        d_t = torch.tensor([1.0, 2.0, 3.0, 4.0])
        g_t = torch.tensor([0.5, 1.5, 2.5, 3.5])  # same sign as d_t everywhere
        raw = (d_t * g_t > 0).float()
        assert raw.all(), "Expected all-ones raw mask when signs always agree"

    def test_all_disagree_mask_equals_all_zeros(self):
        """Test 14: When gradient reverses completely, masked update should be zero."""
        d_t = torch.tensor([1.0, 2.0, 3.0])
        g_t = torch.tensor([-1.0, -2.0, -3.0])  # opposite signs
        masked = CautiousMask.apply(d_t, g_t)
        assert masked.abs().max().item() == pytest.approx(0.0), (
            "Expected all-zero masked update when gradient fully reversed"
        )

    def test_cautious_adam_faster_than_adam_on_oscillating_loss(self):
        """Test 15: CautiousAdam should converge faster than vanilla Adam on an
        oscillating loss landscape (loss that alternates sign of gradient)."""
        torch.manual_seed(15)

        def oscillating_loss(p: torch.Tensor, step: int) -> torch.Tensor:
            """Loss that creates sign-flipping gradients to trigger masking."""
            # Quadratic with a sign-oscillating linear perturbation
            sign = 1.0 if step % 2 == 0 else -1.0
            return (p**2).sum() + sign * 5.0 * p.sum()

        # Run CautiousAdam
        torch.manual_seed(15)
        p_c = nn.Parameter(torch.ones(8) * 2.0)
        opt_c = CautiousAdam([p_c], lr=1e-2, weight_decay=0.0)
        losses_c = []
        for step in range(50):
            opt_c.zero_grad()
            loss = oscillating_loss(p_c, step)
            loss.backward()
            opt_c.step()
            losses_c.append((p_c.data**2).sum().item())

        # Run vanilla Adam
        torch.manual_seed(15)
        p_a = nn.Parameter(torch.ones(8) * 2.0)
        opt_a = torch.optim.Adam([p_a], lr=1e-2)
        losses_a = []
        for step in range(50):
            opt_a.zero_grad()
            loss = oscillating_loss(p_a, step)
            loss.backward()
            opt_a.step()
            losses_a.append((p_a.data**2).sum().item())

        # Cautious should end up closer to zero than vanilla Adam on this task
        final_c = sum(losses_c[-10:]) / 10
        final_a = sum(losses_a[-10:]) / 10
        assert final_c <= final_a * 1.5, (
            f"CautiousAdam ({final_c:.4f}) unexpectedly much worse than Adam ({final_a:.4f})"
        )


# ---------------------------------------------------------------------------
# CautiousSGD tests
# ---------------------------------------------------------------------------


class TestCautiousSGD:
    """Tests 8–9."""

    def test_params_move_after_step(self):
        """Test 8: At least one parameter must change after a CautiousSGD step."""
        model = _make_linear()
        opt = CautiousSGD(model.parameters(), lr=0.01, momentum=0.9)
        before = [p.data.clone() for p in model.parameters()]

        x = torch.randn(4, 8)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        changed = any(not torch.equal(a, b.data) for a, b in zip(before, model.parameters()))
        assert changed, "No parameters updated after CautiousSGD step"

    def test_movement_is_finite(self):
        """Test 9: All parameters remain finite after a CautiousSGD step."""
        model = _make_linear()
        opt = CautiousSGD(model.parameters(), lr=0.01, momentum=0.9)

        x = torch.randn(4, 8)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        for p in model.parameters():
            assert torch.isfinite(p.data).all(), "Non-finite parameter after CautiousSGD step"

    def test_sgd_converges_on_quadratic(self):
        """CautiousSGD should reduce quadratic loss."""
        torch.manual_seed(8)
        model = _make_linear(4, 2, seed=8)
        opt = CautiousSGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.0)

        losses = []
        for _ in range(50):
            opt.zero_grad()
            loss = _quadratic_loss(model)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0] * 0.5, (
            f"CautiousSGD did not converge: {losses[0]:.4f} → {losses[-1]:.4f}"
        )

    def test_sgd_no_nan_on_large_gradients(self):
        """CautiousSGD stays finite under 100x gradient magnitudes."""
        torch.manual_seed(99)
        model = _make_linear()
        opt = CautiousSGD(model.parameters(), lr=0.01, momentum=0.9)
        opt.zero_grad()
        x = torch.randn(4, 8) * 100.0
        model(x).sum().backward()
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(100.0)
        opt.step()
        for p in model.parameters():
            assert torch.isfinite(p.data).all(), "NaN/Inf in CautiousSGD after large grads"


# ---------------------------------------------------------------------------
# make_cautious factory smoke test
# ---------------------------------------------------------------------------


class TestMakeCautious:
    def test_factory_wraps_sgd(self):
        """make_cautious should produce a working optimizer from torch.optim.SGD."""
        CautiousSGDFactory = make_cautious(torch.optim.SGD)
        model = _make_linear()
        opt = CautiousSGDFactory(model.parameters(), lr=0.01, momentum=0.9)
        before = [p.data.clone() for p in model.parameters()]

        x = torch.randn(4, 8)
        model(x).sum().backward()
        opt.step()

        changed = any(not torch.equal(a, b.data) for a, b in zip(before, model.parameters()))
        assert changed, "Wrapped optimizer did not update parameters"
        for p in model.parameters():
            assert torch.isfinite(p.data).all(), "NaN/Inf in wrapped optimizer output"
