"""Tests for Token Budget Predictor."""

from __future__ import annotations

import torch
from src.inference.token_budget_predictor import (
    AdaptiveStopCriteria,
    BudgetCalibrator,
    BudgetCategory,
    BudgetLoss,
    TokenBudgetPredictor,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

D_MODEL = 16
B = 4
MAX_BUDGET = 512


def _make_predictor() -> TokenBudgetPredictor:
    torch.manual_seed(42)
    return TokenBudgetPredictor(d_model=D_MODEL, max_budget=MAX_BUDGET, hidden_size=64)


def _make_hidden(batch: int = B) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(batch, D_MODEL)


# ---------------------------------------------------------------------------
# BudgetCategory tests
# ---------------------------------------------------------------------------


class TestBudgetCategory:
    def test_from_length_short(self):
        """length=10 → SHORT."""
        assert BudgetCategory.from_length(10) == BudgetCategory.SHORT

    def test_from_length_medium(self):
        """length=100 → MEDIUM."""
        assert BudgetCategory.from_length(100) == BudgetCategory.MEDIUM

    def test_from_length_long(self):
        """length=300 → LONG."""
        assert BudgetCategory.from_length(300) == BudgetCategory.LONG


# ---------------------------------------------------------------------------
# TokenBudgetPredictor tests
# ---------------------------------------------------------------------------


class TestTokenBudgetPredictor:
    def test_predict_budget_shape(self):
        """predict_budget output shape is (B,)."""
        predictor = _make_predictor()
        hidden = _make_hidden()
        out = predictor.predict_budget(hidden)
        assert out.shape == (B,), f"Expected shape ({B},), got {out.shape}"

    def test_predict_budget_in_range(self):
        """predict_budget values are in [1, max_budget]."""
        predictor = _make_predictor()
        hidden = _make_hidden()
        out = predictor.predict_budget(hidden)
        assert (out >= 1.0).all(), "Some predictions < 1"
        assert (out <= MAX_BUDGET).all(), f"Some predictions > {MAX_BUDGET}"

    def test_predict_budget_positive(self):
        """predict_budget output is strictly positive."""
        predictor = _make_predictor()
        hidden = _make_hidden()
        out = predictor.predict_budget(hidden)
        assert (out > 0).all(), "predict_budget returned non-positive values"

    def test_predict_category_shape(self):
        """predict_category output shape is (B,) long tensor."""
        predictor = _make_predictor()
        hidden = _make_hidden()
        out = predictor.predict_category(hidden)
        assert out.shape == (B,), f"Expected shape ({B},), got {out.shape}"
        assert out.dtype == torch.long, f"Expected long dtype, got {out.dtype}"

    def test_predict_category_valid_values(self):
        """predict_category values are in {0, 1, 2}."""
        predictor = _make_predictor()
        hidden = _make_hidden()
        out = predictor.predict_category(hidden)
        assert set(out.tolist()).issubset({0, 1, 2}), f"Unexpected category values: {out.tolist()}"


# ---------------------------------------------------------------------------
# BudgetLoss tests
# ---------------------------------------------------------------------------


class TestBudgetLoss:
    def _make_inputs(self):
        predictor = _make_predictor()
        hidden = _make_hidden()
        target_lengths = torch.tensor([30, 120, 250, 60], dtype=torch.long)
        return predictor, hidden, target_lengths

    def test_returns_expected_keys(self):
        """BudgetLoss returns dict with 'total', 'reg_loss', 'cls_loss'."""
        loss_fn = BudgetLoss()
        predictor, hidden, targets = self._make_inputs()
        result = loss_fn(hidden, predictor, targets)
        assert isinstance(result, dict)
        assert "total" in result
        assert "reg_loss" in result
        assert "cls_loss" in result

    def test_total_is_finite(self):
        """BudgetLoss total is finite."""
        loss_fn = BudgetLoss()
        predictor, hidden, targets = self._make_inputs()
        result = loss_fn(hidden, predictor, targets)
        assert torch.isfinite(result["total"]), "total loss is not finite"

    def test_total_combines_losses(self):
        """total == reg_weight * reg_loss + cls_weight * cls_loss."""
        reg_weight, cls_weight = 2.0, 0.5
        loss_fn = BudgetLoss(reg_weight=reg_weight, cls_weight=cls_weight)
        predictor, hidden, targets = self._make_inputs()
        result = loss_fn(hidden, predictor, targets)
        expected = reg_weight * result["reg_loss"] + cls_weight * result["cls_loss"]
        assert torch.isclose(result["total"], expected), (
            f"total={result['total'].item():.6f}, expected={expected.item():.6f}"
        )


# ---------------------------------------------------------------------------
# AdaptiveStopCriteria tests
# ---------------------------------------------------------------------------


class TestAdaptiveStopCriteria:
    def test_should_stop_false_at_zero(self):
        """should_stop returns False when n_generated=0."""
        predictor = _make_predictor()
        criteria = AdaptiveStopCriteria(predictor, slack_factor=1.2)
        hidden = _make_hidden(batch=1)
        assert criteria.should_stop(hidden, n_generated=0) is False

    def test_remaining_budget_non_negative(self):
        """remaining_budget is always >= 0."""
        predictor = _make_predictor()
        criteria = AdaptiveStopCriteria(predictor, slack_factor=1.2)
        hidden = _make_hidden(batch=1)
        # Test with a very large n_generated that would exceed the budget
        remaining = criteria.remaining_budget(hidden, n_generated=10_000)
        assert remaining >= 0, f"remaining_budget returned negative value: {remaining}"


# ---------------------------------------------------------------------------
# BudgetCalibrator tests
# ---------------------------------------------------------------------------


class TestBudgetCalibrator:
    def test_fit_and_calibrate_reduces_error(self):
        """After fitting, calibrated predictions have lower error than raw ones."""
        torch.manual_seed(7)
        # Simulate raw predictions with a systematic bias: true = 2*pred + 10
        predicted = torch.rand(50) * 100 + 50  # values in [50, 150]
        actual = 2.0 * predicted + 10.0 + torch.randn(50) * 2  # true relationship

        cal = BudgetCalibrator()
        cal.fit(predicted, actual)
        calibrated = cal.calibrate(predicted)

        raw_error = (predicted - actual).abs().mean().item()
        cal_error = (calibrated - actual).abs().mean().item()
        assert cal_error < raw_error, (
            f"Calibration did not reduce error: raw={raw_error:.4f}, cal={cal_error:.4f}"
        )

    def test_calibrate_returns_same_shape(self):
        """calibrate returns a tensor with the same shape as the input."""
        torch.manual_seed(3)
        predicted = torch.rand(20) * 200
        actual = predicted * 1.5 + 5.0

        cal = BudgetCalibrator()
        cal.fit(predicted, actual)
        result = cal.calibrate(predicted)

        assert result.shape == predicted.shape, (
            f"Shape mismatch: got {result.shape}, expected {predicted.shape}"
        )
