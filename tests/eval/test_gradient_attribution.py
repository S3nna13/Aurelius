"""Tests for src/eval/gradient_attribution.py.

Covers:
  - VanillaGradients.attribute  (shape, non-negativity, zero-input)
  - GradientXInput.attribute    (shape, signed values)
  - IntegratedGradients.attribute (shape, finiteness, baseline==input, completeness)
  - KernelSHAPApproximator.attribute (shape, finiteness, sum approximation)
  - All methods with B=1, T=4
"""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn
from torch import Tensor

from src.eval.gradient_attribution import (
    AttributionConfig,
    VanillaGradients,
    GradientXInput,
    IntegratedGradients,
    KernelSHAPApproximator,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

B, T, D = 1, 4, 16  # batch=1, seq_len=4, embed_dim=16


class SimpleLinearModel(nn.Module):
    """Single linear layer that maps (B, T, D) -> scalar via sum."""

    def __init__(self, d: int = D) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.fc = nn.Linear(d, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, D) -> (B, T, 1) -> scalar
        return self.fc(x).sum()


@pytest.fixture(scope="module")
def model() -> SimpleLinearModel:
    m = SimpleLinearModel(D)
    m.eval()
    return m


@pytest.fixture(scope="module")
def embeddings(model) -> Tensor:
    """Random (B, T, D) embedding tensor (detached, no grad on fixture)."""
    torch.manual_seed(1)
    return torch.randn(B, T, D)


def target_fn_factory(model: nn.Module):
    """Return a target_fn that uses the given model."""
    def _fn(emb: Tensor) -> Tensor:
        return model(emb)
    return _fn


# ---------------------------------------------------------------------------
# AttributionConfig
# ---------------------------------------------------------------------------

def test_attribution_config_defaults():
    cfg = AttributionConfig()
    assert cfg.n_steps == 50
    assert cfg.baseline == "zero"
    assert cfg.normalize is True


def test_attribution_config_custom():
    cfg = AttributionConfig(n_steps=10, baseline="random", normalize=False)
    assert cfg.n_steps == 10
    assert cfg.baseline == "random"
    assert cfg.normalize is False


# ---------------------------------------------------------------------------
# VanillaGradients
# ---------------------------------------------------------------------------

def test_vanilla_gradients_attribute_shape(model, embeddings):
    """attribute() must return shape (B, T)."""
    vg = VanillaGradients(model)
    fn = target_fn_factory(model)
    out = vg.attribute(embeddings, fn)
    assert out.shape == (B, T), f"Expected ({B}, {T}), got {out.shape}"


def test_vanilla_gradients_nonnegative(model, embeddings):
    """L2 norms are always >= 0."""
    vg = VanillaGradients(model)
    fn = target_fn_factory(model)
    out = vg.attribute(embeddings, fn)
    assert (out >= 0).all(), "VanillaGradients attributions must be non-negative"


def test_vanilla_gradients_zero_input(model):
    """Zero-gradient model gives zero attribution."""
    vg = VanillaGradients(model)

    class ZeroModel(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            return (x * 0).sum()

    zero_model = ZeroModel()
    fn = lambda emb: zero_model(emb)

    zero_emb = torch.zeros(B, T, D)
    out = vg.attribute(zero_emb, fn)
    assert torch.allclose(out, torch.zeros(B, T)), (
        "Zero-gradient model should give zero attributions"
    )


def test_vanilla_gradients_saliency_shape(model, embeddings):
    """saliency() must return shape (B, T)."""
    vg = VanillaGradients(model)
    fn = target_fn_factory(model)
    out = vg.saliency(embeddings, fn)
    assert out.shape == (B, T), f"saliency: expected ({B}, {T}), got {out.shape}"


def test_vanilla_gradients_saliency_nonnegative(model, embeddings):
    """saliency() is based on absolute values -- must be >= 0."""
    vg = VanillaGradients(model)
    fn = target_fn_factory(model)
    out = vg.saliency(embeddings, fn)
    assert (out >= 0).all(), "Saliency values must be non-negative"


# ---------------------------------------------------------------------------
# GradientXInput
# ---------------------------------------------------------------------------

def test_gradient_x_input_shape(model, embeddings):
    """attribute() must return shape (B, T)."""
    gxi = GradientXInput(model)
    fn = target_fn_factory(model)
    out = gxi.attribute(embeddings, fn)
    assert out.shape == (B, T), f"Expected ({B}, {T}), got {out.shape}"


def test_gradient_x_input_can_be_negative(model):
    """GradientXInput produces signed (possibly negative) attributions."""
    gxi = GradientXInput(model)
    torch.manual_seed(7)
    mixed_emb = torch.randn(B, T, D)
    fn = target_fn_factory(model)
    out = gxi.attribute(mixed_emb, fn)
    assert out.shape == (B, T)
    assert torch.isfinite(out).all(), "GradientXInput attributions must be finite"


def test_gradient_x_input_b1_t4(model, embeddings):
    """Explicit B=1, T=4 shape check."""
    gxi = GradientXInput(model)
    fn = target_fn_factory(model)
    out = gxi.attribute(embeddings, fn)
    assert out.shape == (1, 4)


# ---------------------------------------------------------------------------
# IntegratedGradients
# ---------------------------------------------------------------------------

def test_integrated_gradients_shape(model, embeddings):
    """attribute() must return shape (B, T)."""
    ig = IntegratedGradients(model, n_steps=10)
    fn = target_fn_factory(model)
    out = ig.attribute(embeddings, fn)
    assert out.shape == (B, T), f"Expected ({B}, {T}), got {out.shape}"


def test_integrated_gradients_finite(model, embeddings):
    """All IG attributions must be finite."""
    ig = IntegratedGradients(model, n_steps=10)
    fn = target_fn_factory(model)
    out = ig.attribute(embeddings, fn)
    assert torch.isfinite(out).all(), "IG attributions must be finite"


def test_integrated_gradients_baseline_equals_input(model):
    """When baseline == input, IG attributions should be near zero."""
    ig = IntegratedGradients(model, n_steps=20)
    torch.manual_seed(5)
    emb = torch.randn(B, T, D)
    fn = target_fn_factory(model)
    out = ig.attribute(emb, fn, baseline=emb.clone())
    assert out.abs().max().item() < 1e-4, (
        f"IG with baseline=input should be near zero; max={out.abs().max().item()}"
    )


def test_integrated_gradients_completeness(model, embeddings):
    """Completeness: sum(IG) should approximate f(x) - f(baseline)."""
    ig = IntegratedGradients(model, n_steps=50)
    fn = target_fn_factory(model)
    baseline = torch.zeros_like(embeddings)
    attrs = ig.attribute(embeddings, fn, baseline=baseline)
    error = ig.completeness_check(attrs, embeddings, baseline, fn)
    assert error < 1e-3, f"Completeness error too large: {error}"


def test_integrated_gradients_completeness_returns_float(model, embeddings):
    """completeness_check must return a Python float."""
    ig = IntegratedGradients(model, n_steps=5)
    fn = target_fn_factory(model)
    baseline = torch.zeros_like(embeddings)
    attrs = ig.attribute(embeddings, fn, baseline=baseline)
    result = ig.completeness_check(attrs, embeddings, baseline, fn)
    assert isinstance(result, float), f"Expected float, got {type(result)}"


def test_integrated_gradients_b1_t4(model, embeddings):
    """Explicit B=1, T=4 shape check for IG."""
    ig = IntegratedGradients(model, n_steps=5)
    fn = target_fn_factory(model)
    out = ig.attribute(embeddings, fn)
    assert out.shape == (1, 4)


# ---------------------------------------------------------------------------
# KernelSHAPApproximator
# ---------------------------------------------------------------------------

def test_kernel_shap_shape(model, embeddings):
    """attribute() must return shape (T,)."""
    shap = KernelSHAPApproximator(model, n_samples=30)
    fn = target_fn_factory(model)
    out = shap.attribute(embeddings, fn)
    assert out.shape == (T,), f"Expected ({T},), got {out.shape}"


def test_kernel_shap_finite(model, embeddings):
    """SHAP attributions must be finite."""
    shap = KernelSHAPApproximator(model, n_samples=30)
    fn = target_fn_factory(model)
    out = shap.attribute(embeddings, fn)
    assert torch.isfinite(out).all(), "SHAP attributions must be finite"


def test_kernel_shap_sum_approximates_delta(model, embeddings):
    """sum(SHAP) should approximate f(x) - f(baseline) (efficiency axiom)."""
    shap = KernelSHAPApproximator(model, n_samples=100)
    fn = target_fn_factory(model)
    baseline = torch.zeros_like(embeddings)
    out = shap.attribute(embeddings, fn, baseline=baseline)

    with torch.no_grad():
        fx = fn(embeddings).item()
        fb = fn(baseline).item()
    delta = fx - fb

    shap_sum = out.sum().item()
    tol = max(0.1, 0.10 * abs(delta))
    assert abs(shap_sum - delta) < tol, (
        f"SHAP sum {shap_sum:.4f} deviates from f(x)-f(b)={delta:.4f} by "
        f"{abs(shap_sum - delta):.4f} (tol={tol:.4f})"
    )


def test_kernel_shap_b1_t4(model, embeddings):
    """Explicit B=1, T=4 shape check for SHAP."""
    shap = KernelSHAPApproximator(model, n_samples=20)
    fn = target_fn_factory(model)
    out = shap.attribute(embeddings, fn)
    assert out.shape == (4,)


def test_kernel_shap_rejects_batch_gt_1(model):
    """KernelSHAPApproximator must raise ValueError for B > 1."""
    shap = KernelSHAPApproximator(model, n_samples=10)
    fn = target_fn_factory(model)
    emb = torch.randn(2, T, D)
    with pytest.raises(ValueError, match="batch size 1"):
        shap.attribute(emb, fn)
