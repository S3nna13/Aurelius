"""Integration test for FlashMLAAttention.

Covers the full lifecycle:
  - Build from config
  - Forward in both modes
  - Numerical equivalence between modes
  - absorb_projections()
  - kv_cache_size_ratio < 1.0
  - Backward pass
  - Registry entry
"""
from __future__ import annotations

import torch
import pytest

from src.model.flash_mla import FlashMLAAttention, FlashMLAConfig
from src.model import MODEL_COMPONENT_REGISTRY


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

INTEGRATION_CFG = FlashMLAConfig(
    d_model=64,
    n_heads=4,
    head_dim=16,
    kv_lrank=16,
    q_lrank=32,
    rope_dim=8,
    dropout=0.0,
)

B, T = 2, 8


@pytest.fixture(scope="module")
def model_and_input():
    """Return a fresh model (eval mode) and a fixed input tensor."""
    torch.manual_seed(99)
    model = FlashMLAAttention(INTEGRATION_CFG).eval()
    x = torch.randn(B, T, INTEGRATION_CFG.d_model)
    return model, x


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestFlashMLAIntegration:
    """End-to-end integration tests for FlashMLAAttention."""

    def test_build_from_config(self, model_and_input):
        """Module builds without error from FlashMLAConfig."""
        model, _ = model_and_input
        assert isinstance(model, FlashMLAAttention)

    def test_forward_standard_mode(self, model_and_input):
        """Standard path (use_absorbed=False) produces correct-shaped finite output."""
        model, x = model_and_input
        with torch.no_grad():
            out = model(x, use_absorbed=False)
        assert out.shape == (B, T, INTEGRATION_CFG.d_model)
        assert torch.isfinite(out).all()

    def test_forward_absorbed_mode(self, model_and_input):
        """Absorbed path (use_absorbed=True) produces correct-shaped finite output."""
        model, x = model_and_input
        model.absorb_projections()
        with torch.no_grad():
            out = model(x, use_absorbed=True)
        assert out.shape == (B, T, INTEGRATION_CFG.d_model)
        assert torch.isfinite(out).all()

    def test_both_modes_match(self, model_and_input):
        """Both forward paths produce identical outputs to tolerance 1e-4."""
        model, x = model_and_input
        model.absorb_projections()
        with torch.no_grad():
            out_std = model(x, use_absorbed=False)
            out_abs = model(x, use_absorbed=True)
        max_diff = (out_std - out_abs).abs().max().item()
        assert torch.allclose(out_std, out_abs, rtol=1e-4, atol=1e-5), (
            f"Paths diverge: max |diff| = {max_diff:.6f}"
        )

    def test_absorb_projections(self, model_and_input):
        """absorb_projections() stores a non-None buffer of correct shape.

        absorbed_qk is stored per-head: [n_heads, d_model, kv_lrank].
        Each head h holds W_q_h^T @ W_k_h which maps d_model -> kv_lrank.
        """
        model, _ = model_and_input
        model.absorb_projections()
        assert model.absorbed_qk is not None
        assert model.absorbed_qk.shape == (
            INTEGRATION_CFG.n_heads,
            INTEGRATION_CFG.d_model,
            INTEGRATION_CFG.kv_lrank,
        )

    def test_kv_cache_ratio_lt_one(self, model_and_input):
        """kv_cache_size_ratio() < 1.0 confirms genuine compression."""
        model, _ = model_and_input
        ratio = model.kv_cache_size_ratio()
        assert ratio < 1.0, f"Expected ratio < 1.0, got {ratio}"
        assert abs(ratio - 0.25) < 1e-9

    def test_backward_pass(self):
        """Gradients flow through the standard path during training."""
        model = FlashMLAAttention(INTEGRATION_CFG).train()
        torch.manual_seed(5)
        x = torch.randn(B, T, INTEGRATION_CFG.d_model, requires_grad=True)
        out = model(x, use_absorbed=False)
        loss = out.mean()
        loss.backward()
        assert x.grad is not None, "Input gradient is None"
        assert torch.isfinite(x.grad).all(), "Input gradient contains non-finite values"
        params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
        assert params_with_grad > 0, "No parameter received a gradient"

    def test_registry_entry(self):
        """MODEL_COMPONENT_REGISTRY['flash_mla'] resolves to FlashMLAAttention."""
        assert "flash_mla" in MODEL_COMPONENT_REGISTRY
        cls = MODEL_COMPONENT_REGISTRY["flash_mla"]
        assert cls is FlashMLAAttention
        instance = cls(INTEGRATION_CFG)
        assert isinstance(instance, FlashMLAAttention)
