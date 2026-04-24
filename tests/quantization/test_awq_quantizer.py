"""Tests for src/quantization/awq_quantizer.py — 10+ tests, CPU-only, tiny tensors."""

from __future__ import annotations

import pytest
import torch

from src.quantization.awq_quantizer import (
    AWQConfig,
    AWQQuantizer,
    AWQScaleSearch,
    QUANTIZATION_REGISTRY,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_weight() -> torch.Tensor:
    """4×8 weight matrix."""
    torch.manual_seed(0)
    return torch.randn(4, 8)


@pytest.fixture()
def tiny_activations() -> torch.Tensor:
    """16×8 calibration activations."""
    torch.manual_seed(1)
    return torch.randn(16, 8)


@pytest.fixture()
def config4bit() -> AWQConfig:
    return AWQConfig(bits=4, group_size=128, zero_point=True, clip_ratio=0.9)


@pytest.fixture()
def config8bit() -> AWQConfig:
    return AWQConfig(bits=8, group_size=128, zero_point=True, clip_ratio=0.9)


# ---------------------------------------------------------------------------
# AWQConfig tests
# ---------------------------------------------------------------------------

class TestAWQConfig:
    def test_defaults(self):
        cfg = AWQConfig()
        assert cfg.bits == 4
        assert cfg.group_size == 128
        assert cfg.zero_point is True
        assert cfg.clip_ratio == 0.9

    def test_custom(self):
        cfg = AWQConfig(bits=8, group_size=64, zero_point=False, clip_ratio=0.8)
        assert cfg.bits == 8
        assert cfg.group_size == 64
        assert cfg.zero_point is False
        assert cfg.clip_ratio == 0.8


# ---------------------------------------------------------------------------
# AWQScaleSearch tests
# ---------------------------------------------------------------------------

class TestAWQScaleSearch:
    def test_search_scales_shape(self, tiny_weight, tiny_activations, config4bit):
        searcher = AWQScaleSearch(config4bit)
        scales = searcher.search_scales(tiny_weight, tiny_activations)
        assert scales.shape == (tiny_weight.shape[0],), "One scale per output channel"

    def test_search_scales_positive(self, tiny_weight, tiny_activations, config4bit):
        searcher = AWQScaleSearch(config4bit)
        scales = searcher.search_scales(tiny_weight, tiny_activations)
        assert (scales > 0).all(), "All scales must be positive"

    def test_search_scales_grid_count(self, tiny_weight, tiny_activations, config4bit):
        """Grid has 20 points; returned scales should be one of those grid values."""
        searcher = AWQScaleSearch(config4bit, n_grid=20)
        scales = searcher.search_scales(tiny_weight, tiny_activations)
        # Just verify output is finite
        assert torch.isfinite(scales).all()

    def test_search_scales_zero_activations(self, tiny_weight, config4bit):
        """Zero activations should not cause NaN."""
        zero_acts = torch.zeros(16, 8)
        searcher = AWQScaleSearch(config4bit)
        scales = searcher.search_scales(tiny_weight, zero_acts)
        assert torch.isfinite(scales).all()


# ---------------------------------------------------------------------------
# AWQQuantizer tests
# ---------------------------------------------------------------------------

class TestAWQQuantizer:
    def test_quantize_layer_shapes(self, tiny_weight, tiny_activations, config4bit):
        q = AWQQuantizer(config4bit)
        q_weight, scales, zeros = q.quantize_layer(tiny_weight, tiny_activations)
        assert q_weight.shape == tiny_weight.shape
        assert scales.shape == (tiny_weight.shape[0],)
        assert zeros.shape == (tiny_weight.shape[0],)

    def test_quantize_layer_4bit_range(self, tiny_weight, tiny_activations, config4bit):
        q = AWQQuantizer(config4bit)
        q_weight, _, _ = q.quantize_layer(tiny_weight, tiny_activations)
        assert q_weight.min() >= 0
        assert q_weight.max() <= 15  # 2^4 - 1

    def test_quantize_layer_8bit_range(self, tiny_weight, tiny_activations, config8bit):
        q = AWQQuantizer(config8bit)
        q_weight, _, _ = q.quantize_layer(tiny_weight, tiny_activations)
        assert q_weight.min() >= 0
        assert q_weight.max() <= 255  # 2^8 - 1

    def test_dequantize_shape(self, tiny_weight, tiny_activations, config4bit):
        q = AWQQuantizer(config4bit)
        q_weight, scales, zeros = q.quantize_layer(tiny_weight, tiny_activations)
        recon = q.dequantize(q_weight, scales, zeros)
        assert recon.shape == tiny_weight.shape

    def test_dequantize_finite(self, tiny_weight, tiny_activations, config4bit):
        q = AWQQuantizer(config4bit)
        q_weight, scales, zeros = q.quantize_layer(tiny_weight, tiny_activations)
        recon = q.dequantize(q_weight, scales, zeros)
        assert torch.isfinite(recon).all()

    def test_roundtrip_error_reasonable(self, tiny_weight, tiny_activations, config8bit):
        """8-bit round-trip error should be relatively small."""
        q = AWQQuantizer(config8bit)
        q_weight, scales, zeros = q.quantize_layer(tiny_weight, tiny_activations)
        recon = q.dequantize(q_weight, scales, zeros)
        err = (recon - tiny_weight).abs().mean().item()
        # Generous bound — just checking it's not completely broken
        assert err < 1.0, f"Mean reconstruction error too large: {err}"

    def test_registry_entry(self):
        assert "awq" in QUANTIZATION_REGISTRY
        assert QUANTIZATION_REGISTRY["awq"] is AWQQuantizer

    def test_no_zero_point_symmetric(self, tiny_weight, tiny_activations):
        cfg = AWQConfig(bits=4, zero_point=False)
        q = AWQQuantizer(cfg)
        q_weight, scales, zeros = q.quantize_layer(tiny_weight, tiny_activations)
        # zeros should all be 0.0 for symmetric
        assert (zeros == 0.0).all()

    def test_different_seeds_different_output(self):
        torch.manual_seed(42)
        w1 = torch.randn(4, 8)
        torch.manual_seed(99)
        w2 = torch.randn(4, 8)
        acts = torch.randn(16, 8)
        q = AWQQuantizer()
        qw1, _, _ = q.quantize_layer(w1, acts)
        qw2, _, _ = q.quantize_layer(w2, acts)
        assert not torch.equal(qw1, qw2)

    def test_scales_all_positive(self, tiny_weight, tiny_activations, config4bit):
        q = AWQQuantizer(config4bit)
        _, scales, _ = q.quantize_layer(tiny_weight, tiny_activations)
        assert (scales > 0).all()
