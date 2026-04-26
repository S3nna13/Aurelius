"""Tests for nf4_quantizer — 4-bit NormalFloat quantization."""

from __future__ import annotations

import torch

from src.quantization.nf4_quantizer import NF4Quantizer, nf4_dequantize, nf4_quantize


class TestNF4Quantizer:
    def test_quantize_roundtrip_preserves_approx_value(self):
        x = torch.randn(32, 64)
        q, scale = nf4_quantize(x)
        x_hat = nf4_dequantize(q, scale, x.shape)
        assert x.shape == x_hat.shape
        mse = ((x - x_hat) ** 2).mean()
        assert mse < 1.0

    def test_quantize_zero_tensor(self):
        x = torch.zeros(16, 16)
        q, scale = nf4_quantize(x)
        x_hat = nf4_dequantize(q, scale, x.shape)
        assert (x_hat == 0).all()

    def test_quantize_updates_absmax(self):
        x = torch.tensor([[1.0, -2.0, 3.0]])
        q, scale = nf4_quantize(x)
        assert scale > 0

    def test_dequantize_wrong_shape_raises(self):
        import pytest

        q = torch.zeros(10, dtype=torch.uint8)
        with pytest.raises(ValueError, match="shape"):
            nf4_dequantize(q, 1.0, torch.Size((5, 5)))

    def test_class_based_roundtrip(self):
        x = torch.randn(8, 16)
        quantizer = NF4Quantizer()
        q, scale = quantizer.quantize(x)
        x_hat = quantizer.dequantize(q, scale, x.shape)
        assert x.shape == x_hat.shape

    def test_different_scales(self):
        x_small = torch.randn(16) * 0.01
        x_large = torch.randn(16) * 10.0
        _, s1 = nf4_quantize(x_small)
        _, s2 = nf4_quantize(x_large)
        assert s2 > s1

    def test_quantile_based_levels(self):
        quantizer = NF4Quantizer()
        levels = quantizer._nf4_levels
        assert len(levels) == 16
        assert levels[0] < 0
        assert levels[-1] > 0
