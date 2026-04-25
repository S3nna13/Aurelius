"""Tests for int8_dynamic_quant — INT8 dynamic per-tensor quantization."""
from __future__ import annotations

import torch

from src.quantization.int8_dynamic_quant import Int8DynamicQuantizer


class TestInt8DynamicQuantizer:
    def test_quantize_roundtrip_2d(self):
        x = torch.randn(16, 32)
        quantizer = Int8DynamicQuantizer()
        q, scale, zp = quantizer.quantize(x)
        x_hat = quantizer.dequantize(q, scale, zp, x.shape)
        assert x.shape == x_hat.shape
        mse = ((x.float() - x_hat.float()) ** 2).mean()
        assert mse < 5.0

    def test_quantize_zero_tensor(self):
        x = torch.zeros(8, 8)
        quantizer = Int8DynamicQuantizer()
        q, scale, zp = quantizer.quantize(x)
        x_hat = quantizer.dequantize(q, scale, zp, x.shape)
        assert (x_hat == 0).all()

    def test_quantize_constant_tensor(self):
        x = torch.ones(4, 4) * 3.14
        quantizer = Int8DynamicQuantizer()
        q, scale, zp = quantizer.quantize(x)
        x_hat = quantizer.dequantize(q, scale, zp, x.shape)
        assert (x_hat - 3.14).abs().mean() < 0.02

    def test_symmetric_quantize(self):
        x = torch.randn(8, 16)
        quantizer = Int8DynamicQuantizer(symmetric=True)
        q, scale, zp = quantizer.quantize(x)
        x_hat = quantizer.dequantize(q, scale, zp, x.shape)
        assert x.shape == x_hat.shape

    def test_per_channel_quantize(self):
        x = torch.randn(4, 32)
        quantizer = Int8DynamicQuantizer(per_channel=True)
        q, scale, zp = quantizer.quantize(x)
        assert scale.numel() == x.shape[0]
        x_hat = quantizer.dequantize(q, scale, zp, x.shape)
        assert x.shape == x_hat.shape

    def test_negative_values_roundtrip(self):
        x = torch.tensor([[-10.0, -5.0, 0.0, 5.0, 10.0]])
        quantizer = Int8DynamicQuantizer()
        q, scale, zp = quantizer.quantize(x)
        x_hat = quantizer.dequantize(q, scale, zp, x.shape)
        # Ordering preserved: each value should be >= previous
        assert (x_hat[0, :-1] <= x_hat[0, 1:] + 0.01).all()

    def test_1d_tensor(self):
        x = torch.randn(64)
        quantizer = Int8DynamicQuantizer()
        q, scale, zp = quantizer.quantize(x)
        x_hat = quantizer.dequantize(q, scale, zp, x.shape)
        assert x.shape == x_hat.shape
