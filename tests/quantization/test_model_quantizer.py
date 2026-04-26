"""Tests for model_quantizer."""
from __future__ import annotations

import pytest
import torch

from src.quantization.model_quantizer import ModelQuantizer, MODEL_QUANTIZER_REGISTRY


class TestModelQuantizer:
    def test_quantize_dequantize_roundtrip(self):
        t = torch.randn(16, 32)
        quantizer = ModelQuantizer(bits=8, scheme="symmetric")
        q, scale, zp = quantizer.quantize_tensor(t)
        t_hat = quantizer.dequantize_tensor(q, scale, zp)
        assert t.shape == t_hat.shape
        mse = ((t - t_hat) ** 2).mean()
        assert mse < 1.0

    def test_symmetric_zero_point_is_zero(self):
        t = torch.randn(8, 8)
        quantizer = ModelQuantizer(bits=8, scheme="symmetric")
        q, scale, zp = quantizer.quantize_tensor(t)
        assert zp.item() == 0

    def test_asymmetric_non_zero_zero_point_for_mixed_sign(self):
        t = torch.tensor([-5.0, 0.0, 5.0])
        quantizer = ModelQuantizer(bits=8, scheme="asymmetric")
        q, scale, zp = quantizer.quantize_tensor(t)
        assert zp.item() != 0

    def test_bits_4_symmetric(self):
        t = torch.randn(8, 8)
        quantizer = ModelQuantizer(bits=4, scheme="symmetric")
        q, scale, zp = quantizer.quantize_tensor(t)
        t_hat = quantizer.dequantize_tensor(q, scale, zp)
        assert q.dtype == torch.int8
        assert t.shape == t_hat.shape
        mse = ((t - t_hat) ** 2).mean()
        assert mse < 5.0

    def test_bits_4_asymmetric(self):
        t = torch.randn(8, 8)
        quantizer = ModelQuantizer(bits=4, scheme="asymmetric")
        q, scale, zp = quantizer.quantize_tensor(t)
        t_hat = quantizer.dequantize_tensor(q, scale, zp)
        assert q.dtype == torch.int8
        assert t.shape == t_hat.shape
        mse = ((t - t_hat) ** 2).mean()
        assert mse < 5.0

    def test_bits_8_symmetric(self):
        t = torch.randn(8, 8)
        quantizer = ModelQuantizer(bits=8, scheme="symmetric")
        q, scale, zp = quantizer.quantize_tensor(t)
        t_hat = quantizer.dequantize_tensor(q, scale, zp)
        assert q.dtype == torch.int8
        assert t.shape == t_hat.shape
        mse = ((t - t_hat) ** 2).mean()
        assert mse < 1.0

    def test_bits_8_asymmetric(self):
        t = torch.randn(8, 8)
        quantizer = ModelQuantizer(bits=8, scheme="asymmetric")
        q, scale, zp = quantizer.quantize_tensor(t)
        t_hat = quantizer.dequantize_tensor(q, scale, zp)
        assert q.dtype == torch.uint8
        assert t.shape == t_hat.shape
        mse = ((t - t_hat) ** 2).mean()
        assert mse < 1.0

    def test_bits_16_symmetric(self):
        t = torch.randn(8, 8)
        quantizer = ModelQuantizer(bits=16, scheme="symmetric")
        q, scale, zp = quantizer.quantize_tensor(t)
        t_hat = quantizer.dequantize_tensor(q, scale, zp)
        assert q.dtype == torch.int16
        assert t.shape == t_hat.shape
        mse = ((t - t_hat) ** 2).mean()
        assert mse < 1e-3

    def test_bits_16_asymmetric(self):
        t = torch.randn(8, 8)
        quantizer = ModelQuantizer(bits=16, scheme="asymmetric")
        q, scale, zp = quantizer.quantize_tensor(t)
        t_hat = quantizer.dequantize_tensor(q, scale, zp)
        assert q.dtype == torch.int32
        assert t.shape == t_hat.shape
        mse = ((t - t_hat) ** 2).mean()
        assert mse < 1e-3

    def test_state_dict_roundtrip(self):
        state_dict = {
            "weight": torch.randn(16, 32),
            "bias": torch.randn(32),
        }
        quantizer = ModelQuantizer(bits=8, scheme="symmetric")
        q_state = quantizer.quantize_state_dict(state_dict)
        assert set(q_state.keys()) == set(state_dict.keys())
        for v in q_state.values():
            assert isinstance(v, tuple)
            assert len(v) == 3
        deq_state = quantizer.dequantize_state_dict(q_state)
        for k in state_dict:
            assert deq_state[k].shape == state_dict[k].shape
            mse = ((state_dict[k] - deq_state[k]) ** 2).mean()
            assert mse < 1.0

    def test_invalid_bits(self):
        with pytest.raises(ValueError, match="bits must be in"):
            ModelQuantizer(bits=2)

    def test_invalid_scheme(self):
        with pytest.raises(ValueError, match="scheme must be"):
            ModelQuantizer(bits=8, scheme="linear")

    def test_zero_tensor(self):
        t = torch.zeros(8, 8)
        quantizer = ModelQuantizer(bits=8, scheme="symmetric")
        q, scale, zp = quantizer.quantize_tensor(t)
        t_hat = quantizer.dequantize_tensor(q, scale, zp)
        assert (t_hat == 0).all()

    def test_constant_tensor(self):
        t = torch.ones(4, 4) * 3.14
        quantizer = ModelQuantizer(bits=8, scheme="symmetric")
        q, scale, zp = quantizer.quantize_tensor(t)
        t_hat = quantizer.dequantize_tensor(q, scale, zp)
        assert (t_hat - 3.14).abs().mean() < 0.02

    def test_negative_constant_tensor_asymmetric(self):
        t = torch.full((4, 4), -2.5)
        quantizer = ModelQuantizer(bits=8, scheme="asymmetric")
        q, scale, zp = quantizer.quantize_tensor(t)
        t_hat = quantizer.dequantize_tensor(q, scale, zp)
        assert (t_hat + 2.5).abs().mean() < 0.02

    def test_registry_contains_default(self):
        assert "default" in MODEL_QUANTIZER_REGISTRY
        assert MODEL_QUANTIZER_REGISTRY["default"] is ModelQuantizer
