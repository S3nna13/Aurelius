"""Tests for GPTQ post-training quantization (src/inference/gptq.py)."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.inference.gptq import (
    GPTQConfig,
    GPTQLinear,
    apply_gptq_to_model,
    compute_hessian,
    quantize_to_bits,
    quantize_weight_gptq,
)

# ---------------------------------------------------------------------------
# Shared constants (small sizes for fast tests)
# ---------------------------------------------------------------------------
IN_FEATURES = 16
OUT_FEATURES = 8
BITS = 4
GROUP_SIZE = 4


def _make_hessian(n: int, seed: int = 0) -> torch.Tensor:
    """Create a random positive-definite Hessian of shape (n, n)."""
    torch.manual_seed(seed)
    A = torch.randn(n, n)
    return A.T @ A + 0.1 * torch.eye(n)


# ---------------------------------------------------------------------------
# 1. GPTQConfig defaults
# ---------------------------------------------------------------------------

class TestGPTQConfigDefaults:
    def test_gptq_config_defaults(self):
        cfg = GPTQConfig()
        assert cfg.bits == 4
        assert cfg.group_size == 128
        assert cfg.damp_percent == 0.01
        assert cfg.block_size == 128
        assert cfg.actorder is False


# ---------------------------------------------------------------------------
# 2. quantize_to_bits — output same shape as input
# ---------------------------------------------------------------------------

class TestQuantizeToBitsShape:
    def test_quantize_to_bits_shape(self):
        torch.manual_seed(1)
        x = torch.randn(OUT_FEATURES, IN_FEATURES)
        scale = torch.ones(OUT_FEATURES, IN_FEATURES)
        zero = torch.zeros(OUT_FEATURES, IN_FEATURES)
        out = quantize_to_bits(x, BITS, scale, zero)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# 3. quantize_to_bits — values close to original for a well-scaled input
# ---------------------------------------------------------------------------

class TestQuantizeToBitsRange:
    def test_quantize_to_bits_range(self):
        """Dequantized values stay close to original for a low-range matrix."""
        torch.manual_seed(2)
        # Use small values so quantization error is small relative to range
        x = torch.rand(OUT_FEATURES, IN_FEATURES) * 10.0
        x_min = x.min()
        x_max = x.max()
        scale = (x_max - x_min) / (2 ** BITS - 1)
        zero = (-x_min / scale).round()

        out = quantize_to_bits(x, BITS, scale, zero)
        # Max absolute error should be ≤ one quantization step
        assert (out - x).abs().max().item() <= scale.item() + 1e-5


# ---------------------------------------------------------------------------
# 4. quantize_to_bits — 8-bit has lower error than 4-bit
# ---------------------------------------------------------------------------

class TestQuantizeToBitsInt8Precision:
    def test_quantize_to_bits_int8_precision(self):
        torch.manual_seed(3)
        x = torch.randn(32, 64)
        x_min, x_max = x.min(), x.max()

        def _err(bits: int) -> float:
            scale = (x_max - x_min) / (2 ** bits - 1)
            zero = (-x_min / scale).round()
            out = quantize_to_bits(x, bits, scale, zero)
            return (out - x).abs().mean().item()

        assert _err(8) < _err(4)


# ---------------------------------------------------------------------------
# 5. quantize_weight_gptq — W_quantized same shape as W
# ---------------------------------------------------------------------------

class TestQuantizeWeightGPTQShape:
    def test_quantize_weight_gptq_shape(self):
        torch.manual_seed(4)
        W = torch.randn(OUT_FEATURES, IN_FEATURES)
        H = _make_hessian(IN_FEATURES, seed=4)
        cfg = GPTQConfig(bits=BITS, group_size=GROUP_SIZE)
        W_q, scales, zeros = quantize_weight_gptq(W, H, cfg)

        assert W_q.shape == W.shape


# ---------------------------------------------------------------------------
# 6. quantize_weight_gptq — error bounded for identity H
# ---------------------------------------------------------------------------

class TestQuantizeWeightGPTQErrorBounded:
    def test_quantize_weight_gptq_error_bounded(self):
        """Quantization error on unit-range weights should be reasonable."""
        torch.manual_seed(5)
        # Weights in [0, 1] so max quantization error ≤ 1/2^bits per entry
        W = torch.rand(OUT_FEATURES, IN_FEATURES)
        H = torch.eye(IN_FEATURES)
        cfg = GPTQConfig(bits=BITS, group_size=GROUP_SIZE)
        W_q, _, _ = quantize_weight_gptq(W, H, cfg)

        mse = (W_q - W).pow(2).mean().item()
        # Loosely bounded: average MSE well below 1 for 4-bit quantization
        assert mse < 0.5


# ---------------------------------------------------------------------------
# 7. compute_hessian — H shape is (in_features, in_features)
# ---------------------------------------------------------------------------

class TestComputeHessianShape:
    def test_compute_hessian_shape(self):
        torch.manual_seed(6)
        layer = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)
        cal_data = [torch.randn(4, IN_FEATURES) for _ in range(3)]
        H = compute_hessian(layer, cal_data)
        assert H.shape == (IN_FEATURES, IN_FEATURES)


# ---------------------------------------------------------------------------
# 8. compute_hessian — H is approximately symmetric
# ---------------------------------------------------------------------------

class TestComputeHessianSymmetric:
    def test_compute_hessian_symmetric(self):
        torch.manual_seed(7)
        layer = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)
        cal_data = [torch.randn(8, IN_FEATURES) for _ in range(5)]
        H = compute_hessian(layer, cal_data)
        assert torch.allclose(H, H.T, atol=1e-5), "Hessian should be symmetric"


# ---------------------------------------------------------------------------
# 9. GPTQLinear forward — output shape (B, T, out_features)
# ---------------------------------------------------------------------------

class TestGPTQLinearForwardShape:
    def test_gptq_linear_forward_shape(self):
        q_layer = GPTQLinear(IN_FEATURES, OUT_FEATURES, bits=BITS, group_size=GROUP_SIZE)
        B, T = 2, 5
        x = torch.randn(B, T, IN_FEATURES)
        out = q_layer(x)
        assert out.shape == (B, T, OUT_FEATURES)


# ---------------------------------------------------------------------------
# 10. GPTQLinear.from_linear — creates GPTQLinear correctly
# ---------------------------------------------------------------------------

class TestGPTQLinearFromLinear:
    def test_gptq_linear_from_linear(self):
        torch.manual_seed(8)
        linear = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=True)
        cal_data = [torch.randn(4, IN_FEATURES) for _ in range(3)]
        cfg = GPTQConfig(bits=BITS, group_size=GROUP_SIZE)

        q_layer = GPTQLinear.from_linear(linear, cfg, cal_data)

        assert isinstance(q_layer, GPTQLinear)
        assert q_layer.weight_q.shape == (OUT_FEATURES, IN_FEATURES)
        assert q_layer.bias is not None
        assert q_layer.bias.shape == (OUT_FEATURES,)

        # Forward pass must work and produce correct output shape
        x = torch.randn(2, IN_FEATURES)
        out = q_layer(x)
        assert out.shape == (2, OUT_FEATURES)


# ---------------------------------------------------------------------------
# 11. apply_gptq_to_model — replaces Linear layers with GPTQLinear
# ---------------------------------------------------------------------------

class TestApplyGPTQToModel:
    def test_apply_gptq_to_model(self):
        torch.manual_seed(9)

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(IN_FEATURES, OUT_FEATURES)
                self.fc2 = nn.Linear(OUT_FEATURES, OUT_FEATURES)

            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        model = TinyModel()
        cfg = GPTQConfig(bits=BITS, group_size=GROUP_SIZE)
        cal_data = {
            "fc1": [torch.randn(4, IN_FEATURES)],
            "fc2": [torch.randn(4, OUT_FEATURES)],
        }

        model = apply_gptq_to_model(model, cfg, cal_data)

        assert isinstance(model.fc1, GPTQLinear), "fc1 should be GPTQLinear"
        assert isinstance(model.fc2, GPTQLinear), "fc2 should be GPTQLinear"

        # Model should still be callable
        x = torch.randn(2, IN_FEATURES)
        out = model(x)
        assert out.shape == (2, OUT_FEATURES)
