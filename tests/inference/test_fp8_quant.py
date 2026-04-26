"""Tests for DeepSeek-V3-style FP8 block quantization (src/inference/fp8_quant.py)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.inference.fp8_quant import (
    BLOCK_SIZE,
    FP8Linear,
    compute_fp8_memory_savings,
    fp8_block_dequantize,
    fp8_block_quantize,
    quantize_model_fp8,
)

# ---------------------------------------------------------------------------
# Core quantize / dequantize tests
# ---------------------------------------------------------------------------


class TestFP8BlockQuantize:
    def test_fp8_quantize_shape(self):
        """quantized has same shape as input; scales has correct n_blocks."""
        weight = torch.randn(64, 128)
        quantized, scales = fp8_block_quantize(weight)

        assert quantized.shape == weight.shape

        n_elements = weight.numel()
        expected_n_blocks = math.ceil(n_elements / BLOCK_SIZE)
        assert scales.shape == (expected_n_blocks,)

    def test_fp8_roundtrip_accuracy(self):
        """dequantize(quantize(W)) is close to W.

        FP8 e4m3fn has 3 mantissa bits, giving ~1/8 quantization step per
        binade.  Empirically the max absolute error stays below ~5% of
        max(|W|) for standard-normal weights; we verify that bound here.
        Per-block scaling (one scale per 128 elements) keeps each block's
        dynamic range well-covered, and the result is far more accurate
        than a single global INT8 scale.
        """
        torch.manual_seed(42)
        weight = torch.randn(128, 256)
        quantized, scales = fp8_block_quantize(weight)
        reconstructed = fp8_block_dequantize(quantized, scales, tuple(weight.shape))

        max_weight = weight.abs().max().item()
        max_error = (weight.float() - reconstructed).abs().max().item()
        assert max_error < 0.05 * max_weight, (
            f"Roundtrip error {max_error:.6f} exceeds 5% of max(|W|)={max_weight:.6f}"
        )

    def test_fp8_scales_positive(self):
        """All per-block scales must be strictly positive."""
        torch.manual_seed(0)
        weight = torch.randn(32, 64)
        _, scales = fp8_block_quantize(weight)
        assert (scales > 0).all(), "Some scales are non-positive"

    def test_fp8_block_count(self):
        """Weight (256, 512): n_blocks = ceil(256*512 / 128) = 1024."""
        weight = torch.randn(256, 512)
        _, scales = fp8_block_quantize(weight)
        expected = math.ceil(256 * 512 / BLOCK_SIZE)
        assert expected == 1024
        assert scales.numel() == expected

    def test_fp8_scales_dtype(self):
        """Scales are stored as float32."""
        weight = torch.randn(32, 64)
        _, scales = fp8_block_quantize(weight)
        assert scales.dtype == torch.float32

    def test_fp8_dequantize_shape(self):
        """Dequantized tensor has the original shape."""
        weight = torch.randn(48, 96)
        quantized, scales = fp8_block_quantize(weight)
        result = fp8_block_dequantize(quantized, scales, tuple(weight.shape))
        assert result.shape == weight.shape
        assert result.dtype == torch.float32


# ---------------------------------------------------------------------------
# FP8Linear tests
# ---------------------------------------------------------------------------


class TestFP8Linear:
    def test_fp8_linear_forward_shape(self):
        """FP8Linear(64, 128): (4, 64) input → (4, 128) output."""
        layer = FP8Linear(64, 128)
        x = torch.randn(4, 64)
        out = layer(x)
        assert out.shape == (4, 128)

    def test_fp8_linear_from_linear(self):
        """FP8Linear.from_linear: forward output close to original nn.Linear output."""
        torch.manual_seed(7)
        linear = nn.Linear(64, 128)
        fp8_layer = FP8Linear.from_linear(linear)

        x = torch.randn(8, 64)
        with torch.no_grad():
            expected = linear(x)
            actual = fp8_layer(x)

        # FP8 e4m3fn weight quantization error accumulates across in_features=64
        # dot products; allow up to 10% of the output range as tolerance.
        max_range = expected.abs().max().item()
        max_diff = (expected - actual).abs().max().item()
        assert max_diff < 0.10 * max_range, (
            f"FP8Linear output differs too much: max_diff={max_diff:.6f}, max_range={max_range:.6f}"
        )

    def test_fp8_linear_no_bias(self):
        """FP8Linear(64, 128, bias=False): forward works, bias attribute is None."""
        layer = FP8Linear(64, 128, bias=False)
        assert layer.bias is None
        x = torch.randn(3, 64)
        out = layer(x)
        assert out.shape == (3, 128)

    def test_fp8_linear_has_buffers(self):
        """FP8Linear registers weight_q and fp8_scales as buffers, not parameters."""
        layer = FP8Linear(32, 64)
        buffer_names = {name for name, _ in layer.named_buffers()}
        assert "weight_q" in buffer_names
        assert "fp8_scales" in buffer_names
        param_names = {name for name, _ in layer.named_parameters()}
        assert "weight_q" not in param_names
        assert "fp8_scales" not in param_names

    def test_fp8_linear_from_linear_preserves_bias(self):
        """from_linear preserves bias values from the source nn.Linear."""
        torch.manual_seed(3)
        linear = nn.Linear(16, 32)
        fp8_layer = FP8Linear.from_linear(linear)
        assert fp8_layer.bias is not None
        assert torch.allclose(fp8_layer.bias, linear.bias, atol=1e-6)


# ---------------------------------------------------------------------------
# Model quantization tests
# ---------------------------------------------------------------------------


class _SmallModel(nn.Module):
    """Tiny model for testing model-level quantization."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 64)
        self.lm_head = nn.Linear(64, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.lm_head(x)


class TestQuantizeModelFP8:
    def test_quantize_model_replaces_linears(self):
        """After quantize_model_fp8, all non-skipped Linear layers become FP8Linear."""
        model = _SmallModel()
        quantize_model_fp8(model, skip_layers=[])

        for name, module in model.named_modules():
            if name in ("fc1", "fc2", "lm_head"):
                assert isinstance(module, FP8Linear), (
                    f"Expected FP8Linear for '{name}', got {type(module)}"
                )

    def test_skip_layers(self):
        """skip_layers=['lm_head'] leaves lm_head as nn.Linear."""
        model = _SmallModel()
        quantize_model_fp8(model, skip_layers=["lm_head"])

        assert isinstance(model.fc1, FP8Linear), "fc1 should be FP8Linear"
        assert isinstance(model.fc2, FP8Linear), "fc2 should be FP8Linear"
        assert isinstance(model.lm_head, nn.Linear), "lm_head should remain nn.Linear"
        assert not isinstance(model.lm_head, FP8Linear), "lm_head should NOT be FP8Linear"

    def test_quantize_model_forward_works(self):
        """Model forward pass works after quantization."""
        model = _SmallModel()
        quantize_model_fp8(model)
        x = torch.randn(4, 32)
        out = model(x)
        assert out.shape == (4, 16)


# ---------------------------------------------------------------------------
# Memory savings tests
# ---------------------------------------------------------------------------


class TestComputeFP8MemorySavings:
    def test_memory_savings_factor(self):
        """savings_factor >= 1.5 (FP8 uses substantially less memory than BF16)."""
        model = _SmallModel()
        quantize_model_fp8(model)
        stats = compute_fp8_memory_savings(model)
        assert stats["savings_factor"] >= 1.5, (
            f"Expected savings_factor >= 1.5, got {stats['savings_factor']:.4f}"
        )

    def test_memory_savings_keys(self):
        """compute_fp8_memory_savings returns all expected keys."""
        model = _SmallModel()
        quantize_model_fp8(model)
        stats = compute_fp8_memory_savings(model)
        for key in (
            "fp8_params",
            "total_params",
            "fp8_memory_mb",
            "bf16_memory_mb",
            "savings_factor",
        ):
            assert key in stats, f"Missing key: {key}"

    def test_memory_savings_fp8_params_count(self):
        """fp8_params counts parameters in FP8Linear layers."""
        model = _SmallModel()
        quantize_model_fp8(model, skip_layers=["lm_head"])
        stats = compute_fp8_memory_savings(model)

        # fc1: 32*64 = 2048, fc2: 64*64 = 4096
        expected_fp8 = 32 * 64 + 64 * 64
        assert stats["fp8_params"] == expected_fp8

    def test_memory_bf16_greater_than_fp8(self):
        """bf16_memory_mb > fp8_memory_mb for any model with FP8 layers."""
        model = _SmallModel()
        quantize_model_fp8(model)
        stats = compute_fp8_memory_savings(model)
        assert stats["bf16_memory_mb"] > stats["fp8_memory_mb"]
