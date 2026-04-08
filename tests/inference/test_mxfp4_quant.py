"""Tests for MXFP4 microscaling quantization."""
import torch
import torch.nn as nn
import pytest
from src.inference.mxfp4_quant import (
    MXFPConfig,
    mxfp4_quantize,
    mxfp4_dequantize,
    mxfp4_quantization_error,
    MXFP4Linear,
    quantize_model,
    estimate_quantization_impact,
)


# ---------------------------------------------------------------------------
# Core quantize / dequantize
# ---------------------------------------------------------------------------

def test_mxfp4_quantize_output_shapes():
    """Quantized tensor and block_scales must have correct shapes."""
    torch.manual_seed(0)
    cfg = MXFPConfig(block_size=32)
    weight = torch.randn(64, 128)
    q, scales = mxfp4_quantize(weight, cfg)
    # quantized same shape as weight
    assert q.shape == weight.shape
    # scales: one per block along last dim
    assert scales.shape == (64, 128 // 32)


def test_mxfp4_quantize_int_range():
    """Quantized values must lie in [-8, 7] for 4-bit symmetric."""
    torch.manual_seed(1)
    cfg = MXFPConfig(block_size=32, bits=4, symmetric=True)
    weight = torch.randn(32, 64)
    q, _ = mxfp4_quantize(weight, cfg)
    assert q.dtype == torch.int8
    assert q.min().item() >= -8
    assert q.max().item() <= 7


def test_mxfp4_dequantize_shape():
    """Dequantized tensor must match original weight shape."""
    torch.manual_seed(2)
    cfg = MXFPConfig(block_size=32)
    weight = torch.randn(48, 96)
    q, scales = mxfp4_quantize(weight, cfg)
    reconstructed = mxfp4_dequantize(q, scales, cfg, weight.shape)
    assert reconstructed.shape == weight.shape
    assert reconstructed.dtype == torch.float32


def test_mxfp4_roundtrip_low_error():
    """MSE after quantize -> dequantize should be < 0.01 for Gaussian weights."""
    torch.manual_seed(3)
    cfg = MXFPConfig(block_size=32)
    weight = torch.randn(64, 128)
    q, scales = mxfp4_quantize(weight, cfg)
    reconstructed = mxfp4_dequantize(q, scales, cfg, weight.shape)
    mse = ((weight - reconstructed) ** 2).mean().item()
    assert mse < 0.01, f"MSE {mse:.6f} too high"


def test_mxfp4_block_size_16():
    """block_size=16 must work end-to-end."""
    torch.manual_seed(4)
    cfg = MXFPConfig(block_size=16)
    weight = torch.randn(32, 64)
    q, scales = mxfp4_quantize(weight, cfg)
    assert scales.shape == (32, 64 // 16)
    reconstructed = mxfp4_dequantize(q, scales, cfg, weight.shape)
    assert reconstructed.shape == weight.shape


def test_mxfp4_block_size_32():
    """block_size=32 must work end-to-end."""
    torch.manual_seed(5)
    cfg = MXFPConfig(block_size=32)
    weight = torch.randn(32, 128)
    q, scales = mxfp4_quantize(weight, cfg)
    assert scales.shape == (32, 128 // 32)
    reconstructed = mxfp4_dequantize(q, scales, cfg, weight.shape)
    assert reconstructed.shape == weight.shape


def test_quantization_error_metrics():
    """mxfp4_quantization_error must return dict with required keys."""
    torch.manual_seed(6)
    cfg = MXFPConfig(block_size=32)
    weight = torch.randn(32, 64)
    metrics = mxfp4_quantization_error(weight, cfg)
    for key in ("mse", "max_error", "relative_error", "snr_db"):
        assert key in metrics, f"Missing key: {key}"
        assert isinstance(metrics[key], float)


# ---------------------------------------------------------------------------
# MXFP4Linear
# ---------------------------------------------------------------------------

def test_mxfp4_linear_forward_shape():
    """MXFP4Linear forward must produce (B, out_features) output."""
    torch.manual_seed(7)
    cfg = MXFPConfig(block_size=32)
    layer = MXFP4Linear(in_features=64, out_features=32, bias=True, cfg=cfg)
    x = torch.randn(4, 64)
    out = layer(x)
    assert out.shape == (4, 32)


def test_mxfp4_linear_from_linear():
    """MXFP4Linear.from_linear must create MXFP4Linear from nn.Linear."""
    torch.manual_seed(8)
    linear = nn.Linear(64, 32, bias=True)
    cfg = MXFPConfig(block_size=32)
    mxfp4_layer = MXFP4Linear.from_linear(linear, cfg)
    assert isinstance(mxfp4_layer, MXFP4Linear)
    # Forward must run without error
    x = torch.randn(2, 64)
    out = mxfp4_layer(x)
    assert out.shape == (2, 32)


def test_mxfp4_linear_compression_ratio():
    """compression_ratio must be > 1.0 (quantized uses less memory)."""
    cfg = MXFPConfig(block_size=32)
    layer = MXFP4Linear(in_features=128, out_features=64, bias=False, cfg=cfg)
    ratio = layer.compression_ratio()
    assert ratio > 1.0, f"Expected compression ratio > 1, got {ratio}"


def test_mxfp4_linear_effective_bits():
    """effective_bits must be between 4.0 and 4.5 (scale overhead is small)."""
    cfg = MXFPConfig(block_size=32, bits=4, scale_bits=8)
    layer = MXFP4Linear(in_features=128, out_features=64, bias=False, cfg=cfg)
    eff = layer.effective_bits()
    assert 4.0 <= eff <= 4.5, f"Expected effective bits in [4.0, 4.5], got {eff}"


# ---------------------------------------------------------------------------
# quantize_model
# ---------------------------------------------------------------------------

def test_quantize_model_replaces_linear():
    """quantize_model must replace all nn.Linear with MXFP4Linear."""
    model = nn.Sequential(
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
    )
    cfg = MXFPConfig(block_size=32)
    quantize_model(model, cfg)
    for module in model.modules():
        if not isinstance(module, (nn.Sequential, nn.ReLU, MXFP4Linear)):
            assert not isinstance(module, nn.Linear), \
                f"nn.Linear not replaced: {type(module)}"


def test_quantize_model_skip_layers():
    """Layers matching skip_layers patterns must remain as nn.Linear."""

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(32, 64)
            self.hidden = nn.Linear(64, 64)
            self.lm_head = nn.Linear(64, 32)

        def forward(self, x):
            return self.lm_head(self.hidden(self.embed(x)))

    model = TinyModel()
    cfg = MXFPConfig(block_size=32)
    quantize_model(model, cfg, skip_layers=["embed", "lm_head"])

    assert isinstance(model.embed, nn.Linear), "embed should be skipped"
    assert isinstance(model.lm_head, nn.Linear), "lm_head should be skipped"
    assert isinstance(model.hidden, MXFP4Linear), "hidden should be quantized"


# ---------------------------------------------------------------------------
# estimate_quantization_impact
# ---------------------------------------------------------------------------

def test_estimate_quantization_impact_keys():
    """estimate_quantization_impact must return dict with required keys."""
    model = nn.Sequential(
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
    )
    cfg = MXFPConfig(block_size=32)
    x = torch.randn(2, 64)
    result = estimate_quantization_impact(model, x, cfg)
    for key in ("mean_snr_db", "min_snr_db", "compression_ratio",
                "total_params", "quantized_params"):
        assert key in result, f"Missing key: {key}"
