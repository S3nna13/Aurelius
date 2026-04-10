"""Tests for src/training/quantization.py — post-training quantization."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.training.quantization import (
    CalibrationDataset,
    QuantizationConfig,
    QuantizedLinear,
    dequantize_tensor,
    quantization_error,
    quantize_model,
    quantize_tensor,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_cfg() -> AureliusConfig:
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )


def _tiny_model() -> AureliusTransformer:
    return AureliusTransformer(_tiny_cfg())


def _sample_weight(out=16, in_=32) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(out, in_)


# ---------------------------------------------------------------------------
# 1. QuantizationConfig defaults
# ---------------------------------------------------------------------------

def test_quantization_config_defaults():
    cfg = QuantizationConfig()
    assert cfg.bits == 8
    assert cfg.symmetric is True
    assert cfg.per_channel is True
    assert cfg.calibration_batches == 10


# ---------------------------------------------------------------------------
# 2. quantize_tensor — output shape matches input
# ---------------------------------------------------------------------------

def test_quantize_tensor_output_shape():
    w = _sample_weight()
    q_w, scale, zp = quantize_tensor(w, bits=8)
    assert q_w.shape == w.shape, "Quantized weight must have same shape as input"


# ---------------------------------------------------------------------------
# 3. quantize_tensor symmetric — zero_point is 0
# ---------------------------------------------------------------------------

def test_quantize_tensor_symmetric_zero_point():
    w = _sample_weight()
    _, _, zp = quantize_tensor(w, bits=8, symmetric=True, per_channel=True)
    assert torch.all(zp == 0), "Symmetric quantization must have zero_point = 0"


# ---------------------------------------------------------------------------
# 4. quantize_tensor per_channel — scale shape is (out_features,)
# ---------------------------------------------------------------------------

def test_quantize_tensor_per_channel_scale_shape():
    w = _sample_weight(out=16, in_=32)
    _, scale, _ = quantize_tensor(w, bits=8, symmetric=True, per_channel=True)
    assert scale.shape == (16,), f"Expected scale shape (16,), got {scale.shape}"


# ---------------------------------------------------------------------------
# 5. quantize_tensor per_tensor — scale is scalar
# ---------------------------------------------------------------------------

def test_quantize_tensor_per_tensor_scale_scalar():
    w = _sample_weight()
    _, scale, _ = quantize_tensor(w, bits=8, symmetric=True, per_channel=False)
    assert scale.dim() == 0 or scale.numel() == 1, (
        f"Per-tensor scale should be scalar, got shape {scale.shape}"
    )


# ---------------------------------------------------------------------------
# 6. INT4 vs INT8 — INT8 has lower error
# ---------------------------------------------------------------------------

def test_int8_lower_error_than_int4():
    torch.manual_seed(42)
    w = torch.randn(32, 64)

    q8, s8, zp8 = quantize_tensor(w, bits=8)
    dq8 = dequantize_tensor(q8, s8, zp8)
    err8 = quantization_error(w, dq8)

    q4, s4, zp4 = quantize_tensor(w, bits=4)
    dq4 = dequantize_tensor(q4, s4, zp4)
    err4 = quantization_error(w, dq4)

    assert err8 < err4, (
        f"INT8 error ({err8:.4f}) should be less than INT4 error ({err4:.4f})"
    )


# ---------------------------------------------------------------------------
# 7. dequantize_tensor — output shape matches quantized input
# ---------------------------------------------------------------------------

def test_dequantize_output_shape():
    w = _sample_weight()
    q_w, scale, zp = quantize_tensor(w, bits=8)
    dq = dequantize_tensor(q_w, scale, zp)
    assert dq.shape == w.shape, "Dequantized shape must match original"


# ---------------------------------------------------------------------------
# 8. dequantize_tensor — roundtrip close to original
# ---------------------------------------------------------------------------

def test_dequantize_roundtrip():
    torch.manual_seed(7)
    w = torch.randn(32, 64)
    q_w, scale, zp = quantize_tensor(w, bits=8, symmetric=True, per_channel=True)
    dq = dequantize_tensor(q_w, scale, zp)
    err = quantization_error(w, dq)
    assert err < 0.05, f"Roundtrip error too large: {err:.4f} (should be < 0.05 for INT8)"


# ---------------------------------------------------------------------------
# 9. quantization_error — returns float in [0, 1] for reasonable weights
# ---------------------------------------------------------------------------

def test_quantization_error_range():
    torch.manual_seed(1)
    w = torch.randn(16, 32)
    q_w, scale, zp = quantize_tensor(w, bits=8)
    dq = dequantize_tensor(q_w, scale, zp)
    err = quantization_error(w, dq)
    assert isinstance(err, float)
    assert 0.0 <= err <= 1.0, f"Error {err} out of [0, 1]"


# ---------------------------------------------------------------------------
# 10. QuantizedLinear — output shape matches nn.Linear output
# ---------------------------------------------------------------------------

def test_quantized_linear_output_shape():
    linear = nn.Linear(32, 16)
    cfg = QuantizationConfig(bits=8)
    ql = QuantizedLinear(linear, cfg)

    x = torch.randn(4, 32)
    out_ref = linear(x)
    out_ql = ql(x)

    assert out_ql.shape == out_ref.shape, (
        f"Expected {out_ref.shape}, got {out_ql.shape}"
    )


# ---------------------------------------------------------------------------
# 11. QuantizedLinear.compression_ratio — 4.0 for INT8 (32/8)
# ---------------------------------------------------------------------------

def test_quantized_linear_compression_ratio_int8():
    linear = nn.Linear(32, 16)
    cfg = QuantizationConfig(bits=8)
    ql = QuantizedLinear(linear, cfg)
    assert ql.compression_ratio == 4.0, (
        f"Expected 4.0 for INT8, got {ql.compression_ratio}"
    )


# ---------------------------------------------------------------------------
# 12. quantize_model — returns modified model + stats dict
# ---------------------------------------------------------------------------

def test_quantize_model_returns_model_and_stats():
    model = _tiny_model()
    cfg = QuantizationConfig(bits=8)
    q_model, stats = quantize_model(model, cfg)
    assert isinstance(q_model, nn.Module)
    assert isinstance(stats, dict)


# ---------------------------------------------------------------------------
# 13. quantize_model — stats has correct keys
# ---------------------------------------------------------------------------

def test_quantize_model_stats_keys():
    model = _tiny_model()
    cfg = QuantizationConfig(bits=8)
    _, stats = quantize_model(model, cfg)
    assert "n_quantized" in stats
    assert "total_params_saved" in stats
    assert "mean_error" in stats


# ---------------------------------------------------------------------------
# 14. quantize_model — n_quantized > 0 for AureliusTransformer
# ---------------------------------------------------------------------------

def test_quantize_model_n_quantized_positive():
    model = _tiny_model()
    cfg = QuantizationConfig(bits=8)
    _, stats = quantize_model(model, cfg)
    assert stats["n_quantized"] > 0, (
        f"Expected n_quantized > 0, got {stats['n_quantized']}"
    )


# ---------------------------------------------------------------------------
# 15. CalibrationDataset — len matches input
# ---------------------------------------------------------------------------

def test_calibration_dataset_len():
    seqs = [torch.randint(0, 100, (64,)) for _ in range(12)]
    ds = CalibrationDataset(seqs)
    assert len(ds) == 12


# ---------------------------------------------------------------------------
# 16. CalibrationDataset.get_batch — returns tensor of shape (batch_size, T)
# ---------------------------------------------------------------------------

def test_calibration_dataset_get_batch_shape():
    torch.manual_seed(0)
    seqs = [torch.randint(0, 100, (torch.randint(32, 64, ()).item(),)) for _ in range(20)]
    ds = CalibrationDataset(seqs)
    batch = ds.get_batch(batch_size=5)
    assert batch.dim() == 2, f"Expected 2D tensor, got {batch.dim()}D"
    assert batch.shape[0] == 5, f"Expected batch_size=5, got {batch.shape[0]}"
