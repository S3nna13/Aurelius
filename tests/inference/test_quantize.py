"""Tests for weight quantization."""
import torch
import pytest
from src.inference.quantize import (
    quantize_tensor_int8, dequantize_int8,
    quantize_tensor_int4, dequantize_int4,
    QuantizedLinear, QuantConfig, quantize_model, estimate_memory_savings,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


def test_int8_roundtrip_close():
    """INT8 quantize -> dequantize must approximately recover original weights."""
    torch.manual_seed(0)
    w = torch.randn(32, 64)
    w_q, scale = quantize_tensor_int8(w, per_channel=True)
    w_deq = dequantize_int8(w_q, scale)
    # INT8 per-channel: absolute error should be well under 1% of weight range
    abs_err = (w - w_deq).abs().mean().item()
    assert abs_err < 0.02


def test_int8_per_tensor():
    """Per-tensor INT8 should return scale of shape (1,)."""
    w = torch.randn(16, 32)
    _, scale = quantize_tensor_int8(w, per_channel=False)
    assert scale.numel() == 1


def test_int4_roundtrip_close():
    """INT4 quantize -> dequantize must approximately recover original weights."""
    torch.manual_seed(0)
    w = torch.randn(16, 128)  # in_features divisible by group_size=128
    w_q, scale, zp = quantize_tensor_int4(w, group_size=128)
    w_deq = dequantize_int4(w_q, scale, zp, group_size=128)
    # INT4 has coarse steps (~scale/step); use absolute error which is bounded by scale/2
    abs_err = (w - w_deq).abs().mean().item()
    assert abs_err < 0.15  # INT4 is less precise, ~0.09 abs err expected


def test_quantized_linear_forward():
    """QuantizedLinear forward must produce same shape as nn.Linear."""
    w = torch.randn(32, 64)
    q_linear = QuantizedLinear(w, bias=None, cfg=QuantConfig(bits=8))
    x = torch.randn(2, 10, 64)
    out = q_linear(x)
    assert out.shape == (2, 10, 32)
    assert torch.isfinite(out).all()


def test_quantized_linear_close_to_linear():
    """QuantizedLinear must produce outputs close to original nn.Linear."""
    torch.manual_seed(0)
    linear = torch.nn.Linear(64, 32, bias=False)
    q_linear = QuantizedLinear(linear.weight.data, bias=None, cfg=QuantConfig(bits=8))

    x = torch.randn(4, 64)
    out_orig = linear(x)
    out_q = q_linear(x)

    # Outputs should be close (INT8 approximation)
    rel_err = (out_orig - out_q).abs() / (out_orig.abs() + 1e-8)
    assert rel_err.mean().item() < 0.02


def test_quantize_model_replaces_linears():
    """quantize_model must replace nn.Linear layers (except skipped) with QuantizedLinear."""
    torch.manual_seed(0)
    cfg = AureliusConfig(
        n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
        head_dim=32, d_ff=128, vocab_size=256, max_seq_len=32,
    )
    model = AureliusTransformer(cfg)

    quantize_model(model, QuantConfig(bits=8), skip_modules=("lm_head", "embed"))

    # At least some layers should now be QuantizedLinear
    n_quantized = sum(1 for m in model.modules() if isinstance(m, QuantizedLinear))
    assert n_quantized > 0


def test_quantized_model_forward():
    """Quantized model must still produce valid outputs."""
    torch.manual_seed(0)
    cfg = AureliusConfig(
        n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
        head_dim=32, d_ff=128, vocab_size=256, max_seq_len=32,
    )
    model = AureliusTransformer(cfg)
    quantize_model(model, QuantConfig(bits=8))

    ids = torch.randint(0, 256, (1, 16))
    _, logits, _ = model(ids)
    assert logits.shape == (1, 16, 256)
    assert torch.isfinite(logits).all()
