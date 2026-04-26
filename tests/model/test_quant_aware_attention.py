"""Tests for Quantization-Aware Attention (src/model/quant_aware_attention.py).

Uses tiny dimensions so tests run quickly on CPU.
"""

import torch

from src.model.quant_aware_attention import (
    FakeQuantize,
    QuantAwareAttention,
    QuantCalibrator,
    QuantConfig,
)

# ---------------------------------------------------------------------------
# Common test dimensions
# ---------------------------------------------------------------------------
D_MODEL = 16
N_HEADS = 4
D_HEAD = D_MODEL // N_HEADS  # 4
B = 2
T = 8


# ===========================================================================
# FakeQuantize
# ===========================================================================


def test_fake_quantize_output_shape():
    """FakeQuantize output shape must match input shape."""
    fq = FakeQuantize(bits=8)
    x = torch.randn(B, T, D_MODEL)
    out = fq(x)
    assert out.shape == x.shape


def test_fake_quantize_output_dtype():
    """FakeQuantize output dtype must match input dtype."""
    fq = FakeQuantize(bits=8)
    x = torch.randn(B, T, D_MODEL)
    out = fq(x)
    assert out.dtype == x.dtype


def test_fake_quantize_values_on_grid():
    """Fake-quantized values should be close to scale * integer grid points.

    After quantization x_q = round(x/scale)*scale; the difference
    x_q - round(x_q / scale)*scale should be near zero.
    """
    fq = FakeQuantize(bits=8)
    x = torch.randn(B, T, D_MODEL)
    x_q = fq(x).detach()

    q_max = 2 ** (8 - 1) - 1  # 127
    scale = x.abs().max() / (q_max + 1e-8)
    residual = (x_q / scale) - torch.round(x_q / scale)
    assert residual.abs().max().item() < 1e-4, "Values are not on the quantization grid"


def test_fake_quantize_ste_gradient():
    """STE: gradient of fake-quantized output w.r.t. input should be ~1."""
    fq = FakeQuantize(bits=8)
    x = torch.randn(4, requires_grad=True)
    out = fq(x)
    out.sum().backward()
    # STE means dx == 1 for all elements
    assert x.grad is not None
    assert torch.allclose(x.grad, torch.ones_like(x.grad)), (
        f"Expected STE gradient of 1, got: {x.grad}"
    )


def test_fake_quantize_bits4_more_error_than_bits8():
    """4-bit quantization must produce more error than 8-bit quantization."""
    torch.manual_seed(0)
    x = torch.randn(32, 32)
    fq4 = FakeQuantize(bits=4)
    fq8 = FakeQuantize(bits=8)

    err4 = (fq4(x).detach() - x).pow(2).mean().item()
    err8 = (fq8(x).detach() - x).pow(2).mean().item()
    assert err4 > err8, f"Expected 4-bit error ({err4}) > 8-bit error ({err8})"


# ===========================================================================
# QuantConfig
# ===========================================================================


def test_quant_config_defaults():
    """QuantConfig default values must match the specification."""
    cfg = QuantConfig()
    assert cfg.weight_bits == 8
    assert cfg.activation_bits == 8
    assert cfg.kv_bits == 8
    assert cfg.quantize_weights is True
    assert cfg.quantize_activations is False
    assert cfg.quantize_kv is False


# ===========================================================================
# QuantAwareAttention — basic forward
# ===========================================================================


def test_quant_aware_attention_output_shape():
    """QuantAwareAttention must return (B, T, d_model)."""
    model = QuantAwareAttention(D_MODEL, N_HEADS)
    x = torch.randn(B, T, D_MODEL)
    out = model(x)
    assert out.shape == (B, T, D_MODEL)


def test_quant_aware_attention_output_finite():
    """QuantAwareAttention output must contain no NaN or Inf."""
    model = QuantAwareAttention(D_MODEL, N_HEADS)
    x = torch.randn(B, T, D_MODEL)
    out = model(x)
    assert torch.isfinite(out).all(), "Output contains NaN or Inf"


def test_quant_aware_attention_gradients_flow():
    """Gradients must reach input tensor through QuantAwareAttention."""
    model = QuantAwareAttention(D_MODEL, N_HEADS)
    x = torch.randn(B, T, D_MODEL, requires_grad=True)
    out = model(x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum().item() > 0.0, "Input gradient is all zeros"


# ===========================================================================
# QuantAwareAttention — weight quantization
# ===========================================================================


def test_quantize_weights_changes_effective_weight():
    """With quantize_weights=True, the effective weight used differs from raw weight.

    We check that the quantized weight differs from the original for at least
    one element (which is expected since rounding introduces error).
    """
    cfg_on = QuantConfig(quantize_weights=True, weight_bits=4)
    cfg_off = QuantConfig(quantize_weights=False)

    torch.manual_seed(42)
    model_on = QuantAwareAttention(D_MODEL, N_HEADS, config=cfg_on)
    torch.manual_seed(42)
    model_off = QuantAwareAttention(D_MODEL, N_HEADS, config=cfg_off)

    w = model_on.q_proj.weight
    q_w = model_on._quantize_weight(w)
    assert not torch.allclose(w, q_w, atol=1e-6), (
        "Quantized weight should differ from full-precision weight (4-bit)"
    )

    # The model with quantization disabled should pass weight through unchanged
    w_off = model_off.q_proj.weight
    assert torch.allclose(w_off, model_off._quantize_weight(w_off)), (
        "Identity quantizer should return weight unchanged"
    )


def test_quantize_kv_4bit_finite():
    """quantize_kv=True with kv_bits=4 must still produce finite output."""
    cfg = QuantConfig(quantize_kv=True, kv_bits=4)
    model = QuantAwareAttention(D_MODEL, N_HEADS, config=cfg)
    x = torch.randn(B, T, D_MODEL)
    out = model(x)
    assert torch.isfinite(out).all(), "Output with 4-bit KV quantization contains NaN/Inf"


def test_all_quantization_disabled_output_shape():
    """With all quantization disabled, output shape must still be (B, T, d_model)."""
    cfg = QuantConfig(
        quantize_weights=False,
        quantize_activations=False,
        quantize_kv=False,
    )
    model = QuantAwareAttention(D_MODEL, N_HEADS, config=cfg)
    x = torch.randn(B, T, D_MODEL)
    out = model(x)
    assert out.shape == (B, T, D_MODEL)


# ===========================================================================
# QuantCalibrator
# ===========================================================================


def test_quant_calibrator_get_quantizer_stats():
    """get_quantizer_stats must return a dict mapping name -> bits."""
    cfg = QuantConfig(
        weight_bits=4,
        quantize_weights=True,
        quantize_activations=True,
        activation_bits=8,
        quantize_kv=True,
        kv_bits=4,
    )
    model = QuantAwareAttention(D_MODEL, N_HEADS, config=cfg)
    calibrator = QuantCalibrator(model)
    stats = calibrator.get_quantizer_stats()

    assert isinstance(stats, dict)
    assert len(stats) > 0, "Expected at least one FakeQuantize module"

    # All values must be integer bit counts
    for name, bits in stats.items():
        assert isinstance(bits, int), f"Expected int bits for {name}, got {type(bits)}"
        assert bits in {4, 8, 16}, f"Unexpected bit width {bits} for {name}"

    # We expect weight_quantizer, act_quantizer, kv_quantizer
    found_4bit = any(b == 4 for b in stats.values())
    assert found_4bit, "Expected at least one 4-bit quantizer in stats"


# ===========================================================================
# Different weight_bits gives different coarseness
# ===========================================================================


def test_different_weight_bits_different_coarseness():
    """Higher weight_bits should produce smaller quantization error."""
    torch.manual_seed(7)
    w = torch.randn(D_MODEL, D_MODEL)

    fq4 = FakeQuantize(bits=4, per_channel=True)
    fq8 = FakeQuantize(bits=8, per_channel=True)

    err4 = (fq4(w).detach() - w).pow(2).mean().item()
    err8 = (fq8(w).detach() - w).pow(2).mean().item()
    assert err4 > err8, (
        f"4-bit per-channel error ({err4:.6f}) should exceed 8-bit per-channel error ({err8:.6f})"
    )


# ===========================================================================
# Edge case: B=1, T=1
# ===========================================================================


def test_quant_aware_attention_b1_t1_edge_case():
    """QuantAwareAttention must handle single-token batch (B=1, T=1)."""
    model = QuantAwareAttention(D_MODEL, N_HEADS)
    x = torch.randn(1, 1, D_MODEL)
    out = model(x)
    assert out.shape == (1, 1, D_MODEL)
    assert torch.isfinite(out).all(), "B=1,T=1 output contains NaN/Inf"
