"""Tests for BitNet b1.58 quantization (src/model/bitnet_quant.py).

Uses tiny dimensions so tests run quickly on CPU.
"""

import torch
import torch.nn as nn

from src.model.bitnet_quant import (
    BitLinear,
    BitNetConfig,
    BitNetFFN,
    apply_bitnet,
    compute_effective_bits,
    quantize_activation,
    ternarize_weight,
)

# ---------------------------------------------------------------------------
# Tiny test dimensions
# ---------------------------------------------------------------------------
D_MODEL = 16
D_FF = 32
BATCH = 2
SEQ = 4


# ===========================================================================
# ternarize_weight
# ===========================================================================


def test_ternarize_weight_values_in_ternary_set():
    """All elements of ternary_W must be in {-1, 0, +1}."""
    W = torch.randn(8, 8)
    ternary_W, _ = ternarize_weight(W)
    # Detach to check raw values without gradient
    vals = ternary_W.detach().unique().tolist()
    for v in vals:
        assert v in {-1.0, 0.0, 1.0}, f"Unexpected value {v} in ternary weight"


def test_ternarize_weight_scale_positive():
    """Scale must be strictly positive."""
    W = torch.randn(4, 4)
    _, scale = ternarize_weight(W)
    assert scale.item() > 0.0


def test_ternarize_weight_preserves_shape():
    """Output shape must match input shape."""
    W = torch.randn(D_MODEL, D_FF)
    ternary_W, scale = ternarize_weight(W)
    assert ternary_W.shape == W.shape
    assert scale.shape == torch.Size([])


def test_ternarize_weight_ste_backward():
    """Backward pass through ternarize_weight must not raise."""
    W = torch.randn(8, 8, requires_grad=True)
    ternary_W, _ = ternarize_weight(W)
    loss = ternary_W.sum()
    loss.backward()  # should not raise
    assert W.grad is not None


# ===========================================================================
# quantize_activation
# ===========================================================================


def test_quantize_activation_range():
    """Quantized float values must lie within [-127, 127] for 8-bit."""
    x = torch.randn(BATCH, SEQ, D_MODEL) * 5.0
    q, scale = quantize_activation(x, bits=8)
    q_vals = q.detach()
    assert q_vals.min().item() >= -127.0 - 1e-5
    assert q_vals.max().item() <= 127.0 + 1e-5


def test_quantize_activation_scale_positive():
    """Activation scale must be positive."""
    x = torch.randn(BATCH, SEQ, D_MODEL)
    _, scale = quantize_activation(x)
    assert scale.item() > 0.0


def test_quantize_activation_shape_preserved():
    """Output quantized tensor must have the same shape as input."""
    x = torch.randn(BATCH, SEQ, D_MODEL)
    q, _ = quantize_activation(x)
    assert q.shape == x.shape


# ===========================================================================
# BitLinear
# ===========================================================================


def test_bitlinear_output_shape_matches_linear():
    """BitLinear and nn.Linear must produce the same output shape."""
    x = torch.randn(BATCH, SEQ, D_MODEL)
    linear = nn.Linear(D_MODEL, D_FF, bias=False)
    bitlinear = BitLinear(D_MODEL, D_FF, bias=False)

    ref_shape = linear(x).shape
    bit_shape = bitlinear(x).shape
    assert bit_shape == ref_shape


def test_bitlinear_gradient_flows():
    """Gradients must reach BitLinear.weight from output loss."""
    x = torch.randn(BATCH, SEQ, D_MODEL, requires_grad=True)
    layer = BitLinear(D_MODEL, D_FF, bias=False)
    out = layer(x)
    loss = out.sum()
    loss.backward()
    assert layer.weight.grad is not None
    assert x.grad is not None


def test_bitlinear_weight_stays_float32_after_forward():
    """Weight parameter must remain float32 after a forward pass."""
    x = torch.randn(BATCH, SEQ, D_MODEL)
    layer = BitLinear(D_MODEL, D_FF)
    _ = layer(x)
    assert layer.weight.dtype == torch.float32


def test_bitlinear_extra_repr_contains_dims():
    """extra_repr should mention d_in and d_out."""
    layer = BitLinear(D_MODEL, D_FF)
    r = layer.extra_repr()
    assert "d_in" in r and "d_out" in r


# ===========================================================================
# apply_bitnet
# ===========================================================================


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, D_MODEL)
        self.linear1 = nn.Linear(D_MODEL, D_FF, bias=False)
        self.linear2 = nn.Linear(D_FF, D_MODEL, bias=False)

    def forward(self, idx):
        x = self.embed(idx)
        return self.linear2(self.linear1(x))


def test_apply_bitnet_replaces_linear():
    """apply_bitnet must replace nn.Linear with BitLinear."""
    model = _TinyModel()
    cfg = BitNetConfig()
    apply_bitnet(model, cfg)
    assert isinstance(model.linear1, BitLinear)
    assert isinstance(model.linear2, BitLinear)


def test_apply_bitnet_leaves_embedding_untouched():
    """apply_bitnet must not replace nn.Embedding layers."""
    model = _TinyModel()
    cfg = BitNetConfig()
    apply_bitnet(model, cfg)
    assert isinstance(model.embed, nn.Embedding)


# ===========================================================================
# compute_effective_bits
# ===========================================================================


def test_compute_effective_bits_keys_present():
    """Return dict must contain n_bitlinear, n_linear, compression_ratio."""
    layer = BitLinear(D_MODEL, D_FF)
    stats = compute_effective_bits(layer)
    assert "n_bitlinear" in stats
    assert "n_linear" in stats
    assert "compression_ratio" in stats


def test_compute_effective_bits_counts():
    """Mixed model should report correct BitLinear / Linear counts."""
    model = _TinyModel()
    cfg = BitNetConfig()
    apply_bitnet(model, cfg)
    stats = compute_effective_bits(model)
    assert stats["n_bitlinear"] == 2
    assert stats["n_linear"] == 0


def test_compute_effective_bits_compression_ratio_range():
    """Compression ratio must be in [0, 1) for a model with BitLinear layers."""
    model = BitLinear(D_MODEL, D_FF)
    stats = compute_effective_bits(model)
    assert 0.0 <= stats["compression_ratio"] < 1.0


# ===========================================================================
# BitNetFFN
# ===========================================================================


def test_bitnetffn_output_shape():
    """BitNetFFN must return tensor with same shape as input."""
    ffn = BitNetFFN(D_MODEL, D_FF)
    x = torch.randn(BATCH, SEQ, D_MODEL)
    out = ffn(x)
    assert out.shape == (BATCH, SEQ, D_MODEL)


def test_bitnetffn_gradient_flows():
    """Gradients must propagate through both BitLinear layers in BitNetFFN."""
    ffn = BitNetFFN(D_MODEL, D_FF)
    x = torch.randn(BATCH, SEQ, D_MODEL, requires_grad=True)
    out = ffn(x)
    out.sum().backward()
    assert ffn.fc1.weight.grad is not None
    assert ffn.fc2.weight.grad is not None
    assert x.grad is not None
