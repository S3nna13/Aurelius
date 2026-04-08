"""Tests for LoftQ: Low-Rank Fine-Tuning with Quantization."""
import torch
import torch.nn as nn
import pytest

from src.training.loftq import (
    LoftQConfig,
    quantize_nf4,
    dequantize_nf4,
    quantize_int8,
    dequantize_int8,
    loftq_init,
    LoftQLinear,
    apply_loftq,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Config used across tests
# ---------------------------------------------------------------------------
MODEL_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=4,
    n_kv_heads=2,
    head_dim=16,
    d_ff=128,
    vocab_size=256,
    max_seq_len=64,
)

# Small weight tensors for loftq_init tests
OUT_FEATURES = 32
IN_FEATURES = 64


# ---------------------------------------------------------------------------
# Test 1: quantize_nf4 output shape
# ---------------------------------------------------------------------------
def test_quantize_nf4_shape():
    """quantize_nf4 must return quantized tensor of same shape as input."""
    w = torch.randn(OUT_FEATURES, IN_FEATURES)
    q, scales = quantize_nf4(w, group_size=64)
    assert q.shape == w.shape, f"Expected shape {w.shape}, got {q.shape}"
    assert q.dtype == torch.int8


# ---------------------------------------------------------------------------
# Test 2: dequantize roundtrip error bounded
# ---------------------------------------------------------------------------
def test_dequantize_roundtrip_error():
    """NF4 dequant(quant(w)) should approximate w within a bounded error.

    4-bit quantization is lossy; the Frobenius norm of the error should be
    less than 20% of the Frobenius norm of the original weight.
    """
    torch.manual_seed(42)
    w = torch.randn(OUT_FEATURES, IN_FEATURES)
    q, scales = quantize_nf4(w, group_size=64)
    w_deq = dequantize_nf4(q, scales, group_size=64)
    assert w_deq.shape == w.shape

    # Frobenius norm relative error: ||w_deq - w||_F / ||w||_F < 0.2
    err_norm = (w_deq - w).norm(p="fro").item()
    ref_norm = w.norm(p="fro").item()
    rel_err = err_norm / ref_norm
    assert rel_err < 0.2, (
        f"Frobenius relative error {rel_err:.4f} exceeds 0.2 threshold"
    )


# ---------------------------------------------------------------------------
# Test 3: quantize_int8 output shape and scales shape
# ---------------------------------------------------------------------------
def test_quantize_int8_shape():
    """quantize_int8 must return quantized of same shape and scales of (out_features,)."""
    w = torch.randn(OUT_FEATURES, IN_FEATURES)
    q, scales = quantize_int8(w)
    assert q.shape == w.shape, f"Expected quantized shape {w.shape}, got {q.shape}"
    assert q.dtype == torch.int8
    assert scales.shape == (OUT_FEATURES,), (
        f"Expected scales shape ({OUT_FEATURES},), got {scales.shape}"
    )


# ---------------------------------------------------------------------------
# Test 4: loftq_init output shapes
# ---------------------------------------------------------------------------
def test_loftq_init_shapes():
    """loftq_init must return A: (out, rank), B: (rank, in), W_q: (out, in)."""
    torch.manual_seed(0)
    rank = 8
    w = torch.randn(OUT_FEATURES, IN_FEATURES)
    A, B, W_q = loftq_init(w, rank=rank, n_bits=4, n_iter=3)

    assert A.shape == (OUT_FEATURES, rank), f"Expected A shape ({OUT_FEATURES}, {rank}), got {A.shape}"
    assert B.shape == (rank, IN_FEATURES), f"Expected B shape ({rank}, {IN_FEATURES}), got {B.shape}"
    assert W_q.shape == (OUT_FEATURES, IN_FEATURES), f"Expected W_q shape ({OUT_FEATURES}, {IN_FEATURES}), got {W_q.shape}"


# ---------------------------------------------------------------------------
# Test 5: loftq_init improves on quantization alone
# ---------------------------------------------------------------------------
def test_loftq_init_approximation():
    """||W_q + A@B - W||_F < ||W_q - W||_F: LoRA must reduce quantization error."""
    torch.manual_seed(7)
    rank = 8
    w = torch.randn(OUT_FEATURES, IN_FEATURES)
    A, B, W_q = loftq_init(w, rank=rank, n_bits=4, n_iter=5)

    err_quant_only = (W_q - w).norm(p="fro").item()
    err_loftq = (W_q + A @ B - w).norm(p="fro").item()

    assert err_loftq < err_quant_only, (
        f"LoftQ err {err_loftq:.4f} should be < quant-only err {err_quant_only:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 6: LoftQLinear forward shape
# ---------------------------------------------------------------------------
def test_loftq_linear_forward_shape():
    """LoftQLinear forward must produce shape (B, T, out_features)."""
    torch.manual_seed(0)
    lin = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    loftq_lin = LoftQLinear(lin, rank=8, n_bits=4, n_iter=3)

    batch, seq_len = 2, 10
    x = torch.randn(batch, seq_len, IN_FEATURES)
    out = loftq_lin(x)
    assert out.shape == (batch, seq_len, OUT_FEATURES), (
        f"Expected ({batch}, {seq_len}, {OUT_FEATURES}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# Test 7: W_q is frozen (requires_grad=False)
# ---------------------------------------------------------------------------
def test_loftq_linear_w_frozen():
    """LoftQLinear.W_q must have requires_grad=False."""
    lin = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    loftq_lin = LoftQLinear(lin, rank=8, n_bits=4, n_iter=2)
    assert not loftq_lin.W_q.requires_grad, "W_q must be a frozen parameter"


# ---------------------------------------------------------------------------
# Test 8: A and B are trainable
# ---------------------------------------------------------------------------
def test_loftq_linear_ab_trainable():
    """LoftQLinear.A and .B must have requires_grad=True."""
    lin = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    loftq_lin = LoftQLinear(lin, rank=8, n_bits=4, n_iter=2)
    assert loftq_lin.A.requires_grad, "A must be trainable"
    assert loftq_lin.B.requires_grad, "B must be trainable"


# ---------------------------------------------------------------------------
# Test 9: apply_loftq replaces layers with LoftQLinear
# ---------------------------------------------------------------------------
def test_apply_loftq_replaces_layers():
    """apply_loftq must replace target nn.Linear layers with LoftQLinear."""
    torch.manual_seed(0)
    model = AureliusTransformer(MODEL_CFG)
    config = LoftQConfig(rank=8, n_bits=4, n_iter=2)
    model, stats = apply_loftq(model, config)

    n_loftq = sum(1 for m in model.modules() if isinstance(m, LoftQLinear))
    assert n_loftq > 0, "No LoftQLinear layers found after apply_loftq"
    assert stats["n_replaced"] == n_loftq, (
        f"stats['n_replaced']={stats['n_replaced']} != actual count {n_loftq}"
    )


# ---------------------------------------------------------------------------
# Test 10: apply_loftq returns correct stats keys
# ---------------------------------------------------------------------------
def test_apply_loftq_stats_keys():
    """apply_loftq must return dict with 'n_replaced', 'total_params', 'lora_params'."""
    torch.manual_seed(0)
    model = AureliusTransformer(MODEL_CFG)
    config = LoftQConfig(rank=8, n_bits=4, n_iter=2)
    _, stats = apply_loftq(model, config)

    assert "n_replaced" in stats, "Missing key 'n_replaced'"
    assert "total_params" in stats, "Missing key 'total_params'"
    assert "lora_params" in stats, "Missing key 'lora_params'"
    assert isinstance(stats["n_replaced"], int)
    assert isinstance(stats["total_params"], int)
    assert isinstance(stats["lora_params"], int)
    assert stats["lora_params"] > 0, "lora_params should be > 0 after replacement"
