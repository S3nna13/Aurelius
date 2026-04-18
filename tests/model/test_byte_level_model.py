"""Tests for byte_level_model.py — 16+ tests covering all public classes.

Test parameters:
    B          = 2   (batch size)
    T_bytes    = 16  (byte sequence length)
    d_model    = 16
    patch_size = 4
    n_layers   = 2
    n_heads    = 4
"""

from __future__ import annotations

import math

import pytest
import torch

from src.model.byte_level_model import (
    ByteDecoder,
    ByteEncoder,
    ByteLevelLM,
    ByteMetrics,
    ByteModelConfig,
    CrossPatchAttention,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

B = 2
T_BYTES = 16
D_MODEL = 16
PATCH_SIZE = 4
N_LAYERS = 2
N_HEADS = 4

T_PATCHES = T_BYTES // PATCH_SIZE  # == 4


def make_byte_ids(B: int = B, T: int = T_BYTES) -> torch.Tensor:
    """Random byte values in [0, 255]."""
    return torch.randint(0, 256, (B, T))


def make_lm() -> ByteLevelLM:
    return ByteLevelLM(d_model=D_MODEL, n_layers=N_LAYERS, patch_size=PATCH_SIZE, n_heads=N_HEADS)


# ---------------------------------------------------------------------------
# ByteEncoder tests
# ---------------------------------------------------------------------------

def test_byte_encoder_output_shape():
    """Forward should return [B, T//patch_size, d_model]."""
    enc = ByteEncoder(d_model=D_MODEL, patch_size=PATCH_SIZE)
    byte_ids = make_byte_ids()
    out = enc(byte_ids)
    assert out.shape == (B, T_PATCHES, D_MODEL), (
        f"Expected {(B, T_PATCHES, D_MODEL)}, got {out.shape}"
    )


def test_byte_encoder_non_divisible_length():
    """T not divisible by patch_size should truncate, not crash."""
    enc = ByteEncoder(d_model=D_MODEL, patch_size=PATCH_SIZE)
    # T=17 -> trunc to 16 -> 4 patches
    byte_ids = make_byte_ids(T=17)
    out = enc(byte_ids)
    assert out.shape == (B, 4, D_MODEL)


def test_byte_encoder_single_patch():
    """T == patch_size yields exactly 1 patch."""
    enc = ByteEncoder(d_model=D_MODEL, patch_size=PATCH_SIZE)
    byte_ids = make_byte_ids(T=PATCH_SIZE)
    out = enc(byte_ids)
    assert out.shape == (B, 1, D_MODEL)


def test_byte_encoder_all_256_bytes_embeddable():
    """All 256 distinct byte values should pass through the embedding."""
    enc = ByteEncoder(d_model=D_MODEL, patch_size=PATCH_SIZE)
    # Build a sequence containing all 256 bytes; pad to multiple of patch_size
    all_bytes = torch.arange(0, 256, dtype=torch.long).unsqueeze(0)  # [1, 256]
    out = enc(all_bytes)
    T_patches_expected = 256 // PATCH_SIZE  # 64
    assert out.shape == (1, T_patches_expected, D_MODEL)
    assert torch.isfinite(out).all(), "Encoder output contains non-finite values"


def test_byte_encoder_output_finite():
    """Encoder output should contain only finite values."""
    enc = ByteEncoder(d_model=D_MODEL, patch_size=PATCH_SIZE)
    out = enc(make_byte_ids())
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# ByteDecoder tests
# ---------------------------------------------------------------------------

def test_byte_decoder_output_shape():
    """Forward should return [B, T_patches*patch_size, 256]."""
    dec = ByteDecoder(d_model=D_MODEL, patch_size=PATCH_SIZE)
    patch_repr = torch.randn(B, T_PATCHES, D_MODEL)
    out = dec(patch_repr)
    assert out.shape == (B, T_PATCHES * PATCH_SIZE, 256), (
        f"Expected {(B, T_PATCHES * PATCH_SIZE, 256)}, got {out.shape}"
    )


def test_byte_decoder_output_finite():
    """Decoder logits should be finite (no NaN/Inf)."""
    dec = ByteDecoder(d_model=D_MODEL, patch_size=PATCH_SIZE)
    patch_repr = torch.randn(B, T_PATCHES, D_MODEL)
    out = dec(patch_repr)
    assert torch.isfinite(out).all(), "Decoder output contains non-finite values"


# ---------------------------------------------------------------------------
# CrossPatchAttention tests
# ---------------------------------------------------------------------------

def test_cross_patch_attention_output_shape():
    """CrossPatchAttention should return [B, T_patches, d_model]."""
    cpa = CrossPatchAttention(
        d_model=D_MODEL, n_heads=N_HEADS, patch_size=PATCH_SIZE,
        n_local_layers=1, n_global_layers=1,
    )
    byte_ids = make_byte_ids()
    out = cpa(byte_ids)
    assert out.shape == (B, T_PATCHES, D_MODEL), (
        f"Expected {(B, T_PATCHES, D_MODEL)}, got {out.shape}"
    )


def test_cross_patch_attention_output_finite():
    """CrossPatchAttention output should be finite."""
    cpa = CrossPatchAttention(
        d_model=D_MODEL, n_heads=N_HEADS, patch_size=PATCH_SIZE,
        n_local_layers=1, n_global_layers=1,
    )
    out = cpa(make_byte_ids())
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# ByteLevelLM tests
# ---------------------------------------------------------------------------

def test_byte_lm_forward_output_shape():
    """ByteLevelLM.forward should return [B, T_bytes_trunc, 256]."""
    model = make_lm()
    byte_ids = make_byte_ids()
    logits = model(byte_ids)
    # T_bytes is already divisible by patch_size here
    assert logits.shape == (B, T_BYTES, 256), (
        f"Expected {(B, T_BYTES, 256)}, got {logits.shape}"
    )


def test_byte_lm_compute_loss_finite_positive():
    """compute_loss should return a finite positive scalar."""
    model = make_lm()
    byte_ids = make_byte_ids()
    loss = model.compute_loss(byte_ids)
    assert loss.ndim == 0, "Loss should be a scalar"
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"


def test_byte_lm_compute_loss_backward():
    """Gradients should flow through compute_loss."""
    model = make_lm()
    byte_ids = make_byte_ids()
    loss = model.compute_loss(byte_ids)
    loss.backward()
    # Check that at least some parameters received a gradient
    grads_found = [
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.parameters()
        if p.requires_grad
    ]
    assert any(grads_found), "No gradient found for any parameter"


def test_byte_lm_generate_bytes_length():
    """generate_bytes should return a tensor of length prefix + max_new."""
    model = make_lm()
    prefix = torch.randint(0, 256, (T_BYTES,))
    max_new = 8
    out = model.generate_bytes(prefix, max_new)
    assert out.shape[0] == T_BYTES + max_new, (
        f"Expected {T_BYTES + max_new}, got {out.shape[0]}"
    )


def test_byte_lm_generated_bytes_in_valid_range():
    """All generated bytes should be in [0, 255]."""
    model = make_lm()
    prefix = torch.randint(0, 256, (T_BYTES,))
    out = model.generate_bytes(prefix, max_new=4)
    assert out.min().item() >= 0, "Generated byte < 0"
    assert out.max().item() <= 255, "Generated byte > 255"


# ---------------------------------------------------------------------------
# ByteMetrics tests
# ---------------------------------------------------------------------------

def test_byte_metrics_bits_per_byte_positive():
    """bits_per_byte should return a positive float."""
    model = make_lm()
    byte_ids = make_byte_ids()
    bpb = ByteMetrics.bits_per_byte(model, byte_ids)
    assert isinstance(bpb, float), f"Expected float, got {type(bpb)}"
    assert bpb > 0, f"BPB should be positive, got {bpb}"


def test_byte_metrics_bits_per_byte_random_approx_8():
    """A freshly-initialized (random) model should have BPB close to 8."""
    torch.manual_seed(42)
    model = make_lm()
    # Use a larger batch for stability
    byte_ids = torch.randint(0, 256, (8, 32))
    bpb = ByteMetrics.bits_per_byte(model, byte_ids)
    # For uniform random distribution BPB = log2(256) = 8; allow wide tolerance
    assert 2.0 < bpb < 20.0, (
        f"Random model BPB expected near 8, got {bpb:.3f}"
    )


def test_byte_metrics_byte_accuracy_in_range():
    """byte_accuracy should return a value in [0, 1]."""
    logits = torch.randn(B, T_BYTES, 256)
    targets = make_byte_ids()
    acc = ByteMetrics.byte_accuracy(logits, targets)
    assert 0.0 <= acc <= 1.0, f"byte_accuracy out of [0,1]: {acc}"


def test_byte_metrics_top_k_accuracy_ge_accuracy():
    """top_k_byte_accuracy(k=5) should be >= byte_accuracy."""
    logits = torch.randn(B, T_BYTES, 256)
    targets = make_byte_ids()
    acc1 = ByteMetrics.byte_accuracy(logits, targets)
    acc5 = ByteMetrics.top_k_byte_accuracy(logits, targets, k=5)
    assert acc5 >= acc1 - 1e-6, (
        f"top-5 accuracy {acc5:.4f} < top-1 accuracy {acc1:.4f}"
    )


# ---------------------------------------------------------------------------
# ByteModelConfig tests
# ---------------------------------------------------------------------------

def test_byte_model_config_defaults():
    """ByteModelConfig should have the specified default values."""
    cfg = ByteModelConfig()
    assert cfg.d_model == 32
    assert cfg.n_layers == 2
    assert cfg.patch_size == 4
    assert cfg.n_heads == 4
    assert cfg.n_local_layers == 1
    assert cfg.n_global_layers == 1


def test_byte_model_config_custom():
    """ByteModelConfig should accept custom values."""
    cfg = ByteModelConfig(d_model=64, n_layers=4, patch_size=8, n_heads=8)
    assert cfg.d_model == 64
    assert cfg.n_layers == 4
    assert cfg.patch_size == 8
    assert cfg.n_heads == 8
