"""Tests for src/model/xpos_attention.py"""

from __future__ import annotations

import pytest
import torch

from src.model.xpos_attention import (
    XPosAttention,
    XPosConfig,
    _build_xpos_scale,
)

# ---------------------------------------------------------------------------
# Config used across tests
# ---------------------------------------------------------------------------

D_MODEL = 64
N_HEADS = 4
N_KV_HEADS = 2
HEAD_DIM = 16  # D_MODEL // N_HEADS
SCALE_BASE = 512
THETA = 10000.0


def _cfg(use_xpos: bool = True) -> XPosConfig:
    return XPosConfig(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        head_dim=HEAD_DIM,
        base_theta=THETA,
        scale_base=SCALE_BASE,
        use_xpos=use_xpos,
    )


# ---------------------------------------------------------------------------
# 1. Output shape correctness
# ---------------------------------------------------------------------------


def test_output_shape_batch4():
    """Forward pass with batch=4, seq=16 returns (4, 16, d_model)."""
    B, S = 4, 16
    cfg = _cfg()
    model = XPosAttention(cfg, n_kv_heads=N_KV_HEADS)
    x = torch.randn(B, S, D_MODEL)
    out = model(x)
    assert out.shape == (B, S, D_MODEL)


# ---------------------------------------------------------------------------
# 2. Output dtype
# ---------------------------------------------------------------------------


def test_output_dtype():
    """Output tensor must be float32."""
    cfg = _cfg()
    model = XPosAttention(cfg, n_kv_heads=N_KV_HEADS)
    x = torch.randn(1, 8, D_MODEL)
    out = model(x)
    assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# 3. XPos scale shape
# ---------------------------------------------------------------------------


def test_xpos_scale_shape():
    """_build_xpos_scale should return (seq_len, head_dim)."""
    seq_len = 32
    scale = _build_xpos_scale(seq_len, HEAD_DIM, SCALE_BASE, torch.device("cpu"))
    assert scale.shape == (seq_len, HEAD_DIM)


# ---------------------------------------------------------------------------
# 4. Disabled mode falls back to standard RoPE (output still valid shape)
# ---------------------------------------------------------------------------


def test_disabled_xpos_falls_back():
    """use_xpos=False should still produce correct output shape."""
    cfg = _cfg(use_xpos=False)
    model = XPosAttention(cfg, n_kv_heads=N_KV_HEADS)
    x = torch.randn(2, 12, D_MODEL)
    out = model(x)
    assert out.shape == (2, 12, D_MODEL)


# ---------------------------------------------------------------------------
# 5. Gradient flow
# ---------------------------------------------------------------------------


def test_gradient_flow():
    """Loss.backward() should populate gradients on all parameters."""
    cfg = _cfg()
    model = XPosAttention(cfg, n_kv_heads=N_KV_HEADS)
    x = torch.randn(1, 8, D_MODEL)
    out = model(x)
    loss = out.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


# ---------------------------------------------------------------------------
# 6. Causal mask application (output at position 0-3 unaffected by tokens 4+)
# ---------------------------------------------------------------------------


def test_causal_mask_applied():
    """The model uses a causal mask; past positions unaffected by future tokens."""
    torch.manual_seed(42)
    cfg = _cfg()
    model = XPosAttention(cfg, n_kv_heads=N_KV_HEADS)
    model.eval()

    x = torch.randn(1, 8, D_MODEL)
    out1 = model(x)

    # Changing future tokens should not affect past positions
    x2 = x.clone()
    x2[:, 4:, :] = torch.randn(1, 4, D_MODEL)
    out2 = model(x2)

    # Position 0-3 outputs should be identical
    assert torch.allclose(out1[:, :4, :], out2[:, :4, :], atol=1e-5)


# ---------------------------------------------------------------------------
# 7. Different sequence lengths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_len", [4, 16, 32])
def test_different_seq_lengths(seq_len: int):
    """XPosAttention must handle arbitrary sequence lengths."""
    cfg = _cfg()
    model = XPosAttention(cfg, n_kv_heads=N_KV_HEADS)
    x = torch.randn(1, seq_len, D_MODEL)
    out = model(x)
    assert out.shape == (1, seq_len, D_MODEL)


# ---------------------------------------------------------------------------
# 8. Determinism
# ---------------------------------------------------------------------------


def test_determinism():
    """Two forward passes with same input should produce identical output."""
    cfg = _cfg()
    model = XPosAttention(cfg, n_kv_heads=N_KV_HEADS)
    model.eval()
    x = torch.randn(2, 8, D_MODEL)
    out1 = model(x)
    out2 = model(x)
    assert torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# 9. Head dim correctness validated by config
# ---------------------------------------------------------------------------


def test_head_dim_correctness():
    """XPosConfig raises if d_model != n_heads * head_dim."""
    with pytest.raises((AssertionError, Exception)):
        # 63 != 4 * 16 = 64
        XPosConfig(d_model=63, n_heads=4, head_dim=16)


# ---------------------------------------------------------------------------
# 10. Batch size 1 and 4 produce correct shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 4])
def test_batch_sizes(batch_size: int):
    """Batch size 1 and 4 should both return correct output shapes."""
    cfg = _cfg()
    model = XPosAttention(cfg, n_kv_heads=N_KV_HEADS)
    x = torch.randn(batch_size, 8, D_MODEL)
    out = model(x)
    assert out.shape == (batch_size, 8, D_MODEL)
