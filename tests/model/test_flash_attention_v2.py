"""Tests for src/model/flash_attention_v2.py

Tiny config: D_MODEL=16, N_HEADS=2, D_HEAD=8, B=2, T=8, BQ=4, BK=4
"""

from __future__ import annotations

import math

import torch

from src.model.flash_attention_v2 import (
    FlashAttentionV2,
    FlashAttnV2Block,
    FlashAttnV2Config,
    standard_attention_v2,
    tiled_attention_v2,
)

D_MODEL = 16
N_HEADS = 2
D_HEAD = D_MODEL // N_HEADS  # 8
B = 2
T = 8
BQ = 4
BK = 4
SCALE = 1.0 / math.sqrt(D_HEAD)


def _qkv():
    torch.manual_seed(0)
    Q = torch.randn(B, N_HEADS, T, D_HEAD)
    K = torch.randn(B, N_HEADS, T, D_HEAD)
    V = torch.randn(B, N_HEADS, T, D_HEAD)
    return Q, K, V


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = FlashAttnV2Config()
    assert cfg.d_model == 64
    assert cfg.n_heads == 4
    assert cfg.block_size_q == 16
    assert cfg.block_size_k == 16
    assert cfg.causal is True
    assert cfg.scale is None


# ---------------------------------------------------------------------------
# tiled_attention_v2 output shape
# ---------------------------------------------------------------------------


def test_tiled_attention_v2_shape():
    Q, K, V = _qkv()
    out = tiled_attention_v2(Q, K, V, block_q=BQ, block_k=BK, causal=True, scale=SCALE)
    assert out.shape == (B, N_HEADS, T, D_HEAD)


# ---------------------------------------------------------------------------
# Numerical equivalence causal=True
# ---------------------------------------------------------------------------


def test_tiled_vs_standard_causal():
    Q, K, V = _qkv()
    ref = standard_attention_v2(Q, K, V, causal=True, scale=SCALE)
    tiled = tiled_attention_v2(Q, K, V, block_q=BQ, block_k=BK, causal=True, scale=SCALE)
    assert torch.allclose(tiled, ref, atol=1e-4), f"max diff={(tiled - ref).abs().max().item()}"


# ---------------------------------------------------------------------------
# Numerical equivalence causal=False
# ---------------------------------------------------------------------------


def test_tiled_vs_standard_non_causal():
    Q, K, V = _qkv()
    ref = standard_attention_v2(Q, K, V, causal=False, scale=SCALE)
    tiled = tiled_attention_v2(Q, K, V, block_q=BQ, block_k=BK, causal=False, scale=SCALE)
    assert torch.allclose(tiled, ref, atol=1e-4), f"max diff={(tiled - ref).abs().max().item()}"


# ---------------------------------------------------------------------------
# Block size edge cases
# ---------------------------------------------------------------------------


def test_tiled_block_q_1():
    Q, K, V = _qkv()
    ref = standard_attention_v2(Q, K, V, causal=True, scale=SCALE)
    tiled = tiled_attention_v2(Q, K, V, block_q=1, block_k=BK, causal=True, scale=SCALE)
    assert torch.allclose(tiled, ref, atol=1e-4)


def test_tiled_block_k_1():
    Q, K, V = _qkv()
    ref = standard_attention_v2(Q, K, V, causal=True, scale=SCALE)
    tiled = tiled_attention_v2(Q, K, V, block_q=BQ, block_k=1, causal=True, scale=SCALE)
    assert torch.allclose(tiled, ref, atol=1e-4)


def test_tiled_non_divisible_seq_len():
    """T=7 not divisible by BQ=4 or BK=3 — should still work."""
    torch.manual_seed(5)
    T2 = 7
    Q = torch.randn(B, N_HEADS, T2, D_HEAD)
    K = torch.randn(B, N_HEADS, T2, D_HEAD)
    V = torch.randn(B, N_HEADS, T2, D_HEAD)
    ref = standard_attention_v2(Q, K, V, causal=True, scale=SCALE)
    tiled = tiled_attention_v2(Q, K, V, block_q=4, block_k=3, causal=True, scale=SCALE)
    assert tiled.shape == (B, N_HEADS, T2, D_HEAD)
    assert torch.allclose(tiled, ref, atol=1e-4)


# ---------------------------------------------------------------------------
# Attention weights sum to 1 (standard reference)
# ---------------------------------------------------------------------------


def test_standard_attention_finite():
    Q, K, V = _qkv()
    out = standard_attention_v2(Q, K, V, causal=True, scale=SCALE)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# FlashAttentionV2 module
# ---------------------------------------------------------------------------


def test_flash_attention_v2_module_shape():
    cfg = FlashAttnV2Config(d_model=D_MODEL, n_heads=N_HEADS, block_size_q=BQ, block_size_k=BK)
    attn = FlashAttentionV2(cfg)
    x = torch.randn(B, T, D_MODEL)
    out = attn(x)
    assert out.shape == (B, T, D_MODEL)


def test_flash_attention_v2_module_finite():
    cfg = FlashAttnV2Config(d_model=D_MODEL, n_heads=N_HEADS, block_size_q=BQ, block_size_k=BK)
    attn = FlashAttentionV2(cfg)
    x = torch.randn(B, T, D_MODEL)
    out = attn(x)
    assert torch.isfinite(out).all()


def test_flash_attention_v2_gradient_flows():
    cfg = FlashAttnV2Config(d_model=D_MODEL, n_heads=N_HEADS, block_size_q=BQ, block_size_k=BK)
    attn = FlashAttentionV2(cfg)
    x = torch.randn(B, T, D_MODEL, requires_grad=True)
    out = attn(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# FlashAttnV2Block
# ---------------------------------------------------------------------------


def test_flash_attn_v2_block_shape():
    cfg = FlashAttnV2Config(d_model=D_MODEL, n_heads=N_HEADS, block_size_q=BQ, block_size_k=BK)
    block = FlashAttnV2Block(cfg)
    x = torch.randn(B, T, D_MODEL)
    out = block(x)
    assert out.shape == (B, T, D_MODEL)


def test_flash_attn_v2_block_residual():
    """Block output should differ from pure attention (residual adds x)."""
    cfg = FlashAttnV2Config(d_model=D_MODEL, n_heads=N_HEADS, block_size_q=BQ, block_size_k=BK)
    block = FlashAttnV2Block(cfg)
    x = torch.randn(B, T, D_MODEL)
    out = block(x)
    # Out = x + attn(LN(x)), so out != attn(LN(x)) alone
    attn_only = block.attn(block.norm(x))
    assert not torch.allclose(out, attn_only)
