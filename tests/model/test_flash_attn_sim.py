"""Tests for flash_attn_sim.py — tiled flash attention simulation."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from src.model.flash_attn_sim import (
    FlashAttnConfig,
    TiledAttention,
    compare_with_standard_attention,
    compute_memory_usage,
    flash_attention_forward,
    online_softmax_update,
)

# ---------------------------------------------------------------------------
# Shared test dimensions
# ---------------------------------------------------------------------------
B, H, T, D = 2, 2, 32, 16       # batch, heads, seq_len, head_dim
d_model = 32                      # H * D
n_heads = H
BLOCK = 8                         # block_size for tests


def make_qkv(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    return q, k, v


# ---------------------------------------------------------------------------
# 1. FlashAttnConfig defaults
# ---------------------------------------------------------------------------
def test_flash_attn_config_defaults():
    cfg = FlashAttnConfig()
    assert cfg.block_size == 32
    assert cfg.kv_block_size == 32
    assert cfg.causal is True
    assert cfg.dropout_p == 0.0
    assert cfg.scale is None


# ---------------------------------------------------------------------------
# 2. flash_attention_forward output shape
# ---------------------------------------------------------------------------
def test_flash_attention_forward_shape():
    q, k, v = make_qkv()
    cfg = FlashAttnConfig(block_size=BLOCK, kv_block_size=BLOCK, causal=True)
    out = flash_attention_forward(q, k, v, cfg)
    assert out.shape == (B, H, T, D), f"Expected {(B, H, T, D)}, got {out.shape}"


# ---------------------------------------------------------------------------
# 3. Causal masking — position i must not attend to positions j > i
# ---------------------------------------------------------------------------
def test_flash_attention_causal_mask():
    """Verify causality by checking outputs shift when future values change."""
    torch.manual_seed(0)
    q = torch.randn(1, 1, 8, 8)
    k = torch.randn(1, 1, 8, 8)
    v1 = torch.randn(1, 1, 8, 8)
    v2 = v1.clone()
    # Modify only the last 4 positions of v2
    v2[:, :, 4:, :] += 100.0

    cfg = FlashAttnConfig(block_size=4, kv_block_size=4, causal=True)
    out1 = flash_attention_forward(q, k, v1, cfg)
    out2 = flash_attention_forward(q, k, v2, cfg)

    # First 4 query positions cannot attend to positions 4-7 → outputs must be identical
    assert torch.allclose(out1[:, :, :4, :], out2[:, :, :4, :], atol=1e-5), \
        "Causal masking failed: early query positions should not see future values"


# ---------------------------------------------------------------------------
# 4. flash_attention_forward matches F.scaled_dot_product_attention
# ---------------------------------------------------------------------------
def test_flash_attention_matches_standard():
    q, k, v = make_qkv()
    cfg = FlashAttnConfig(block_size=BLOCK, kv_block_size=BLOCK, causal=True)
    tiled_out = flash_attention_forward(q, k, v, cfg)

    std_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    assert torch.allclose(tiled_out.float(), std_out.float(), atol=1e-4), \
        f"Max diff: {(tiled_out - std_out).abs().max().item():.6f}"


# ---------------------------------------------------------------------------
# 5. online_softmax_update — output shapes correct
# ---------------------------------------------------------------------------
def test_online_softmax_update_shapes():
    torch.manual_seed(0)
    T_q, T_kv = 8, 8
    prev_max = torch.full((B, H, T_q, 1), float("-inf"))
    prev_sum = torch.zeros(B, H, T_q, 1)
    prev_out = torch.zeros(B, H, T_q, D)
    new_scores = torch.randn(B, H, T_q, T_kv)
    new_values = torch.randn(B, H, T_kv, D)

    new_max, new_sum, new_out = online_softmax_update(
        prev_max, prev_sum, prev_out, new_scores, new_values
    )

    assert new_max.shape == (B, H, T_q, 1)
    assert new_sum.shape == (B, H, T_q, 1)
    assert new_out.shape == (B, H, T_q, D)


# ---------------------------------------------------------------------------
# 6. online_softmax_correctness — single-step matches direct softmax
# ---------------------------------------------------------------------------
def test_online_softmax_correctness():
    """For a single KV chunk starting from zero, result equals direct softmax."""
    torch.manual_seed(0)
    T_q, T_kv = 4, 8
    prev_max = torch.full((1, 1, T_q, 1), float("-inf"))
    prev_sum = torch.zeros(1, 1, T_q, 1)
    prev_out = torch.zeros(1, 1, T_q, D)
    scores = torch.randn(1, 1, T_q, T_kv)
    values = torch.randn(1, 1, T_kv, D)

    new_max, new_sum, new_out = online_softmax_update(
        prev_max, prev_sum, prev_out, scores, values
    )

    # Direct computation
    weights = torch.softmax(scores, dim=-1)          # (1, 1, T_q, T_kv)
    expected_out = torch.matmul(weights, values)     # (1, 1, T_q, D)
    online_normalized = new_out / new_sum.clamp(min=1e-12)

    assert torch.allclose(online_normalized, expected_out, atol=1e-5), \
        f"Max diff: {(online_normalized - expected_out).abs().max().item():.6f}"


# ---------------------------------------------------------------------------
# 7. TiledAttention output shape
# ---------------------------------------------------------------------------
def test_tiled_attention_output_shape():
    torch.manual_seed(0)
    cfg = FlashAttnConfig(block_size=BLOCK, kv_block_size=BLOCK, causal=True)
    model = TiledAttention(d_model=d_model, n_heads=n_heads, config=cfg)
    x = torch.randn(B, T, d_model)
    out = model(x)
    assert out.shape == (B, T, d_model), f"Expected {(B, T, d_model)}, got {out.shape}"


# ---------------------------------------------------------------------------
# 8. TiledAttention gradient flow
# ---------------------------------------------------------------------------
def test_tiled_attention_gradient_flow():
    torch.manual_seed(0)
    cfg = FlashAttnConfig(block_size=BLOCK, kv_block_size=BLOCK, causal=True)
    model = TiledAttention(d_model=d_model, n_heads=n_heads, config=cfg)
    x = torch.randn(B, T, d_model, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "Input gradient is None"
    assert x.grad.shape == x.shape, "Input gradient shape mismatch"
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"


# ---------------------------------------------------------------------------
# 9. compare_with_standard_attention — results are close
# ---------------------------------------------------------------------------
def test_compare_with_standard_close():
    q, k, v = make_qkv()
    std_out, tiled_out = compare_with_standard_attention(q, k, v, causal=True)
    assert std_out.shape == tiled_out.shape
    assert torch.allclose(std_out.float(), tiled_out.float(), atol=1e-3), \
        f"Max diff: {(std_out - tiled_out).abs().max().item():.6f}"


# ---------------------------------------------------------------------------
# 10. compute_memory_usage — standard uses more memory than tiled for large T
# ---------------------------------------------------------------------------
def test_compute_memory_usage_standard_larger():
    info = compute_memory_usage(seq_len=512, n_heads=8, head_dim=64, block_size=32)
    assert info["standard_bytes"] > info["tiled_bytes"], \
        "Standard attention should use more memory than tiled for large sequences"


# ---------------------------------------------------------------------------
# 11. compute_memory_usage — reduction_factor > 1
# ---------------------------------------------------------------------------
def test_compute_memory_usage_reduction_positive():
    info = compute_memory_usage(seq_len=1024, n_heads=4, head_dim=64, block_size=32)
    assert info["reduction_factor"] > 1.0, \
        f"Expected reduction_factor > 1, got {info['reduction_factor']}"


# ---------------------------------------------------------------------------
# 12. flash_attention_forward — same result for different block sizes
# ---------------------------------------------------------------------------
def test_flash_attention_different_block_sizes():
    q, k, v = make_qkv()

    cfg8 = FlashAttnConfig(block_size=8, kv_block_size=8, causal=True)
    cfg16 = FlashAttnConfig(block_size=16, kv_block_size=16, causal=True)

    out8 = flash_attention_forward(q, k, v, cfg8)
    out16 = flash_attention_forward(q, k, v, cfg16)

    assert torch.allclose(out8.float(), out16.float(), atol=1e-4), \
        f"Results differ between block_size=8 and block_size=16; max diff: {(out8 - out16).abs().max().item():.6f}"
