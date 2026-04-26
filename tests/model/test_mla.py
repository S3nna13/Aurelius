"""Tests for Multi-head Latent Attention (MLA) — DeepSeek-V2 KV cache compression."""

from __future__ import annotations

import pytest
import torch

from src.model.mla import (
    DownProjectKV,
    MLABlock,
    MLAConfig,
    MultiHeadLatentAttention,
    UpProjectKV,
    compute_kv_cache_savings,
)

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------
B = 2
T = 8
D_MODEL = 64
N_HEADS = 4
HEAD_DIM = 16
KV_LORA_RANK = 16
D_FF = 128


def make_config(**overrides) -> MLAConfig:
    defaults = dict(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        head_dim=HEAD_DIM,
        kv_lora_rank=KV_LORA_RANK,
    )
    defaults.update(overrides)
    return MLAConfig(**defaults)


# ---------------------------------------------------------------------------
# 1. MLAConfig defaults
# ---------------------------------------------------------------------------
def test_mla_config_defaults():
    cfg = MLAConfig()
    assert cfg.d_model == 512
    assert cfg.n_heads == 8
    assert cfg.head_dim == 64
    assert cfg.kv_lora_rank == 64
    assert cfg.q_lora_rank == 0
    assert cfg.rope_dim == 32
    assert cfg.dropout == 0.0


# ---------------------------------------------------------------------------
# 2. DownProjectKV output shape
# ---------------------------------------------------------------------------
def test_down_project_kv_shape():
    down = DownProjectKV(D_MODEL, KV_LORA_RANK)
    x = torch.randn(B, T, D_MODEL)
    out = down(x)
    assert out.shape == (B, T, KV_LORA_RANK)


# ---------------------------------------------------------------------------
# 3. UpProjectKV output shapes
# ---------------------------------------------------------------------------
def test_up_project_kv_shapes():
    up = UpProjectKV(KV_LORA_RANK, N_HEADS, HEAD_DIM)
    c = torch.randn(B, T, KV_LORA_RANK)
    K, V = up(c)
    assert K.shape == (B, N_HEADS, T, HEAD_DIM)
    assert V.shape == (B, N_HEADS, T, HEAD_DIM)


# ---------------------------------------------------------------------------
# 4. MultiHeadLatentAttention output shape
# ---------------------------------------------------------------------------
def test_mla_output_shape():
    cfg = make_config()
    mla = MultiHeadLatentAttention(cfg)
    x = torch.randn(B, T, D_MODEL)
    out, _ = mla(x)
    assert out.shape == (B, T, D_MODEL)


# ---------------------------------------------------------------------------
# 5. MultiHeadLatentAttention returns latent cache of correct shape
# ---------------------------------------------------------------------------
def test_mla_latent_cache_shape():
    cfg = make_config()
    mla = MultiHeadLatentAttention(cfg)
    x = torch.randn(B, T, D_MODEL)
    _, latent = mla(x)
    assert latent.shape == (B, T, KV_LORA_RANK)


# ---------------------------------------------------------------------------
# 6. MultiHeadLatentAttention with past_kv extends sequence length
# ---------------------------------------------------------------------------
def test_mla_past_kv_extends_seq():
    cfg = make_config()
    mla = MultiHeadLatentAttention(cfg)
    x = torch.randn(B, T, D_MODEL)
    _, past_cache = mla(x)

    # Single new token with past cache
    x_new = torch.randn(B, 1, D_MODEL)
    # Need mask since we have past_kv and T>1 total
    out, new_cache = mla(x_new, past_kv=past_cache)
    assert out.shape == (B, 1, D_MODEL)
    assert new_cache.shape == (B, T + 1, KV_LORA_RANK)


# ---------------------------------------------------------------------------
# 7. MultiHeadLatentAttention with causal mask doesn't error
# ---------------------------------------------------------------------------
def test_mla_with_causal_mask():
    cfg = make_config()
    mla = MultiHeadLatentAttention(cfg)
    x = torch.randn(B, T, D_MODEL)
    # Create a causal mask: (1, 1, T, T)
    mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
    mask = mask.unsqueeze(0).unsqueeze(0)
    out, _ = mla(x, mask=mask)
    assert out.shape == (B, T, D_MODEL)


# ---------------------------------------------------------------------------
# 8. MLABlock output shape
# ---------------------------------------------------------------------------
def test_mla_block_output_shape():
    cfg = make_config()
    block = MLABlock(cfg, d_ff=D_FF)
    x = torch.randn(B, T, D_MODEL)
    out, _ = block(x)
    assert out.shape == (B, T, D_MODEL)


# ---------------------------------------------------------------------------
# 9. MLABlock residual connection (output != 0)
# ---------------------------------------------------------------------------
def test_mla_block_residual_nonzero():
    cfg = make_config()
    block = MLABlock(cfg, d_ff=D_FF)
    x = torch.randn(B, T, D_MODEL)
    out, _ = block(x)
    assert out.abs().sum().item() > 0.0


# ---------------------------------------------------------------------------
# 10. compute_kv_cache_savings correct compression ratio
# ---------------------------------------------------------------------------
def test_kv_cache_savings_compression_ratio():
    cfg = make_config()
    savings = compute_kv_cache_savings(cfg)
    expected_standard = 2 * N_HEADS * HEAD_DIM  # 2 * 4 * 16 = 128
    expected_mla = KV_LORA_RANK  # 16
    expected_ratio = expected_standard / expected_mla  # 8.0
    assert savings["standard_per_token"] == expected_standard
    assert savings["mla_per_token"] == expected_mla
    assert savings["compression_ratio"] == pytest.approx(expected_ratio)


# ---------------------------------------------------------------------------
# 11. compute_kv_cache_savings MLA < standard per token
# ---------------------------------------------------------------------------
def test_kv_cache_savings_mla_smaller():
    cfg = make_config()
    savings = compute_kv_cache_savings(cfg)
    assert savings["mla_per_token"] < savings["standard_per_token"]


# ---------------------------------------------------------------------------
# 12. MultiHeadLatentAttention gradients flow
# ---------------------------------------------------------------------------
def test_mla_gradients_flow():
    cfg = make_config()
    mla = MultiHeadLatentAttention(cfg)
    x = torch.randn(B, T, D_MODEL, requires_grad=True)
    out, _ = mla(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum().item() > 0.0


# ---------------------------------------------------------------------------
# 13. MLABlock with past_kv caching works across two steps
# ---------------------------------------------------------------------------
def test_mla_block_past_kv_two_steps():
    cfg = make_config()
    block = MLABlock(cfg, d_ff=D_FF)

    # Step 1: full sequence
    x1 = torch.randn(B, T, D_MODEL)
    out1, cache1 = block(x1)
    assert cache1.shape == (B, T, KV_LORA_RANK)

    # Step 2: single token with past cache
    x2 = torch.randn(B, 1, D_MODEL)
    out2, cache2 = block(x2, past_kv=cache1)
    assert out2.shape == (B, 1, D_MODEL)
    assert cache2.shape == (B, T + 1, KV_LORA_RANK)
