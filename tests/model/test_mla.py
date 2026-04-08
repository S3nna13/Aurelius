"""Tests for Multi-head Latent Attention (MLA) — DeepSeek-V2 KV cache compression."""
from __future__ import annotations

import torch
import pytest

from src.model.config import AureliusConfig
from src.model.mla import MLAConfig, MultiHeadLatentAttention
from src.model.attention import precompute_rope_frequencies


# ---------------------------------------------------------------------------
# Small test fixtures
# ---------------------------------------------------------------------------

SMALL_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=4,
    n_kv_heads=2,
    head_dim=16,
    d_ff=128,
    vocab_size=256,
    max_seq_len=32,
)

SMALL_MLA_CFG = MLAConfig(
    kv_lora_rank=32,
    q_lora_rank=48,
    rope_head_dim=8,
)

B, S = 2, 8  # batch size and sequence length


def make_mla() -> MultiHeadLatentAttention:
    return MultiHeadLatentAttention(SMALL_CFG, SMALL_MLA_CFG)


def make_inputs(seq_len: int = S):
    x = torch.randn(B, seq_len, SMALL_CFG.d_model)
    freqs = precompute_rope_frequencies(SMALL_CFG.head_dim, seq_len)
    return x, freqs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_mla_forward_shape():
    """Output tensor has shape (B, S, D)."""
    mla = make_mla()
    x, freqs = make_inputs()
    out, _ = mla(x, freqs)
    assert out.shape == (B, S, SMALL_CFG.d_model), f"Expected {(B, S, SMALL_CFG.d_model)}, got {out.shape}"


def test_mla_kv_cache_shape():
    """Returned c_KV has shape (B, S, kv_lora_rank)."""
    mla = make_mla()
    x, freqs = make_inputs()
    _, c_KV = mla(x, freqs)
    assert c_KV.shape == (B, S, SMALL_MLA_CFG.kv_lora_rank), (
        f"Expected {(B, S, SMALL_MLA_CFG.kv_lora_rank)}, got {c_KV.shape}"
    )


def test_mla_kv_cache_compression_ratio():
    """kv_cache_memory_ratio() > 1.0 — MLA is more memory-efficient than GQA."""
    mla = make_mla()
    ratio = mla.kv_cache_memory_ratio()
    assert ratio > 1.0, f"Expected ratio > 1.0, got {ratio}"


def test_mla_with_past_kv():
    """forward() with past_kv_latent doesn't crash and output shape is correct."""
    mla = make_mla()
    x, freqs = make_inputs()
    # First forward to get initial cache
    _, past_cache = mla(x, freqs)

    # Second forward: single new token with past cache
    x_new = torch.randn(B, 1, SMALL_CFG.d_model)
    freqs_new = precompute_rope_frequencies(SMALL_CFG.head_dim, 1)
    out, new_cache = mla(x_new, freqs_new, past_kv_latent=past_cache)

    assert out.shape == (B, 1, SMALL_CFG.d_model)
    assert new_cache.shape == (B, S + 1, SMALL_MLA_CFG.kv_lora_rank)


def test_mla_past_kv_cache_grows():
    """Cache shape increases from (B, S, r) to (B, 2S, r) after two full-sequence passes."""
    mla = make_mla()
    x, freqs = make_inputs()

    _, cache1 = mla(x, freqs)
    assert cache1.shape == (B, S, SMALL_MLA_CFG.kv_lora_rank)

    # Pass the whole sequence again with the cache (mask to avoid is_causal conflict)
    mask = torch.zeros(B, 1, S, S + S, dtype=torch.float32)  # allow all attention
    _, cache2 = mla(x, freqs, mask=mask, past_kv_latent=cache1)
    assert cache2.shape == (B, S + S, SMALL_MLA_CFG.kv_lora_rank), (
        f"Expected {(B, S + S, SMALL_MLA_CFG.kv_lora_rank)}, got {cache2.shape}"
    )


def test_mla_no_bias_in_projections():
    """All projection linear layers have bias=False."""
    mla = make_mla()
    proj_names = ["W_DKV", "W_UK", "W_UV", "q_proj", "o_proj"]
    for name in proj_names:
        layer = getattr(mla, name)
        assert layer.bias is None, f"{name} should have bias=False"


def test_mla_output_dtype():
    """Output tensor has the same dtype as input."""
    mla = make_mla()
    x, freqs = make_inputs()
    out, _ = mla(x, freqs)
    assert out.dtype == x.dtype, f"Expected dtype {x.dtype}, got {out.dtype}"


def test_mla_kv_ratio_formula():
    """kv_cache_memory_ratio() == 2 * n_kv_heads * head_dim / kv_lora_rank."""
    mla = make_mla()
    expected = (2 * SMALL_CFG.n_kv_heads * SMALL_CFG.head_dim) / SMALL_MLA_CFG.kv_lora_rank
    assert mla.kv_cache_memory_ratio() == pytest.approx(expected), (
        f"Expected {expected}, got {mla.kv_cache_memory_ratio()}"
    )
