"""Tests for MLA attention."""
from __future__ import annotations

import torch

from src.model.mla import MLAConfig, MultiHeadLatentAttention


def test_mla_kv_cache_shape():
    cfg = MLAConfig(d_model=128, n_heads=4, head_dim=32, kv_lora_rank=64)
    mla = MultiHeadLatentAttention(cfg)
    x = torch.randn(2, 8, 128)
    out, latent = mla(x)
    assert out.shape == (2, 8, 128)
    assert latent.shape == (2, 8, 64)


def test_mla_past_kv_concat():
    cfg = MLAConfig(d_model=128, n_heads=4, head_dim=32, kv_lora_rank=64)
    mla = MultiHeadLatentAttention(cfg)
    past_kv = torch.randn(2, 4, 64)
    x = torch.randn(2, 8, 128)
    out, latent = mla(x, past_kv=past_kv)
    assert latent.shape == (2, 12, 64)


def test_mla_output_not_nan():
    cfg = MLAConfig()
    mla = MultiHeadLatentAttention(cfg)
    x = torch.randn(2, 8, cfg.d_model)
    out, _ = mla(x)
    assert not torch.isnan(out).any()


def test_kv_cache_savings():
    cfg = MLAConfig(n_heads=8, head_dim=64, kv_lora_rank=128)
    from src.model.mla import compute_kv_cache_savings
    result = compute_kv_cache_savings(cfg)
    assert result["standard_per_token"] == 1024
    assert result["mla_per_token"] == 128
    assert result["compression_ratio"] > 1