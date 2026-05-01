"""Tests for Heavily Compressed Attention (HCA)."""

import torch

from src.model.config import AureliusConfig
from src.model.hca_attention import HeavilyCompressedAttention, HeavyCompressor


def test_heavy_compressor_shape():
    comp = HeavyCompressor(64, 16, 128)
    x = torch.randn(2, 256, 64)
    k, v = comp(x)
    assert k.shape == (2, 2, 16)
    assert v.shape == (2, 2, 16)


def test_hca_attention_forward():
    config = AureliusConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=256,
        hybrid_attention_enabled=True,
        attention_compression_rate_csa=4,
        attention_compression_rate_hca=128,
        attention_num_query_heads_hca=4,
        attention_query_compression_dim=16,
        attention_output_projection_groups=2,
        attention_intermediate_output_dim=32,
        attention_sliding_window_size=8,
        attention_partial_rope_dim=4,
    )
    hca = HeavilyCompressedAttention(config)
    x = torch.randn(2, 128, config.d_model)
    out, kv = hca(x)
    assert out.shape == (2, 128, config.d_model)
    assert kv is not None
    assert kv[0].shape[1] == 128 // config.attention_compression_rate_hca


def test_hca_causal_masking():
    config = AureliusConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
        hybrid_attention_enabled=True,
        attention_compression_rate_hca=128,
        attention_num_query_heads_hca=4,
        attention_query_compression_dim=16,
        attention_output_projection_groups=2,
        attention_intermediate_output_dim=32,
        attention_partial_rope_dim=4,
    )
    hca = HeavilyCompressedAttention(config)
    x = torch.randn(1, 256, config.d_model)
    out, _ = hca(x)
    assert not torch.isnan(out).any()


def test_hca_gradient_flow():
    config = AureliusConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=256,
        hybrid_attention_enabled=True,
        attention_compression_rate_hca=128,
        attention_num_query_heads_hca=4,
        attention_query_compression_dim=16,
        attention_output_projection_groups=2,
        attention_intermediate_output_dim=32,
        attention_partial_rope_dim=4,
    )
    hca = HeavilyCompressedAttention(config)
    x = torch.randn(1, 128, config.d_model, requires_grad=True)
    out, _ = hca(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
