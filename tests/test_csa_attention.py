"""Tests for Compressed Sparse Attention (CSA)."""

import torch

from src.model.config import AureliusConfig
from src.model.csa_attention import (
    CompressedSparseAttention,
    LightningIndexer,
    TokenLevelCompressor,
)


def test_token_level_compressor_shape():
    config = AureliusConfig(
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
        attention_compression_rate_csa=4,
    )
    comp = TokenLevelCompressor(config.d_model, config.head_dim, 4)
    x = torch.randn(2, 32, config.d_model)
    k, v = comp(x)
    assert k.shape == (2, 32 // 4, config.head_dim)
    assert v.shape == (2, 32 // 4, config.head_dim)


def test_token_level_compressor_values():
    comp = TokenLevelCompressor(32, 8, 4)
    x = torch.ones(1, 16, 32)
    k, v = comp(x)
    assert not torch.isnan(k).any()
    assert not torch.isnan(v).any()
    assert k.shape == (1, 4, 8)


def test_lightning_indexer_shape():
    d_model, d_c, c_i, n_h_i = 64, 16, 8, 4
    indexer = LightningIndexer(d_model, d_c, c_i, n_h_i, compression_rate=4, top_k=16)
    h_q = torch.randn(2, 8, d_model)
    h_kv = torch.randn(2, 16, d_model)
    scores, indices, c_q = indexer(h_q, h_kv, compressed_rate=4)
    assert scores.shape == (2, 8, 4)
    assert indices.shape == (2, 8, 4)


def test_csa_attention_forward():
    config = AureliusConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
        hybrid_attention_enabled=True,
        attention_compression_rate_csa=4,
        attention_compression_rate_hca=128,
        attention_num_indexer_heads=4,
        attention_indexer_head_dim=8,
        attention_top_k=8,
        attention_num_query_heads_csa=4,
        attention_num_query_heads_hca=4,
        attention_query_compression_dim=16,
        attention_output_projection_groups=2,
        attention_intermediate_output_dim=32,
        attention_sliding_window_size=8,
        attention_partial_rope_dim=4,
    )
    csa = CompressedSparseAttention(config)
    x = torch.randn(2, 16, config.d_model)
    out, kv = csa(x)
    assert out.shape == (2, 16, config.d_model)
    assert kv is not None
    assert kv[0].shape[1] == 16 // config.attention_compression_rate_csa


def test_csa_attention_gradient_flow():
    config = AureliusConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
        hybrid_attention_enabled=True,
        attention_compression_rate_csa=4,
        attention_compression_rate_hca=128,
        attention_num_indexer_heads=4,
        attention_indexer_head_dim=8,
        attention_top_k=8,
        attention_num_query_heads_csa=4,
        attention_num_query_heads_hca=4,
        attention_query_compression_dim=16,
        attention_output_projection_groups=2,
        attention_intermediate_output_dim=32,
        attention_sliding_window_size=8,
        attention_partial_rope_dim=4,
    )
    csa = CompressedSparseAttention(config)
    x = torch.randn(1, 8, config.d_model, requires_grad=True)
    out, _ = csa(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
