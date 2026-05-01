"""Tests for transformer with hybrid attention and mHC integration."""

import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


def test_transformer_default_config_still_works():
    config = AureliusConfig(
        d_model=64,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )
    model = AureliusTransformer(config)
    tokens = torch.randint(0, 256, (2, 16))
    loss, logits, past_kv = model(tokens, labels=tokens)
    assert logits.shape == (2, 16, 256)
    assert loss is not None
    assert loss > 0


def test_transformer_with_hybrid_attention():
    config = AureliusConfig(
        d_model=64,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=128,
        hybrid_attention_enabled=True,
        attention_compression_rate_csa=4,
        attention_compression_rate_hca=16,
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
    model = AureliusTransformer(config)
    tokens = torch.randint(0, 256, (1, 32))
    loss, logits, past_kv = model(tokens, labels=tokens)
    assert logits.shape == (1, 32, 256)
    assert loss is not None
    assert loss > 0


def test_transformer_with_mhc():
    config = AureliusConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
        mhc_enabled=True,
        mhc_expansion_factor=4,
        mhc_sinkhorn_iterations=10,
    )
    model = AureliusTransformer(config)
    tokens = torch.randint(0, 256, (1, 8))
    loss, logits, past_kv = model(tokens, labels=tokens)
    assert logits.shape == (1, 8, 256)
    assert loss is not None
    assert loss > 0


def test_transformer_with_hybrid_and_mhc():
    config = AureliusConfig(
        d_model=64,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
        hybrid_attention_enabled=True,
        attention_compression_rate_csa=4,
        attention_compression_rate_hca=16,
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
        mhc_enabled=True,
        mhc_expansion_factor=4,
        mhc_sinkhorn_iterations=10,
    )
    model = AureliusTransformer(config)
    tokens = torch.randint(0, 256, (1, 16))
    loss, logits, past_kv = model(tokens, labels=tokens)
    assert logits.shape == (1, 16, 256)
    assert loss is not None
    assert loss > 0


def test_transformer_with_moe_and_mhc():
    config = AureliusConfig(
        d_model=64,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
        moe_enabled=True,
        moe_num_experts=4,
        moe_top_k=2,
        moe_every_n_layers=2,
        mhc_enabled=True,
        mhc_expansion_factor=4,
        mhc_sinkhorn_iterations=10,
    )
    model = AureliusTransformer(config)
    tokens = torch.randint(0, 256, (1, 8))
    loss, logits, past_kv = model(tokens, labels=tokens)
    assert logits.shape == (1, 8, 256)


def test_transformer_gradient_flow_hybrid():
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
        attention_compression_rate_hca=16,
        attention_num_indexer_heads=4,
        attention_indexer_head_dim=8,
        attention_top_k=4,
        attention_num_query_heads_csa=4,
        attention_num_query_heads_hca=4,
        attention_query_compression_dim=16,
        attention_output_projection_groups=2,
        attention_intermediate_output_dim=32,
        attention_sliding_window_size=8,
        attention_partial_rope_dim=4,
    )
    model = AureliusTransformer(config)
    tokens = torch.randint(0, 256, (1, 8))
    loss, logits, _ = model(tokens, labels=tokens)
    loss.backward()
    has_grad = False
    for p in model.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad


def test_transformer_generate_hybrid():
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
        attention_compression_rate_hca=16,
        attention_num_indexer_heads=4,
        attention_indexer_head_dim=8,
        attention_top_k=4,
        attention_num_query_heads_csa=4,
        attention_num_query_heads_hca=4,
        attention_query_compression_dim=16,
        attention_output_projection_groups=2,
        attention_intermediate_output_dim=32,
        attention_sliding_window_size=8,
        attention_partial_rope_dim=4,
    )
    model = AureliusTransformer(config)
    model.eval()
    tokens = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        out = model.generate(tokens, max_new_tokens=4, temperature=0.8, top_p=0.9)
    assert out.shape[1] >= 8
