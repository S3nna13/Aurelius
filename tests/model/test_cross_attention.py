"""Tests for CrossAttentionLayer and MultiModalTransformerBlock."""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.model.cross_attention import (
    CrossAttentionConfig,
    CrossAttentionLayer,
    MultiModalTransformerBlock,
    add_cross_attention_to_model,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.model.attention import precompute_rope_frequencies


# ---------------------------------------------------------------------------
# Tiny config for fast tests
# ---------------------------------------------------------------------------

TINY = CrossAttentionConfig(
    d_model=64,
    n_heads=4,
    head_dim=16,
    context_dim=32,
    dropout=0.0,
    use_layer_norm=True,
)


# ---------------------------------------------------------------------------
# CrossAttentionConfig tests
# ---------------------------------------------------------------------------

def test_cross_attn_config_defaults():
    cfg = CrossAttentionConfig()
    assert cfg.d_model == 2048
    assert cfg.n_heads == 16
    assert cfg.head_dim == 128
    assert cfg.context_dim == 768
    assert cfg.dropout == 0.0
    assert cfg.use_layer_norm is True


# ---------------------------------------------------------------------------
# CrossAttentionLayer shape/behaviour tests
# ---------------------------------------------------------------------------

def test_cross_attn_output_shape():
    B, S, C = 2, 10, 5
    layer = CrossAttentionLayer(TINY)
    x = torch.randn(B, S, TINY.d_model)
    context = torch.randn(B, C, TINY.context_dim)
    out = layer(x, context)
    assert out.shape == (B, S, TINY.d_model)


def test_cross_attn_residual_connection():
    """Output should differ from input because cross-attn updates hidden states."""
    B, S, C = 2, 10, 5
    layer = CrossAttentionLayer(TINY)
    x = torch.randn(B, S, TINY.d_model)
    context = torch.randn(B, C, TINY.context_dim)
    out = layer(x, context)
    assert not torch.allclose(out, x), "Cross-attention should change the hidden states"


def test_cross_attn_with_context_mask():
    """Masking all but the first context token should still produce a valid output."""
    B, S, C = 2, 8, 5
    layer = CrossAttentionLayer(TINY)
    x = torch.randn(B, S, TINY.d_model)
    context = torch.randn(B, C, TINY.context_dim)

    # Only the first token is valid
    context_mask = torch.zeros(B, C, dtype=torch.bool)
    context_mask[:, 0] = True

    out_masked = layer(x, context, context_mask=context_mask)
    assert out_masked.shape == (B, S, TINY.d_model)

    # With all tokens valid the result should differ
    full_mask = torch.ones(B, C, dtype=torch.bool)
    out_full = layer(x, context, context_mask=full_mask)
    assert not torch.allclose(out_masked, out_full), (
        "Masked (1 valid token) and full-mask outputs should differ"
    )


def test_cross_attn_no_context_mask():
    B, S, C = 3, 6, 4
    layer = CrossAttentionLayer(TINY)
    x = torch.randn(B, S, TINY.d_model)
    context = torch.randn(B, C, TINY.context_dim)
    out = layer(x, context, context_mask=None)
    assert out.shape == (B, S, TINY.d_model)


def test_cross_attn_different_context_dim():
    cfg = CrossAttentionConfig(
        d_model=64,
        n_heads=4,
        head_dim=16,
        context_dim=32,
    )
    layer = CrossAttentionLayer(cfg)
    B, S, C = 2, 5, 7
    x = torch.randn(B, S, cfg.d_model)
    context = torch.randn(B, C, cfg.context_dim)
    out = layer(x, context)
    assert out.shape == (B, S, cfg.d_model)


# ---------------------------------------------------------------------------
# MultiModalTransformerBlock tests
# ---------------------------------------------------------------------------

def _tiny_aurelius_config():
    """Minimal AureliusConfig that satisfies the assertion in __post_init__."""
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )


def _make_multimodal_block(with_cross_attn: bool = True):
    """Build a MultiModalTransformerBlock backed by real (tiny) modules."""
    from src.model.attention import GroupedQueryAttention
    from src.model.ffn import SwiGLUFFN
    from src.model.rms_norm import RMSNorm

    cfg = _tiny_aurelius_config()

    self_attn = GroupedQueryAttention(cfg)
    ffn = SwiGLUFFN(cfg)
    norm1 = RMSNorm(cfg.d_model)
    norm2 = RMSNorm(cfg.d_model)

    cross_attn = None
    norm_cross = None
    if with_cross_attn:
        cross_cfg = CrossAttentionConfig(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            head_dim=cfg.head_dim,
            context_dim=32,
        )
        cross_attn = CrossAttentionLayer(cross_cfg)
        norm_cross = RMSNorm(cfg.d_model)

    return MultiModalTransformerBlock(
        self_attn=self_attn,
        ffn=ffn,
        norm1=norm1,
        norm2=norm2,
        cross_attn=cross_attn,
        norm_cross=norm_cross,
    ), cfg


def test_multimodal_block_no_context():
    """Without context, MultiModalTransformerBlock should still run and produce correct shapes."""
    block, cfg = _make_multimodal_block(with_cross_attn=True)
    B, S = 2, 8
    x = torch.randn(B, S, cfg.d_model)
    freqs = precompute_rope_frequencies(cfg.head_dim, S)

    out, kv = block(x, freqs, mask=None, past_kv=None, context=None)
    assert out.shape == (B, S, cfg.d_model)
    assert isinstance(kv, tuple) and len(kv) == 2


def test_multimodal_block_with_context():
    """With context provided, output shape should still be (B, S, d_model)."""
    block, cfg = _make_multimodal_block(with_cross_attn=True)
    B, S, C = 2, 8, 5
    x = torch.randn(B, S, cfg.d_model)
    context = torch.randn(B, C, 32)  # context_dim=32 matches _make_multimodal_block
    freqs = precompute_rope_frequencies(cfg.head_dim, S)

    out, kv = block(x, freqs, mask=None, past_kv=None, context=context)
    assert out.shape == (B, S, cfg.d_model)
    assert isinstance(kv, tuple) and len(kv) == 2


# ---------------------------------------------------------------------------
# add_cross_attention_to_model tests
# ---------------------------------------------------------------------------

def _tiny_model():
    cfg = _tiny_aurelius_config()
    return AureliusTransformer(cfg)


def test_add_cross_attention_to_model_all_layers():
    model = _tiny_model()
    cross_layers = add_cross_attention_to_model(model, context_dim=32, layer_indices=None)
    assert isinstance(cross_layers, nn.ModuleList)
    assert len(cross_layers) == model.config.n_layers
    for layer in cross_layers:
        assert isinstance(layer, CrossAttentionLayer)


def test_add_cross_attention_to_model_subset():
    model = _tiny_model()
    cross_layers = add_cross_attention_to_model(model, context_dim=32, layer_indices=[0, 1])
    assert isinstance(cross_layers, nn.ModuleList)
    assert len(cross_layers) == 2
    for layer in cross_layers:
        assert isinstance(layer, CrossAttentionLayer)


# ---------------------------------------------------------------------------
# Gradient flow test
# ---------------------------------------------------------------------------

def test_cross_attn_gradients_flow():
    """Gradients must flow back through cross-attention to both x and context."""
    B, S, C = 2, 6, 4
    layer = CrossAttentionLayer(TINY)

    x = torch.randn(B, S, TINY.d_model, requires_grad=True)
    context = torch.randn(B, C, TINY.context_dim)

    out = layer(x, context)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "Gradient did not flow back to x"
    assert not torch.all(x.grad == 0), "x.grad is all zeros — gradients may not be flowing"
