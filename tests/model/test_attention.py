"""Tests for Grouped Query Attention and RoPE."""
import torch
import pytest
from src.model.config import AureliusConfig
from src.model.attention import GroupedQueryAttention, precompute_rope_frequencies, apply_rope


@pytest.fixture
def small_cfg():
    return AureliusConfig(
        n_layers=2, d_model=256, n_heads=4, n_kv_heads=2,
        head_dim=64, d_ff=512, vocab_size=1000, max_seq_len=128,
    )


@pytest.fixture
def attn(small_cfg):
    return GroupedQueryAttention(small_cfg)


def test_output_shape(attn, small_cfg):
    x = torch.randn(2, 16, small_cfg.d_model)
    freqs = precompute_rope_frequencies(small_cfg.head_dim, 16)
    out = attn(x, freqs)
    assert out.shape == (2, 16, small_cfg.d_model)


def test_no_bias(attn):
    for name, _ in attn.named_parameters():
        assert "bias" not in name


def test_gqa_kv_weight_shape(attn, small_cfg):
    # KV projections should be smaller than Q projection (GQA)
    q_weight = attn.q_proj.weight
    k_weight = attn.k_proj.weight
    assert q_weight.shape[0] == small_cfg.n_heads * small_cfg.head_dim
    assert k_weight.shape[0] == small_cfg.n_kv_heads * small_cfg.head_dim


def test_rope_frequencies_shape():
    head_dim, seq_len = 64, 128
    freqs = precompute_rope_frequencies(head_dim, seq_len)
    assert freqs.shape == (seq_len, head_dim // 2)
    assert freqs.is_complex()


def test_rope_preserves_shape():
    x = torch.randn(2, 16, 4, 64)  # (batch, seq, heads, head_dim)
    freqs = precompute_rope_frequencies(64, 16)
    out = apply_rope(x, freqs)
    assert out.shape == x.shape


def test_rope_changes_values():
    """RoPE should rotate embeddings — output must differ from input."""
    x = torch.randn(1, 8, 2, 64)
    freqs = precompute_rope_frequencies(64, 8)
    out = apply_rope(x, freqs)
    assert not torch.allclose(x, out)


def test_no_nan_in_attention_output(attn, small_cfg):
    x = torch.randn(2, 16, small_cfg.d_model)
    freqs = precompute_rope_frequencies(small_cfg.head_dim, 16)
    with torch.no_grad():
        out = attn(x, freqs)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_causal_mask_applied(attn, small_cfg):
    """With causal masking, position i should not attend to position j > i."""
    torch.manual_seed(42)
    x = torch.randn(1, 8, small_cfg.d_model)
    freqs = precompute_rope_frequencies(small_cfg.head_dim, 8)
    with torch.no_grad():
        out1 = attn(x, freqs)
    # Modify future tokens and verify past token outputs don't change
    x2 = x.clone()
    x2[:, 4:, :] = torch.randn_like(x2[:, 4:, :])
    with torch.no_grad():
        out2 = attn(x2, freqs)
    # First 4 positions should be identical (causal attention)
    assert torch.allclose(out1[:, :4, :], out2[:, :4, :], atol=1e-5)


def test_batch_size_one(attn, small_cfg):
    x = torch.randn(1, 8, small_cfg.d_model)
    freqs = precompute_rope_frequencies(small_cfg.head_dim, 8)
    out = attn(x, freqs)
    assert out.shape == (1, 8, small_cfg.d_model)
