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
    out, kv = attn(x, freqs)
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
        out, kv = attn(x, freqs)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_causal_mask_applied(attn, small_cfg):
    """With causal masking, position i should not attend to position j > i."""
    torch.manual_seed(42)
    x = torch.randn(1, 8, small_cfg.d_model)
    freqs = precompute_rope_frequencies(small_cfg.head_dim, 8)
    with torch.no_grad():
        out1, _ = attn(x, freqs)
    # Modify future tokens and verify past token outputs don't change
    x2 = x.clone()
    x2[:, 4:, :] = torch.randn_like(x2[:, 4:, :])
    with torch.no_grad():
        out2, _ = attn(x2, freqs)
    # First 4 positions should be identical (causal attention)
    assert torch.allclose(out1[:, :4, :], out2[:, :4, :], atol=1e-5)


def test_batch_size_one(attn, small_cfg):
    x = torch.randn(1, 8, small_cfg.d_model)
    freqs = precompute_rope_frequencies(small_cfg.head_dim, 8)
    out, kv = attn(x, freqs)
    assert out.shape == (1, 8, small_cfg.d_model)


def test_kv_cache_incremental_matches_full(attn, small_cfg):
    """Prefill + one decode step must match the full-sequence result."""
    torch.manual_seed(0)
    seq = torch.randn(1, 8, small_cfg.d_model)
    freqs_full = precompute_rope_frequencies(small_cfg.head_dim, 8)

    # Full forward pass
    out_full, _ = attn(seq, freqs_full)

    # Prefill (first 7 tokens)
    freqs_prefix = precompute_rope_frequencies(small_cfg.head_dim, 8)[:7]
    out_prefix, kv = attn(seq[:, :7, :], freqs_prefix)

    # Decode (token 8, offset 7)
    freqs_decode = precompute_rope_frequencies(small_cfg.head_dim, 8)[7:8]
    out_decode, _ = attn(seq[:, 7:8, :], freqs_decode, past_kv=kv)

    # The decode output for position 7 must match full forward at position 7
    assert torch.allclose(out_full[:, 7:8, :], out_decode, atol=1e-4), \
        f"Max diff: {(out_full[:, 7:8, :] - out_decode).abs().max()}"


def test_kv_cache_shape(attn, small_cfg):
    """Cache should store pre-expansion KV (n_kv_heads, not n_heads)."""
    x = torch.randn(2, 10, small_cfg.d_model)
    freqs = precompute_rope_frequencies(small_cfg.head_dim, 10)
    _, (k, v) = attn(x, freqs)
    # Cache stores n_kv_heads (pre-expansion), seq dim grows with tokens
    assert k.shape == (2, 10, small_cfg.n_kv_heads, small_cfg.head_dim)
    assert v.shape == (2, 10, small_cfg.n_kv_heads, small_cfg.head_dim)


def test_kv_cache_grows(attn, small_cfg):
    """Each decode step the cache seq dim should grow by 1."""
    x = torch.randn(1, 4, small_cfg.d_model)
    freqs = precompute_rope_frequencies(small_cfg.head_dim, 4)
    _, kv = attn(x, freqs)
    assert kv[0].shape[1] == 4

    x_next = torch.randn(1, 1, small_cfg.d_model)
    freqs_next = precompute_rope_frequencies(small_cfg.head_dim, 5)[4:5]
    _, kv2 = attn(x_next, freqs_next, past_kv=kv)
    assert kv2[0].shape[1] == 5


def test_kv_cache_batch_decode(attn, small_cfg):
    """KV cache incremental decode works correctly for batch_size > 1."""
    torch.manual_seed(1)
    B = 2
    seq = torch.randn(B, 6, small_cfg.d_model)
    freqs_full = precompute_rope_frequencies(small_cfg.head_dim, 6)

    # Full forward (reference)
    out_full, _ = attn(seq, freqs_full)

    # Prefill 5 tokens
    freqs_prefix = freqs_full[:5]
    _, kv = attn(seq[:, :5, :], freqs_prefix)

    # Decode token 6
    freqs_decode = freqs_full[5:6]
    out_decode, _ = attn(seq[:, 5:6, :], freqs_decode, past_kv=kv)

    assert out_decode.shape == (B, 1, small_cfg.d_model)
    assert torch.allclose(out_full[:, 5:6, :], out_decode, atol=1e-4), \
        f"Max diff: {(out_full[:, 5:6, :] - out_decode).abs().max()}"
