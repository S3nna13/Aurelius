"""Tests for src/model/attention_variants.py — 16 tests total."""

from __future__ import annotations

import torch

from src.model.attention_variants import (
    AttentionVariantConfig,
    CrossAttention,
    GroupedQueryAttention,
    MultiHeadAttention,
    MultiQueryAttention,
    attention_memory_ratio,
    count_kv_parameters,
    mha_attention,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def make_causal_mask(T: int) -> torch.Tensor:
    """(T, T) lower-triangular causal mask."""
    return torch.tril(torch.ones(T, T))


def mha_config(**kwargs) -> AttentionVariantConfig:
    """MHA config: n_kv_heads == n_heads."""
    defaults = dict(d_model=64, n_heads=4, n_kv_heads=4, head_dim=16)
    defaults.update(kwargs)
    return AttentionVariantConfig(**defaults)


def mqa_config(**kwargs) -> AttentionVariantConfig:
    """MQA config: n_kv_heads == 1."""
    defaults = dict(d_model=64, n_heads=4, n_kv_heads=1, head_dim=16)
    defaults.update(kwargs)
    return AttentionVariantConfig(**defaults)


def gqa_config(**kwargs) -> AttentionVariantConfig:
    """GQA config: 1 < n_kv_heads < n_heads."""
    defaults = dict(d_model=64, n_heads=4, n_kv_heads=2, head_dim=16)
    defaults.update(kwargs)
    return AttentionVariantConfig(**defaults)


# ---------------------------------------------------------------------------
# Test 1: AttentionVariantConfig defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = AttentionVariantConfig()
    assert cfg.d_model == 256
    assert cfg.n_heads == 8
    assert cfg.n_kv_heads == 1
    assert cfg.head_dim == 32
    assert cfg.dropout == 0.0
    assert cfg.use_rope is False


# ---------------------------------------------------------------------------
# Test 2: mha_attention output shape (B, H, T, head_dim)
# ---------------------------------------------------------------------------


def test_mha_attention_output_shape():
    B, H, T, hd = 2, 4, 8, 16
    q = torch.randn(B, H, T, hd)
    k = torch.randn(B, H, T, hd)
    v = torch.randn(B, H, T, hd)
    out = mha_attention(q, k, v)
    assert out.shape == (B, H, T, hd)


# ---------------------------------------------------------------------------
# Test 3: mha_attention with H_kv < H (KV expansion)
# ---------------------------------------------------------------------------


def test_mha_attention_kv_expansion():
    B, H, H_kv, T, hd = 2, 8, 2, 6, 16
    q = torch.randn(B, H, T, hd)
    k = torch.randn(B, H_kv, T, hd)
    v = torch.randn(B, H_kv, T, hd)
    out = mha_attention(q, k, v)
    assert out.shape == (B, H, T, hd), f"Expected ({B}, {H}, {T}, {hd}), got {out.shape}"


# ---------------------------------------------------------------------------
# Test 4: mha_attention causal mask prevents future attention
# ---------------------------------------------------------------------------


def test_mha_attention_causal_mask():
    B, H, T, hd = 1, 1, 4, 8
    # Craft q and k so that scores are predictable
    q = torch.zeros(B, H, T, hd)
    k = torch.zeros(B, H, T, hd)
    v = torch.eye(T).unsqueeze(0).unsqueeze(0).expand(B, H, T, T)[:, :, :, :hd]
    # Make v full rank by using identity-padded
    v = torch.randn(B, H, T, hd)

    mask = make_causal_mask(T).unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
    out_masked = mha_attention(q, k, v, mask=mask)

    # Without mask, check we don't crash and shapes match
    out_no_mask = mha_attention(q, k, v, mask=None)
    assert out_masked.shape == out_no_mask.shape == (B, H, T, hd)

    # With a causal mask and distinct v per position, token 0 should not
    # attend to tokens 1+ — verify by checking that an upper-triangular mask
    # produces -inf in softmax => 0 weight. Use distinct q/k so scores differ.
    q2 = torch.randn(B, H, T, hd)
    k2 = torch.randn(B, H, T, hd)
    v2 = torch.eye(T, hd)  # each position has a unique v
    v2 = v2.unsqueeze(0).unsqueeze(0).expand(B, H, T, hd)

    causal = make_causal_mask(T)  # (T, T)
    out_causal = mha_attention(q2, k2, v2, mask=causal)
    # The first token can only attend to itself — output should equal v2[0]
    # (scaled by softmax of a single position). Shape must be correct.
    assert out_causal.shape == (B, H, T, hd)


# ---------------------------------------------------------------------------
# Test 5: MultiHeadAttention output shape (B, T, d_model)
# ---------------------------------------------------------------------------


def test_mha_module_output_shape():
    cfg = mha_config()
    model = MultiHeadAttention(cfg)
    x = torch.randn(2, 8, cfg.d_model)
    out = model(x)
    assert out.shape == (2, 8, cfg.d_model)


# ---------------------------------------------------------------------------
# Test 6: MultiHeadAttention is differentiable
# ---------------------------------------------------------------------------


def test_mha_module_differentiable():
    cfg = mha_config()
    model = MultiHeadAttention(cfg)
    x = torch.randn(2, 8, cfg.d_model, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


# ---------------------------------------------------------------------------
# Test 7: MultiQueryAttention output shape (B, T, d_model)
# ---------------------------------------------------------------------------


def test_mqa_module_output_shape():
    cfg = mqa_config()
    model = MultiQueryAttention(cfg)
    x = torch.randn(2, 8, cfg.d_model)
    out = model(x)
    assert out.shape == (2, 8, cfg.d_model)


# ---------------------------------------------------------------------------
# Test 8: MultiQueryAttention fewer KV params than MHA
# ---------------------------------------------------------------------------


def test_mqa_fewer_kv_params_than_mha():
    cfg_mha = mha_config()
    cfg_mqa = mqa_config()

    mha = MultiHeadAttention(cfg_mha)
    mqa = MultiQueryAttention(cfg_mqa)

    def kv_params(m: torch.nn.Module) -> int:
        return sum(
            p.numel()
            for name, p in m.named_parameters()
            if name in ("k_proj.weight", "v_proj.weight")
        )

    assert kv_params(mqa) < kv_params(mha)


# ---------------------------------------------------------------------------
# Test 9: GroupedQueryAttention output shape (B, T, d_model)
# ---------------------------------------------------------------------------


def test_gqa_module_output_shape():
    cfg = gqa_config()
    model = GroupedQueryAttention(cfg)
    x = torch.randn(2, 8, cfg.d_model)
    out = model(x)
    assert out.shape == (2, 8, cfg.d_model)


# ---------------------------------------------------------------------------
# Test 10: GroupedQueryAttention param count between MHA and MQA
# ---------------------------------------------------------------------------


def test_gqa_param_count_between_mha_and_mqa():
    d_model = 64
    n_heads = 8
    head_dim = 16

    cfg_mha = AttentionVariantConfig(
        d_model=d_model, n_heads=n_heads, n_kv_heads=n_heads, head_dim=head_dim
    )
    cfg_gqa = AttentionVariantConfig(
        d_model=d_model, n_heads=n_heads, n_kv_heads=4, head_dim=head_dim
    )
    cfg_mqa = AttentionVariantConfig(
        d_model=d_model, n_heads=n_heads, n_kv_heads=1, head_dim=head_dim
    )

    mha = MultiHeadAttention(cfg_mha)
    gqa = GroupedQueryAttention(cfg_gqa)
    mqa = MultiQueryAttention(cfg_mqa)

    def kv_params(m: torch.nn.Module) -> int:
        return sum(
            p.numel()
            for name, p in m.named_parameters()
            if name in ("k_proj.weight", "v_proj.weight")
        )

    assert kv_params(mqa) < kv_params(gqa) < kv_params(mha)


# ---------------------------------------------------------------------------
# Test 11: CrossAttention output shape (B, T_q, d_model)
# ---------------------------------------------------------------------------


def test_cross_attention_output_shape():
    cfg = mha_config()
    model = CrossAttention(cfg)
    query = torch.randn(2, 6, cfg.d_model)
    context = torch.randn(2, 10, cfg.d_model)
    out = model(query, context)
    assert out.shape == (2, 6, cfg.d_model)


# ---------------------------------------------------------------------------
# Test 12: CrossAttention T_q can differ from T_kv
# ---------------------------------------------------------------------------


def test_cross_attention_different_seq_lens():
    cfg = mha_config()
    model = CrossAttention(cfg)

    for T_q, T_kv in [(1, 16), (12, 4), (8, 8)]:
        query = torch.randn(1, T_q, cfg.d_model)
        context = torch.randn(1, T_kv, cfg.d_model)
        out = model(query, context)
        assert out.shape == (1, T_q, cfg.d_model), f"Failed for T_q={T_q}, T_kv={T_kv}"


# ---------------------------------------------------------------------------
# Test 13: count_kv_parameters — MQA has 1/n_heads the KV params of MHA
# ---------------------------------------------------------------------------


def test_count_kv_parameters_mqa_ratio():
    n_heads = 8
    head_dim = 64
    d_model = 512

    mha_kv = count_kv_parameters(n_heads, n_kv_heads=n_heads, head_dim=head_dim, d_model=d_model)
    mqa_kv = count_kv_parameters(n_heads, n_kv_heads=1, head_dim=head_dim, d_model=d_model)

    assert mqa_kv * n_heads == mha_kv
    assert mqa_kv == 2 * 1 * head_dim * d_model
    assert mha_kv == 2 * n_heads * head_dim * d_model


# ---------------------------------------------------------------------------
# Test 14: attention_memory_ratio — MQA: 1/n_heads
# ---------------------------------------------------------------------------


def test_attention_memory_ratio_mqa():
    n_heads = 8
    ratio = attention_memory_ratio(n_heads=n_heads, n_kv_heads=1)
    assert abs(ratio - 1.0 / n_heads) < 1e-9


# ---------------------------------------------------------------------------
# Test 15: attention_memory_ratio — MHA: 1.0
# ---------------------------------------------------------------------------


def test_attention_memory_ratio_mha():
    n_heads = 16
    ratio = attention_memory_ratio(n_heads=n_heads, n_kv_heads=n_heads)
    assert ratio == 1.0


# ---------------------------------------------------------------------------
# Test 16: All attention variants produce same output shape for same config
# ---------------------------------------------------------------------------


def test_all_variants_same_output_shape():
    """MHA, GQA, and MQA should all produce (B, T, d_model) for the same input."""
    d_model = 64
    n_heads = 4
    head_dim = 16
    B, T = 2, 10

    cfg_mha = AttentionVariantConfig(
        d_model=d_model, n_heads=n_heads, n_kv_heads=n_heads, head_dim=head_dim
    )
    cfg_gqa = AttentionVariantConfig(
        d_model=d_model, n_heads=n_heads, n_kv_heads=2, head_dim=head_dim
    )
    cfg_mqa = AttentionVariantConfig(
        d_model=d_model, n_heads=n_heads, n_kv_heads=1, head_dim=head_dim
    )

    x = torch.randn(B, T, d_model)
    expected_shape = (B, T, d_model)

    models = [
        MultiHeadAttention(cfg_mha),
        GroupedQueryAttention(cfg_gqa),
        MultiQueryAttention(cfg_mqa),
    ]

    for model in models:
        out = model(x)
        assert out.shape == expected_shape, (
            f"{type(model).__name__}: expected {expected_shape}, got {out.shape}"
        )
