"""Tests for src/inference/token_merging.py — Token Merging (ToMe)."""

from __future__ import annotations

import torch

from src.inference.token_merging import (
    ToMeConfig,
    ToMeLayer,
    ToMeWrapper,
    bipartite_soft_matching,
    merge_tokens,
    unmerge_tokens,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tiny_model():
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )
    return AureliusTransformer(cfg)


# ---------------------------------------------------------------------------
# 1. ToMeConfig defaults
# ---------------------------------------------------------------------------


def test_tome_config_defaults():
    cfg = ToMeConfig()
    assert cfg.r == 4
    assert cfg.merge_mode == "mean"
    assert cfg.similarity == "cosine"


# ---------------------------------------------------------------------------
# 2. bipartite_soft_matching returns 3 tensors
# ---------------------------------------------------------------------------


def test_bipartite_soft_matching_returns_three_tensors():
    B, T, d = 2, 10, 32
    x = torch.randn(B, T, d)
    result = bipartite_soft_matching(x, r=3)
    assert isinstance(result, tuple)
    assert len(result) == 3
    for item in result:
        assert isinstance(item, torch.Tensor)


# ---------------------------------------------------------------------------
# 3. bipartite_soft_matching src_idx shape is (B, r') where r' <= min(T//2, r)
# ---------------------------------------------------------------------------


def test_bipartite_soft_matching_src_idx_shape():
    B, T, d = 2, 10, 32
    r = 3
    x = torch.randn(B, T, d)
    src_idx, dst_idx, unmerge_map = bipartite_soft_matching(x, r=r)
    r_prime = src_idx.shape[1]
    assert src_idx.shape == (B, r_prime)
    assert r_prime <= min(T // 2, r)


# ---------------------------------------------------------------------------
# 4. bipartite_soft_matching indices are valid (in range)
# ---------------------------------------------------------------------------


def test_bipartite_soft_matching_indices_in_range():
    B, T, d = 2, 12, 16
    r = 4
    x = torch.randn(B, T, d)
    src_idx, dst_idx, unmerge_map = bipartite_soft_matching(x, r=r)
    assert src_idx.min() >= 0
    assert src_idx.max() < T
    assert dst_idx.min() >= 0
    assert dst_idx.max() < T


# ---------------------------------------------------------------------------
# 5. bipartite_soft_matching with r=0 returns empty indices
# ---------------------------------------------------------------------------


def test_bipartite_soft_matching_r_zero():
    B, T, d = 2, 8, 16
    x = torch.randn(B, T, d)
    src_idx, dst_idx, unmerge_map = bipartite_soft_matching(x, r=0)
    assert src_idx.shape == (B, 0)
    assert dst_idx.shape == (B, 0)
    assert unmerge_map.shape == (B, T)


# ---------------------------------------------------------------------------
# 6. merge_tokens output shape is (B, T-r, d)
# ---------------------------------------------------------------------------


def test_merge_tokens_output_shape():
    B, T, d = 2, 10, 32
    r = 3
    x = torch.randn(B, T, d)
    src_idx, dst_idx, _ = bipartite_soft_matching(x, r=r)
    r_actual = src_idx.shape[1]
    merged, size = merge_tokens(x, src_idx, dst_idx, mode="mean")
    assert merged.shape == (B, T - r_actual, d)


# ---------------------------------------------------------------------------
# 7. merge_tokens size tensor sums to T (conservation of tokens)
# ---------------------------------------------------------------------------


def test_merge_tokens_size_sums_to_T():
    B, T, d = 2, 10, 32
    r = 3
    x = torch.randn(B, T, d)
    src_idx, dst_idx, _ = bipartite_soft_matching(x, r=r)
    merged, size = merge_tokens(x, src_idx, dst_idx, mode="mean")
    for b in range(B):
        total = size[b].sum().item()
        assert abs(total - T) < 1e-5, f"Batch {b}: size sum {total} != T={T}"


# ---------------------------------------------------------------------------
# 8. unmerge_tokens restores shape to (B, T_orig, d)
# ---------------------------------------------------------------------------


def test_unmerge_tokens_restores_shape():
    B, T, d = 2, 10, 32
    r = 3
    x = torch.randn(B, T, d)
    src_idx, dst_idx, _ = bipartite_soft_matching(x, r=r)
    merged, size = merge_tokens(x, src_idx, dst_idx)
    reconstructed = unmerge_tokens(merged, src_idx, dst_idx, size, T_orig=T)
    assert reconstructed.shape == (B, T, d)


# ---------------------------------------------------------------------------
# 9. unmerge_tokens with r=0 is identity
# ---------------------------------------------------------------------------


def test_unmerge_tokens_r_zero_is_identity():
    B, T, d = 2, 8, 16
    x = torch.randn(B, T, d)
    src_idx, dst_idx, _ = bipartite_soft_matching(x, r=0)
    merged, size = merge_tokens(x, src_idx, dst_idx)
    reconstructed = unmerge_tokens(merged, src_idx, dst_idx, size, T_orig=T)
    assert reconstructed.shape == (B, T, d)
    assert torch.allclose(reconstructed, x, atol=1e-5), (
        "With r=0, unmerge should return the original sequence unchanged"
    )


# ---------------------------------------------------------------------------
# 10. ToMeLayer output shape matches input shape
# ---------------------------------------------------------------------------


def test_tome_layer_output_shape_matches_input():
    B, T, d = 2, 10, 32
    cfg = ToMeConfig(r=2)
    inner = torch.nn.Linear(d, d, bias=False)
    layer = ToMeLayer(inner, cfg)
    x = torch.randn(B, T, d)
    out = layer(x)
    assert out.shape == (B, T, d), f"Expected ({B}, {T}, {d}), got {out.shape}"


# ---------------------------------------------------------------------------
# 11. ToMeLayer with r > T//2 doesn't crash (caps merging)
# ---------------------------------------------------------------------------


def test_tome_layer_r_larger_than_half_T_does_not_crash():
    B, T, d = 2, 6, 16
    cfg = ToMeConfig(r=100)
    inner = torch.nn.Linear(d, d, bias=False)
    layer = ToMeLayer(inner, cfg)
    x = torch.randn(B, T, d)
    out = layer(x)
    assert out.shape == (B, T, d)


# ---------------------------------------------------------------------------
# 12. ToMeWrapper instantiates from AureliusTransformer
# ---------------------------------------------------------------------------


def test_tome_wrapper_instantiation():
    model = make_tiny_model()
    cfg = ToMeConfig(r=2)
    wrapper = ToMeWrapper(model, cfg)
    assert isinstance(wrapper, torch.nn.Module)
    assert hasattr(wrapper, "embed")
    assert hasattr(wrapper, "norm")
    assert hasattr(wrapper, "lm_head")
    assert hasattr(wrapper, "layers")
    assert len(wrapper.layers) == 2


# ---------------------------------------------------------------------------
# 13. ToMeWrapper.forward returns 3-tuple with logits shape (B, T, vocab)
# ---------------------------------------------------------------------------


def test_tome_wrapper_forward_shape():
    model = make_tiny_model()
    cfg = ToMeConfig(r=2)
    wrapper = ToMeWrapper(model, cfg)
    wrapper.eval()

    B, T = 2, 8
    input_ids = torch.randint(0, 256, (B, T))

    with torch.no_grad():
        result = wrapper(input_ids)

    assert isinstance(result, tuple)
    assert len(result) == 3
    loss, logits, pkv = result
    assert logits.shape == (B, T, 256), f"Expected ({B}, {T}, 256), got {logits.shape}"
    assert pkv is None


# ---------------------------------------------------------------------------
# 14. ToMeWrapper.get_compression_ratio returns float in (0, 1]
# ---------------------------------------------------------------------------


def test_tome_wrapper_get_compression_ratio():
    model = make_tiny_model()
    cfg = ToMeConfig(r=2)
    wrapper = ToMeWrapper(model, cfg)
    ratio = wrapper.get_compression_ratio()
    assert isinstance(ratio, float)
    assert 0.0 < ratio <= 1.0, f"Compression ratio {ratio} not in (0, 1]"


# ---------------------------------------------------------------------------
# 15. ToMeWrapper with r=0 gives same output as base model (no merging)
# ---------------------------------------------------------------------------


def test_tome_wrapper_r_zero_matches_base_model():
    model = make_tiny_model()
    model.eval()

    cfg = ToMeConfig(r=0)
    wrapper = ToMeWrapper(model, cfg)
    wrapper.eval()

    B, T = 1, 6
    input_ids = torch.randint(0, 256, (B, T))

    with torch.no_grad():
        _loss_base, logits_base, _pkv_base = model(input_ids)
        _loss_wrap, logits_wrap, _pkv_wrap = wrapper(input_ids)

    assert torch.allclose(logits_base, logits_wrap, atol=1e-5), (
        f"With r=0, wrapper output should match base model. "
        f"Max diff: {(logits_base - logits_wrap).abs().max().item()}"
    )
