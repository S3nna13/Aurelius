"""Tests for sliding-window sink attention."""

from __future__ import annotations

import pytest
import torch

from src.inference.sliding_window_sink import (
    SlidingWindowSinkAttention,
    SlidingWindowSinkCache,
    SlidingWindowSinkConfig,
    sliding_window_sink_indices,
    sliding_window_sink_mask,
)
from src.model.config import AureliusConfig

TINY_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=4,
    n_kv_heads=2,
    head_dim=16,
    d_ff=128,
    vocab_size=256,
    max_seq_len=64,
)


def make_module(S: int = 4, W: int = 8) -> SlidingWindowSinkAttention:
    return SlidingWindowSinkAttention(
        d_model=TINY_CFG.d_model,
        n_heads=TINY_CFG.n_heads,
        n_kv_heads=TINY_CFG.n_kv_heads,
        head_dim=TINY_CFG.head_dim,
        S=S,
        W=W,
    )


def reference_indices(T: int, S: int, W: int) -> torch.Tensor:
    kept = []
    for j in range(T):
        if j < S or j >= max(S, T - W):
            kept.append(j)
    return torch.tensor(kept, dtype=torch.long)


def reference_mask(
    q_positions: torch.Tensor, k_positions: torch.Tensor, S: int, W: int
) -> torch.Tensor:
    mask = torch.zeros(q_positions.numel(), k_positions.numel(), dtype=torch.bool)
    for i, q_i in enumerate(q_positions.tolist()):
        for j, k_j in enumerate(k_positions.tolist()):
            if k_j <= q_i and (k_j < S or k_j >= q_i - W + 1):
                mask[i, j] = True
    return mask


def test_config_defaults_and_cache_size():
    cfg = SlidingWindowSinkConfig()
    assert cfg.S == 4
    assert cfg.W == 256
    assert cfg.cache_size == 260


def test_rejects_invalid_config():
    with pytest.raises(ValueError, match="S"):
        SlidingWindowSinkConfig(S=-1)
    with pytest.raises(ValueError, match="W"):
        SlidingWindowSinkConfig(W=0)


def test_indices_match_reference_formulation():
    idx = sliding_window_sink_indices(T=19, S=3, W=5)
    ref = reference_indices(T=19, S=3, W=5)
    assert torch.equal(idx, ref)


def test_mask_matches_reference_formulation():
    q_positions = torch.arange(10, dtype=torch.long)
    k_positions = torch.arange(10, dtype=torch.long)
    mask = sliding_window_sink_mask(q_positions, k_positions, S=2, W=4)
    ref = reference_mask(q_positions, k_positions, S=2, W=4)
    assert torch.equal(mask, ref)


def test_cache_keeps_sink_and_recent_positions():
    cache = SlidingWindowSinkCache(S=2, W=3)
    for t in range(8):
        K_new = torch.full((1, TINY_CFG.n_kv_heads, 1, TINY_CFG.head_dim), float(t))
        V_new = K_new.clone()
        cache.update(K_new, V_new)

    K_cache, V_cache, positions = cache.get()
    assert K_cache.shape == (1, TINY_CFG.n_kv_heads, 5, TINY_CFG.head_dim)
    assert V_cache.shape == K_cache.shape
    assert torch.equal(positions, torch.tensor([0, 1, 5, 6, 7]))


def test_attention_shape_and_dtype_on_tiny_config():
    torch.manual_seed(0)
    module = make_module(S=4, W=8)
    x = torch.randn(2, 6, TINY_CFG.d_model)
    out = module(x)
    assert out.shape == (2, 6, TINY_CFG.d_model)
    assert out.dtype == x.dtype


def test_backward_produces_finite_gradients_for_all_params():
    torch.manual_seed(1)
    module = make_module(S=4, W=8)
    x = torch.randn(2, 7, TINY_CFG.d_model, requires_grad=True)
    out = module(x)
    loss = out.square().mean()
    loss.backward()

    for name, param in module.named_parameters():
        assert param.grad is not None, f"missing grad for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite grad for {name}"


def test_deterministic_under_manual_seed():
    torch.manual_seed(123)
    module_a = make_module(S=4, W=8)
    x = torch.randn(2, 5, TINY_CFG.d_model)
    out_a = module_a(x)

    torch.manual_seed(123)
    module_b = make_module(S=4, W=8)
    x_b = torch.randn(2, 5, TINY_CFG.d_model)
    out_b = module_b(x_b)

    assert torch.equal(x, x_b)
    assert torch.allclose(out_a, out_b)


def test_batch_one_and_seq_len_one():
    torch.manual_seed(2)
    module = make_module(S=4, W=8)
    x = torch.randn(1, 1, TINY_CFG.d_model)
    out = module(x)
    assert out.shape == (1, 1, TINY_CFG.d_model)
    assert torch.isfinite(out).all()


def test_padded_queries_are_zeroed_and_finite():
    torch.manual_seed(3)
    module = make_module(S=2, W=4)
    x = torch.randn(1, 5, TINY_CFG.d_model)
    key_padding_mask = torch.tensor([[True, True, True, False, False]])
    out = module(x, key_padding_mask=key_padding_mask)
    assert torch.isfinite(out).all()
    assert torch.allclose(out[:, 3:, :], torch.zeros_like(out[:, 3:, :]), atol=1e-7)


def test_numerical_stability_on_extreme_inputs():
    torch.manual_seed(4)
    module = make_module(S=2, W=4)
    x = torch.full((2, 6, TINY_CFG.d_model), 1.0e4)
    out, attn = module(x, return_attn_weights=True)
    assert torch.isfinite(out).all()
    assert torch.isfinite(attn).all()


def test_streaming_matches_full_reference_before_eviction():
    torch.manual_seed(5)
    module = make_module(S=4, W=8)
    x = torch.randn(1, 10, TINY_CFG.d_model)

    full = module(x)
    module.reset_cache()

    streamed = []
    for t in range(x.size(1)):
        streamed.append(module(x[:, t : t + 1, :], use_cache=True))
    streamed_out = torch.cat(streamed, dim=1)

    assert torch.allclose(streamed_out, full, atol=1e-5)


def test_streaming_cache_size_is_bounded_after_long_sequence():
    torch.manual_seed(6)
    module = make_module(S=3, W=5)
    x = torch.randn(1, 17, TINY_CFG.d_model)

    for t in range(x.size(1)):
        module(x[:, t : t + 1, :], use_cache=True)

    K_cache, _, positions = module.cache.get()
    assert K_cache.size(2) == 8
    assert module.cache.size == 8
    assert torch.equal(positions, torch.tensor([0, 1, 2, 12, 13, 14, 15, 16]))
