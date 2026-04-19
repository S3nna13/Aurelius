"""Unit tests for AttentionSinkCache (StreamingLLM windowing)."""

from __future__ import annotations

import time

import pytest
import torch

from src.longcontext.attention_sinks import AttentionSinkCache


B = 2
H_KV = 4
D = 16


def _rand(t_new: int, *, dtype=torch.float32, seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    if seed is not None:
        gen = torch.Generator().manual_seed(seed)
        k = torch.randn(B, H_KV, t_new, D, generator=gen, dtype=dtype)
        v = torch.randn(B, H_KV, t_new, D, generator=gen, dtype=dtype)
    else:
        k = torch.randn(B, H_KV, t_new, D, dtype=dtype)
        v = torch.randn(B, H_KV, t_new, D, dtype=dtype)
    return k, v


# ---------------------------------------------------------------------------
# Basic growth and eviction semantics
# ---------------------------------------------------------------------------


def test_growth_below_budget_no_eviction():
    cache = AttentionSinkCache(n_sinks=4, window_size=16, head_dim=D, n_kv_heads=H_KV)
    pos = 0
    for step in range(1, 11):  # 10 tokens total, budget 20
        k, v = _rand(1, seed=step)
        ck, cv, cp = cache.append(k, v, pos)
        pos += 1
        assert ck.shape[2] == step
        assert cv.shape[2] == step
        assert cp.shape[0] == step
        assert cache.num_cached_tokens() == step


def test_exact_budget_full_no_eviction():
    n_sinks, window = 4, 16
    cache = AttentionSinkCache(n_sinks=n_sinks, window_size=window, head_dim=D, n_kv_heads=H_KV)
    k, v = _rand(n_sinks + window, seed=0)
    ck, cv, cp = cache.append(k, v, 0)
    assert ck.shape[2] == n_sinks + window
    assert cache.num_cached_tokens() == n_sinks + window
    # Contents match exactly (no eviction yet).
    torch.testing.assert_close(ck, k)
    torch.testing.assert_close(cv, v)


def test_overflow_final_size_and_sinks_intact():
    n_sinks, window = 4, 16
    total = 1024
    cache = AttentionSinkCache(n_sinks=n_sinks, window_size=window, head_dim=D, n_kv_heads=H_KV)
    k_all, v_all = _rand(total, seed=42)
    ck, cv, cp = cache.append(k_all, v_all, 0)

    assert ck.shape[2] == n_sinks + window == 20
    assert cache.num_cached_tokens() == 20

    # Sinks: first n_sinks slots of output must equal the first n_sinks input tokens.
    torch.testing.assert_close(ck[:, :, :n_sinks, :], k_all[:, :, :n_sinks, :])
    torch.testing.assert_close(cv[:, :, :n_sinks, :], v_all[:, :, :n_sinks, :])
    # Window: final window tokens of output must equal last window_size input tokens.
    torch.testing.assert_close(ck[:, :, n_sinks:, :], k_all[:, :, total - window:, :])
    torch.testing.assert_close(cv[:, :, n_sinks:, :], v_all[:, :, total - window:, :])

    # Positions: sinks get 0..n_sinks-1; window gets n_sinks..n_sinks+window-1 (shifted).
    expected_pos = torch.arange(n_sinks + window, dtype=torch.long)
    torch.testing.assert_close(cp, expected_pos)


def test_sinks_are_exactly_first_n_tokens_across_appends():
    n_sinks, window = 3, 8
    cache = AttentionSinkCache(n_sinks=n_sinks, window_size=window, head_dim=D, n_kv_heads=H_KV)
    # Feed 64 tokens in small chunks.
    chunks = []
    pos = 0
    for i in range(16):
        k, v = _rand(4, seed=100 + i)
        chunks.append((k, v))
        cache.append(k, v, pos)
        pos += 4
    full_k = torch.cat([c[0] for c in chunks], dim=2)
    full_v = torch.cat([c[1] for c in chunks], dim=2)

    ck, cv, _ = cache.append(*_rand(1, seed=999), pos)
    torch.testing.assert_close(ck[:, :, :n_sinks, :], full_k[:, :, :n_sinks, :])
    torch.testing.assert_close(cv[:, :, :n_sinks, :], full_v[:, :, :n_sinks, :])


def test_window_contains_most_recent_tokens():
    n_sinks, window = 2, 5
    cache = AttentionSinkCache(n_sinks=n_sinks, window_size=window, head_dim=D, n_kv_heads=H_KV)
    total = 30
    k_all, v_all = _rand(total, seed=7)
    # Feed one at a time.
    for t in range(total):
        cache.append(k_all[:, :, t:t + 1, :], v_all[:, :, t:t + 1, :], t)

    ck, cv, _ = cache._materialize_view()  # peek without appending
    torch.testing.assert_close(
        ck[:, :, n_sinks:, :], k_all[:, :, total - window:, :]
    )
    torch.testing.assert_close(
        cv[:, :, n_sinks:, :], v_all[:, :, total - window:, :]
    )


def test_num_cached_tokens_progression():
    cache = AttentionSinkCache(n_sinks=2, window_size=3, head_dim=D, n_kv_heads=H_KV)
    assert cache.num_cached_tokens() == 0
    expected = [1, 2, 3, 4, 5, 5, 5, 5]  # budget = 5
    for i, exp in enumerate(expected):
        k, v = _rand(1, seed=i)
        cache.append(k, v, i)
        assert cache.num_cached_tokens() == exp


def test_reset_clears():
    cache = AttentionSinkCache(n_sinks=2, window_size=4, head_dim=D, n_kv_heads=H_KV)
    k, v = _rand(10, seed=3)
    cache.append(k, v, 0)
    assert cache.num_cached_tokens() > 0
    cache.reset()
    assert cache.num_cached_tokens() == 0
    # Must be usable again, potentially with a different B.
    k2, v2 = torch.randn(1, H_KV, 3, D), torch.randn(1, H_KV, 3, D)
    ck, cv, cp = cache.append(k2, v2, 0)
    assert ck.shape == (1, H_KV, 3, D)
    assert cv.shape == (1, H_KV, 3, D)
    assert cp.shape == (3,)


def test_shapes_preserved_across_appends():
    cache = AttentionSinkCache(n_sinks=4, window_size=8, head_dim=D, n_kv_heads=H_KV)
    for step in range(20):
        k, v = _rand(1, seed=step)
        ck, cv, cp = cache.append(k, v, step)
        assert ck.shape[0] == B
        assert ck.shape[1] == H_KV
        assert ck.shape[3] == D
        assert cv.shape == ck.shape
        assert cp.shape[0] == ck.shape[2]
        assert cp.dim() == 1


def test_dtype_preserved_float32():
    cache = AttentionSinkCache(n_sinks=2, window_size=4, head_dim=D, n_kv_heads=H_KV)
    k, v = _rand(10, dtype=torch.float32, seed=11)
    ck, cv, _ = cache.append(k, v, 0)
    assert ck.dtype == torch.float32
    assert cv.dtype == torch.float32


def test_determinism_with_fixed_seed():
    def run():
        cache = AttentionSinkCache(n_sinks=3, window_size=7, head_dim=D, n_kv_heads=H_KV)
        gen = torch.Generator().manual_seed(1234)
        result = None
        for _ in range(25):
            k = torch.randn(B, H_KV, 1, D, generator=gen)
            v = torch.randn(B, H_KV, 1, D, generator=gen)
            result = cache.append(k, v, 0)
        return result

    a_k, a_v, a_p = run()
    b_k, b_v, b_p = run()
    torch.testing.assert_close(a_k, b_k)
    torch.testing.assert_close(a_v, b_v)
    torch.testing.assert_close(a_p, b_p)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_n_sinks_zero_pure_rolling_window():
    window = 5
    cache = AttentionSinkCache(n_sinks=0, window_size=window, head_dim=D, n_kv_heads=H_KV)
    total = 20
    k_all, v_all = _rand(total, seed=17)
    ck, cv, cp = cache.append(k_all, v_all, 0)
    assert ck.shape[2] == window
    torch.testing.assert_close(ck, k_all[:, :, total - window:, :])
    torch.testing.assert_close(cv, v_all[:, :, total - window:, :])
    # Shifted positions start at n_sinks=0.
    torch.testing.assert_close(cp, torch.arange(window, dtype=torch.long))


def test_window_size_one_edge_case():
    cache = AttentionSinkCache(n_sinks=2, window_size=1, head_dim=D, n_kv_heads=H_KV)
    total = 10
    k_all, v_all = _rand(total, seed=5)
    for t in range(total):
        cache.append(k_all[:, :, t:t + 1, :], v_all[:, :, t:t + 1, :], t)
    ck, cv, cp = cache._materialize_view()
    assert ck.shape[2] == 3
    # Only the most-recent single token sits in the window.
    torch.testing.assert_close(ck[:, :, 2:3, :], k_all[:, :, 9:10, :])
    torch.testing.assert_close(cv[:, :, 2:3, :], v_all[:, :, 9:10, :])
    torch.testing.assert_close(cp, torch.tensor([0, 1, 2], dtype=torch.long))


def test_very_long_sequence_fast_and_correct():
    n_sinks, window = 4, 32
    total = 4096
    cache = AttentionSinkCache(n_sinks=n_sinks, window_size=window, head_dim=D, n_kv_heads=H_KV)
    k_all, v_all = _rand(total, seed=2026)
    t0 = time.perf_counter()
    # Feed one-at-a-time to exercise the incremental eviction path.
    for t in range(total):
        cache.append(k_all[:, :, t:t + 1, :], v_all[:, :, t:t + 1, :], t)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"too slow: {elapsed:.3f}s"

    ck, cv, cp = cache._materialize_view()
    assert ck.shape[2] == n_sinks + window
    torch.testing.assert_close(ck[:, :, :n_sinks, :], k_all[:, :, :n_sinks, :])
    torch.testing.assert_close(ck[:, :, n_sinks:, :], k_all[:, :, total - window:, :])
    torch.testing.assert_close(cv[:, :, :n_sinks, :], v_all[:, :, :n_sinks, :])
    torch.testing.assert_close(cv[:, :, n_sinks:, :], v_all[:, :, total - window:, :])
    # Positions are shifted, *not* the true absolute source positions.
    expected_pos = torch.arange(n_sinks + window, dtype=torch.long)
    torch.testing.assert_close(cp, expected_pos)


# ---------------------------------------------------------------------------
# Fail-loud validation
# ---------------------------------------------------------------------------


def test_zero_total_budget_rejected():
    with pytest.raises(ValueError):
        AttentionSinkCache(n_sinks=0, window_size=0, head_dim=D, n_kv_heads=H_KV)


def test_bad_head_dim_rejected():
    with pytest.raises(ValueError):
        AttentionSinkCache(n_sinks=2, window_size=2, head_dim=0, n_kv_heads=H_KV)
    with pytest.raises(ValueError):
        AttentionSinkCache(n_sinks=2, window_size=2, head_dim=-1, n_kv_heads=H_KV)


def test_bad_shape_rejected():
    cache = AttentionSinkCache(n_sinks=2, window_size=2, head_dim=D, n_kv_heads=H_KV)
    with pytest.raises(ValueError):
        cache.append(torch.zeros(2, H_KV, 1), torch.zeros(2, H_KV, 1), 0)  # 3-D
    with pytest.raises(ValueError):
        cache.append(
            torch.zeros(2, H_KV + 1, 1, D), torch.zeros(2, H_KV + 1, 1, D), 0
        )  # wrong heads
    with pytest.raises(ValueError):
        cache.append(
            torch.zeros(2, H_KV, 1, D + 1), torch.zeros(2, H_KV, 1, D + 1), 0
        )  # wrong head_dim
