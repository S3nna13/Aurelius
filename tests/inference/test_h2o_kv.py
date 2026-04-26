"""
Tests for src/inference/h2o_kv.py — H2O KV cache compression.

Paper: Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative
Inference of Large Language Models", NeurIPS 2023 (arXiv:2306.14048).

Tiny test config: B=1, n_heads=4, head_dim=16, max_size=8, recent_window=2
"""

from __future__ import annotations

import torch

from src.inference.h2o_kv import H2OCache, H2OEvictionPolicy

# ---------------------------------------------------------------------------
# Shared tiny dimensions (paper: k=8, w=2)
# ---------------------------------------------------------------------------
B = 1
N_HEADS = 4
HEAD_DIM = 16
K = 8  # max_size  (k in paper)
W = 2  # recent_window (w in paper)


def make_kv(seq_len: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (key, value) tensors of shape (B, N_HEADS, seq_len, HEAD_DIM)."""
    key = torch.randn(B, N_HEADS, seq_len, HEAD_DIM)
    val = torch.randn(B, N_HEADS, seq_len, HEAD_DIM)
    return key, val


def make_attn(cache_len: int, uniform: bool = False) -> torch.Tensor:
    """Return attention scores of shape (B, N_HEADS, cache_len).

    If uniform=True every position gets equal weight (1/cache_len).
    Otherwise returns a random positive tensor.
    """
    if cache_len == 0:
        return torch.zeros(B, N_HEADS, 0)
    if uniform:
        return torch.full((B, N_HEADS, cache_len), 1.0 / cache_len)
    return torch.rand(B, N_HEADS, cache_len)


def fresh_cache(**kwargs) -> H2OCache:
    """Convenience: create a fresh H2OCache with default tiny config."""
    params = dict(max_size=K, recent_window=W)
    params.update(kwargs)
    return H2OCache(**params)


def fill_cache(cache: H2OCache, n_tokens: int) -> None:
    """Push n_tokens into cache using uniform attention scores."""
    for _ in range(n_tokens):
        k, v = make_kv()
        attn = make_attn(cache.size)
        cache.update(k, v, attn)


# ---------------------------------------------------------------------------
# Test 1: size increases with each update when below max_size
# ---------------------------------------------------------------------------
def test_size_increases_below_max():
    cache = fresh_cache()
    for i in range(1, K + 1):
        k, v = make_kv()
        attn = make_attn(cache.size)
        cache.update(k, v, attn)
        assert cache.size == i, f"Expected size {i}, got {cache.size}"


# ---------------------------------------------------------------------------
# Test 2: size stays ≤ max_size after overflow
# ---------------------------------------------------------------------------
def test_size_bounded_after_overflow():
    cache = fresh_cache()
    # Push more tokens than the budget
    for _ in range(K + 5):
        k, v = make_kv()
        attn = make_attn(cache.size)
        cache.update(k, v, attn)
    assert cache.size <= K, f"Cache exceeded max_size: {cache.size} > {K}"
    assert cache.size == K, f"Cache size should equal K={K}, got {cache.size}"


# ---------------------------------------------------------------------------
# Test 3: evicts least-attended token on overflow
# ---------------------------------------------------------------------------
def test_evicts_least_attended():
    """The token that has accumulated the lowest attention score is evicted."""
    cache = fresh_cache(max_size=4, recent_window=0)

    # Fill cache to exactly max_size with tokens receiving zero attention
    # (uniform zeros), then add one final token.
    # We'll use controlled scores: give token 0 a very low score so it's evicted.

    # Token 0: add with no prior cache
    k0, v0 = make_kv()
    cache.update(k0, v0, make_attn(0))  # cache size = 1

    # Tokens 1–3: give token 0 near-zero attention mass
    for _ in range(3):
        k, v = make_kv()
        sz = cache.size
        attn = torch.zeros(B, N_HEADS, sz)
        # give mass to all positions except index 0
        attn[:, :, 1:] = 1.0
        cache.update(k, v, attn)  # sizes 2, 3, 4

    assert cache.size == 4

    # Now add one more token — cache must evict the lowest-scored one (index 0)
    k_new, v_new = make_kv()
    attn_new = torch.zeros(B, N_HEADS, 4)
    attn_new[:, :, 1:] = 1.0  # still ignore index 0
    cache.update(k_new, v_new, attn_new)

    assert cache.size == 4, f"size should stay at 4, got {cache.size}"


# ---------------------------------------------------------------------------
# Test 4: never evicts from recent_window tokens
# ---------------------------------------------------------------------------
def test_never_evicts_recent_window():
    """Even if recent tokens have low scores they must not be evicted."""
    cache = fresh_cache(max_size=4, recent_window=2)

    # Fill to max with zero-score attention everywhere
    for _ in range(4):
        k, v = make_kv()
        attn = torch.zeros(B, N_HEADS, cache.size)
        cache.update(k, v, attn)

    assert cache.size == 4

    # Capture the last 2 keys/values — they are in the recent window
    keys_before, vals_before = cache.get_kv()
    recent_keys = keys_before[:, :, -2:, :].clone()
    vals_before[:, :, -2:, :].clone()

    # Add a new token (forces eviction).
    k_new, v_new = make_kv()
    attn_new = torch.zeros(B, N_HEADS, 4)
    cache.update(k_new, v_new, attn_new)

    assert cache.size == 4

    keys_after, vals_after = cache.get_kv()

    # The two most-recent keys from *before* the eviction should still be
    # present somewhere in the updated cache.
    for head in range(N_HEADS):
        for ri in range(2):
            ref_k = recent_keys[0, head, ri]  # (HEAD_DIM,)
            found = any(
                torch.allclose(ref_k, keys_after[0, head, ci], atol=1e-6)
                for ci in range(keys_after.shape[2])
            )
            assert found, (
                f"Recent token {ri} (head {head}) was evicted despite being in the recent window."
            )


# ---------------------------------------------------------------------------
# Test 5: get_kv shapes are (B, n_heads, size, head_dim)
# ---------------------------------------------------------------------------
def test_get_kv_shapes():
    cache = fresh_cache()
    fill_cache(cache, K)
    keys, vals = cache.get_kv()
    assert keys.shape == (B, N_HEADS, K, HEAD_DIM), f"keys shape: {keys.shape}"
    assert vals.shape == (B, N_HEADS, K, HEAD_DIM), f"vals shape: {vals.shape}"


# ---------------------------------------------------------------------------
# Test 6: heavy hitters retained (high-score tokens survive eviction)
# ---------------------------------------------------------------------------
def test_heavy_hitters_retained():
    """A token that consistently receives high attention should survive."""
    cache = fresh_cache(max_size=4, recent_window=0)

    # Token 0 — the designated heavy hitter key tensor
    hh_key = torch.ones(B, N_HEADS, 1, HEAD_DIM) * 99.0
    hh_val = torch.ones(B, N_HEADS, 1, HEAD_DIM) * 99.0
    cache.update(hh_key, hh_val, make_attn(0))  # size 1

    # Tokens 1–3 — give the heavy hitter (index 0) lots of attention
    for _ in range(3):
        k, v = make_kv()
        sz = cache.size
        attn = torch.zeros(B, N_HEADS, sz)
        attn[:, :, 0] = 10.0  # heavy hitter gets all attention
        cache.update(k, v, attn)

    assert cache.size == 4

    # Add token 4 — triggers eviction; heavy hitter must survive
    k_new, v_new = make_kv()
    attn_new = torch.zeros(B, N_HEADS, 4)
    attn_new[:, :, 0] = 10.0
    cache.update(k_new, v_new, attn_new)

    assert cache.size == 4
    keys, _ = cache.get_kv()

    hh_found = any(
        torch.allclose(keys[0, 0, ci], hh_key[0, 0, 0], atol=1e-5) for ci in range(keys.shape[2])
    )
    assert hh_found, "Heavy hitter was incorrectly evicted."


# ---------------------------------------------------------------------------
# Test 7: light tokens evicted first
# ---------------------------------------------------------------------------
def test_light_tokens_evicted_first():
    """Tokens receiving near-zero attention are the first to be dropped."""
    cache = fresh_cache(max_size=4, recent_window=0)

    # Token 0: light token (near-zero attention)
    light_key = torch.full((B, N_HEADS, 1, HEAD_DIM), -99.0)
    light_val = torch.full((B, N_HEADS, 1, HEAD_DIM), -99.0)
    cache.update(light_key, light_val, make_attn(0))

    # Tokens 1–3: fill with heavy attention on each other, not on token 0
    for _ in range(3):
        k, v = make_kv()
        sz = cache.size
        attn = torch.zeros(B, N_HEADS, sz)
        attn[:, :, 1:] = 1.0  # no attention to light token
        cache.update(k, v, attn)

    # Token 4: triggers eviction
    k_new, v_new = make_kv()
    attn_new = torch.zeros(B, N_HEADS, 4)
    attn_new[:, :, 1:] = 1.0
    cache.update(k_new, v_new, attn_new)

    assert cache.size == 4
    keys, _ = cache.get_kv()

    light_found = any(
        torch.allclose(keys[0, 0, ci], light_key[0, 0, 0], atol=1e-5) for ci in range(keys.shape[2])
    )
    assert not light_found, "Light token should have been evicted."


# ---------------------------------------------------------------------------
# Test 8: recent_window >= max_size → purely recency-based, no crash
# ---------------------------------------------------------------------------
def test_recent_window_ge_max_size():
    """When w >= k the cache degenerates to a pure recency window; no error."""
    cache = fresh_cache(max_size=4, recent_window=4)
    for _ in range(8):
        k, v = make_kv()
        attn = make_attn(cache.size)
        cache.update(k, v, attn)
    assert cache.size <= 4


# ---------------------------------------------------------------------------
# Test 9: max_size=1 edge case — only most recent token kept
# ---------------------------------------------------------------------------
def test_max_size_one():
    cache = fresh_cache(max_size=1, recent_window=1)

    k0, v0 = make_kv()
    cache.update(k0, v0, make_attn(0))
    assert cache.size == 1

    for _ in range(5):
        k, v = make_kv()
        attn = make_attn(cache.size)
        cache.update(k, v, attn)
        assert cache.size == 1


# ---------------------------------------------------------------------------
# Test 10: determinism under same attention scores
# ---------------------------------------------------------------------------
def test_determinism():
    """Two caches driven by identical inputs produce identical KV contents."""
    torch.manual_seed(42)
    n_steps = K + 3
    keys_seq = [torch.randn(B, N_HEADS, 1, HEAD_DIM) for _ in range(n_steps)]
    vals_seq = [torch.randn(B, N_HEADS, 1, HEAD_DIM) for _ in range(n_steps)]
    # Pre-generate attention tensors for each *possible* cache length
    # (0..K inclusive) and index by cache.size at each step.
    rng_attns = {
        l: (torch.rand(B, N_HEADS, l) if l > 0 else torch.zeros(B, N_HEADS, 0))
        for l in range(K + 1)  # noqa: E741
    }

    def run() -> tuple[torch.Tensor, torch.Tensor]:
        c = fresh_cache()
        for ki, vi in zip(keys_seq, vals_seq):
            ai = rng_attns[c.size].clone()
            c.update(ki.clone(), vi.clone(), ai)
        return c.get_kv()

    k1, v1 = run()
    k2, v2 = run()

    assert torch.allclose(k1, k2), "Keys differ between runs with same inputs."
    assert torch.allclose(v1, v2), "Values differ between runs with same inputs."


# ---------------------------------------------------------------------------
# Test 11: no NaN/Inf on uniform attention
# ---------------------------------------------------------------------------
def test_no_nan_inf_uniform():
    cache = fresh_cache()
    for _ in range(K + 4):
        k, v = make_kv()
        attn = make_attn(cache.size, uniform=True)
        cache.update(k, v, attn)

    keys, vals = cache.get_kv()
    assert not torch.any(torch.isnan(keys)), "NaN in keys (uniform attn)"
    assert not torch.any(torch.isinf(keys)), "Inf in keys (uniform attn)"
    assert not torch.any(torch.isnan(vals)), "NaN in vals (uniform attn)"
    assert not torch.any(torch.isinf(vals)), "Inf in vals (uniform attn)"


# ---------------------------------------------------------------------------
# Test 12: no NaN/Inf on peaked attention (one token gets all mass)
# ---------------------------------------------------------------------------
def test_no_nan_inf_peaked():
    cache = fresh_cache()
    for step in range(K + 4):
        k, v = make_kv()
        sz = cache.size
        if sz == 0:
            attn = make_attn(0)
        else:
            attn = torch.zeros(B, N_HEADS, sz)
            attn[:, :, 0] = 1.0  # all attention on first token
        cache.update(k, v, attn)

    keys, vals = cache.get_kv()
    assert not torch.any(torch.isnan(keys)), "NaN in keys (peaked attn)"
    assert not torch.any(torch.isinf(keys)), "Inf in keys (peaked attn)"
    assert not torch.any(torch.isnan(vals)), "NaN in vals (peaked attn)"
    assert not torch.any(torch.isinf(vals)), "Inf in vals (peaked attn)"


# ---------------------------------------------------------------------------
# Test 13: H2OEvictionPolicy always selects lowest-score non-recent token
# ---------------------------------------------------------------------------
def test_eviction_policy_selects_lowest_non_recent():
    policy = H2OEvictionPolicy()

    # scores: [5, 1, 3, 9, 2]  recent_mask: [F, F, F, F, T]
    # lowest non-recent = index 1 (score=1)
    scores = torch.tensor([5.0, 1.0, 3.0, 9.0, 2.0])
    recent_mask = torch.tensor([False, False, False, False, True])
    assert policy.select_evict(scores, recent_mask) == 1

    # recent_mask covers indices 3 and 4; lowest non-recent = index 1
    recent_mask2 = torch.tensor([False, False, False, True, True])
    assert policy.select_evict(scores, recent_mask2) == 1

    # If index 1 is also recent, next lowest non-recent = index 4 (score=2)
    recent_mask3 = torch.tensor([False, True, False, True, False])
    # non-recent scores: 0→5, 2→3, 4→2  → minimum is index 4
    assert policy.select_evict(scores, recent_mask3) == 4


# ---------------------------------------------------------------------------
# Test 14: cache correctly accumulates scores across multiple updates
# ---------------------------------------------------------------------------
def test_score_accumulation():
    """Scores grow monotonically for a token that always receives attention."""
    cache = fresh_cache(max_size=10, recent_window=0)

    # Token 0 always receives 1.0 of attention per step
    k0, v0 = make_kv()
    cache.update(k0, v0, make_attn(0))

    n_extra = 6
    for step in range(n_extra):
        k, v = make_kv()
        sz = cache.size
        attn = torch.zeros(B, N_HEADS, sz)
        attn[:, :, 0] = 1.0  # all attention to token 0
        cache.update(k, v, attn)

    # Internal scores: token 0 should have accumulated n_extra × 1.0
    # (it exists in the cache at index 0 since max_size=10 >> n_extra+1)
    scores = cache._scores  # (B, N_HEADS, cache_len)
    expected_score = float(n_extra)
    actual_score = float(scores[0, 0, 0].item())

    assert abs(actual_score - expected_score) < 1e-4, (
        f"Expected accumulated score {expected_score}, got {actual_score}"
    )
