"""Tests for src/model/rope_cache.py.

Uses tiny configs to keep tests fast:
    D_HEAD=8, MAX_LEN=16, B=2, T=8
"""

import torch

from src.model.rope_cache import (
    CachedRoPEAttention,
    RopeCache,
    RopeCacheConfig,
    apply_rotary_with_cache,
    build_cos_sin_cache,
    rotate_half,
)

# ---------------------------------------------------------------------------
# Tiny constants used throughout
# ---------------------------------------------------------------------------
D_HEAD = 8
MAX_LEN = 16
B = 2
T = 8
D_MODEL = 16
N_HEADS = 2  # d_head = D_MODEL // N_HEADS = 8


# ---------------------------------------------------------------------------
# 1. RopeCacheConfig defaults
# ---------------------------------------------------------------------------
def test_rope_cache_config_defaults():
    """RopeCacheConfig must have the documented default values."""
    cfg = RopeCacheConfig()
    assert cfg.d_head == 64
    assert cfg.max_seq_len == 2048
    assert cfg.base == 10000.0
    assert cfg.dtype == "float32"


# ---------------------------------------------------------------------------
# 2. build_cos_sin_cache shapes
# ---------------------------------------------------------------------------
def test_build_cos_sin_cache_shapes():
    """build_cos_sin_cache must return (max_seq_len, d_head//2) for both tensors."""
    cos, sin = build_cos_sin_cache(D_HEAD, MAX_LEN)
    assert cos.shape == (MAX_LEN, D_HEAD // 2), (
        f"cos shape expected ({MAX_LEN}, {D_HEAD // 2}), got {cos.shape}"
    )
    assert sin.shape == (MAX_LEN, D_HEAD // 2), (
        f"sin shape expected ({MAX_LEN}, {D_HEAD // 2}), got {sin.shape}"
    )


# ---------------------------------------------------------------------------
# 3. cos values at position 0 are all ones (cos(0) = 1)
# ---------------------------------------------------------------------------
def test_cos_at_position_zero_all_ones():
    """cos_cache[0] must be all ones because cos(0 * theta) = cos(0) = 1."""
    cos, _ = build_cos_sin_cache(D_HEAD, MAX_LEN)
    assert torch.allclose(cos[0], torch.ones(D_HEAD // 2), atol=1e-6), (
        f"cos[0] must be all-ones, got {cos[0]}"
    )


# ---------------------------------------------------------------------------
# 4. sin values at position 0 are all zeros (sin(0) = 0)
# ---------------------------------------------------------------------------
def test_sin_at_position_zero_all_zeros():
    """sin_cache[0] must be all zeros because sin(0 * theta) = sin(0) = 0."""
    _, sin = build_cos_sin_cache(D_HEAD, MAX_LEN)
    assert torch.allclose(sin[0], torch.zeros(D_HEAD // 2), atol=1e-6), (
        f"sin[0] must be all-zeros, got {sin[0]}"
    )


# ---------------------------------------------------------------------------
# 5. apply_rotary_with_cache preserves shape
# ---------------------------------------------------------------------------
def test_apply_rotary_with_cache_preserves_shape():
    """apply_rotary_with_cache must return a tensor with the same shape as x."""
    cos, sin = build_cos_sin_cache(D_HEAD, MAX_LEN)
    x = torch.randn(B, T, D_HEAD)
    out = apply_rotary_with_cache(x, cos, sin)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


# ---------------------------------------------------------------------------
# 6. rotate_half shape preserved
# ---------------------------------------------------------------------------
def test_rotate_half_shape_preserved():
    """rotate_half must preserve the input shape."""
    x = torch.randn(B, T, D_HEAD)
    out = rotate_half(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


# ---------------------------------------------------------------------------
# 7. RopeCache.get returns correct shape
# ---------------------------------------------------------------------------
def test_rope_cache_get_shape():
    """RopeCache.get(seq_len) must return (seq_len, d_head//2) for both cos and sin."""
    cfg = RopeCacheConfig(d_head=D_HEAD, max_seq_len=MAX_LEN)
    cache = RopeCache(cfg)
    cos, sin = cache.get(T)
    assert cos.shape == (T, D_HEAD // 2), (
        f"cos shape expected ({T}, {D_HEAD // 2}), got {cos.shape}"
    )
    assert sin.shape == (T, D_HEAD // 2), (
        f"sin shape expected ({T}, {D_HEAD // 2}), got {sin.shape}"
    )


# ---------------------------------------------------------------------------
# 8. RopeCache.apply preserves shape
# ---------------------------------------------------------------------------
def test_rope_cache_apply_preserves_shape():
    """RopeCache.apply must return a tensor with the same shape as x."""
    cfg = RopeCacheConfig(d_head=D_HEAD, max_seq_len=MAX_LEN)
    cache = RopeCache(cfg)
    x = torch.randn(B, T, D_HEAD)
    out = cache.apply(x, T)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


# ---------------------------------------------------------------------------
# 9. RopeCache.apply is numerically consistent with build_cos_sin_cache
# ---------------------------------------------------------------------------
def test_rope_cache_apply_consistent_with_build():
    """RopeCache.apply must produce the same result as applying build_cos_sin_cache directly."""
    cfg = RopeCacheConfig(d_head=D_HEAD, max_seq_len=MAX_LEN)
    cache = RopeCache(cfg)
    x = torch.randn(B, T, D_HEAD)

    # Via RopeCache
    out_cache = cache.apply(x, T)

    # Via build_cos_sin_cache directly
    cos, sin = build_cos_sin_cache(D_HEAD, MAX_LEN)
    out_direct = apply_rotary_with_cache(x, cos[:T], sin[:T])

    assert torch.allclose(out_cache, out_direct, atol=1e-5), (
        "RopeCache.apply must be numerically consistent with build_cos_sin_cache"
    )


# ---------------------------------------------------------------------------
# 10. CachedRoPEAttention output shape
# ---------------------------------------------------------------------------
def test_cached_rope_attention_output_shape():
    """CachedRoPEAttention forward must return (B, T, d_model)."""
    cfg = RopeCacheConfig(d_head=D_HEAD, max_seq_len=MAX_LEN)
    attn = CachedRoPEAttention(D_MODEL, N_HEADS, cfg)
    x = torch.randn(B, T, D_MODEL)
    out = attn(x)
    assert out.shape == (B, T, D_MODEL), f"Expected ({B}, {T}, {D_MODEL}), got {out.shape}"


# ---------------------------------------------------------------------------
# 11. CachedRoPEAttention output is finite (no NaN/Inf)
# ---------------------------------------------------------------------------
def test_cached_rope_attention_output_finite():
    """CachedRoPEAttention output must not contain NaN or Inf values."""
    cfg = RopeCacheConfig(d_head=D_HEAD, max_seq_len=MAX_LEN)
    attn = CachedRoPEAttention(D_MODEL, N_HEADS, cfg)
    x = torch.randn(B, T, D_MODEL)
    out = attn(x)
    assert torch.isfinite(out).all(), "CachedRoPEAttention output contains NaN or Inf"


# ---------------------------------------------------------------------------
# 12. RoPE applied twice is different from once (not idempotent)
# ---------------------------------------------------------------------------
def test_rope_not_idempotent():
    """Applying RoPE twice must differ from applying it once."""
    cos, sin = build_cos_sin_cache(D_HEAD, MAX_LEN)
    x = torch.randn(B, T, D_HEAD)
    once = apply_rotary_with_cache(x, cos[:T], sin[:T])
    twice = apply_rotary_with_cache(once, cos[:T], sin[:T])
    assert not torch.allclose(once, twice, atol=1e-5), (
        "Applying RoPE twice must differ from once (RoPE is not idempotent)"
    )


# ---------------------------------------------------------------------------
# 13. Pythagorean identity: cos^2 + sin^2 = 1 for every (pos, dim) entry
# ---------------------------------------------------------------------------
def test_pythagorean_identity():
    """cos^2 + sin^2 must equal 1 for every (position, dimension) entry."""
    cos, sin = build_cos_sin_cache(D_HEAD, MAX_LEN)
    identity = cos**2 + sin**2
    assert torch.allclose(identity, torch.ones_like(identity), atol=1e-5), (
        "cos^2 + sin^2 must equal 1 (Pythagorean identity)"
    )


# ---------------------------------------------------------------------------
# 14. RoPE is position-dependent: same vector at pos 0 vs pos 5 differs
# ---------------------------------------------------------------------------
def test_rope_position_dependent():
    """The same vector rotated at position 0 vs position 5 must differ."""
    cos, sin = build_cos_sin_cache(D_HEAD, MAX_LEN)
    v = torch.randn(D_HEAD)

    x0 = v.unsqueeze(0).unsqueeze(0)  # (1, 1, D_HEAD)
    out0 = apply_rotary_with_cache(x0, cos[:1], sin[:1])

    x5 = v.unsqueeze(0).unsqueeze(0)
    out5 = apply_rotary_with_cache(x5, cos[5:6], sin[5:6])

    assert not torch.allclose(out0, out5, atol=1e-5), (
        "RoPE output must differ for position 0 vs position 5"
    )


# ---------------------------------------------------------------------------
# 15. base parameter changes the frequencies
# ---------------------------------------------------------------------------
def test_base_changes_frequencies():
    """Different base values must produce different cos/sin caches."""
    cos1, sin1 = build_cos_sin_cache(D_HEAD, MAX_LEN, base=10000.0)
    cos2, sin2 = build_cos_sin_cache(D_HEAD, MAX_LEN, base=500.0)
    # Position 0 is always 1/0; compare from position 1 onward
    assert not torch.allclose(cos1[1:], cos2[1:], atol=1e-5), (
        "Different base values must yield different cosine caches"
    )
