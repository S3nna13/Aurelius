"""Tests for src/model/rope_embeddings.py.

Uses tiny configs to keep tests fast:
    D_HEAD=8, D_MODEL=16, N_HEADS=2, MAX_SEQ=16, BATCH=2, SEQ=6
"""

import torch

from src.model.rope_embeddings import (
    RoPEAttention,
    RoPEConfig,
    apply_rotary_emb,
    apply_rotary_emb_real,
    build_rope_cache,
    compute_freqs_cis,
    rotate_half,
)

# ---------------------------------------------------------------------------
# Tiny constants used throughout
# ---------------------------------------------------------------------------
D_HEAD = 8
D_MODEL = 16
N_HEADS = 2
MAX_SEQ = 16
BATCH = 2
SEQ = 6


# ---------------------------------------------------------------------------
# 1. compute_freqs_cis — shape
# ---------------------------------------------------------------------------
def test_compute_freqs_cis_shape():
    """compute_freqs_cis must return (SEQ, D_HEAD//2)."""
    freqs = compute_freqs_cis(D_HEAD, SEQ)
    assert freqs.shape == (SEQ, D_HEAD // 2), f"Expected ({SEQ}, {D_HEAD // 2}), got {freqs.shape}"


# ---------------------------------------------------------------------------
# 2. compute_freqs_cis — complex dtype
# ---------------------------------------------------------------------------
def test_compute_freqs_cis_is_complex():
    """compute_freqs_cis must return a complex tensor."""
    freqs = compute_freqs_cis(D_HEAD, SEQ)
    assert freqs.is_complex(), "freqs_cis must be a complex tensor"


# ---------------------------------------------------------------------------
# 3. apply_rotary_emb — output shape matches input
# ---------------------------------------------------------------------------
def test_apply_rotary_emb_output_shape():
    """apply_rotary_emb must return a tensor with the same shape as x."""
    freqs = compute_freqs_cis(D_HEAD, MAX_SEQ)
    x = torch.randn(BATCH, SEQ, N_HEADS, D_HEAD)
    out = apply_rotary_emb(x, freqs)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


# ---------------------------------------------------------------------------
# 4. RoPEAttention — output shape
# ---------------------------------------------------------------------------
def test_rope_attention_output_shape():
    """RoPEAttention forward must return (B, T, d_model)."""
    config = RoPEConfig(d_head=D_HEAD, max_seq_len=MAX_SEQ)
    attn = RoPEAttention(D_MODEL, N_HEADS, config)
    x = torch.randn(BATCH, SEQ, D_MODEL)
    out = attn(x)
    assert out.shape == (BATCH, SEQ, D_MODEL), (
        f"Expected ({BATCH}, {SEQ}, {D_MODEL}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 5. Gradient flows through apply_rotary_emb
# ---------------------------------------------------------------------------
def test_apply_rotary_emb_gradient_flows():
    """Gradient must flow through apply_rotary_emb back to the input."""
    freqs = compute_freqs_cis(D_HEAD, MAX_SEQ)
    x = torch.randn(BATCH, SEQ, N_HEADS, D_HEAD, requires_grad=True)
    out = apply_rotary_emb(x, freqs)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "Gradient did not flow back to x"
    assert x.grad.shape == x.shape


# ---------------------------------------------------------------------------
# 6. RoPE is position-dependent (different positions give different outputs)
# ---------------------------------------------------------------------------
def test_rope_is_position_dependent():
    """Rotating the same vector at position 0 vs position 5 must differ."""
    freqs = compute_freqs_cis(D_HEAD, MAX_SEQ)
    v = torch.randn(D_HEAD)

    # Position 0: use a (1, 1, 1, D_HEAD) tensor
    x0 = v.view(1, 1, 1, D_HEAD)
    out0 = apply_rotary_emb(x0, freqs[:1])

    # Position 5: slice freqs starting at position 5
    x5 = v.view(1, 1, 1, D_HEAD)
    out5 = apply_rotary_emb(x5, freqs[5:6])

    assert not torch.allclose(out0, out5, atol=1e-5), (
        "RoPE output must differ between position 0 and position 5"
    )


# ---------------------------------------------------------------------------
# 7. build_rope_cache — shape matches config
# ---------------------------------------------------------------------------
def test_build_rope_cache_shape():
    """build_rope_cache must return (max_seq_len, d_head // 2)."""
    config = RoPEConfig(d_head=D_HEAD, max_seq_len=MAX_SEQ)
    cache = build_rope_cache(config)
    assert cache.shape == (MAX_SEQ, D_HEAD // 2), (
        f"Expected ({MAX_SEQ}, {D_HEAD // 2}), got {cache.shape}"
    )


# ---------------------------------------------------------------------------
# 8. rotate_half — output shape
# ---------------------------------------------------------------------------
def test_rotate_half_output_shape():
    """rotate_half must preserve input shape."""
    x = torch.randn(BATCH, SEQ, D_HEAD)
    out = rotate_half(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


# ---------------------------------------------------------------------------
# 9. rotate_half — applying twice equals negation (rotate_half(rotate_half(x)) == -x)
# ---------------------------------------------------------------------------
def test_rotate_half_twice_is_negation():
    """Applying rotate_half twice must equal -x (180-degree rotation)."""
    x = torch.randn(BATCH, SEQ, D_HEAD)
    result = rotate_half(rotate_half(x))
    assert torch.allclose(result, -x, atol=1e-6), "rotate_half(rotate_half(x)) must equal -x"


# ---------------------------------------------------------------------------
# 10. apply_rotary_emb_real — output shapes
# ---------------------------------------------------------------------------
def test_apply_rotary_emb_real_output_shapes():
    """apply_rotary_emb_real must return q and k with same shapes as inputs.

    Uses 3-D q/k (B, T, D_HEAD) so that cos/sin (T, D_HEAD) broadcast cleanly.
    The function is shape-agnostic on the leading dims.
    """
    q = torch.randn(BATCH, SEQ, D_HEAD)
    k = torch.randn(BATCH, SEQ, D_HEAD)
    t = torch.arange(SEQ, dtype=torch.float32)
    i = torch.arange(0, D_HEAD, 2, dtype=torch.float32)
    theta = 1.0 / (10000.0 ** (i / D_HEAD))
    angles = torch.outer(t, theta)  # (SEQ, D_HEAD//2)
    # Expand angles to full d_head (rotate_half splits into two equal halves)
    angles_full = torch.cat([angles, angles], dim=-1)  # (SEQ, D_HEAD)
    cos = angles_full.cos()
    sin = angles_full.sin()

    q_rot, k_rot = apply_rotary_emb_real(q, k, cos, sin)
    assert q_rot.shape == q.shape, f"Expected {q.shape}, got {q_rot.shape}"
    assert k_rot.shape == k.shape, f"Expected {k.shape}, got {k_rot.shape}"


# ---------------------------------------------------------------------------
# 11. apply_rotary_emb_real — cos=1, sin=0 is identity (no rotation)
# ---------------------------------------------------------------------------
def test_apply_rotary_emb_real_identity():
    """When cos=1 and sin=0, apply_rotary_emb_real must be a no-op."""
    q = torch.randn(BATCH, SEQ, D_HEAD)
    k = torch.randn(BATCH, SEQ, D_HEAD)
    cos = torch.ones(SEQ, D_HEAD)
    sin = torch.zeros(SEQ, D_HEAD)

    q_rot, k_rot = apply_rotary_emb_real(q, k, cos, sin)
    assert torch.allclose(q_rot, q, atol=1e-6), "q must be unchanged when cos=1, sin=0"
    assert torch.allclose(k_rot, k, atol=1e-6), "k must be unchanged when cos=1, sin=0"


# ---------------------------------------------------------------------------
# 12. freqs_cis unit magnitude (|exp(i*theta)| = 1)
# ---------------------------------------------------------------------------
def test_freqs_cis_unit_magnitude():
    """All entries of freqs_cis must have magnitude exactly 1."""
    freqs = compute_freqs_cis(D_HEAD, MAX_SEQ)
    magnitudes = freqs.abs()
    expected = torch.ones_like(magnitudes)
    assert torch.allclose(magnitudes, expected, atol=1e-5), (
        "freqs_cis entries must all have magnitude 1 (they are unit complex numbers)"
    )


# ---------------------------------------------------------------------------
# 13. build_rope_cache matches compute_freqs_cis directly
# ---------------------------------------------------------------------------
def test_build_rope_cache_matches_compute_freqs_cis():
    """build_rope_cache must produce the same tensor as compute_freqs_cis."""
    config = RoPEConfig(d_head=D_HEAD, max_seq_len=MAX_SEQ, base=5000.0, scale=2.0)
    cache = build_rope_cache(config)
    direct = compute_freqs_cis(D_HEAD, MAX_SEQ, base=5000.0, scale=2.0)
    assert torch.allclose(cache.real, direct.real, atol=1e-6)
    assert torch.allclose(cache.imag, direct.imag, atol=1e-6)


# ---------------------------------------------------------------------------
# 14. RoPEAttention gradient flows end-to-end
# ---------------------------------------------------------------------------
def test_rope_attention_gradient_flows():
    """Gradient must propagate through RoPEAttention to the input."""
    config = RoPEConfig(d_head=D_HEAD, max_seq_len=MAX_SEQ)
    attn = RoPEAttention(D_MODEL, N_HEADS, config)
    x = torch.randn(BATCH, SEQ, D_MODEL, requires_grad=True)
    out = attn(x)
    out.sum().backward()
    assert x.grad is not None, "Gradient must flow back through RoPEAttention"
    assert x.grad.shape == x.shape


# ---------------------------------------------------------------------------
# 15. scale parameter changes the frequencies
# ---------------------------------------------------------------------------
def test_scale_changes_frequencies():
    """Different scale values must produce different freqs_cis tensors."""
    freqs_1 = compute_freqs_cis(D_HEAD, MAX_SEQ, scale=1.0)
    freqs_2 = compute_freqs_cis(D_HEAD, MAX_SEQ, scale=2.0)
    # Angles differ by factor 2, so the complex values must differ (beyond pos 0)
    assert not torch.allclose(freqs_1[1:].real, freqs_2[1:].real, atol=1e-5), (
        "scale=1.0 and scale=2.0 must yield different frequency tensors"
    )
