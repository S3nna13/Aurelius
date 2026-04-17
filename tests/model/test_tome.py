"""Tests for src/model/tome.py — ToMe (Token Merging) module."""

from __future__ import annotations

import torch
import pytest

from aurelius.model.tome import (
    BipartiteSoftMatching,
    ToMeMerger,
    ToMeAttention,
    ToMeBlock,
)

# ---------------------------------------------------------------------------
# Shared fixtures / constants
# ---------------------------------------------------------------------------

B   = 2   # batch size
T   = 16  # sequence length
D   = 32  # embedding dim
R   = 4   # tokens to merge


def _rand(B: int, T: int, D: int) -> torch.Tensor:
    return torch.randn(B, T, D)


# ===========================================================================
# BipartiteSoftMatching
# ===========================================================================


def test_match_output_shape():
    """match() returns src and dst tensors of shape (B, r)."""
    bsm = BipartiteSoftMatching(r=R)
    metric = _rand(B, T, D)
    src, dst = bsm.match(metric)
    assert src.shape == (B, R), f"src shape {src.shape} != ({B}, {R})"
    assert dst.shape == (B, R), f"dst shape {dst.shape} != ({B}, {R})"


def test_match_indices_distinct():
    """src and dst indices must be different (no token merged with itself)."""
    bsm = BipartiteSoftMatching(r=R)
    metric = _rand(B, T, D)
    src, dst = bsm.match(metric)
    # src comes from even positions, dst from odd — they can never be equal
    assert (src != dst).all(), "some src == dst indices found"


def test_match_r_zero():
    """r=0 returns empty index tensors without error."""
    bsm = BipartiteSoftMatching(r=0)
    metric = _rand(B, T, D)
    src, dst = bsm.match(metric)
    assert src.shape == (B, 0)
    assert dst.shape == (B, 0)


def test_match_r_exceeds_half_T():
    """r > T//2 is clamped to T//2 and does not raise."""
    bsm = BipartiteSoftMatching(r=T * 10)  # way too large
    metric = _rand(B, T, D)
    src, dst = bsm.match(metric)
    assert src.shape[0] == B
    assert src.shape[1] <= T // 2
    assert dst.shape[1] <= T // 2


# ===========================================================================
# ToMeMerger
# ===========================================================================


def test_merge_output_shape():
    """merge() reduces sequence length from T to T-r."""
    bsm = BipartiteSoftMatching(r=R)
    metric = _rand(B, T, D)
    src, dst = bsm.match(metric)

    x = _rand(B, T, D)
    x_merged, sz = ToMeMerger.merge(x, src, dst)
    assert x_merged.shape == (B, T - R, D), (
        f"merged shape {x_merged.shape} != ({B}, {T - R}, {D})"
    )


def test_merge_output_finite():
    """merge() output contains no NaN or Inf values."""
    bsm = BipartiteSoftMatching(r=R)
    src, dst = bsm.match(_rand(B, T, D))
    x = _rand(B, T, D)
    x_merged, _ = ToMeMerger.merge(x, src, dst)
    assert torch.isfinite(x_merged).all()


def test_unmerge_restores_shape():
    """unmerge() restores the original sequence length T."""
    bsm = BipartiteSoftMatching(r=R)
    src, dst = bsm.match(_rand(B, T, D))
    x = _rand(B, T, D)
    x_merged, _ = ToMeMerger.merge(x, src, dst)
    x_restored = ToMeMerger.unmerge(x_merged, src, dst, T_orig=T)
    assert x_restored.shape == (B, T, D), (
        f"restored shape {x_restored.shape} != ({B}, {T}, {D})"
    )


def test_unmerge_after_merge_identity_when_r0():
    """When r=0, merge→unmerge is an identity operation."""
    bsm = BipartiteSoftMatching(r=0)
    src, dst = bsm.match(_rand(B, T, D))
    x = _rand(B, T, D)
    x_merged, _ = ToMeMerger.merge(x, src, dst)
    x_restored = ToMeMerger.unmerge(x_merged, src, dst, T_orig=T)
    assert torch.allclose(x, x_restored, atol=1e-5), "r=0 should be identity"


# ===========================================================================
# ToMeAttention
# ===========================================================================


def test_tome_attention_output_shape():
    """ToMeAttention output has the same shape as the input."""
    attn = ToMeAttention(d_model=D, n_heads=4, r=R)
    x = _rand(B, T, D)
    out = attn(x)
    assert out.shape == (B, T, D), f"output shape {out.shape} != ({B}, {T}, {D})"


def test_tome_attention_output_finite():
    """ToMeAttention output is finite (no NaN/Inf)."""
    attn = ToMeAttention(d_model=D, n_heads=4, r=R)
    x = _rand(B, T, D)
    out = attn(x)
    assert torch.isfinite(out).all()


def test_tome_attention_r0_shape():
    """With r=0, output shape equals input shape exactly."""
    attn = ToMeAttention(d_model=D, n_heads=4, r=0)
    x = _rand(B, T, D)
    out = attn(x)
    assert out.shape == (B, T, D)


def test_tome_attention_gradient_flows():
    """Gradients flow back through ToMeAttention."""
    attn = ToMeAttention(d_model=D, n_heads=4, r=R)
    x = _rand(B, T, D).requires_grad_(True)
    out = attn(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "no gradient on input"
    assert torch.isfinite(x.grad).all(), "gradient contains NaN/Inf"


# ===========================================================================
# ToMeBlock
# ===========================================================================


def test_tome_block_output_shape():
    """ToMeBlock output shape equals input shape."""
    block = ToMeBlock(d_model=D, n_heads=4, d_ff=64, r=R)
    x = _rand(B, T, D)
    out = block(x)
    assert out.shape == (B, T, D), f"block output shape {out.shape} != ({B}, {T}, {D})"


def test_tome_block_gradient_flows():
    """Gradients flow back through the full ToMeBlock."""
    block = ToMeBlock(d_model=D, n_heads=4, d_ff=64, r=R)
    x = _rand(B, T, D).requires_grad_(True)
    out = block(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "no gradient on input"
    assert torch.isfinite(x.grad).all(), "gradient contains NaN/Inf"
