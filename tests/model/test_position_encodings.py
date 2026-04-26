"""Tests for src/model/position_encodings.py."""

import torch

from src.model.position_encodings import (
    ALiBiPositionBias,
    NoPE,
    T5RelativePositionBias,
    get_position_encoding,
    xPosEmbedding,
)

N_HEADS = 4
HEAD_DIM = 16
DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# ALiBi tests
# ---------------------------------------------------------------------------


def test_alibi_slopes_shape():
    alibi = ALiBiPositionBias(n_heads=N_HEADS)
    assert alibi.slopes.shape == (N_HEADS,), (
        f"Expected slopes shape ({N_HEADS},), got {alibi.slopes.shape}"
    )


def test_alibi_get_bias_shape():
    alibi = ALiBiPositionBias(n_heads=N_HEADS)
    bias = alibi.get_bias(16, DEVICE)
    assert bias.shape == (1, N_HEADS, 16, 16), (
        f"Expected bias shape (1, {N_HEADS}, 16, 16), got {bias.shape}"
    )


def test_alibi_bias_causal_decreases_with_distance():
    """|bias[0, 0, i, j]| should increase (or stay equal) as |i-j| grows."""
    alibi = ALiBiPositionBias(n_heads=N_HEADS)
    bias = alibi.get_bias(16, DEVICE)
    # Check row i=8: bias at j=7 should be <= bias at j=0 (further = more negative)
    row = bias[0, 0, 8, :]  # head 0, query position 8
    # |i - j| increases as j moves away from 8
    # bias should be non-positive and more negative for larger distances
    distances = torch.arange(16).float()
    abs_dist_from_8 = (distances - 8).abs()
    # Verify: where distance is larger, bias should be more negative (smaller)
    for j1 in range(16):
        for j2 in range(16):
            if abs_dist_from_8[j1] > abs_dist_from_8[j2]:
                assert row[j1] <= row[j2], (
                    f"bias at j={j1} (dist={abs_dist_from_8[j1]:.0f}) should be <= "
                    f"bias at j={j2} (dist={abs_dist_from_8[j2]:.0f})"
                )


def test_alibi_forward_changes_scores():
    """ALiBi forward should change the attention scores."""
    alibi = ALiBiPositionBias(n_heads=N_HEADS)
    scores = torch.zeros(2, N_HEADS, 8, 8)
    out = alibi(scores)
    assert not torch.allclose(out, scores), "ALiBi forward should modify attention scores"


# ---------------------------------------------------------------------------
# xPos tests
# ---------------------------------------------------------------------------


def test_xpos_output_shapes():
    xpos = xPosEmbedding(head_dim=HEAD_DIM)
    q = torch.randn(2, N_HEADS, 8, HEAD_DIM)
    k = torch.randn(2, N_HEADS, 8, HEAD_DIM)
    scaled_q, scaled_k = xpos.apply_xpos(q, k)
    assert scaled_q.shape == q.shape, f"scaled_q shape mismatch: {scaled_q.shape} vs {q.shape}"
    assert scaled_k.shape == k.shape, f"scaled_k shape mismatch: {scaled_k.shape} vs {k.shape}"


def test_xpos_scales_are_different_q_k():
    """scaled_q and scaled_k should differ because they are scaled by s and 1/s."""
    xpos = xPosEmbedding(head_dim=HEAD_DIM)
    # Use identical q and k to make comparison straightforward
    x = torch.ones(2, N_HEADS, 8, HEAD_DIM)
    scaled_q, scaled_k = xpos.apply_xpos(x.clone(), x.clone())
    assert not torch.allclose(scaled_q, scaled_k), (
        "scaled_q and scaled_k should differ (one multiplied by s, other by 1/s)"
    )


# ---------------------------------------------------------------------------
# NoPE tests
# ---------------------------------------------------------------------------


def test_nope_passthrough():
    nope = NoPE()
    q = torch.randn(2, N_HEADS, 8, HEAD_DIM)
    k = torch.randn(2, N_HEADS, 8, HEAD_DIM)
    out_q, out_k = nope(q, k)
    assert torch.equal(out_q, q), "NoPE should return q unchanged"
    assert torch.equal(out_k, k), "NoPE should return k unchanged"


# ---------------------------------------------------------------------------
# T5 Relative Position Bias tests
# ---------------------------------------------------------------------------


def test_t5_bias_shape():
    t5 = T5RelativePositionBias(n_heads=N_HEADS)
    bias = t5(8, 8, DEVICE)
    assert bias.shape == (1, N_HEADS, 8, 8), (
        f"Expected T5 bias shape (1, {N_HEADS}, 8, 8), got {bias.shape}"
    )


def test_t5_bias_learnable():
    t5 = T5RelativePositionBias(n_heads=N_HEADS)
    assert isinstance(t5.embedding, torch.nn.Embedding), (
        "T5RelativePositionBias.embedding should be nn.Embedding"
    )
    assert t5.embedding.weight.requires_grad, "T5 embedding weights should require grad (learnable)"


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


def test_get_position_encoding_factory():
    alibi = get_position_encoding("alibi", n_heads=N_HEADS, head_dim=HEAD_DIM)
    assert isinstance(alibi, ALiBiPositionBias)

    xpos = get_position_encoding("xpos", n_heads=N_HEADS, head_dim=HEAD_DIM)
    assert isinstance(xpos, xPosEmbedding)

    nope = get_position_encoding("nope", n_heads=N_HEADS, head_dim=HEAD_DIM)
    assert isinstance(nope, NoPE)

    t5 = get_position_encoding("t5", n_heads=N_HEADS, head_dim=HEAD_DIM)
    assert isinstance(t5, T5RelativePositionBias)

    rope = get_position_encoding("rope", n_heads=N_HEADS, head_dim=HEAD_DIM)
    assert rope is None, "rope pe_type should return None (handled by attention.py)"
