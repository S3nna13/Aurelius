"""Tests for ALiBi (Attention with Linear Biases) attention module."""

from __future__ import annotations

import math

import pytest
import torch

from src.model.alibi_attention import (
    ALiBiAttention,
    ALiBiBlock,
    ALiBiConfig,
    build_alibi_bias,
    get_alibi_slopes,
    get_relative_positions,
)

# ---------------------------------------------------------------------------
# Tiny test constants — keep everything small so tests run fast on CPU
# ---------------------------------------------------------------------------

D_MODEL = 16
N_HEADS = 4
MAX_SEQ = 32
BATCH = 2
SEQ = 8


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg() -> ALiBiConfig:
    return ALiBiConfig(d_model=D_MODEL, n_heads=N_HEADS, causal=True, max_seq_len=MAX_SEQ)


@pytest.fixture
def attn(cfg: ALiBiConfig) -> ALiBiAttention:
    return ALiBiAttention(cfg)


@pytest.fixture
def block(cfg: ALiBiConfig) -> ALiBiBlock:
    return ALiBiBlock(cfg)


@pytest.fixture
def x() -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(BATCH, SEQ, D_MODEL)


# ---------------------------------------------------------------------------
# 1. ALiBiConfig defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = ALiBiConfig()
    assert cfg.d_model == 64
    assert cfg.n_heads == 8
    assert cfg.causal is True
    assert cfg.max_seq_len == 2048


# ---------------------------------------------------------------------------
# 2. get_alibi_slopes — shape
# ---------------------------------------------------------------------------


def test_slopes_shape():
    slopes = get_alibi_slopes(N_HEADS)
    assert slopes.shape == (N_HEADS,), f"Expected ({N_HEADS},), got {slopes.shape}"


# ---------------------------------------------------------------------------
# 3. get_alibi_slopes — all positive
# ---------------------------------------------------------------------------


def test_slopes_all_positive():
    slopes = get_alibi_slopes(N_HEADS)
    assert (slopes > 0).all(), "All slopes must be positive"


# ---------------------------------------------------------------------------
# 4. get_alibi_slopes — strictly decreasing with head index
# ---------------------------------------------------------------------------


def test_slopes_decreasing():
    slopes = get_alibi_slopes(N_HEADS)
    # Each subsequent slope should be strictly smaller
    diffs = slopes[1:] - slopes[:-1]
    assert (diffs < 0).all(), f"Slopes must be strictly decreasing, diffs={diffs}"


# ---------------------------------------------------------------------------
# 5. get_alibi_slopes — non-power-of-2 heads still returns correct shape
# ---------------------------------------------------------------------------


def test_slopes_non_power_of_2():
    for n in (3, 5, 6, 7, 12):
        slopes = get_alibi_slopes(n)
        assert slopes.shape == (n,), f"n={n}: expected ({n},), got {slopes.shape}"
        assert (slopes > 0).all(), f"n={n}: slopes must be positive"


# ---------------------------------------------------------------------------
# 6. build_alibi_bias — shape (n_heads, T, T)
# ---------------------------------------------------------------------------


def test_bias_shape():
    bias = build_alibi_bias(N_HEADS, SEQ)
    assert bias.shape == (N_HEADS, SEQ, SEQ), f"Expected ({N_HEADS},{SEQ},{SEQ}), got {bias.shape}"


# ---------------------------------------------------------------------------
# 7. build_alibi_bias — diagonal is 0 (|i-i| = 0)
# ---------------------------------------------------------------------------


def test_bias_diagonal_zero():
    # Use non-causal so we can inspect all values cleanly
    bias = build_alibi_bias(N_HEADS, SEQ, causal=False)
    for h in range(N_HEADS):
        diag = torch.diagonal(bias[h])
        assert torch.allclose(diag, torch.zeros(SEQ)), (
            f"Head {h}: diagonal should be 0, got {diag}"
        )


# ---------------------------------------------------------------------------
# 8. build_alibi_bias — causal: future positions are -inf
# ---------------------------------------------------------------------------


def test_causal_bias_upper_triangle_is_neg_inf():
    bias = build_alibi_bias(N_HEADS, SEQ, causal=True)
    for h in range(N_HEADS):
        # Upper triangle (j > i) should be -inf
        for i in range(SEQ):
            for j in range(i + 1, SEQ):
                assert bias[h, i, j] == float("-inf"), (
                    f"Head {h}, position ({i},{j}) should be -inf for causal"
                )


# ---------------------------------------------------------------------------
# 9. build_alibi_bias — non-causal: no -inf values
# ---------------------------------------------------------------------------


def test_non_causal_bias_no_neg_inf():
    bias = build_alibi_bias(N_HEADS, SEQ, causal=False)
    assert not torch.isinf(bias).any(), "Non-causal bias should have no -inf values"


# ---------------------------------------------------------------------------
# 10. get_relative_positions — shape and diagonal = 0
# ---------------------------------------------------------------------------


def test_relative_positions_shape_and_diagonal():
    rel = get_relative_positions(SEQ)
    assert rel.shape == (SEQ, SEQ), f"Expected ({SEQ},{SEQ}), got {rel.shape}"
    diag = torch.diagonal(rel)
    assert (diag == 0).all(), "Diagonal (|i-i|) must be 0"


# ---------------------------------------------------------------------------
# 11. get_relative_positions — values are non-negative and symmetric
# ---------------------------------------------------------------------------


def test_relative_positions_non_negative_and_symmetric():
    rel = get_relative_positions(SEQ)
    assert (rel >= 0).all(), "All distances must be non-negative"
    assert torch.equal(rel, rel.T), "|i-j| must equal |j-i|"


# ---------------------------------------------------------------------------
# 12. ALiBiAttention — output shape
# ---------------------------------------------------------------------------


def test_attention_output_shape(attn, x):
    out = attn(x)
    assert out.shape == (BATCH, SEQ, D_MODEL), (
        f"Expected ({BATCH},{SEQ},{D_MODEL}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 13. ALiBiAttention — gradient flows through
# ---------------------------------------------------------------------------


def test_attention_gradient_flows(attn, x):
    x_req = x.requires_grad_(True)
    out = attn(x_req)
    loss = out.sum()
    loss.backward()
    assert x_req.grad is not None, "Gradient must flow back to input"
    assert not torch.isnan(x_req.grad).any(), "Gradient contains NaN"


# ---------------------------------------------------------------------------
# 14. ALiBiBlock — output shape
# ---------------------------------------------------------------------------


def test_block_output_shape(block, x):
    out = block(x)
    assert out.shape == (BATCH, SEQ, D_MODEL), (
        f"Expected ({BATCH},{SEQ},{D_MODEL}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 15. ALiBiBlock — residual (output should differ from input)
# ---------------------------------------------------------------------------


def test_block_residual_changes_input(block, x):
    out = block(x)
    assert not torch.allclose(out, x), (
        "ALiBiBlock output should not equal input (residual + attention)"
    )


# ---------------------------------------------------------------------------
# 16. Single-token sequence
# ---------------------------------------------------------------------------


def test_single_token_sequence(attn):
    x_single = torch.randn(1, 1, D_MODEL)
    out = attn(x_single)
    assert out.shape == (1, 1, D_MODEL)


# ---------------------------------------------------------------------------
# 17. Longer sequence than MAX_SEQ would fail, but a shorter one is fine
# ---------------------------------------------------------------------------


def test_sequence_shorter_than_max_seq(attn):
    """Any T <= max_seq_len should work via bias slicing."""
    for seq_len in (1, 4, SEQ, MAX_SEQ):
        x_var = torch.randn(1, seq_len, D_MODEL)
        out = attn(x_var)
        assert out.shape == (1, seq_len, D_MODEL), (
            f"seq_len={seq_len}: shape mismatch"
        )


# ---------------------------------------------------------------------------
# 18. ALiBiAttention has no bias parameters
# ---------------------------------------------------------------------------


def test_no_bias_parameters(attn):
    for name, _ in attn.named_parameters():
        assert "bias" not in name, f"Unexpected bias parameter: {name}"


# ---------------------------------------------------------------------------
# 19. Slopes match expected formula for power-of-2 heads
# ---------------------------------------------------------------------------


def test_slopes_match_formula_power_of_2():
    n = 4
    slopes = get_alibi_slopes(n)
    for i, s in enumerate(slopes.tolist(), start=1):
        expected = 2.0 ** (-8.0 * i / n)
        assert abs(s - expected) < 1e-6, (
            f"Head {i}: expected {expected:.6f}, got {s:.6f}"
        )


# ---------------------------------------------------------------------------
# 20. build_alibi_bias lower triangle values are negative (non-zero distance)
# ---------------------------------------------------------------------------


def test_bias_lower_triangle_negative(cfg):
    bias = build_alibi_bias(cfg.n_heads, SEQ, causal=False)
    # Below diagonal (i > j): distances > 0, slopes > 0 -> bias < 0
    for h in range(cfg.n_heads):
        for i in range(1, SEQ):
            for j in range(i):
                assert bias[h, i, j] < 0, (
                    f"Head {h}, pos ({i},{j}): expected negative bias, got {bias[h,i,j]}"
                )
