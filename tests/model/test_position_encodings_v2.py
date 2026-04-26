"""Tests for src/model/position_encodings_v2.py.

Uses intentionally small configs (tiny H, T, d_model) so tests run quickly
without GPU resources.
"""

import math

import pytest
import torch

from src.model.position_encodings_v2 import (
    ALiBiAttention,
    LearnedPositionalEncoding,
    PosEncConfig,
    build_alibi_bias,
    build_sinusoidal_encoding,
    compute_alibi_slopes,
    get_position_encoding,
)

# ---------------------------------------------------------------------------
# Small test constants
# ---------------------------------------------------------------------------
N_HEADS = 4
SEQ_LEN = 8
D_MODEL = 16
BATCH = 2


# ===========================================================================
# PosEncConfig
# ===========================================================================


class TestPosEncConfig:
    def test_defaults(self):
        cfg = PosEncConfig()
        assert cfg.n_heads == 8
        assert cfg.max_seq_len == 2048
        assert cfg.alibi_max_slope == 1.0
        assert cfg.encoding_type == "alibi"

    def test_custom_values(self):
        cfg = PosEncConfig(n_heads=4, max_seq_len=512, encoding_type="sinusoidal")
        assert cfg.n_heads == 4
        assert cfg.max_seq_len == 512
        assert cfg.encoding_type == "sinusoidal"

    def test_invalid_encoding_type_raises(self):
        with pytest.raises(ValueError, match="encoding_type"):
            PosEncConfig(encoding_type="banana")

    def test_all_valid_types_accepted(self):
        for t in ("alibi", "sinusoidal", "learned", "none"):
            cfg = PosEncConfig(encoding_type=t)
            assert cfg.encoding_type == t


# ===========================================================================
# compute_alibi_slopes
# ===========================================================================


class TestComputeAlibiSlopes:
    def test_shape(self):
        slopes = compute_alibi_slopes(N_HEADS)
        assert slopes.shape == (N_HEADS,), f"Expected ({N_HEADS},), got {slopes.shape}"

    def test_all_positive(self):
        slopes = compute_alibi_slopes(N_HEADS)
        assert (slopes > 0).all(), "All slopes must be positive"

    def test_strictly_decreasing(self):
        slopes = compute_alibi_slopes(N_HEADS)
        # slope_h = 1/2^(h*8/H); larger h → smaller slope
        diffs = slopes[1:] - slopes[:-1]
        assert (diffs < 0).all(), f"Slopes must be strictly decreasing; diffs={diffs}"

    def test_values_in_zero_one(self):
        slopes = compute_alibi_slopes(N_HEADS)
        assert (slopes > 0).all() and (slopes <= 1).all(), (
            f"All slopes must be in (0, 1]; got min={slopes.min()}, max={slopes.max()}"
        )

    def test_first_slope_formula(self):
        # slope_1 = 1 / 2^(1 * 8 / n_heads)
        for n in (1, 2, 4, 8, 16):
            slopes = compute_alibi_slopes(n)
            expected_first = 0.5 ** (1.0 * 8.0 / n)
            assert abs(slopes[0].item() - expected_first) < 1e-6, (
                f"slope[0] mismatch for n_heads={n}: {slopes[0]} vs {expected_first}"
            )


# ===========================================================================
# build_alibi_bias
# ===========================================================================


class TestBuildAlibiBias:
    def test_shape(self):
        bias = build_alibi_bias(N_HEADS, SEQ_LEN)
        assert bias.shape == (N_HEADS, SEQ_LEN, SEQ_LEN), (
            f"Expected ({N_HEADS}, {SEQ_LEN}, {SEQ_LEN}), got {bias.shape}"
        )

    def test_diagonal_is_zero(self):
        """Self-attention (i == j) should incur no ALiBi penalty."""
        bias = build_alibi_bias(N_HEADS, SEQ_LEN)
        for h in range(N_HEADS):
            diag = torch.diagonal(bias[h])
            assert torch.allclose(diag, torch.zeros(SEQ_LEN)), (
                f"Head {h}: diagonal entries should be 0, got {diag}"
            )

    def test_upper_triangle_is_neg_inf(self):
        """Causal masking: positions j > i must be -inf."""
        bias = build_alibi_bias(N_HEADS, SEQ_LEN)
        for h in range(N_HEADS):
            for i in range(SEQ_LEN):
                for j in range(i + 1, SEQ_LEN):
                    assert bias[h, i, j].isinf() and bias[h, i, j] < 0, (
                        f"bias[{h},{i},{j}] should be -inf, got {bias[h, i, j]}"
                    )

    def test_lower_triangle_non_positive(self):
        """For j <= i the bias value must be <= 0 (0 on diagonal, negative below)."""
        bias = build_alibi_bias(N_HEADS, SEQ_LEN)
        for h in range(N_HEADS):
            for i in range(SEQ_LEN):
                for j in range(i + 1):  # j <= i
                    assert bias[h, i, j] <= 0, (
                        f"bias[{h},{i},{j}] should be <= 0, got {bias[h, i, j]}"
                    )

    def test_penalty_increases_with_distance(self):
        """Farther apart positions should receive a more negative bias."""
        bias = build_alibi_bias(N_HEADS, SEQ_LEN)
        # Row i=5, compare columns j=4 (dist 1) and j=0 (dist 5)
        for h in range(N_HEADS):
            close_val = bias[h, 5, 4]  # |5-4| = 1
            far_val = bias[h, 5, 0]  # |5-0| = 5
            assert far_val < close_val, (
                f"Head {h}: bias at dist=5 ({far_val}) should be < bias at dist=1 ({close_val})"
            )

    def test_dtype_float32(self):
        bias = build_alibi_bias(N_HEADS, SEQ_LEN)
        assert bias.dtype == torch.float32


# ===========================================================================
# build_sinusoidal_encoding
# ===========================================================================


class TestBuildSinusoidalEncoding:
    def test_shape(self):
        pe = build_sinusoidal_encoding(SEQ_LEN, D_MODEL)
        assert pe.shape == (SEQ_LEN, D_MODEL), f"Expected ({SEQ_LEN}, {D_MODEL}), got {pe.shape}"

    def test_values_in_neg1_pos1(self):
        pe = build_sinusoidal_encoding(SEQ_LEN, D_MODEL)
        assert pe.min() >= -1.0 - 1e-6 and pe.max() <= 1.0 + 1e-6, (
            f"Sinusoidal values should be in [-1, 1]; min={pe.min()}, max={pe.max()}"
        )

    def test_position_0_distinct_from_position_1(self):
        pe = build_sinusoidal_encoding(SEQ_LEN, D_MODEL)
        assert not torch.allclose(pe[0], pe[1]), (
            "Position 0 and position 1 encodings should be distinct"
        )

    def test_even_dims_are_sin(self):
        """Even-indexed dimensions should equal sin(t / div_term)."""
        T, D = 4, 8
        pe = build_sinusoidal_encoding(T, D)
        for t in range(T):
            for i in range(D // 2):
                div = 10000.0 ** (2 * i / D)
                expected = math.sin(t / div)
                assert abs(pe[t, 2 * i].item() - expected) < 1e-5, (
                    f"pe[{t},{2 * i}] should be sin: {pe[t, 2 * i]} vs {expected}"
                )

    def test_odd_dims_are_cos(self):
        """Odd-indexed dimensions should equal cos(t / div_term)."""
        T, D = 4, 8
        pe = build_sinusoidal_encoding(T, D)
        for t in range(T):
            for i in range(D // 2):
                div = 10000.0 ** (2 * i / D)
                expected = math.cos(t / div)
                assert abs(pe[t, 2 * i + 1].item() - expected) < 1e-5, (
                    f"pe[{t},{2 * i + 1}] should be cos: {pe[t, 2 * i + 1]} vs {expected}"
                )

    def test_different_base_changes_encoding(self):
        pe_default = build_sinusoidal_encoding(SEQ_LEN, D_MODEL, base=10000.0)
        pe_other = build_sinusoidal_encoding(SEQ_LEN, D_MODEL, base=1000.0)
        assert not torch.allclose(pe_default, pe_other), (
            "Different base values should produce different encodings"
        )


# ===========================================================================
# LearnedPositionalEncoding
# ===========================================================================


class TestLearnedPositionalEncoding:
    def test_forward_shape(self):
        lpe = LearnedPositionalEncoding(max_seq_len=SEQ_LEN, d_model=D_MODEL)
        x = torch.zeros(BATCH, SEQ_LEN, D_MODEL)
        out = lpe(x)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL), (
            f"Expected ({BATCH}, {SEQ_LEN}, {D_MODEL}), got {out.shape}"
        )

    def test_output_changes_input(self):
        """Output should differ from the zero input (embeddings are non-zero by default)."""
        lpe = LearnedPositionalEncoding(max_seq_len=SEQ_LEN, d_model=D_MODEL)
        # Initialise embedding to non-zero values to guarantee change
        nn_init = torch.nn.init.normal_
        nn_init(lpe.embedding.weight)
        x = torch.zeros(BATCH, SEQ_LEN, D_MODEL)
        out = lpe(x)
        assert not torch.allclose(out, x), (
            "LearnedPositionalEncoding should add non-zero offsets to the input"
        )

    def test_embedding_is_learnable(self):
        lpe = LearnedPositionalEncoding(max_seq_len=SEQ_LEN, d_model=D_MODEL)
        assert lpe.embedding.weight.requires_grad, "Embedding weights should require grad"

    def test_shorter_seq_than_max(self):
        """Should work when the actual sequence is shorter than max_seq_len."""
        lpe = LearnedPositionalEncoding(max_seq_len=64, d_model=D_MODEL)
        x = torch.randn(BATCH, 5, D_MODEL)
        out = lpe(x)
        assert out.shape == (BATCH, 5, D_MODEL)


# ===========================================================================
# ALiBiAttention
# ===========================================================================


class TestALiBiAttention:
    def test_forward_shape(self):
        attn = ALiBiAttention(d_model=D_MODEL, n_heads=N_HEADS)
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        out = attn(x)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL), (
            f"Expected ({BATCH}, {SEQ_LEN}, {D_MODEL}), got {out.shape}"
        )

    def test_output_finite(self):
        attn = ALiBiAttention(d_model=D_MODEL, n_heads=N_HEADS)
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        out = attn(x)
        assert torch.isfinite(out).all(), "ALiBiAttention output must be finite"

    def test_slopes_buffer_shape(self):
        attn = ALiBiAttention(d_model=D_MODEL, n_heads=N_HEADS)
        assert attn.slopes.shape == (N_HEADS,), (
            f"slopes buffer shape: expected ({N_HEADS},), got {attn.slopes.shape}"
        )

    def test_slopes_not_trainable(self):
        attn = ALiBiAttention(d_model=D_MODEL, n_heads=N_HEADS)
        # Buffers should not appear in named_parameters
        param_names = {n for n, _ in attn.named_parameters()}
        assert "slopes" not in param_names, "slopes should be a buffer, not a parameter"

    def test_output_differs_across_positions(self):
        """Different token positions should generally produce different outputs."""
        torch.manual_seed(0)
        attn = ALiBiAttention(d_model=D_MODEL, n_heads=N_HEADS)
        x = torch.randn(1, SEQ_LEN, D_MODEL)
        out = attn(x)
        # At least one pair of positions should differ
        all_same = all(torch.allclose(out[0, i], out[0, i + 1]) for i in range(SEQ_LEN - 1))
        assert not all_same, "Output should differ across positions"


# ===========================================================================
# get_position_encoding (dispatcher)
# ===========================================================================


class TestGetPositionEncoding:
    def test_sinusoidal_shape(self):
        cfg = PosEncConfig(encoding_type="sinusoidal", n_heads=N_HEADS)
        out = get_position_encoding(cfg, d_model=D_MODEL, seq_len=SEQ_LEN)
        assert out.shape == (SEQ_LEN, D_MODEL), (
            f"sinusoidal: expected ({SEQ_LEN}, {D_MODEL}), got {out.shape}"
        )

    def test_alibi_shape(self):
        cfg = PosEncConfig(encoding_type="alibi", n_heads=N_HEADS)
        out = get_position_encoding(cfg, d_model=D_MODEL, seq_len=SEQ_LEN)
        assert out.shape == (N_HEADS, SEQ_LEN, SEQ_LEN), (
            f"alibi: expected ({N_HEADS},{SEQ_LEN},{SEQ_LEN}), got {out.shape}"
        )

    def test_learned_returns_zeros(self):
        cfg = PosEncConfig(encoding_type="learned", n_heads=N_HEADS)
        out = get_position_encoding(cfg, d_model=D_MODEL, seq_len=SEQ_LEN)
        assert out.shape == (SEQ_LEN, D_MODEL)
        assert torch.all(out == 0), "learned encoding tensor should be all zeros"

    def test_none_returns_zeros(self):
        cfg = PosEncConfig(encoding_type="none", n_heads=N_HEADS)
        out = get_position_encoding(cfg, d_model=D_MODEL, seq_len=SEQ_LEN)
        assert out.shape == (SEQ_LEN, D_MODEL)
        assert torch.all(out == 0), "none encoding tensor should be all zeros"

    def test_sinusoidal_values_in_range(self):
        cfg = PosEncConfig(encoding_type="sinusoidal")
        out = get_position_encoding(cfg, d_model=D_MODEL, seq_len=SEQ_LEN)
        assert out.min() >= -1.0 - 1e-6 and out.max() <= 1.0 + 1e-6
