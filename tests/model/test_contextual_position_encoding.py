"""Tests for contextual position encoding: ALiBi, KERPLE, T5 Relative, Learned Absolute."""

from __future__ import annotations

import torch

from src.model.contextual_position_encoding import (
    ALiBiPositionBias,
    KERPLEBias,
    LearnedAbsolutePositionEncoding,
    PositionEncodingConfig,
    T5RelativePositionBias,
)

# Common test dimensions
N_HEADS = 4
T = 8
D_MODEL = 16
N_BUCKETS = 8


# ---------------------------------------------------------------------------
# PositionEncodingConfig
# ---------------------------------------------------------------------------


class TestPositionEncodingConfig:
    def test_defaults(self):
        cfg = PositionEncodingConfig(n_heads=N_HEADS, d_model=D_MODEL)
        assert cfg.max_seq_len == 2048
        assert cfg.alibi_slopes is None

    def test_custom_slopes(self):
        slopes = [0.5, 0.25, 0.125, 0.0625]
        cfg = PositionEncodingConfig(n_heads=N_HEADS, d_model=D_MODEL, alibi_slopes=slopes)
        assert cfg.alibi_slopes == slopes


# ---------------------------------------------------------------------------
# ALiBiPositionBias
# ---------------------------------------------------------------------------


class TestALiBiPositionBias:
    def test_get_slopes_length(self):
        """ALiBi slopes length equals n_heads."""
        module = ALiBiPositionBias(n_heads=N_HEADS)
        slopes = module.get_slopes(N_HEADS)
        assert len(slopes) == N_HEADS

    def test_slopes_strictly_decreasing(self):
        """ALiBi slopes are strictly decreasing (larger penalty for later heads)."""
        module = ALiBiPositionBias(n_heads=N_HEADS)
        slopes = module.get_slopes(N_HEADS)
        for i in range(len(slopes) - 1):
            assert slopes[i] > slopes[i + 1], (
                f"slopes[{i}]={slopes[i].item():.6f} should be > slopes[{i + 1}]={slopes[i + 1].item():.6f}"  # noqa: E501
            )

    def test_forward_shape(self):
        """ALiBi forward output has shape (n_heads, T, T)."""
        module = ALiBiPositionBias(n_heads=N_HEADS)
        bias = module(seq_len=T)
        assert bias.shape == (N_HEADS, T, T)

    def test_diagonal_zero(self):
        """ALiBi bias[h, i, i] == 0 for all heads (no penalty for same position)."""
        module = ALiBiPositionBias(n_heads=N_HEADS)
        bias = module(seq_len=T)
        for h in range(N_HEADS):
            diag = torch.diagonal(bias[h])
            assert torch.allclose(diag, torch.zeros(T)), f"Head {h} diagonal not zero"

    def test_off_diagonal_negative(self):
        """ALiBi bias[h, i, j] < 0 for i > j (past positions penalized)."""
        module = ALiBiPositionBias(n_heads=N_HEADS)
        bias = module(seq_len=T)
        # Check lower triangle (i > j), excluding diagonal
        for h in range(N_HEADS):
            for i in range(1, T):
                for j in range(i):
                    assert bias[h, i, j].item() < 0, (
                        f"Head {h}, i={i}, j={j}: expected negative, got {bias[h, i, j].item()}"
                    )

    def test_causal_future_masked(self):
        """ALiBi bias[h, i, j] == -inf for i < j (future positions masked)."""
        module = ALiBiPositionBias(n_heads=N_HEADS)
        bias = module(seq_len=T)
        for h in range(N_HEADS):
            for i in range(T - 1):
                for j in range(i + 1, T):
                    assert bias[h, i, j].item() == float("-inf"), (
                        f"Head {h}, i={i}, j={j}: expected -inf, got {bias[h, i, j].item()}"
                    )

    def test_custom_slopes(self):
        """ALiBi accepts custom slopes."""
        slopes = [0.5, 0.25, 0.125, 0.0625]
        module = ALiBiPositionBias(n_heads=N_HEADS, slopes=slopes)
        bias = module(seq_len=T)
        assert bias.shape == (N_HEADS, T, T)


# ---------------------------------------------------------------------------
# T5RelativePositionBias
# ---------------------------------------------------------------------------


class TestT5RelativePositionBias:
    def test_forward_shape(self):
        """T5 bias forward shape is (n_heads, T, T)."""
        module = T5RelativePositionBias(n_heads=N_HEADS, n_buckets=N_BUCKETS, max_distance=32)
        bias = module(seq_len=T)
        assert bias.shape == (N_HEADS, T, T)

    def test_bias_is_finite(self):
        """T5 bias values are all finite (no inf or nan)."""
        module = T5RelativePositionBias(n_heads=N_HEADS, n_buckets=N_BUCKETS, max_distance=32)
        bias = module(seq_len=T)
        assert torch.isfinite(bias).all(), "T5 bias contains non-finite values"

    def test_gradient_flows(self):
        """T5 bias is learnable — gradient flows through the embedding."""
        module = T5RelativePositionBias(n_heads=N_HEADS, n_buckets=N_BUCKETS, max_distance=32)
        bias = module(seq_len=T)
        loss = bias.sum()
        loss.backward()
        grad = module.relative_attention_bias.weight.grad
        assert grad is not None, "No gradient on T5 embedding weight"
        assert grad.shape == module.relative_attention_bias.weight.shape

    def test_different_seq_lens(self):
        """T5 bias works for various sequence lengths."""
        module = T5RelativePositionBias(n_heads=N_HEADS, n_buckets=N_BUCKETS, max_distance=32)
        for seq_len in [1, 4, 16]:
            bias = module(seq_len=seq_len)
            assert bias.shape == (N_HEADS, seq_len, seq_len)


# ---------------------------------------------------------------------------
# LearnedAbsolutePositionEncoding
# ---------------------------------------------------------------------------


class TestLearnedAbsolutePositionEncoding:
    def test_forward_shape(self):
        """Learned absolute PE forward shape is (1, T, d_model)."""
        module = LearnedAbsolutePositionEncoding(max_seq_len=T * 4, d_model=D_MODEL)
        out = module(seq_len=T)
        assert out.shape == (1, T, D_MODEL)

    def test_interpolate_changes_weight_size(self):
        """Interpolate updates weight to the new max sequence length."""
        module = LearnedAbsolutePositionEncoding(max_seq_len=T, d_model=D_MODEL)
        new_len = T * 4
        module.interpolate(new_max_len=new_len)
        assert module.weight.shape[0] == new_len
        assert module.weight.shape[1] == D_MODEL
        assert module.max_seq_len == new_len

    def test_weight_is_learnable(self):
        """Learned PE weight is a trainable parameter."""
        module = LearnedAbsolutePositionEncoding(max_seq_len=T, d_model=D_MODEL)
        assert module.weight.requires_grad

    def test_forward_sub_slice(self):
        """Forward with seq_len < max_seq_len returns correct slice."""
        max_len = 32
        module = LearnedAbsolutePositionEncoding(max_seq_len=max_len, d_model=D_MODEL)
        out = module(seq_len=T)
        assert out.shape == (1, T, D_MODEL)
        # Should match first T rows of weight
        assert torch.allclose(out.squeeze(0), module.weight[:T])


# ---------------------------------------------------------------------------
# KERPLEBias
# ---------------------------------------------------------------------------


class TestKERPLEBias:
    def test_forward_shape(self):
        """KERPLE forward shape is (n_heads, T, T)."""
        module = KERPLEBias(n_heads=N_HEADS)
        bias = module(seq_len=T)
        assert bias.shape == (N_HEADS, T, T)

    def test_diagonal_is_zero(self):
        """KERPLE bias[h, i, i] == 0 because |i - i|^2 = 0."""
        module = KERPLEBias(n_heads=N_HEADS)
        bias = module(seq_len=T)
        for h in range(N_HEADS):
            diag = torch.diagonal(bias[h])
            assert torch.allclose(diag, torch.zeros(T)), f"Head {h} diagonal not zero"

    def test_off_diagonal_negative(self):
        """KERPLE off-diagonal entries are negative when r > 0."""
        module = KERPLEBias(n_heads=N_HEADS, init_r=1.0)
        bias = module(seq_len=T)
        # All off-diagonal elements should be strictly negative
        for h in range(N_HEADS):
            for i in range(T):
                for j in range(T):
                    if i != j:
                        assert bias[h, i, j].item() < 0, (
                            f"Head {h}, i={i}, j={j}: expected negative, got {bias[h, i, j].item()}"
                        )

    def test_r_is_learnable(self):
        """KERPLE r parameters are learnable."""
        module = KERPLEBias(n_heads=N_HEADS)
        bias = module(seq_len=T)
        loss = bias.sum()
        loss.backward()
        assert module.r_log.grad is not None, "No gradient on KERPLE r_log"

    def test_symmetry(self):
        """KERPLE bias is symmetric: bias[h, i, j] == bias[h, j, i]."""
        module = KERPLEBias(n_heads=N_HEADS)
        bias = module(seq_len=T)
        for h in range(N_HEADS):
            assert torch.allclose(bias[h], bias[h].T), f"Head {h} bias not symmetric"


# ---------------------------------------------------------------------------
# Edge cases: n_heads=1, T=1
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_alibi_n_heads_1_T_1(self):
        """ALiBi works with n_heads=1 and T=1."""
        module = ALiBiPositionBias(n_heads=1)
        bias = module(seq_len=1)
        assert bias.shape == (1, 1, 1)
        assert bias[0, 0, 0].item() == 0.0

    def test_t5_n_heads_1_T_1(self):
        """T5 relative bias works with n_heads=1 and T=1."""
        module = T5RelativePositionBias(n_heads=1, n_buckets=4, max_distance=16)
        bias = module(seq_len=1)
        assert bias.shape == (1, 1, 1)

    def test_learned_abs_T_1(self):
        """Learned absolute PE works with T=1."""
        module = LearnedAbsolutePositionEncoding(max_seq_len=16, d_model=D_MODEL)
        out = module(seq_len=1)
        assert out.shape == (1, 1, D_MODEL)

    def test_kerple_n_heads_1_T_1(self):
        """KERPLE works with n_heads=1 and T=1."""
        module = KERPLEBias(n_heads=1)
        bias = module(seq_len=1)
        assert bias.shape == (1, 1, 1)
        assert bias[0, 0, 0].item() == 0.0
