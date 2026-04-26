"""
Tests for src/model/sliding_window_attention.py

Uses tiny configs (small D, T, W) to keep execution fast.
All tests are pure PyTorch — no HuggingFace, no scipy, no sklearn.
"""

import pytest
import torch

from src.model.sliding_window_attention import (
    SlidingWindowAttention,
    SWABlock,
    SWAConfig,
    build_sliding_window_mask,
    count_attended_positions,
    sliding_window_attention,
)

# ---------------------------------------------------------------------------
# Tiny test parameters
# ---------------------------------------------------------------------------
B, T, D = 2, 12, 16  # batch, seq_len, d_model
H = 2  # heads
W = 4  # window_size
d_head = D // H  # 8


# ---------------------------------------------------------------------------
# 1. SWAConfig defaults
# ---------------------------------------------------------------------------


class TestSWAConfig:
    def test_defaults(self):
        cfg = SWAConfig()
        assert cfg.d_model == 512
        assert cfg.n_heads == 8
        assert cfg.window_size == 256
        assert cfg.causal is True
        assert cfg.dropout == 0.0

    def test_custom_values(self):
        cfg = SWAConfig(d_model=D, n_heads=H, window_size=W, causal=False, dropout=0.1)
        assert cfg.d_model == D
        assert cfg.n_heads == H
        assert cfg.window_size == W
        assert cfg.causal is False
        assert cfg.dropout == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# 2. build_sliding_window_mask — shape
# ---------------------------------------------------------------------------


class TestBuildMaskShape:
    def test_shape_causal(self):
        mask = build_sliding_window_mask(T, W, causal=True)
        assert mask.shape == (T, T), f"Expected ({T},{T}), got {mask.shape}"

    def test_shape_noncausal(self):
        mask = build_sliding_window_mask(T, W, causal=False)
        assert mask.shape == (T, T)

    def test_shape_seq_len_1(self):
        mask = build_sliding_window_mask(1, W, causal=True)
        assert mask.shape == (1, 1)


# ---------------------------------------------------------------------------
# 3. build_sliding_window_mask — causal: no future attending
# ---------------------------------------------------------------------------


class TestBuildMaskCausal:
    def setup_method(self):
        self.mask = build_sliding_window_mask(T, W, causal=True)

    def test_no_future_in_causal(self):
        """Upper triangle (j > i) must all be -inf."""
        for i in range(T):
            for j in range(i + 1, T):
                assert self.mask[i, j] == float("-inf"), (
                    f"mask[{i},{j}] should be -inf (future), got {self.mask[i, j]}"
                )

    def test_diagonal_is_zero_causal(self):
        """Each token can attend to itself."""
        for i in range(T):
            assert self.mask[i, i] == 0.0, f"mask[{i},{i}] should be 0.0 (self-attend)"

    def test_within_window_is_zero_causal(self):
        """Positions within the window (and not future) must be 0.0."""
        half = W // 2
        for i in range(T):
            for j in range(max(0, i - half), i + 1):
                assert self.mask[i, j] == 0.0, (
                    f"mask[{i},{j}] should be 0.0 (in window), got {self.mask[i, j]}"
                )

    def test_outside_window_is_neg_inf_causal(self):
        """Positions outside the window (past) must be -inf."""
        half = W // 2
        for i in range(T):
            for j in range(0, max(0, i - half)):
                assert self.mask[i, j] == float("-inf"), (
                    f"mask[{i},{j}] should be -inf (outside window)"
                )


# ---------------------------------------------------------------------------
# 4. build_sliding_window_mask — non-causal: symmetric
# ---------------------------------------------------------------------------


class TestBuildMaskNonCausal:
    def setup_method(self):
        self.mask = build_sliding_window_mask(T, W, causal=False)

    def test_symmetric(self):
        diff = (self.mask - self.mask.T).abs()
        # Replace inf-inf = nan with 0 for comparison
        diff = torch.nan_to_num(diff, nan=0.0)
        assert diff.max().item() == 0.0, "Non-causal mask must be symmetric"

    def test_diagonal_is_zero_noncausal(self):
        for i in range(T):
            assert self.mask[i, i] == 0.0


# ---------------------------------------------------------------------------
# 5. build_sliding_window_mask — values are only 0.0 or -inf
# ---------------------------------------------------------------------------


class TestBuildMaskValues:
    def test_only_zero_or_neg_inf_causal(self):
        mask = build_sliding_window_mask(T, W, causal=True)
        finite_vals = mask[torch.isfinite(mask)]
        assert (finite_vals == 0.0).all(), "Finite values must all be 0.0"
        inf_vals = mask[~torch.isfinite(mask)]
        assert (inf_vals == float("-inf")).all(), "Non-finite values must be -inf"

    def test_only_zero_or_neg_inf_noncausal(self):
        mask = build_sliding_window_mask(T, W, causal=False)
        finite_vals = mask[torch.isfinite(mask)]
        assert (finite_vals == 0.0).all()
        inf_vals = mask[~torch.isfinite(mask)]
        assert (inf_vals == float("-inf")).all()


# ---------------------------------------------------------------------------
# 6. sliding_window_attention — output shape (B, T, d_head)
# ---------------------------------------------------------------------------


class TestSlidingWindowAttentionShape:
    def _make_qkv(self):
        q = torch.randn(B, T, d_head)
        k = torch.randn(B, T, d_head)
        v = torch.randn(B, T, d_head)
        return q, k, v

    def test_output_shape_causal(self):
        q, k, v = self._make_qkv()
        out = sliding_window_attention(q, k, v, window_size=W, causal=True)
        assert out.shape == (B, T, d_head), f"Expected {(B, T, d_head)}, got {out.shape}"

    def test_output_shape_noncausal(self):
        q, k, v = self._make_qkv()
        out = sliding_window_attention(q, k, v, window_size=W, causal=False)
        assert out.shape == (B, T, d_head)

    def test_output_shape_window_larger_than_T(self):
        """Window larger than sequence length should behave like full attention."""
        q, k, v = self._make_qkv()
        out = sliding_window_attention(q, k, v, window_size=T * 4, causal=True)
        assert out.shape == (B, T, d_head)


# ---------------------------------------------------------------------------
# 7. sliding_window_attention — output finite
# ---------------------------------------------------------------------------


class TestSlidingWindowAttentionFinite:
    def test_output_finite_causal(self):
        q = torch.randn(B, T, d_head)
        k = torch.randn(B, T, d_head)
        v = torch.randn(B, T, d_head)
        out = sliding_window_attention(q, k, v, window_size=W, causal=True)
        assert torch.isfinite(out).all(), "Output must be finite (no NaN/inf)"

    def test_output_finite_noncausal(self):
        q = torch.randn(B, T, d_head)
        k = torch.randn(B, T, d_head)
        v = torch.randn(B, T, d_head)
        out = sliding_window_attention(q, k, v, window_size=W, causal=False)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 8. SlidingWindowAttention — forward shape (B, T, d_model)
# ---------------------------------------------------------------------------


class TestSlidingWindowAttentionModuleShape:
    def test_output_shape(self):
        cfg = SWAConfig(d_model=D, n_heads=H, window_size=W)
        attn = SlidingWindowAttention(cfg)
        x = torch.randn(B, T, D)
        out = attn(x)
        assert out.shape == (B, T, D), f"Expected {(B, T, D)}, got {out.shape}"

    def test_output_shape_noncausal(self):
        cfg = SWAConfig(d_model=D, n_heads=H, window_size=W, causal=False)
        attn = SlidingWindowAttention(cfg)
        x = torch.randn(B, T, D)
        out = attn(x)
        assert out.shape == (B, T, D)


# ---------------------------------------------------------------------------
# 9. SlidingWindowAttention — output finite
# ---------------------------------------------------------------------------


class TestSlidingWindowAttentionModuleFinite:
    def test_output_finite(self):
        cfg = SWAConfig(d_model=D, n_heads=H, window_size=W)
        attn = SlidingWindowAttention(cfg)
        x = torch.randn(B, T, D)
        out = attn(x)
        assert torch.isfinite(out).all(), "SlidingWindowAttention output must be finite"

    def test_gradients_flow(self):
        cfg = SWAConfig(d_model=D, n_heads=H, window_size=W)
        attn = SlidingWindowAttention(cfg)
        x = torch.randn(B, T, D, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# 10. SWABlock — forward shape (B, T, d_model)
# ---------------------------------------------------------------------------


class TestSWABlockShape:
    def test_output_shape(self):
        cfg = SWAConfig(d_model=D, n_heads=H, window_size=W)
        block = SWABlock(cfg)
        x = torch.randn(B, T, D)
        out = block(x)
        assert out.shape == (B, T, D), f"Expected {(B, T, D)}, got {out.shape}"

    def test_output_shape_noncausal(self):
        cfg = SWAConfig(d_model=D, n_heads=H, window_size=W, causal=False)
        block = SWABlock(cfg)
        x = torch.randn(B, T, D)
        out = block(x)
        assert out.shape == (B, T, D)


# ---------------------------------------------------------------------------
# 11. SWABlock — output finite
# ---------------------------------------------------------------------------


class TestSWABlockFinite:
    def test_output_finite(self):
        cfg = SWAConfig(d_model=D, n_heads=H, window_size=W)
        block = SWABlock(cfg)
        x = torch.randn(B, T, D)
        out = block(x)
        assert torch.isfinite(out).all(), "SWABlock output must be finite"

    def test_residual_connection_preserves_shape(self):
        """Residual path: output must be same shape as input."""
        cfg = SWAConfig(d_model=D, n_heads=H, window_size=W)
        block = SWABlock(cfg)
        x = torch.zeros(B, T, D)
        out = block(x)
        assert out.shape == x.shape

    def test_gradients_flow_through_block(self):
        cfg = SWAConfig(d_model=D, n_heads=H, window_size=W)
        block = SWABlock(cfg)
        x = torch.randn(B, T, D, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# 12. count_attended_positions — causal < T*T
# ---------------------------------------------------------------------------


class TestCountAttendedPositions:
    def test_causal_less_than_full(self):
        total = count_attended_positions(T, W, causal=True)
        assert total < T * T, f"Causal attended count {total} should be less than T^2={T * T}"

    def test_causal_positive(self):
        total = count_attended_positions(T, W, causal=True)
        assert total > 0

    def test_causal_at_most_window_per_token(self):
        """Each token attends to at most window_size//2 + 1 positions when causal."""
        half = W // 2
        max_per_token = half + 1
        total = count_attended_positions(T, W, causal=True)
        assert total <= T * max_per_token

    def test_noncausal_symmetric_total(self):
        """Non-causal count should be >= causal count (more positions visible)."""
        causal_total = count_attended_positions(T, W, causal=True)
        noncausal_total = count_attended_positions(T, W, causal=False)
        assert noncausal_total >= causal_total, (
            f"Non-causal ({noncausal_total}) should be >= causal ({causal_total})"
        )

    def test_noncausal_less_than_full(self):
        total = count_attended_positions(T, W, causal=False)
        assert total < T * T or W >= T, (
            "With W < T, non-causal attended count should be less than T^2"
        )

    def test_count_matches_mask_causal(self):
        """count_attended_positions should match the number of 0.0 entries in mask."""
        mask = build_sliding_window_mask(T, W, causal=True)
        mask_count = (mask == 0.0).sum().item()
        func_count = count_attended_positions(T, W, causal=True)
        assert func_count == mask_count, (
            f"count_attended_positions={func_count} but mask has {mask_count} zeros"
        )

    def test_count_matches_mask_noncausal(self):
        """count_attended_positions should match the number of 0.0 entries in mask."""
        mask = build_sliding_window_mask(T, W, causal=False)
        mask_count = (mask == 0.0).sum().item()
        func_count = count_attended_positions(T, W, causal=False)
        assert func_count == mask_count, (
            f"count_attended_positions={func_count} but mask has {mask_count} zeros"
        )

    def test_window_equals_T_causal_is_lower_triangular(self):
        """When window covers entire sequence (W >= 2T), causal becomes lower-triangular."""
        big_W = T * 3
        total = count_attended_positions(T, big_W, causal=True)
        expected = T * (T + 1) // 2  # full lower triangle
        assert total == expected, f"With big window causal, expected {expected} but got {total}"
