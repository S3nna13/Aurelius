"""Tests for src/model/positional_encodings.py."""

import torch
import pytest

from src.model.positional_encodings import (
    get_alibi_slopes,
    compute_alibi_bias,
    ALiBiAttention,
    sinusoidal_encoding,
    LearnedPositionalEncoding,
    t5_relative_position_bias,
    SandwichPositionalEncoding,
)

# Common test dimensions
D_MODEL = 32
N_HEADS = 4
B = 2
T = 8


# ---------------------------------------------------------------------------
# get_alibi_slopes
# ---------------------------------------------------------------------------

def test_get_alibi_slopes_shape():
    """Slopes tensor must have shape (n_heads,)."""
    slopes = get_alibi_slopes(N_HEADS)
    assert slopes.shape == (N_HEADS,), f"Expected ({N_HEADS},), got {slopes.shape}"


def test_get_alibi_slopes_values_positive_less_than_one():
    """All slope values must be in (0, 1)."""
    slopes = get_alibi_slopes(N_HEADS)
    assert (slopes > 0).all(), "All slopes must be positive"
    assert (slopes < 1).all(), "All slopes must be < 1"


def test_get_alibi_slopes_monotonically_decreasing():
    """Slopes should decrease monotonically (larger head index -> smaller slope)."""
    slopes = get_alibi_slopes(N_HEADS)
    for i in range(len(slopes) - 1):
        assert slopes[i] > slopes[i + 1], (
            f"slopes[{i}]={slopes[i].item():.6f} should be > slopes[{i+1}]={slopes[i+1].item():.6f}"
        )


def test_get_alibi_slopes_non_power_of_2():
    """Non-power-of-2 head counts should still return a valid (n_heads,) tensor."""
    for n in [3, 5, 6, 7, 12]:
        slopes = get_alibi_slopes(n)
        assert slopes.shape == (n,), f"Expected ({n},), got {slopes.shape}"
        assert (slopes > 0).all(), f"All slopes must be positive for n_heads={n}"
        assert (slopes < 1).all(), f"All slopes must be < 1 for n_heads={n}"


# ---------------------------------------------------------------------------
# compute_alibi_bias
# ---------------------------------------------------------------------------

def test_compute_alibi_bias_shape():
    """Bias tensor must have shape (n_heads, T, T)."""
    slopes = get_alibi_slopes(N_HEADS)
    bias = compute_alibi_bias(T, slopes)
    assert bias.shape == (N_HEADS, T, T), f"Expected ({N_HEADS}, {T}, {T}), got {bias.shape}"


def test_compute_alibi_bias_causal_upper_triangle_zero():
    """Upper triangle (j > i, future tokens) must be 0."""
    slopes = get_alibi_slopes(N_HEADS)
    bias = compute_alibi_bias(T, slopes)
    # Upper triangle excluding diagonal: rows i, cols j > i
    for h in range(N_HEADS):
        for i in range(T):
            for j in range(i + 1, T):
                assert bias[h, i, j].item() == 0.0, (
                    f"Expected 0 at head={h}, i={i}, j={j}, got {bias[h, i, j].item()}"
                )


def test_compute_alibi_bias_diagonal_is_zero():
    """Self-attention positions (i == j, distance=0) must have 0 bias."""
    slopes = get_alibi_slopes(N_HEADS)
    bias = compute_alibi_bias(T, slopes)
    for h in range(N_HEADS):
        for i in range(T):
            assert bias[h, i, i].item() == 0.0, (
                f"Expected 0 on diagonal at head={h}, i={i}, got {bias[h, i, i].item()}"
            )


# ---------------------------------------------------------------------------
# ALiBiAttention
# ---------------------------------------------------------------------------

def test_alibi_attention_forward_output_shape():
    """ALiBiAttention forward must return (B, T, d_model)."""
    model = ALiBiAttention(d_model=D_MODEL, n_heads=N_HEADS)
    x = torch.randn(B, T, D_MODEL)
    out = model(x)
    assert out.shape == (B, T, D_MODEL), f"Expected ({B}, {T}, {D_MODEL}), got {out.shape}"


# ---------------------------------------------------------------------------
# sinusoidal_encoding
# ---------------------------------------------------------------------------

def test_sinusoidal_encoding_output_shape():
    """sinusoidal_encoding must return (seq_len, d_model)."""
    pe = sinusoidal_encoding(T, D_MODEL)
    assert pe.shape == (T, D_MODEL), f"Expected ({T}, {D_MODEL}), got {pe.shape}"


def test_sinusoidal_encoding_values_in_range():
    """All sinusoidal encoding values must be in [-1, 1]."""
    pe = sinusoidal_encoding(T, D_MODEL)
    assert pe.min().item() >= -1.0 - 1e-6, f"Min value {pe.min().item()} < -1"
    assert pe.max().item() <= 1.0 + 1e-6, f"Max value {pe.max().item()} > 1"


# ---------------------------------------------------------------------------
# LearnedPositionalEncoding
# ---------------------------------------------------------------------------

def test_learned_positional_encoding_output_shape():
    """LearnedPositionalEncoding forward must return (seq_len, d_model)."""
    lpe = LearnedPositionalEncoding(max_seq_len=128, d_model=D_MODEL)
    out = lpe(T)
    assert out.shape == (T, D_MODEL), f"Expected ({T}, {D_MODEL}), got {out.shape}"


# ---------------------------------------------------------------------------
# t5_relative_position_bias
# ---------------------------------------------------------------------------

def test_t5_relative_position_bias_shape():
    """t5_relative_position_bias must return (n_heads, T, T)."""
    bias = t5_relative_position_bias(N_HEADS, T)
    assert bias.shape == (N_HEADS, T, T), f"Expected ({N_HEADS}, {T}, {T}), got {bias.shape}"


# ---------------------------------------------------------------------------
# SandwichPositionalEncoding
# ---------------------------------------------------------------------------

def test_sandwich_positional_encoding_output_shape():
    """SandwichPositionalEncoding must return (seq_len, d_model)."""
    sandwich = SandwichPositionalEncoding(max_seq_len=128, d_model=D_MODEL)
    out = sandwich(T)
    assert out.shape == (T, D_MODEL), f"Expected ({T}, {D_MODEL}), got {out.shape}"


def test_sandwich_positional_encoding_different_seq_lens():
    """SandwichPositionalEncoding should work for varying sequence lengths."""
    sandwich = SandwichPositionalEncoding(max_seq_len=256, d_model=D_MODEL)
    for seq_len in [1, 4, 8, 16, 32]:
        out = sandwich(seq_len)
        assert out.shape == (seq_len, D_MODEL), (
            f"Expected ({seq_len}, {D_MODEL}), got {out.shape} for seq_len={seq_len}"
        )
