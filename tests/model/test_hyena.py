"""Tests for the Hyena operator (Poli et al. 2023)."""

import torch

from src.model.config import AureliusConfig
from src.model.hyena import HyenaBlock, HyenaFilter, HyenaOperator

# ---------------------------------------------------------------------------
# Shared tiny config for HyenaBlock tests
# ---------------------------------------------------------------------------


def make_config():
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )


# ---------------------------------------------------------------------------
# HyenaFilter tests
# ---------------------------------------------------------------------------


def test_hyena_filter_output_shape():
    """HyenaFilter(64, order=2, kernel_len=32).forward(16) → (2, 64, 16)."""
    filt = HyenaFilter(d_model=64, order=2, kernel_len=32)
    out = filt(16)
    assert out.shape == (2, 64, 16), f"Expected (2, 64, 16), got {out.shape}"


def test_hyena_filter_different_lengths():
    """filter(8) and filter(16) both return the correct shapes."""
    filt = HyenaFilter(d_model=64, order=2, kernel_len=32)
    out8 = filt(8)
    out16 = filt(16)
    assert out8.shape == (2, 64, 8)
    assert out16.shape == (2, 64, 16)


# ---------------------------------------------------------------------------
# fft_conv tests
# ---------------------------------------------------------------------------


def test_fft_conv_output_shape():
    """fft_conv on (2, 64, 32) input → (2, 64, 32) output."""
    op = HyenaOperator(d_model=64, order=2, kernel_len=64)
    u = torch.randn(2, 64, 32)
    k = torch.randn(64, 32)
    out = op.fft_conv(u, k)
    assert out.shape == (2, 64, 32), f"Expected (2, 64, 32), got {out.shape}"


def test_fft_conv_causality():
    """Output at position t must not depend on input at positions > t.

    Strategy: run fft_conv on a full input u, then zero out positions [t+1:]
    and verify the output at position t is unchanged.
    """
    torch.manual_seed(0)
    op = HyenaOperator(d_model=4, order=2, kernel_len=32)

    L = 16
    t = 7  # check causality at this position

    u = torch.randn(1, 4, L)
    k = torch.randn(4, L)

    out_full = op.fft_conv(u, k)

    # Zero out future positions (t+1 onward)
    u_causal = u.clone()
    u_causal[..., t + 1 :] = 0.0
    out_causal = op.fft_conv(u_causal, k)

    # Output at position t should be identical
    torch.testing.assert_close(
        out_full[..., t],
        out_causal[..., t],
        msg="fft_conv output at t changed when future inputs were zeroed (not causal)",
    )


# ---------------------------------------------------------------------------
# HyenaOperator tests
# ---------------------------------------------------------------------------


def test_hyena_operator_output_shape():
    """HyenaOperator: (2, 32, 64) → (2, 32, 64)."""
    op = HyenaOperator(d_model=64, order=2, kernel_len=64)
    x = torch.randn(2, 32, 64)
    out = op(x)
    assert out.shape == (2, 32, 64), f"Expected (2, 32, 64), got {out.shape}"


def test_hyena_operator_gradient_flow():
    """Backward pass through HyenaOperator must not raise."""
    op = HyenaOperator(d_model=64, order=2, kernel_len=64)
    x = torch.randn(2, 32, 64, requires_grad=True)
    out = op(x)
    loss = out.sum()
    loss.backward()  # must not raise
    assert x.grad is not None, "No gradient flowed back to input"


# ---------------------------------------------------------------------------
# HyenaBlock tests
# ---------------------------------------------------------------------------


def test_hyena_block_output_shape():
    """HyenaBlock: (2, 16, 64) → (2, 16, 64)."""
    config = make_config()
    block = HyenaBlock(config)
    x = torch.randn(2, 16, 64)
    out = block(x)
    assert out.shape == (2, 16, 64), f"Expected (2, 16, 64), got {out.shape}"


def test_hyena_block_residual():
    """Output of HyenaBlock must differ from input (something was computed)."""
    config = make_config()
    block = HyenaBlock(config)
    x = torch.randn(2, 16, 64)
    out = block(x)
    assert not torch.allclose(out, x), (
        "HyenaBlock output is identical to input — residual did nothing"
    )


# ---------------------------------------------------------------------------
# Sub-quadratic parameter count test
# ---------------------------------------------------------------------------


def test_hyena_subquadratic_params():
    """HyenaOperator param count should be O(d_model * d_filter), not O(L²).

    Specifically: doubling L (kernel_len) should not significantly increase
    the trainable parameter count (the MLP is fixed-size; only the positional
    buffer, which is not a parameter, grows with L).
    """
    d_model = 64
    d_filter = 64

    op_short = HyenaOperator(d_model=d_model, order=2, kernel_len=128, d_filter=d_filter)
    op_long = HyenaOperator(d_model=d_model, order=2, kernel_len=4096, d_filter=d_filter)

    params_short = sum(p.numel() for p in op_short.parameters())
    params_long = sum(p.numel() for p in op_long.parameters())

    # Trainable parameters must be equal (kernel length only affects the buffer)
    assert params_short == params_long, (
        f"Parameter count changed with kernel_len: {params_short} vs {params_long}. "
        "HyenaOperator is not sub-quadratic in L."
    )
