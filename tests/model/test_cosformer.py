"""Tests for src/model/cosformer.py — cosFormer attention.

Test inventory (14 tests):
  1.  test_kernel_doubles_last_dim       — CosformerKernel output last dim = 2*input last dim
  2.  test_kernel_outputs_nonneg         — all kernel outputs >= 0 (relu construction)
  3.  test_attention_output_shape        — CosformerAttention (B, T, d_head) non-causal
  4.  test_attention_noncausal_finite    — non-causal output contains no NaN/Inf
  5.  test_attention_causal_finite       — causal output contains no NaN/Inf
  6.  test_causal_prefix_consistency     — out[:t] unchanged when seq extended by padding
  7.  test_layer_output_shape            — CosformerLayer output (B, T, d_model)
  8.  test_layer_output_finite           — layer output contains no NaN/Inf
  9.  test_layer_gradients               — loss.backward() propagates gradients
 10.  test_batch1_seqlen1                — works with B=1, T=1
 11.  test_block_output_shape            — CosformerBlock output (B, T, d_model)
 12.  test_block_output_finite           — block output contains no NaN/Inf
 13.  test_block_gradients               — loss.backward() through block
 14.  test_different_inputs_differ       — distinct inputs produce distinct outputs
"""

import pytest
import torch

from aurelius.model.cosformer import (
    CosformerBlock,
    CosformerAttention,
    CosformerKernel,
    CosformerLayer,
)

# ---------------------------------------------------------------------------
# Fixtures / constants
# ---------------------------------------------------------------------------

B, T, D_HEAD = 2, 8, 16
D_MODEL, N_HEADS = 32, 4
D_FF = 64


def _rand(*shape):
    return torch.randn(*shape)


# ---------------------------------------------------------------------------
# CosformerKernel tests
# ---------------------------------------------------------------------------

def test_kernel_doubles_last_dim():
    """Output last dim should be exactly 2 * input last dim."""
    kernel = CosformerKernel()
    x = _rand(B, T, D_HEAD)
    out = kernel(x)
    assert out.shape == (B, T, 2 * D_HEAD), f"Expected ({B}, {T}, {2*D_HEAD}), got {out.shape}"


def test_kernel_outputs_nonneg():
    """All values produced by CosformerKernel must be >= 0 (relu construction)."""
    kernel = CosformerKernel()
    x = _rand(4, 10, 8)
    out = kernel(x)
    assert (out >= 0).all(), "CosformerKernel produced negative values"


# ---------------------------------------------------------------------------
# CosformerAttention tests
# ---------------------------------------------------------------------------

def test_attention_output_shape():
    """Non-causal CosformerAttention must return (B, T, d_head)."""
    attn = CosformerAttention(d_head=D_HEAD)
    q = _rand(B, T, D_HEAD)
    k = _rand(B, T, D_HEAD)
    v = _rand(B, T, D_HEAD)
    out = attn(q, k, v, causal=False)
    assert out.shape == (B, T, D_HEAD), f"Expected ({B}, {T}, {D_HEAD}), got {out.shape}"


def test_attention_noncausal_finite():
    """Non-causal output must contain no NaN or Inf."""
    attn = CosformerAttention(d_head=D_HEAD)
    q = _rand(B, T, D_HEAD)
    k = _rand(B, T, D_HEAD)
    v = _rand(B, T, D_HEAD)
    out = attn(q, k, v, causal=False)
    assert torch.isfinite(out).all(), "Non-causal output contains NaN or Inf"


def test_attention_causal_finite():
    """Causal output must contain no NaN or Inf."""
    attn = CosformerAttention(d_head=D_HEAD)
    q = _rand(B, T, D_HEAD)
    k = _rand(B, T, D_HEAD)
    v = _rand(B, T, D_HEAD)
    out = attn(q, k, v, causal=True)
    assert out.shape == (B, T, D_HEAD), f"Expected ({B}, {T}, {D_HEAD}), got {out.shape}"
    assert torch.isfinite(out).all(), "Causal output contains NaN or Inf"


def test_causal_prefix_consistency():
    """Causal attention: output for the first t steps must not change when
    the sequence is extended by appending zero-padding tokens."""
    torch.manual_seed(42)
    attn = CosformerAttention(d_head=D_HEAD)

    T_short = 5
    T_long = T_short + 4

    q_short = _rand(1, T_short, D_HEAD)
    k_short = _rand(1, T_short, D_HEAD)
    v_short = _rand(1, T_short, D_HEAD)

    # Use random (non-zero) extension — linear-attention causality means
    # changing k/v at position T_short should NOT affect outputs at 0..T_short-1
    torch.manual_seed(99)
    extension = torch.randn(1, T_long - T_short, D_HEAD)
    q_long = torch.cat([q_short, extension], dim=1)
    k_long_v1 = torch.cat([k_short, extension], dim=1)
    k_long_v2 = torch.cat([k_short, extension * 2], dim=1)  # different future K
    v_long_v1 = torch.cat([v_short, extension], dim=1)
    v_long_v2 = torch.cat([v_short, -extension], dim=1)    # different future V

    with torch.no_grad():
        out_v1 = attn(q_long, k_long_v1, v_long_v1, causal=True)
        out_v2 = attn(q_long, k_long_v2, v_long_v2, causal=True)

    # Past outputs (positions 0..T_short-1) must be identical regardless of future
    assert torch.allclose(
        out_v1[:, :T_short, :], out_v2[:, :T_short, :], atol=1e-4
    ), "Causal prefix changed when future tokens were modified"


# ---------------------------------------------------------------------------
# CosformerLayer tests
# ---------------------------------------------------------------------------

def test_layer_output_shape():
    """CosformerLayer must return (B, T, d_model)."""
    layer = CosformerLayer(D_MODEL, N_HEADS)
    x = _rand(B, T, D_MODEL)
    out = layer(x)
    assert out.shape == (B, T, D_MODEL), f"Expected ({B}, {T}, {D_MODEL}), got {out.shape}"


def test_layer_output_finite():
    """CosformerLayer output must contain no NaN or Inf."""
    layer = CosformerLayer(D_MODEL, N_HEADS)
    x = _rand(B, T, D_MODEL)
    out = layer(x)
    assert torch.isfinite(out).all(), "CosformerLayer output contains NaN or Inf"


def test_layer_gradients():
    """Gradients must flow through CosformerLayer back to the input."""
    layer = CosformerLayer(D_MODEL, N_HEADS)
    x = _rand(B, T, D_MODEL).requires_grad_(True)
    out = layer(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "No gradient reached input tensor"
    assert torch.isfinite(x.grad).all(), "Input gradient contains NaN or Inf"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_batch1_seqlen1():
    """CosformerLayer must handle B=1, T=1 without error."""
    layer = CosformerLayer(D_MODEL, N_HEADS)
    x = _rand(1, 1, D_MODEL)
    out = layer(x)
    assert out.shape == (1, 1, D_MODEL)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# CosformerBlock tests
# ---------------------------------------------------------------------------

def test_block_output_shape():
    """CosformerBlock must return (B, T, d_model)."""
    block = CosformerBlock(D_MODEL, N_HEADS, D_FF)
    x = _rand(B, T, D_MODEL)
    out = block(x)
    assert out.shape == (B, T, D_MODEL), f"Expected ({B}, {T}, {D_MODEL}), got {out.shape}"


def test_block_output_finite():
    """CosformerBlock output must contain no NaN or Inf."""
    block = CosformerBlock(D_MODEL, N_HEADS, D_FF)
    x = _rand(B, T, D_MODEL)
    out = block(x)
    assert torch.isfinite(out).all(), "CosformerBlock output contains NaN or Inf"


def test_block_gradients():
    """Gradients must flow through CosformerBlock."""
    block = CosformerBlock(D_MODEL, N_HEADS, D_FF)
    x = _rand(B, T, D_MODEL).requires_grad_(True)
    out = block(x)
    out.sum().backward()
    assert x.grad is not None, "No gradient reached input tensor"
    assert torch.isfinite(x.grad).all(), "Input gradient contains NaN or Inf"


def test_different_inputs_differ():
    """Two distinct inputs must produce distinct outputs."""
    block = CosformerBlock(D_MODEL, N_HEADS, D_FF)
    torch.manual_seed(0)
    x1 = _rand(B, T, D_MODEL)
    x2 = _rand(B, T, D_MODEL)
    with torch.no_grad():
        out1 = block(x1)
        out2 = block(x2)
    assert not torch.allclose(out1, out2, atol=1e-6), "Different inputs produced identical outputs"
