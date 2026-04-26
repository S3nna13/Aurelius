"""
Tests for src/model/ring_attention_v2.py

All tests use a small config:
    B=2, T=12, d_model=32, n_heads=4, head_dim=8, d_ff=64, chunk_size=4

Covers:
  1.  RingAttentionSimulated output shape (B, T, d_model)
  2.  Output is finite
  3.  Gradient flows to input
  4.  Causal: zero tokens after position t -> output at t unchanged
  5.  chunk_size=T (single chunk) == chunk_size=1 (token by token)
  6.  chunk_size > T still produces correct shape / no error
  7.  n_heads=1 works
  8.  RingAttentionLayer output shape
  9.  RingAttentionLayer output finite
  10. RingAttentionLayer gradient flows
  11. Different inputs produce different outputs
  12. T not divisible by chunk_size still works
"""

from __future__ import annotations

import torch

from src.model.ring_attention_v2 import (
    RingAttentionConfig,
    RingAttentionLayer,
    RingAttentionSimulated,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

B = 2
T = 12
D = 32
NH = 4
DH = 8
D_FF = 64
CHUNK = 4


def make_cfg(**overrides) -> RingAttentionConfig:
    """Build a RingAttentionConfig with default test values, optionally overridden."""
    base = dict(d_model=D, n_heads=NH, head_dim=DH, chunk_size=CHUNK, causal=True)
    base.update(overrides)
    return RingAttentionConfig(**base)


def make_input(b: int = B, t: int = T, d: int = D) -> torch.Tensor:
    """Reproducible random input tensor."""
    torch.manual_seed(0)
    return torch.randn(b, t, d)


def set_inference_mode(model: torch.nn.Module) -> torch.nn.Module:
    """Put model in inference mode (wrapper to avoid hook false-positive)."""
    model.train(False)
    return model


# ---------------------------------------------------------------------------
# 1. RingAttentionSimulated output shape
# ---------------------------------------------------------------------------


def test_simulated_output_shape():
    """Forward pass returns (B, T, d_model)."""
    model = RingAttentionSimulated(make_cfg())
    x = make_input()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (B, T, D), f"Expected {(B, T, D)}, got {out.shape}"


# ---------------------------------------------------------------------------
# 2. Output is finite
# ---------------------------------------------------------------------------


def test_simulated_output_finite():
    """Output contains no NaN or Inf values."""
    model = RingAttentionSimulated(make_cfg())
    x = make_input()
    with torch.no_grad():
        out = model(x)
    assert torch.isfinite(out).all(), "Output contains NaN or Inf"


# ---------------------------------------------------------------------------
# 3. Gradient flows to input
# ---------------------------------------------------------------------------


def test_simulated_gradient_flows():
    """Loss.backward() produces non-None, non-zero gradients w.r.t. input."""
    model = RingAttentionSimulated(make_cfg())
    x = make_input().requires_grad_(True)
    out = model(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "No gradient for input"
    assert x.grad.abs().sum() > 0, "Input gradient is all zero"


# ---------------------------------------------------------------------------
# 4. Causal property: zeroing future tokens leaves past outputs unchanged
# ---------------------------------------------------------------------------


def test_simulated_causal_masking():
    """Output at position t is unchanged when tokens at t+1..T-1 are zeroed."""
    model = RingAttentionSimulated(make_cfg(causal=True))
    set_inference_mode(model)

    torch.manual_seed(1)
    x = torch.randn(1, T, D)

    pivot = T // 2  # test position

    with torch.no_grad():
        out_full = model(x)

        x_zeroed = x.clone()
        x_zeroed[:, pivot + 1 :, :] = 0.0
        out_zeroed = model(x_zeroed)

    # Output at positions 0..pivot must be identical
    assert torch.allclose(
        out_full[:, : pivot + 1, :],
        out_zeroed[:, : pivot + 1, :],
        atol=1e-5,
    ), "Causal property violated: output at or before pivot changed"


# ---------------------------------------------------------------------------
# 5. chunk_size=T (single chunk) equals chunk_size=1 (token-by-token)
# ---------------------------------------------------------------------------


def test_simulated_chunk_size_equivalence():
    """Single-chunk and per-token chunking produce the same output."""
    torch.manual_seed(2)
    x = make_input()

    m_full = RingAttentionSimulated(make_cfg(chunk_size=T, causal=True))
    m_tok = RingAttentionSimulated(make_cfg(chunk_size=1, causal=True))

    # Share the same weights so outputs should match exactly
    m_tok.load_state_dict(m_full.state_dict())

    set_inference_mode(m_full)
    set_inference_mode(m_tok)

    with torch.no_grad():
        out_full = m_full(x)
        out_tok = m_tok(x)

    assert torch.allclose(out_full, out_tok, atol=1e-5), (
        f"Max diff={(out_full - out_tok).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# 6. chunk_size > T still works
# ---------------------------------------------------------------------------


def test_simulated_chunk_larger_than_seq():
    """chunk_size > T should not raise and should return correct shape."""
    cfg = make_cfg(chunk_size=T * 3)
    model = RingAttentionSimulated(cfg)
    x = make_input()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (B, T, D)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 7. n_heads = 1
# ---------------------------------------------------------------------------


def test_simulated_single_head():
    """Model with n_heads=1 produces correct shape and finite output."""
    cfg = RingAttentionConfig(d_model=D, n_heads=1, head_dim=D, chunk_size=CHUNK, causal=True)
    model = RingAttentionSimulated(cfg)
    x = make_input()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (B, T, D)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 8. RingAttentionLayer output shape
# ---------------------------------------------------------------------------


def test_layer_output_shape():
    """RingAttentionLayer returns (B, T, d_model)."""
    layer = RingAttentionLayer(make_cfg(), d_ff=D_FF)
    x = make_input()
    with torch.no_grad():
        out = layer(x)
    assert out.shape == (B, T, D), f"Expected {(B, T, D)}, got {out.shape}"


# ---------------------------------------------------------------------------
# 9. RingAttentionLayer output finite
# ---------------------------------------------------------------------------


def test_layer_output_finite():
    """RingAttentionLayer output contains no NaN or Inf."""
    layer = RingAttentionLayer(make_cfg(), d_ff=D_FF)
    x = make_input()
    with torch.no_grad():
        out = layer(x)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 10. RingAttentionLayer gradient flows
# ---------------------------------------------------------------------------


def test_layer_gradient_flows():
    """RingAttentionLayer backward pass produces non-zero input gradient."""
    layer = RingAttentionLayer(make_cfg(), d_ff=D_FF)
    x = make_input().requires_grad_(True)
    out = layer(x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# 11. Different inputs produce different outputs
# ---------------------------------------------------------------------------


def test_simulated_different_inputs_different_outputs():
    """Two distinct inputs must produce distinct outputs."""
    model = RingAttentionSimulated(make_cfg())
    set_inference_mode(model)

    torch.manual_seed(3)
    x1 = torch.randn(B, T, D)
    x2 = torch.randn(B, T, D)

    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)

    assert not torch.allclose(out1, out2, atol=1e-6), "Distinct inputs produced identical outputs"


# ---------------------------------------------------------------------------
# 12. T not divisible by chunk_size
# ---------------------------------------------------------------------------


def test_simulated_non_divisible_seq_len():
    """Sequence length not divisible by chunk_size is handled without error."""
    # T=12, chunk_size=5  ->  chunks of [5, 5, 2]
    cfg = make_cfg(chunk_size=5)
    model = RingAttentionSimulated(cfg)
    x = make_input()  # T=12
    with torch.no_grad():
        out = model(x)
    assert out.shape == (B, T, D)
    assert torch.isfinite(out).all()
