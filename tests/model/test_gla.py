"""Tests for GLA — Gated Linear Attention (Yang et al. 2024)."""

import pytest
import torch

from src.model.gla import GatedLinearAttention, GLABlock
from src.model.config import AureliusConfig


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def make_config():
    """Tiny AureliusConfig for block-level tests."""
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


def make_gla():
    """Small standalone GLA for unit tests."""
    return GatedLinearAttention(d_model=64, n_heads=4, head_dim=16)


# ---------------------------------------------------------------------------
# GLA forward shape
# ---------------------------------------------------------------------------


def test_gla_forward_shape():
    """GLA.forward: (2, 16, 64) -> (2, 16, 64)."""
    gla = make_gla()
    x = torch.randn(2, 16, 64)
    out = gla(x)
    assert out.shape == (2, 16, 64), f"Expected (2, 16, 64), got {out.shape}"


# ---------------------------------------------------------------------------
# Recurrent mode
# ---------------------------------------------------------------------------


def test_gla_recurrent_output_shape():
    """forward_recurrent output shape is (B, L, d_model)."""
    gla = make_gla()
    x = torch.randn(2, 16, 64)
    out, _ = gla.forward_recurrent(x)
    assert out.shape == (2, 16, 64), f"Expected (2, 16, 64), got {out.shape}"


def test_gla_state_shape():
    """final_state from forward_recurrent has shape (B, n_heads, head_dim, head_dim)."""
    n_heads, head_dim = 4, 16
    gla = GatedLinearAttention(d_model=64, n_heads=n_heads, head_dim=head_dim)
    x = torch.randn(2, 8, 64)
    _, state = gla.forward_recurrent(x)
    expected = (2, n_heads, head_dim, head_dim)
    assert state.shape == expected, f"Expected {expected}, got {state.shape}"


# ---------------------------------------------------------------------------
# Gate value range
# ---------------------------------------------------------------------------


def test_gate_in_zero_one():
    """Gate values from sigmoid projection must lie strictly in (0, 1)."""
    gla = make_gla()
    x = torch.randn(2, 8, 64)

    # Extract raw gate logits and apply sigmoid (mirrors forward_recurrent internals)
    with torch.no_grad():
        B, L, _ = x.shape
        H, D = gla.n_heads, gla.head_dim
        gate = torch.sigmoid(gla.gate_proj(x).view(B, L, H, D))

    assert (gate > 0).all().item(), "Gate has values <= 0 (sigmoid should ensure > 0)"
    assert (gate < 1).all().item(), "Gate has values >= 1 (sigmoid should ensure < 1)"


# ---------------------------------------------------------------------------
# GLABlock
# ---------------------------------------------------------------------------


def test_gla_block_shape():
    """GLABlock: (2, 8, 64) -> (2, 8, 64)."""
    config = make_config()
    block = GLABlock(config)
    x = torch.randn(2, 8, 64)
    out = block(x)
    assert out.shape == (2, 8, 64), f"Expected (2, 8, 64), got {out.shape}"


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gla_gradient_flow():
    """Backward pass through GLA must not error and must produce gradients."""
    gla = make_gla()
    x = torch.randn(2, 8, 64, requires_grad=True)
    out = gla(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "No gradient flowed back to input"
    assert not torch.isnan(x.grad).any(), "NaN in gradients"
