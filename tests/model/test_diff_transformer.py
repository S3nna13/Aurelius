"""Tests for Differential Transformer Attention (Microsoft, 2024).

Uses a deliberately tiny configuration so every test runs in <1 s on CPU:
    d_model=64, n_heads=4, head_dim=16, n_layers=2,
    vocab_size=256, seq_len=8, batch_size=2
"""

from __future__ import annotations

import pytest
import torch

from src.model.diff_transformer import (
    DiffAttnConfig,
    DifferentialAttention,
    DiffTransformerBlock,
    DiffTransformerLayer,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

BATCH = 2
SEQ = 8
D_MODEL = 64
N_HEADS = 4
HEAD_DIM = 16
N_LAYERS = 2
VOCAB_SIZE = 256


@pytest.fixture
def cfg() -> DiffAttnConfig:
    return DiffAttnConfig(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        head_dim=HEAD_DIM,
        dropout=0.0,
        lambda_init=0.8,
        n_layers=N_LAYERS,
    )


@pytest.fixture
def attn(cfg: DiffAttnConfig) -> DifferentialAttention:
    return DifferentialAttention(cfg)


@pytest.fixture
def block(cfg: DiffAttnConfig) -> DiffTransformerBlock:
    return DiffTransformerBlock(cfg)


@pytest.fixture
def layer(cfg: DiffAttnConfig) -> DiffTransformerLayer:
    return DiffTransformerLayer(cfg)


@pytest.fixture
def x() -> torch.Tensor:
    """Random input tensor (B, T, d_model)."""
    torch.manual_seed(0)
    return torch.randn(BATCH, SEQ, D_MODEL)


def _causal_mask(seq_len: int) -> torch.Tensor:
    """Build a standard additive causal mask (upper triangle = -inf)."""
    mask = torch.zeros(seq_len, seq_len)
    mask.fill_(float("-inf"))
    mask = torch.triu(mask, diagonal=1)
    # Shape: (1, 1, T, T) — broadcast over batch and heads
    return mask.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Test 1 — Output shape is correct (B, T, d_model)
# ---------------------------------------------------------------------------


def test_output_shape(attn: DifferentialAttention, x: torch.Tensor) -> None:
    with torch.no_grad():
        out = attn(x)
    assert out.shape == (BATCH, SEQ, D_MODEL), (
        f"Expected ({BATCH}, {SEQ}, {D_MODEL}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# Test 2 — Lambda is learnable and clamped in [0, 1]
# ---------------------------------------------------------------------------


def test_lambda_learnable_and_clamped(attn: DifferentialAttention) -> None:
    # lambda_param must be an nn.Parameter (i.e. requires_grad=True)
    assert isinstance(attn.lambda_param, torch.nn.Parameter)
    assert attn.lambda_param.requires_grad

    # lambda_values must stay in [0, 1] even after arbitrary raw param values
    with torch.no_grad():
        attn.lambda_param.fill_(5.0)  # raw value outside valid range
    clamped = attn.lambda_values
    assert (clamped >= 0.0).all() and (clamped <= 1.0).all(), (
        f"lambda_values out of [0,1]: {clamped}"
    )

    with torch.no_grad():
        attn.lambda_param.fill_(-3.0)  # another out-of-range value
    clamped = attn.lambda_values
    assert (clamped >= 0.0).all() and (clamped <= 1.0).all()


# ---------------------------------------------------------------------------
# Test 3 — Causal mask zeros upper triangle (future positions get ~0 weight)
# ---------------------------------------------------------------------------


def test_causal_mask_zeros_upper_triangle(attn: DifferentialAttention) -> None:
    """With a causal mask, token i must not be influenced by token j > i.

    Strategy: run two passes with identical tokens except that future tokens
    differ.  The output at positions that should be causally shielded must be
    unchanged.
    """
    torch.manual_seed(1)
    x1 = torch.randn(1, SEQ, D_MODEL)
    x2 = x1.clone()
    # Perturb every token beyond position 0
    x2[:, 1:, :] = torch.randn_like(x2[:, 1:, :])

    mask = _causal_mask(SEQ)

    # Switch to inference mode (no dropout)
    attn.train(False)
    with torch.no_grad():
        out1 = attn(x1, mask)
        out2 = attn(x2, mask)
    attn.train(True)

    # Position 0 only attends to itself, so its output must be equal
    assert torch.allclose(out1[:, 0, :], out2[:, 0, :], atol=1e-5), (
        "Causal mask broken: position 0 output changed when future tokens changed."
    )


# ---------------------------------------------------------------------------
# Test 4 — Forward pass produces no NaN or Inf
# ---------------------------------------------------------------------------


def test_no_nan_inf(attn: DifferentialAttention, x: torch.Tensor) -> None:
    with torch.no_grad():
        out = attn(x)
    assert not torch.isnan(out).any(), "NaN detected in attention output."
    assert not torch.isinf(out).any(), "Inf detected in attention output."


# ---------------------------------------------------------------------------
# Test 5 — Backward pass computes gradients (loss.backward() succeeds)
# ---------------------------------------------------------------------------


def test_backward_pass(attn: DifferentialAttention, x: torch.Tensor) -> None:
    x_req = x.detach().requires_grad_(True)
    out = attn(x_req)
    loss = out.sum()
    loss.backward()  # must not raise
    assert x_req.grad is not None, "No gradient flowed back to the input."


# ---------------------------------------------------------------------------
# Test 6 — Lambda gradient is non-zero after backward
# ---------------------------------------------------------------------------


def test_lambda_gradient_nonzero(attn: DifferentialAttention, x: torch.Tensor) -> None:
    # Reset param to a mid-range value so clamping doesn't zero the gradient
    with torch.no_grad():
        attn.lambda_param.fill_(0.5)

    out = attn(x)
    loss = out.sum()
    loss.backward()

    assert attn.lambda_param.grad is not None, "No gradient for lambda_param."
    assert attn.lambda_param.grad.abs().sum() > 0, (
        "Lambda gradient is zero — lambda is not affecting the output."
    )


# ---------------------------------------------------------------------------
# Test 7 — DiffTransformerBlock output shape
# ---------------------------------------------------------------------------


def test_block_output_shape(block: DiffTransformerBlock, x: torch.Tensor) -> None:
    with torch.no_grad():
        out = block(x)
    assert out.shape == (BATCH, SEQ, D_MODEL), (
        f"DiffTransformerBlock output shape mismatch: {out.shape}"
    )


# ---------------------------------------------------------------------------
# Test 8 — DiffTransformerLayer stack output shape
# ---------------------------------------------------------------------------


def test_layer_stack_output_shape(layer: DiffTransformerLayer, x: torch.Tensor) -> None:
    with torch.no_grad():
        out = layer(x)
    assert out.shape == (BATCH, SEQ, D_MODEL), (
        f"DiffTransformerLayer stack output shape mismatch: {out.shape}"
    )


# ---------------------------------------------------------------------------
# Test 9 — Different lambda values produce different outputs
# ---------------------------------------------------------------------------


def test_different_lambda_different_output(cfg: DiffAttnConfig, x: torch.Tensor) -> None:
    attn_a = DifferentialAttention(cfg)
    attn_b = DifferentialAttention(cfg)

    # Copy all weights except lambda so only lambda differs
    with torch.no_grad():
        attn_b.qkv_proj.weight.copy_(attn_a.qkv_proj.weight)
        attn_b.out_proj.weight.copy_(attn_a.out_proj.weight)
        attn_a.lambda_param.fill_(0.1)
        attn_b.lambda_param.fill_(0.9)

    # Inference mode (no dropout)
    attn_a.train(False)
    attn_b.train(False)
    with torch.no_grad():
        out_a = attn_a(x)
        out_b = attn_b(x)

    assert not torch.allclose(out_a, out_b, atol=1e-6), (
        "Different lambda values should produce different outputs."
    )


# ---------------------------------------------------------------------------
# Test 10 — Deterministic with same random seed
# ---------------------------------------------------------------------------


def test_deterministic_with_same_seed(cfg: DiffAttnConfig) -> None:
    def _run() -> torch.Tensor:
        torch.manual_seed(42)
        model = DifferentialAttention(cfg)
        model.train(False)
        inp = torch.randn(BATCH, SEQ, D_MODEL)
        with torch.no_grad():
            return model(inp)

    out1 = _run()
    out2 = _run()
    assert torch.allclose(out1, out2), "Same seed should yield identical outputs."


# ---------------------------------------------------------------------------
# Bonus Test 11 — Block no NaN/Inf with causal mask
# ---------------------------------------------------------------------------


def test_block_no_nan_with_causal_mask(block: DiffTransformerBlock, x: torch.Tensor) -> None:
    mask = _causal_mask(SEQ)
    block.train(False)
    with torch.no_grad():
        out = block(x, mask)
    assert not torch.isnan(out).any(), "NaN in block output with causal mask."
    assert not torch.isinf(out).any(), "Inf in block output with causal mask."


# ---------------------------------------------------------------------------
# Bonus Test 12 — Layer stack no NaN/Inf
# ---------------------------------------------------------------------------


def test_layer_no_nan(layer: DiffTransformerLayer, x: torch.Tensor) -> None:
    layer.train(False)
    with torch.no_grad():
        out = layer(x)
    assert not torch.isnan(out).any(), "NaN in DiffTransformerLayer output."
    assert not torch.isinf(out).any(), "Inf in DiffTransformerLayer output."
