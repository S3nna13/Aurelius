"""
Tests for Native Sparse Attention (NSA) — src/model/nsa.py
arXiv:2502.11089

Tiny config used throughout:
  d_model=64, n_heads=4, head_dim=16, block_size=4, r_blocks=2, window_size=8
"""

import math
import pytest
import torch
import torch.nn as nn

from src.model.nsa import NativeSparseAttention


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TINY = dict(d_model=64, n_heads=4, block_size=4, r_blocks=2, window_size=8)


def make_model(**kwargs) -> NativeSparseAttention:
    cfg = {**TINY, **kwargs}
    return NativeSparseAttention(**cfg)


def make_input(B: int = 2, T: int = 32, d_model: int = 64, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(B, T, d_model)


# ---------------------------------------------------------------------------
# Test 1: Output shape matches input shape
# ---------------------------------------------------------------------------

def test_output_shape():
    model = make_model()
    x = make_input(B=2, T=32)
    out = model(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


# ---------------------------------------------------------------------------
# Test 2: Gradient flow — all trainable params receive finite gradients
# ---------------------------------------------------------------------------

def test_gradient_flow():
    model = make_model()
    x = make_input(B=2, T=32)
    out = model(x)
    loss = out.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No grad for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite grad for {name}"


# ---------------------------------------------------------------------------
# Test 3: Determinism under torch.manual_seed
# ---------------------------------------------------------------------------

def test_determinism():
    torch.manual_seed(42)
    model = make_model()
    x = make_input(B=2, T=32, seed=42)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.equal(out1, out2), "Outputs differ between two forward passes with same input"


# ---------------------------------------------------------------------------
# Test 4: batch=1, seq_len=1 edge case
# ---------------------------------------------------------------------------

def test_single_token():
    model = make_model()
    x = make_input(B=1, T=1)
    out = model(x)
    assert out.shape == (1, 1, 64)
    assert torch.isfinite(out).all(), "Non-finite output for single token"


# ---------------------------------------------------------------------------
# Test 5: seq_len shorter than one block (T < block_size=4)
# ---------------------------------------------------------------------------

def test_seq_shorter_than_block():
    model = make_model()
    x = make_input(B=2, T=3)   # T=3 < block_size=4
    out = model(x)
    assert out.shape == (2, 3, 64)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Test 6: seq_len not divisible by block_size — pads gracefully
# ---------------------------------------------------------------------------

def test_seq_not_divisible_by_block():
    model = make_model()
    x = make_input(B=2, T=14)  # 14 % 4 = 2, not divisible
    out = model(x)
    assert out.shape == (2, 14, 64)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Test 7: Numerical stability — all-zero input
# ---------------------------------------------------------------------------

def test_stability_zero_input():
    model = make_model()
    x = torch.zeros(2, 32, 64)
    out = model(x)
    assert torch.isfinite(out).all(), "NaN/Inf on all-zero input"


# ---------------------------------------------------------------------------
# Test 8: Numerical stability — large input (100x scale)
# ---------------------------------------------------------------------------

def test_stability_large_input():
    model = make_model()
    x = make_input(B=2, T=32) * 100.0
    out = model(x)
    assert torch.isfinite(out).all(), "NaN/Inf on large-scale input"


# ---------------------------------------------------------------------------
# Test 9: Causal masking — output at t must not depend on t+1
# ---------------------------------------------------------------------------

def test_causal_masking():
    """
    Verify that changing token at position t+1 does NOT change the output
    at position t (i.e., no future information leaks).
    """
    model = make_model(causal=True)
    model.eval()

    T = 16
    x1 = make_input(B=1, T=T, seed=1)
    x2 = x1.clone()
    # Perturb the last token only
    x2[:, -1, :] = x2[:, -1, :] + 1.0

    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)

    # All positions except the last should be identical (causal)
    assert torch.allclose(out1[:, :-1, :], out2[:, :-1, :], atol=1e-5), (
        "Causal violation: earlier positions changed when later token was modified"
    )


# ---------------------------------------------------------------------------
# Test 10: Gate weights sum to 1 (softmax property)
# ---------------------------------------------------------------------------

def test_gate_weights_sum_to_one():
    """
    Manually run the gate MLP on a test input and verify alpha+beta+gamma ~ 1.
    """
    model = make_model()
    x = make_input(B=2, T=32)
    with torch.no_grad():
        q = model._split_heads(model.W_q(x))           # (B, H, T, D)
        alpha, beta, gamma = model._compute_gates(q)    # each (B, T, 1)
    gate_sum = alpha + beta + gamma                     # (B, T, 1)
    assert torch.allclose(gate_sum, torch.ones_like(gate_sum), atol=1e-5), (
        f"Gate weights do not sum to 1; max deviation {(gate_sum - 1).abs().max()}"
    )


# ---------------------------------------------------------------------------
# Test 11: All three branches are active (non-zero contribution)
# ---------------------------------------------------------------------------

def test_all_branches_active():
    """
    Each gate weight should be strictly positive on normal random input.
    """
    model = make_model()
    x = make_input(B=2, T=32, seed=7)
    with torch.no_grad():
        q = model._split_heads(model.W_q(x))
        alpha, beta, gamma = model._compute_gates(q)

    assert (alpha > 0).all(), "alpha gate is zero somewhere"
    assert (beta  > 0).all(), "beta gate is zero somewhere"
    assert (gamma > 0).all(), "gamma gate is zero somewhere"


# ---------------------------------------------------------------------------
# Test 12: r_blocks > available blocks — clamps gracefully
# ---------------------------------------------------------------------------

def test_r_blocks_exceeds_available():
    """
    With T=8 and block_size=4, there are only 2 blocks.
    Setting r_blocks=10 should clamp to 2 and not crash.
    """
    model = make_model(r_blocks=10)
    x = make_input(B=1, T=8)
    out = model(x)
    assert out.shape == (1, 8, 64)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Test 13: attention_mask (padding) handled without crash
# ---------------------------------------------------------------------------

def test_attention_mask_no_crash():
    model = make_model()
    B, T = 2, 32
    x = make_input(B=B, T=T)
    # Second half of second sequence is padding
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[1, T // 2:] = False
    out = model(x, attention_mask=mask)
    assert out.shape == (B, T, 64)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Test 14: Parameter count is reasonable
# ---------------------------------------------------------------------------

def test_parameter_count():
    """
    For d_model=64, n_heads=4, block_size=4:
      - 4 linear projections (Q,K,V,O): 4 * 64*64 = 16384
      - 2 compression projections: 2 * (4*16)*16 = 2048
      - gate MLP bias + weight: small
    Total should be well under 100k for tiny config.
    """
    model = make_model()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 100_000, f"Unexpectedly large parameter count: {n_params}"
    assert n_params > 1_000,   f"Suspiciously small parameter count: {n_params}"


# ---------------------------------------------------------------------------
# Test 15: Non-causal mode runs without error
# ---------------------------------------------------------------------------

def test_non_causal_mode():
    model = make_model(causal=False)
    x = make_input(B=2, T=32)
    out = model(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Test 16: Larger sequence with multiple complete blocks
# ---------------------------------------------------------------------------

def test_larger_sequence():
    model = make_model()
    x = make_input(B=2, T=64)   # 64 / 4 = 16 blocks
    out = model(x)
    assert out.shape == (2, 64, 64)
    assert torch.isfinite(out).all()
