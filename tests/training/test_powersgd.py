"""Tests for PowerSGD gradient compressor (arXiv:1905.13727).

All tests use pure PyTorch — no scipy, sklearn, or heavy ML frameworks.
"""
from __future__ import annotations

import math

import torch
import pytest

from src.training.powersgd import PowerSGDCompressor, PowerSGDOptimizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_low_rank_matrix(m: int, n: int, rank: int, seed: int = 0) -> torch.Tensor:
    """Return a random matrix with exact rank `rank`."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    A = torch.randn(m, rank, generator=gen)
    B = torch.randn(n, rank, generator=gen)
    return A @ B.t()


# ===========================================================================
# Test 1 — compress() output shapes: M is (m, r), Q is (n, r)
# ===========================================================================

def test_compress_output_shapes():
    m, n, r = 32, 16, 4
    G = torch.randn(m, n)
    comp = PowerSGDCompressor(rank=r)
    M, Q, residual = comp.compress(G)

    assert M is not None and Q is not None
    assert M.shape == (m, r), f"expected M shape ({m}, {r}), got {M.shape}"
    assert Q.shape == (n, r), f"expected Q shape ({n}, {r}), got {Q.shape}"
    assert residual is not None and residual.shape == G.shape


# ===========================================================================
# Test 2 — Q columns are orthonormal: Q^T @ Q ≈ I
# ===========================================================================

def test_q_columns_orthonormal():
    m, n, r = 64, 32, 6
    G = torch.randn(m, n)
    comp = PowerSGDCompressor(rank=r, n_power_iter=1)
    _, Q, _ = comp.compress(G)

    I_approx = Q.t() @ Q
    I_ref = torch.eye(r, dtype=Q.dtype)
    assert torch.allclose(I_approx, I_ref, atol=1e-5), (
        f"Q is not orthonormal: max deviation {(I_approx - I_ref).abs().max():.2e}"
    )


# ===========================================================================
# Test 3 — decompress(compress(G)) ≈ G for low-rank matrices
#           (exact reconstruction when rank >= true rank)
# ===========================================================================

def test_exact_reconstruction_low_rank():
    m, n, true_rank = 40, 20, 3
    G = _make_low_rank_matrix(m, n, true_rank, seed=42)

    comp = PowerSGDCompressor(rank=true_rank, n_power_iter=3)
    M, Q, _ = comp.compress(G)
    G_hat = comp.decompress(M, Q, original_shape=G.shape)

    frob_error = torch.norm(G - G_hat, p="fro")
    frob_G = torch.norm(G, p="fro")
    # Relative error should be very small for rank-matching compression
    assert frob_error / (frob_G + 1e-9) < 1e-3, (
        f"Reconstruction error too large: {frob_error:.4f} vs G norm {frob_G:.4f}"
    )


# ===========================================================================
# Test 4 — Reconstruction error < Frobenius norm of G for random G, r=1
# ===========================================================================

def test_reconstruction_error_less_than_grad_norm():
    torch.manual_seed(7)
    m, n = 50, 30
    G = torch.randn(m, n)

    comp = PowerSGDCompressor(rank=1, n_power_iter=2)
    M, Q, _ = comp.compress(G)
    G_hat = comp.decompress(M, Q, original_shape=G.shape)

    frob_error = torch.norm(G - G_hat, p="fro").item()
    frob_G = torch.norm(G, p="fro").item()
    assert frob_error < frob_G, (
        f"Reconstruction error {frob_error:.4f} >= G norm {frob_G:.4f}"
    )


# ===========================================================================
# Test 5 — Error feedback: residual stored; next call adds e_t to input
# ===========================================================================

def test_error_feedback_stored_and_applied():
    torch.manual_seed(13)
    m, n = 24, 16

    comp = PowerSGDCompressor(rank=2, n_power_iter=1)

    G1 = torch.randn(m, n)
    # First step: no prior feedback
    out1 = comp.step(G1)
    assert comp._error_feedback is not None, "error_feedback should be stored after first step"
    assert comp._error_feedback.shape == G1.shape

    # Capture the residual stored after step 1
    stored_residual = comp._error_feedback.clone()

    # Second step: feedback should be added to G2 before compression
    G2 = torch.randn(m, n)
    # Manually compress G2 + stored_residual to verify the feedback is used
    comp2 = PowerSGDCompressor(rank=2, n_power_iter=1)
    G2_with_feedback = G2 + stored_residual
    M_ref, Q_ref, _ = comp2.compress(G2_with_feedback)

    # The two compressors use different random Q initialisations, so we
    # can't compare outputs directly — instead verify that the internal
    # buffer is updated after the second step
    comp.step(G2)
    assert comp._error_feedback is not None
    assert comp._error_feedback.shape == G2.shape


# ===========================================================================
# Test 6 — 1-D tensors pass through unmodified
# ===========================================================================

def test_1d_tensor_passthrough():
    bias = torch.randn(64)
    comp = PowerSGDCompressor(rank=4)

    M, Q, residual = comp.compress(bias)
    # For 1-D: M == grad, Q is None, residual is None
    assert Q is None, "Q should be None for 1-D gradient"
    assert residual is None, "residual should be None for 1-D gradient"
    assert torch.equal(M, bias), "1-D gradient should be returned unchanged"


def test_1d_tensor_step_passthrough():
    bias = torch.randn(32)
    comp = PowerSGDCompressor(rank=4)
    out = comp.step(bias)
    assert torch.equal(out, bias), "step() should return 1-D gradient unchanged"
    assert comp._error_feedback is None, "no error feedback for 1-D"


# ===========================================================================
# Test 7 — n_power_iter=0 still produces valid (M, Q) without iteration
# ===========================================================================

def test_zero_power_iterations():
    m, n, r = 20, 10, 3
    G = torch.randn(m, n)
    comp = PowerSGDCompressor(rank=r, n_power_iter=0)
    M, Q, residual = comp.compress(G)

    assert M.shape == (m, r)
    assert Q.shape == (n, r)
    # Q should still be orthonormal (initialised via QR)
    I_approx = Q.t() @ Q
    assert torch.allclose(I_approx, torch.eye(r), atol=1e-5)


# ===========================================================================
# Test 8 — PowerSGDOptimizer: params move after step
# ===========================================================================

def test_optimizer_params_move():
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.randn(16, 8))
    param_before = param.data.clone()

    param.grad = torch.randn_like(param)

    opt = PowerSGDOptimizer([param], lr=0.1, rank=2)
    opt.step()

    assert not torch.equal(param.data, param_before), "param should change after optimizer step"


# ===========================================================================
# Test 9 — PowerSGDOptimizer: param movement is finite (no NaN/Inf)
# ===========================================================================

def test_optimizer_params_finite():
    torch.manual_seed(1)
    param = torch.nn.Parameter(torch.randn(32, 16))
    param.grad = torch.randn_like(param)

    opt = PowerSGDOptimizer([param], lr=0.01, rank=4, n_power_iter=1)
    opt.step()

    assert torch.isfinite(param.data).all(), "param contains NaN/Inf after step"


# ===========================================================================
# Test 10 — Determinism: same seed → same compressed output
# ===========================================================================

def test_determinism_same_seed():
    G = torch.randn(24, 12)

    torch.manual_seed(42)
    comp1 = PowerSGDCompressor(rank=3, n_power_iter=2)
    M1, Q1, _ = comp1.compress(G)

    torch.manual_seed(42)
    comp2 = PowerSGDCompressor(rank=3, n_power_iter=2)
    M2, Q2, _ = comp2.compress(G)

    assert torch.allclose(M1, M2, atol=1e-6), "M outputs differ with same seed"
    assert torch.allclose(Q1, Q2, atol=1e-6), "Q outputs differ with same seed"


# ===========================================================================
# Test 11 — rank >= min(m, n) → clamped gracefully (no error)
# ===========================================================================

def test_rank_clamped_gracefully():
    m, n = 8, 6
    G = torch.randn(m, n)
    # Request rank larger than min(m, n)
    comp = PowerSGDCompressor(rank=100, n_power_iter=1)
    M, Q, residual = comp.compress(G)

    effective_r = min(100, min(m, n))
    assert M.shape == (m, effective_r), f"expected M shape ({m}, {effective_r})"
    assert Q.shape == (n, effective_r), f"expected Q shape ({n}, {effective_r})"


# ===========================================================================
# Test 12 — Error feedback accumulates correctly across 2 steps
# ===========================================================================

def test_error_feedback_accumulates_two_steps():
    torch.manual_seed(99)
    m, n = 20, 12
    G1 = torch.randn(m, n)
    G2 = torch.randn(m, n)

    comp = PowerSGDCompressor(rank=2, n_power_iter=1)

    out1 = comp.step(G1)
    e1 = comp._error_feedback.clone()

    # Reconstruct step 1: G1 approximation = out1, residual = G1 - out1
    expected_e1 = (G1 - out1)
    assert torch.allclose(e1, expected_e1, atol=1e-5), (
        "error feedback after step 1 should equal G1 - G1_approx"
    )

    out2 = comp.step(G2)
    e2 = comp._error_feedback.clone()

    # After step 2: error feedback = (G2 + e1) - out2
    expected_e2 = (G2 + e1 - out2)
    assert torch.allclose(e2, expected_e2, atol=1e-5), (
        "error feedback after step 2 incorrect"
    )


# ===========================================================================
# Test 13 — Compression ratio: M+Q storage < G storage for r < n/2
# ===========================================================================

def test_compression_ratio():
    m, n, r = 128, 64, 8   # r=8 << n/2=32 → compressed < original
    G = torch.randn(m, n)

    comp = PowerSGDCompressor(rank=r, n_power_iter=1)
    M, Q, _ = comp.compress(G)

    original_elements = G.numel()            # m * n = 8192
    compressed_elements = M.numel() + Q.numel()  # m*r + n*r = (m+n)*r

    assert compressed_elements < original_elements, (
        f"Compressed size {compressed_elements} >= original size {original_elements}"
    )


# ===========================================================================
# Test 14 — Large gradient stability: no NaN/Inf on inputs scaled to 1000
# ===========================================================================

def test_large_gradient_stability():
    torch.manual_seed(5)
    G = torch.randn(32, 16) * 1000.0

    comp = PowerSGDCompressor(rank=4, n_power_iter=2)
    M, Q, residual = comp.compress(G)

    assert torch.isfinite(M).all(), "M contains NaN/Inf for large gradient"
    assert torch.isfinite(Q).all(), "Q contains NaN/Inf for large gradient"
    assert torch.isfinite(residual).all(), "residual contains NaN/Inf for large gradient"

    G_hat = comp.decompress(M, Q, original_shape=G.shape)
    assert torch.isfinite(G_hat).all(), "reconstructed G contains NaN/Inf for large gradient"
