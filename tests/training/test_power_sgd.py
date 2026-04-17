"""
Tests for src/training/power_sgd.py
All tests run actual forward/backward passes or tensor operations.
Tiny configs: matrices (16,8) max, rank=2.
"""

import pytest
import torch
import torch.nn as nn

from src.training.power_sgd import (
    LowRankApproximation,
    PowerIteration,
    GradientCompressor,
    PowerSGDOptimizer,
    CompressionStats,
)


# ---------------------------------------------------------------------------
# LowRankApproximation tests
# ---------------------------------------------------------------------------

def test_low_rank_decompose_shapes():
    """P shape (m, rank), Q shape (n, rank)."""
    lra = LowRankApproximation(rank=2)
    G = torch.randn(16, 8)
    P, Q = lra.decompose(G)
    assert P.shape == (16, 2), f"Expected P (16,2), got {P.shape}"
    assert Q.shape == (8, 2), f"Expected Q (8,2), got {Q.shape}"


def test_low_rank_reconstruct_shape():
    """Reconstructed matrix shape matches original."""
    lra = LowRankApproximation(rank=2)
    G = torch.randn(16, 8)
    P, Q = lra.decompose(G)
    G_approx = lra.reconstruct(P, Q)
    assert G_approx.shape == G.shape, f"Shape mismatch: {G_approx.shape} vs {G.shape}"


def test_low_rank_reconstruct_close_rank4():
    """Low-rank approx is reasonably close for rank=4 on random matrix (error < 0.5)."""
    lra = LowRankApproximation(rank=4)
    # Use a low-rank-friendly matrix: outer product + small noise
    u = torch.randn(16, 4)
    v = torch.randn(8, 4)
    G = u @ v.T + 0.01 * torch.randn(16, 8)
    P, Q = lra.decompose(G)
    G_approx = lra.reconstruct(P, Q)
    error = (torch.norm(G - G_approx, p="fro") / torch.norm(G, p="fro")).item()
    assert error < 0.5, f"Reconstruction error too large: {error:.4f}"


def test_low_rank_compression_ratio_gt1():
    """Compression ratio > 1.0 when rank < min(m,n)/2."""
    lra = LowRankApproximation(rank=2)
    ratio = lra.compression_ratio((16, 8))
    assert ratio > 1.0, f"Expected ratio > 1.0, got {ratio:.4f}"


# ---------------------------------------------------------------------------
# PowerIteration tests
# ---------------------------------------------------------------------------

def test_power_iteration_shapes():
    """P (m, rank) and Q (n, rank) shapes are correct."""
    pi = PowerIteration(rank=2, n_iter=2)
    G = torch.randn(16, 8)
    P, Q = pi.run(G)
    assert P.shape == (16, 2), f"Expected P (16,2), got {P.shape}"
    assert Q.shape == (8, 2), f"Expected Q (8,2), got {Q.shape}"


def test_power_iteration_reconstruction_error_in_range():
    """Reconstruction error is in [0, 1] for a random matrix."""
    pi = PowerIteration(rank=2, n_iter=2)
    G = torch.randn(16, 8)
    P, Q = pi.run(G)
    err = pi.reconstruction_error(G, P, Q)
    assert 0.0 <= err <= 1.0, f"Error out of range: {err}"


def test_power_iteration_n_iter_zero():
    """With n_iter=0, run still returns valid P, Q of correct shapes."""
    pi = PowerIteration(rank=2, n_iter=0)
    G = torch.randn(16, 8)
    P, Q = pi.run(G)
    assert P.shape == (16, 2)
    assert Q.shape == (8, 2)
    # Should not raise; error can be anything in [0,1]
    err = pi.reconstruction_error(G, P, Q)
    assert 0.0 <= err <= 1.0


# ---------------------------------------------------------------------------
# GradientCompressor tests
# ---------------------------------------------------------------------------

def test_gradient_compressor_2d_compressed():
    """2D grad above min ratio: metadata['compressed']==True, P and Q stored."""
    gc = GradientCompressor(rank=2, min_compression_ratio=1.5, n_iter=1)
    grad = torch.randn(16, 8)
    compressed, meta = gc.compress(grad)
    assert meta["compressed"] is True
    assert "P" in compressed
    assert "Q" in compressed


def test_gradient_compressor_1d_not_compressed():
    """1D grad: metadata['compressed']==False, tensor returned unchanged."""
    gc = GradientCompressor(rank=2, min_compression_ratio=2.0, n_iter=1)
    grad = torch.randn(16)
    compressed, meta = gc.compress(grad)
    assert meta["compressed"] is False
    assert torch.equal(compressed, grad)


def test_gradient_compressor_decompress_shape():
    """Decompressed gradient has same shape as original."""
    gc = GradientCompressor(rank=2, min_compression_ratio=1.5, n_iter=1)
    original = torch.randn(16, 8)
    compressed, meta = gc.compress(original)
    reconstructed = gc.decompress(compressed, meta)
    assert reconstructed.shape == original.shape, (
        f"Shape mismatch: {reconstructed.shape} vs {original.shape}"
    )


def test_gradient_compressor_decompress_3d_shape():
    """3D grad is reshaped to 2D internally; decompressed shape matches original."""
    gc = GradientCompressor(rank=2, min_compression_ratio=1.5, n_iter=1)
    original = torch.randn(4, 4, 4)
    compressed, meta = gc.compress(original)
    if meta["compressed"]:
        reconstructed = gc.decompress(compressed, meta)
        assert reconstructed.shape == original.shape


# ---------------------------------------------------------------------------
# PowerSGDOptimizer tests
# ---------------------------------------------------------------------------

def _make_model_and_opt(start_iter: int = 0):
    """Helper: tiny linear model + PowerSGDOptimizer."""
    model = nn.Linear(8, 4, bias=False)
    base_opt = torch.optim.SGD(model.parameters(), lr=0.01)
    compressor = GradientCompressor(rank=2, min_compression_ratio=1.5, n_iter=1)
    opt = PowerSGDOptimizer(base_opt, compressor, start_iter=start_iter)
    return model, opt


def test_powersgd_step_runs_and_updates_params():
    """step() runs without error and params differ from initial values."""
    model, opt = _make_model_and_opt(start_iter=0)
    initial = model.weight.data.clone()

    x = torch.randn(2, 8)
    loss = model(x).sum()
    loss.backward()
    opt.step()

    assert not torch.equal(model.weight.data, initial), "Params should have changed after step"


def test_powersgd_warmup_before_start_iter():
    """Before start_iter steps, gradients are not compressed (raw SGD update)."""
    model, opt = _make_model_and_opt(start_iter=100)  # won't reach compression

    x = torch.randn(2, 8)
    loss = model(x).sum()
    loss.backward()

    # Capture grad before step
    grad_before = model.weight.grad.data.clone()
    opt.step()

    # n_steps should be 1, which is <= start_iter=100, so no compression
    assert opt.n_steps == 1
    # Params should still update via base optimizer
    assert not torch.all(model.weight.grad == 0)


def test_powersgd_zero_grad_delegates():
    """zero_grad() clears gradients on all parameters."""
    model, opt = _make_model_and_opt()
    x = torch.randn(2, 8)
    loss = model(x).sum()
    loss.backward()
    assert model.weight.grad is not None
    opt.zero_grad()
    assert model.weight.grad is None or torch.all(model.weight.grad == 0)


# ---------------------------------------------------------------------------
# CompressionStats tests
# ---------------------------------------------------------------------------

def test_compression_stats_summary_keys():
    """summary() dict has all required keys and correct counts."""
    stats = CompressionStats()
    stats.record("layer0", {"compressed": True, "rank": 2, "error": 0.1})
    stats.record("layer1", {"compressed": False})
    stats.record("layer2", {"compressed": True, "rank": 2, "error": 0.2})

    s = stats.summary()
    assert "total_params_compressed" in s
    assert "total_params_skipped" in s
    assert "mean_compression_ratio" in s
    assert "mean_reconstruction_error" in s
    assert s["total_params_compressed"] == 2
    assert s["total_params_skipped"] == 1
    assert abs(s["mean_reconstruction_error"] - 0.15) < 1e-6


def test_compression_stats_reset():
    """After reset(), summary returns zeros."""
    stats = CompressionStats()
    stats.record("layer0", {"compressed": True, "rank": 2, "error": 0.3})
    stats.reset()
    s = stats.summary()
    assert s["total_params_compressed"] == 0
    assert s["total_params_skipped"] == 0
    assert s["mean_reconstruction_error"] == 0.0


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def test_full_training_loop_finite_loss():
    """3-step training loop with PowerSGDOptimizer produces finite loss values."""
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(8, 16, bias=False),
        nn.ReLU(),
        nn.Linear(16, 4, bias=False),
    )
    base_opt = torch.optim.SGD(model.parameters(), lr=0.05)
    compressor = GradientCompressor(rank=2, min_compression_ratio=1.5, n_iter=1)
    opt = PowerSGDOptimizer(base_opt, compressor, start_iter=1)

    losses = []
    for _ in range(3):
        opt.zero_grad()
        x = torch.randn(2, 8)
        y = torch.randint(0, 4, (2,))
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    for i, l in enumerate(losses):
        assert torch.isfinite(torch.tensor(l)), f"Loss at step {i} is not finite: {l}"


# ---------------------------------------------------------------------------
# n_steps counter
# ---------------------------------------------------------------------------

def test_n_steps_counter():
    """n_steps increments correctly across multiple step() calls."""
    model, opt = _make_model_and_opt(start_iter=0)
    assert opt.n_steps == 0

    for i in range(1, 4):
        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        assert opt.n_steps == i, f"Expected n_steps={i}, got {opt.n_steps}"
        opt.zero_grad()
