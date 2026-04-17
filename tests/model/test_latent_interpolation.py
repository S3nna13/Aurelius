"""Tests for src/model/latent_interpolation.py

Covers:
1.  LinearInterpolator.interpolate at t=0 returns z0
2.  LinearInterpolator.interpolate at t=1 returns z1
3.  LinearInterpolator.interpolate at t=0.5 returns midpoint
4.  LinearInterpolator.path returns (n_steps, d) shape
5.  LinearInterpolator.path endpoints match z0 and z1
6.  SphericalInterpolator.interpolate at t=0 returns z0 direction
7.  SphericalInterpolator.interpolate at t=0.5 is equidistant from z0 and z1 (on sphere)
8.  SphericalInterpolator.path shape (n_steps, d)
9.  SphericalInterpolator handles parallel vectors (fall back to lerp)
10. ManifoldInterpolator output shape matches input
11. ManifoldInterpolator gradients flow
12. ManifoldInterpolator.path shape (n_steps, d_model)
13. InterpolationAnalyzer.path_length > 0 for non-trivial path
14. InterpolationAnalyzer.smoothness == 1.0 for linear path
15. InterpolationAnalyzer.midpoint_deviation == 0 for linear path
"""

from __future__ import annotations

import pytest
import torch

from src.model.latent_interpolation import (
    InterpolationAnalyzer,
    LinearInterpolator,
    ManifoldInterpolator,
    SphericalInterpolator,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

D_MODEL = 16
D_LATENT = 8
N_STEPS = 5

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def z_pair():
    """A random pair of 1-D vectors of length D_MODEL."""
    torch.manual_seed(0)
    z0 = torch.randn(D_MODEL)
    z1 = torch.randn(D_MODEL)
    return z0, z1


@pytest.fixture
def linear_interp():
    return LinearInterpolator()


@pytest.fixture
def slerp_interp():
    return SphericalInterpolator()


@pytest.fixture
def manifold_interp():
    torch.manual_seed(42)
    return ManifoldInterpolator(d_model=D_MODEL, d_latent=D_LATENT)


@pytest.fixture
def analyzer():
    return InterpolationAnalyzer()


# ---------------------------------------------------------------------------
# LinearInterpolator tests
# ---------------------------------------------------------------------------


def test_linear_t0_returns_z0(linear_interp, z_pair):
    """t=0 should exactly reproduce z0."""
    z0, z1 = z_pair
    result = linear_interp.interpolate(z0, z1, 0.0)
    assert torch.allclose(result, z0)


def test_linear_t1_returns_z1(linear_interp, z_pair):
    """t=1 should exactly reproduce z1."""
    z0, z1 = z_pair
    result = linear_interp.interpolate(z0, z1, 1.0)
    assert torch.allclose(result, z1)


def test_linear_t05_returns_midpoint(linear_interp, z_pair):
    """t=0.5 should return the arithmetic midpoint."""
    z0, z1 = z_pair
    result = linear_interp.interpolate(z0, z1, 0.5)
    expected = (z0 + z1) / 2.0
    assert torch.allclose(result, expected)


def test_linear_path_shape(linear_interp, z_pair):
    """path() for 1-D inputs should return (n_steps, d)."""
    z0, z1 = z_pair
    path = linear_interp.path(z0, z1, N_STEPS)
    assert path.shape == (N_STEPS, D_MODEL)


def test_linear_path_endpoints(linear_interp, z_pair):
    """First and last points of path() should match z0 and z1."""
    z0, z1 = z_pair
    path = linear_interp.path(z0, z1, N_STEPS)
    assert torch.allclose(path[0], z0)
    assert torch.allclose(path[-1], z1)


# ---------------------------------------------------------------------------
# SphericalInterpolator tests
# ---------------------------------------------------------------------------


def test_slerp_t0_direction(slerp_interp, z_pair):
    """At t=0, SLERP result should point in the direction of z0."""
    z0, z1 = z_pair
    result = slerp_interp.interpolate(z0, z1, 0.0)
    # Normalise both and check they are parallel (cosine sim ≈ 1)
    eps = 1e-6
    cos_sim = torch.dot(result / (result.norm() + eps), z0 / (z0.norm() + eps))
    assert cos_sim.item() > 0.999


def test_slerp_t05_equidistant(slerp_interp, z_pair):
    """At t=0.5, the SLERP midpoint should be equidistant from z0 and z1 on the sphere."""
    z0, z1 = z_pair
    # Normalise inputs to unit sphere for the distance check
    z0_n = z0 / z0.norm()
    z1_n = z1 / z1.norm()
    mid = slerp_interp.interpolate(z0_n, z1_n, 0.5)
    mid_n = mid / (mid.norm() + 1e-9)

    angle_to_z0 = torch.acos(torch.clamp(torch.dot(mid_n, z0_n), -1.0, 1.0))
    angle_to_z1 = torch.acos(torch.clamp(torch.dot(mid_n, z1_n), -1.0, 1.0))
    assert abs(angle_to_z0.item() - angle_to_z1.item()) < 1e-4


def test_slerp_path_shape(slerp_interp, z_pair):
    """SLERP path() should return (n_steps, d)."""
    z0, z1 = z_pair
    path = slerp_interp.path(z0, z1, N_STEPS)
    assert path.shape == (N_STEPS, D_MODEL)


def test_slerp_parallel_vectors(slerp_interp):
    """Parallel (identical) vectors should fall back gracefully to lerp."""
    z = torch.randn(D_MODEL)
    result = slerp_interp.interpolate(z, z.clone(), 0.5)
    # Result should be close to z (same direction)
    cos_sim = torch.dot(result / (result.norm() + 1e-9), z / (z.norm() + 1e-9))
    assert cos_sim.item() > 0.999


# ---------------------------------------------------------------------------
# ManifoldInterpolator tests
# ---------------------------------------------------------------------------


def test_manifold_output_shape_1d(manifold_interp, z_pair):
    """ManifoldInterpolator should preserve (d_model,) shape for 1-D inputs."""
    z0, z1 = z_pair
    out = manifold_interp.interpolate(z0, z1, 0.5)
    assert out.shape == (D_MODEL,)


def test_manifold_output_shape_batched(manifold_interp):
    """ManifoldInterpolator should preserve (B, d_model) shape."""
    torch.manual_seed(1)
    B = 4
    z0 = torch.randn(B, D_MODEL)
    z1 = torch.randn(B, D_MODEL)
    out = manifold_interp.interpolate(z0, z1, 0.5)
    assert out.shape == (B, D_MODEL)


def test_manifold_gradients_flow(manifold_interp, z_pair):
    """Gradients should propagate through ManifoldInterpolator."""
    z0, z1 = z_pair
    z0_req = z0.detach().requires_grad_(True)
    z1_req = z1.detach().requires_grad_(True)
    out = manifold_interp.interpolate(z0_req, z1_req, 0.5)
    loss = out.sum()
    loss.backward()
    assert z0_req.grad is not None
    assert z1_req.grad is not None


def test_manifold_path_shape(manifold_interp, z_pair):
    """ManifoldInterpolator.path() should return (n_steps, d_model)."""
    z0, z1 = z_pair
    path = manifold_interp.path(z0, z1, N_STEPS)
    assert path.shape == (N_STEPS, D_MODEL)


# ---------------------------------------------------------------------------
# InterpolationAnalyzer tests
# ---------------------------------------------------------------------------


def test_analyzer_path_length_positive(analyzer, linear_interp, z_pair):
    """Path length should be > 0 for two distinct vectors."""
    z0, z1 = z_pair
    path = linear_interp.path(z0, z1, N_STEPS)
    length = analyzer.path_length(path)
    assert length > 0.0


def test_analyzer_smoothness_linear_path(analyzer, linear_interp, z_pair):
    """A linear path should have smoothness == 1.0 (all directions aligned)."""
    z0, z1 = z_pair
    path = linear_interp.path(z0, z1, N_STEPS)
    s = analyzer.smoothness(path)
    assert abs(s - 1.0) < 1e-5


def test_analyzer_midpoint_deviation_linear_path(analyzer, linear_interp, z_pair):
    """Midpoint deviation should be 0 for a linear path."""
    z0, z1 = z_pair
    path = linear_interp.path(z0, z1, N_STEPS)
    dev = analyzer.midpoint_deviation(path, z0, z1)
    assert abs(dev) < 1e-5
