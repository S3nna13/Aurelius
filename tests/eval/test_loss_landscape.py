"""Tests for src/eval/loss_landscape.py.

Import via the stable `aurelius.*` namespace as required by the project spec.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from aurelius.eval.loss_landscape import (
    DirectionGenerator,
    LandscapeConfig,
    LandscapeEvaluator,
    LandscapeStats,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_model() -> nn.Module:
    """A tiny Linear+Linear model with known, easy-to-inspect parameters."""
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(8, 16, bias=True),
        nn.ReLU(),
        nn.Linear(16, 4, bias=True),
    )
    return model


@pytest.fixture()
def default_config() -> LandscapeConfig:
    return LandscapeConfig()


@pytest.fixture()
def direction_gen(small_model) -> DirectionGenerator:
    return DirectionGenerator(small_model)


@pytest.fixture()
def evaluator(small_model, default_config) -> LandscapeEvaluator:
    def loss_fn(model, x, y):
        pred = model(x)
        return nn.functional.mse_loss(pred, y)

    return LandscapeEvaluator(small_model, loss_fn, default_config)


@pytest.fixture()
def loss_inputs(small_model):
    torch.manual_seed(7)
    x = torch.randn(4, 8)
    y = torch.randn(4, 4)
    return (x, y)


# ---------------------------------------------------------------------------
# LandscapeConfig tests
# ---------------------------------------------------------------------------


def test_landscape_config_defaults():
    """LandscapeConfig should have documented default values."""
    cfg = LandscapeConfig()
    assert cfg.n_points == 11
    assert cfg.alpha_range == 1.0
    assert cfg.filter_normalize is True


def test_landscape_config_custom():
    """LandscapeConfig should accept custom values."""
    cfg = LandscapeConfig(n_points=5, alpha_range=0.5, filter_normalize=False)
    assert cfg.n_points == 5
    assert cfg.alpha_range == 0.5
    assert cfg.filter_normalize is False


# ---------------------------------------------------------------------------
# DirectionGenerator tests
# ---------------------------------------------------------------------------


def test_random_direction_shapes(direction_gen, small_model):
    """random_direction must return a list with one tensor per param, same shape."""
    direction = direction_gen.random_direction()
    params = list(small_model.parameters())
    assert len(direction) == len(params)
    for d, p in zip(direction, params):
        assert d.shape == p.shape, f"shape mismatch: {d.shape} vs {p.shape}"


def test_random_direction_is_random(direction_gen):
    """Two calls to random_direction should (almost certainly) differ."""
    d1 = direction_gen.random_direction()
    d2 = direction_gen.random_direction()
    # At least one tensor pair should differ
    any_different = any(not torch.allclose(a, b) for a, b in zip(d1, d2))
    assert any_different


def test_filter_normalize_direction_list_length(direction_gen):
    """filter_normalize_direction must return a list of the same length."""
    direction = direction_gen.random_direction()
    normed = direction_gen.filter_normalize_direction(direction)
    assert len(normed) == len(direction)


def test_filter_normalize_direction_norms(direction_gen, small_model):
    """After normalization, each filter norm should match the param filter norm."""
    direction = direction_gen.random_direction()
    normed = direction_gen.filter_normalize_direction(direction)

    for nd, p in zip(normed, small_model.parameters()):
        theta = p.data
        if theta.dim() <= 1:
            # Single filter — whole-tensor norms must match
            assert torch.isclose(nd.norm(), theta.norm(), atol=1e-5), (
                f"1-D norm mismatch: {nd.norm():.6f} vs {theta.norm():.6f}"
            )
        else:
            nd_2d = nd.view(nd.shape[0], -1)
            theta_2d = theta.view(theta.shape[0], -1)
            nd_norms = nd_2d.norm(dim=1)
            theta_norms = theta_2d.norm(dim=1)
            assert torch.allclose(nd_norms, theta_norms, atol=1e-5), (
                "filter-wise norms do not match after normalization"
            )


def test_filter_normalize_direction_shapes(direction_gen, small_model):
    """Shapes must be preserved after filter normalization."""
    direction = direction_gen.random_direction()
    normed = direction_gen.filter_normalize_direction(direction)
    params = list(small_model.parameters())
    for nd, p in zip(normed, params):
        assert nd.shape == p.shape


def test_pca_direction_shapes(direction_gen, small_model):
    """pca_direction must return tensors matching param shapes."""
    torch.manual_seed(0)
    # Create 4 snapshots from random params
    trajectory = []
    for _ in range(4):
        snap = [torch.randn_like(p.data) for p in small_model.parameters()]
        trajectory.append(snap)

    pca_dir = direction_gen.pca_direction(trajectory, component=0)
    params = list(small_model.parameters())
    assert len(pca_dir) == len(params)
    for d, p in zip(pca_dir, params):
        assert d.shape == p.shape


def test_pca_direction_component_selection(direction_gen, small_model):
    """Different components should (in general) give different directions."""
    torch.manual_seed(1)
    trajectory = [[torch.randn_like(p.data) for p in small_model.parameters()] for _ in range(6)]
    d0 = direction_gen.pca_direction(trajectory, component=0)
    d1 = direction_gen.pca_direction(trajectory, component=1)
    any_different = any(not torch.allclose(a, b) for a, b in zip(d0, d1))
    assert any_different


# ---------------------------------------------------------------------------
# LandscapeEvaluator._perturb tests
# ---------------------------------------------------------------------------


def test_perturb_alpha_zero(evaluator, small_model):
    """At alpha=0, _perturb must return base unchanged."""
    base = evaluator._get_params()
    direction = [torch.randn_like(p) for p in base]
    result = evaluator._perturb(base, direction, 0.0)
    for r, b in zip(result, base):
        assert torch.allclose(r, b)


def test_perturb_alpha_one(evaluator, small_model):
    """At alpha=1, _perturb must return base + direction."""
    base = evaluator._get_params()
    direction = [torch.randn_like(p) for p in base]
    result = evaluator._perturb(base, direction, 1.0)
    for r, b, d in zip(result, base, direction):
        assert torch.allclose(r, b + d)


# ---------------------------------------------------------------------------
# scan_1d tests
# ---------------------------------------------------------------------------


def test_scan_1d_returns_correct_keys(evaluator, small_model, loss_inputs):
    """scan_1d must return a dict with 'alphas' and 'losses' keys."""
    gen = DirectionGenerator(small_model)
    direction = gen.random_direction()
    result = evaluator.scan_1d(direction, loss_inputs)
    assert "alphas" in result
    assert "losses" in result


def test_scan_1d_n_points(evaluator, small_model, loss_inputs, default_config):
    """scan_1d must return exactly n_points losses."""
    gen = DirectionGenerator(small_model)
    direction = gen.random_direction()
    result = evaluator.scan_1d(direction, loss_inputs)
    assert result["losses"].shape == (default_config.n_points,)
    assert result["alphas"].shape == (default_config.n_points,)


def test_scan_1d_restores_params(evaluator, small_model, loss_inputs):
    """scan_1d must restore model parameters to their original values."""
    gen = DirectionGenerator(small_model)
    direction = gen.random_direction()

    # Save original params
    original = [p.data.clone() for p in small_model.parameters()]

    evaluator.scan_1d(direction, loss_inputs)

    # Params must be restored
    for orig, p in zip(original, small_model.parameters()):
        assert torch.allclose(orig, p.data), "scan_1d did not restore model params"


def test_scan_1d_center_point_is_zero(evaluator, small_model, loss_inputs, default_config):
    """The middle alpha of n_points should be 0 (for odd n_points)."""
    gen = DirectionGenerator(small_model)
    direction = gen.random_direction()
    result = evaluator.scan_1d(direction, loss_inputs)
    n = default_config.n_points
    center = n // 2
    assert abs(result["alphas"][center].item()) < 1e-6


# ---------------------------------------------------------------------------
# scan_2d tests
# ---------------------------------------------------------------------------


def test_scan_2d_losses_shape(evaluator, small_model, loss_inputs, default_config):
    """scan_2d losses must have shape (n_points, n_points)."""
    gen = DirectionGenerator(small_model)
    d1 = gen.random_direction()
    d2 = gen.random_direction()
    result = evaluator.scan_2d(d1, d2, loss_inputs)
    n = default_config.n_points
    assert result["losses"].shape == (n, n)


def test_scan_2d_returns_correct_keys(evaluator, small_model, loss_inputs):
    """scan_2d must return a dict with 'alphas', 'betas', and 'losses' keys."""
    gen = DirectionGenerator(small_model)
    d1 = gen.random_direction()
    d2 = gen.random_direction()
    result = evaluator.scan_2d(d1, d2, loss_inputs)
    assert "alphas" in result
    assert "betas" in result
    assert "losses" in result


def test_scan_2d_restores_params(evaluator, small_model, loss_inputs):
    """scan_2d must restore model parameters after the scan."""
    gen = DirectionGenerator(small_model)
    d1 = gen.random_direction()
    d2 = gen.random_direction()

    original = [p.data.clone() for p in small_model.parameters()]
    evaluator.scan_2d(d1, d2, loss_inputs)

    for orig, p in zip(original, small_model.parameters()):
        assert torch.allclose(orig, p.data), "scan_2d did not restore model params"


# ---------------------------------------------------------------------------
# LandscapeStats tests
# ---------------------------------------------------------------------------


def test_sharpness_non_negative():
    """sharpness must always be >= 0."""
    stats = LandscapeStats()
    losses = torch.tensor([2.0, 1.5, 1.0, 1.5, 2.0])
    s = stats.sharpness(losses, center_idx=2)
    assert s >= 0.0


def test_sharpness_flat_landscape():
    """sharpness of a flat loss landscape (all equal) is 0."""
    stats = LandscapeStats()
    losses = torch.ones(7)
    s = stats.sharpness(losses, center_idx=3)
    assert abs(s) < 1e-6


def test_sharpness_value():
    """sharpness should equal max-min of the provided losses."""
    stats = LandscapeStats()
    losses = torch.tensor([3.0, 1.0, 2.0, 4.0, 0.5])
    s = stats.sharpness(losses, center_idx=2)
    assert abs(s - (4.0 - 0.5)) < 1e-5


def test_curvature_at_center_convex():
    """A parabola centred at 0 should have positive curvature."""
    stats = LandscapeStats()
    # f(x) = x^2: f''(0) = 2
    alphas = torch.linspace(-1.0, 1.0, 11)
    losses = alphas**2
    c = stats.curvature_at_center(losses, alphas)
    assert c > 0.0


def test_is_convex_returns_bool():
    """is_convex must always return a Python bool."""
    stats = LandscapeStats()
    losses = torch.tensor([2.0, 1.0, 0.5, 1.0, 2.0])
    result = stats.is_convex(losses)
    assert isinstance(result, bool)


def test_is_convex_true_for_parabola():
    """A parabolic (U-shaped) loss should be convex."""
    stats = LandscapeStats()
    alphas = torch.linspace(-1.0, 1.0, 11)
    losses = alphas**2 + 1.0
    assert stats.is_convex(losses) is True


def test_is_convex_false_for_concave():
    """An inverted parabola (∩-shaped) should NOT be convex."""
    stats = LandscapeStats()
    alphas = torch.linspace(-1.0, 1.0, 11)
    losses = -(alphas**2) + 2.0
    assert stats.is_convex(losses) is False
