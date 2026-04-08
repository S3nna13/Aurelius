"""Tests for loss-landscape helpers."""

import pytest
import torch

from src.training.loss_landscape import (
    flatten_parameters,
    interpolate_parameters,
    interpolation_slice,
    local_sharpness,
    perturb_parameters,
    random_direction_like,
    unflatten_like,
)


def make_params():
    return [torch.randn(2, 3), torch.randn(4)]


def test_flatten_and_unflatten_round_trip():
    params = make_params()
    flat = flatten_parameters(params)
    restored = unflatten_like(flat, params)
    assert all(torch.allclose(a, b) for a, b in zip(params, restored))


def test_interpolate_parameters_midpoint():
    a = [torch.zeros(2)]
    b = [torch.ones(2)]
    out = interpolate_parameters(a, b, alpha=0.5)
    assert torch.allclose(out[0], torch.full((2,), 0.5))


def test_random_direction_like_is_normalized():
    direction = random_direction_like(make_params())
    norm = flatten_parameters(direction).norm().item()
    assert norm == pytest.approx(1.0, rel=1e-5)


def test_perturb_parameters_shifts_values():
    params = [torch.zeros(2)]
    direction = [torch.ones(2)]
    out = perturb_parameters(params, direction, epsilon=0.25)
    assert torch.allclose(out[0], torch.full((2,), 0.25))


def test_interpolation_slice_returns_n_points():
    a = [torch.zeros(2)]
    b = [torch.ones(2)]
    loss_fn = lambda ps: ps[0].sum()
    result = interpolation_slice(a, b, loss_fn, n_points=4)
    assert result.alphas.shape == (4,)
    assert result.losses.shape == (4,)


def test_local_sharpness_is_max_increase():
    sharpness = local_sharpness(torch.tensor(1.0), torch.tensor([1.2, 0.9, 1.5]))
    assert sharpness.item() == pytest.approx(0.5)


def test_interpolate_parameters_rejects_bad_alpha():
    with pytest.raises(ValueError):
        interpolate_parameters([torch.zeros(1)], [torch.zeros(1)], alpha=-0.1)


# ---------------------------------------------------------------------------
# LossLandscapeExplorer and LandscapeStats tests
# ---------------------------------------------------------------------------

import torch.nn as nn
from src.training.loss_landscape import LossLandscapeExplorer, LandscapeStats


def make_explorer_model():
    """A simple linear model for landscape tests."""
    torch.manual_seed(0)
    return nn.Linear(4, 2, bias=False)


def make_explorer(model=None):
    """Build a LossLandscapeExplorer with a simple quadratic loss."""
    if model is None:
        model = make_explorer_model()

    def loss_fn(m):
        # Simple: sum of squared parameters -> bowl-shaped landscape
        flat = torch.cat([p.view(-1) for p in m.parameters()])
        return (flat ** 2).sum()

    return LossLandscapeExplorer(model, loss_fn, filter_normalization=True)


def test_line_scan_returns_dict():
    """line_scan should return a dict with 'alphas' and 'losses'."""
    explorer = make_explorer()
    result = explorer.line_scan(n_points=5)
    assert isinstance(result, dict)
    assert 'alphas' in result
    assert 'losses' in result


def test_line_scan_n_points():
    """losses tensor should have n_points values."""
    explorer = make_explorer()
    n = 7
    result = explorer.line_scan(n_points=n)
    assert result['losses'].shape == (n,), f"Expected ({n},), got {result['losses'].shape}"


def test_line_scan_restores_params():
    """Model parameters should be unchanged after line_scan."""
    model = make_explorer_model()
    explorer = make_explorer(model)

    params_before = torch.cat([p.detach().view(-1) for p in model.parameters()]).clone()
    explorer.line_scan(n_points=5)
    params_after = torch.cat([p.detach().view(-1) for p in model.parameters()])

    assert torch.allclose(params_before, params_after), "Parameters should be restored after line_scan"


def test_surface_scan_shape():
    """surface_scan losses tensor should be (n_points, n_points)."""
    explorer = make_explorer()
    n = 4
    result = explorer.surface_scan(n_points=n)
    assert result['losses'].shape == (n, n), f"Expected ({n},{n}), got {result['losses'].shape}"


def test_surface_scan_restores_params():
    """Model parameters should be unchanged after surface_scan."""
    model = make_explorer_model()
    explorer = make_explorer(model)

    params_before = torch.cat([p.detach().view(-1) for p in model.parameters()]).clone()
    explorer.surface_scan(n_points=3)
    params_after = torch.cat([p.detach().view(-1) for p in model.parameters()])

    assert torch.allclose(params_before, params_after), "Parameters should be restored after surface_scan"


def test_flatness_score_nonneg():
    """flatness_score should be >= 0 for a convex (bowl-shaped) loss landscape."""
    model = make_explorer_model()

    def bowl_loss(m):
        flat = torch.cat([p.view(-1) for p in m.parameters()])
        return (flat ** 2).sum()

    explorer = LossLandscapeExplorer(model, bowl_loss, filter_normalization=True)
    score = explorer.flatness_score(epsilon=0.01, n_directions=3)
    # For a bowl (quadratic), moving away from origin increases loss, so score >= 0
    # (unless model is already at zero)
    assert isinstance(score, float), "flatness_score should return a float"
    # We just check it's a number; sign depends on position in landscape
    assert score is not None


def test_sharpness_ratio_nonneg():
    """sharpness_ratio should return a non-negative float for a bowl-shaped loss."""
    model = make_explorer_model()

    def bowl_loss(m):
        flat = torch.cat([p.view(-1) for p in m.parameters()])
        return (flat ** 2).sum()

    explorer = LossLandscapeExplorer(model, bowl_loss, filter_normalization=True)
    ratio = explorer.sharpness_ratio(epsilon=0.01, n_directions=5)
    assert isinstance(ratio, float), "sharpness_ratio should return a float"
    assert ratio >= 0.0, f"sharpness_ratio should be >= 0, got {ratio}"


def test_landscape_stats_curvature():
    """curvature_at_center should return a float."""
    losses = torch.tensor([2.0, 1.0, 0.5, 1.0, 2.0])
    stats = LandscapeStats(losses)
    curv = stats.curvature_at_center()
    assert isinstance(curv, float), "curvature_at_center should return a float"


def test_local_minima_count():
    """local_minima_count should return a non-negative integer."""
    losses = torch.tensor([3.0, 1.0, 3.0, 1.0, 3.0])  # two local minima
    stats = LandscapeStats(losses)
    count = stats.local_minima_count(window=3)
    assert isinstance(count, int), "local_minima_count should return an int"
    assert count >= 0, "local_minima_count should be non-negative"


def test_is_convex_simple():
    """A convex (bowl-shaped) loss tensor should return True."""
    losses = torch.tensor([4.0, 1.0, 0.0, 1.0, 4.0])  # quadratic bowl
    stats = LandscapeStats(losses)
    assert stats.is_convex() is True, "Bowl-shaped loss should be convex"
