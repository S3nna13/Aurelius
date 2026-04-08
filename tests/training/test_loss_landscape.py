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
