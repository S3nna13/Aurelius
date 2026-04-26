"""Tests for route-dropout helpers."""

import pytest
import torch

from src.model.route_dropout import apply_route_dropout, route_dropout_mask, route_survival_rate


def test_route_dropout_mask_shape():
    mask = route_dropout_mask((2, 3, 4), drop_prob=0.2)
    assert mask.shape == (2, 3, 4)


def test_apply_route_dropout_preserves_simplex():
    probs = torch.full((2, 3, 4), 0.25)
    out = apply_route_dropout(probs, drop_prob=0.2)
    assert torch.allclose(out.sum(dim=-1), torch.ones(2, 3), atol=1e-6)


def test_apply_route_dropout_fallbacks_when_all_dropped():
    probs = torch.full((1, 1, 3), 1.0 / 3.0)
    out = apply_route_dropout(probs, drop_prob=0.999999)
    assert torch.allclose(out.sum(dim=-1), torch.ones(1, 1), atol=1e-6)


def test_route_survival_rate_in_unit_interval():
    mask = route_dropout_mask((100,), drop_prob=0.5)
    rate = route_survival_rate(mask)
    assert 0.0 <= rate.item() <= 1.0


def test_route_dropout_rejects_bad_prob():
    with pytest.raises(ValueError):
        route_dropout_mask((1,), drop_prob=1.0)


def test_apply_route_dropout_changes_distribution():
    probs = torch.tensor([[[0.7, 0.2, 0.1]]])
    out = apply_route_dropout(probs, drop_prob=0.5)
    assert out.shape == probs.shape


def test_route_survival_rate_scalar_output():
    rate = route_survival_rate(torch.tensor([True, False, True]))
    assert rate.ndim == 0
