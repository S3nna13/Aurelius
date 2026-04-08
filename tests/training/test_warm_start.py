"""Tests for warm-start helpers."""

import pytest
import torch

from src.training.warm_start import (
    interpolation_warm_start,
    prefix_warm_start,
    warm_start_state,
)


def test_warm_start_state_loads_matching_keys():
    target = {"a": torch.zeros(2), "b": torch.ones(3)}
    source = {"a": torch.full((2,), 5.0), "b": torch.full((3,), 2.0)}
    updated, report = warm_start_state(target, source)
    assert torch.equal(updated["a"], source["a"])
    assert report.loaded_keys == ("a", "b")


def test_warm_start_state_skips_missing_keys():
    updated, report = warm_start_state({"a": torch.zeros(2)}, {})
    assert torch.equal(updated["a"], torch.zeros(2))
    assert report.missing_keys == ("a",)


def test_warm_start_state_skips_shape_mismatch():
    updated, report = warm_start_state({"a": torch.zeros(2)}, {"a": torch.zeros(3)})
    assert updated["a"].shape == (2,)
    assert report.shape_mismatch_keys == ("a",)


def test_interpolation_warm_start_blends_tensors():
    target = torch.zeros(2)
    source = torch.ones(2)
    blended = interpolation_warm_start(target, source, alpha=0.25)
    assert torch.allclose(blended, torch.full((2,), 0.25))


def test_prefix_warm_start_copies_matching_prefix():
    target = torch.zeros(5)
    source = torch.tensor([1.0, 2.0, 3.0])
    result = prefix_warm_start(target, source)
    assert torch.allclose(result[:3], source)
    assert torch.allclose(result[3:], torch.zeros(2))


def test_interpolation_warm_start_rejects_bad_alpha():
    with pytest.raises(ValueError):
        interpolation_warm_start(torch.zeros(1), torch.zeros(1), alpha=1.5)


def test_prefix_warm_start_rejects_bad_dim():
    with pytest.raises(ValueError):
        prefix_warm_start(torch.zeros(2, 2), torch.zeros(2, 2), dim=2)
