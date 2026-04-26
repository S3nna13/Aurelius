"""Tests for judge margin helpers."""

import pytest
import torch

from src.eval.judge_margin import decisive_fraction, mean_margin, score_margin


def test_score_margin_elementwise_difference():
    margin = score_margin(torch.tensor([2.0, 1.0]), torch.tensor([1.0, 0.5]))
    assert torch.allclose(margin, torch.tensor([1.0, 0.5]))


def test_mean_margin_averages_examples():
    margin = mean_margin(torch.tensor([2.0, 1.0]), torch.tensor([1.0, 0.0]))
    assert margin.item() == pytest.approx(1.0)


def test_decisive_fraction_thresholds_absolute_margin():
    frac = decisive_fraction(torch.tensor([2.0, 1.0]), torch.tensor([1.7, 0.0]), threshold=0.5)
    assert frac.item() == pytest.approx(0.5)


def test_score_margin_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        score_margin(torch.tensor([1.0]), torch.tensor([1.0, 2.0]))


def test_mean_margin_scalar_output():
    margin = mean_margin(torch.tensor([1.0]), torch.tensor([0.0]))
    assert margin.ndim == 0


def test_decisive_fraction_zero_when_no_large_margins():
    frac = decisive_fraction(torch.tensor([1.0, 1.1]), torch.tensor([1.0, 1.0]), threshold=0.5)
    assert frac.item() == pytest.approx(0.0)


def test_decisive_fraction_one_when_all_large_margins():
    frac = decisive_fraction(torch.tensor([2.0, -1.0]), torch.tensor([0.0, 1.0]), threshold=0.5)
    assert frac.item() == pytest.approx(1.0)
