"""Tests for judge temperature helpers."""

import pytest
import torch

from src.eval.judge_temperature import (
    calibrated_top1,
    distribution_entropy,
    scale_logits,
    tempered_distribution,
)


def test_scale_logits_divides_by_temperature():
    logits = torch.tensor([2.0, 4.0])
    assert torch.allclose(scale_logits(logits, 2.0), torch.tensor([1.0, 2.0]))


def test_tempered_distribution_sums_to_one():
    probs = tempered_distribution(torch.randn(2, 4), temperature=0.7)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-6)


def test_distribution_entropy_increases_with_temperature():
    logits = torch.tensor([[3.0, 1.0, 0.5]])
    cold = distribution_entropy(logits, 0.5)
    hot = distribution_entropy(logits, 2.0)
    assert hot.item() > cold.item()


def test_calibrated_top1_matches_argmax_for_positive_temperature():
    logits = torch.tensor([[0.1, 0.9, 0.2]])
    assert torch.equal(calibrated_top1(logits, 1.0), torch.tensor([1]))


def test_scale_logits_rejects_bad_temperature():
    with pytest.raises(ValueError):
        scale_logits(torch.tensor([1.0]), 0.0)


def test_distribution_entropy_scalar_output():
    entropy = distribution_entropy(torch.randn(2, 3, 4), 1.0)
    assert entropy.ndim == 0


def test_tempered_distribution_preserves_shape():
    logits = torch.randn(2, 3, 4)
    assert tempered_distribution(logits).shape == logits.shape
