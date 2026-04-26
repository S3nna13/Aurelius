"""Tests for router temperature helpers."""

import pytest
import torch

from src.model.route_temperature import (
    apply_router_temperature,
    router_entropy,
    router_probabilities,
    sharpened_top1,
)


def test_apply_router_temperature_scales_logits():
    logits = torch.tensor([2.0, 4.0])
    scaled = apply_router_temperature(logits, 2.0)
    assert torch.allclose(scaled, torch.tensor([1.0, 2.0]))


def test_router_probabilities_sum_to_one():
    probs = router_probabilities(torch.randn(2, 4), temperature=0.7)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-6)


def test_router_entropy_higher_for_hotter_temperature():
    logits = torch.tensor([[3.0, 1.0, 0.5]])
    cold = router_entropy(logits, temperature=0.5)
    hot = router_entropy(logits, temperature=2.0)
    assert hot.item() > cold.item()


def test_sharpened_top1_matches_argmax():
    logits = torch.tensor([[0.1, 0.9, 0.3]])
    assert torch.equal(sharpened_top1(logits, temperature=1.0), torch.tensor([1]))


def test_apply_router_temperature_rejects_bad_temperature():
    with pytest.raises(ValueError):
        apply_router_temperature(torch.tensor([1.0]), 0.0)


def test_router_entropy_handles_batched_logits():
    entropy = router_entropy(torch.randn(2, 3, 4), temperature=1.0)
    assert entropy.ndim == 0


def test_sharpened_top1_respects_temperature_interface():
    logits = torch.tensor([[2.0, 1.0]])
    assert torch.equal(sharpened_top1(logits, temperature=0.5), torch.tensor([0]))
