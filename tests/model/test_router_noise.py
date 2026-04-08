"""Tests for noisy router helpers."""

import pytest
import torch

from src.model.router_noise import (
    add_gumbel_noise,
    noisy_router_probs,
    noisy_topk_routing,
    routing_mask,
)


def test_add_gumbel_noise_preserves_shape():
    logits = torch.randn(2, 3, 4)
    out = add_gumbel_noise(logits)
    assert out.shape == logits.shape


def test_noisy_topk_routing_returns_indices_and_values():
    logits = torch.tensor([[1.0, 3.0, 2.0]])
    indices, values = noisy_topk_routing(logits, k=2)
    assert indices.shape == (1, 2)
    assert values.shape == (1, 2)


def test_noisy_topk_routing_matches_deterministic_topk_without_noise():
    logits = torch.tensor([[1.0, 3.0, 2.0]])
    indices, _ = noisy_topk_routing(logits, k=2, noise_std=0.0)
    assert torch.equal(indices, torch.tensor([[1, 2]]))


def test_routing_mask_marks_selected_experts():
    mask = routing_mask(torch.tensor([[1, 3]]), n_experts=4)
    assert torch.equal(mask, torch.tensor([[False, True, False, True]]))


def test_noisy_router_probs_sum_to_one():
    probs = noisy_router_probs(torch.randn(2, 4), noise_std=0.1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-6)


def test_noisy_topk_routing_rejects_bad_k():
    with pytest.raises(ValueError):
        noisy_topk_routing(torch.randn(2, 3), k=0)


def test_routing_mask_rejects_bad_indices():
    with pytest.raises(ValueError):
        routing_mask(torch.tensor([[4]]), n_experts=4)
