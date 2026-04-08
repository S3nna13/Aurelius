"""Tests for PairRM helpers."""

import pytest
import torch

from src.alignment.pairrm import (
    PairwisePreference,
    pairrm_accuracy,
    pairrm_loss,
    pairrm_margin_stats,
    rerank_candidates,
)


def test_pairrm_loss_prefers_higher_chosen_scores():
    better = pairrm_loss(torch.tensor([2.0]), torch.tensor([0.0]))
    worse = pairrm_loss(torch.tensor([0.0]), torch.tensor([2.0]))
    assert better.item() < worse.item()


def test_pairrm_accuracy_counts_correct_pairs():
    acc = pairrm_accuracy(torch.tensor([2.0, 0.0]), torch.tensor([1.0, 1.0]))
    assert acc.item() == pytest.approx(0.5)


def test_pairrm_margin_stats_summarizes_preferences():
    stats = pairrm_margin_stats([
        PairwisePreference(2.0, 1.0),
        PairwisePreference(3.0, 0.0),
    ])
    assert stats["mean_margin"] == pytest.approx(2.0)


def test_rerank_candidates_sorts_descending():
    order = rerank_candidates(torch.tensor([0.2, 0.9, 0.5]))
    assert torch.equal(order, torch.tensor([1, 2, 0]))


def test_pairrm_loss_backward():
    chosen = torch.tensor([0.5, 0.7], requires_grad=True)
    rejected = torch.tensor([0.1, 0.8], requires_grad=True)
    loss = pairrm_loss(chosen, rejected)
    loss.backward()
    assert chosen.grad is not None
    assert rejected.grad is not None


def test_pairrm_margin_stats_handles_empty_input():
    stats = pairrm_margin_stats([])
    assert stats["mean_margin"] == pytest.approx(0.0)


def test_pairrm_loss_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        pairrm_loss(torch.tensor([1.0]), torch.tensor([1.0, 2.0]))
