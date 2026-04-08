"""Tests for evidence ranking helpers."""

import pytest
import torch

from src.eval.evidence_ranking import EvidenceItem, mean_reciprocal_rank, rank_evidence, topk_precision


def make_items():
    return [
        EvidenceItem("a", score=0.9, relevant=True),
        EvidenceItem("b", score=0.8, relevant=False),
        EvidenceItem("c", score=0.7, relevant=True),
    ]


def test_rank_evidence_sorts_descending():
    order = rank_evidence(torch.tensor([0.2, 0.9, 0.5]))
    assert torch.equal(order, torch.tensor([1, 2, 0]))


def test_topk_precision_computes_fraction():
    assert topk_precision(make_items(), k=2) == pytest.approx(0.5)


def test_mean_reciprocal_rank_returns_first_relevant_rank():
    assert mean_reciprocal_rank(make_items()) == pytest.approx(1.0)


def test_mean_reciprocal_rank_zero_when_no_relevant_items():
    items = [EvidenceItem("a", 0.9, False)]
    assert mean_reciprocal_rank(items) == pytest.approx(0.0)


def test_topk_precision_handles_empty_items():
    assert topk_precision([], k=1) == pytest.approx(0.0)


def test_rank_evidence_rejects_bad_rank():
    with pytest.raises(ValueError):
        rank_evidence(torch.tensor([[0.1, 0.2]]))


def test_topk_precision_rejects_bad_k():
    with pytest.raises(ValueError):
        topk_precision(make_items(), k=0)
