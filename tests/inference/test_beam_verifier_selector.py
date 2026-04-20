"""Tests for beam_verifier_selector."""

from __future__ import annotations

import pytest
import torch

from src.inference.beam_verifier_selector import BeamVerifierSelector


def test_select_best_1d():
    s = torch.tensor([0.1, 0.5, 0.2])
    idx = BeamVerifierSelector.select_best(s)
    assert idx.item() == 1


def test_select_best_2d():
    s = torch.tensor([[1.0, 2.0], [5.0, 3.0]])
    idx = BeamVerifierSelector.select_best(s)
    assert list(idx.tolist()) == [1, 0]


def test_topk_1d():
    s = torch.tensor([0.2, 0.9, 0.4, 0.7])
    idx = BeamVerifierSelector.select_topk(s, k=2)
    assert set(idx.tolist()) == {1, 3}


def test_topk_2d():
    s = torch.randn(3, 5)
    idx = BeamVerifierSelector.select_topk(s, k=2)
    assert idx.shape == (3, 2)


def test_rejects_bad_rank():
    with pytest.raises(ValueError):
        BeamVerifierSelector.select_best(torch.randn(2, 3, 4))


def test_rejects_nan():
    with pytest.raises(ValueError):
        BeamVerifierSelector.select_best(torch.tensor([1.0, float("nan")]))


def test_topk_k_too_large():
    with pytest.raises(ValueError):
        BeamVerifierSelector.select_topk(torch.tensor([1.0, 2.0]), k=5)


def test_topk_bad_k():
    with pytest.raises(ValueError):
        BeamVerifierSelector.select_topk(torch.tensor([1.0]), k=0)


def test_determinism():
    torch.manual_seed(2)
    s = torch.randn(4)
    a = BeamVerifierSelector.select_best(s)
    torch.manual_seed(2)
    s2 = torch.randn(4)
    b = BeamVerifierSelector.select_best(s2)
    assert a.item() == b.item()
