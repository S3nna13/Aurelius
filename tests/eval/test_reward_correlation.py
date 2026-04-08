"""Tests for reward correlation helpers."""

import pytest
import torch

from src.eval.reward_correlation import (
    pearson_correlation,
    rankdata,
    reward_agreement,
    spearman_correlation,
)


def test_pearson_correlation_one_for_identical_vectors():
    x = torch.tensor([1.0, 2.0, 3.0])
    assert pearson_correlation(x, x).item() == pytest.approx(1.0)


def test_pearson_correlation_zero_for_constant_vector():
    corr = pearson_correlation(torch.tensor([1.0, 1.0]), torch.tensor([1.0, 2.0]))
    assert corr.item() == pytest.approx(0.0)


def test_rankdata_assigns_ordinal_ranks():
    ranks = rankdata(torch.tensor([0.3, 0.1, 0.2]))
    assert torch.equal(ranks, torch.tensor([3.0, 1.0, 2.0]))


def test_spearman_correlation_one_for_same_ordering():
    x = torch.tensor([1.0, 3.0, 2.0])
    y = torch.tensor([10.0, 30.0, 20.0])
    assert spearman_correlation(x, y).item() == pytest.approx(1.0)


def test_reward_agreement_measures_sign_matches():
    agreement = reward_agreement(torch.tensor([1.0, -1.0]), torch.tensor([0.5, -0.5]))
    assert agreement.item() == pytest.approx(1.0)


def test_reward_agreement_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        reward_agreement(torch.tensor([1.0]), torch.tensor([1.0, 2.0]))


def test_rankdata_rejects_bad_rank():
    with pytest.raises(ValueError):
        rankdata(torch.tensor([[1.0, 2.0]]))
