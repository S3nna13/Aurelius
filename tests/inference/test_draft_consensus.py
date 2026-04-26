"""Tests for draft consensus helpers."""

import pytest
import torch

from src.inference.draft_consensus import consensus_rate, consensus_token, majority_token


def test_majority_token_returns_mode():
    assert majority_token(torch.tensor([1, 2, 2])) == 2


def test_consensus_token_returns_token_when_unanimous():
    assert consensus_token(torch.tensor([3, 3, 3])) == 3


def test_consensus_token_returns_none_when_not_unanimous():
    assert consensus_token(torch.tensor([3, 4, 3])) is None


def test_consensus_rate_counts_unanimous_columns():
    proposals = torch.tensor([[1, 2, 3], [1, 2, 4], [1, 2, 5]])
    assert consensus_rate(proposals).item() == pytest.approx(2.0 / 3.0)


def test_majority_token_rejects_bad_rank():
    with pytest.raises(ValueError):
        majority_token(torch.tensor([[1, 2]]))


def test_consensus_token_rejects_empty_input():
    with pytest.raises(ValueError):
        consensus_token(torch.tensor([], dtype=torch.long))


def test_consensus_rate_handles_empty_dimensions():
    assert consensus_rate(torch.empty(0, 0, dtype=torch.long)).item() == pytest.approx(0.0)
