"""Tests for Medusa verification helpers."""

import pytest
import torch

from src.inference.medusa_verification import (
    acceptance_rate,
    truncate_after_rejection,
    verify_draft_tokens,
)


def test_verify_draft_tokens_accepts_full_match():
    result = verify_draft_tokens(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3]))
    assert result.accepted_length == 3
    assert result.rejected_at is None


def test_verify_draft_tokens_stops_on_first_mismatch():
    result = verify_draft_tokens(torch.tensor([1, 2, 9]), torch.tensor([1, 2, 3]))
    assert result.accepted_length == 2
    assert result.rejected_at == 2


def test_truncate_after_rejection_uses_accepted_prefix():
    result = verify_draft_tokens(torch.tensor([1, 5]), torch.tensor([1, 2]))
    kept = truncate_after_rejection(torch.tensor([1, 2]), result)
    assert torch.equal(kept, torch.tensor([1]))


def test_acceptance_rate_averages_results():
    results = [
        verify_draft_tokens(torch.tensor([1, 2]), torch.tensor([1, 2])),
        verify_draft_tokens(torch.tensor([1, 3]), torch.tensor([1, 2])),
    ]
    rate = acceptance_rate(results)
    assert rate > 0.5


def test_verify_draft_tokens_rejects_bad_rank():
    with pytest.raises(ValueError):
        verify_draft_tokens(torch.tensor([[1, 2]]), torch.tensor([1, 2]))


def test_truncate_after_rejection_rejects_bad_rank():
    result = verify_draft_tokens(torch.tensor([1]), torch.tensor([1]))
    with pytest.raises(ValueError):
        truncate_after_rejection(torch.tensor([[1]]), result)


def test_acceptance_rate_is_zero_for_empty_results():
    assert acceptance_rate([]) == pytest.approx(0.0)
