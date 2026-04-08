"""Tests for token replay helpers."""

import pytest
import torch

from src.training.token_replay import (
    TokenReplayRecord,
    replay_distribution,
    replay_weight,
    select_replay_tokens,
    update_replay_counts,
)


def make_records():
    return [
        TokenReplayRecord(1, loss=0.2, count=5),
        TokenReplayRecord(2, loss=1.0, count=0),
        TokenReplayRecord(3, loss=0.5, count=1),
    ]


def test_replay_weight_prefers_high_loss_low_count():
    weights = replay_weight(torch.tensor([0.2, 1.0]), torch.tensor([5.0, 0.0]))
    assert weights[1].item() > weights[0].item()


def test_replay_distribution_sums_to_one():
    probs = replay_distribution(make_records())
    assert probs.sum().item() == pytest.approx(1.0)


def test_select_replay_tokens_returns_top_ids():
    assert select_replay_tokens(make_records(), 2) == [2, 3]


def test_update_replay_counts_increments_selected_records():
    records = make_records()
    update_replay_counts(records, [2, 3])
    assert records[1].count == 1
    assert records[2].count == 2


def test_replay_distribution_handles_empty_input():
    assert replay_distribution([]).numel() == 0


def test_select_replay_tokens_rejects_bad_k():
    with pytest.raises(ValueError):
        select_replay_tokens(make_records(), -1)


def test_replay_weight_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        replay_weight(torch.tensor([1.0]), torch.tensor([1.0, 2.0]))
