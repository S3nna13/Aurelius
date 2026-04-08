"""Tests for data echoing helpers."""

import pytest
import torch

from src.training.data_echoing import (
    EchoExample,
    echo_probabilities,
    echo_score,
    select_echoes,
    update_echo_metadata,
)


def make_examples() -> list[EchoExample]:
    return [
        EchoExample("easy", loss=0.2, difficulty=0.1, last_seen_step=9),
        EchoExample("hard", loss=1.2, difficulty=0.9, last_seen_step=2),
        EchoExample("medium", loss=0.6, difficulty=0.5, last_seen_step=5),
    ]


def test_echo_score_increases_with_loss_and_difficulty():
    low = echo_score(torch.tensor([0.2]), torch.tensor([0.1]), torch.tensor([1.0]))
    high = echo_score(torch.tensor([1.0]), torch.tensor([0.8]), torch.tensor([1.0]))
    assert high.item() > low.item()


def test_echo_probabilities_sum_to_one():
    probs = echo_probabilities(make_examples(), current_step=10)
    assert probs.sum().item() == pytest.approx(1.0)


def test_echo_probabilities_prioritize_harder_examples():
    examples = make_examples()
    probs = echo_probabilities(examples, current_step=10)
    assert probs[1].item() == pytest.approx(probs.max().item())


def test_select_echoes_returns_top_examples():
    selected = select_echoes(make_examples(), current_step=10, n_select=2)
    assert [example.example_id for example in selected] == ["hard", "medium"]


def test_update_echo_metadata_increments_seen_examples():
    examples = make_examples()
    update_echo_metadata(examples, ["hard"], current_step=11)
    hard = next(example for example in examples if example.example_id == "hard")
    assert hard.echo_count == 1
    assert hard.last_seen_step == 11


def test_select_echoes_handles_zero_selection():
    assert select_echoes(make_examples(), current_step=10, n_select=0) == []


def test_echo_probabilities_reject_bad_temperature():
    with pytest.raises(ValueError):
        echo_probabilities(make_examples(), current_step=10, temperature=0.0)
