"""Tests for answer selection helpers."""

import pytest
import torch

from src.alignment.answer_selection import (
    CandidateAnswer,
    answer_selection_probs,
    reward_margin,
    select_best_answer,
    top_k_answers,
)


def make_candidates():
    return [
        CandidateAnswer("a", reward=0.8, length_penalty=0.1),
        CandidateAnswer("b", reward=0.9, length_penalty=0.3),
        CandidateAnswer("c", reward=0.7, length_penalty=0.0),
    ]


def test_select_best_answer_uses_adjusted_score():
    best = select_best_answer(make_candidates())
    assert best.text == "a"


def test_top_k_answers_returns_sorted_subset():
    top = top_k_answers(make_candidates(), 2)
    assert [candidate.text for candidate in top] == ["a", "c"]


def test_answer_selection_probs_sum_to_one():
    probs = answer_selection_probs(torch.tensor([1.0, 2.0, 3.0]))
    assert probs.sum().item() == pytest.approx(1.0)


def test_reward_margin_computes_difference():
    candidates = top_k_answers(make_candidates(), 2)
    assert reward_margin(candidates[0], candidates[1]) == pytest.approx(0.0)


def test_select_best_answer_rejects_empty_input():
    with pytest.raises(ValueError):
        select_best_answer([])


def test_top_k_answers_rejects_bad_k():
    with pytest.raises(ValueError):
        top_k_answers(make_candidates(), -1)


def test_answer_selection_probs_rejects_bad_temperature():
    with pytest.raises(ValueError):
        answer_selection_probs(torch.tensor([1.0]), temperature=0.0)
