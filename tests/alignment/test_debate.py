"""Tests for debate alignment helpers."""

import pytest
import torch

from src.alignment.debate import (
    DebateTurn,
    debate_preference_loss,
    majority_verdict,
    rank_turns,
    self_consistency_score,
    stance_margin,
)


def make_turns() -> list[DebateTurn]:
    return [
        DebateTurn("a", "pro", "Argument one", 0.8),
        DebateTurn("b", "con", "Counter", 0.3),
        DebateTurn("c", "pro", "Argument two", 0.5),
    ]


def test_stance_margin_computes_difference():
    assert stance_margin(make_turns()) == pytest.approx(1.0)


def test_majority_verdict_returns_pro():
    assert majority_verdict(make_turns()) == "pro"


def test_majority_verdict_returns_tie_when_balanced():
    turns = [DebateTurn("a", "pro", "x", 1.0), DebateTurn("b", "con", "y", 1.0)]
    assert majority_verdict(turns) == "tie"


def test_self_consistency_score_measures_fraction():
    score = self_consistency_score(["pro", "pro", "con", "pro"])
    assert score == pytest.approx(0.75)


def test_debate_preference_loss_prefers_higher_pro_scores():
    better = debate_preference_loss(torch.tensor([1.0]), torch.tensor([0.0]))
    worse = debate_preference_loss(torch.tensor([0.0]), torch.tensor([1.0]))
    assert better.item() < worse.item()


def test_debate_preference_loss_backward():
    pro = torch.tensor([0.5, 0.7], requires_grad=True)
    con = torch.tensor([0.2, 0.9], requires_grad=True)
    loss = debate_preference_loss(pro, con)
    loss.backward()
    assert pro.grad is not None
    assert con.grad is not None


def test_rank_turns_sorts_descending():
    ranked = rank_turns(make_turns())
    assert [turn.score for turn in ranked] == [0.8, 0.5, 0.3]
