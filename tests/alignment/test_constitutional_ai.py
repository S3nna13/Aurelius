"""Tests for Constitutional AI (Bai et al., 2022) RL-CAI stage.

Import path: aurelius.alignment.constitutional_ai
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
from aurelius.alignment.constitutional_ai import (
    CAILoss,
    CAITrainer,
    Constitution,
    ConstitutionalPrinciple,
    ConstitutionalScorer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_constitution(n: int = 3) -> Constitution:
    principles = [
        ConstitutionalPrinciple(
            name=f"p{i}",
            critique_prompt=f"Does this response satisfy criterion {i}?",
            weight=float(i + 1),
        )
        for i in range(n)
    ]
    return Constitution(principles)


class _DummyModel(nn.Module):
    """Minimal model with trainable params for CAITrainer tests."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ---------------------------------------------------------------------------
# 1. ConstitutionalPrinciple fields
# ---------------------------------------------------------------------------


def test_constitutional_principle_fields():
    p = ConstitutionalPrinciple(
        name="honesty",
        critique_prompt="Does this response avoid deception?",
        weight=2.5,
    )
    assert p.name == "honesty"
    assert p.critique_prompt == "Does this response avoid deception?"
    assert p.weight == 2.5


# ---------------------------------------------------------------------------
# 2. ConstitutionalPrinciple default weight
# ---------------------------------------------------------------------------


def test_constitutional_principle_default_weight():
    p = ConstitutionalPrinciple(name="safety", critique_prompt="Is it safe?")
    assert p.weight == 1.0


# ---------------------------------------------------------------------------
# 3. Constitution.get_principle returns correct principle
# ---------------------------------------------------------------------------


def test_constitution_get_principle():
    c = make_constitution(3)
    p = c.get_principle("p1")
    assert p.name == "p1"
    assert p.weight == 2.0


# ---------------------------------------------------------------------------
# 4. Constitution.get_principle raises KeyError for unknown name
# ---------------------------------------------------------------------------


def test_constitution_get_principle_keyerror():
    c = make_constitution(2)
    with pytest.raises(KeyError):
        c.get_principle("nonexistent")


# ---------------------------------------------------------------------------
# 5. Constitution.total_weight sums weights
# ---------------------------------------------------------------------------


def test_constitution_total_weight():
    c = make_constitution(4)
    # weights = 1, 2, 3, 4 -> total = 10
    assert c.total_weight() == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# 6. Constitution.weighted_score correct
# ---------------------------------------------------------------------------


def test_constitution_weighted_score():
    principles = [
        ConstitutionalPrinciple(name="a", critique_prompt="a?", weight=1.0),
        ConstitutionalPrinciple(name="b", critique_prompt="b?", weight=3.0),
    ]
    c = Constitution(principles)
    scores = {"a": 0.0, "b": 1.0}
    # weighted mean = (1*0 + 3*1) / (1+3) = 0.75
    result = c.weighted_score(scores)
    assert result == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# 7. Constitution.from_dicts constructs correctly
# ---------------------------------------------------------------------------


def test_constitution_from_dicts():
    dicts = [
        {"name": "helpfulness", "critique_prompt": "Is it helpful?", "weight": 2.0},
        {"name": "safety", "critique_prompt": "Is it safe?"},
    ]
    c = Constitution.from_dicts(dicts)
    assert len(c) == 2
    assert c.get_principle("helpfulness").weight == 2.0
    assert c.get_principle("safety").weight == 1.0  # default


# ---------------------------------------------------------------------------
# 8. ConstitutionalScorer.score_response shape (N,)
# ---------------------------------------------------------------------------


def test_scorer_score_response_shape():
    c = make_constitution(4)
    scorer = ConstitutionalScorer(c)
    log_probs = [torch.randn(10) for _ in range(4)]
    scores = scorer.score_response(log_probs)
    assert scores.shape == (4,)


# ---------------------------------------------------------------------------
# 9. ConstitutionalScorer.score_response values = mean log-prob per response
# ---------------------------------------------------------------------------


def test_scorer_score_response_values():
    c = make_constitution(3)
    scorer = ConstitutionalScorer(c)
    lp0 = torch.tensor([-1.0, -2.0, -3.0])
    lp1 = torch.tensor([-0.5, -0.5])
    lp2 = torch.tensor([-4.0])
    scores = scorer.score_response([lp0, lp1, lp2])
    assert scores[0].item() == pytest.approx(-2.0)
    assert scores[1].item() == pytest.approx(-0.5)
    assert scores[2].item() == pytest.approx(-4.0)


# ---------------------------------------------------------------------------
# 10. ConstitutionalScorer.aggregate_score returns scalar
# ---------------------------------------------------------------------------


def test_scorer_aggregate_score_scalar():
    c = make_constitution(3)
    scorer = ConstitutionalScorer(c)
    principle_scores = torch.tensor([-1.0, -2.0, -3.0])
    agg = scorer.aggregate_score(principle_scores)
    assert agg.ndim == 0
    assert agg.item() == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# 11. ConstitutionalScorer.aggregate_score with weights
# ---------------------------------------------------------------------------


def test_scorer_aggregate_score_weighted():
    c = make_constitution(2)
    scorer = ConstitutionalScorer(c)
    principle_scores = torch.tensor([0.0, 1.0])
    weights = torch.tensor([1.0, 3.0])
    agg = scorer.aggregate_score(principle_scores, weights=weights)
    assert agg.item() == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# 12. ConstitutionalScorer.rank_responses descending order
# ---------------------------------------------------------------------------


def test_scorer_rank_responses_descending():
    c = make_constitution(2)
    scorer = ConstitutionalScorer(c)
    scores = torch.tensor([0.3, 0.9, 0.1, 0.7])
    ranks = scorer.rank_responses(scores)
    # Expected order: 1 (0.9), 3 (0.7), 0 (0.3), 2 (0.1)
    assert ranks.tolist() == [1, 3, 0, 2]


# ---------------------------------------------------------------------------
# 13. CAILoss returns scalar + correct keys
# ---------------------------------------------------------------------------


def test_cai_loss_returns_scalar_and_keys():
    loss_fn = CAILoss(beta=0.1)
    B = 4
    chosen = torch.randn(B)
    rejected = torch.randn(B)
    ref_chosen = torch.randn(B)
    ref_rejected = torch.randn(B)
    loss, metrics = loss_fn(chosen, rejected, ref_chosen, ref_rejected)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    for key in ("loss", "accuracy", "margin"):
        assert key in metrics, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# 14. CAILoss perfect separation -> accuracy = 1.0
# ---------------------------------------------------------------------------


def test_cai_loss_perfect_separation_accuracy():
    loss_fn = CAILoss(beta=1.0)
    B = 8
    # chosen has much higher log-probs than rejected, refs are equal
    chosen = torch.full((B,), 10.0)
    rejected = torch.full((B,), -10.0)
    ref_chosen = torch.zeros(B)
    ref_rejected = torch.zeros(B)
    _, metrics = loss_fn(chosen, rejected, ref_chosen, ref_rejected)
    assert metrics["accuracy"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 15. CAILoss gradient flows
# ---------------------------------------------------------------------------


def test_cai_loss_gradient_flows():
    loss_fn = CAILoss(beta=0.1)
    B = 4
    chosen = torch.randn(B, requires_grad=True)
    rejected = torch.randn(B, requires_grad=True)
    ref_chosen = torch.randn(B)
    ref_rejected = torch.randn(B)
    loss, _ = loss_fn(chosen, rejected, ref_chosen, ref_rejected)
    loss.backward()
    assert chosen.grad is not None
    assert rejected.grad is not None
    assert chosen.grad.shape == (B,)


# ---------------------------------------------------------------------------
# 16. CAITrainer.freeze_ref freezes all ref params
# ---------------------------------------------------------------------------


def test_cai_trainer_freeze_ref():
    c = make_constitution(2)
    model = _DummyModel()
    ref_model = copy.deepcopy(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = CAILoss()
    trainer = CAITrainer(model, ref_model, optimizer, c, loss_fn)
    for param in trainer.ref_model.parameters():
        assert not param.requires_grad, "ref_model param should be frozen"


# ---------------------------------------------------------------------------
# 17. CAITrainer.train_step returns correct keys
# ---------------------------------------------------------------------------


def test_cai_trainer_train_step_keys():
    c = make_constitution(2)
    model = _DummyModel()
    ref_model = copy.deepcopy(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = CAILoss()
    trainer = CAITrainer(model, ref_model, optimizer, c, loss_fn)

    B = 4
    chosen_lp = torch.randn(B, requires_grad=True)
    rejected_lp = torch.randn(B)
    ref_chosen = torch.randn(B)
    ref_rejected = torch.randn(B)
    metrics = trainer.train_step(chosen_lp, rejected_lp, ref_chosen, ref_rejected)
    for key in ("loss", "accuracy", "margin"):
        assert key in metrics, f"Missing key: {key}"
