"""Tests for src/alignment/hierarchical_reward.py"""
from __future__ import annotations

import torch
import pytest

from aurelius.alignment.hierarchical_reward import (
    RewardCriterion,
    CriterionWeights,
    HierarchicalRewardModel,
    HierarchicalRewardLoss,
    MultiObjectiveRewardOptimizer,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

D_MODEL = 16
N_CRITERIA = 3
B = 4


def make_criteria(d_model: int = D_MODEL, n: int = N_CRITERIA) -> list:
    names = ["helpfulness", "harmlessness", "honesty"][:n]
    return [RewardCriterion(name=names[i], d_model=d_model) for i in range(n)]


def make_model(learnable: bool = False) -> HierarchicalRewardModel:
    criteria = make_criteria()
    weights = CriterionWeights(n_criteria=N_CRITERIA, learnable=learnable)
    return HierarchicalRewardModel(d_model=D_MODEL, criteria=criteria, weight_scheme=weights)


def make_hidden(B: int = B, d_model: int = D_MODEL) -> torch.Tensor:
    return torch.randn(B, d_model)


# ---------------------------------------------------------------------------
# 1. RewardCriterion output shape (B,)
# ---------------------------------------------------------------------------

def test_reward_criterion_output_shape():
    criterion = RewardCriterion(name="helpfulness", d_model=D_MODEL)
    hidden = make_hidden()
    out = criterion(hidden)
    assert out.shape == (B,), f"Expected ({B},), got {out.shape}"


# ---------------------------------------------------------------------------
# 2. RewardCriterion output is finite
# ---------------------------------------------------------------------------

def test_reward_criterion_output_finite():
    criterion = RewardCriterion(name="harmlessness", d_model=D_MODEL)
    hidden = make_hidden()
    out = criterion(hidden)
    assert torch.isfinite(out).all(), "Criterion scores contain non-finite values"


# ---------------------------------------------------------------------------
# 3. CriterionWeights.get_weights sums to 1
# ---------------------------------------------------------------------------

def test_criterion_weights_sums_to_one():
    cw = CriterionWeights(n_criteria=N_CRITERIA, learnable=False)
    w = cw.get_weights()
    assert abs(w.sum().item() - 1.0) < 1e-5, f"Weights sum to {w.sum().item()}, expected 1.0"


# ---------------------------------------------------------------------------
# 4. Learnable weights have requires_grad=True
# ---------------------------------------------------------------------------

def test_learnable_weights_requires_grad():
    cw = CriterionWeights(n_criteria=N_CRITERIA, learnable=True)
    assert cw.raw_weights.requires_grad, "Learnable weights should require grad"


# ---------------------------------------------------------------------------
# 5. Fixed weights are uniform
# ---------------------------------------------------------------------------

def test_fixed_weights_are_uniform():
    cw = CriterionWeights(n_criteria=N_CRITERIA, learnable=False)
    w = cw.get_weights()
    expected = 1.0 / N_CRITERIA
    assert (w - expected).abs().max().item() < 1e-5, "Fixed weights should be uniform"


# ---------------------------------------------------------------------------
# 6. HierarchicalRewardModel forward returns expected keys
# ---------------------------------------------------------------------------

def test_hierarchical_reward_model_forward_keys():
    model = make_model()
    hidden = make_hidden()
    out = model(hidden)
    assert "total_reward" in out
    assert "criterion_scores" in out
    assert "weights" in out


# ---------------------------------------------------------------------------
# 7. total_reward shape is (B,)
# ---------------------------------------------------------------------------

def test_total_reward_shape():
    model = make_model()
    hidden = make_hidden()
    out = model(hidden)
    assert out["total_reward"].shape == (B,), f"Expected ({B},), got {out['total_reward'].shape}"


# ---------------------------------------------------------------------------
# 8. criterion_scores has n_criteria entries
# ---------------------------------------------------------------------------

def test_criterion_scores_length():
    model = make_model()
    hidden = make_hidden()
    out = model(hidden)
    assert len(out["criterion_scores"]) == N_CRITERIA, (
        f"Expected {N_CRITERIA} criterion scores, got {len(out['criterion_scores'])}"
    )


# ---------------------------------------------------------------------------
# 9. criterion_names returns list of correct length
# ---------------------------------------------------------------------------

def test_criterion_names_length():
    model = make_model()
    names = model.criterion_names()
    assert isinstance(names, list)
    assert len(names) == N_CRITERIA


# ---------------------------------------------------------------------------
# 10. breakdown returns dict with 'total' and criterion names
# ---------------------------------------------------------------------------

def test_breakdown_keys():
    model = make_model()
    hidden = make_hidden()
    result = model.breakdown(hidden)
    assert "total" in result
    for name in model.criterion_names():
        assert name in result, f"Missing criterion '{name}' in breakdown"


# ---------------------------------------------------------------------------
# 11. HierarchicalRewardLoss.preference_loss returns 'total_loss' and 'margin'
# ---------------------------------------------------------------------------

def test_preference_loss_keys():
    model = make_model()
    loss_fn = HierarchicalRewardLoss(reward_model=model)
    hidden_w = make_hidden()
    hidden_l = make_hidden()
    result = loss_fn.preference_loss(hidden_w, hidden_l)
    assert "total_loss" in result
    assert "margin" in result


# ---------------------------------------------------------------------------
# 12. diversity_regularizer returns a scalar tensor
# ---------------------------------------------------------------------------

def test_diversity_regularizer_is_scalar():
    model = make_model()
    loss_fn = HierarchicalRewardLoss(reward_model=model)
    hidden = make_hidden()
    reg = loss_fn.diversity_regularizer(hidden)
    assert reg.shape == torch.Size([]), f"Expected scalar tensor, got shape {reg.shape}"


# ---------------------------------------------------------------------------
# 13. diversity_regularizer >= 0
# ---------------------------------------------------------------------------

def test_diversity_regularizer_non_negative():
    model = make_model()
    loss_fn = HierarchicalRewardLoss(reward_model=model)
    hidden = make_hidden()
    reg = loss_fn.diversity_regularizer(hidden)
    assert reg.item() >= 0.0, f"Diversity regularizer should be >= 0, got {reg.item()}"


# ---------------------------------------------------------------------------
# 14. MultiObjectiveRewardOptimizer.scalarize output shape (B,)
# ---------------------------------------------------------------------------

def test_scalarize_output_shape():
    optimizer = MultiObjectiveRewardOptimizer(n_criteria=N_CRITERIA)
    names = ["helpfulness", "harmlessness", "honesty"]
    scores = {name: torch.randn(B) for name in names}
    result = optimizer.scalarize(scores, names)
    assert result.shape == (B,), f"Expected ({B},), got {result.shape}"


# ---------------------------------------------------------------------------
# 15. dominated: b=[2,2] dominates a=[1,1]
# ---------------------------------------------------------------------------

def test_dominated_b_dominates_a():
    optimizer = MultiObjectiveRewardOptimizer(n_criteria=2)
    scores_a = torch.tensor([1.0, 1.0])
    scores_b = torch.tensor([2.0, 2.0])
    assert optimizer.dominated(scores_a, scores_b), "b=[2,2] should dominate a=[1,1]"
