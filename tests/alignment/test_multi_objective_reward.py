"""Tests for multi-objective reward modeling."""

from __future__ import annotations

import pytest
import torch

from src.alignment.multi_objective_reward import (
    MultiHeadRewardModel,
    MultiObjectiveRMTrainer,
    ObjectiveConfig,
    chebyshev_scalarize,
    compute_pareto_reward,
    is_pareto_dominated,
    linear_scalarize,
    pareto_front,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HIDDEN_DIM = 16
N_OBJECTIVES = 3
B = 4


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model():
    torch.manual_seed(0)
    return MultiHeadRewardModel(
        hidden_dim=HIDDEN_DIM,
        n_objectives=N_OBJECTIVES,
        objective_names=["helpfulness", "harmlessness", "honesty"],
    )


@pytest.fixture
def hidden():
    torch.manual_seed(1)
    return torch.randn(B, HIDDEN_DIM)


@pytest.fixture
def objectives():
    return [
        ObjectiveConfig(name="helpfulness", weight=1.0),
        ObjectiveConfig(name="harmlessness", weight=1.0),
        ObjectiveConfig(name="honesty", weight=1.0),
    ]


@pytest.fixture
def trainer(model, objectives):
    return MultiObjectiveRMTrainer(model, objectives, scalarization="linear", lr=1e-4)


# ---------------------------------------------------------------------------
# Test 1: forward shape
# ---------------------------------------------------------------------------


def test_forward_shape(model, hidden):
    rewards = model(hidden)
    assert rewards.shape == (B, N_OBJECTIVES), (
        f"Expected ({B}, {N_OBJECTIVES}), got {rewards.shape}"
    )


# ---------------------------------------------------------------------------
# Test 2: get_objective_reward shape
# ---------------------------------------------------------------------------


def test_get_objective_reward_shape(model, hidden):
    for idx in range(N_OBJECTIVES):
        r = model.get_objective_reward(hidden, idx)
        assert r.shape == (B,), f"obj {idx}: expected ({B},), got {r.shape}"


# ---------------------------------------------------------------------------
# Test 3: each head has independent parameters (different outputs same input)
# ---------------------------------------------------------------------------


def test_heads_are_independent(model, hidden):
    rewards = model(hidden)  # (B, N_OBJECTIVES)
    # If all heads were the same, all columns would be equal; they should differ
    cols = [rewards[:, i] for i in range(N_OBJECTIVES)]
    all_same = all(torch.allclose(cols[0], cols[i]) for i in range(1, N_OBJECTIVES))
    # With different random initializations this should be False
    assert not all_same, "All reward heads produced identical outputs — they may share parameters"


# ---------------------------------------------------------------------------
# Test 4: linear_scalarize shape and correctness
# ---------------------------------------------------------------------------


def test_linear_scalarize_shape_and_correctness():
    torch.manual_seed(42)
    rewards = torch.randn(B, N_OBJECTIVES)
    weights = torch.tensor([0.5, 0.3, 0.2])
    out = linear_scalarize(rewards, weights)
    assert out.shape == (B,), f"Expected ({B},), got {out.shape}"
    # Verify against manual computation
    expected = (rewards * weights.unsqueeze(0)).sum(dim=-1)
    assert torch.allclose(out, expected), "linear_scalarize does not match weighted dot product"


# ---------------------------------------------------------------------------
# Test 5: linear_scalarize with uniform weights == mean
# ---------------------------------------------------------------------------


def test_linear_scalarize_uniform_is_mean():
    torch.manual_seed(7)
    rewards = torch.randn(B, N_OBJECTIVES)
    weights = torch.ones(N_OBJECTIVES) / N_OBJECTIVES
    out = linear_scalarize(rewards, weights)
    expected = rewards.mean(dim=-1)
    assert torch.allclose(out, expected, atol=1e-6), (
        "linear_scalarize with uniform weights should equal row mean"
    )


# ---------------------------------------------------------------------------
# Test 6: chebyshev_scalarize shape
# ---------------------------------------------------------------------------


def test_chebyshev_scalarize_shape():
    torch.manual_seed(3)
    rewards = torch.randn(B, N_OBJECTIVES)
    weights = torch.ones(N_OBJECTIVES)
    out = chebyshev_scalarize(rewards, weights)
    assert out.shape == (B,), f"Expected ({B},), got {out.shape}"


# ---------------------------------------------------------------------------
# Test 7: chebyshev_scalarize with one-hot weight collapses to that objective
# ---------------------------------------------------------------------------


def test_chebyshev_one_hot_collapses():
    """With weight = [1,0,0] and reference=0, result = -|reward_0|."""
    torch.manual_seed(9)
    rewards = torch.randn(B, N_OBJECTIVES)
    weights = torch.tensor([1.0, 0.0, 0.0])
    ref = torch.zeros(N_OBJECTIVES)
    out = chebyshev_scalarize(rewards, weights, reference_point=ref)
    expected = -rewards[:, 0].abs()
    assert torch.allclose(out, expected, atol=1e-6), (
        "One-hot chebyshev should collapse to negative abs of that objective"
    )


# ---------------------------------------------------------------------------
# Test 8: is_pareto_dominated — dominated solution returns True
# ---------------------------------------------------------------------------


def test_is_pareto_dominated_true():
    # solution [1,1,1] is dominated by [2,2,2]
    solution = torch.tensor([1.0, 1.0, 1.0])
    population = torch.tensor([[2.0, 2.0, 2.0], [0.5, 0.5, 0.5]])
    assert is_pareto_dominated(solution, population) is True


# ---------------------------------------------------------------------------
# Test 9: is_pareto_dominated — Pareto-optimal returns False
# ---------------------------------------------------------------------------


def test_is_pareto_dominated_false():
    # solution [3,1,1] — no one dominates it (population has [1,3,1] and [1,1,3])
    solution = torch.tensor([3.0, 1.0, 1.0])
    population = torch.tensor([[1.0, 3.0, 1.0], [1.0, 1.0, 3.0]])
    assert is_pareto_dominated(solution, population) is False


# ---------------------------------------------------------------------------
# Test 10: is_pareto_dominated — empty population returns False
# ---------------------------------------------------------------------------


def test_is_pareto_dominated_empty_population():
    solution = torch.tensor([1.0, 2.0, 3.0])
    population = torch.zeros(0, 3)
    assert is_pareto_dominated(solution, population) is False


# ---------------------------------------------------------------------------
# Test 11: pareto_front — all identical solutions → all non-dominated
# ---------------------------------------------------------------------------


def test_pareto_front_identical():
    # Identical solutions: none strictly dominates another
    solutions = torch.ones(4, 3)
    mask = pareto_front(solutions)
    assert mask.shape == (4,)
    assert mask.all(), "All identical solutions should be non-dominated"


# ---------------------------------------------------------------------------
# Test 12: pareto_front — clearly dominated solution excluded
# ---------------------------------------------------------------------------


def test_pareto_front_dominated_excluded():
    # [2,2,2] dominates [1,1,1]
    solutions = torch.tensor(
        [
            [2.0, 2.0, 2.0],  # idx 0: dominates idx 1
            [1.0, 1.0, 1.0],  # idx 1: dominated
            [3.0, 0.5, 0.5],  # idx 2: non-dominated (high on obj 0)
        ]
    )
    mask = pareto_front(solutions)
    assert mask.shape == (3,)
    assert mask[0].item() is True, "Solution 0 should be non-dominated"
    assert mask[1].item() is False, "Solution 1 should be dominated"
    assert mask[2].item() is True, "Solution 2 should be non-dominated"


# ---------------------------------------------------------------------------
# Test 13: compute_loss returns scalar loss + metrics dict
# ---------------------------------------------------------------------------


def test_compute_loss_returns_scalar_and_metrics(trainer, hidden):
    torch.manual_seed(5)
    chosen = hidden
    rejected = torch.randn(B, HIDDEN_DIM)
    loss, metrics = trainer.compute_loss(chosen, rejected)
    assert loss.shape == (), f"Loss should be scalar, got shape {loss.shape}"
    assert isinstance(metrics, dict), "metrics should be a dict"
    assert len(metrics) == N_OBJECTIVES


# ---------------------------------------------------------------------------
# Test 14: per-objective metrics have correct keys
# ---------------------------------------------------------------------------


def test_compute_loss_metric_keys(trainer, hidden):
    torch.manual_seed(6)
    chosen = hidden
    rejected = torch.randn(B, HIDDEN_DIM)
    _, metrics = trainer.compute_loss(chosen, rejected)
    expected_keys = {"helpfulness", "harmlessness", "honesty"}
    assert set(metrics.keys()) == expected_keys, (
        f"Expected keys {expected_keys}, got {set(metrics.keys())}"
    )


# ---------------------------------------------------------------------------
# Test 15: detect_gradient_conflict returns (n_obj, n_obj) with diagonal ≈ 1.0
# ---------------------------------------------------------------------------


def test_detect_gradient_conflict_shape_and_diagonal(trainer, hidden):
    torch.manual_seed(11)
    chosen = hidden
    rejected = torch.randn(B, HIDDEN_DIM)
    sim = trainer.detect_gradient_conflict(chosen, rejected)
    assert sim.shape == (N_OBJECTIVES, N_OBJECTIVES), (
        f"Expected ({N_OBJECTIVES}, {N_OBJECTIVES}), got {sim.shape}"
    )
    for i in range(N_OBJECTIVES):
        assert abs(sim[i, i].item() - 1.0) < 1e-4, (
            f"Diagonal element [{i},{i}] = {sim[i, i].item():.6f}, expected ≈ 1.0"
        )


# ---------------------------------------------------------------------------
# Test 16: compute_pareto_reward — dominating gets +1, dominated gets -1
# ---------------------------------------------------------------------------


def test_compute_pareto_reward():
    # B=3: dominating, dominated, neutral
    rewards = torch.tensor(
        [
            [3.0, 3.0, 3.0],  # dominates ref → +1
            [0.5, 0.5, 0.5],  # dominated by ref → -1
            [2.0, 1.0, 3.0],  # neither (not all >= ref, not all <= ref) → 0
        ]
    )
    reference = torch.tensor(
        [
            [1.0, 1.0, 1.0],  # dominated by rewards[0]
            [2.0, 2.0, 2.0],  # dominates rewards[1]
            [1.0, 2.0, 2.0],  # rewards[2] wins obj0 and obj2, loses obj1 → 0
        ]
    )
    result = compute_pareto_reward(rewards, reference)
    assert result.shape == (3,), f"Expected (3,), got {result.shape}"
    assert result[0].item() == 1, f"Expected +1 for dominating, got {result[0].item()}"
    assert result[1].item() == -1, f"Expected -1 for dominated, got {result[1].item()}"
    assert result[2].item() == 0, f"Expected 0 for neutral, got {result[2].item()}"
