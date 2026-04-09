"""Tests for multi-objective optimization module."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.multi_objective import (
    MOOConfig,
    MOOTrainer,
    chebyshev_scalarization,
    compute_hypervolume,
    compute_pareto_front,
    eps_constrained_loss,
    is_pareto_dominant,
    linear_scalarization,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_config():
    return AureliusConfig(
        n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
        head_dim=32, d_ff=128, vocab_size=256, max_seq_len=512,
    )


@pytest.fixture
def model(small_config):
    torch.manual_seed(0)
    return AureliusTransformer(small_config)


@pytest.fixture
def optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=1e-3)


@pytest.fixture
def input_ids():
    torch.manual_seed(1)
    return torch.randint(0, 256, (2, 8))


# ---------------------------------------------------------------------------
# 1. MOOConfig defaults
# ---------------------------------------------------------------------------

def test_moo_config_defaults():
    cfg = MOOConfig()
    assert cfg.n_objectives == 3
    assert cfg.method == "linear_scalarization"
    assert cfg.reference_point is None
    assert cfg.eps_tolerance == 0.05


# ---------------------------------------------------------------------------
# 2. is_pareto_dominant — a strictly better on all objectives → True
# ---------------------------------------------------------------------------

def test_is_pareto_dominant_strict():
    a = [1.0, 2.0, 3.0]
    b = [2.0, 3.0, 4.0]
    assert is_pareto_dominant(a, b) is True


# ---------------------------------------------------------------------------
# 3. is_pareto_dominant — equal solutions → False
# ---------------------------------------------------------------------------

def test_is_pareto_dominant_equal():
    a = [1.0, 2.0]
    b = [1.0, 2.0]
    assert is_pareto_dominant(a, b) is False


# ---------------------------------------------------------------------------
# 4. is_pareto_dominant — partial dominance (a better on one, worse on other) → False
# ---------------------------------------------------------------------------

def test_is_pareto_dominant_partial():
    a = [1.0, 5.0]   # better on obj0, worse on obj1
    b = [3.0, 2.0]   # worse on obj0, better on obj1
    assert is_pareto_dominant(a, b) is False


# ---------------------------------------------------------------------------
# 5. compute_pareto_front — single solution → returns that solution's index
# ---------------------------------------------------------------------------

def test_compute_pareto_front_single():
    solutions = [[1.0, 2.0, 3.0]]
    front = compute_pareto_front(solutions)
    assert front == [0]


# ---------------------------------------------------------------------------
# 6. compute_pareto_front — two non-dominated solutions → both returned
# ---------------------------------------------------------------------------

def test_compute_pareto_front_two_non_dominated():
    solutions = [
        [1.0, 5.0],   # better on obj0, worse on obj1
        [5.0, 1.0],   # worse on obj0, better on obj1
    ]
    front = compute_pareto_front(solutions)
    assert sorted(front) == [0, 1]


# ---------------------------------------------------------------------------
# 7. compute_pareto_front — one dominated solution excluded
# ---------------------------------------------------------------------------

def test_compute_pareto_front_dominated_excluded():
    solutions = [
        [1.0, 1.0],   # dominates the others
        [2.0, 2.0],   # dominated by [1,1]
        [3.0, 3.0],   # dominated by both
    ]
    front = compute_pareto_front(solutions)
    assert front == [0]


# ---------------------------------------------------------------------------
# 8. compute_hypervolume — 2D simple case
# ---------------------------------------------------------------------------

def test_compute_hypervolume_2d():
    # Single point [1, 1], reference [3, 3]
    # Hypervolume = (3-1) * (3-1) = 4
    pareto_front = [[1.0, 1.0]]
    reference = [3.0, 3.0]
    hv = compute_hypervolume(pareto_front, reference)
    assert abs(hv - 4.0) < 1e-6


# ---------------------------------------------------------------------------
# 9. linear_scalarization — correct weighted sum
# ---------------------------------------------------------------------------

def test_linear_scalarization():
    objectives = torch.tensor([2.0, 3.0, 4.0])
    weights = torch.tensor([0.5, 0.3, 0.2])
    result = linear_scalarization(objectives, weights)
    expected = 0.5 * 2.0 + 0.3 * 3.0 + 0.2 * 4.0  # 1.0 + 0.9 + 0.8 = 2.7
    assert result.dim() == 0
    assert abs(result.item() - expected) < 1e-5


# ---------------------------------------------------------------------------
# 10. chebyshev_scalarization — returns scalar
# ---------------------------------------------------------------------------

def test_chebyshev_scalarization_scalar():
    objectives = torch.tensor([2.0, 3.0])
    weights = torch.tensor([1.0, 1.0])
    reference = torch.tensor([0.0, 0.0])
    result = chebyshev_scalarization(objectives, weights, reference)
    assert result.dim() == 0
    # max(1*|2-0|, 1*|3-0|) = 3.0
    assert abs(result.item() - 3.0) < 1e-5


# ---------------------------------------------------------------------------
# 11. eps_constrained_loss — zero penalty when all constraints satisfied
# ---------------------------------------------------------------------------

def test_eps_constrained_loss_zero_penalty():
    primary = torch.tensor(1.5)
    constraint_losses = [torch.tensor(0.02), torch.tensor(0.03)]
    tolerances = [0.05, 0.05]
    result = eps_constrained_loss(primary, constraint_losses, tolerances)
    # Both constraints satisfied (< tolerance), so penalty = 0
    assert result.dim() == 0
    assert abs(result.item() - primary.item()) < 1e-6


def test_eps_constrained_loss_with_violation():
    primary = torch.tensor(1.0)
    # constraint_loss = 0.1, tolerance = 0.05  → violation = 0.05
    constraint_losses = [torch.tensor(0.1)]
    tolerances = [0.05]
    result = eps_constrained_loss(primary, constraint_losses, tolerances)
    # penalty = (0.1 - 0.05)^2 = 0.0025
    assert result.item() > primary.item()
    assert abs(result.item() - (1.0 + 0.05 ** 2)) < 1e-5


# ---------------------------------------------------------------------------
# 12. MOOTrainer.train_step returns correct keys
# ---------------------------------------------------------------------------

def test_moo_trainer_train_step_keys(model, optimizer, input_ids):
    config = MOOConfig(n_objectives=2, method="linear_scalarization")
    trainer = MOOTrainer(model, config, optimizer)

    target = input_ids[:, 1:]  # shift right as language-model target

    def obj1():
        _, logits, _ = model(input_ids)
        # Cross-entropy over vocab for first slice
        logits_flat = logits[:, :-1, :].reshape(-1, logits.size(-1))
        tgt_flat = target.reshape(-1)
        return nn.CrossEntropyLoss()(logits_flat, tgt_flat)

    def obj2():
        _, logits, _ = model(input_ids)
        return logits.abs().mean()

    result = trainer.train_step([obj1, obj2])

    assert "total_loss" in result
    assert "objectives" in result
    assert "pareto_dominated" in result
    assert len(result["objectives"]) == 2
    assert isinstance(result["total_loss"], float)
    assert isinstance(result["pareto_dominated"], bool)
