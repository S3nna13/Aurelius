"""Unit tests for PCGrad v2 — Cosine-Adaptive Conflicting Gradient Projection."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.training.pcgrad_v2 import GradientBank, PCGradV2, PCGradV2Config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_pcgrad(n_tasks: int = 2, adaptive_weight: bool = True, alpha: float = 1.0) -> PCGradV2:
    cfg = PCGradV2Config(n_tasks=n_tasks, adaptive_weight=adaptive_weight, alpha=alpha)
    return PCGradV2(cfg)


def vec(*values: float) -> torch.Tensor:
    return torch.tensor(list(values), dtype=torch.float32)


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = PCGradV2Config()
    assert cfg.n_tasks == 2
    assert cfg.adaptive_weight is True
    assert cfg.alpha == 1.0
    assert cfg.normalize_gradients is False
    assert cfg.gradient_bank_size == 1


# ---------------------------------------------------------------------------
# 2. test_conflict_score_aligned
# ---------------------------------------------------------------------------


def test_conflict_score_aligned():
    pcg = make_pcgrad()
    g = vec(1.0, 2.0, 3.0)
    score = pcg.conflict_score(g, g)
    assert abs(score - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# 3. test_conflict_score_opposed
# ---------------------------------------------------------------------------


def test_conflict_score_opposed():
    pcg = make_pcgrad()
    g = vec(1.0, 2.0, 3.0)
    score = pcg.conflict_score(g, -g)
    assert abs(score - (-1.0)) < 1e-5


# ---------------------------------------------------------------------------
# 4. test_conflict_score_orthogonal
# ---------------------------------------------------------------------------


def test_conflict_score_orthogonal():
    pcg = make_pcgrad()
    g1 = vec(1.0, 0.0)
    g2 = vec(0.0, 1.0)
    score = pcg.conflict_score(g1, g2)
    assert abs(score) < 1e-5


# ---------------------------------------------------------------------------
# 5. test_project_no_conflict — aligned grads leave g1 unchanged
# ---------------------------------------------------------------------------


def test_project_no_conflict():
    pcg = make_pcgrad()
    g1 = vec(1.0, 2.0, 3.0)
    g2 = vec(1.0, 2.0, 3.0)  # same direction
    resolved = pcg.project(g1, g2)
    assert torch.allclose(resolved, g1, atol=1e-5)


# ---------------------------------------------------------------------------
# 6. test_project_with_conflict — opposed grads: dot product improves
# ---------------------------------------------------------------------------


def test_project_with_conflict():
    pcg = make_pcgrad()
    g1 = vec(1.0, 0.0)
    g2 = vec(-1.0, 0.0)  # directly opposing
    resolved = pcg.project(g1, g2)
    # After projection the dot product of resolved and g2 should be >= original
    original_dot = float(g1 @ g2)
    resolved_dot = float(resolved @ g2)
    assert resolved_dot >= original_dot - 1e-6


# ---------------------------------------------------------------------------
# 7. test_project_reduces_conflict
# ---------------------------------------------------------------------------


def test_project_reduces_conflict():
    pcg = make_pcgrad()
    g1 = vec(1.0, 0.5)
    g2 = vec(-1.0, 0.1)
    resolved = pcg.project(g1, g2)
    cos_before = pcg.conflict_score(g1, g2)
    cos_after = pcg.conflict_score(resolved, g2)
    # After projection the cosine should be >= before (less negative / more positive)
    assert cos_after >= cos_before - 1e-6


# ---------------------------------------------------------------------------
# 8. test_resolve_shape
# ---------------------------------------------------------------------------


def test_resolve_shape():
    pcg = make_pcgrad()
    task_grads = [
        [torch.randn(4, 8), torch.randn(8)],
        [torch.randn(4, 8), torch.randn(8)],
    ]
    resolved = pcg.resolve(task_grads)
    assert len(resolved) == 2
    for t in range(2):
        for p in range(2):
            assert resolved[t][p].shape == task_grads[t][p].shape


# ---------------------------------------------------------------------------
# 9. test_resolve_2_tasks
# ---------------------------------------------------------------------------


def test_resolve_2_tasks():
    pcg = make_pcgrad()
    g_a = [vec(1.0, 0.0)]
    g_b = [vec(-1.0, 0.0)]
    resolved = pcg.resolve([g_a, g_b])
    assert len(resolved) == 2
    assert len(resolved[0]) == 1
    assert len(resolved[1]) == 1


# ---------------------------------------------------------------------------
# 10. test_gradient_bank_fills
# ---------------------------------------------------------------------------


def test_gradient_bank_fills():
    bank = GradientBank(n_tasks=2, bank_size=2)
    grads = [torch.randn(4)]
    # Add first batch for each task — not yet full
    full = bank.add(0, grads)
    assert not full
    full = bank.add(1, grads)
    assert not full
    # Add second batch for each task — now full
    full = bank.add(0, grads)
    assert not full  # task 1 still only has 1
    full = bank.add(1, grads)
    assert full  # now both have >= 2


# ---------------------------------------------------------------------------
# 11. test_gradient_bank_clear
# ---------------------------------------------------------------------------


def test_gradient_bank_clear():
    bank = GradientBank(n_tasks=2, bank_size=1)
    grads = [torch.randn(4)]
    bank.add(0, grads)
    bank.add(1, grads)
    bank.clear()
    # After clear, storage for both tasks is empty
    assert len(bank._storage[0]) == 0
    assert len(bank._storage[1]) == 0


# ---------------------------------------------------------------------------
# 12. test_per_layer_conflict_length
# ---------------------------------------------------------------------------


def test_per_layer_conflict_length():
    pcg = make_pcgrad()
    task_grads = [
        [torch.randn(4), torch.randn(8), torch.randn(2, 3)],
        [torch.randn(4), torch.randn(8), torch.randn(2, 3)],
    ]
    scores = pcg.per_layer_conflict(task_grads)
    assert len(scores) == 3
    for s in scores:
        assert isinstance(s, float)
        assert -1.0 - 1e-6 <= s <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# 13. test_apply_to_optimizer — runs without error, returns dict with keys
# ---------------------------------------------------------------------------


def test_apply_to_optimizer():
    torch.manual_seed(42)
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    pcg = make_pcgrad(n_tasks=2)

    x = torch.randn(3, 4)
    loss_a = model(x).sum()
    loss_b = (model(x) ** 2).sum()

    result = pcg.apply_to_optimizer(optimizer, [loss_a, loss_b])
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 14. test_apply_dict_keys
# ---------------------------------------------------------------------------


def test_apply_dict_keys():
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    pcg = make_pcgrad(n_tasks=2)

    x = torch.randn(3, 4)
    result = pcg.apply_to_optimizer(optimizer, [model(x).sum(), (-model(x)).sum()])
    assert "n_conflicts" in result
    assert "mean_conflict_score" in result
    assert isinstance(result["n_conflicts"], int)
    assert isinstance(result["mean_conflict_score"], float)


# ---------------------------------------------------------------------------
# 15. test_registry
# ---------------------------------------------------------------------------


def test_registry():
    from src.training import TRAINING_REGISTRY

    assert "pcgrad_v2" in TRAINING_REGISTRY
    assert TRAINING_REGISTRY["pcgrad_v2"] is PCGradV2
