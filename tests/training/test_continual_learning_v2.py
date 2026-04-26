"""Tests for Advanced Continual Learning: Online EWC++ and A-GEM."""

from __future__ import annotations

import torch
import torch.nn as nn
from aurelius.training.continual_learning_v2 import (
    AGEMTrainer,
    EpisodicMemory,
    EWCPlusPlusTrainer,
    OnlineFisherEstimate,
)

# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

D_INPUT = 8
D_MODEL = 16
BATCH = 4


def _make_model() -> nn.Module:
    torch.manual_seed(42)
    return nn.Sequential(
        nn.Linear(D_INPUT, D_MODEL),
        nn.ReLU(),
        nn.Linear(D_MODEL, D_MODEL),
    )


def _make_x() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(BATCH, D_INPUT)


def _make_y() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(BATCH, D_MODEL)


def _mse_loss_fn(output: torch.Tensor) -> torch.Tensor:
    """Unsupervised proxy: MSE toward zeros."""
    return (output**2).mean()


def _supervised_loss_fn(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(output, target)


# ---------------------------------------------------------------------------
# 1. OnlineFisherEstimate initializes with zero fisher
# ---------------------------------------------------------------------------


def test_online_fisher_init_zero():
    model = _make_model()
    fe = OnlineFisherEstimate(model, gamma=0.9)
    for name, f in fe.fisher.items():
        assert (f == 0).all(), f"Fisher for {name} should be zero at init"


# ---------------------------------------------------------------------------
# 2. update() sets params_star to current params
# ---------------------------------------------------------------------------


def test_update_sets_params_star():
    model = _make_model()
    fe = OnlineFisherEstimate(model, gamma=0.9)
    x = _make_x()
    loss = _mse_loss_fn(model(x))
    fe.update(loss)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        assert torch.allclose(fe.params_star[name], param.data), (
            f"params_star[{name}] should match current param after update"
        )


# ---------------------------------------------------------------------------
# 3. update() fisher is non-zero after gradient computed
# ---------------------------------------------------------------------------


def test_update_fisher_nonzero():
    model = _make_model()
    fe = OnlineFisherEstimate(model, gamma=0.9)
    x = _make_x()
    loss = _mse_loss_fn(model(x))
    fe.update(loss)

    total_fisher_mass = sum(f.abs().sum().item() for f in fe.fisher.values())
    assert total_fisher_mass > 0.0, "Fisher should be non-zero after update with non-trivial loss"


# ---------------------------------------------------------------------------
# 4. ewc_penalty is zero when params unchanged after update
# ---------------------------------------------------------------------------


def test_ewc_penalty_zero_when_params_unchanged():
    model = _make_model()
    fe = OnlineFisherEstimate(model, gamma=0.9)
    x = _make_x()
    loss = _mse_loss_fn(model(x))
    fe.update(loss)

    # params_star is now set to current params; penalty should be ~0
    penalty = fe.ewc_penalty(model)
    assert abs(penalty.item()) < 1e-6, (
        f"Penalty should be ~0 when params unchanged, got {penalty.item()}"
    )


# ---------------------------------------------------------------------------
# 5. ewc_penalty is positive when params change after update
# ---------------------------------------------------------------------------


def test_ewc_penalty_positive_after_param_change():
    model = _make_model()
    fe = OnlineFisherEstimate(model, gamma=0.9)
    x = _make_x()
    loss = _mse_loss_fn(model(x))
    fe.update(loss)

    # Perturb parameters
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.5)

    penalty = fe.ewc_penalty(model)
    assert penalty.item() > 0.0, (
        f"Penalty should be positive after parameter perturbation, got {penalty.item()}"
    )


# ---------------------------------------------------------------------------
# 6. EWCPlusPlusTrainer.train_step returns expected keys
# ---------------------------------------------------------------------------


def test_ewcplusplus_train_step_returns_expected_keys():
    model = _make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    trainer = EWCPlusPlusTrainer(model, optimizer, ewc_lambda=10.0, gamma=0.9)
    x = _make_x()
    result = trainer.train_step(x, _mse_loss_fn)

    assert "task_loss" in result
    assert "ewc_penalty" in result
    assert "total_loss" in result


# ---------------------------------------------------------------------------
# 7. train_step total_loss is finite
# ---------------------------------------------------------------------------


def test_ewcplusplus_train_step_total_loss_finite():
    model = _make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    trainer = EWCPlusPlusTrainer(model, optimizer, ewc_lambda=10.0, gamma=0.9)
    x = _make_x()
    result = trainer.train_step(x, _mse_loss_fn)

    assert torch.isfinite(torch.tensor(result["total_loss"])), (
        f"total_loss should be finite, got {result['total_loss']}"
    )


# ---------------------------------------------------------------------------
# 8. train_step ewc_penalty is non-negative
# ---------------------------------------------------------------------------


def test_ewcplusplus_train_step_ewc_penalty_nonnegative():
    model = _make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    trainer = EWCPlusPlusTrainer(model, optimizer, ewc_lambda=10.0, gamma=0.9)
    x = _make_x()
    result = trainer.train_step(x, _mse_loss_fn)

    assert result["ewc_penalty"] >= 0.0, (
        f"ewc_penalty should be non-negative, got {result['ewc_penalty']}"
    )


# ---------------------------------------------------------------------------
# 9. EpisodicMemory.add stores samples
# ---------------------------------------------------------------------------


def test_episodic_memory_add_stores_samples():
    mem = EpisodicMemory(max_size=10, d_input=D_INPUT)
    x = _make_x()
    mem.add(x[0])
    mem.add(x[1])
    assert len(mem.memory) == 2


# ---------------------------------------------------------------------------
# 10. add evicts oldest when over max_size
# ---------------------------------------------------------------------------


def test_episodic_memory_evicts_oldest():
    mem = EpisodicMemory(max_size=3, d_input=D_INPUT)
    samples = [torch.full((D_INPUT,), float(i)) for i in range(5)]
    for s in samples:
        mem.add(s)

    assert len(mem.memory) == 3, f"Memory should have 3 items, got {len(mem.memory)}"
    # The oldest (0, 1) should have been evicted; remaining should be 2, 3, 4
    remaining_values = [mem.memory[i][0].item() for i in range(3)]
    assert remaining_values == [2.0, 3.0, 4.0], (
        f"Expected [2, 3, 4] after eviction, got {remaining_values}"
    )


# ---------------------------------------------------------------------------
# 11. sample returns tensor of correct shape
# ---------------------------------------------------------------------------


def test_episodic_memory_sample_correct_shape():
    mem = EpisodicMemory(max_size=20, d_input=D_INPUT)
    for _ in range(10):
        mem.add(torch.randn(D_INPUT))

    samples = mem.sample(5)
    assert samples is not None
    assert samples.shape == (5, D_INPUT), f"Expected shape (5, {D_INPUT}), got {samples.shape}"


# ---------------------------------------------------------------------------
# 12. sample returns None for empty memory
# ---------------------------------------------------------------------------


def test_episodic_memory_sample_empty_returns_none():
    mem = EpisodicMemory(max_size=10, d_input=D_INPUT)
    result = mem.sample(5)
    assert result is None, "sample() on empty memory should return None"


# ---------------------------------------------------------------------------
# 13. AGEMTrainer.train_step returns expected keys
# ---------------------------------------------------------------------------


def test_agem_train_step_returns_expected_keys():
    model = _make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    memory = EpisodicMemory(max_size=20, d_input=D_INPUT)
    for _ in range(5):
        memory.add(torch.randn(D_INPUT))

    trainer = AGEMTrainer(model, optimizer, memory)
    x = _make_x()
    y = _make_y()
    result = trainer.train_step(x, y, _supervised_loss_fn)

    assert "task_loss" in result
    assert "projected" in result


# ---------------------------------------------------------------------------
# 14. AGEMTrainer.train_step with empty memory still works (no projection)
# ---------------------------------------------------------------------------


def test_agem_train_step_empty_memory():
    model = _make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    memory = EpisodicMemory(max_size=20, d_input=D_INPUT)

    trainer = AGEMTrainer(model, optimizer, memory)
    x = _make_x()
    y = _make_y()
    result = trainer.train_step(x, y, _supervised_loss_fn)

    assert "task_loss" in result
    assert result["projected"] is False, "projected should be False when memory is empty"
    assert torch.isfinite(torch.tensor(result["task_loss"]))


# ---------------------------------------------------------------------------
# 15. _project_gradient: when dot >= 0, returns current unchanged
# ---------------------------------------------------------------------------


def test_project_gradient_no_projection_when_dot_nonnegative():
    model = _make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    memory = EpisodicMemory(max_size=20, d_input=D_INPUT)
    trainer = AGEMTrainer(model, optimizer, memory)

    # Construct gradients where dot product >= 0 (identical grads)
    g1 = [torch.ones(3, 3), torch.ones(5)]
    g2 = [torch.ones(3, 3), torch.ones(5)]  # same direction -> dot > 0

    result = trainer._project_gradient(g1, g2)

    for r, orig in zip(result, g1):
        assert torch.allclose(r, orig), "Gradient should be unchanged when dot(g_c, g_r) >= 0"
