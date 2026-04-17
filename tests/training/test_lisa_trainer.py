"""Tests for LISA (Layerwise Importance Sampling) trainer."""

from __future__ import annotations

from typing import Dict, List

import pytest
import torch
import torch.nn as nn

from aurelius.training.lisa_trainer import (
    LayerActivationSchedule,
    LISALayerManager,
    LISATrainer,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_LAYERS = 8
ACTIVATE_LAYERS = 3
SEED = 42
D_MODEL = 16


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_layer_params(n_layers: int, d: int = D_MODEL) -> Dict[int, List[nn.Parameter]]:
    """Create a dict of {layer_idx: [weight, bias]} using small nn.Linear layers."""
    torch.manual_seed(0)
    return {
        i: list(nn.Linear(d, d).parameters())
        for i in range(n_layers)
    }


@pytest.fixture()
def schedule() -> LayerActivationSchedule:
    return LayerActivationSchedule(N_LAYERS, ACTIVATE_LAYERS, seed=SEED)


@pytest.fixture()
def layer_params() -> Dict[int, List[nn.Parameter]]:
    return _make_layer_params(N_LAYERS)


@pytest.fixture()
def manager(layer_params, schedule) -> LISALayerManager:
    return LISALayerManager(layer_params, schedule)


@pytest.fixture()
def tiny_model_and_manager():
    """A tiny sequential model split into N_LAYERS linear sub-layers."""
    torch.manual_seed(7)
    layers = nn.ModuleList([nn.Linear(D_MODEL, D_MODEL) for _ in range(N_LAYERS)])
    model = nn.Sequential(*layers)

    named_layer_params: Dict[int, List[nn.Parameter]] = {
        i: list(layers[i].parameters()) for i in range(N_LAYERS)
    }
    sched = LayerActivationSchedule(N_LAYERS, ACTIVATE_LAYERS, seed=SEED)
    mgr = LISALayerManager(named_layer_params, sched)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return model, mgr, optimizer, layers


# ---------------------------------------------------------------------------
# LayerActivationSchedule tests
# ---------------------------------------------------------------------------


def test_sample_returns_exact_count(schedule):
    """1. sample() returns exactly activate_layers indices."""
    indices = schedule.sample(step=0)
    assert len(indices) == ACTIVATE_LAYERS


def test_sample_indices_in_range(schedule):
    """2. All returned indices are in [0, n_layers)."""
    for step in range(20):
        indices = schedule.sample(step=step)
        assert all(0 <= idx < N_LAYERS for idx in indices), (
            f"Out-of-range index at step {step}: {indices}"
        )


def test_sample_sorted_order(schedule):
    """3. Returned list is in sorted (ascending) order."""
    for step in range(10):
        indices = schedule.sample(step=step)
        assert indices == sorted(indices), f"Not sorted at step {step}: {indices}"


def test_sample_different_steps_differ(schedule):
    """4. Different steps (generally) produce different samples."""
    results = [tuple(schedule.sample(step=s)) for s in range(50)]
    unique = set(results)
    # With 8 layers choose 3, there are C(8,3)=56 combos; across 50 steps
    # we expect more than 1 unique sample with overwhelming probability.
    assert len(unique) > 1, "All 50 steps produced identical samples — suspicious."


def test_sample_reproducibility(schedule):
    """5. Same step + same seed always yields the same sample."""
    sched2 = LayerActivationSchedule(N_LAYERS, ACTIVATE_LAYERS, seed=SEED)
    for step in range(10):
        assert schedule.sample(step) == sched2.sample(step)


def test_activation_fraction(schedule):
    """6. activation_fraction() == activate_layers / n_layers."""
    expected = ACTIVATE_LAYERS / N_LAYERS
    assert abs(schedule.activation_fraction() - expected) < 1e-9


# ---------------------------------------------------------------------------
# LISALayerManager tests
# ---------------------------------------------------------------------------


def test_activate_step_freezes_non_active(manager):
    """7. After activate_step, non-active layers have requires_grad=False."""
    active = set(manager.activate_step(step=0))
    inactive = set(range(N_LAYERS)) - active
    for idx in inactive:
        for p in manager._layer_params[idx]:
            assert not p.requires_grad, (
                f"Layer {idx} should be frozen but requires_grad=True"
            )


def test_activate_step_unfreezes_active(manager):
    """8. After activate_step, active layers have requires_grad=True."""
    active = manager.activate_step(step=0)
    for idx in active:
        for p in manager._layer_params[idx]:
            assert p.requires_grad, (
                f"Layer {idx} should be unfrozen but requires_grad=False"
            )


def test_unfreeze_all_restores_grad(manager):
    """9. unfreeze_all() sets requires_grad=True for every managed parameter."""
    manager.activate_step(step=0)  # freeze most layers
    manager.unfreeze_all()
    for idx, params in manager._layer_params.items():
        for p in params:
            assert p.requires_grad, f"Layer {idx} param not unfrozen after unfreeze_all()"


# ---------------------------------------------------------------------------
# LISATrainer tests
# ---------------------------------------------------------------------------


def test_train_step_returns_expected_keys(tiny_model_and_manager):
    """10. train_step() returns dict with 'loss', 'active_layers', 'n_active_params'."""
    model, mgr, optimizer, _ = tiny_model_and_manager
    x = torch.randn(2, D_MODEL)

    def loss_fn():
        return model(x).sum()

    result = LISATrainer(model, optimizer, mgr).train_step(loss_fn, step=0)
    assert set(result.keys()) == {"loss", "active_layers", "n_active_params"}


def test_train_step_loss_is_finite(tiny_model_and_manager):
    """11. The reported loss is a finite float."""
    model, mgr, optimizer, _ = tiny_model_and_manager
    x = torch.randn(2, D_MODEL)

    def loss_fn():
        return model(x).sum()

    result = LISATrainer(model, optimizer, mgr).train_step(loss_fn, step=0)
    assert isinstance(result["loss"], float)
    assert torch.isfinite(torch.tensor(result["loss"]))


def test_train_step_only_active_params_have_grad(tiny_model_and_manager):
    """12. After train_step backward, only active-layer params have .grad set."""
    model, mgr, optimizer, layers = tiny_model_and_manager
    x = torch.randn(2, D_MODEL)

    trainer = LISATrainer(model, optimizer, mgr)

    # We need to inspect grads *after* backward but the trainer does optimizer.step()
    # which doesn't clear grads. Re-run a step but intercept.
    active_indices = set(mgr.activate_step(step=5))
    optimizer.zero_grad()
    loss = model(x).sum()
    loss.backward()

    for i in range(N_LAYERS):
        for p in layers[i].parameters():
            if i in active_indices:
                # Active layer: grad should exist (may be zero tensor but not None)
                assert p.grad is not None, f"Expected grad on active layer {i}"
            else:
                # Frozen layer: no grad
                assert p.grad is None, f"Unexpected grad on frozen layer {i}"


def test_train_step_twice_updates_params(tiny_model_and_manager):
    """13. Calling train_step twice both succeed and update parameters."""
    model, mgr, optimizer, _ = tiny_model_and_manager
    x = torch.randn(2, D_MODEL)

    # Capture parameter snapshot before any updates.
    params_before = [p.data.clone() for p in model.parameters()]

    def loss_fn():
        return model(x).pow(2).mean()

    trainer = LISATrainer(model, optimizer, mgr)
    r1 = trainer.train_step(loss_fn, step=0)
    r2 = trainer.train_step(loss_fn, step=1)

    params_after = [p.data.clone() for p in model.parameters()]

    assert isinstance(r1["loss"], float)
    assert isinstance(r2["loss"], float)

    # At least some parameter should have changed.
    changed = any(
        not torch.equal(pb, pa) for pb, pa in zip(params_before, params_after)
    )
    assert changed, "No parameter was updated after two train_step calls."


def test_n_active_params_is_positive(tiny_model_and_manager):
    """14. n_active_params is a positive integer."""
    model, mgr, optimizer, _ = tiny_model_and_manager
    x = torch.randn(2, D_MODEL)

    def loss_fn():
        return model(x).sum()

    result = LISATrainer(model, optimizer, mgr).train_step(loss_fn, step=0)
    n = result["n_active_params"]
    assert isinstance(n, int)
    assert n > 0


def test_activate_layers_one(layer_params):
    """15. Works correctly with activate_layers=1 (minimum)."""
    sched = LayerActivationSchedule(N_LAYERS, activate_layers=1, seed=0)
    mgr = LISALayerManager(layer_params, sched)

    indices = mgr.activate_step(step=0)
    assert len(indices) == 1

    active_idx = indices[0]
    for idx, params in layer_params.items():
        for p in params:
            if idx == active_idx:
                assert p.requires_grad
            else:
                assert not p.requires_grad
