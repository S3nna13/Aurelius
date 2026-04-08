"""Tests for src/training/nas.py — DARTS-style differentiable architecture search."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.training.nas import (
    ArchitectureStats,
    DARTSCell,
    DARTSSearcher,
    MixedOperation,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_mixed_op(d_model: int = 64, n_ops: int = 3) -> MixedOperation:
    ops = {f'op_{i}': nn.Linear(d_model, d_model) for i in range(n_ops)}
    return MixedOperation(ops)


def make_simple_model(d_model: int = 32) -> nn.Module:
    """Simple model containing a DARTSCell for searcher tests."""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.cell = DARTSCell(d_model, n_candidates=2)

        def forward(self, x):
            return self.cell(x)

    return SimpleModel()


# ---------------------------------------------------------------------------
# MixedOperation tests
# ---------------------------------------------------------------------------

def test_mixed_operation_output_shape():
    """(2, 16, 64) -> (2, 16, 64)."""
    d_model = 64
    mixed_op = make_mixed_op(d_model=d_model)
    x = torch.randn(2, 16, d_model)
    out = mixed_op(x)
    assert out.shape == (2, 16, d_model)


def test_mixed_operation_weights_sum_to_one():
    """softmax weights should sum to ~1.0."""
    mixed_op = make_mixed_op()
    weights = mixed_op.architecture_weights()
    total = sum(weights.values())
    assert abs(total - 1.0) < 1e-5, f"Weights sum to {total}, expected ~1.0"


def test_best_operation_valid():
    """best_operation() returns a key from the ops dict."""
    mixed_op = make_mixed_op()
    best = mixed_op.best_operation()
    assert best in mixed_op.ops, f"'{best}' not found in ops keys: {list(mixed_op.ops.keys())}"


def test_architecture_weights_dict():
    """architecture_weights() returns dict with all op names as keys."""
    mixed_op = make_mixed_op(n_ops=4)
    weights = mixed_op.architecture_weights()
    assert set(weights.keys()) == set(mixed_op._op_names)
    assert len(weights) == 4


# ---------------------------------------------------------------------------
# DARTSCell tests
# ---------------------------------------------------------------------------

def test_darts_cell_output_shape():
    """DARTSCell(64): (2, 8, 64) -> (2, 8, 64)."""
    d_model = 64
    cell = DARTSCell(d_model)
    x = torch.randn(2, 8, d_model)
    out = cell(x)
    assert out.shape == (2, 8, d_model)


# ---------------------------------------------------------------------------
# DARTSSearcher tests
# ---------------------------------------------------------------------------

def test_darts_searcher_model_step():
    """model_step returns float."""
    model = make_simple_model()
    searcher = DARTSSearcher(model)
    x = torch.randn(2, 4, 32)
    out = model(x)
    loss = out.mean()
    result = searcher.model_step(loss)
    assert isinstance(result, float)


def test_darts_searcher_arch_step():
    """arch_step returns float."""
    model = make_simple_model()
    searcher = DARTSSearcher(model)
    x = torch.randn(2, 4, 32)
    out = model(x)
    loss = out.mean()
    result = searcher.arch_step(loss)
    assert isinstance(result, float)


def test_arch_weights_change_after_arch_step():
    """arch weights change after arch_step."""
    model = make_simple_model()
    searcher = DARTSSearcher(model)

    # Capture arch weights before
    arch_params_before = {
        n: p.clone().detach()
        for n, p in model.named_parameters()
        if 'arch_weights' in n
    }

    # Forward + arch step
    x = torch.randn(2, 4, 32)
    out = model(x)
    loss = out.mean()
    searcher.arch_step(loss)

    # Check that at least one arch weight changed
    changed = False
    for n, p in model.named_parameters():
        if 'arch_weights' in n:
            if not torch.allclose(p.detach(), arch_params_before[n]):
                changed = True
                break
    assert changed, "Architecture weights did not change after arch_step"


def test_model_weights_unchanged_after_arch_step():
    """model weights unchanged after arch_step only."""
    model = make_simple_model()
    searcher = DARTSSearcher(model)

    # Capture model (non-arch) weights before
    model_params_before = {
        n: p.clone().detach()
        for n, p in model.named_parameters()
        if 'arch_weights' not in n
    }

    # Forward + arch step only (NOT model_step)
    x = torch.randn(2, 4, 32)
    out = model(x)
    loss = out.mean()
    searcher.arch_step(loss)

    # Check that all model weights remain unchanged
    for n, p in model.named_parameters():
        if 'arch_weights' not in n:
            assert torch.allclose(p.detach(), model_params_before[n]), (
                f"Model param '{n}' changed after arch_step only"
            )


def test_get_best_architecture():
    """get_best_architecture returns dict with module names as keys."""
    model = make_simple_model()
    searcher = DARTSSearcher(model)
    best_arch = searcher.get_best_architecture(model)
    assert isinstance(best_arch, dict)
    assert len(best_arch) > 0
    # All keys should be strings (module names) and values should be op name strings
    for k, v in best_arch.items():
        assert isinstance(k, str)
        assert isinstance(v, str)


# ---------------------------------------------------------------------------
# ArchitectureStats tests
# ---------------------------------------------------------------------------

def test_arch_stats_record():
    """Recording updates history length."""
    stats = ArchitectureStats()
    mixed_op = make_mixed_op()
    assert len(stats.history) == 0
    stats.record(0, mixed_op)
    stats.record(1, mixed_op)
    assert len(stats.history) == 2
    assert stats.history[0]['step'] == 0
    assert stats.history[1]['step'] == 1


def test_convergence_step_found():
    """After setting one arch_weight very high, convergence_step returns a step."""
    stats = ArchitectureStats()
    mixed_op = make_mixed_op(n_ops=3)

    # Step 0: uniform weights — no convergence
    stats.record(0, mixed_op)

    # Step 1: set one weight very high so softmax pushes it >= 0.8
    with torch.no_grad():
        mixed_op.arch_weights[0] = 10.0
        mixed_op.arch_weights[1] = 0.0
        mixed_op.arch_weights[2] = 0.0
    stats.record(1, mixed_op)

    step = stats.convergence_step(threshold=0.8)
    assert step is not None, "Expected convergence_step to return a step"
    assert step == 1, f"Expected convergence at step 1, got step {step}"
