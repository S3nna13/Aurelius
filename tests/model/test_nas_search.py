"""Tests for DARTS-style NAS search module."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.model.nas_search import (
    NASConfig,
    MixedOp,
    gumbel_softmax_weights,
    DARTSCell,
    DARTSSearcher,
)

torch.manual_seed(42)

# Fast test dimensions
B, T, D = 2, 4, 32
N_OPS = 3
N_CELLS = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_input() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(B, T, D)


def _make_ops(d_model: int = D, n: int = N_OPS) -> nn.ModuleList:
    ops = [
        nn.Identity(),
        nn.Linear(d_model, d_model),
        nn.Sequential(nn.ReLU(), nn.Linear(d_model, d_model)),
    ]
    return nn.ModuleList(ops[:n])


def _make_config(**kwargs) -> NASConfig:
    return NASConfig(n_ops=N_OPS, d_model=D, **kwargs)


def _make_cells(n: int = N_CELLS) -> nn.ModuleList:
    cfg = _make_config()
    return nn.ModuleList([DARTSCell(D, N_OPS, cfg) for _ in range(n)])


# ---------------------------------------------------------------------------
# 1. NASConfig defaults
# ---------------------------------------------------------------------------

def test_nas_config_defaults():
    cfg = NASConfig()
    assert cfg.n_ops == 4
    assert cfg.d_model == 64
    assert cfg.temperature == 1.0
    assert cfg.hard is False


# ---------------------------------------------------------------------------
# 2. MixedOp output shape matches input
# ---------------------------------------------------------------------------

def test_mixed_op_output_shape():
    cfg = _make_config()
    ops = _make_ops()
    mixed = MixedOp(ops, cfg)
    x = _sample_input()
    out = mixed(x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# 3. MixedOp differentiable through arch_weights
# ---------------------------------------------------------------------------

def test_mixed_op_differentiable():
    cfg = _make_config(hard=False)
    ops = _make_ops()
    mixed = MixedOp(ops, cfg)
    x = _sample_input()
    out = mixed(x)
    loss = out.sum()
    loss.backward()
    assert mixed.arch_weights.grad is not None
    assert mixed.arch_weights.grad.shape == mixed.arch_weights.shape


# ---------------------------------------------------------------------------
# 4. gumbel_softmax_weights output sums to 1
# ---------------------------------------------------------------------------

def test_gumbel_softmax_sums_to_one():
    logits = torch.randn(N_OPS)
    weights = gumbel_softmax_weights(logits, temperature=1.0, hard=False)
    assert abs(weights.sum().item() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# 5. gumbel_softmax_weights hard=True returns one-hot-like (max ≈ 1)
# ---------------------------------------------------------------------------

def test_gumbel_softmax_hard_one_hot():
    logits = torch.randn(N_OPS)
    weights = gumbel_softmax_weights(logits, temperature=1.0, hard=True)
    assert abs(weights.max().item() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# 6. gumbel_softmax_weights shape matches input
# ---------------------------------------------------------------------------

def test_gumbel_softmax_shape():
    logits = torch.randn(N_OPS)
    weights = gumbel_softmax_weights(logits, temperature=1.0)
    assert weights.shape == logits.shape


# ---------------------------------------------------------------------------
# 7. DARTSCell output shape correct
# ---------------------------------------------------------------------------

def test_darts_cell_output_shape():
    cfg = _make_config()
    cell = DARTSCell(D, N_OPS, cfg)
    x = _sample_input()
    out = cell(x)
    assert out.shape == (B, T, D)


# ---------------------------------------------------------------------------
# 8. DARTSCell get_arch_params returns non-empty list
# ---------------------------------------------------------------------------

def test_darts_cell_arch_params_nonempty():
    cfg = _make_config()
    cell = DARTSCell(D, N_OPS, cfg)
    params = cell.get_arch_params()
    assert len(params) > 0


# ---------------------------------------------------------------------------
# 9. DARTSCell arch params have requires_grad=True
# ---------------------------------------------------------------------------

def test_darts_cell_arch_params_requires_grad():
    cfg = _make_config()
    cell = DARTSCell(D, N_OPS, cfg)
    for p in cell.get_arch_params():
        assert p.requires_grad is True


# ---------------------------------------------------------------------------
# 10. DARTSSearcher.arch_params returns parameters
# ---------------------------------------------------------------------------

def test_darts_searcher_arch_params():
    cells = _make_cells()
    cfg = _make_config()
    searcher = DARTSSearcher(cells, cfg)
    params = searcher.arch_params()
    assert len(params) > 0
    for p in params:
        assert isinstance(p, torch.nn.Parameter)


# ---------------------------------------------------------------------------
# 11. DARTSSearcher.model_params returns parameters
# ---------------------------------------------------------------------------

def test_darts_searcher_model_params():
    cells = _make_cells()
    cfg = _make_config()
    searcher = DARTSSearcher(cells, cfg)
    params = searcher.model_params()
    assert len(params) > 0
    for p in params:
        assert isinstance(p, torch.nn.Parameter)


# ---------------------------------------------------------------------------
# 12. DARTSSearcher.discretize returns list of ints
# ---------------------------------------------------------------------------

def test_darts_searcher_discretize_returns_ints():
    cells = _make_cells()
    cfg = _make_config()
    searcher = DARTSSearcher(cells, cfg)
    result = searcher.discretize()
    assert isinstance(result, list)
    for v in result:
        assert isinstance(v, int)


# ---------------------------------------------------------------------------
# 13. DARTSSearcher.discretize length equals n_cells
# ---------------------------------------------------------------------------

def test_darts_searcher_discretize_length():
    cells = _make_cells(N_CELLS)
    cfg = _make_config()
    searcher = DARTSSearcher(cells, cfg)
    result = searcher.discretize()
    assert len(result) == N_CELLS


# ---------------------------------------------------------------------------
# 14. Separate arch and model optimizers work independently
# ---------------------------------------------------------------------------

def test_separate_optimizers_independent():
    cells = _make_cells()
    cfg = _make_config(hard=False)
    searcher = DARTSSearcher(cells, cfg)

    arch_opt = torch.optim.Adam(searcher.arch_params(), lr=1e-3)
    model_opt = torch.optim.Adam(searcher.model_params(), lr=1e-3)

    x = _sample_input()

    # Step arch optimizer only, check model params are unaffected
    model_params_before = [p.clone().detach() for p in searcher.model_params()]

    arch_opt.zero_grad()
    out = sum(cell(x) for cell in cells)
    loss = out.sum()
    loss.backward()
    arch_opt.step()

    for p_before, p_after in zip(model_params_before, searcher.model_params()):
        assert torch.allclose(p_before, p_after), "model params changed during arch step"

    # Now step model optimizer
    arch_params_before = [p.clone().detach() for p in searcher.arch_params()]

    model_opt.zero_grad()
    out2 = sum(cell(x) for cell in cells)
    loss2 = out2.sum()
    loss2.backward()
    model_opt.step()

    for p_before, p_after in zip(arch_params_before, searcher.arch_params()):
        assert torch.allclose(p_before, p_after), "arch params changed during model step"
