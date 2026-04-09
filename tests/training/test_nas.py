"""Tests for src/training/nas.py — DARTS/ENAS-style NAS primitives."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.nas import (
    NASConfig,
    ArchitectureStats,
    gumbel_softmax,
    compute_arch_entropy,
    compute_arch_dominance,
    MixedOp,
    DARTSCell,
    DARTSSearcher,
    random_architecture_search,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

TINY_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=2,
    n_kv_heads=2,
    head_dim=32,
    d_ff=128,
    vocab_size=256,
    max_seq_len=512,
)

NAS_CFG = NASConfig(n_candidates=4, d_model=64, temperature=1.0)


def make_cells(n_cells: int = 3, cfg: NASConfig | None = None) -> list[DARTSCell]:
    cfg = cfg or NAS_CFG
    return [DARTSCell(cfg) for _ in range(n_cells)]


# ---------------------------------------------------------------------------
# 1. NASConfig defaults
# ---------------------------------------------------------------------------

def test_nas_config_defaults():
    """NASConfig() should have n_candidates=4 and temperature=1.0."""
    cfg = NASConfig()
    assert cfg.n_candidates == 4
    assert cfg.temperature == 1.0


# ---------------------------------------------------------------------------
# 2. gumbel_softmax — shape
# ---------------------------------------------------------------------------

def test_gumbel_softmax_shape():
    """Output shape must match input shape."""
    logits = torch.randn(4)
    out = gumbel_softmax(logits, temperature=1.0, hard=False)
    assert out.shape == logits.shape


# ---------------------------------------------------------------------------
# 3. gumbel_softmax — sums to one
# ---------------------------------------------------------------------------

def test_gumbel_softmax_sums_to_one():
    """Soft Gumbel-Softmax output must sum to ~1."""
    logits = torch.randn(4)
    out = gumbel_softmax(logits, temperature=1.0, hard=False)
    assert abs(out.sum().item() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# 4. gumbel_softmax — hard mode gives exactly one 1.0
# ---------------------------------------------------------------------------

def test_gumbel_softmax_hard_one_hot():
    """Hard Gumbel-Softmax must produce a one-hot vector (straight-through)."""
    logits = torch.randn(4)
    out = gumbel_softmax(logits, temperature=1.0, hard=True)
    # In hard mode the detached data values should form a one-hot vector.
    data = out.detach()
    assert (data == 1.0).sum().item() == 1
    assert (data == 0.0).sum().item() == len(logits) - 1


# ---------------------------------------------------------------------------
# 5. compute_arch_entropy — uniform weights → max entropy
# ---------------------------------------------------------------------------

def test_compute_arch_entropy_uniform():
    """Uniform weights should give maximum entropy = log(n_candidates)."""
    n = 4
    weights = torch.full((3, n), 1.0 / n)
    entropy = compute_arch_entropy(weights)
    max_entropy = torch.log(torch.tensor(float(n))).item()
    assert abs(entropy - max_entropy) < 1e-4, f"Expected ~{max_entropy:.4f}, got {entropy:.4f}"


# ---------------------------------------------------------------------------
# 6. compute_arch_entropy — one-hot weights → ~0 entropy
# ---------------------------------------------------------------------------

def test_compute_arch_entropy_peaked():
    """One-hot weights should give ~0 entropy."""
    n = 4
    weights = torch.zeros(3, n)
    weights[:, 0] = 1.0
    entropy = compute_arch_entropy(weights)
    assert entropy < 1e-3, f"Expected entropy ~0, got {entropy:.6f}"


# ---------------------------------------------------------------------------
# 7. compute_arch_dominance — uniform → low dominance (~0)
# ---------------------------------------------------------------------------

def test_compute_arch_dominance_uniform():
    """Uniform weights: max == mean, dominance should be ~0."""
    n = 4
    weights = torch.full((3, n), 1.0 / n)
    dominance = compute_arch_dominance(weights)
    assert dominance < 1e-5, f"Expected dominance ~0 for uniform, got {dominance:.6f}"


# ---------------------------------------------------------------------------
# 8. compute_arch_dominance — peaked → high dominance
# ---------------------------------------------------------------------------

def test_compute_arch_dominance_peaked():
    """One-hot weights: max = 1, mean = 1/n → dominance = (n-1)/n."""
    n = 4
    weights = torch.zeros(3, n)
    weights[:, 0] = 1.0
    dominance = compute_arch_dominance(weights)
    expected = 1.0 - 1.0 / n  # (n-1)/n
    assert abs(dominance - expected) < 1e-5, f"Expected {expected:.4f}, got {dominance:.6f}"


# ---------------------------------------------------------------------------
# 9. MixedOp — output shape preserved
# ---------------------------------------------------------------------------

def test_mixed_op_output_shape():
    """MixedOp forward must preserve input shape."""
    d = 64
    ops = [nn.Linear(d, d) for _ in range(4)]
    mixed = MixedOp(ops)
    x = torch.randn(2, 8, d)
    weights = torch.softmax(torch.randn(4), dim=0)
    out = mixed(x, weights)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# 10. MixedOp — one-hot weights selects the single op
# ---------------------------------------------------------------------------

def test_mixed_op_weighted_sum():
    """With one-hot weights, MixedOp output equals the selected op's output."""
    d = 64
    ops = [nn.Linear(d, d, bias=False) for _ in range(4)]
    mixed = MixedOp(ops)
    x = torch.randn(2, 8, d)

    # Select op index 2 with one-hot weights
    weights = torch.zeros(4)
    weights[2] = 1.0

    with torch.no_grad():
        out = mixed(x, weights)
        expected = ops[2](x)

    assert torch.allclose(out, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# 11. DARTSCell — forward shape equals input shape
# ---------------------------------------------------------------------------

def test_darts_cell_forward_shape():
    """DARTSCell forward must return same shape as input."""
    cfg = NASConfig(n_candidates=4, d_model=64)
    cell = DARTSCell(cfg)
    x = torch.randn(2, 8, 64)
    out = cell(x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# 12. DARTSCell — get_weights sums to ~1
# ---------------------------------------------------------------------------

def test_darts_cell_get_weights_sums_to_one():
    """DARTSCell.get_weights() must return a vector summing to ~1."""
    cfg = NASConfig(n_candidates=4, d_model=64)
    cell = DARTSCell(cfg)
    weights = cell.get_weights()
    assert abs(weights.sum().item() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# 13. DARTSSearcher — arch_parameters returns one param per cell
# ---------------------------------------------------------------------------

def test_darts_searcher_arch_params():
    """arch_parameters() must return a list of length == n_cells."""
    model = AureliusTransformer(TINY_CFG)
    n_cells = 3
    cells = make_cells(n_cells)
    searcher = DARTSSearcher(model, cells, NAS_CFG)
    arch_params = searcher.arch_parameters()
    assert isinstance(arch_params, list)
    assert len(arch_params) == n_cells
    for p in arch_params:
        assert isinstance(p, nn.Parameter)


# ---------------------------------------------------------------------------
# 14. DARTSSearcher — discretize returns list of ints, len == n_cells
# ---------------------------------------------------------------------------

def test_darts_searcher_discretize():
    """discretize() must return a list of ints with length == n_cells."""
    model = AureliusTransformer(TINY_CFG)
    n_cells = 3
    cells = make_cells(n_cells)
    searcher = DARTSSearcher(model, cells, NAS_CFG)
    arch = searcher.discretize()
    assert isinstance(arch, list)
    assert len(arch) == n_cells
    for idx in arch:
        assert isinstance(idx, int)
        assert 0 <= idx < NAS_CFG.n_candidates


# ---------------------------------------------------------------------------
# 15. DARTSSearcher — get_architecture_stats has required attributes
# ---------------------------------------------------------------------------

def test_darts_searcher_stats_keys():
    """get_architecture_stats() must return ArchitectureStats with correct fields."""
    model = AureliusTransformer(TINY_CFG)
    n_cells = 3
    cells = make_cells(n_cells)
    searcher = DARTSSearcher(model, cells, NAS_CFG)
    stats = searcher.get_architecture_stats()
    assert isinstance(stats, ArchitectureStats)
    assert hasattr(stats, "selected_ops")
    assert hasattr(stats, "entropy")
    assert hasattr(stats, "dominance")
    assert isinstance(stats.selected_ops, list)
    assert len(stats.selected_ops) == n_cells
    assert isinstance(stats.entropy, float)
    assert isinstance(stats.dominance, float)


# ---------------------------------------------------------------------------
# 16. random_architecture_search — returns (list[int], float)
# ---------------------------------------------------------------------------

def test_random_arch_search_returns_best():
    """random_architecture_search must return (list[int], float)."""
    model = AureliusTransformer(TINY_CFG)
    n_cells = 3
    cells = make_cells(n_cells)

    call_count = {"n": 0}

    def score_fn(m: nn.Module) -> float:
        call_count["n"] += 1
        return float(torch.randn(1).item())

    best_arch, best_score = random_architecture_search(model, cells, n_trials=5, eval_fn=score_fn)

    assert isinstance(best_arch, list)
    assert len(best_arch) == n_cells
    for idx in best_arch:
        assert isinstance(idx, int)

    assert isinstance(best_score, float)
    # Should have called score_fn exactly n_trials times
    assert call_count["n"] == 5
