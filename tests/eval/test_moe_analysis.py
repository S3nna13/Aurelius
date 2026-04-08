"""Tests for src/eval/moe_analysis.py."""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from src.eval.moe_analysis import (
    MoERoutingStats,
    ExpertUtilizationTracker,
    compute_expert_specialization,
    detect_expert_collapse,
    routing_entropy,
    jensen_shannon_divergence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_moe_config():
    """Return a small AureliusConfig suitable for unit tests."""
    from src.model.config import AureliusConfig
    from src.model.moe import MoEConfig
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )
    moe_cfg = MoEConfig(n_experts=4, top_k=2)
    return cfg, moe_cfg


class TinyMoEModel(nn.Module):
    """Minimal wrapper that holds a single SparseMoEFFN for testing."""

    def __init__(self):
        super().__init__()
        from src.model.moe import SparseMoEFFN
        cfg, moe_cfg = make_moe_config()
        self.ffn = SparseMoEFFN(cfg, moe_cfg)

    def forward(self, x: torch.Tensor):
        # x: (B, S, D)
        return self.ffn(x)


def make_routing_stats(expert_loads: list[float], layer_idx: int = 0) -> MoERoutingStats:
    """Build a MoERoutingStats from a list of expert load fractions."""
    load_tensor = torch.tensor(expert_loads, dtype=torch.float32)
    n = len(expert_loads)
    mean_val = load_tensor.mean()
    std_val = load_tensor.std()
    imbalance = (std_val / (mean_val + 1e-10)).item()
    return MoERoutingStats(
        layer_idx=layer_idx,
        n_experts=n,
        n_tokens=100,
        expert_load=load_tensor,
        routing_entropy=routing_entropy(load_tensor / load_tensor.sum()),
        load_imbalance=imbalance,
        top_expert_fraction=load_tensor.max().item(),
    )


# ---------------------------------------------------------------------------
# routing_entropy tests
# ---------------------------------------------------------------------------

def test_routing_entropy_uniform():
    """Uniform distribution should give maximum entropy = log(n_experts)."""
    n = 8
    probs = torch.ones(n) / n
    h = routing_entropy(probs)
    expected = math.log(n)
    assert abs(h - expected) < 1e-4, f"Expected {expected:.4f}, got {h:.4f}"


def test_routing_entropy_concentrated():
    """All weight on one expert should give entropy close to 0."""
    n = 8
    probs = torch.zeros(n)
    probs[0] = 1.0
    h = routing_entropy(probs)
    assert h < 1e-4, f"Expected ~0, got {h}"


# ---------------------------------------------------------------------------
# jensen_shannon_divergence tests
# ---------------------------------------------------------------------------

def test_jsd_identical():
    """JSD(P, P) should be approximately 0."""
    p = torch.tensor([0.1, 0.5, 0.3, 0.1])
    jsd = jensen_shannon_divergence(p, p)
    assert jsd < 1e-5, f"Expected ~0, got {jsd}"


def test_jsd_disjoint():
    """JSD of two non-overlapping distributions should be 1.0."""
    p = torch.tensor([1.0, 0.0, 0.0, 0.0])
    q = torch.tensor([0.0, 0.0, 0.0, 1.0])
    jsd = jensen_shannon_divergence(p, q)
    assert abs(jsd - 1.0) < 1e-3, f"Expected ~1.0, got {jsd}"


# ---------------------------------------------------------------------------
# detect_expert_collapse tests
# ---------------------------------------------------------------------------

def test_detect_collapse_balanced():
    """Uniform routing should not be considered collapsed."""
    n = 4
    uniform_load = 1.0 / n
    stats = [make_routing_stats([uniform_load] * n)]
    result = detect_expert_collapse(stats, collapse_threshold=0.8)
    assert not result["is_collapsed"]
    assert result["collapsed_layers"] == []


def test_detect_collapse_concentrated():
    """One expert getting 95% should trigger collapse detection."""
    loads = [0.95, 0.02, 0.02, 0.01]
    stats = [make_routing_stats(loads)]
    result = detect_expert_collapse(stats, collapse_threshold=0.8)
    assert result["is_collapsed"]
    assert 0 in result["collapsed_layers"]
    assert result["max_single_expert_load"] == pytest.approx(0.95, abs=1e-4)


# ---------------------------------------------------------------------------
# MoERoutingStats imbalance test
# ---------------------------------------------------------------------------

def test_moe_routing_stats_imbalance():
    """High std in expert loads should produce a high load_imbalance value."""
    # Skewed distribution: one expert gets everything
    loads = [0.9, 0.05, 0.03, 0.02]
    stat = make_routing_stats(loads)
    # Compare with balanced
    balanced_stat = make_routing_stats([0.25, 0.25, 0.25, 0.25])
    assert stat.load_imbalance > balanced_stat.load_imbalance


# ---------------------------------------------------------------------------
# ExpertUtilizationTracker tests
# ---------------------------------------------------------------------------

def test_utilization_tracker_attach_detach():
    """attach/detach should not crash on a valid MoE model."""
    model = TinyMoEModel()
    tracker = ExpertUtilizationTracker(model)
    tracker.attach()
    assert len(tracker._hooks) == 1
    tracker.detach()
    assert len(tracker._hooks) == 0


def test_utilization_tracker_records_stats():
    """After a forward pass, get_stats() should be non-empty."""
    model = TinyMoEModel()
    tracker = ExpertUtilizationTracker(model)
    tracker.attach()

    cfg, _ = make_moe_config()
    x = torch.randn(2, 8, cfg.d_model)
    with torch.no_grad():
        model(x)

    stats = tracker.get_stats()
    tracker.detach()

    assert len(stats) >= 1
    s = stats[0]
    assert s.n_experts == 4
    assert s.n_tokens == 2 * 8
    assert s.expert_load.shape == (4,)
    assert abs(s.expert_load.sum().item() - 1.0) < 1e-4


def test_summary_dead_experts():
    """An expert with 0 utilization should appear in the dead_experts list."""
    model = TinyMoEModel()
    tracker = ExpertUtilizationTracker(model)

    # Manually inject a stat with one completely dead expert
    dead_load = torch.tensor([0.50, 0.50, 0.00, 0.00], dtype=torch.float32)
    stat = MoERoutingStats(
        layer_idx=0,
        n_experts=4,
        n_tokens=100,
        expert_load=dead_load,
        routing_entropy=routing_entropy(dead_load),
        load_imbalance=0.5,
        top_expert_fraction=0.50,
    )
    tracker._routing_history.append(stat)

    summary = tracker.summary()
    dead = summary["dead_experts"]

    # Experts 2 and 3 have 0 utilization
    assert (0, 2) in dead
    assert (0, 3) in dead
