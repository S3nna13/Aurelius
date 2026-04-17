"""Tests for src/eval/moe_routing_analysis.py.

Covers RoutingStats, RoutingEntropyAnalyzer, ExpertSpecializationAnalyzer,
and RoutingDiagnostics.  Uses n_experts=4, N=20 throughout.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from src.eval.moe_routing_analysis import (
    RoutingStats,
    RoutingEntropyAnalyzer,
    ExpertSpecializationAnalyzer,
    RoutingDiagnostics,
)

N_EXPERTS = 4
N = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_weights(n: int = N, n_experts: int = N_EXPERTS) -> torch.Tensor:
    """Uniform routing — every expert gets equal probability."""
    return torch.full((n, n_experts), 1.0 / n_experts)


def _one_hot_weights(expert_idx: int, n: int = N, n_experts: int = N_EXPERTS) -> torch.Tensor:
    """All tokens routed deterministically to *expert_idx*."""
    w = torch.zeros(n, n_experts)
    w[:, expert_idx] = 1.0
    return w


def _balanced_weights(n: int = N, n_experts: int = N_EXPERTS) -> torch.Tensor:
    """Perfectly balanced hard assignment (each expert gets exactly n//n_experts tokens)."""
    assert n % n_experts == 0, "n must be divisible by n_experts for perfect balance"
    w = torch.zeros(n, n_experts)
    per_expert = n // n_experts
    for e in range(n_experts):
        w[e * per_expert:(e + 1) * per_expert, e] = 1.0
    return w


# ---------------------------------------------------------------------------
# RoutingStats tests (1–7)
# ---------------------------------------------------------------------------

def test_routing_stats_total_tokens_increments():
    """update() increments total_tokens by the batch size."""
    stats = RoutingStats(N_EXPERTS)
    w = _uniform_weights()
    stats.update(w)
    assert stats.total_tokens == N


def test_expert_counts_sum_to_total_tokens():
    """expert_counts sums to total_tokens after one update."""
    stats = RoutingStats(N_EXPERTS)
    w = _uniform_weights()
    stats.update(w)
    assert stats.expert_counts.sum().item() == stats.total_tokens


def test_load_balance_score_zero_for_balanced():
    """load_balance_score() = 0 when routing is perfectly balanced."""
    stats = RoutingStats(N_EXPERTS)
    stats.update(_balanced_weights())
    assert stats.load_balance_score() == pytest.approx(0.0, abs=1e-6)


def test_load_balance_score_positive_for_imbalanced():
    """load_balance_score() > 0 when routing is imbalanced."""
    stats = RoutingStats(N_EXPERTS)
    # Send all tokens to expert 0.
    stats.update(_one_hot_weights(0))
    assert stats.load_balance_score() > 0.0


def test_router_collapse_true_when_single_expert():
    """router_collapse() is True when one expert receives all tokens."""
    stats = RoutingStats(N_EXPERTS)
    stats.update(_one_hot_weights(0))
    assert stats.router_collapse() is True


def test_router_collapse_false_for_balanced():
    """router_collapse() is False for perfectly balanced routing."""
    stats = RoutingStats(N_EXPERTS)
    stats.update(_balanced_weights())
    assert stats.router_collapse() is False


def test_expert_utilization_sums_to_one():
    """expert_utilization() should sum to approximately 1.0."""
    stats = RoutingStats(N_EXPERTS)
    stats.update(_balanced_weights())
    total = stats.expert_utilization().sum().item()
    assert total == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# RoutingEntropyAnalyzer tests (8–11)
# ---------------------------------------------------------------------------

def test_token_entropy_shape():
    """token_entropy() returns a (N,) tensor."""
    analyzer = RoutingEntropyAnalyzer()
    w = _uniform_weights()
    H = analyzer.token_entropy(w)
    assert H.shape == (N,)


def test_entropy_zero_for_deterministic_routing():
    """Entropy is (near) 0 for one-hot routing weights."""
    analyzer = RoutingEntropyAnalyzer()
    w = _one_hot_weights(2)
    H = analyzer.token_entropy(w)
    assert H.max().item() == pytest.approx(0.0, abs=1e-4)


def test_entropy_equals_log_n_experts_for_uniform():
    """Entropy equals log(n_experts) for uniform routing."""
    analyzer = RoutingEntropyAnalyzer()
    w = _uniform_weights()
    expected = math.log(N_EXPERTS)
    H = analyzer.token_entropy(w)
    assert H.mean().item() == pytest.approx(expected, abs=1e-5)


def test_entropy_efficiency_one_for_uniform():
    """entropy_efficiency() ≈ 1.0 for perfectly uniform routing."""
    analyzer = RoutingEntropyAnalyzer()
    w = _uniform_weights()
    eff = analyzer.entropy_efficiency(w)
    assert eff == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# ExpertSpecializationAnalyzer tests (12–13)
# ---------------------------------------------------------------------------

def test_specialization_update_accumulates_co_occurrence():
    """update() accumulates non-zero entries in co_occurrence."""
    analyzer = ExpertSpecializationAnalyzer(n_experts=N_EXPERTS, n_token_types=2)
    w = _balanced_weights()
    # Alternate token types: first half = type 0, second half = type 1.
    token_types = torch.cat([
        torch.zeros(N // 2, dtype=torch.long),
        torch.ones(N // 2, dtype=torch.long),
    ])
    analyzer.update(w, token_types)
    # Some counts must be non-zero.
    assert analyzer.co_occurrence.sum().item() > 0
    assert analyzer.co_occurrence.shape == (2, N_EXPERTS)


def test_specialization_score_in_range():
    """specialization_score() is in [0, 1]."""
    analyzer = ExpertSpecializationAnalyzer(n_experts=N_EXPERTS, n_token_types=4)
    w = torch.rand(N, N_EXPERTS)
    w = w / w.sum(dim=-1, keepdim=True)  # normalise to simplex
    token_types = torch.arange(N) % 4
    analyzer.update(w, token_types)
    score = analyzer.specialization_score()
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# RoutingDiagnostics tests (14–15)
# ---------------------------------------------------------------------------

def test_routing_diagnostics_returns_expected_keys():
    """analyze() returns a dict with all required keys."""
    diag = RoutingDiagnostics(N_EXPERTS)
    w = _uniform_weights()
    result = diag.analyze(w)
    expected_keys = {
        "load_balance_score",
        "router_collapse",
        "mean_entropy",
        "entropy_efficiency",
        "min_expert_fraction",
        "max_expert_fraction",
    }
    assert set(result.keys()) == expected_keys


def test_routing_diagnostics_uniform_entropy_efficiency():
    """entropy_efficiency ≈ 1.0 for uniform routing weights."""
    diag = RoutingDiagnostics(N_EXPERTS)
    w = _uniform_weights()
    result = diag.analyze(w)
    assert result["entropy_efficiency"] == pytest.approx(1.0, abs=1e-5)
