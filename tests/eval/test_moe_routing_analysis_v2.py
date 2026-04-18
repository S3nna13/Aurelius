"""Tests for src/eval/moe_routing_analysis_v2.py.

Covers RouterStats, ExpertActivationTracker, RoutingDiversityMetrics,
MoEDiagnostics, and MoEAnalysisConfig.

Test matrix uses n_experts=4, top_k=2, d=16, N=32, B=2, T=8.
"""
from __future__ import annotations

import math

import pytest
import torch

from src.eval.moe_routing_analysis_v2 import (
    ExpertActivationTracker,
    MoEAnalysisConfig,
    MoEDiagnostics,
    RouterStats,
    RoutingDiversityMetrics,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

N_EXPERTS = 4
TOP_K = 2
D = 16
N = 32   # B*T tokens
B = 2
T = 8    # B*T = 16; we use N=32 for standalone tensors


def _uniform_routing(n: int = N) -> torch.Tensor:
    """Uniform distribution over N_EXPERTS for each of n tokens."""
    return torch.full((n, N_EXPERTS), 1.0 / N_EXPERTS)


def _random_routing(n: int = N) -> torch.Tensor:
    """Random softmax routing weights."""
    return torch.softmax(torch.randn(n, N_EXPERTS), dim=-1)


def _topk_indices(routing: torch.Tensor) -> torch.Tensor:
    """Return top-k expert indices from a routing weight matrix."""
    return torch.topk(routing, TOP_K, dim=-1).indices


def _make_router_stats_with_data() -> RouterStats:
    """Build a RouterStats with two accumulated batches of random data."""
    rs = RouterStats(n_experts=N_EXPERTS, top_k=TOP_K)
    for _ in range(2):
        w = _random_routing()
        idx = _topk_indices(w)
        rs.update(w, idx)
    return rs


def _make_tracker_with_data() -> ExpertActivationTracker:
    tracker = ExpertActivationTracker(n_experts=N_EXPERTS)
    expert_ids = torch.randint(0, N_EXPERTS, (N, TOP_K))
    tracker.record_batch(expert_ids)
    return tracker


# ---------------------------------------------------------------------------
# RouterStats tests
# ---------------------------------------------------------------------------

class TestRouterStatsUpdate:
    def test_update_increments_total_tokens(self):
        rs = RouterStats(n_experts=N_EXPERTS, top_k=TOP_K)
        w = _random_routing(N)
        idx = _topk_indices(w)
        rs.update(w, idx)
        assert rs._total_tokens == N

    def test_update_accumulates_across_batches(self):
        rs = RouterStats(n_experts=N_EXPERTS, top_k=TOP_K)
        w1 = _random_routing(10)
        w2 = _random_routing(6)
        rs.update(w1, _topk_indices(w1))
        rs.update(w2, _topk_indices(w2))
        assert rs._total_tokens == 16

    def test_update_token_counts_non_negative(self):
        rs = _make_router_stats_with_data()
        assert (rs._expert_token_counts >= 0).all()

    def test_update_total_assignments_equals_n_times_topk(self):
        """Sum of per-expert token counts must equal N * TOP_K."""
        rs = RouterStats(n_experts=N_EXPERTS, top_k=TOP_K)
        w = _random_routing(N)
        idx = _topk_indices(w)
        rs.update(w, idx)
        assert rs._expert_token_counts.sum().item() == pytest.approx(N * TOP_K, abs=1e-3)


class TestRouterStatsUtilization:
    def test_expert_utilization_sums_to_one(self):
        """Utilization fractions should sum to 1.0 (top_k assignments / total)."""
        rs = _make_router_stats_with_data()
        util = rs.expert_utilization()
        assert util.sum().item() == pytest.approx(1.0, abs=1e-4)

    def test_expert_utilization_shape(self):
        rs = _make_router_stats_with_data()
        assert rs.expert_utilization().shape == (N_EXPERTS,)

    def test_expert_utilization_non_negative(self):
        rs = _make_router_stats_with_data()
        assert (rs.expert_utilization() >= 0).all()

    def test_expert_utilization_zero_before_update(self):
        rs = RouterStats(n_experts=N_EXPERTS, top_k=TOP_K)
        assert (rs.expert_utilization() == 0).all()


class TestRouterStatsLoss:
    def test_load_balance_loss_non_negative(self):
        rs = _make_router_stats_with_data()
        assert rs.load_balance_loss() >= 0.0

    def test_load_balance_loss_zero_before_update(self):
        rs = RouterStats(n_experts=N_EXPERTS, top_k=TOP_K)
        assert rs.load_balance_loss() == 0.0

    def test_load_balance_loss_is_float(self):
        rs = _make_router_stats_with_data()
        assert isinstance(rs.load_balance_loss(), float)


class TestRouterStatsEntropy:
    def test_routing_entropy_non_negative(self):
        rs = _make_router_stats_with_data()
        assert rs.routing_entropy() >= 0.0

    def test_routing_entropy_zero_before_update(self):
        rs = RouterStats(n_experts=N_EXPERTS, top_k=TOP_K)
        assert rs.routing_entropy() == 0.0

    def test_routing_entropy_uniform_is_max(self):
        """Uniform routing should yield the maximum entropy ln(n_experts)."""
        rs = RouterStats(n_experts=N_EXPERTS, top_k=TOP_K)
        w = _uniform_routing(N)
        idx = _topk_indices(w)
        rs.update(w, idx)
        expected_max = math.log(N_EXPERTS)
        assert rs.routing_entropy() == pytest.approx(expected_max, abs=1e-4)


class TestRouterStatsCollapse:
    def test_collapse_score_in_unit_interval(self):
        rs = _make_router_stats_with_data()
        cs = rs.collapse_score()
        assert 0.0 <= cs <= 1.0

    def test_collapse_score_is_float(self):
        rs = _make_router_stats_with_data()
        assert isinstance(rs.collapse_score(), float)

    def test_collapse_score_zero_when_all_experts_used_equally(self):
        """Force perfectly balanced usage — no expert should be below 1 %."""
        rs = RouterStats(n_experts=N_EXPERTS, top_k=TOP_K)
        # construct routing that forces exactly balanced expert selection
        bt = N_EXPERTS * TOP_K  # 8 tokens, each assigned to one expert slot
        w = _uniform_routing(bt)
        # manually craft indices: token i selects expert (i % N_EXPERTS)
        idx = torch.zeros(bt, TOP_K, dtype=torch.long)
        for i in range(bt):
            idx[i, 0] = i % N_EXPERTS
            idx[i, 1] = (i + 1) % N_EXPERTS
        rs.update(w, idx)
        assert rs.collapse_score() == pytest.approx(0.0, abs=1e-6)


class TestRouterStatsReset:
    def test_reset_clears_token_counts(self):
        rs = _make_router_stats_with_data()
        rs.reset()
        assert rs._expert_token_counts.sum().item() == 0.0

    def test_reset_clears_total_tokens(self):
        rs = _make_router_stats_with_data()
        rs.reset()
        assert rs._total_tokens == 0

    def test_reset_allows_fresh_accumulation(self):
        rs = _make_router_stats_with_data()
        rs.reset()
        w = _random_routing(5)
        rs.update(w, _topk_indices(w))
        assert rs._total_tokens == 5


# ---------------------------------------------------------------------------
# ExpertActivationTracker tests
# ---------------------------------------------------------------------------

class TestExpertActivationTracker:
    def test_co_activation_matrix_shape(self):
        tracker = _make_tracker_with_data()
        mat = tracker.co_activation_matrix()
        assert mat.shape == (N_EXPERTS, N_EXPERTS)

    def test_co_activation_matrix_diagonal_ge_off_diagonal(self):
        """Diagonal (self-activation) must be >= any off-diagonal entry."""
        tracker = _make_tracker_with_data()
        mat = tracker.co_activation_matrix()
        diag = mat.diagonal()
        for i in range(N_EXPERTS):
            for j in range(N_EXPERTS):
                if i != j:
                    assert diag[i].item() >= mat[i, j].item() - 1e-6, (
                        f"diag[{i}]={diag[i].item():.4f} < off-diag[{i},{j}]={mat[i,j].item():.4f}"
                    )

    def test_co_activation_matrix_non_negative(self):
        tracker = _make_tracker_with_data()
        assert (tracker.co_activation_matrix() >= 0).all()

    def test_co_activation_matrix_symmetric(self):
        """Co-activation is symmetric: expert i & j == expert j & i."""
        tracker = _make_tracker_with_data()
        mat = tracker.co_activation_matrix()
        assert torch.allclose(mat, mat.T, atol=1e-6)

    def test_expert_specialization_shape(self):
        tracker = ExpertActivationTracker(n_experts=N_EXPERTS)
        features = torch.randn(N, D)
        expert_ids = torch.randint(0, N_EXPERTS, (N,))
        spec = tracker.expert_specialization(features, expert_ids)
        assert spec.shape == (N_EXPERTS, D)

    def test_expert_specialization_centroid_correctness(self):
        """Centroid of expert 0 should match manual mean computation."""
        tracker = ExpertActivationTracker(n_experts=N_EXPERTS)
        features = torch.randn(N, D)
        expert_ids = torch.zeros(N, dtype=torch.long)  # all tokens → expert 0
        spec = tracker.expert_specialization(features, expert_ids)
        expected = features.mean(dim=0)
        assert torch.allclose(spec[0], expected, atol=1e-5)

    def test_top_tokens_per_expert_returns_dict(self):
        tracker = ExpertActivationTracker(n_experts=N_EXPERTS)
        token_ids = torch.randint(0, 50, (N,))
        expert_ids = torch.randint(0, N_EXPERTS, (N,))
        result = tracker.top_tokens_per_expert(token_ids, expert_ids, k=5)
        assert isinstance(result, dict)

    def test_top_tokens_per_expert_all_experts_in_dict(self):
        tracker = ExpertActivationTracker(n_experts=N_EXPERTS)
        token_ids = torch.randint(0, 50, (N,))
        expert_ids = torch.randint(0, N_EXPERTS, (N,))
        result = tracker.top_tokens_per_expert(token_ids, expert_ids, k=5)
        for e in range(N_EXPERTS):
            assert e in result

    def test_top_tokens_per_expert_length_bounded_by_k(self):
        tracker = ExpertActivationTracker(n_experts=N_EXPERTS)
        token_ids = torch.randint(0, 100, (N,))
        expert_ids = torch.randint(0, N_EXPERTS, (N,))
        k = 3
        result = tracker.top_tokens_per_expert(token_ids, expert_ids, k=k)
        for e in range(N_EXPERTS):
            assert len(result[e]) <= k


# ---------------------------------------------------------------------------
# RoutingDiversityMetrics tests
# ---------------------------------------------------------------------------

class TestRoutingDiversityMetrics:
    def test_token_routing_entropy_shape(self):
        w = _random_routing(N)
        ent = RoutingDiversityMetrics.token_routing_entropy(w)
        assert ent.shape == (N,)

    def test_token_routing_entropy_non_negative(self):
        w = _random_routing(N)
        ent = RoutingDiversityMetrics.token_routing_entropy(w)
        assert (ent >= 0).all()

    def test_token_routing_entropy_uniform_is_max(self):
        """Uniform distribution should achieve maximum entropy ln(n_experts)."""
        w = _uniform_routing(N)
        ent = RoutingDiversityMetrics.token_routing_entropy(w)
        expected = math.log(N_EXPERTS)
        assert ent.mean().item() == pytest.approx(expected, abs=1e-4)

    def test_token_routing_entropy_peaked_is_low(self):
        """One-hot routing should give near-zero entropy."""
        w = torch.zeros(N, N_EXPERTS)
        w[:, 0] = 1.0  # all tokens → expert 0
        ent = RoutingDiversityMetrics.token_routing_entropy(w)
        assert ent.mean().item() < 0.01

    def test_expert_similarity_shape(self):
        ew = torch.randn(N_EXPERTS, D)
        sim = RoutingDiversityMetrics.expert_similarity(ew)
        assert sim.shape == (N_EXPERTS, N_EXPERTS)

    def test_expert_similarity_diagonal_is_one(self):
        """Cosine similarity of a vector with itself is 1."""
        ew = torch.randn(N_EXPERTS, D)
        sim = RoutingDiversityMetrics.expert_similarity(ew)
        assert torch.allclose(sim.diagonal(), torch.ones(N_EXPERTS), atol=1e-5)

    def test_expert_similarity_range(self):
        ew = torch.randn(N_EXPERTS, D)
        sim = RoutingDiversityMetrics.expert_similarity(ew)
        assert (sim >= -1.0 - 1e-5).all()
        assert (sim <= 1.0 + 1e-5).all()

    def test_routing_consistency_in_unit_interval(self):
        w1 = _random_routing(N)
        w2 = _random_routing(N)
        rc = RoutingDiversityMetrics.routing_consistency(w1, w2)
        assert 0.0 <= rc <= 1.0

    def test_routing_consistency_identical_is_one(self):
        w = _random_routing(N)
        rc = RoutingDiversityMetrics.routing_consistency(w, w)
        assert rc == pytest.approx(1.0, abs=1e-5)

    def test_routing_consistency_is_float(self):
        w1 = _random_routing(N)
        w2 = _random_routing(N)
        assert isinstance(RoutingDiversityMetrics.routing_consistency(w1, w2), float)


# ---------------------------------------------------------------------------
# MoEDiagnostics tests
# ---------------------------------------------------------------------------

REQUIRED_REPORT_KEYS = {
    "utilization_std",
    "load_balance_loss",
    "routing_entropy",
    "collapse_score",
    "mean_co_activation",
}


def _make_diagnostics() -> MoEDiagnostics:
    rs = _make_router_stats_with_data()
    tracker = _make_tracker_with_data()
    return MoEDiagnostics(rs, tracker)


class TestMoEDiagnostics:
    def test_full_report_returns_dict(self):
        diag = _make_diagnostics()
        assert isinstance(diag.full_report(), dict)

    def test_full_report_has_all_required_keys(self):
        diag = _make_diagnostics()
        report = diag.full_report()
        assert REQUIRED_REPORT_KEYS.issubset(report.keys())

    def test_full_report_values_are_floats(self):
        diag = _make_diagnostics()
        for k, v in diag.full_report().items():
            assert isinstance(v, float), f"Key '{k}' is {type(v)}, expected float"

    def test_full_report_utilization_std_non_negative(self):
        diag = _make_diagnostics()
        assert diag.full_report()["utilization_std"] >= 0.0

    def test_full_report_collapse_score_in_unit_interval(self):
        diag = _make_diagnostics()
        cs = diag.full_report()["collapse_score"]
        assert 0.0 <= cs <= 1.0

    def test_detect_issues_returns_list(self):
        diag = _make_diagnostics()
        issues = diag.detect_issues()
        assert isinstance(issues, list)

    def test_detect_issues_load_imbalance_triggered(self):
        """Force routing collapse onto expert 0 to trigger LOAD_IMBALANCE."""
        rs = RouterStats(n_experts=N_EXPERTS, top_k=TOP_K)
        w = torch.zeros(N, N_EXPERTS)
        w[:, 0] = 1.0  # all weight on expert 0
        idx = torch.zeros(N, TOP_K, dtype=torch.long)  # both slots → expert 0
        rs.update(w, idx)
        tracker = ExpertActivationTracker(n_experts=N_EXPERTS)
        tracker.record_batch(idx)
        diag = MoEDiagnostics(rs, tracker)
        assert "LOAD_IMBALANCE" in diag.detect_issues()

    def test_detect_issues_routing_collapse_triggered(self):
        """Route > 50 % of experts to a single expert to trigger ROUTING_COLLAPSE."""
        rs = RouterStats(n_experts=N_EXPERTS, top_k=TOP_K)
        # Only expert 0 ever selected — 3 of 4 experts are dead (75 %)
        w = torch.zeros(N, N_EXPERTS)
        w[:, 0] = 1.0
        idx = torch.zeros(N, TOP_K, dtype=torch.long)
        rs.update(w, idx)
        tracker = ExpertActivationTracker(n_experts=N_EXPERTS)
        tracker.record_batch(idx)
        diag = MoEDiagnostics(rs, tracker)
        assert "ROUTING_COLLAPSE" in diag.detect_issues()

    def test_detect_issues_low_entropy_triggered(self):
        """One-hot routing should produce near-zero entropy → LOW_ENTROPY warning."""
        rs = RouterStats(n_experts=N_EXPERTS, top_k=TOP_K)
        w = torch.zeros(N, N_EXPERTS)
        w[:, 0] = 1.0
        idx = torch.zeros(N, TOP_K, dtype=torch.long)
        rs.update(w, idx)
        tracker = ExpertActivationTracker(n_experts=N_EXPERTS)
        tracker.record_batch(idx)
        diag = MoEDiagnostics(rs, tracker)
        assert "LOW_ENTROPY" in diag.detect_issues()

    def test_detect_issues_no_warnings_for_balanced_routing(self):
        """Perfectly balanced uniform routing should produce no warnings."""
        rs = RouterStats(n_experts=N_EXPERTS, top_k=TOP_K)
        w = _uniform_routing(N)
        idx = _topk_indices(w)
        rs.update(w, idx)
        tracker = ExpertActivationTracker(n_experts=N_EXPERTS)
        # record balanced expert ids
        tracker.record_batch(idx)
        diag = MoEDiagnostics(rs, tracker)
        issues = diag.detect_issues()
        # Uniform routing should not trigger LOW_ENTROPY (entropy = ln(4) ≈ 1.39)
        assert "LOW_ENTROPY" not in issues


# ---------------------------------------------------------------------------
# MoEAnalysisConfig tests
# ---------------------------------------------------------------------------

class TestMoEAnalysisConfig:
    def test_default_values(self):
        cfg = MoEAnalysisConfig()
        assert cfg.n_experts == 8
        assert cfg.top_k == 2
        assert cfg.collapse_threshold == pytest.approx(0.01)
        assert cfg.entropy_threshold == pytest.approx(0.5)
        assert cfg.imbalance_threshold == pytest.approx(0.3)

    def test_custom_values(self):
        cfg = MoEAnalysisConfig(n_experts=16, top_k=4)
        assert cfg.n_experts == 16
        assert cfg.top_k == 4
