"""Tests for src/training/moe_analysis.py."""

from __future__ import annotations

import math

import torch

from src.training.moe_analysis import (
    MoEAnalyzer,
    compute_expert_specialization,
    compute_load_imbalance,
    compute_routing_entropy,
    gini_coefficient,
    track_routing_over_time,
)

N_EXPERTS = 4
N_TOKENS = 12


# ---------------------------------------------------------------------------
# compute_routing_entropy
# ---------------------------------------------------------------------------


def test_compute_routing_entropy_shape():
    """Output shape is (N,) matching the number of tokens."""
    probs = torch.softmax(torch.randn(N_TOKENS, N_EXPERTS), dim=-1)
    entropy = compute_routing_entropy(probs)
    assert entropy.shape == (N_TOKENS,), f"expected ({N_TOKENS},), got {entropy.shape}"


def test_compute_routing_entropy_uniform():
    """Uniform probs → entropy = log(n_experts) for every token."""
    probs = torch.full((N_TOKENS, N_EXPERTS), 1.0 / N_EXPERTS)
    entropy = compute_routing_entropy(probs)
    expected = math.log(N_EXPERTS)
    assert torch.allclose(entropy, torch.full_like(entropy, expected), atol=1e-5), (
        f"expected {expected:.6f}, got {entropy}"
    )


def test_compute_routing_entropy_one_hot():
    """One-hot probs → entropy ≈ 0 for every token."""
    probs = torch.zeros(N_TOKENS, N_EXPERTS)
    probs[:, 0] = 1.0
    entropy = compute_routing_entropy(probs)
    assert (entropy.abs() < 1e-4).all(), f"expected ~0, got {entropy}"


# ---------------------------------------------------------------------------
# compute_expert_specialization
# ---------------------------------------------------------------------------


def test_compute_expert_specialization_shape():
    """Output shape is (n_experts, n_classes)."""
    n_classes = 3
    probs = torch.softmax(torch.randn(N_TOKENS, N_EXPERTS), dim=-1)
    labels = torch.randint(0, n_classes, (N_TOKENS,))
    spec = compute_expert_specialization(probs, labels, n_classes)
    assert spec.shape == (N_EXPERTS, n_classes), (
        f"expected ({N_EXPERTS}, {n_classes}), got {spec.shape}"
    )


def test_compute_expert_specialization_rows_sum_to_one():
    """Each row (expert) sums to 1 after normalization."""
    n_classes = 5
    probs = torch.softmax(torch.randn(16, N_EXPERTS), dim=-1)
    labels = torch.randint(0, n_classes, (16,))
    spec = compute_expert_specialization(probs, labels, n_classes)
    row_sums = spec.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(N_EXPERTS), atol=1e-5), (
        f"row sums not all 1: {row_sums}"
    )


# ---------------------------------------------------------------------------
# compute_load_imbalance
# ---------------------------------------------------------------------------


def test_compute_load_imbalance_keys_present():
    """Result dict contains all required keys."""
    probs = torch.softmax(torch.randn(N_TOKENS, N_EXPERTS), dim=-1)
    result = compute_load_imbalance(probs)
    required = {"load_per_expert", "cv", "max_load", "min_load", "gini"}
    assert required.issubset(result.keys()), f"missing keys: {required - result.keys()}"


def test_compute_load_imbalance_load_per_expert_shape():
    """load_per_expert tensor has shape (n_experts,)."""
    probs = torch.softmax(torch.randn(N_TOKENS, N_EXPERTS), dim=-1)
    result = compute_load_imbalance(probs)
    assert result["load_per_expert"].shape == (N_EXPERTS,)


def test_compute_load_imbalance_uniform_router_cv_near_zero():
    """Uniform router (all experts equally loaded) → CV ≈ 0."""
    probs = torch.full((N_TOKENS, N_EXPERTS), 1.0 / N_EXPERTS)
    result = compute_load_imbalance(probs)
    assert abs(result["cv"]) < 1e-4, f"expected CV ≈ 0, got {result['cv']}"


# ---------------------------------------------------------------------------
# gini_coefficient
# ---------------------------------------------------------------------------


def test_gini_coefficient_all_equal_is_zero():
    """Equal values → Gini = 0."""
    x = torch.ones(N_EXPERTS)
    g = gini_coefficient(x)
    assert abs(g) < 1e-6, f"expected 0.0, got {g}"


def test_gini_coefficient_one_expert_gets_all_near_one():
    """One non-zero entry → Gini ≈ (n-1)/n, approaching 1 for large n."""
    x = torch.zeros(N_EXPERTS)
    x[0] = 1.0
    g = gini_coefficient(x)
    expected = (N_EXPERTS - 1) / N_EXPERTS
    assert abs(g - expected) < 1e-5, f"expected {expected:.6f}, got {g}"


# ---------------------------------------------------------------------------
# track_routing_over_time
# ---------------------------------------------------------------------------


def test_track_routing_over_time_load_trajectory_shape():
    """load_trajectory shape is (n_steps, n_experts)."""
    n_steps = 5
    history = [torch.softmax(torch.randn(N_TOKENS, N_EXPERTS), dim=-1) for _ in range(n_steps)]
    result = track_routing_over_time(history)
    assert result["load_trajectory"].shape == (n_steps, N_EXPERTS), (
        f"expected ({n_steps}, {N_EXPERTS}), got {result['load_trajectory'].shape}"
    )


def test_track_routing_over_time_keys():
    """Result dict contains load_trajectory, mean_cv, load_variance."""
    history = [torch.softmax(torch.randn(8, N_EXPERTS), dim=-1) for _ in range(3)]
    result = track_routing_over_time(history)
    assert "load_trajectory" in result
    assert "mean_cv" in result
    assert "load_variance" in result


def test_track_routing_over_time_load_variance_shape():
    """load_variance shape is (n_experts,)."""
    history = [torch.softmax(torch.randn(8, N_EXPERTS), dim=-1) for _ in range(4)]
    result = track_routing_over_time(history)
    assert result["load_variance"].shape == (N_EXPERTS,)


# ---------------------------------------------------------------------------
# MoEAnalyzer
# ---------------------------------------------------------------------------


def test_moe_analyzer_record_and_summary():
    """record_batch + summary returns expected keys and valid mean_entropy."""
    analyzer = MoEAnalyzer(n_experts=N_EXPERTS)
    probs = torch.softmax(torch.randn(N_TOKENS, N_EXPERTS), dim=-1)
    analyzer.record_batch(probs)
    result = analyzer.summary()
    assert "mean_entropy" in result
    assert "load_imbalance" in result
    assert "n_tokens_seen" in result
    assert result["mean_entropy"] >= 0.0


def test_moe_analyzer_n_tokens_seen_accumulates():
    """n_tokens_seen increments correctly across multiple record_batch calls."""
    analyzer = MoEAnalyzer(n_experts=N_EXPERTS)
    batch_sizes = [8, 10, 12]
    for bs in batch_sizes:
        probs = torch.softmax(torch.randn(bs, N_EXPERTS), dim=-1)
        analyzer.record_batch(probs)
    result = analyzer.summary()
    assert result["n_tokens_seen"] == sum(batch_sizes), (
        f"expected {sum(batch_sizes)}, got {result['n_tokens_seen']}"
    )


def test_moe_analyzer_reset_clears_state():
    """After reset(), n_tokens_seen is 0 and internal buffer is empty."""
    analyzer = MoEAnalyzer(n_experts=N_EXPERTS)
    probs = torch.softmax(torch.randn(N_TOKENS, N_EXPERTS), dim=-1)
    analyzer.record_batch(probs)
    analyzer.reset()
    assert analyzer._n_tokens_seen == 0
    assert len(analyzer._router_probs) == 0


def test_moe_analyzer_summary_after_reset_and_new_batch():
    """Summary after reset only reflects the new batch, not the old one."""
    analyzer = MoEAnalyzer(n_experts=N_EXPERTS)
    # Record a large batch
    probs_old = torch.softmax(torch.randn(100, N_EXPERTS), dim=-1)
    analyzer.record_batch(probs_old)
    analyzer.reset()
    # Record a small batch after reset
    probs_new = torch.softmax(torch.randn(5, N_EXPERTS), dim=-1)
    analyzer.record_batch(probs_new)
    result = analyzer.summary()
    assert result["n_tokens_seen"] == 5, f"expected 5 after reset, got {result['n_tokens_seen']}"
