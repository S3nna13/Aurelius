"""Tests for src/eval/moe_analysis.py.

Covers MoEAnalysisConfig, simulate_expert_routing, compute_expert_utilization,
compute_load_balance_score, compute_expert_specialization, and MoEAnalyzer.
"""

from __future__ import annotations

import torch

from src.eval.moe_analysis import (
    MoEAnalysisConfig,
    MoEAnalyzer,
    compute_expert_specialization,
    compute_expert_utilization,
    compute_load_balance_score,
    simulate_expert_routing,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_model():
    """Create a small AureliusTransformer for testing."""
    from src.model.config import AureliusConfig
    from src.model.transformer import AureliusTransformer

    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )
    model = AureliusTransformer(cfg)
    model.eval()
    return model


def make_hidden(B=2, T=8, D=64) -> torch.Tensor:
    """Return a random hidden state tensor."""
    return torch.randn(B, T, D)


def make_input_ids(B=2, S=8, vocab_size=256) -> torch.Tensor:
    """Return random token ids."""
    return torch.randint(0, vocab_size, (B, S))


# ---------------------------------------------------------------------------
# 1. MoEAnalysisConfig defaults
# ---------------------------------------------------------------------------


def test_moe_analysis_config_defaults():
    """MoEAnalysisConfig should have correct default values."""
    cfg = MoEAnalysisConfig()
    assert cfg.n_experts == 8
    assert cfg.top_k == 2
    assert cfg.n_samples == 50
    assert cfg.track_layers == []


# ---------------------------------------------------------------------------
# 2. simulate_expert_routing - correct shapes
# ---------------------------------------------------------------------------


def test_simulate_expert_routing_shapes():
    """simulate_expert_routing should return (B, T, top_k) tensors."""
    B, T, D = 2, 8, 64
    n_experts, top_k = 8, 2
    hidden = make_hidden(B, T, D)
    indices, weights = simulate_expert_routing(hidden, n_experts, top_k)
    assert indices.shape == (B, T, top_k), f"indices shape: {indices.shape}"
    assert weights.shape == (B, T, top_k), f"weights shape: {weights.shape}"


# ---------------------------------------------------------------------------
# 3. simulate_expert_routing - top_k indices in [0, n_experts)
# ---------------------------------------------------------------------------


def test_simulate_expert_routing_index_range():
    """All expert indices should be in [0, n_experts)."""
    B, T, D = 3, 10, 32
    n_experts, top_k = 6, 2
    hidden = make_hidden(B, T, D)
    indices, _ = simulate_expert_routing(hidden, n_experts, top_k)
    assert (indices >= 0).all(), "Some indices are negative"
    assert (indices < n_experts).all(), "Some indices exceed n_experts"


# ---------------------------------------------------------------------------
# 4. simulate_expert_routing - weights sum to 1 per token (after softmax top_k)
# ---------------------------------------------------------------------------


def test_simulate_expert_routing_weights_positive():
    """Expert weights should be positive (they come from softmax)."""
    B, T, D = 2, 6, 32
    n_experts, top_k = 4, 2
    hidden = make_hidden(B, T, D)
    _, weights = simulate_expert_routing(hidden, n_experts, top_k)
    assert (weights > 0).all(), "Some weights are non-positive"


# ---------------------------------------------------------------------------
# 5. compute_expert_utilization - returns shape (n_experts,)
# ---------------------------------------------------------------------------


def test_compute_expert_utilization_shape():
    """compute_expert_utilization should return shape (n_experts,)."""
    B, T, top_k, n_experts = 2, 8, 2, 6
    hidden = make_hidden(B, T, 32)
    indices, _ = simulate_expert_routing(hidden, n_experts, top_k)
    util = compute_expert_utilization(indices, n_experts)
    assert util.shape == (n_experts,), f"Expected ({n_experts},), got {util.shape}"


# ---------------------------------------------------------------------------
# 6. compute_expert_utilization - values in [0, 1]
# ---------------------------------------------------------------------------


def test_compute_expert_utilization_values_in_range():
    """Utilization values should be in [0, 1]."""
    B, T, top_k, n_experts = 2, 8, 2, 8
    hidden = make_hidden(B, T, 64)
    indices, _ = simulate_expert_routing(hidden, n_experts, top_k)
    util = compute_expert_utilization(indices, n_experts)
    assert (util >= 0).all(), "Some utilization values are negative"
    assert (util <= 1.0 + 1e-6).all(), f"Some utilization values exceed 1: {util}"


# ---------------------------------------------------------------------------
# 7. compute_load_balance_score - uniform utilization -> score near 1.0
# ---------------------------------------------------------------------------


def test_compute_load_balance_score_uniform():
    """Uniform utilization should yield a load balance score near 1.0."""
    n_experts = 8
    uniform = torch.full((n_experts,), 1.0 / n_experts)
    score = compute_load_balance_score(uniform)
    assert score >= 0.95, f"Expected near 1.0, got {score}"


# ---------------------------------------------------------------------------
# 8. compute_load_balance_score - all-one-expert -> score near 0.0
# ---------------------------------------------------------------------------


def test_compute_load_balance_score_concentrated():
    """Highly concentrated utilization should yield a low load balance score."""
    n_experts = 8
    concentrated = torch.zeros(n_experts)
    concentrated[0] = 1.0
    score = compute_load_balance_score(concentrated)
    assert score <= 0.2, f"Expected near 0.0, got {score}"


# ---------------------------------------------------------------------------
# 9. compute_expert_specialization - returns shape (n_experts,)
# ---------------------------------------------------------------------------


def test_compute_expert_specialization_shape():
    """compute_expert_specialization should return shape (n_experts,)."""
    B, T, n_experts, top_k = 2, 8, 6, 2
    hidden = make_hidden(B, T, 32)
    indices, _ = simulate_expert_routing(hidden, n_experts, top_k)
    token_positions = torch.arange(T)
    spec = compute_expert_specialization(indices, token_positions, n_experts)
    assert spec.shape == (n_experts,), f"Expected ({n_experts},), got {spec.shape}"


# ---------------------------------------------------------------------------
# 10. compute_expert_specialization - values in [0, 1]
# ---------------------------------------------------------------------------


def test_compute_expert_specialization_values_in_range():
    """Specialization scores should be in [0, 1]."""
    B, T, n_experts, top_k = 2, 12, 8, 2
    hidden = make_hidden(B, T, 64)
    indices, _ = simulate_expert_routing(hidden, n_experts, top_k)
    token_positions = torch.arange(T)
    spec = compute_expert_specialization(indices, token_positions, n_experts)
    assert (spec >= -1e-6).all(), f"Some specialization values are negative: {spec}"
    assert (spec <= 1.0 + 1e-6).all(), f"Some specialization values exceed 1: {spec}"


# ---------------------------------------------------------------------------
# 11. MoEAnalyzer.register_hooks - returns handles
# ---------------------------------------------------------------------------


def test_moe_analyzer_register_hooks_returns_handles():
    """register_hooks should return a non-empty list of hook handles."""
    model = make_model()
    cfg = MoEAnalysisConfig(n_experts=4, top_k=2)
    analyzer = MoEAnalyzer(model, cfg)
    handles = analyzer.register_hooks()
    assert isinstance(handles, list), "register_hooks should return a list"
    assert len(handles) > 0, "register_hooks should return at least one handle"
    analyzer.remove_hooks(handles)


# ---------------------------------------------------------------------------
# 12. MoEAnalyzer.analyze_batch - returns dict with layer keys
# ---------------------------------------------------------------------------


def test_moe_analyzer_analyze_batch_layer_keys():
    """analyze_batch should return a dict keyed by layer indices."""
    model = make_model()
    cfg = MoEAnalysisConfig(n_experts=4, top_k=2)
    analyzer = MoEAnalyzer(model, cfg)
    handles = analyzer.register_hooks()
    input_ids = make_input_ids(B=2, S=8)
    results = analyzer.analyze_batch(input_ids)
    analyzer.remove_hooks(handles)
    assert isinstance(results, dict), "analyze_batch should return a dict"
    assert len(results) > 0, "analyze_batch should return results for at least one layer"
    for key in results:
        assert isinstance(key, int), f"Layer key should be int, got {type(key)}"


# ---------------------------------------------------------------------------
# 13. MoEAnalyzer.analyze_batch - each value has "load_balance" key
# ---------------------------------------------------------------------------


def test_moe_analyzer_analyze_batch_has_load_balance():
    """Each layer result in analyze_batch should contain 'load_balance' key."""
    model = make_model()
    cfg = MoEAnalysisConfig(n_experts=4, top_k=2)
    analyzer = MoEAnalyzer(model, cfg)
    handles = analyzer.register_hooks()
    input_ids = make_input_ids(B=2, S=8)
    results = analyzer.analyze_batch(input_ids)
    analyzer.remove_hooks(handles)
    for layer_idx, stats in results.items():
        assert "load_balance" in stats, (
            f"Layer {layer_idx} result missing 'load_balance' key: {list(stats.keys())}"
        )
        assert "utilization" in stats, f"Layer {layer_idx} result missing 'utilization' key"
        assert "mean_expert_stats" in stats, (
            f"Layer {layer_idx} result missing 'mean_expert_stats' key"
        )


# ---------------------------------------------------------------------------
# 14. MoEAnalyzer.run_analysis - returns required keys
# ---------------------------------------------------------------------------


def test_moe_analyzer_run_analysis_required_keys():
    """run_analysis should return a dict with all required keys."""
    model = make_model()
    cfg = MoEAnalysisConfig(n_experts=4, top_k=2)
    analyzer = MoEAnalyzer(model, cfg)
    dataset = [make_input_ids(B=2, S=8) for _ in range(3)]
    results = analyzer.run_analysis(dataset)
    required_keys = {
        "mean_load_balance",
        "per_layer_balance",
        "most_used_expert",
        "least_used_expert",
    }
    for key in required_keys:
        assert key in results, f"Missing required key '{key}' in run_analysis output"


# ---------------------------------------------------------------------------
# 15. MoEAnalyzer.run_analysis - mean_load_balance in [0, 1]
# ---------------------------------------------------------------------------


def test_moe_analyzer_run_analysis_mean_load_balance_range():
    """mean_load_balance in run_analysis output should be in [0, 1]."""
    model = make_model()
    cfg = MoEAnalysisConfig(n_experts=4, top_k=2)
    analyzer = MoEAnalyzer(model, cfg)
    dataset = [make_input_ids(B=2, S=8) for _ in range(3)]
    results = analyzer.run_analysis(dataset)
    lb = results["mean_load_balance"]
    assert 0.0 <= lb <= 1.0, f"mean_load_balance {lb} not in [0, 1]"
