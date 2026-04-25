"""
Tests for src/interpretability/attention_flow.py

Covers: AttentionFlow dataclass, AttentionFlowAnalyzer, and ATTENTION_FLOW_REGISTRY.
Minimum 28 tests.
"""

from __future__ import annotations

import pytest

from src.interpretability.attention_flow import (
    AttentionFlow,
    AttentionFlowAnalyzer,
    ATTENTION_FLOW_REGISTRY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_weights(n: int) -> list[list[float]]:
    """n x n uniform attention matrix (each row sums to 1)."""
    val = 1.0 / n
    return [[val] * n for _ in range(n)]


def _identity_weights(n: int) -> list[list[float]]:
    """n x n identity attention matrix."""
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _zero_weights(n: int) -> list[list[float]]:
    """n x n all-zero matrix."""
    return [[0.0] * n for _ in range(n)]


def _make_analyzer(num_layers: int = 3, num_heads: int = 4) -> AttentionFlowAnalyzer:
    return AttentionFlowAnalyzer(num_layers=num_layers, num_heads=num_heads)


# ---------------------------------------------------------------------------
# 1. AttentionFlow dataclass — basic fields
# ---------------------------------------------------------------------------

class TestAttentionFlowDataclass:
    def test_fields_stored_correctly(self):
        af = AttentionFlow(layer=0, head=1, from_pos=2, to_pos=3, weight=0.75)
        assert af.layer == 0
        assert af.head == 1
        assert af.from_pos == 2
        assert af.to_pos == 3
        assert af.weight == 0.75

    def test_frozen_layer(self):
        af = AttentionFlow(layer=0, head=0, from_pos=0, to_pos=0, weight=0.5)
        with pytest.raises((AttributeError, TypeError)):
            af.layer = 99  # type: ignore[misc]

    def test_frozen_weight(self):
        af = AttentionFlow(layer=0, head=0, from_pos=0, to_pos=0, weight=0.5)
        with pytest.raises((AttributeError, TypeError)):
            af.weight = 0.99  # type: ignore[misc]

    def test_frozen_from_pos(self):
        af = AttentionFlow(layer=1, head=2, from_pos=3, to_pos=4, weight=0.1)
        with pytest.raises((AttributeError, TypeError)):
            af.from_pos = 0  # type: ignore[misc]

    def test_equality(self):
        af1 = AttentionFlow(layer=0, head=0, from_pos=0, to_pos=1, weight=0.3)
        af2 = AttentionFlow(layer=0, head=0, from_pos=0, to_pos=1, weight=0.3)
        assert af1 == af2

    def test_inequality_different_weight(self):
        af1 = AttentionFlow(layer=0, head=0, from_pos=0, to_pos=1, weight=0.3)
        af2 = AttentionFlow(layer=0, head=0, from_pos=0, to_pos=1, weight=0.4)
        assert af1 != af2


# ---------------------------------------------------------------------------
# 2. AttentionFlowAnalyzer — constructor
# ---------------------------------------------------------------------------

class TestAnalyzerConstructor:
    def test_num_layers_stored(self):
        a = AttentionFlowAnalyzer(num_layers=6, num_heads=8)
        assert a.num_layers == 6

    def test_num_heads_stored(self):
        a = AttentionFlowAnalyzer(num_layers=6, num_heads=8)
        assert a.num_heads == 8

    def test_starts_empty(self):
        a = _make_analyzer()
        assert a.top_flows(n=100) == []


# ---------------------------------------------------------------------------
# 3. record — return type and filtering
# ---------------------------------------------------------------------------

class TestRecord:
    def test_record_returns_list(self):
        a = _make_analyzer()
        result = a.record(0, 0, _uniform_weights(3))
        assert isinstance(result, list)

    def test_record_returns_attention_flow_objects(self):
        a = _make_analyzer()
        result = a.record(0, 0, _uniform_weights(3))
        assert all(isinstance(f, AttentionFlow) for f in result)

    def test_record_zero_weights_returns_empty(self):
        a = _make_analyzer()
        result = a.record(0, 0, _zero_weights(4))
        assert result == []

    def test_record_filters_zero_weights(self):
        a = _make_analyzer()
        weights = [[0.5, 0.0], [0.0, 0.5]]
        result = a.record(0, 0, weights)
        # Only the two non-zero entries should produce flows
        assert len(result) == 2
        for f in result:
            assert f.weight > 0.0

    def test_record_identity_n_flows_equals_n(self):
        n = 4
        a = _make_analyzer()
        result = a.record(0, 0, _identity_weights(n))
        assert len(result) == n

    def test_record_uniform_n_flows_equals_n_squared(self):
        n = 3
        a = _make_analyzer()
        result = a.record(0, 0, _uniform_weights(n))
        assert len(result) == n * n

    def test_record_sets_layer_and_head(self):
        a = _make_analyzer()
        result = a.record(2, 3, _identity_weights(2))
        for f in result:
            assert f.layer == 2
            assert f.head == 3

    def test_record_sets_from_pos_and_to_pos(self):
        a = _make_analyzer()
        weights = [[0.8, 0.2], [0.1, 0.9]]
        result = a.record(0, 0, weights)
        pairs = {(f.from_pos, f.to_pos) for f in result}
        assert (0, 0) in pairs
        assert (0, 1) in pairs
        assert (1, 0) in pairs
        assert (1, 1) in pairs

    def test_record_weight_value_correct(self):
        a = _make_analyzer()
        weights = [[0.6, 0.4]]
        result = a.record(0, 0, weights)
        weight_vals = sorted([f.weight for f in result])
        assert abs(weight_vals[0] - 0.4) < 1e-9
        assert abs(weight_vals[1] - 0.6) < 1e-9

    def test_record_accumulates_across_calls(self):
        a = _make_analyzer()
        a.record(0, 0, _uniform_weights(2))
        a.record(1, 0, _uniform_weights(2))
        assert len(a.top_flows(n=100)) == 8  # 4 + 4

    def test_record_returns_only_current_call_flows(self):
        a = _make_analyzer()
        a.record(0, 0, _uniform_weights(2))
        result2 = a.record(1, 0, _identity_weights(3))
        assert len(result2) == 3  # only current call


# ---------------------------------------------------------------------------
# 4. top_flows
# ---------------------------------------------------------------------------

class TestTopFlows:
    def test_top_flows_returns_list(self):
        a = _make_analyzer()
        a.record(0, 0, _uniform_weights(3))
        assert isinstance(a.top_flows(), list)

    def test_top_flows_sorted_descending(self):
        a = _make_analyzer()
        weights = [[0.1, 0.9], [0.4, 0.6]]
        a.record(0, 0, weights)
        flows = a.top_flows(n=10)
        for i in range(len(flows) - 1):
            assert flows[i].weight >= flows[i + 1].weight

    def test_top_flows_n_limits_results(self):
        a = _make_analyzer()
        a.record(0, 0, _uniform_weights(4))  # 16 flows
        result = a.top_flows(n=5)
        assert len(result) == 5

    def test_top_flows_n_larger_than_available_returns_all(self):
        a = _make_analyzer()
        a.record(0, 0, _identity_weights(3))  # 3 flows
        result = a.top_flows(n=100)
        assert len(result) == 3

    def test_top_flows_default_n_is_10(self):
        a = _make_analyzer()
        a.record(0, 0, _uniform_weights(4))  # 16 flows
        result = a.top_flows()
        assert len(result) == 10

    def test_top_flows_empty_after_zero_weights(self):
        a = _make_analyzer()
        a.record(0, 0, _zero_weights(3))
        assert a.top_flows() == []

    def test_top_flows_highest_weight_first(self):
        a = _make_analyzer()
        weights = [[0.1, 0.9]]
        a.record(0, 0, weights)
        flows = a.top_flows(n=2)
        assert abs(flows[0].weight - 0.9) < 1e-9


# ---------------------------------------------------------------------------
# 5. layer_summary
# ---------------------------------------------------------------------------

class TestLayerSummary:
    def test_layer_summary_keys(self):
        a = _make_analyzer()
        a.record(0, 0, _uniform_weights(2))
        summary = a.layer_summary(0)
        assert "layer" in summary
        assert "total_flows" in summary
        assert "mean_weight" in summary
        assert "max_weight" in summary

    def test_layer_summary_layer_value(self):
        a = _make_analyzer()
        a.record(1, 0, _uniform_weights(2))
        summary = a.layer_summary(1)
        assert summary["layer"] == 1

    def test_layer_summary_total_flows(self):
        a = _make_analyzer()
        a.record(0, 0, _uniform_weights(3))  # 9 flows
        summary = a.layer_summary(0)
        assert summary["total_flows"] == 9

    def test_layer_summary_mean_weight_uniform(self):
        a = _make_analyzer()
        n = 2
        a.record(0, 0, _uniform_weights(n))
        summary = a.layer_summary(0)
        expected_mean = 1.0 / n
        assert abs(summary["mean_weight"] - expected_mean) < 1e-9

    def test_layer_summary_max_weight(self):
        a = _make_analyzer()
        weights = [[0.2, 0.8], [0.3, 0.7]]
        a.record(0, 0, weights)
        summary = a.layer_summary(0)
        assert abs(summary["max_weight"] - 0.8) < 1e-9

    def test_layer_summary_unrecorded_layer_zeros(self):
        a = _make_analyzer()
        a.record(0, 0, _uniform_weights(2))
        summary = a.layer_summary(99)
        assert summary["total_flows"] == 0
        assert summary["mean_weight"] == 0.0
        assert summary["max_weight"] == 0.0

    def test_layer_summary_aggregates_across_heads(self):
        a = _make_analyzer()
        a.record(0, 0, _identity_weights(2))  # 2 flows
        a.record(0, 1, _identity_weights(2))  # 2 flows
        summary = a.layer_summary(0)
        assert summary["total_flows"] == 4


# ---------------------------------------------------------------------------
# 6. head_importance
# ---------------------------------------------------------------------------

class TestHeadImportance:
    def test_head_importance_length_equals_num_heads(self):
        a = _make_analyzer(num_layers=2, num_heads=4)
        a.record(0, 0, _uniform_weights(2))
        result = a.head_importance(0)
        assert len(result) == 4

    def test_head_importance_unrecorded_head_is_zero(self):
        a = _make_analyzer(num_layers=2, num_heads=3)
        a.record(0, 0, _uniform_weights(2))
        result = a.head_importance(0)
        # Heads 1 and 2 were not recorded
        assert result[1] == 0.0
        assert result[2] == 0.0

    def test_head_importance_unrecorded_layer_all_zeros(self):
        a = _make_analyzer(num_layers=2, num_heads=3)
        a.record(0, 0, _uniform_weights(2))
        result = a.head_importance(1)
        assert result == [0.0, 0.0, 0.0]

    def test_head_importance_value_is_mean_weight(self):
        a = _make_analyzer(num_layers=2, num_heads=2)
        weights = [[1.0]]  # 1 flow with weight 1.0
        a.record(0, 0, weights)
        result = a.head_importance(0)
        assert abs(result[0] - 1.0) < 1e-9

    def test_head_importance_uniform_mean(self):
        n = 2
        a = _make_analyzer(num_layers=1, num_heads=1)
        a.record(0, 0, _uniform_weights(n))
        result = a.head_importance(0)
        assert abs(result[0] - 1.0 / n) < 1e-9

    def test_head_importance_multiple_heads(self):
        a = _make_analyzer(num_layers=1, num_heads=2)
        # head 0: single flow weight=0.8
        a.record(0, 0, [[0.8]])
        # head 1: single flow weight=0.3
        a.record(0, 1, [[0.3]])
        result = a.head_importance(0)
        assert abs(result[0] - 0.8) < 1e-9
        assert abs(result[1] - 0.3) < 1e-9


# ---------------------------------------------------------------------------
# 7. reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_top_flows(self):
        a = _make_analyzer()
        a.record(0, 0, _uniform_weights(3))
        a.reset()
        assert a.top_flows(n=100) == []

    def test_reset_clears_layer_summary(self):
        a = _make_analyzer()
        a.record(0, 0, _uniform_weights(3))
        a.reset()
        summary = a.layer_summary(0)
        assert summary["total_flows"] == 0

    def test_reset_clears_head_importance(self):
        a = _make_analyzer(num_layers=1, num_heads=2)
        a.record(0, 0, _uniform_weights(2))
        a.reset()
        result = a.head_importance(0)
        assert result == [0.0, 0.0]

    def test_reset_allows_fresh_recording(self):
        a = _make_analyzer()
        a.record(0, 0, _uniform_weights(2))
        a.reset()
        a.record(1, 0, _identity_weights(3))
        assert len(a.top_flows(n=100)) == 3


# ---------------------------------------------------------------------------
# 8. REGISTRY
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_has_default_key(self):
        assert "default" in ATTENTION_FLOW_REGISTRY

    def test_registry_default_is_class(self):
        assert ATTENTION_FLOW_REGISTRY["default"] is AttentionFlowAnalyzer

    def test_registry_default_is_instantiable(self):
        cls = ATTENTION_FLOW_REGISTRY["default"]
        instance = cls(num_layers=2, num_heads=4)
        assert isinstance(instance, AttentionFlowAnalyzer)
