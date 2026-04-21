"""Integration test for ToolBench — end-to-end evaluation pipeline.

Verifies:
- 3 ToolBenchSamples (perfect / partial / wrong)
- evaluate() returns all metrics in [0, 1]
- format_compliance on 3 raw responses
- BENCHMARK_REGISTRY["tool_bench"] is wired
"""
from __future__ import annotations

import pytest

from src.eval.tool_bench import (
    ToolBench,
    ToolBenchConfig,
    ToolBenchSample,
    ToolCall,
)
from src.eval import BENCHMARK_REGISTRY


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _tc(name: str, **params) -> ToolCall:
    return ToolCall(tool_name=name, parameters=dict(params))


def _sample(sid, exp, pred):
    available = list({tc.tool_name for tc in exp + pred})
    return ToolBenchSample(
        sample_id=sid,
        task_description=f"Task {sid}",
        available_tools=available,
        expected_tool_calls=exp,
        predicted_tool_calls=pred,
    )


# Perfect match: same tool, same params
SAMPLE_PERFECT = _sample(
    "perfect",
    exp=[_tc("web_search", query="LLM benchmarks"), _tc("read_file", path="/data/results.json")],
    pred=[_tc("web_search", query="LLM benchmarks"), _tc("read_file", path="/data/results.json")],
)

# Partial match: right first tool wrong params, second tool entirely wrong
SAMPLE_PARTIAL = _sample(
    "partial",
    exp=[_tc("web_search", query="deep learning"), _tc("calculator", expr="2+2")],
    pred=[_tc("web_search", query="machine learning"), _tc("translator", text="hi")],
)

# Completely wrong: different tool names
SAMPLE_WRONG = _sample(
    "wrong",
    exp=[_tc("database_query", sql="SELECT * FROM t")],
    pred=[_tc("send_email", to="user@example.com", body="hi")],
)

SAMPLES = [SAMPLE_PERFECT, SAMPLE_PARTIAL, SAMPLE_WRONG]

RAW_RESPONSES = [
    '{"tool": "web_search", "parameters": {"query": "LLM benchmarks"}}',   # valid
    'Call the calculator with 2+2',                                          # invalid
    '{"tool": "database_query", "parameters": {"sql": "SELECT 1"}}',        # valid
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_registry_wired():
    """BENCHMARK_REGISTRY must contain a 'tool_bench' entry."""
    assert "tool_bench" in BENCHMARK_REGISTRY
    assert BENCHMARK_REGISTRY["tool_bench"] is ToolBench


def test_evaluate_returns_all_metric_keys():
    bench = ToolBench()
    result = bench.evaluate(SAMPLES, raw_responses=RAW_RESPONSES)
    for key in ("tool_selection", "parameter_accuracy", "sequence_exact_match",
                "format_compliance", "n_samples"):
        assert key in result, f"Missing key: {key}"


def test_evaluate_metrics_in_unit_range():
    bench = ToolBench()
    result = bench.evaluate(SAMPLES, raw_responses=RAW_RESPONSES)
    for key in ("tool_selection", "parameter_accuracy", "sequence_exact_match",
                "format_compliance"):
        val = result[key]
        assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"


def test_evaluate_n_samples():
    bench = ToolBench()
    result = bench.evaluate(SAMPLES, raw_responses=RAW_RESPONSES)
    assert result["n_samples"] == len(SAMPLES)


def test_perfect_sample_drives_up_scores():
    """Evaluating only the perfect sample must yield 1.0 on all metrics."""
    bench = ToolBench()
    result = bench.evaluate(
        [SAMPLE_PERFECT],
        raw_responses=[RAW_RESPONSES[0]],
    )
    assert result["tool_selection"] == pytest.approx(1.0)
    assert result["parameter_accuracy"] == pytest.approx(1.0)
    assert result["sequence_exact_match"] == pytest.approx(1.0)
    assert result["format_compliance"] == pytest.approx(1.0)


def test_wrong_sample_scores_zero_on_selection():
    """A completely wrong sample should have 0.0 tool selection accuracy."""
    bench = ToolBench()
    result = bench.evaluate([SAMPLE_WRONG])
    assert result["tool_selection"] == pytest.approx(0.0)


def test_format_compliance_two_of_three():
    bench = ToolBench()
    result = bench.evaluate(SAMPLES, raw_responses=RAW_RESPONSES)
    # 2 valid out of 3
    assert result["format_compliance"] == pytest.approx(2 / 3)


def test_mixed_samples_tool_selection_between_zero_and_one():
    bench = ToolBench()
    result = bench.evaluate(SAMPLES)
    assert 0.0 < result["tool_selection"] < 1.0


def test_config_order_insensitive_partial_sample():
    """With order_sensitive=False, name matching is set-based."""
    cfg = ToolBenchConfig(order_sensitive=False)
    bench = ToolBench(cfg)
    # SAMPLE_PARTIAL first tool matches by name (regardless of position)
    result = bench.evaluate([SAMPLE_PARTIAL])
    # At least 1 of 2 tools matches by name → > 0
    assert result["tool_selection"] > 0.0
