"""Unit tests for src/eval/tool_bench.py — 15 tests total."""
from __future__ import annotations

import pytest

from src.eval.tool_bench import (
    ToolBench,
    ToolBenchConfig,
    ToolBenchSample,
    ToolCall,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_call(name: str, **params) -> ToolCall:
    return ToolCall(tool_name=name, parameters=dict(params))


def _make_sample(
    sample_id: str,
    expected: list[ToolCall],
    predicted: list[ToolCall],
    available: list[str] | None = None,
) -> ToolBenchSample:
    if available is None:
        available = list({tc.tool_name for tc in expected + predicted})
    return ToolBenchSample(
        sample_id=sample_id,
        task_description="test task",
        available_tools=available,
        expected_tool_calls=expected,
        predicted_tool_calls=predicted,
    )


# ---------------------------------------------------------------------------
# 1. Config defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = ToolBenchConfig()
    assert cfg.param_match_threshold == 0.8
    assert cfg.order_sensitive is True
    assert cfg.partial_credit is True


# ---------------------------------------------------------------------------
# 2–4. Tool selection accuracy
# ---------------------------------------------------------------------------

def test_tool_selection_perfect():
    bench = ToolBench()
    sample = _make_sample(
        "s1",
        expected=[_make_call("search", q="foo"), _make_call("read", path="/x")],
        predicted=[_make_call("search", q="bar"), _make_call("read", path="/y")],
    )
    assert bench.tool_selection_accuracy([sample]) == pytest.approx(1.0)


def test_tool_selection_partial():
    bench = ToolBench()
    sample = _make_sample(
        "s1",
        expected=[_make_call("search"), _make_call("read")],
        predicted=[_make_call("search"), _make_call("write")],
    )
    # Position 0 matches, position 1 doesn't → 0.5
    assert bench.tool_selection_accuracy([sample]) == pytest.approx(0.5)


def test_tool_selection_wrong_tool():
    bench = ToolBench()
    sample = _make_sample(
        "s1",
        expected=[_make_call("search")],
        predicted=[_make_call("delete")],
    )
    assert bench.tool_selection_accuracy([sample]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 5–7. Parameter accuracy
# ---------------------------------------------------------------------------

def test_parameter_accuracy_perfect():
    bench = ToolBench()
    sample = _make_sample(
        "s1",
        expected=[_make_call("search", q="hello", limit=10)],
        predicted=[_make_call("search", q="hello", limit=10)],
    )
    assert bench.parameter_accuracy([sample]) == pytest.approx(1.0)


def test_parameter_accuracy_partial():
    bench = ToolBench()
    # 2 params expected; only 1 correct → 0.5
    sample = _make_sample(
        "s1",
        expected=[_make_call("search", q="hello", limit=10)],
        predicted=[_make_call("search", q="hello", limit=99)],
    )
    assert bench.parameter_accuracy([sample]) == pytest.approx(0.5)


def test_parameter_accuracy_no_match():
    bench = ToolBench()
    # Tool names differ → no matched pairs → 0.0
    sample = _make_sample(
        "s1",
        expected=[_make_call("search", q="hello")],
        predicted=[_make_call("read", path="/etc")],
    )
    assert bench.parameter_accuracy([sample]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 8–10. Sequence exact match
# ---------------------------------------------------------------------------

def test_sequence_exact_match_perfect():
    bench = ToolBench()
    calls = [_make_call("search", q="foo"), _make_call("read", path="/x")]
    sample = _make_sample("s1", expected=calls, predicted=list(calls))
    # Use separate ToolCall objects with identical values
    pred = [_make_call("search", q="foo"), _make_call("read", path="/x")]
    sample2 = _make_sample("s1", expected=calls, predicted=pred)
    assert bench.sequence_exact_match([sample2]) == pytest.approx(1.0)


def test_sequence_exact_match_wrong_order():
    bench = ToolBench(ToolBenchConfig(order_sensitive=True))
    expected = [_make_call("search", q="foo"), _make_call("read", path="/x")]
    # Swap order
    predicted = [_make_call("read", path="/x"), _make_call("search", q="foo")]
    sample = _make_sample("s1", expected=expected, predicted=predicted)
    # Names don't align position-by-position → not exact
    assert bench.sequence_exact_match([sample]) == pytest.approx(0.0)


def test_sequence_exact_match_wrong_params():
    bench = ToolBench()
    expected = [_make_call("search", q="foo")]
    predicted = [_make_call("search", q="bar")]   # right tool, wrong param
    sample = _make_sample("s1", expected=expected, predicted=predicted)
    assert bench.sequence_exact_match([sample]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 11–13. Format compliance
# ---------------------------------------------------------------------------

VALID_RESPONSE = '{"tool": "search", "parameters": {"q": "hello"}}'
INVALID_RESPONSE = "Just call the search function with hello."


def test_format_compliance_valid():
    bench = ToolBench()
    assert bench.format_compliance([VALID_RESPONSE]) == pytest.approx(1.0)


def test_format_compliance_invalid():
    bench = ToolBench()
    assert bench.format_compliance([INVALID_RESPONSE]) == pytest.approx(0.0)


def test_format_compliance_mixed():
    bench = ToolBench()
    responses = [VALID_RESPONSE, INVALID_RESPONSE, VALID_RESPONSE]
    # 2 valid out of 3
    assert bench.format_compliance(responses) == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# 14–15. evaluate() API
# ---------------------------------------------------------------------------

def test_evaluate_keys():
    bench = ToolBench()
    expected = [_make_call("search", q="x")]
    predicted = [_make_call("search", q="x")]
    sample = _make_sample("s1", expected=expected, predicted=predicted)
    result = bench.evaluate([sample], raw_responses=[VALID_RESPONSE])
    for key in ("tool_selection", "parameter_accuracy", "sequence_exact_match",
                "format_compliance", "n_samples"):
        assert key in result, f"Missing key: {key}"
    assert result["n_samples"] == 1


def test_evaluate_no_raw_responses():
    bench = ToolBench()
    expected = [_make_call("read", path="/x")]
    predicted = [_make_call("read", path="/x")]
    sample = _make_sample("s1", expected=expected, predicted=predicted)
    result = bench.evaluate([sample])
    assert "format_compliance" not in result
    assert "tool_selection" in result
    assert "parameter_accuracy" in result
    assert "sequence_exact_match" in result
    assert result["n_samples"] == 1
