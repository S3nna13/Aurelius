"""Tests for src.training.model_evaluator."""

from __future__ import annotations

import pytest

from src.training.model_evaluator import (
    MODEL_EVALUATOR_REGISTRY,
    ComparisonResult,
    EvalResult,
    ModelEvaluator,
)

# ---------------------------------------------------------------------------
# Dataclass basics
# ---------------------------------------------------------------------------


def test_eval_result_defaults():
    r = EvalResult(adapter_path="/tmp/a")
    assert r.perplexity == 0.0
    assert r.accuracy == 0.0
    assert r.n_samples == 0
    assert r.errors == []


def test_eval_result_frozen():
    r = EvalResult(adapter_path="/tmp/a")
    with pytest.raises(Exception):
        r.perplexity = 1.0  # type: ignore[misc]


def test_eval_result_errors_independent():
    a = EvalResult(adapter_path="a")
    b = EvalResult(adapter_path="b")
    a.errors.append("oops")
    assert b.errors == []


def test_comparison_result_defaults():
    old = EvalResult(adapter_path="o")
    new = EvalResult(adapter_path="n")
    c = ComparisonResult(old=old, new=new)
    assert c.regressed is False
    assert c.perplexity_delta_pct == 0.0
    assert c.accuracy_delta_pct == 0.0
    assert c.rejection_reason == ""


def test_comparison_result_frozen():
    old = EvalResult(adapter_path="o")
    new = EvalResult(adapter_path="n")
    c = ComparisonResult(old=old, new=new)
    with pytest.raises(Exception):
        c.regressed = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# compare()
# ---------------------------------------------------------------------------


def test_compare_not_regressed_identical():
    ev = ModelEvaluator()
    old = EvalResult(adapter_path="o", perplexity=10.0, accuracy=0.8, n_samples=50)
    new = EvalResult(adapter_path="n", perplexity=10.0, accuracy=0.8, n_samples=50)
    c = ev.compare(old, new)
    assert not c.regressed
    assert c.perplexity_delta_pct == 0.0
    assert c.accuracy_delta_pct == 0.0


def test_compare_improvement_not_regressed():
    ev = ModelEvaluator()
    old = EvalResult(adapter_path="o", perplexity=10.0, accuracy=0.7)
    new = EvalResult(adapter_path="n", perplexity=9.0, accuracy=0.8)
    c = ev.compare(old, new)
    assert not c.regressed
    assert c.perplexity_delta_pct == pytest.approx(-10.0)
    assert c.accuracy_delta_pct == pytest.approx((0.1 / 0.7) * 100.0)


def test_compare_regression_exceeds_threshold():
    ev = ModelEvaluator(max_regression_pct=5.0)
    old = EvalResult(adapter_path="o", perplexity=10.0)
    new = EvalResult(adapter_path="n", perplexity=12.0)  # +20%
    c = ev.compare(old, new)
    assert c.regressed
    assert "Perplexity" in c.rejection_reason
    assert c.perplexity_delta_pct == pytest.approx(20.0)


def test_compare_regression_within_threshold():
    ev = ModelEvaluator(max_regression_pct=5.0)
    old = EvalResult(adapter_path="o", perplexity=10.0)
    new = EvalResult(adapter_path="n", perplexity=10.3)  # +3%
    c = ev.compare(old, new)
    assert not c.regressed


def test_compare_exact_threshold_not_regressed():
    ev = ModelEvaluator(max_regression_pct=5.0)
    old = EvalResult(adapter_path="o", perplexity=10.0)
    new = EvalResult(adapter_path="n", perplexity=10.5)  # +5.0%, not > 5.0
    c = ev.compare(old, new)
    assert not c.regressed


def test_compare_min_accuracy_floor_triggers_regression():
    ev = ModelEvaluator(max_regression_pct=50.0, min_accuracy=0.9)
    old = EvalResult(adapter_path="o", perplexity=10.0, accuracy=0.95)
    new = EvalResult(adapter_path="n", perplexity=10.0, accuracy=0.8)
    c = ev.compare(old, new)
    assert c.regressed
    assert "below floor" in c.rejection_reason


def test_compare_min_accuracy_floor_zero_disabled():
    ev = ModelEvaluator(min_accuracy=0.0)
    old = EvalResult(adapter_path="o", perplexity=10.0, accuracy=0.95)
    new = EvalResult(adapter_path="n", perplexity=10.0, accuracy=0.1)
    c = ev.compare(old, new)
    assert not c.regressed


def test_compare_zero_old_perplexity_delta_zero():
    ev = ModelEvaluator()
    old = EvalResult(adapter_path="o", perplexity=0.0)
    new = EvalResult(adapter_path="n", perplexity=10.0)
    c = ev.compare(old, new)
    assert c.perplexity_delta_pct == 0.0


def test_compare_zero_old_accuracy_delta_zero():
    ev = ModelEvaluator()
    old = EvalResult(adapter_path="o", accuracy=0.0)
    new = EvalResult(adapter_path="n", accuracy=0.5)
    c = ev.compare(old, new)
    assert c.accuracy_delta_pct == 0.0


def test_compare_perplexity_delta_pct_math():
    ev = ModelEvaluator(max_regression_pct=100.0)
    old = EvalResult(adapter_path="o", perplexity=20.0)
    new = EvalResult(adapter_path="n", perplexity=25.0)
    c = ev.compare(old, new)
    assert c.perplexity_delta_pct == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# evaluate_samples()
# ---------------------------------------------------------------------------


def test_evaluate_samples_all_correct():
    ev = ModelEvaluator()
    r = ev.evaluate_samples(["hello", "world"], ["hello", "world"])
    assert r.accuracy == 1.0
    assert r.n_samples == 2


def test_evaluate_samples_none_correct():
    ev = ModelEvaluator()
    r = ev.evaluate_samples(["a", "b"], ["x", "y"])
    assert r.accuracy == 0.0
    assert r.n_samples == 2


def test_evaluate_samples_partial():
    ev = ModelEvaluator()
    r = ev.evaluate_samples(["a", "b", "c"], ["a", "x", "c"])
    assert r.accuracy == pytest.approx(2 / 3)


def test_evaluate_samples_case_insensitive():
    ev = ModelEvaluator()
    r = ev.evaluate_samples(["Hello"], ["hello"])
    assert r.accuracy == 1.0


def test_evaluate_samples_strips_whitespace():
    ev = ModelEvaluator()
    r = ev.evaluate_samples(["  hello  "], ["hello"])
    assert r.accuracy == 1.0


def test_evaluate_samples_length_mismatch_errors():
    ev = ModelEvaluator()
    r = ev.evaluate_samples(["a"], ["a", "b"])
    assert r.errors
    assert "mismatch" in r.errors[0]


def test_evaluate_samples_empty():
    ev = ModelEvaluator()
    r = ev.evaluate_samples([], [])
    assert r.n_samples == 0
    assert r.perplexity == 1.0


def test_evaluate_samples_perplexity_proxy():
    ev = ModelEvaluator()
    r = ev.evaluate_samples(["a b c", "d e"], ["x", "y"])
    assert r.perplexity == pytest.approx((3 + 2) / 2)


def test_evaluate_samples_adapter_path_stored():
    ev = ModelEvaluator()
    r = ev.evaluate_samples(["a"], ["a"], adapter_path="/tmp/adapter")
    assert r.adapter_path == "/tmp/adapter"


# ---------------------------------------------------------------------------
# is_deployment_safe()
# ---------------------------------------------------------------------------


def test_is_deployment_safe_no_baseline_no_errors():
    ev = ModelEvaluator()
    r = EvalResult(adapter_path="n", perplexity=10.0, accuracy=0.9)
    assert ev.is_deployment_safe(r) is True


def test_is_deployment_safe_errors_block():
    ev = ModelEvaluator()
    r = EvalResult(adapter_path="n", errors=["bad"])
    assert ev.is_deployment_safe(r) is False


def test_is_deployment_safe_baseline_no_regression():
    ev = ModelEvaluator(max_regression_pct=5.0)
    base = EvalResult(adapter_path="o", perplexity=10.0, accuracy=0.8)
    new = EvalResult(adapter_path="n", perplexity=10.2, accuracy=0.82)
    assert ev.is_deployment_safe(new, baseline=base) is True


def test_is_deployment_safe_baseline_regression():
    ev = ModelEvaluator(max_regression_pct=5.0)
    base = EvalResult(adapter_path="o", perplexity=10.0)
    new = EvalResult(adapter_path="n", perplexity=20.0)
    assert ev.is_deployment_safe(new, baseline=base) is False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_default_present():
    assert "default" in MODEL_EVALUATOR_REGISTRY
    assert MODEL_EVALUATOR_REGISTRY["default"] is ModelEvaluator


def test_registry_default_instantiable():
    ev = MODEL_EVALUATOR_REGISTRY["default"]()
    assert isinstance(ev, ModelEvaluator)
    assert ev.max_regression_pct == 5.0


def test_evaluator_constructor_args():
    ev = ModelEvaluator(max_regression_pct=10.0, min_accuracy=0.5)
    assert ev.max_regression_pct == 10.0
    assert ev.min_accuracy == 0.5
