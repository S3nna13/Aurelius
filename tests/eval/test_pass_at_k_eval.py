"""Unit and integration tests for src/eval/pass_at_k_eval.py.

Covers:
  - PassAtKConfig defaults
  - estimate_single edge cases and numerical stability
  - estimate_batch mean and empty-list handling
  - evaluate / summary output shape and value ranges
  - difficulty_distribution bucketing
  - Integration: monotone pass@k across k values
"""

from __future__ import annotations

import math

import pytest

from src.eval import BENCHMARK_REGISTRY
from src.eval.pass_at_k_eval import (
    PassAtKConfig,
    PassAtKEvaluator,
    PassAtKResult,
    ProblemResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_evaluator(k_values=None, n_samples=200):
    if k_values is None:
        k_values = [1, 5, 10]
    return PassAtKEvaluator(PassAtKConfig(k_values=k_values, n_samples=n_samples))


# ---------------------------------------------------------------------------
# 1. Config defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = PassAtKConfig(k_values=[1, 5, 10])
    assert cfg.n_samples == 200
    assert cfg.k_values == [1, 5, 10]


# ---------------------------------------------------------------------------
# 2. estimate_single — all correct
# ---------------------------------------------------------------------------


def test_estimate_single_all_correct():
    ev = _make_evaluator()
    # c == n means every draw succeeds
    for k in [1, 5, 10, 50]:
        assert ev.estimate_single(n=100, c=100, k=k) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3. estimate_single — none correct
# ---------------------------------------------------------------------------


def test_estimate_single_none_correct():
    ev = _make_evaluator()
    for k in [1, 5, 10, 100]:
        assert ev.estimate_single(n=200, c=0, k=k) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 4. estimate_single — k=1 equals c/n
# ---------------------------------------------------------------------------


def test_estimate_single_k1():
    ev = _make_evaluator()
    n, c = 200, 50
    expected = c / n  # pass@1 = 1 - (n-c)/n = c/n
    result = ev.estimate_single(n=n, c=c, k=1)
    assert result == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# 5. estimate_single — k == n, c >= 1 → very close to 1
# ---------------------------------------------------------------------------


def test_estimate_single_k_equals_n():
    ev = _make_evaluator()
    # With k = n and c >= 1, C(n-c, k) = 0 so pass@k = 1
    result = ev.estimate_single(n=10, c=1, k=10)
    assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 6. estimate_single — numerical stability (n=200, c=100, k=10)
# ---------------------------------------------------------------------------


def test_estimate_single_stability():
    ev = _make_evaluator()
    result = ev.estimate_single(n=200, c=100, k=10)
    assert 0.0 < result < 1.0
    assert math.isfinite(result)


# ---------------------------------------------------------------------------
# 7. estimate_single — n < k handled without error
# ---------------------------------------------------------------------------


def test_estimate_single_n_lt_k():
    ev = _make_evaluator()
    # n=5 < k=10
    assert ev.estimate_single(n=5, c=3, k=10) == pytest.approx(1.0)
    assert ev.estimate_single(n=5, c=0, k=10) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 8. estimate_batch — mean across problems
# ---------------------------------------------------------------------------


def test_estimate_batch_mean():
    ev = _make_evaluator()
    results = [
        ProblemResult("p0", n_samples=10, n_correct=0),
        ProblemResult("p1", n_samples=10, n_correct=10),
    ]
    # pass@1 for p0 = 0.0, p1 = 1.0 → mean = 0.5
    assert ev.estimate_batch(results, k=1) == pytest.approx(0.5, rel=1e-9)


# ---------------------------------------------------------------------------
# 9. estimate_batch — empty list returns 0.0
# ---------------------------------------------------------------------------


def test_estimate_batch_empty():
    ev = _make_evaluator()
    result = ev.estimate_batch([], k=5)
    assert result == 0.0


# ---------------------------------------------------------------------------
# 10. evaluate — returns one PassAtKResult per k_value
# ---------------------------------------------------------------------------


def test_evaluate_returns_all_k():
    k_values = [1, 5, 10, 100]
    ev = _make_evaluator(k_values=k_values)
    results = [ProblemResult("p0", n_samples=200, n_correct=100)]
    evals = ev.evaluate(results)
    assert len(evals) == len(k_values)
    assert all(isinstance(e, PassAtKResult) for e in evals)
    returned_ks = [e.k for e in evals]
    assert returned_ks == k_values


# ---------------------------------------------------------------------------
# 11. summary — correct key format
# ---------------------------------------------------------------------------


def test_summary_keys():
    k_values = [1, 5, 10]
    ev = _make_evaluator(k_values=k_values)
    results = [ProblemResult("p0", n_samples=100, n_correct=50)]
    s = ev.summary(results)
    assert set(s.keys()) == {"pass@1", "pass@5", "pass@10"}


# ---------------------------------------------------------------------------
# 12. summary — all values in [0, 1]
# ---------------------------------------------------------------------------


def test_summary_values_range():
    k_values = [1, 5, 10, 50]
    ev = _make_evaluator(k_values=k_values)
    results = [ProblemResult(f"p{i}", n_samples=200, n_correct=i * 20) for i in range(5)]
    s = ev.summary(results)
    for key, val in s.items():
        assert 0.0 <= val <= 1.0, f"{key}={val} out of range"


# ---------------------------------------------------------------------------
# 13. difficulty_distribution — correct keys present
# ---------------------------------------------------------------------------


def test_difficulty_distribution_keys():
    ev = _make_evaluator()
    dist = ev.difficulty_distribution([])
    assert set(dist.keys()) == {"unsolved", "easy", "medium", "all_correct"}


# ---------------------------------------------------------------------------
# 14. difficulty_distribution — counts sum to n_problems
# ---------------------------------------------------------------------------


def test_difficulty_distribution_counts():
    ev = _make_evaluator()
    results = [
        ProblemResult("unsolved", n_samples=10, n_correct=0),  # unsolved
        ProblemResult("easy", n_samples=10, n_correct=3),  # easy: 0.3 < 0.5
        ProblemResult("medium", n_samples=10, n_correct=6),  # medium: 0.6 >= 0.5
        ProblemResult("perfect", n_samples=10, n_correct=10),  # all_correct
    ]
    dist = ev.difficulty_distribution(results)
    assert dist["unsolved"] == 1
    assert dist["easy"] == 1
    assert dist["medium"] == 1
    assert dist["all_correct"] == 1
    assert sum(dist.values()) == len(results)


# ---------------------------------------------------------------------------
# 15. Registry — PassAtKEvaluator registered under "pass_at_k"
# ---------------------------------------------------------------------------


def test_registry_key():
    assert "pass_at_k" in BENCHMARK_REGISTRY
    assert BENCHMARK_REGISTRY["pass_at_k"] is PassAtKEvaluator


# ---------------------------------------------------------------------------
# Integration test — 10 problems, monotone pass@k, complete summary
# ---------------------------------------------------------------------------


def test_integration_monotone_pass_at_k():
    """With 10 problems of varying difficulty, pass@1 <= pass@5 <= pass@10
    and the summary dict contains all expected keys."""
    k_values = [1, 5, 10]
    ev = PassAtKEvaluator(PassAtKConfig(k_values=k_values, n_samples=10))

    # 10 problems: problem i has n=10, c=i (0..9)
    results = [ProblemResult(problem_id=f"prob_{i}", n_samples=10, n_correct=i) for i in range(10)]

    evals = ev.evaluate(results)
    assert len(evals) == 3

    p1 = evals[0].pass_at_k  # pass@1
    p5 = evals[1].pass_at_k  # pass@5
    p10 = evals[2].pass_at_k  # pass@10

    # Monotone: more samples → higher (or equal) chance of finding a solution
    assert p1 <= p5 + 1e-10, f"Expected pass@1 <= pass@5, got {p1} > {p5}"
    assert p5 <= p10 + 1e-10, f"Expected pass@5 <= pass@10, got {p5} > {p10}"

    # Values in range
    for val in (p1, p5, p10):
        assert 0.0 <= val <= 1.0

    # Summary has all keys
    s = ev.summary(results)
    assert set(s.keys()) == {"pass@1", "pass@5", "pass@10"}

    # n_problems recorded correctly
    for e in evals:
        assert e.n_problems == 10
