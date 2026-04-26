"""Integration tests: humaneval registered and reachable via src.eval."""

from __future__ import annotations

import pytest

import src.eval as ev


def test_metric_registry_has_humaneval():
    assert "humaneval" in ev.METRIC_REGISTRY


def test_benchmark_registry_has_humaneval():
    assert "humaneval" in ev.BENCHMARK_REGISTRY


def test_existing_niah_entries_unchanged():
    assert "niah" in ev.METRIC_REGISTRY
    assert "niah" in ev.BENCHMARK_REGISTRY
    assert ev.METRIC_REGISTRY["niah"] is ev.NeedleInHaystackBenchmark
    assert ev.BENCHMARK_REGISTRY["niah"] is ev.NeedleInHaystackBenchmark


def test_existing_ruler_entries_unchanged():
    assert "ruler" in ev.METRIC_REGISTRY
    assert "ruler" in ev.BENCHMARK_REGISTRY
    assert ev.METRIC_REGISTRY["ruler"] is ev.RULERBenchmark
    assert ev.BENCHMARK_REGISTRY["ruler"] is ev.RULERBenchmark


def test_end_to_end_pass_at_1_on_toy_problem():
    HumanEvalProblem = ev.HumanEvalProblem
    score_problems = ev.METRIC_REGISTRY["humaneval"]

    problem = HumanEvalProblem(
        task_id="int/0",
        prompt='def add(a, b):\n    """Return a+b."""\n',
        canonical_solution="    return a + b\n",
        test=(
            "def check(candidate):\n"
            "    assert candidate(1, 2) == 3\n"
            "    assert candidate(-5, 5) == 0\n"
        ),
        entry_point="add",
    )

    completions = [["    return a + b\n", "    return a - b\n"]]

    out = score_problems([problem], completions, k_values=[1], max_workers=2, timeout_seconds=5.0)
    assert out["n_problems"] == 1
    assert out["pass@1"] == pytest.approx(0.5)
