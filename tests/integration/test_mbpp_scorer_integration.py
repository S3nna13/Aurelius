"""Integration tests: mbpp registered and reachable via src.eval."""

from __future__ import annotations

import pytest

import src.eval as ev


def test_metric_registry_has_mbpp():
    assert "mbpp" in ev.METRIC_REGISTRY


def test_benchmark_registry_has_mbpp():
    assert "mbpp" in ev.BENCHMARK_REGISTRY


def test_existing_humaneval_entries_unchanged():
    assert "humaneval" in ev.METRIC_REGISTRY
    assert "humaneval" in ev.BENCHMARK_REGISTRY
    assert ev.METRIC_REGISTRY["humaneval"] is ev.humaneval_score_problems
    assert ev.BENCHMARK_REGISTRY["humaneval"] is ev.HumanEvalProblem


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


def test_end_to_end_two_problem_score():
    MBPPProblem = ev.MBPPProblem
    score_problems = ev.METRIC_REGISTRY["mbpp"]

    p1 = MBPPProblem(
        task_id=101,
        text="add",
        code="def add(a, b):\n    return a + b\n",
        test_list=["assert add(1, 2) == 3", "assert add(0, 0) == 0"],
    )
    p2 = MBPPProblem(
        task_id=102,
        text="mul",
        code="def mul(a, b):\n    return a * b\n",
        test_list=["assert mul(3, 4) == 12"],
    )

    completions = [
        ["def add(a, b):\n    return a + b\n"],  # correct
        ["def mul(a, b):\n    return a + b\n"],  # wrong
    ]

    out = score_problems([p1, p2], completions, k_values=[1], max_workers=2, timeout_seconds=5.0)
    assert out["n_problems"] == 2
    assert out["pass@1"] == pytest.approx(0.5)
