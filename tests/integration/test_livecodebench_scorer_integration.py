"""Integration tests: livecodebench registered and reachable via src.eval."""

from __future__ import annotations

import pytest

import src.eval as ev


def test_metric_registry_has_livecodebench():
    assert "livecodebench" in ev.METRIC_REGISTRY


def test_benchmark_registry_has_livecodebench():
    assert "livecodebench" in ev.BENCHMARK_REGISTRY


def test_existing_humaneval_entries_unchanged():
    assert "humaneval" in ev.METRIC_REGISTRY
    assert "humaneval" in ev.BENCHMARK_REGISTRY
    assert ev.METRIC_REGISTRY["humaneval"] is ev.humaneval_score_problems
    assert ev.BENCHMARK_REGISTRY["humaneval"] is ev.HumanEvalProblem


def test_existing_mbpp_entries_unchanged():
    assert "mbpp" in ev.METRIC_REGISTRY
    assert "mbpp" in ev.BENCHMARK_REGISTRY
    assert ev.METRIC_REGISTRY["mbpp"] is ev.mbpp_score_problems
    assert ev.BENCHMARK_REGISTRY["mbpp"] is ev.MBPPProblem


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


def test_existing_swebench_entries_unchanged():
    assert "swebench_lite" in ev.METRIC_REGISTRY
    assert "swebench_lite" in ev.BENCHMARK_REGISTRY


def test_existing_ifeval_entries_unchanged():
    assert "ifeval" in ev.METRIC_REGISTRY
    assert "ifeval" in ev.BENCHMARK_REGISTRY


def test_existing_gpqa_entries_unchanged():
    assert "gpqa" in ev.METRIC_REGISTRY
    assert "gpqa" in ev.BENCHMARK_REGISTRY


def test_end_to_end_two_problem_score():
    LiveCodeProblem = ev.LiveCodeProblem
    score_problems = ev.METRIC_REGISTRY["livecodebench"]

    p1 = LiveCodeProblem(
        task_id="lcb_e2e_1",
        prompt="echo",
        starter_code="",
        test_cases=[("hello\n", "hello\n"), ("world\n", "world\n")],
        difficulty="easy",
        release_date="2025-06-01",
        contest_source="leetcode",
    )
    p2 = LiveCodeProblem(
        task_id="lcb_e2e_2",
        prompt="square",
        starter_code="",
        test_cases=[("3\n", "9\n")],
        difficulty="medium",
        release_date="2025-07-01",
        contest_source="codeforces",
    )

    completions = [
        ["print(input())\n"],  # passes both cases
        ["x=int(input()); print(x + x)\n"],  # returns 6 not 9: fails
    ]

    out = score_problems([p1, p2], completions, k_values=[1], timeout_seconds=5.0)
    assert out["n_problems"] == 2
    assert out["pass@1"] == pytest.approx(0.5)
    assert out["per_difficulty"]["easy"] == pytest.approx(1.0)
    assert out["per_difficulty"]["medium"] == pytest.approx(0.0)
    assert out["per_source"]["leetcode"] == pytest.approx(1.0)
    assert out["per_source"]["codeforces"] == pytest.approx(0.0)
