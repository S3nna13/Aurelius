"""Integration tests for HumanEval+ registration and end-to-end scoring."""

from __future__ import annotations

import os
import sys
import textwrap

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import src.eval as eval_pkg  # noqa: E402
from src.eval.humaneval_plus_scorer import (  # noqa: E402
    HumanEvalPlusProblem,
    score_problems,
)


def test_metric_registry_has_humaneval_plus():
    assert "humaneval_plus" in eval_pkg.METRIC_REGISTRY
    assert "humaneval_plus" in eval_pkg.BENCHMARK_REGISTRY
    assert eval_pkg.METRIC_REGISTRY["humaneval_plus"] is (eval_pkg.humaneval_plus_score_problems)
    assert eval_pkg.BENCHMARK_REGISTRY["humaneval_plus"] is (eval_pkg.HumanEvalPlusProblem)


def test_prior_registry_entries_still_present():
    # Spot-check that earlier additive registrations survived our append.
    for name in (
        "niah",
        "ruler",
        "humaneval",
        "mbpp",
        "swebench_lite",
        "ifeval",
        "mtbench",
        "alpacaeval",
        "arena_hard",
        "gpqa",
        "livecodebench",
        "mmlu",
    ):
        assert name in eval_pkg.METRIC_REGISTRY, f"missing {name}"
        assert name in eval_pkg.BENCHMARK_REGISTRY, f"missing {name}"


def test_end_to_end_two_problems():
    prompt = "def add(a, b):\n"
    base_test = textwrap.dedent(
        """
        def check(candidate):
            assert candidate(1, 2) == 3
            assert candidate(0, 0) == 0
        """
    )
    plus_tests = [
        "def check(candidate):\n    assert candidate(-1, 1) == 0\n",
        "def check(candidate):\n    assert candidate(10, 20) == 30\n",
    ]
    p1 = HumanEvalPlusProblem(
        task_id="e2e/1",
        prompt=prompt,
        canonical_solution="    return a + b\n",
        base_test=base_test,
        plus_tests=plus_tests,
        entry_point="add",
    )
    p2 = HumanEvalPlusProblem(
        task_id="e2e/2",
        prompt=prompt,
        canonical_solution="    return a + b\n",
        base_test=base_test,
        plus_tests=plus_tests,
        entry_point="add",
    )
    # p1: correct. p2: base-only (fails plus).
    completions = [
        ["    return a + b\n"],
        ["    return abs(a) + abs(b)\n"],
    ]
    rep = score_problems([p1, p2], completions, k_values=[1], timeout_seconds=10.0)
    assert rep["n_problems"] == 2
    assert rep["base_pass@1"] == pytest.approx(1.0)
    assert rep["plus_pass@1"] == pytest.approx(0.5)
    assert rep["robustness_gap"] == pytest.approx(0.5)
    assert {t["task_id"] for t in rep["per_task"]} == {"e2e/1", "e2e/2"}
