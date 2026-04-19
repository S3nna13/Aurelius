"""Tests for src.eval.humaneval_plus_scorer."""

from __future__ import annotations

import os
import sys
import textwrap

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.eval.humaneval_plus_scorer import (  # noqa: E402
    HumanEvalPlusProblem,
    HumanEvalPlusResult,
    pass_at_k,
    score_problems,
    score_single,
)
from src.eval import humaneval_scorer as _hm_scorer  # noqa: E402


def _add_problem(plus_tests=None, task_id="add/1"):
    prompt = "def add(a, b):\n"
    canonical = "    return a + b\n"
    base_test = textwrap.dedent(
        """
        def check(candidate):
            assert candidate(1, 2) == 3
            assert candidate(0, 0) == 0
        """
    )
    if plus_tests is None:
        plus_tests = [
            "def check(candidate):\n    assert candidate(-1, 1) == 0\n",
            "def check(candidate):\n    assert candidate(100, 200) == 300\n",
        ]
    return HumanEvalPlusProblem(
        task_id=task_id,
        prompt=prompt,
        canonical_solution=canonical,
        base_test=base_test,
        plus_tests=plus_tests,
        entry_point="add",
    )


_CORRECT = "    return a + b\n"
_BASE_ONLY = "    return abs(a) + abs(b)\n"
_WRONG = "    return a - b\n"
_SYNTAX_ERR = "    return a +\n"


def test_correct_completion_passes_base_and_plus():
    p = _add_problem()
    res = score_single(p, _CORRECT, timeout_seconds=10.0)
    assert isinstance(res, HumanEvalPlusResult)
    assert res.passed_base is True
    assert res.passed_plus is True
    assert res.base_fail_count == 0
    assert res.plus_fail_count == 0
    assert res.error is None
    assert res.duration_ms > 0


def test_base_only_completion_fails_plus():
    p = _add_problem()
    res = score_single(p, _BASE_ONLY, timeout_seconds=10.0)
    assert res.passed_base is True
    assert res.passed_plus is False
    assert res.base_fail_count == 0
    assert res.plus_fail_count >= 1
    assert res.error is not None


def test_robustness_gap_on_three_problems():
    problems = [_add_problem(task_id=f"add/{i}") for i in range(3)]
    completions = [[_BASE_ONLY] for _ in problems]
    report = score_problems(problems, completions, k_values=[1], timeout_seconds=10.0)
    assert report["n_problems"] == 3
    assert report["base_pass@1"] == pytest.approx(1.0)
    assert report["plus_pass@1"] == pytest.approx(0.0)
    assert report["robustness_gap"] == pytest.approx(1.0)


def test_timeout_enforced():
    prompt = "def slow():\n"
    base_test = "def check(candidate):\n    candidate()\n"
    body = "    while True:\n        pass\n"
    p = HumanEvalPlusProblem(
        task_id="slow/1",
        prompt=prompt,
        canonical_solution=body,
        base_test=base_test,
        plus_tests=[],
        entry_point="slow",
    )
    res = score_single(p, body, timeout_seconds=1.0)
    assert res.passed_base is False
    assert res.error is not None
    assert "Timeout" in res.error or "timeout" in res.error.lower()


def test_syntax_error_captured():
    p = _add_problem(plus_tests=[])
    res = score_single(p, _SYNTAX_ERR, timeout_seconds=5.0)
    assert res.passed_base is False
    assert res.passed_plus is False
    assert res.error is not None


def test_pass_at_k_reused_from_humaneval():
    assert pass_at_k is _hm_scorer.pass_at_k
    assert pass_at_k(10, 5, 1) == pytest.approx(0.5)


def test_fail_counts_populated():
    p = _add_problem()
    res_bad_base = score_single(p, _WRONG, timeout_seconds=5.0)
    assert res_bad_base.base_fail_count == 1
    assert res_bad_base.plus_fail_count == 2

    res_base_only = score_single(p, _BASE_ONLY, timeout_seconds=10.0)
    assert res_base_only.base_fail_count == 0
    assert res_base_only.plus_fail_count >= 1


def test_empty_plus_tests_yields_passed_plus_true():
    p = _add_problem(plus_tests=[])
    res = score_single(p, _CORRECT, timeout_seconds=5.0)
    assert res.passed_base is True
    assert res.passed_plus is True
    assert res.plus_fail_count == 0


def test_score_problems_aggregates():
    problems = [_add_problem(task_id="a/1"), _add_problem(task_id="a/2")]
    completions = [[_CORRECT], [_BASE_ONLY]]
    rep = score_problems(problems, completions, k_values=[1], timeout_seconds=10.0)
    assert rep["n_problems"] == 2
    assert rep["base_pass@1"] == pytest.approx(1.0)
    assert rep["plus_pass@1"] == pytest.approx(0.5)
    assert rep["robustness_gap"] == pytest.approx(0.5)
    assert len(rep["per_task"]) == 2


def test_determinism():
    p = _add_problem()
    r1 = score_single(p, _CORRECT, timeout_seconds=10.0)
    r2 = score_single(p, _CORRECT, timeout_seconds=10.0)
    assert (r1.passed_base, r1.passed_plus) == (r2.passed_base, r2.passed_plus)
    assert r1.base_fail_count == r2.base_fail_count
    assert r1.plus_fail_count == r2.plus_fail_count


def test_subprocess_isolation_from_parent_globals():
    parent_only_name = "SENTINEL_IN_PARENT_ONLY_12345"  # noqa: F841
    completion = "    return SENTINEL_IN_PARENT_ONLY_12345\n"
    p = _add_problem(plus_tests=[])
    res = score_single(p, completion, timeout_seconds=5.0)
    assert res.passed_base is False
    assert res.error is not None


def test_sandbox_env_scrubbed(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/nonexistent/should-not-leak")
    prompt = "import os\ndef f():\n"
    base_test = textwrap.dedent(
        """
        def check(candidate):
            val = candidate()
            assert '/nonexistent/should-not-leak' not in val, val
        """
    )
    body = "    return os.environ.get('PYTHONPATH', '')\n"
    p = HumanEvalPlusProblem(
        task_id="env/1",
        prompt=prompt,
        canonical_solution=body,
        base_test=base_test,
        plus_tests=[],
        entry_point="f",
    )
    res = score_single(p, body, timeout_seconds=5.0)
    assert res.passed_base is True


def test_pass_at_k_raises_when_k_exceeds_n():
    with pytest.raises(ValueError):
        pass_at_k(3, 1, 5)


def test_base_precedence_plus_skipped_when_base_fails():
    exploding_plus = [
        "def check(candidate):\n    raise RuntimeError('PLUS_WAS_RUN_SHOULD_NOT_HAVE')\n",
    ]
    p = _add_problem(plus_tests=exploding_plus)
    res = score_single(p, _WRONG, timeout_seconds=5.0)
    assert res.passed_base is False
    assert res.passed_plus is False
    assert res.error is not None
    assert "PLUS_WAS_RUN" not in res.error
    assert res.plus_fail_count == len(exploding_plus)
