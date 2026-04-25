"""Unit tests for src/eval/humaneval_runner.py — 12 tests.

These tests exercise loading, sandboxed execution, and batch evaluation.
All tests run on CPU without any GPU requirement.
"""

from __future__ import annotations

import pytest

from src.eval.humaneval_runner import (
    ExecutionResult,
    HumanEvalProblem,
    HumanEvalRunner,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def runner() -> HumanEvalRunner:
    return HumanEvalRunner()


@pytest.fixture()
def stub_problems(runner: HumanEvalRunner) -> list[HumanEvalProblem]:
    return runner.load_problems()  # returns 3 built-in stubs


# ---------------------------------------------------------------------------
# load_problems
# ---------------------------------------------------------------------------


def test_load_problems_no_path_returns_stubs(runner: HumanEvalRunner):
    problems = runner.load_problems()
    assert len(problems) == 3


def test_stub_problems_have_required_fields(stub_problems: list[HumanEvalProblem]):
    for p in stub_problems:
        assert p.task_id
        assert p.prompt
        assert p.canonical_solution
        assert p.test
        assert p.entry_point


def test_load_problems_missing_path_falls_back_to_stubs(runner: HumanEvalRunner):
    problems = runner.load_problems(path="/nonexistent/path/humaneval.jsonl")
    assert len(problems) == 3


# ---------------------------------------------------------------------------
# HumanEvalProblem dataclass
# ---------------------------------------------------------------------------


def test_humaneval_problem_fields():
    p = HumanEvalProblem(
        task_id="test/0",
        prompt="def foo(): pass\n",
        canonical_solution="    pass\n",
        test="def check(c): assert c() is None\n",
        entry_point="foo",
    )
    assert p.task_id == "test/0"
    assert p.entry_point == "foo"


# ---------------------------------------------------------------------------
# execute_solution — passing solutions
# ---------------------------------------------------------------------------


def test_execute_correct_solution_passes(
    runner: HumanEvalRunner, stub_problems: list[HumanEvalProblem]
):
    problem = stub_problems[0]  # add(a, b)
    solution = "    return a + b\n"
    result = runner.execute_solution(problem, solution)
    assert result.passed is True
    assert result.error is None


def test_execute_solution_is_even(
    runner: HumanEvalRunner, stub_problems: list[HumanEvalProblem]
):
    problem = stub_problems[1]  # is_even(n)
    solution = "    return n % 2 == 0\n"
    result = runner.execute_solution(problem, solution)
    assert result.passed is True


def test_execute_solution_maximum(
    runner: HumanEvalRunner, stub_problems: list[HumanEvalProblem]
):
    problem = stub_problems[2]  # maximum(lst)
    solution = "    return max(lst)\n"
    result = runner.execute_solution(problem, solution)
    assert result.passed is True


# ---------------------------------------------------------------------------
# execute_solution — failing solutions
# ---------------------------------------------------------------------------


def test_execute_wrong_solution_fails(
    runner: HumanEvalRunner, stub_problems: list[HumanEvalProblem]
):
    problem = stub_problems[0]  # add(a, b) — expects sum
    solution = "    return a - b\n"  # subtraction is wrong
    result = runner.execute_solution(problem, solution)
    assert result.passed is False
    assert result.error is not None


def test_execute_syntax_error_fails(
    runner: HumanEvalRunner, stub_problems: list[HumanEvalProblem]
):
    problem = stub_problems[0]
    solution = "    not_valid_python = \n"
    result = runner.execute_solution(problem, solution)
    assert result.passed is False


# ---------------------------------------------------------------------------
# ExecutionResult fields
# ---------------------------------------------------------------------------


def test_execution_result_has_runtime_ms(
    runner: HumanEvalRunner, stub_problems: list[HumanEvalProblem]
):
    result = runner.execute_solution(stub_problems[0], "    return a + b\n")
    assert result.runtime_ms >= 0.0


def test_execution_result_task_id_matches(
    runner: HumanEvalRunner, stub_problems: list[HumanEvalProblem]
):
    problem = stub_problems[0]
    result = runner.execute_solution(problem, "    return a + b\n")
    assert result.task_id == problem.task_id


# ---------------------------------------------------------------------------
# batch_evaluate
# ---------------------------------------------------------------------------


def test_batch_evaluate_all_correct(
    runner: HumanEvalRunner, stub_problems: list[HumanEvalProblem]
):
    solutions = [
        "    return a + b\n",
        "    return n % 2 == 0\n",
        "    return max(lst)\n",
    ]
    metrics = runner.batch_evaluate(stub_problems, solutions)
    assert metrics["pass_at_1"] == pytest.approx(1.0)
    assert metrics["n_passed"] == 3
    assert metrics["n_total"] == 3


def test_batch_evaluate_none_correct(
    runner: HumanEvalRunner, stub_problems: list[HumanEvalProblem]
):
    solutions = [
        "    return None\n",
        "    return None\n",
        "    return None\n",
    ]
    metrics = runner.batch_evaluate(stub_problems, solutions)
    assert metrics["pass_at_1"] == pytest.approx(0.0)
    assert metrics["n_passed"] == 0


def test_batch_evaluate_returns_results_list(
    runner: HumanEvalRunner, stub_problems: list[HumanEvalProblem]
):
    solutions = ["    return a + b\n", "    return n % 2 == 0\n", "    return max(lst)\n"]
    metrics = runner.batch_evaluate(stub_problems, solutions)
    assert len(metrics["results"]) == 3
    assert all(isinstance(r, ExecutionResult) for r in metrics["results"])


def test_batch_evaluate_length_mismatch_raises(
    runner: HumanEvalRunner, stub_problems: list[HumanEvalProblem]
):
    with pytest.raises(ValueError):
        runner.batch_evaluate(stub_problems, ["    pass\n"])


# ---------------------------------------------------------------------------
# BENCHMARK_REGISTRY
# ---------------------------------------------------------------------------


def test_benchmark_registry_humaneval():
    from src.eval.humaneval_runner import BENCHMARK_REGISTRY
    assert "humaneval" in BENCHMARK_REGISTRY
    assert isinstance(BENCHMARK_REGISTRY["humaneval"], HumanEvalRunner)
