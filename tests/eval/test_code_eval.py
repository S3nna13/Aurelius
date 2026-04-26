"""Tests for code generation evaluation module."""

from __future__ import annotations

import pytest

from src.eval.code_eval import (
    CodeEvalConfig,
    CodeEvaluator,
    execute_code_safely,
    generate_synthetic_problems,
    is_safe_code,
    pass_at_k,
)

# ---------------------------------------------------------------------------
# 1. CodeEvalConfig defaults
# ---------------------------------------------------------------------------


def test_code_eval_config_defaults():
    """CodeEvalConfig has expected default values."""
    cfg = CodeEvalConfig()
    assert cfg.language == "python"
    assert cfg.timeout_seconds == 5.0
    assert cfg.n_samples == 1
    assert cfg.k_values == [1, 5, 10]
    assert cfg.safety_check is True


# ---------------------------------------------------------------------------
# 2 & 3. generate_synthetic_problems
# ---------------------------------------------------------------------------


def test_generate_synthetic_problems_count():
    """generate_synthetic_problems returns exactly n problems."""
    problems = generate_synthetic_problems(5, seed=42)
    assert len(problems) == 5


def test_generate_synthetic_problems_zero():
    """generate_synthetic_problems with n=0 returns empty list."""
    problems = generate_synthetic_problems(0)
    assert problems == []


def test_generate_synthetic_problems_have_test_cases():
    """Each generated problem has at least 3 assert test cases."""
    problems = generate_synthetic_problems(6, seed=0)
    for p in problems:
        assert isinstance(p.test_cases, list)
        assert len(p.test_cases) >= 3, f"Problem {p.problem_id} has <3 test cases"


def test_generate_synthetic_problems_fields():
    """Each generated problem exposes all required fields."""
    problems = generate_synthetic_problems(3, seed=7)
    for p in problems:
        assert p.problem_id
        assert p.prompt
        assert p.canonical_solution
        assert p.entry_point
        assert isinstance(p.test_cases, list)


# ---------------------------------------------------------------------------
# 4, 5, 6. is_safe_code
# ---------------------------------------------------------------------------


def test_is_safe_code_os_system():
    """is_safe_code returns False for code containing os.system."""
    code = "import os\nos.system('ls')\n"
    assert is_safe_code(code) is False


def test_is_safe_code_eval():
    """is_safe_code returns False for code using eval()."""
    code = "result = eval('1 + 2')\n"
    assert is_safe_code(code) is False


def test_is_safe_code_subprocess():
    """is_safe_code returns False for code using subprocess."""
    code = "import subprocess\nsubprocess.run(['ls'])\n"
    assert is_safe_code(code) is False


def test_is_safe_code_exec():
    """is_safe_code returns False for code using exec()."""
    code = "exec('x = 1')\n"
    assert is_safe_code(code) is False


def test_is_safe_code_open():
    """is_safe_code returns False for code using open()."""
    code = "f = open('file.txt')\n"
    assert is_safe_code(code) is False


def test_is_safe_code_import():
    """is_safe_code returns False for code using __import__()."""
    code = "mod = __import__('os')\n"
    assert is_safe_code(code) is False


def test_is_safe_code_safe():
    """is_safe_code returns True for benign code."""
    code = "def add(a, b):\n    return a + b\n"
    assert is_safe_code(code) is True


# ---------------------------------------------------------------------------
# 7, 8, 9. execute_code_safely
# ---------------------------------------------------------------------------


def test_execute_code_safely_correct_solution():
    """execute_code_safely passes a correct solution."""
    code = "def add(a, b):\n    return a + b\n"
    tests = ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]
    passed, error = execute_code_safely(code, tests, timeout=5.0)
    assert passed is True
    assert error == ""


def test_execute_code_safely_wrong_solution():
    """execute_code_safely fails a solution that returns wrong answers."""
    code = "def add(a, b):\n    return a - b\n"
    tests = ["assert add(1, 2) == 3"]
    passed, error = execute_code_safely(code, tests, timeout=5.0)
    assert passed is False
    assert error != ""


def test_execute_code_safely_syntax_error():
    """execute_code_safely catches SyntaxError in the solution."""
    code = "def broken(:\n    pass\n"
    tests = ["assert broken() is None"]
    passed, error = execute_code_safely(code, tests, timeout=5.0)
    assert passed is False
    assert "SyntaxError" in error


def test_execute_code_safely_runtime_error():
    """execute_code_safely catches runtime errors in the solution body."""
    code = "def bad():\n    raise ValueError('boom')\n"
    tests = ["assert bad() is None"]
    passed, error = execute_code_safely(code, tests, timeout=5.0)
    assert passed is False


# ---------------------------------------------------------------------------
# 10, 11, 12. pass_at_k
# ---------------------------------------------------------------------------


def test_pass_at_k_c_zero():
    """pass_at_k returns 0.0 when c=0."""
    assert pass_at_k(n=10, c=0, k=1) == 0.0


def test_pass_at_k_c_equals_n():
    """pass_at_k returns 1.0 when c=n (all samples correct)."""
    assert pass_at_k(n=10, c=10, k=1) == 1.0


def test_pass_at_k_c_geq_k():
    """pass_at_k returns 1.0 when c >= k."""
    assert pass_at_k(n=10, c=5, k=3) == 1.0


def test_pass_at_k_partial():
    """pass_at_k returns a value in (0, 1) for partial correctness.

    Use n=10, c=3, k=5: c < k so the shortcut does not fire;
    n-c=7 >= k=5 so C(7,5)/C(10,5) is a valid fraction < 1.
    """
    result = pass_at_k(n=10, c=3, k=5)
    assert 0.0 < result < 1.0


def test_pass_at_k_formula():
    """pass_at_k matches the exact formula for n=10, c=3, k=5.

    Expected = 1 - C(7, 5) / C(10, 5) = 1 - 21/252 = 1 - 1/12.
    """
    import math

    expected = 1.0 - math.comb(7, 5) / math.comb(10, 5)
    assert abs(pass_at_k(n=10, c=3, k=5) - expected) < 1e-9


# ---------------------------------------------------------------------------
# 13, 14. CodeEvaluator
# ---------------------------------------------------------------------------


@pytest.fixture
def evaluator():
    return CodeEvaluator(CodeEvalConfig())


def test_evaluate_solution_returns_correct_keys(evaluator):
    """evaluate_solution returns dict with 'passed', 'safe', 'error' keys."""
    problem = generate_synthetic_problems(1, seed=42)[0]
    solution = problem.canonical_solution
    result = evaluator.evaluate_solution(problem, solution)
    assert "passed" in result
    assert "safe" in result
    assert "error" in result


def test_evaluate_solution_canonical_passes(evaluator):
    """evaluate_solution marks canonical_solution as passed."""
    problem = generate_synthetic_problems(1, seed=0)[0]
    result = evaluator.evaluate_solution(problem, problem.canonical_solution)
    assert result["passed"] is True
    assert result["safe"] is True


def test_evaluate_solution_unsafe_rejected():
    """evaluate_solution rejects unsafe code when safety_check=True."""
    cfg = CodeEvalConfig(safety_check=True)
    ev = CodeEvaluator(cfg)
    problem = generate_synthetic_problems(1, seed=0)[0]
    unsafe_solution = "import os\nos.system('ls')\n"
    result = ev.evaluate_solution(problem, unsafe_solution)
    assert result["passed"] is False
    assert result["safe"] is False


def test_evaluate_problems_returns_pass_at_1(evaluator):
    """evaluate_problems returns a dict with 'pass@1' key."""
    problems = generate_synthetic_problems(3, seed=42)
    solutions = [p.canonical_solution for p in problems]
    metrics = evaluator.evaluate_problems(problems, solutions)
    assert "pass@1" in metrics
    assert "accuracy" in metrics
    assert "n_safe" in metrics


def test_evaluate_problems_all_correct(evaluator):
    """evaluate_problems gives accuracy=1.0 when all canonical solutions used."""
    problems = generate_synthetic_problems(4, seed=1)
    solutions = [p.canonical_solution for p in problems]
    metrics = evaluator.evaluate_problems(problems, solutions)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["pass@1"] == pytest.approx(1.0)


def test_evaluate_problems_length_mismatch(evaluator):
    """evaluate_problems raises ValueError when problems/solutions differ in length."""
    problems = generate_synthetic_problems(3, seed=0)
    with pytest.raises(ValueError):
        evaluator.evaluate_problems(problems, ["solution_a", "solution_b"])


def test_evaluate_problems_empty(evaluator):
    """evaluate_problems returns zero metrics for empty inputs."""
    metrics = evaluator.evaluate_problems([], [])
    assert metrics["pass@1"] == 0.0
    assert metrics["accuracy"] == 0.0
    assert metrics["n_safe"] == 0.0
