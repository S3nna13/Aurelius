"""Tests for src/evaluation/code_eval.py — ≥28 test cases."""

from __future__ import annotations

import pytest

from src.evaluation.code_eval import (
    CODE_EVALUATOR_REGISTRY,
    CodeEvalConfig,
    CodeEvalResult,
    CodeEvaluator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIMPLE_SOLUTION = "def solution(x):\n    return x * 2\n"
WRONG_SOLUTION = "def solution(x):\n    return x + 1\n"
SYNTAX_ERROR_CODE = "def solution(x:\n    return x\n"
PRINT_CODE = "def solution(x):\n    print('hello')\n    return x\n"
NO_SOLUTION_CODE = "result = 42\n"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def evaluator():
    return CodeEvaluator()


# ---------------------------------------------------------------------------
# CodeEvalResult dataclass
# ---------------------------------------------------------------------------


class TestCodeEvalResultFrozen:
    def test_is_frozen(self, evaluator):
        result = evaluator.evaluate(SIMPLE_SOLUTION, [2], [4])
        with pytest.raises((AttributeError, TypeError)):
            result.passed = False  # type: ignore[misc]

    def test_fields_accessible(self, evaluator):
        result = evaluator.evaluate(SIMPLE_SOLUTION, [3], [6])
        assert hasattr(result, "code")
        assert hasattr(result, "passed")
        assert hasattr(result, "output")
        assert hasattr(result, "error")
        assert hasattr(result, "execution_time_ms")


# ---------------------------------------------------------------------------
# Correct solution
# ---------------------------------------------------------------------------


class TestCorrectSolution:
    def test_simple_correct_passes(self, evaluator):
        result = evaluator.evaluate(SIMPLE_SOLUTION, [2], [4])
        assert result.passed is True

    def test_multiple_correct_inputs_all_pass(self, evaluator):
        result = evaluator.evaluate(SIMPLE_SOLUTION, [1, 2, 3], [2, 4, 6])
        assert result.passed is True

    def test_correct_solution_no_error(self, evaluator):
        result = evaluator.evaluate(SIMPLE_SOLUTION, [5], [10])
        assert result.error == ""

    def test_code_stored_in_result(self, evaluator):
        result = evaluator.evaluate(SIMPLE_SOLUTION, [1], [2])
        assert result.code == SIMPLE_SOLUTION


# ---------------------------------------------------------------------------
# Wrong answer
# ---------------------------------------------------------------------------


class TestWrongAnswer:
    def test_wrong_answer_fails(self, evaluator):
        # solution returns x+1, expected x*2 → wrong for x=5
        result = evaluator.evaluate(WRONG_SOLUTION, [5], [10])
        assert result.passed is False

    def test_wrong_answer_no_exception_error(self, evaluator):
        result = evaluator.evaluate(WRONG_SOLUTION, [5], [10])
        # No exception — error field should be empty
        assert result.error == ""

    def test_partial_wrong_fails(self, evaluator):
        # First input passes (x+1 == x*2 when x=2), second fails
        result = evaluator.evaluate(WRONG_SOLUTION, [2, 5], [4, 10])
        # x=2: 2+1=3 ≠ 4 → fails immediately
        assert result.passed is False


# ---------------------------------------------------------------------------
# Syntax error
# ---------------------------------------------------------------------------


class TestSyntaxError:
    def test_syntax_error_fails(self, evaluator):
        result = evaluator.evaluate(SYNTAX_ERROR_CODE, [], [])
        assert result.passed is False

    def test_syntax_error_populates_error(self, evaluator):
        result = evaluator.evaluate(SYNTAX_ERROR_CODE, [], [])
        assert result.error != ""

    def test_syntax_error_error_is_string(self, evaluator):
        result = evaluator.evaluate(SYNTAX_ERROR_CODE, [], [])
        assert isinstance(result.error, str)


# ---------------------------------------------------------------------------
# No solution function / empty test_inputs
# ---------------------------------------------------------------------------


class TestNoSolutionFunction:
    def test_empty_inputs_no_solution_fn_passes(self, evaluator):
        # No solution() defined, no test cases — should pass (code ran fine)
        result = evaluator.evaluate(NO_SOLUTION_CODE, [], [])
        assert result.passed is True

    def test_no_solution_fn_with_inputs_fails(self, evaluator):
        result = evaluator.evaluate(NO_SOLUTION_CODE, [1], [2])
        assert result.passed is False


# ---------------------------------------------------------------------------
# execution_time_ms
# ---------------------------------------------------------------------------


class TestExecutionTime:
    def test_execution_time_ms_positive(self, evaluator):
        result = evaluator.evaluate(SIMPLE_SOLUTION, [1], [2])
        assert result.execution_time_ms > 0

    def test_execution_time_ms_is_float(self, evaluator):
        result = evaluator.evaluate(SIMPLE_SOLUTION, [1], [2])
        assert isinstance(result.execution_time_ms, float)


# ---------------------------------------------------------------------------
# Output capture
# ---------------------------------------------------------------------------


class TestOutputCapture:
    def test_print_output_captured(self, evaluator):
        result = evaluator.evaluate(PRINT_CODE, [7], [7])
        assert "hello" in result.output

    def test_no_print_empty_output(self, evaluator):
        result = evaluator.evaluate(SIMPLE_SOLUTION, [1], [2])
        assert result.output == ""


# ---------------------------------------------------------------------------
# batch_evaluate
# ---------------------------------------------------------------------------


class TestBatchEvaluate:
    def test_batch_returns_correct_length(self, evaluator):
        problems = [
            {"code": SIMPLE_SOLUTION, "inputs": [2], "expected": [4]},
            {"code": WRONG_SOLUTION, "inputs": [5], "expected": [10]},
            {"code": NO_SOLUTION_CODE, "inputs": [], "expected": []},
        ]
        results = evaluator.batch_evaluate(problems)
        assert len(results) == 3

    def test_batch_empty_list(self, evaluator):
        assert evaluator.batch_evaluate([]) == []

    def test_batch_all_results_are_code_eval_results(self, evaluator):
        problems = [
            {"code": SIMPLE_SOLUTION, "inputs": [1], "expected": [2]},
        ]
        results = evaluator.batch_evaluate(problems)
        assert isinstance(results[0], CodeEvalResult)

    def test_batch_first_passes_second_fails(self, evaluator):
        problems = [
            {"code": SIMPLE_SOLUTION, "inputs": [3], "expected": [6]},
            {"code": WRONG_SOLUTION, "inputs": [3], "expected": [6]},
        ]
        results = evaluator.batch_evaluate(problems)
        assert results[0].passed is True
        assert results[1].passed is False


# ---------------------------------------------------------------------------
# Allowed builtins restriction
# ---------------------------------------------------------------------------


class TestAllowedBuiltins:
    def test_disallowed_builtin_raises_and_fails(self, evaluator):
        # open() is not in the allowed list — should fail
        code = "def solution(x):\n    f = open('/etc/passwd')\n    return x\n"
        result = evaluator.evaluate(code, [1], [1])
        assert result.passed is False
        assert result.error != ""

    def test_allowed_builtin_len_works(self, evaluator):
        code = "def solution(x):\n    return len(x)\n"
        result = evaluator.evaluate(code, [[1, 2, 3]], [3])
        assert result.passed is True


# ---------------------------------------------------------------------------
# CodeEvalConfig
# ---------------------------------------------------------------------------


class TestCodeEvalConfig:
    def test_default_timeout(self):
        cfg = CodeEvalConfig()
        assert cfg.timeout_seconds == 5.0

    def test_default_allowed_builtins_includes_print(self):
        cfg = CodeEvalConfig()
        assert "print" in cfg.allowed_builtins

    def test_custom_config_accepted(self):
        cfg = CodeEvalConfig(timeout_seconds=10.0)
        evaluator = CodeEvaluator(config=cfg)
        assert evaluator.config.timeout_seconds == 10.0


# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registry_has_default(self):
        assert "default" in CODE_EVALUATOR_REGISTRY

    def test_registry_default_is_code_evaluator_class(self):
        assert CODE_EVALUATOR_REGISTRY["default"] is CodeEvaluator

    def test_registry_default_instantiable(self):
        cls = CODE_EVALUATOR_REGISTRY["default"]
        instance = cls()
        assert isinstance(instance, CodeEvaluator)
