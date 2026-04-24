"""Tests for src/eval/sandboxed_evaluator.py."""

from __future__ import annotations

import pytest
from src.eval.sandboxed_evaluator import EvalConfig, EvalResult, SandboxedEvaluator


@pytest.fixture
def evaluator() -> SandboxedEvaluator:
    return SandboxedEvaluator()


# ---------------------------------------------------------------------------
# EvalConfig defaults
# ---------------------------------------------------------------------------

def test_evalconfig_defaults() -> None:
    cfg = EvalConfig()
    assert cfg.timeout_s == 5.0
    assert cfg.max_output_bytes == 65536
    assert "math" in cfg.allowed_imports


def test_evalconfig_custom() -> None:
    cfg = EvalConfig(timeout_s=1.0, max_output_bytes=1024)
    assert cfg.timeout_s == 1.0
    assert cfg.max_output_bytes == 1024


# ---------------------------------------------------------------------------
# is_safe / denylist
# ---------------------------------------------------------------------------

def test_is_safe_clean_code(evaluator: SandboxedEvaluator) -> None:
    ok, reason = evaluator.is_safe("x = 1 + 1\nprint(x)")
    assert ok is True
    assert reason == ""


def test_denylist_os_system(evaluator: SandboxedEvaluator) -> None:
    ok, reason = evaluator.is_safe("import os\nos.system('ls')")
    assert ok is False
    assert "os.system" in reason


def test_denylist_subprocess_shell_true(evaluator: SandboxedEvaluator) -> None:
    ok, reason = evaluator.is_safe("subprocess.call(shell=True)")
    assert ok is False
    assert "subprocess.call(shell=True)" in reason


def test_denylist_popen_shell_true(evaluator: SandboxedEvaluator) -> None:
    ok, reason = evaluator.is_safe("subprocess.Popen(shell=True)")
    assert ok is False


def test_denylist_import_os_dunder(evaluator: SandboxedEvaluator) -> None:
    ok, reason = evaluator.is_safe("__import__('os').system('id')")
    assert ok is False


def test_denylist_exec(evaluator: SandboxedEvaluator) -> None:
    ok, reason = evaluator.is_safe("exec('print(1)')")
    assert ok is False
    assert "exec(" in reason


def test_denylist_eval(evaluator: SandboxedEvaluator) -> None:
    ok, reason = evaluator.is_safe("result = eval('1+1')")
    assert ok is False


def test_denylist_open(evaluator: SandboxedEvaluator) -> None:
    ok, reason = evaluator.is_safe("f = open('/etc/passwd')")
    assert ok is False


def test_denylist_importlib(evaluator: SandboxedEvaluator) -> None:
    ok, reason = evaluator.is_safe("import importlib\nimportlib.import_module('os')")
    assert ok is False


# ---------------------------------------------------------------------------
# run() — structure and safe code
# ---------------------------------------------------------------------------

def test_run_safe_code_passes(evaluator: SandboxedEvaluator) -> None:
    result = evaluator.run("print(1 + 1)")
    assert isinstance(result, EvalResult)
    assert result.passed is True
    assert result.exit_code == 0
    assert result.timed_out is False
    assert "2" in result.stdout


def test_run_blocked_code_not_passed(evaluator: SandboxedEvaluator) -> None:
    result = evaluator.run("os.system('ls')")
    assert result.passed is False
    assert "BLOCKED" in result.stderr
    assert result.exit_code != 0


def test_run_syntax_error_fails(evaluator: SandboxedEvaluator) -> None:
    result = evaluator.run("def f(\n")
    assert result.passed is False
    assert result.exit_code != 0


def test_run_result_fields_present(evaluator: SandboxedEvaluator) -> None:
    result = evaluator.run("x = 42")
    assert hasattr(result, "code")
    assert hasattr(result, "stdout")
    assert hasattr(result, "stderr")
    assert hasattr(result, "exit_code")
    assert hasattr(result, "timed_out")
    assert hasattr(result, "passed")


# ---------------------------------------------------------------------------
# run_batch / pass_rate
# ---------------------------------------------------------------------------

def test_run_batch_returns_list(evaluator: SandboxedEvaluator) -> None:
    results = evaluator.run_batch(["print(1)", "print(2)"])
    assert len(results) == 2
    assert all(isinstance(r, EvalResult) for r in results)


def test_pass_rate_all_pass(evaluator: SandboxedEvaluator) -> None:
    results = evaluator.run_batch(["print(1)", "x = 2"])
    rate = evaluator.pass_rate(results)
    assert rate == 1.0


def test_pass_rate_empty(evaluator: SandboxedEvaluator) -> None:
    assert evaluator.pass_rate([]) == 0.0


def test_pass_rate_mixed(evaluator: SandboxedEvaluator) -> None:
    results = evaluator.run_batch(["print(1)", "os.system('x')"])
    rate = evaluator.pass_rate(results)
    assert 0.0 < rate < 1.0
