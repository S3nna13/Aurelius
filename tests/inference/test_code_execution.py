"""Tests for src/inference/code_execution.py."""

from __future__ import annotations

import pytest

from src.inference.code_execution import (
    CodeGenerationEvaluator,
    CodeInterpreterSession,
    ExecutionConfig,
    execute_python,
    extract_code_blocks,
    sanitize_code,
)

# ---------------------------------------------------------------------------
# 1. ExecutionConfig defaults
# ---------------------------------------------------------------------------


def test_execution_config_defaults():
    cfg = ExecutionConfig()
    assert cfg.timeout_seconds == 5.0
    assert cfg.max_output_len == 1000
    assert cfg.capture_exceptions is True
    assert "math" in cfg.allowed_modules
    assert "re" in cfg.allowed_modules
    assert "json" in cfg.allowed_modules
    assert len(cfg.allowed_modules) == 8


# ---------------------------------------------------------------------------
# 2. extract_code_blocks — markdown fenced block
# ---------------------------------------------------------------------------


def test_extract_code_blocks_markdown():
    text = "Some intro\n```python\nprint('hi')\n```\nSome outro"
    blocks = extract_code_blocks(text)
    assert len(blocks) == 1
    assert "print('hi')" in blocks[0]


# ---------------------------------------------------------------------------
# 3. extract_code_blocks — multiple blocks
# ---------------------------------------------------------------------------


def test_extract_code_blocks_multiple():
    text = "```python\nx = 1\n```\nSome text\n```python\ny = 2\n```"
    blocks = extract_code_blocks(text)
    assert len(blocks) == 2
    assert any("x = 1" in b for b in blocks)
    assert any("y = 2" in b for b in blocks)


# ---------------------------------------------------------------------------
# 4. extract_code_blocks — no blocks
# ---------------------------------------------------------------------------


def test_extract_code_blocks_empty():
    text = "No code blocks here."
    blocks = extract_code_blocks(text)
    assert blocks == []


# ---------------------------------------------------------------------------
# 5. sanitize_code — safe code
# ---------------------------------------------------------------------------


def test_sanitize_code_safe():
    code = "x = 1 + 1\nprint(x)"
    is_safe, reason = sanitize_code(code, ["math"])
    assert is_safe is True
    assert reason == ""


# ---------------------------------------------------------------------------
# 6. sanitize_code — import os blocked
# ---------------------------------------------------------------------------


def test_sanitize_code_import_os_blocked():
    code = "import os\nprint(os.getcwd())"
    is_safe, reason = sanitize_code(code, ["math"])
    assert is_safe is False
    assert "os" in reason


# ---------------------------------------------------------------------------
# 7. sanitize_code — eval blocked
# ---------------------------------------------------------------------------


def test_sanitize_code_eval_blocked():
    code = 'result = eval("1+1")'
    is_safe, reason = sanitize_code(code, ["math"])
    assert is_safe is False
    assert "eval" in reason


# ---------------------------------------------------------------------------
# 8. execute_python — simple print
# ---------------------------------------------------------------------------


def test_execute_python_simple():
    cfg = ExecutionConfig()
    result = execute_python('print("hello")', cfg)
    assert "hello" in result.stdout


# ---------------------------------------------------------------------------
# 9. execute_python — success flag for clean code
# ---------------------------------------------------------------------------


def test_execute_python_success_flag():
    cfg = ExecutionConfig()
    result = execute_python("x = 42", cfg)
    assert result.success is True
    assert result.error is None


# ---------------------------------------------------------------------------
# 10. execute_python — syntax error sets success=False
# ---------------------------------------------------------------------------


def test_execute_python_syntax_error():
    cfg = ExecutionConfig()
    result = execute_python("def broken(:\n    pass", cfg)
    assert result.success is False
    assert result.error is not None


# ---------------------------------------------------------------------------
# 11. CodeInterpreterSession — persistent state
# ---------------------------------------------------------------------------


def test_code_interpreter_session_persistent_state():
    cfg = ExecutionConfig()
    session = CodeInterpreterSession(cfg)

    r1 = session.execute("my_var = 99")
    assert r1.success is True

    r2 = session.execute("print(my_var)")
    assert r2.success is True
    assert "99" in r2.stdout

    assert len(session.history()) == 2


# ---------------------------------------------------------------------------
# 12. CodeGenerationEvaluator — pass rate
# ---------------------------------------------------------------------------


def test_code_generation_evaluator_pass_rate():
    cfg = ExecutionConfig()
    evaluator = CodeGenerationEvaluator(cfg)

    code = "def add(a, b):\n    return a + b"
    test_cases = [
        ("print(add(1, 2))", "3"),
        ("print(add(10, 20))", "30"),
        ("print(add(-1, 1))", "0"),
    ]

    metrics = evaluator.evaluate(code, test_cases)
    assert metrics["n_total"] == 3
    assert metrics["n_passed"] == 3
    assert metrics["pass_rate"] == pytest.approx(1.0)
