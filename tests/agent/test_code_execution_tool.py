"""Unit tests for src.agent.code_execution_tool (16 tests).

Code execution: Inspired by Gemini 2.5 code execution tool (Google DeepMind 2025).
"""

from __future__ import annotations

from src.agent.code_execution_tool import (
    CODE_EXECUTION_TOOL_REGISTRY,
    CodeExecutionTool,
    ExecutionLanguage,
    ExecutionRequest,
    ExecutionResult,
)

# ---------------------------------------------------------------------------
# Dataclass / enum basics
# ---------------------------------------------------------------------------


def test_execution_language_enum_members() -> None:
    """ExecutionLanguage must expose PYTHON, BASH, JAVASCRIPT."""
    assert ExecutionLanguage.PYTHON is not None
    assert ExecutionLanguage.BASH is not None
    assert ExecutionLanguage.JAVASCRIPT is not None


def test_execution_language_values() -> None:
    assert ExecutionLanguage.PYTHON.value == "python"
    assert ExecutionLanguage.BASH.value == "bash"
    assert ExecutionLanguage.JAVASCRIPT.value == "javascript"


def test_execution_request_defaults() -> None:
    """ExecutionRequest must default language=PYTHON, timeout_s=10.0."""
    req = ExecutionRequest(code="pass")
    assert req.language is ExecutionLanguage.PYTHON
    assert req.timeout_s == 10.0
    assert req.capture_output is True
    assert req.env_vars == {}


def test_execution_request_custom_fields() -> None:
    req = ExecutionRequest(
        code="x=1",
        language=ExecutionLanguage.BASH,
        timeout_s=5.0,
        env_vars={"FOO": "bar"},
        capture_output=False,
    )
    assert req.language is ExecutionLanguage.BASH
    assert req.timeout_s == 5.0
    assert req.env_vars == {"FOO": "bar"}
    assert req.capture_output is False


def test_execution_result_defaults() -> None:
    res = ExecutionResult(stdout="ok", stderr="", exit_code=0, duration_ms=1.0)
    assert res.timed_out is False
    assert res.error is None


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------


def test_validate_safe_code_empty() -> None:
    """validate() on safe code must return an empty list."""
    tool = CodeExecutionTool()
    req = ExecutionRequest(code="print('hello world')\nx = 1 + 2\n")
    assert tool.validate(req) == []


def test_validate_import_os_blocked() -> None:
    """'import os' must appear in violations."""
    tool = CodeExecutionTool()
    req = ExecutionRequest(code="import os\nprint(os.getcwd())")
    violations = tool.validate(req)
    assert len(violations) > 0
    assert any("os" in v for v in violations)


def test_validate_subprocess_blocked() -> None:
    """'subprocess' literal must trigger a violation."""
    tool = CodeExecutionTool()
    req = ExecutionRequest(code="import subprocess\nsubprocess.run(['ls'])")
    violations = tool.validate(req)
    assert any("subprocess" in v for v in violations)


def test_validate_import_sys_blocked() -> None:
    tool = CodeExecutionTool()
    req = ExecutionRequest(code="import sys; print(sys.version)")
    violations = tool.validate(req)
    assert any("sys" in v for v in violations)


def test_validate_dunder_import_blocked() -> None:
    tool = CodeExecutionTool()
    req = ExecutionRequest(code="m = __import__('os')")
    violations = tool.validate(req)
    assert any("__import__" in v for v in violations)


def test_validate_multiple_violations() -> None:
    tool = CodeExecutionTool()
    req = ExecutionRequest(code="import os\nimport sys\n")
    violations = tool.validate(req)
    assert any("os" in v for v in violations)
    assert any("sys" in v for v in violations)


# ---------------------------------------------------------------------------
# execute()
# ---------------------------------------------------------------------------


def test_execute_safe_print_hello() -> None:
    """Safe code that prints 'hello' must exit 0 and capture stdout."""
    tool = CodeExecutionTool()
    req = ExecutionRequest(code="print('hello')")
    result = tool.execute(req)
    assert result.exit_code == 0
    assert "hello" in result.stdout
    assert result.timed_out is False
    assert result.error is None


def test_execute_blocked_code_returns_blocked() -> None:
    """Blocked code must return exit_code=1 with 'Blocked' in stderr."""
    tool = CodeExecutionTool()
    req = ExecutionRequest(code="import os\nprint(os.listdir('.'))")
    result = tool.execute(req)
    assert result.exit_code == 1
    assert "Blocked" in result.stderr


def test_execute_javascript_not_supported() -> None:
    """JavaScript execution must always return exit_code=1 and the stub message."""
    tool = CodeExecutionTool()
    req = ExecutionRequest(code="console.log('hi')", language=ExecutionLanguage.JAVASCRIPT)
    result = tool.execute(req)
    assert result.exit_code == 1
    assert "not yet supported" in result.stderr


def test_execute_timeout() -> None:
    """An infinite loop with a very short timeout must set timed_out=True."""
    tool = CodeExecutionTool()
    req = ExecutionRequest(
        code="while True: pass",
        timeout_s=0.1,
    )
    result = tool.execute(req)
    assert result.timed_out is True
    assert result.exit_code == 1


def test_execute_duration_ms_positive() -> None:
    tool = CodeExecutionTool()
    req = ExecutionRequest(code="x = 1 + 1")
    result = tool.execute(req)
    assert result.duration_ms >= 0.0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_code_execution_tool_registry_has_default() -> None:
    """CODE_EXECUTION_TOOL_REGISTRY must contain 'default' key."""
    assert "default" in CODE_EXECUTION_TOOL_REGISTRY
    assert CODE_EXECUTION_TOOL_REGISTRY["default"] is CodeExecutionTool


def test_deny_patterns_is_frozenset() -> None:
    assert isinstance(CodeExecutionTool.DENY_PATTERNS, frozenset)
    assert len(CodeExecutionTool.DENY_PATTERNS) >= 10
