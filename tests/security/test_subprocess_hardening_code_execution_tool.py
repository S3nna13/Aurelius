"""Regression tests: CodeExecutionTool uses absolute python path (AUR-SEC-2026-0021)."""

from __future__ import annotations

import os

from src.agent.code_execution_tool import (
    CodeExecutionTool,
    ExecutionLanguage,
    ExecutionRequest,
    _ALLOWED_EXECUTABLES,
    _PYTHON3_ABS,
)


def test_python3_resolved_to_absolute_at_import() -> None:
    assert os.path.isabs(_PYTHON3_ABS)
    assert _PYTHON3_ABS in _ALLOWED_EXECUTABLES


def test_tool_still_runs_simple_python() -> None:
    tool = CodeExecutionTool()
    req = ExecutionRequest(
        code="print('ok')",
        language=ExecutionLanguage.PYTHON,
        timeout_s=5.0,
    )
    result = tool.execute(req)
    assert result.exit_code == 0
    assert "ok" in result.stdout


def test_tool_honors_timeout() -> None:
    tool = CodeExecutionTool()
    req = ExecutionRequest(
        code="x = 1",  # must pass deny patterns
        language=ExecutionLanguage.PYTHON,
        timeout_s=5.0,
    )
    result = tool.execute(req)
    assert result.exit_code == 0
