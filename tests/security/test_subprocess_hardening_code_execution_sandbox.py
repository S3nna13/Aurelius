"""Regression tests: sandbox uses run_safe (AUR-SEC-2026-0021)."""

from __future__ import annotations

import os
import sys

import pytest

from src.agent.code_execution_sandbox import (
    CodeExecutionSandbox,
    _DEFAULT_PYTHON_ABS,
)
from src.security.safe_subprocess import UnsafeSubprocessError


def test_python_path_resolved_to_absolute() -> None:
    sb = CodeExecutionSandbox()
    assert os.path.isabs(sb.python_path)
    assert sb.python_path == _DEFAULT_PYTHON_ABS


def test_relative_python_path_is_resolved() -> None:
    # realpath makes any existing path absolute; a non-existent relative
    # path resolves to an absolute path too (cwd-joined).
    sb = CodeExecutionSandbox(python_path=sys.executable)
    assert os.path.isabs(sb.python_path)


def test_allowed_executables_includes_interpreter() -> None:
    sb = CodeExecutionSandbox()
    assert sb.python_path in sb._allowed_executables


def test_sandbox_still_runs_code() -> None:
    sb = CodeExecutionSandbox(timeout=5.0)
    res = sb.execute("print('hardened')")
    assert res.exit_code == 0
    assert "hardened" in res.stdout


def test_sandbox_timeout_propagates() -> None:
    sb = CodeExecutionSandbox(timeout=0.5)
    res = sb.execute("import time; time.sleep(30)")
    assert res.timed_out is True
