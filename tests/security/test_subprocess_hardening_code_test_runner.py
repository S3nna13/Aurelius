"""Regression tests: CodeTestRunner goes through run_safe (AUR-SEC-2026-0021)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from src.agent.code_test_runner import CodeTestRunner


def test_python_path_is_absolute() -> None:
    runner = CodeTestRunner()
    assert os.path.isabs(runner.python_path)


def test_runner_runs_pytest_against_empty_dir(tmp_path: Path) -> None:
    # With no tests present, pytest exits 5 (no tests collected). We
    # only care that we actually dispatched the process through
    # run_safe without raising UnsafeSubprocessError.
    runner = CodeTestRunner(timeout=20.0, working_dir=str(tmp_path))
    result = runner.run()
    # 5 == "no tests collected". Allow 0 too for distributions that
    # print "no tests ran" and return 0.
    assert result.exit_code in {0, 5}


def test_custom_cwd_uses_hardened_wrapper(tmp_path: Path) -> None:
    # Cover the custom-cwd branch of _run_argv_via_sandbox.
    runner = CodeTestRunner(
        test_command=[sys.executable, "-c", "print('ok')"],
        timeout=10.0,
        working_dir=str(tmp_path),
    )
    # _invoke routes via _run_argv_via_sandbox with cwd set.
    result = runner.run()
    assert result.exit_code == 0
