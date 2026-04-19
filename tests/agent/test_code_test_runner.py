"""Unit tests for :mod:`src.agent.code_test_runner`.

These tests generate tiny test files inside pytest's ``tmp_path`` and
invoke :class:`CodeTestRunner` against them. We use ``--rootdir`` /
``-p no:cacheprovider`` in custom commands where relevant to keep
nested pytest invocations hermetic.
"""

from __future__ import annotations

import sys
import textwrap

import pytest

from src.agent.code_test_runner import CodeTestRunner, TestResult
from src.agent.code_test_runner import (
    _parse_counts,
    _parse_failed_names,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(path, contents: str) -> None:
    path.write_text(textwrap.dedent(contents).lstrip("\n"))


def _make_runner(working_dir, timeout: float = 60.0) -> CodeTestRunner:
    # Add a hermetic rootdir and disable cache so we don't pollute the
    # parent project or pick up outer conftests.
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--tb=short",
        "-p",
        "no:cacheprovider",
        "--rootdir",
        str(working_dir),
    ]
    return CodeTestRunner(
        test_command=cmd,
        timeout=timeout,
        working_dir=str(working_dir),
    )


# ---------------------------------------------------------------------------
# Core behaviour
# ---------------------------------------------------------------------------


def test_single_passing_test(tmp_path) -> None:
    _write(
        tmp_path / "test_ok.py",
        """
        def test_pass():
            assert 1 + 1 == 2
        """,
    )
    runner = _make_runner(tmp_path)
    result = runner.run()
    assert isinstance(result, TestResult)
    assert result.passed == 1
    assert result.failed == 0
    assert result.total == 1


def test_one_pass_one_fail(tmp_path) -> None:
    _write(
        tmp_path / "test_mixed.py",
        """
        def test_pass():
            assert True

        def test_fail():
            assert False
        """,
    )
    runner = _make_runner(tmp_path)
    result = runner.run()
    assert result.passed == 1
    assert result.failed == 1


def test_failed_names_extracted(tmp_path) -> None:
    _write(
        tmp_path / "test_named.py",
        """
        def test_bad():
            assert 0 == 1
        """,
    )
    runner = _make_runner(tmp_path)
    result = runner.run()
    assert result.failed == 1
    assert len(result.failed_names) == 1
    assert "test_bad" in result.failed_names[0]


def test_timeout_enforced(tmp_path) -> None:
    _write(
        tmp_path / "test_slow.py",
        """
        import time

        def test_slow():
            time.sleep(30)
        """,
    )
    runner = _make_runner(tmp_path, timeout=1.0)
    result = runner.run()
    assert result.timed_out is True
    assert result.exit_code == 124


def test_syntax_error_yields_errors(tmp_path) -> None:
    _write(
        tmp_path / "test_syntax.py",
        """
        def test_broken(:
            pass
        """,
    )
    runner = _make_runner(tmp_path)
    result = runner.run()
    # pytest reports collection failures via "errors" in its summary.
    assert result.errors > 0 or result.exit_code != 0


def test_run_file_restricts_scope(tmp_path) -> None:
    _write(
        tmp_path / "test_a.py",
        """
        def test_a():
            assert True
        """,
    )
    _write(
        tmp_path / "test_b.py",
        """
        def test_b():
            assert True
        """,
    )
    runner = _make_runner(tmp_path)
    result = runner.run_file(str(tmp_path / "test_a.py"))
    assert result.passed == 1
    assert result.total == 1


def test_run_function_filters_via_k(tmp_path) -> None:
    _write(
        tmp_path / "test_k.py",
        """
        def test_alpha():
            assert True

        def test_beta():
            assert True

        def test_gamma():
            assert True
        """,
    )
    runner = _make_runner(tmp_path)
    result = runner.run_function(str(tmp_path / "test_k.py"), "test_beta")
    assert result.passed == 1
    assert result.total == 1


def test_empty_test_dir_total_zero(tmp_path) -> None:
    # No test files present.
    runner = _make_runner(tmp_path)
    result = runner.run()
    assert result.total == 0
    assert result.passed == 0
    assert result.failed == 0


def test_custom_test_command_works(tmp_path) -> None:
    # Use a custom command that prints a fake pytest-ish summary line.
    runner = CodeTestRunner(
        test_command=[sys.executable, "-c", "print('1 passed in 0.01s')"],
        timeout=10.0,
        working_dir=str(tmp_path),
    )
    result = runner.run()
    assert result.passed == 1
    assert result.total == 1


def test_stdout_captured(tmp_path) -> None:
    runner = CodeTestRunner(
        test_command=[sys.executable, "-c", "print('hello from child')"],
        timeout=10.0,
        working_dir=str(tmp_path),
    )
    result = runner.run()
    assert "hello from child" in result.stdout


def test_stderr_captured(tmp_path) -> None:
    runner = CodeTestRunner(
        test_command=[
            sys.executable,
            "-c",
            "import sys; sys.stderr.write('err!\\n')",
        ],
        timeout=10.0,
        working_dir=str(tmp_path),
    )
    result = runner.run()
    assert "err!" in result.stderr


def test_determinism_on_same_fixtures(tmp_path) -> None:
    _write(
        tmp_path / "test_det.py",
        """
        def test_one():
            assert True

        def test_two():
            assert 2 == 2
        """,
    )
    runner = _make_runner(tmp_path)
    r1 = runner.run()
    r2 = runner.run()
    assert r1.passed == r2.passed == 2
    assert r1.failed == r2.failed == 0
    assert r1.total == r2.total == 2


def test_parse_summary_three_failed_five_passed() -> None:
    text = "===== 3 failed, 5 passed in 0.42s ====="
    counts = _parse_counts(text)
    assert counts["failed"] == 3
    assert counts["passed"] == 5


def test_parse_failed_names_direct() -> None:
    text = (
        "FAILED tests/test_x.py::test_one - AssertionError\n"
        "FAILED tests/test_x.py::test_two - AssertionError\n"
    )
    names = _parse_failed_names(text)
    assert names == [
        "tests/test_x.py::test_one",
        "tests/test_x.py::test_two",
    ]


def test_duration_ms_populated(tmp_path) -> None:
    _write(
        tmp_path / "test_dur.py",
        """
        def test_dur():
            assert True
        """,
    )
    runner = _make_runner(tmp_path)
    result = runner.run()
    assert result.duration_ms > 0.0


def test_timeout_must_be_positive(tmp_path) -> None:
    with pytest.raises(ValueError):
        CodeTestRunner(timeout=0, working_dir=str(tmp_path))


def test_bad_working_dir_raises(tmp_path) -> None:
    with pytest.raises(NotADirectoryError):
        CodeTestRunner(working_dir=str(tmp_path / "nope"))
