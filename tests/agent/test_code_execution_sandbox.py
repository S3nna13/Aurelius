"""Unit tests for :mod:`src.agent.code_execution_sandbox`."""

from __future__ import annotations

import os
import sys

import pytest

from src.agent.code_execution_sandbox import (
    TRUNCATION_MARKER,
    CodeExecutionSandbox,
    ExecutionResult,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sbx() -> CodeExecutionSandbox:
    return CodeExecutionSandbox(timeout=5.0, max_memory_mb=256)


# ---------------------------------------------------------------------------
# basic execution
# ---------------------------------------------------------------------------


def test_hello_world_exit_zero(sbx: CodeExecutionSandbox) -> None:
    res = sbx.execute('print("hello")')
    assert isinstance(res, ExecutionResult)
    assert res.exit_code == 0
    assert "hello" in res.stdout
    assert res.timed_out is False
    assert res.duration_ms >= 0.0


def test_system_exit_nonzero(sbx: CodeExecutionSandbox) -> None:
    res = sbx.execute("raise SystemExit(1)")
    assert res.exit_code == 1
    assert res.timed_out is False


def test_empty_code_exit_zero(sbx: CodeExecutionSandbox) -> None:
    res = sbx.execute("")
    assert res.exit_code == 0
    assert res.timed_out is False


def test_syntax_error_reported(sbx: CodeExecutionSandbox) -> None:
    res = sbx.execute("def (:\n")
    assert res.exit_code != 0
    assert "SyntaxError" in res.stderr


# ---------------------------------------------------------------------------
# timeout
# ---------------------------------------------------------------------------


def test_infinite_loop_times_out() -> None:
    sbx = CodeExecutionSandbox(timeout=1.0, max_memory_mb=128)
    res = sbx.execute("while True: pass")
    assert res.timed_out is True
    assert res.exit_code == 124


def test_negative_timeout_raises() -> None:
    with pytest.raises(ValueError):
        CodeExecutionSandbox(timeout=-1.0)
    with pytest.raises(ValueError):
        CodeExecutionSandbox(timeout=0.0)


# ---------------------------------------------------------------------------
# output handling
# ---------------------------------------------------------------------------


def test_large_stdout_truncated() -> None:
    sbx = CodeExecutionSandbox(timeout=5.0, max_output_chars=1024)
    code = 'import sys; sys.stdout.write("x" * 100000)'
    res = sbx.execute(code)
    assert res.exit_code == 0
    assert res.truncated_stdout is True
    assert res.stdout.endswith(TRUNCATION_MARKER)
    assert len(res.stdout) <= 1024


def test_stdin_text_passed(sbx: CodeExecutionSandbox) -> None:
    code = "import sys; data = sys.stdin.read(); print(data.upper())"
    res = sbx.execute(code, stdin_text="hello\n")
    assert res.exit_code == 0
    assert "HELLO" in res.stdout


# ---------------------------------------------------------------------------
# isolation
# ---------------------------------------------------------------------------


def test_subprocess_tmpdir_isolation(tmp_path) -> None:
    """Files created via relative path inside the child land in a tmp
    cwd and do not persist in ``tmp_path`` (which is not the child's
    cwd)."""

    sbx = CodeExecutionSandbox(timeout=5.0)
    marker = "sbx_marker_file.txt"
    # Child writes a relative file (lands in its own ephemeral cwd).
    code = f'open({marker!r}, "w").write("x")'
    res = sbx.execute(code)
    assert res.exit_code == 0
    assert not (tmp_path / marker).exists()
    assert not os.path.exists(marker)


def test_scrub_env_strips_pythonpath(monkeypatch) -> None:
    # Put this repo's src on PYTHONPATH for the parent; child with
    # scrub_env=True should not inherit it and thus cannot import the
    # project's package.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    monkeypatch.setenv("PYTHONPATH", repo_root)

    sbx = CodeExecutionSandbox(timeout=5.0, scrub_env=True)
    code = (
        "import sys\n"
        "try:\n"
        "    import src.agent  # noqa: F401\n"
        "    print('OK')\n"
        "except Exception as e:\n"
        "    print('NOPKG', type(e).__name__)\n"
    )
    res = sbx.execute(code)
    assert res.exit_code == 0
    assert "NOPKG" in res.stdout
    assert "OK" not in res.stdout


# ---------------------------------------------------------------------------
# execute_file
# ---------------------------------------------------------------------------


def test_execute_file(tmp_path, sbx: CodeExecutionSandbox) -> None:
    script = tmp_path / "script.py"
    script.write_text("import sys; print('args=' + ','.join(sys.argv[1:]))\n")
    res = sbx.execute_file(str(script), args=["a", "b", "c"])
    assert res.exit_code == 0
    assert "args=a,b,c" in res.stdout


def test_execute_file_missing(sbx: CodeExecutionSandbox) -> None:
    with pytest.raises(FileNotFoundError):
        sbx.execute_file("/nonexistent/path/script.py")


# ---------------------------------------------------------------------------
# memory (Unix only)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name != "posix", reason="RLIMIT_AS only on POSIX")
@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="macOS does not reliably enforce RLIMIT_AS",
)
def test_memory_limit_enforced_on_linux() -> None:
    sbx = CodeExecutionSandbox(timeout=10.0, max_memory_mb=64)
    # Allocate ~512 MB of bytes; should crash with MemoryError or
    # nonzero exit.
    code = "x = bytearray(512 * 1024 * 1024)\nprint('done', len(x))"
    res = sbx.execute(code)
    assert res.exit_code != 0 or res.timed_out
    if not res.timed_out:
        assert "MemoryError" in res.stderr or res.exit_code != 0


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def test_determinism(sbx: CodeExecutionSandbox) -> None:
    code = 'print("42")'
    r1 = sbx.execute(code)
    r2 = sbx.execute(code)
    assert r1.exit_code == r2.exit_code == 0
    assert r1.stdout == r2.stdout
    assert r1.stderr == r2.stderr
    assert r1.truncated_stdout == r2.truncated_stdout


def test_invalid_config_values() -> None:
    with pytest.raises(ValueError):
        CodeExecutionSandbox(max_memory_mb=0)
    with pytest.raises(ValueError):
        CodeExecutionSandbox(max_output_chars=-1)
