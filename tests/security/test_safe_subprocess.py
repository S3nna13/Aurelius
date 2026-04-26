"""Tests for :mod:`src.security.safe_subprocess`.

Finding AUR-SEC-2026-0021; CWE-78 (OS command injection),
CWE-426 (untrusted search path).
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import pytest

from src.security.safe_subprocess import (
    STREAM_CAP_BYTES,
    SafeRunResult,
    UnsafeSubprocessError,
    run_safe,
)

PY = os.path.realpath(sys.executable)


# ---------------------------------------------------------------------------
# 1. relative path rejected
# ---------------------------------------------------------------------------


def test_relative_executable_rejected() -> None:
    with pytest.raises(UnsafeSubprocessError):
        run_safe(
            ["python3", "-c", "print(1)"],
            timeout=2.0,
            allowed_executables={PY},
        )


# ---------------------------------------------------------------------------
# 2. non-allowlisted absolute path rejected
# ---------------------------------------------------------------------------


def test_non_allowlisted_absolute_rejected(tmp_path: Path) -> None:
    # Some existing absolute path not in the allowlist.
    with pytest.raises(UnsafeSubprocessError):
        run_safe(
            ["/bin/ls", "/"],
            timeout=2.0,
            allowed_executables={PY},
        )


# ---------------------------------------------------------------------------
# 3. timeout kills long-running child
# ---------------------------------------------------------------------------


def test_timeout_kills_child() -> None:
    res = run_safe(
        [PY, "-c", "import time; time.sleep(30)"],
        timeout=0.5,
        allowed_executables={PY},
    )
    assert res.killed_on_timeout is True
    assert res.duration_s < 5.0


# ---------------------------------------------------------------------------
# 4. env allowlist applied
# ---------------------------------------------------------------------------


def test_env_allowlist_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AURELIUS_TEST_ALLOWED", "hello")
    monkeypatch.setenv("AURELIUS_TEST_SECRET", "nope")
    code = (
        "import os; "
        "print('A=' + os.environ.get('AURELIUS_TEST_ALLOWED','')); "
        "print('B=' + os.environ.get('AURELIUS_TEST_SECRET',''))"
    )
    res = run_safe(
        [PY, "-c", code],
        timeout=5.0,
        allowed_executables={PY},
        env_allowlist={"AURELIUS_TEST_ALLOWED", "PATH"},
    )
    assert res.returncode == 0
    assert "A=hello" in res.stdout
    assert "B=" in res.stdout and "B=nope" not in res.stdout


# ---------------------------------------------------------------------------
# 5. minimal env when no allowlist supplied
# ---------------------------------------------------------------------------


def test_minimal_env_when_no_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AURELIUS_LEAK", "leaky")
    code = (
        "import os; "
        "print('LEAK=' + os.environ.get('AURELIUS_LEAK','none')); "
        "print('PATH=' + os.environ.get('PATH',''));"
        "print('LANG=' + os.environ.get('LANG',''));"
    )
    res = run_safe(
        [PY, "-c", code],
        timeout=5.0,
        allowed_executables={PY},
    )
    assert res.returncode == 0
    assert "LEAK=none" in res.stdout
    assert "PATH=/usr/bin:/bin" in res.stdout
    assert "LANG=C.UTF-8" in res.stdout


# ---------------------------------------------------------------------------
# 6. stdout/stderr captured
# ---------------------------------------------------------------------------


def test_stdout_stderr_captured() -> None:
    code = "import sys; sys.stdout.write('O'); sys.stderr.write('E'); "
    res = run_safe(
        [PY, "-c", code],
        timeout=5.0,
        allowed_executables={PY},
    )
    assert "O" in res.stdout
    assert "E" in res.stderr


# ---------------------------------------------------------------------------
# 7. returncode propagated
# ---------------------------------------------------------------------------


def test_returncode_propagated() -> None:
    res = run_safe(
        [PY, "-c", "import sys; sys.exit(7)"],
        timeout=5.0,
        allowed_executables={PY},
    )
    assert res.returncode == 7


# ---------------------------------------------------------------------------
# 8. allowlist empty rejects all
# ---------------------------------------------------------------------------


def test_empty_allowlist_rejects_all() -> None:
    with pytest.raises(UnsafeSubprocessError):
        run_safe(
            [PY, "-c", "print(1)"],
            timeout=2.0,
            allowed_executables=set(),
        )


# ---------------------------------------------------------------------------
# 9. concurrent calls thread-safe
# ---------------------------------------------------------------------------


def test_concurrent_calls_thread_safe() -> None:
    results: list[SafeRunResult] = []
    errors: list[BaseException] = []

    def worker(n: int) -> None:
        try:
            r = run_safe(
                [PY, "-c", f"print({n})"],
                timeout=10.0,
                allowed_executables={PY},
            )
            results.append(r)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(results) == 8
    assert all(r.returncode == 0 for r in results)


# ---------------------------------------------------------------------------
# 10. shell=False always (no shell injection via argv)
# ---------------------------------------------------------------------------


def test_no_shell_injection_via_argv() -> None:
    # If shell were True, '; echo HACKED' would run. We pass it as one
    # literal argument to python, which rejects it as a syntax error
    # OR treats it as a literal string. Either way, 'HACKED' must not
    # appear in stdout of an echo'd string.
    code = "print('safe')"
    injected = code + "; echo HACKED"
    res = run_safe(
        [PY, "-c", injected],
        timeout=5.0,
        allowed_executables={PY},
    )
    assert "HACKED" not in res.stdout
    # Either syntax error (nonzero) or printed text; but never invoked shell.


# ---------------------------------------------------------------------------
# 11. output truncation (bomb protection)
# ---------------------------------------------------------------------------


def test_output_truncation_caps_stdout() -> None:
    # Produce ~12MB, should be capped at STREAM_CAP_BYTES (10MB).
    code = (
        "import sys;"
        "chunk = 'A' * 65536;"
        "total = 0;"
        "target = 12 * 1024 * 1024;"
        "\nwhile total < target:\n"
        "    sys.stdout.write(chunk)\n"
        "    total += len(chunk)\n"
    )
    res = run_safe(
        [PY, "-c", code],
        timeout=30.0,
        allowed_executables={PY},
    )
    assert len(res.stdout.encode("utf-8")) <= STREAM_CAP_BYTES + 1024


# ---------------------------------------------------------------------------
# 12. argv must be non-empty
# ---------------------------------------------------------------------------


def test_empty_argv_rejected() -> None:
    with pytest.raises(UnsafeSubprocessError):
        run_safe(
            [],
            timeout=2.0,
            allowed_executables={PY},
        )


# ---------------------------------------------------------------------------
# 13. timeout is mandatory (must be > 0)
# ---------------------------------------------------------------------------


def test_timeout_must_be_positive() -> None:
    with pytest.raises((ValueError, UnsafeSubprocessError)):
        run_safe(
            [PY, "-c", "print(1)"],
            timeout=0.0,
            allowed_executables={PY},
        )


# ---------------------------------------------------------------------------
# 14. cwd is honored
# ---------------------------------------------------------------------------


def test_cwd_is_honored(tmp_path: Path) -> None:
    res = run_safe(
        [PY, "-c", "import os; print(os.getcwd())"],
        timeout=5.0,
        allowed_executables={PY},
        cwd=str(tmp_path),
    )
    assert res.returncode == 0
    # tmp_path may resolve symlinks; compare resolved
    assert str(tmp_path.resolve()) in os.path.realpath(
        res.stdout.strip()
    ) or res.stdout.strip() == str(tmp_path)


# ---------------------------------------------------------------------------
# 15. duration reported
# ---------------------------------------------------------------------------


def test_duration_reported() -> None:
    res = run_safe(
        [PY, "-c", "print(1)"],
        timeout=5.0,
        allowed_executables={PY},
    )
    assert res.duration_s >= 0.0
    assert res.killed_on_timeout is False


# ---------------------------------------------------------------------------
# 16. non-absolute first arg (e.g. plain name) rejected even if it exists on PATH
# ---------------------------------------------------------------------------


def test_bare_command_rejected() -> None:
    with pytest.raises(UnsafeSubprocessError):
        run_safe(
            ["ls"],
            timeout=2.0,
            allowed_executables={"/bin/ls"},
        )
