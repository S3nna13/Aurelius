"""Safe-ish per-call Python code execution sandbox.

The :class:`CodeExecutionSandbox` runs short Python snippets in a
fresh child process launched via :mod:`subprocess`. It applies basic
hardening:

* Isolated interpreter mode (``python -I -S``) so site-packages, user
  ``PYTHONSTARTUP`` files, and the current ``sys.path[0]`` are not
  injected into the child.
* A temporary working directory that is torn down after the call so
  stray files created by the snippet do not pollute the parent.
* A scrubbed environment that strips ``PYTHONPATH``, ``PYTHONHOME``,
  ``PYTHONSTARTUP``, and proxy variables.
* Resource limits via :mod:`resource` (Unix only): CPU time, address
  space, data segment, and core dump size.
* A wall-clock timeout that forcibly terminates runaway children.
* Truncation of captured stdout/stderr with an explicit marker.

This is a defense-in-depth wrapper, NOT a real security sandbox.

Pure standard library.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Optional

try:  # pragma: no cover - platform branch
    import resource  # type: ignore[attr-defined]

    _HAS_RESOURCE = True
except ImportError:  # pragma: no cover - Windows
    resource = None  # type: ignore[assignment]
    _HAS_RESOURCE = False


__all__ = [
    "ExecutionResult",
    "CodeExecutionSandbox",
    "TRUNCATION_MARKER",
    "SCRUBBED_ENV_VARS",
    "NETWORK_ENV_VARS",
]


TRUNCATION_MARKER = "\n...[truncated]..."

SCRUBBED_ENV_VARS: tuple[str, ...] = (
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "PYTHONHOME",
    "PYTHONUSERBASE",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONINSPECT",
    "PYTHONNOUSERSITE",
    "PYTHONCASEOK",
)

NETWORK_ENV_VARS: tuple[str, ...] = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "FTP_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    "ftp_proxy",
)


@dataclass
class ExecutionResult:
    """Captured result of a single sandboxed Python invocation."""

    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool
    memory_mb_peak: Optional[float] = None
    truncated_stdout: bool = False
    truncated_stderr: bool = False


def _build_preexec(max_memory_mb: int, cpu_seconds: int):
    if not _HAS_RESOURCE:
        return None

    mem_bytes = int(max_memory_mb) * 1024 * 1024
    cpu_lim = max(1, int(cpu_seconds))

    def _preexec() -> None:  # pragma: no cover - runs in child
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_lim, cpu_lim))
        except (ValueError, OSError):
            pass
        for rl_name in ("RLIMIT_AS", "RLIMIT_DATA"):
            rl = getattr(resource, rl_name, None)
            if rl is None:
                continue
            try:
                resource.setrlimit(rl, (mem_bytes, mem_bytes))
            except (ValueError, OSError):
                pass
        try:
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        except (ValueError, OSError):
            pass

    return _preexec


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if limit <= 0 or len(text) <= limit:
        return text, False
    keep = max(0, limit - len(TRUNCATION_MARKER))
    return text[:keep] + TRUNCATION_MARKER, True


class CodeExecutionSandbox:
    """Per-call wrapper around :func:`subprocess.run` for untrusted Python."""

    def __init__(
        self,
        python_path: Optional[str] = None,
        timeout: float = 10.0,
        max_memory_mb: int = 512,
        max_output_chars: int = 65536,
        scrub_env: bool = True,
        disallow_network_env: bool = True,
    ) -> None:
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout!r}")
        if max_memory_mb <= 0:
            raise ValueError(
                f"max_memory_mb must be > 0, got {max_memory_mb!r}"
            )
        if max_output_chars < 0:
            raise ValueError(
                f"max_output_chars must be >= 0, got {max_output_chars!r}"
            )
        self.python_path = python_path or sys.executable
        self.timeout = float(timeout)
        self.max_memory_mb = int(max_memory_mb)
        self.max_output_chars = int(max_output_chars)
        self.scrub_env = bool(scrub_env)
        self.disallow_network_env = bool(disallow_network_env)

    def execute(
        self,
        code: str,
        stdin_text: Optional[str] = None,
    ) -> ExecutionResult:
        argv = [self.python_path, "-I", "-S", "-c", code]
        return self._run(argv, stdin_text=stdin_text)

    def execute_file(
        self,
        path: str,
        args: Optional[list[str]] = None,
    ) -> ExecutionResult:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        argv = [self.python_path, "-I", "-S", path]
        if args:
            argv.extend(str(a) for a in args)
        return self._run(argv, stdin_text=None)

    def _build_env(self) -> dict[str, str]:
        if self.scrub_env:
            env: dict[str, str] = {}
            for key in ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR"):
                if key in os.environ:
                    env[key] = os.environ[key]
        else:
            env = dict(os.environ)
        for key in SCRUBBED_ENV_VARS:
            env.pop(key, None)
        if self.disallow_network_env:
            for key in NETWORK_ENV_VARS:
                env.pop(key, None)
        env["PYTHONUNBUFFERED"] = "1"
        return env

    def _run(
        self,
        argv: list[str],
        stdin_text: Optional[str],
    ) -> ExecutionResult:
        tmpdir = tempfile.mkdtemp(prefix="aurelius_sbx_")
        env = self._build_env()
        preexec = _build_preexec(
            self.max_memory_mb, cpu_seconds=int(self.timeout) + 1
        )

        mem_before: Optional[int] = None
        if _HAS_RESOURCE:
            try:
                mem_before = resource.getrusage(
                    resource.RUSAGE_CHILDREN
                ).ru_maxrss
            except OSError:
                mem_before = None

        t0 = time.perf_counter()
        timed_out = False
        try:
            kwargs: dict[str, object] = dict(
                input=stdin_text,
                capture_output=True,
                text=True,
                cwd=tmpdir,
                env=env,
                timeout=self.timeout,
            )
            if preexec is not None and os.name == "posix":
                kwargs["preexec_fn"] = preexec
            # argv is built from validated/canoncalized paths and the caller
            # has already scrubbed the environment; S603 is acceptable here.
            proc = subprocess.run(argv, **kwargs)  # noqa: S603 # type: ignore[arg-type]
            exit_code = proc.returncode
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            exit_code = 124
            stdout_raw = exc.stdout or b""
            stderr_raw = exc.stderr or b""
            if isinstance(stdout_raw, bytes):
                stdout = stdout_raw.decode("utf-8", errors="replace")
            else:
                stdout = stdout_raw
            if isinstance(stderr_raw, bytes):
                stderr = stderr_raw.decode("utf-8", errors="replace")
            else:
                stderr = stderr_raw
        finally:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            shutil.rmtree(tmpdir, ignore_errors=True)

        memory_mb_peak: Optional[float] = None
        if _HAS_RESOURCE:
            try:
                after = resource.getrusage(
                    resource.RUSAGE_CHILDREN
                ).ru_maxrss
                raw = max(after - (mem_before or 0), after)
                if raw > 10**9:
                    memory_mb_peak = raw / (1024.0 * 1024.0)
                else:
                    memory_mb_peak = raw / 1024.0
            except OSError:
                memory_mb_peak = None

        stdout, trunc_out = _truncate(stdout, self.max_output_chars)
        stderr, trunc_err = _truncate(stderr, self.max_output_chars)

        return ExecutionResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_ms=duration_ms,
            timed_out=timed_out,
            memory_mb_peak=memory_mb_peak,
            truncated_stdout=trunc_out,
            truncated_stderr=trunc_err,
        )
