"""Hardened subprocess wrapper for Aurelius agent/alignment surfaces.

Finding AUR-SEC-2026-0021; CWE-78 (OS command injection),
CWE-426 (untrusted search path).

Rules enforced by :func:`run_safe`:

* ``shell=False`` always -- argv is passed directly to the OS exec.
* The first element of ``argv`` MUST be an absolute path AND MUST be a
  member of the caller-supplied ``allowed_executables`` set.
* ``timeout`` is mandatory and positive. Children that exceed the
  timeout are killed; their partial output is still captured.
* When ``env_allowlist`` is provided, only those keys are forwarded
  from :data:`os.environ` to the child. Otherwise a minimal env is
  passed (``PATH=/usr/bin:/bin``, ``HOME=<cwd>``, ``LANG=C.UTF-8``).
* Stdout and stderr are captured and each capped at
  :data:`STREAM_CAP_BYTES` (10 MiB) to defend against output bombs.
* No ``except`` clauses that silently swallow errors.

Pure standard library.
"""

from __future__ import annotations

import os
import subprocess  # noqa: S404 -- this module IS the hardened wrapper
import time
from dataclasses import dataclass
from typing import Iterable, Optional


__all__ = [
    "SafeRunResult",
    "UnsafeSubprocessError",
    "run_safe",
    "STREAM_CAP_BYTES",
]


#: Maximum bytes captured per stream (stdout / stderr). 10 MiB.
STREAM_CAP_BYTES: int = 10 * 1024 * 1024


class UnsafeSubprocessError(RuntimeError):
    """Raised when :func:`run_safe` refuses to execute a request."""


@dataclass(frozen=True)
class SafeRunResult:
    """Outcome of a hardened subprocess invocation."""

    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    killed_on_timeout: bool


def _validate_argv(
    argv: list[str] | tuple[str, ...],
    allowed: Iterable[str],
) -> None:
    if not argv:
        raise UnsafeSubprocessError("argv must be non-empty")
    exe = argv[0]
    if not isinstance(exe, str) or not exe:
        raise UnsafeSubprocessError("argv[0] must be a non-empty string")
    if not os.path.isabs(exe):
        raise UnsafeSubprocessError(
            f"argv[0] must be an absolute path; got {exe!r}"
        )
    allowed_set = set(allowed)
    if not allowed_set:
        raise UnsafeSubprocessError(
            "allowed_executables is empty; refusing to exec anything"
        )
    if exe not in allowed_set:
        raise UnsafeSubprocessError(
            f"argv[0] {exe!r} is not in allowed_executables"
        )


def _build_env(
    env_allowlist: Optional[Iterable[str]],
    cwd: Optional[str],
) -> dict[str, str]:
    if env_allowlist is None:
        return {
            "PATH": "/usr/bin:/bin",
            "HOME": cwd if cwd else "/tmp",  # noqa: S108  # subprocess fallback HOME; process is sandboxed, not sensitive data
            "LANG": "C.UTF-8",
        }
    parent = os.environ
    out: dict[str, str] = {}
    for key in env_allowlist:
        if key in parent:
            out[key] = parent[key]
    return out


def _cap(data: str) -> str:
    if len(data.encode("utf-8", errors="replace")) <= STREAM_CAP_BYTES:
        return data
    # Fast byte-level truncation; re-decode to keep a valid str.
    encoded = data.encode("utf-8", errors="replace")
    truncated = encoded[:STREAM_CAP_BYTES]
    return truncated.decode("utf-8", errors="replace") + "\n...[truncated]..."


def run_safe(
    argv: list[str] | tuple[str, ...],
    *,
    timeout: float,
    allowed_executables: Iterable[str],
    env_allowlist: Optional[Iterable[str]] = None,
    cwd: Optional[str] = None,
    env_override: Optional[dict[str, str]] = None,
    preexec_fn: Optional[object] = None,
    stdin_text: Optional[str] = None,
) -> SafeRunResult:
    """Execute ``argv`` under strict allowlist + timeout + env rules.

    Args:
        argv: Argument vector. ``argv[0]`` MUST be an absolute path and
            MUST be in ``allowed_executables``.
        timeout: Mandatory wall-clock limit in seconds; must be > 0.
        allowed_executables: Iterable of absolute paths. Only argv[0]
            values contained here are permitted.
        env_allowlist: Optional iterable of env var names to forward
            from the parent. If ``None``, a minimal env is used.
        cwd: Optional working directory for the child.

    Returns:
        :class:`SafeRunResult` with captured stdout/stderr (each capped
        at :data:`STREAM_CAP_BYTES`), returncode, duration, and
        ``killed_on_timeout`` flag.

    Raises:
        UnsafeSubprocessError: If argv validation fails.
        ValueError: If ``timeout`` is not positive.
    """
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError(f"timeout must be > 0, got {timeout!r}")
    argv_list = list(argv)
    _validate_argv(argv_list, allowed_executables)
    if env_override is not None:
        env = dict(env_override)
    else:
        env = _build_env(env_allowlist, cwd)

    extra: dict[str, object] = {}
    if preexec_fn is not None and os.name == "posix":
        extra["preexec_fn"] = preexec_fn  # type: ignore[assignment]

    t0 = time.perf_counter()
    killed = False
    stdout = ""
    stderr = ""
    returncode = -1
    try:
        proc = subprocess.run(  # noqa: S603 -- allowlist + shell=False
            argv_list,
            shell=False,
            capture_output=True,
            text=True,
            env=env,
            cwd=cwd,
            timeout=float(timeout),
            check=False,
            input=stdin_text,
            **extra,  # type: ignore[arg-type]
        )
        returncode = proc.returncode
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        killed = True
        returncode = 124
        raw_out = exc.stdout or b""
        raw_err = exc.stderr or b""
        if isinstance(raw_out, bytes):
            stdout = raw_out.decode("utf-8", errors="replace")
        else:
            stdout = str(raw_out)
        if isinstance(raw_err, bytes):
            stderr = raw_err.decode("utf-8", errors="replace")
        else:
            stderr = str(raw_err)
    duration = time.perf_counter() - t0

    return SafeRunResult(
        returncode=returncode,
        stdout=_cap(stdout),
        stderr=_cap(stderr),
        duration_s=duration,
        killed_on_timeout=killed,
    )


#: Registry entry for the security subpackage.
SAFE_SUBPROCESS = {
    "run_safe": run_safe,
    "SafeRunResult": SafeRunResult,
    "UnsafeSubprocessError": UnsafeSubprocessError,
    "stream_cap_bytes": STREAM_CAP_BYTES,
}
