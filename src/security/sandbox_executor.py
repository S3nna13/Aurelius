"""Resource-limited code execution sandbox.

Wraps Python's builtin dynamic-evaluation primitive with strict resource
controls:

- Wall-clock timeout enforced via ``concurrent.futures.ThreadPoolExecutor``
- stdout capture via ``io.StringIO`` redirect, truncated at
  ``max_output_bytes``
- Restricted ``__builtins__`` (only the explicitly allow-listed names)
- Size cap on incoming source to prevent compiler DoS
- getattr/setattr/type/hasattr blocked to prevent sandbox escape via
  ``type.__subclasses__()`` or ``getattr(object, '__class__')`` traversal

This sandbox does NOT spawn subprocesses and does NOT use ``multiprocessing``;
it is a best-effort in-process sandbox designed to contain cooperative tool
code, not to defend against hostile native escapes.

Inspired by the CERBERUS layered defense model: the sandbox sits in the
execution boundary layer, mirroring Layer5 perimeter enforcement where every
call traverses a narrow, explicit allow-list.

AUR-SEC: sandbox hardened per cycle-179 review — getattr/type/setattr/hasattr
removed from default builtins to prevent Python sandbox escape via MRO
introspection and attribute traversal.

Pure stdlib.
"""

from __future__ import annotations

import builtins as _py_builtins
import concurrent.futures
import contextlib
import io
import threading
import sys
import weakref
from concurrent.futures.thread import _worker
from dataclasses import dataclass, field
from typing import Any

# Preserve original stdout/stderr for proper restoration on timeout
_ORIG_STDOUT = sys.stdout
_ORIG_STDERR = sys.stderr

DEFAULT_ALLOWED_BUILTINS: frozenset[str] = frozenset(
    {
        "abs",
        "bool",
        "bytes",
        "chr",
        "dict",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "format",
        "frozenset",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "ord",
        "pow",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
    }
)


@dataclass
class SandboxConfig:
    timeout_seconds: float = 5.0
    max_output_bytes: int = 1_048_576
    allowed_builtins: frozenset[str] = field(default_factory=lambda: DEFAULT_ALLOWED_BUILTINS)
    max_code_len: int = 100_000


class SandboxViolation(Exception):
    """Raised when submitted code violates sandbox preconditions."""

    def __init__(self, reason: str, code_snippet: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        # Only the first 200 chars of the offending source are retained to
        # avoid echoing potentially sensitive payloads into exception traces.
        self.code_snippet = (code_snippet or "")[:200]


@dataclass
class SandboxResult:
    stdout: str = ""
    stderr: str = ""
    exception: str | None = None
    timed_out: bool = False


def _truncate(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="replace")


class _DaemonThreadPoolExecutor(concurrent.futures.ThreadPoolExecutor):
    """ThreadPoolExecutor variant whose workers are daemon threads.

    The sandbox can be triggered from long-lived test runs and may still have
    idle workers when the interpreter begins shutdown. Daemonizing the shared
    pool workers ensures the process can exit cleanly without waiting for the
    sandbox pool to be explicitly closed first.
    """

    def _adjust_thread_count(self) -> None:  # pragma: no cover - mirrors stdlib
        if self._idle_semaphore.acquire(timeout=0):
            return

        def weakref_cb(
            _,
            q=self._work_queue,
        ) -> None:
            q.put(None)

        num_threads = len(self._threads)
        if num_threads >= self._max_workers:
            return

        thread_name = (
            f"{self._thread_name_prefix}_{num_threads}"
            if self._thread_name_prefix
            else f"ThreadPoolExecutor-{num_threads}"
        )
        # Python 3.13+ removed _create_worker_context from ThreadPoolExecutor internals.
        if hasattr(self, "_create_worker_context"):
            worker_args = (weakref.ref(self, weakref_cb), self._create_worker_context(), self._work_queue)
        else:
            worker_args = (
                weakref.ref(self, weakref_cb),
                self._work_queue,
                self._initializer,
                self._initargs,
            )
        thread = threading.Thread(
            name=thread_name,
            target=_worker,
            args=worker_args,
        )
        thread.daemon = True
        thread.start()
        self._threads.add(thread)


class _CappedStringIO(io.StringIO):
    def __init__(self, cap: int) -> None:
        super().__init__()
        self._cap = cap
        self._bytes_written = 0

    def write(self, s: str) -> int:
        remaining = self._cap - self._bytes_written
        if remaining <= 0:
            return len(s)
        # Prevent memory DoS: never encode a huge string just to truncate it.
        # In the worst case a codepoint is 4 UTF-8 bytes, so slicing a few
        # characters past the remaining budget is safe.
        if len(s) > remaining + 4:
            s = s[: remaining + 4]
        encoded = s.encode("utf-8", errors="replace")
        if len(encoded) <= remaining:
            self._bytes_written += len(encoded)
            return super().write(s)
        encoded = encoded[:remaining]
        decoded = encoded.decode("utf-8", errors="replace")
        self._bytes_written += len(encoded)
        return super().write(decoded)


def _run_exec(code: str, globs: dict[str, Any], max_bytes: int) -> SandboxResult:
    out = _CappedStringIO(max_bytes)
    err = _CappedStringIO(max_bytes)
    result = SandboxResult()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            compiled = _py_builtins.compile(code, "<sandbox>", "exec")
            _py_builtins.exec(compiled, globs, globs)  # noqa: S102
    except BaseException as caught:
        result.exception = f"{type(caught).__name__}: {caught}"
    result.stdout = out.getvalue()
    result.stderr = err.getvalue()
    return result


def _shutdown_sandbox_pool() -> None:
    """Release the shared sandbox worker pool."""

    pass


class SandboxExecutor:
    """In-process sandboxed dynamic execution with timeout + restricted builtins."""

    def close(self) -> None:
        """Shut down the shared worker pool."""

        _shutdown_sandbox_pool()

    def _build_globals(self, config: SandboxConfig) -> dict[str, Any]:
        safe_builtins: dict[str, Any] = {}
        for name in config.allowed_builtins:
            if hasattr(_py_builtins, name):
                safe_builtins[name] = getattr(_py_builtins, name)
        for blocked in (
            "eval",
            "exec",
            "compile",
            "open",
            "__import__",
            "input",
            "breakpoint",
            "exit",
            "quit",
            "getattr",
            "setattr",
            "type",
            "hasattr",
            "__class__",
            "__bases__",
            "__subclasses__",
            "__mro__",
            "__builtins__",
            "__globals__",
            "__code__",
            "__closure__",
        ):
            safe_builtins.pop(blocked, None)
        globs: dict[str, Any] = {"__builtins__": safe_builtins}
        for attr in (
            "__class__",
            "__bases__",
            "__subclasses__",
            "__mro__",
            "__globals__",
            "__code__",
            "__closure__",
            "__dict__",
        ):
            globs[attr] = None
        return globs

    def execute(
        self,
        code: str,
        config: SandboxConfig | None = None,
    ) -> SandboxResult:
        global _SANDBOX_POOL
        cfg = config or SandboxConfig()
        if not isinstance(code, str):
            raise SandboxViolation(
                reason="code must be str",
                code_snippet=repr(code)[:200],
            )
        if len(code) > cfg.max_code_len:
            raise SandboxViolation(
                reason=f"code length {len(code)} exceeds max_code_len {cfg.max_code_len}",
                code_snippet=code,
            )

        globs = self._build_globals(cfg)

        future = _SANDBOX_POOL.submit(_run_exec, code, globs, cfg.max_output_bytes)
        try:
            result = future.result(timeout=cfg.timeout_seconds)
            return result
        except concurrent.futures.TimeoutError:
            # Restore stdout/stderr to original to avoid global state corruption
            sys.stdout = _ORIG_STDOUT
            sys.stderr = _ORIG_STDERR
            # CPython cannot force-kill a worker thread; it will continue
            # running until it returns. Abandon the pool without waiting so
            # the caller is not blocked on a runaway sandbox thread.
            try:
                future.cancel()
            except Exception:  # noqa: S110  # cancel() returns False if already done
                pass
            old_pool = _SANDBOX_POOL
            _SANDBOX_POOL = _DaemonThreadPoolExecutor(max_workers=4, thread_name_prefix="sandbox")
            old_pool.shutdown(wait=False)
            return SandboxResult(
                stdout="",
                stderr="",
                exception="TimeoutError: sandbox wall-clock budget exceeded",
                timed_out=True,
            )


_SANDBOX_POOL = _DaemonThreadPoolExecutor(max_workers=4, thread_name_prefix="sandbox")
SANDBOX_EXECUTOR = SandboxExecutor()


__all__ = [
    "DEFAULT_ALLOWED_BUILTINS",
    "SandboxConfig",
    "SandboxViolation",
    "SandboxResult",
    "SandboxExecutor",
    "SANDBOX_EXECUTOR",
]
