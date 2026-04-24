"""Resource-limited code execution sandbox.

Wraps Python's builtin dynamic-evaluation primitive with strict resource
controls:

- Wall-clock timeout enforced via ``concurrent.futures.ThreadPoolExecutor``
- stdout capture via ``io.StringIO`` redirect, truncated at
  ``max_output_bytes``
- Restricted ``__builtins__`` (only the explicitly allow-listed names)
- Size cap on incoming source to prevent compiler DoS

This sandbox does NOT spawn subprocesses and does NOT use ``multiprocessing``;
it is a best-effort in-process sandbox designed to contain cooperative tool
code, not to defend against hostile native escapes.

Inspired by the CERBERUS layered defense model: the sandbox sits in the
execution boundary layer, mirroring Layer5 perimeter enforcement where every
call traverses a narrow, explicit allow-list.

Pure stdlib.
"""

from __future__ import annotations

import builtins as _py_builtins
import concurrent.futures
import contextlib
import io
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Optional


DEFAULT_ALLOWED_BUILTINS: FrozenSet[str] = frozenset({
    "abs", "bool", "bytes", "chr", "dict", "divmod", "enumerate", "filter",
    "float", "format", "frozenset", "getattr", "hasattr", "int", "isinstance",
    "issubclass", "iter", "len", "list", "map", "max", "min", "next", "object",
    "ord", "pow", "print", "range", "repr", "reversed", "round", "set", "setattr",
    "slice", "sorted", "str", "sum", "tuple", "type", "zip",
})


@dataclass
class SandboxConfig:
    timeout_seconds: float = 5.0
    max_output_bytes: int = 1_048_576
    allowed_builtins: FrozenSet[str] = field(
        default_factory=lambda: DEFAULT_ALLOWED_BUILTINS
    )
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
    exception: Optional[str] = None
    timed_out: bool = False


def _truncate(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="replace")


def _run_exec(code: str, globs: Dict[str, Any], max_bytes: int) -> SandboxResult:
    out = io.StringIO()
    err = io.StringIO()
    result = SandboxResult()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            compiled = _py_builtins.compile(code, "<sandbox>", "exec")
            _py_builtins.exec(compiled, globs, globs)
    except BaseException as caught:
        result.exception = f"{type(caught).__name__}: {caught}"
    result.stdout = _truncate(out.getvalue(), max_bytes)
    result.stderr = _truncate(err.getvalue(), max_bytes)
    return result


class SandboxExecutor:
    """In-process sandboxed dynamic execution with timeout + restricted builtins."""

    def _build_globals(self, config: SandboxConfig) -> Dict[str, Any]:
        safe_builtins: Dict[str, Any] = {}
        for name in config.allowed_builtins:
            if hasattr(_py_builtins, name):
                safe_builtins[name] = getattr(_py_builtins, name)
        # Explicitly ensure dangerous names are absent even if someone extended
        # ``allowed_builtins`` with them by accident.
        for blocked in (
            "eval", "exec", "compile", "open", "__import__",
            "input", "breakpoint", "exit", "quit",
        ):
            safe_builtins.pop(blocked, None)
        return {"__builtins__": safe_builtins}

    def execute(
        self,
        code: str,
        config: Optional[SandboxConfig] = None,
    ) -> SandboxResult:
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

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(_run_exec, code, globs, cfg.max_output_bytes)
        try:
            result = future.result(timeout=cfg.timeout_seconds)
            pool.shutdown(wait=True)
            return result
        except concurrent.futures.TimeoutError:
            # CPython cannot force-kill a worker thread; it will continue
            # running until it returns. Abandon the pool without waiting so
            # the caller is not blocked on a runaway sandbox thread.
            try:
                future.cancel()
            except Exception:
                pass
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                # Older Pythons lack ``cancel_futures``; fall back silently.
                pool.shutdown(wait=False)
            return SandboxResult(
                stdout="",
                stderr="",
                exception="TimeoutError: sandbox wall-clock budget exceeded",
                timed_out=True,
            )


SANDBOX_EXECUTOR = SandboxExecutor()


__all__ = [
    "DEFAULT_ALLOWED_BUILTINS",
    "SandboxConfig",
    "SandboxViolation",
    "SandboxResult",
    "SandboxExecutor",
    "SANDBOX_EXECUTOR",
]
