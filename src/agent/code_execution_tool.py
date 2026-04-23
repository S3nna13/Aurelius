"""Code execution tool: Inspired by Gemini 2.5 code execution tool (Google DeepMind 2025).

Gemini-style sandboxed code execution for the Aurelius agent surface.
Runs short Python (or stub) snippets in a subprocess with deny-pattern
pre-validation. Pure standard library.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "ExecutionLanguage",
    "ExecutionRequest",
    "ExecutionResult",
    "CodeExecutionTool",
    "CODE_EXECUTION_TOOL_REGISTRY",
]


# ---------------------------------------------------------------------------
# Enums and dataclasses
# ---------------------------------------------------------------------------


class ExecutionLanguage(str, Enum):
    """Languages supported by :class:`CodeExecutionTool`."""

    PYTHON = "python"
    BASH = "bash"
    JAVASCRIPT = "javascript"


@dataclass
class ExecutionRequest:
    """Parameters for a single code-execution call."""

    code: str
    language: ExecutionLanguage = ExecutionLanguage.PYTHON
    timeout_s: float = 10.0
    env_vars: dict[str, str] = field(default_factory=dict)
    capture_output: bool = True


@dataclass
class ExecutionResult:
    """Result returned by :class:`CodeExecutionTool.execute`."""

    stdout: str
    stderr: str
    exit_code: int
    duration_ms: float
    timed_out: bool = False
    error: str | None = None


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------


class CodeExecutionTool:
    """Gemini-style code execution tool with deny-pattern validation.

    Validates the requested code against :attr:`DENY_PATTERNS` (substring
    checks for speed), then runs it via the system Python interpreter with
    a wall-clock timeout.  Every exception is caught and surfaced through
    :attr:`ExecutionResult.error` so callers always get a structured result.
    """

    DENY_PATTERNS: frozenset[str] = frozenset(
        {
            "import os",
            "import sys",
            "subprocess",
            "__import__",
            "exec(",
            "eval(",
            "open(",
            "rmdir",
            "shutil",
            "socket",
        }
    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, req: ExecutionRequest) -> list[str]:
        """Return a list of deny-pattern violations found in *req.code*.

        An empty list means the code passed all checks.
        Checks are plain substring membership (not regex) for speed.
        """
        violations: list[str] = []
        for pattern in self.DENY_PATTERNS:
            if pattern in req.code:
                violations.append(pattern)
        return violations

    def execute(self, req: ExecutionRequest) -> ExecutionResult:
        """Execute *req* and return an :class:`ExecutionResult`.

        Decision tree:
        1. JavaScript language -> immediately rejected (stub not implemented).
        2. validate() returns violations -> blocked, exit_code=1.
        3. Otherwise -> run with python3 -c <code>, catch all exceptions.
        """
        # Stub: JavaScript not yet supported
        if req.language is ExecutionLanguage.JAVASCRIPT:
            return ExecutionResult(
                stdout="",
                stderr="JavaScript execution not yet supported",
                exit_code=1,
                duration_ms=0.0,
            )

        # Deny-pattern check
        violations = self.validate(req)
        if violations:
            return ExecutionResult(
                stdout="",
                stderr=f"Blocked: {violations}",
                exit_code=1,
                duration_ms=0.0,
            )

        # Run in a child process
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(  # noqa: S603
                ["python3", "-c", req.code],
                capture_output=req.capture_output,
                text=True,
                timeout=req.timeout_s,
                env=req.env_vars if req.env_vars else None,
            )
            duration_ms = (time.perf_counter() - t0) * 1000.0
            return ExecutionResult(
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                exit_code=proc.returncode,
                duration_ms=duration_ms,
            )
        except subprocess.TimeoutExpired:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            return ExecutionResult(
                stdout="",
                stderr="",
                exit_code=1,
                duration_ms=duration_ms,
                timed_out=True,
            )
        except Exception as exc:  # noqa: BLE001
            duration_ms = (time.perf_counter() - t0) * 1000.0
            return ExecutionResult(
                stdout="",
                stderr="",
                exit_code=1,
                duration_ms=duration_ms,
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# Module-level registries
# ---------------------------------------------------------------------------

#: Registry of CodeExecutionTool implementations keyed by variant name.
CODE_EXECUTION_TOOL_REGISTRY: dict[str, type[CodeExecutionTool]] = {
    "default": CodeExecutionTool,
}

# ---------------------------------------------------------------------------
# Wire into TOOL_REGISTRY from src.agent
# ---------------------------------------------------------------------------
# Deferred import to avoid circular dependency; guarded so the module is
# usable in isolation (e.g., direct unit tests before package init).
try:
    from src.agent import TOOL_REGISTRY as _TOOL_REGISTRY  # type: ignore[attr-defined]

    _TOOL_REGISTRY.setdefault(
        "code_execution",
        {
            "name": "code_execution",
            "description": (
                "Execute a Python code snippet in a sandboxed subprocess "
                "and return stdout, stderr, and exit code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Source code to run.",
                    },
                    "language": {
                        "type": "string",
                        "enum": ["python", "bash", "javascript"],
                        "default": "python",
                    },
                    "timeout_s": {
                        "type": "number",
                        "default": 10.0,
                        "description": "Wall-clock timeout in seconds.",
                    },
                },
                "required": ["code"],
            },
        },
    )
except Exception:  # noqa: BLE001
    pass
