"""Code execution tool: Inspired by Gemini 2.5 code execution tool (Google DeepMind 2025).

Gemini-style sandboxed code execution for the Aurelius agent surface.
Runs short Python (or stub) snippets in a subprocess with AST-based
pre-validation.  Pure standard library.
"""

from __future__ import annotations

import ast
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import StrEnum

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


class ExecutionLanguage(StrEnum):
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
# AST-based validation
# ---------------------------------------------------------------------------

# Modules that must not be imported / used in any form.
_FORBIDDEN_MODULES: frozenset[str] = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "socket",
        "pty",
        "pickle",
        "marshal",
        "ctypes",
        "cffi",
        "builtins",
        "importlib",
        "imp",
        "runpy",
        "code",
        "codeop",
        "tempfile",
        "pathlib",
        "urllib",
        "http",
        "ftplib",
        "smtplib",
        "poplib",
        "imaplib",
        "nntplib",
        "telnetlib",
        "webbrowser",
        "xmlrpc",
        "concurrent",
        "multiprocessing",
        "threading",
        "asyncio",
        "sqlite3",
        "dbm",
        "shelve",
    }
)

# Built-in callables that must not be invoked.
_FORBIDDEN_BUILTINS: frozenset[str] = frozenset(
    {
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
        "delattr",
        "vars",
        "globals",
        "locals",
        "dir",
        "hasattr",
    }
)


def _validate_ast(code: str) -> list[str]:
    """Return violations found by parsing *code* with the ``ast`` module.

    Checks for:
    - imports of forbidden modules
    - calls to forbidden builtins
    - ``__import__``, ``eval``, ``exec``, ``compile`` anywhere
    """
    violations: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in _FORBIDDEN_MODULES:
                    violations.append(f"forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top in _FORBIDDEN_MODULES:
                violations.append(f"forbidden import from: {node.module}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_BUILTINS:
                violations.append(f"forbidden builtin call: {node.func.id}()")
            elif isinstance(node.func, ast.Attribute) and node.func.attr in _FORBIDDEN_BUILTINS:
                violations.append(f"forbidden attribute call: {node.func.attr}()")

    return violations


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------


class CodeExecutionTool:
    """Gemini-style code execution tool with AST-based validation.

    Validates the requested code by parsing it with the ``ast`` module,
    then runs it via the system Python interpreter with a wall-clock
    timeout.  Every exception is caught and surfaced through
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
        """
        return _validate_ast(req.code)

    def execute(self, req: ExecutionRequest) -> ExecutionResult:
        """Execute *req* and return an :class:`ExecutionResult`.

        Decision tree:
        1. BASH / JavaScript language -> immediately rejected (not implemented).
        2. validate() returns violations -> blocked, exit_code=1.
        3. Otherwise -> run with python3 -c <code>, catch all exceptions.
        """
        # Stub: BASH and JavaScript not yet supported
        if req.language in (ExecutionLanguage.BASH, ExecutionLanguage.JAVASCRIPT):
            return ExecutionResult(
                stdout="",
                stderr=f"{req.language.value} execution not yet supported",
                exit_code=1,
                duration_ms=0.0,
            )

        # Deny-pattern check (AST-based)
        violations = self.validate(req)
        if violations:
            return ExecutionResult(
                stdout="",
                stderr=f"Blocked: {violations}",
                exit_code=1,
                duration_ms=0.0,
            )

        # Build a clean environment — remove known-dangerous variables and
        # reject attacker-controlled overrides of sensitive keys.
        env = self._build_env(req.env_vars)

        # Run in a child process
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(  # noqa: S603
                [sys.executable, "-c", req.code],
                capture_output=req.capture_output,
                text=True,
                timeout=req.timeout_s,
                env=env,
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

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @staticmethod
    def _build_env(user_env: dict[str, str] | None) -> dict[str, str] | None:
        """Return a scrubbed environment dict.

        If *user_env* is empty/None, inherit the current environment after
        stripping known-dangerous keys.  If *user_env* is non-empty, start
        from the current environment, scrub it, then layer the user-provided
        values on top — but reject any attempt to set a dangerous key.
        """
        dangerous_keys = frozenset(
            {
                "LD_PRELOAD",
                "LD_LIBRARY_PATH",
                "LD_AUDIT",
                "LD_PROFILE",
                "PATH",
                "HOME",
                "SHELL",
                "PYTHONPATH",
                "PYTHONHOME",
                "PYTHONSTARTUP",
                "PYTHONINSPECT",
                "PYTHONIOENCODING",
                "PYTHONDONTWRITEBYTECODE",
                "PYTHONOPTIMIZE",
                "PYTHONNOUSERSITE",
                "BROWSER",
                "EDITOR",
                "PAGER",
                "TERM",
                "TERMCAP",
                "DISPLAY",
                "XAUTHORITY",
                "XDG_CONFIG_DIRS",
                "XDG_DATA_DIRS",
            }
        )

        if not user_env:
            # No user overrides — just scrub the inherited environment
            base = dict(os.environ)
            for key in dangerous_keys:
                base.pop(key, None)
            return base

        # User provided overrides — scrub inherited env, then apply allowed
        # user values (rejecting dangerous keys outright).
        base = dict(os.environ)
        for key in dangerous_keys:
            base.pop(key, None)

        for key, value in user_env.items():
            if key.upper() in dangerous_keys:
                # Reject rather than silently ignore — caller gets an error
                raise ValueError(f"env_vars key {key!r} is not allowed for security reasons")
            base[key] = value

        return base


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
    logging.getLogger(__name__).debug(
        "TOOL_REGISTRY wiring failed (expected when imported in isolation)",
        exc_info=True,
    )
