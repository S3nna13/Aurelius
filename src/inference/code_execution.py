"""Safe code generation, extraction, and sandboxed execution utilities.

Security note
-------------
This module intentionally runs model-generated Python in a restricted
namespace (``_SAFE_BUILTINS``) under a threaded timeout. The sandboxed-run
sites below are annotated with ``# nosec B102``. Contract for callers:

  * Input MUST pass ``sanitize_code()`` (enforced at both run sites).
  * Input MUST satisfy the ``_MAX_EXEC_LEN`` length guard.
  * The ``globals`` dict passed in MUST originate from
    ``_make_restricted_globals()`` — never pass real module globals.

Violating these conditions may allow arbitrary-code execution in the host
process.
"""

from __future__ import annotations

import ast
import contextlib
import io
import multiprocessing
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

# Maximum length of a code string accepted for sandboxed evaluation.
_MAX_EXEC_LEN = 100_000


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ExecutionConfig:
    """Configuration for sandboxed code execution."""

    timeout_seconds: float = 5.0
    max_output_len: int = 1000
    allowed_modules: list[str] = field(
        default_factory=lambda: [
            "math",
            "re",
            "json",
            "collections",
            "itertools",
            "functools",
            "string",
            "datetime",
        ]
    )
    capture_exceptions: bool = True


@dataclass
class ExecutionResult:
    """Result of a sandboxed code execution."""

    code: str
    stdout: str
    stderr: str
    return_value: str
    success: bool
    execution_time: float
    error: str | None = None


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------


def extract_code_blocks(text: str, language: str = "python") -> list[str]:
    """Extract code from markdown fenced code blocks and <code> tags.

    Handles:
    - ```python ... ```  (language-labeled)
    - ``` ... ```        (bare, no language tag)
    - <code>...</code>
    """
    blocks: list[str] = []

    # Single-pass: match any fenced block, capturing the optional language tag
    # Group 1 = language tag (may be empty), Group 2 = body
    pattern_fence = re.compile(
        r"```([a-zA-Z0-9_+-]*)\s*\n(.*?)```",
        re.DOTALL,
    )
    for m in pattern_fence.finditer(text):
        tag = m.group(1).strip().lower()
        body = m.group(2).strip()
        target = language.lower()
        # Include if tag matches the requested language OR tag is empty (bare fence)
        if tag == target or tag == "":
            if body not in blocks:
                blocks.append(body)

    # <code>...</code> tags
    pattern_code_tag = re.compile(r"<code>(.*?)</code>", re.DOTALL)
    for m in pattern_code_tag.finditer(text):
        content = m.group(1).strip()
        if content not in blocks:
            blocks.append(content)

    return blocks


# ---------------------------------------------------------------------------
# Sanitization
# ---------------------------------------------------------------------------

_BLOCKED_ATTRS = frozenset({
    "__class__", "__bases__", "__subclasses__", "__builtins__",
    "__globals__", "__code__", "__closure__", "__dict__",
    "__mro__", "__import__",
})


class _AstChecker(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            self.violations.append(f"import of '{root}' is not allowed")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            root = node.module.split(".")[0]
            self.violations.append(f"import from '{root}' is not allowed")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in _BLOCKED_ATTRS:
            self.violations.append(f"access to '{node.attr}' is not allowed")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in ("exec", "eval", "compile", "open", "__import__"):
            self.violations.append(f"use of '{node.func.id}()' is not allowed")
        if isinstance(node.func, ast.Attribute) and node.func.attr in ("__import__",):
            self.violations.append(f"use of '{node.func.attr}()' is not allowed")
        self.generic_visit(node)


def sanitize_code(code: str, allowed_modules: list[str]) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    checker = _AstChecker()
    checker.visit(tree)

    allowed_roots = set(allowed_modules)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in allowed_roots:
                    return False, f"import of '{root}' is not in the allowed modules list"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root not in allowed_roots:
                    return False, f"import from '{root}' is not in the allowed modules list"

    if checker.violations:
        return False, checker.violations[0]

    return True, ""


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

import builtins as _builtins_mod  # noqa: E402

_SAFE_BUILTINS = {
    name: getattr(_builtins_mod, name)
    for name in dir(_builtins_mod)
    if name
    not in (
        "open",
        "exec",
        "eval",
        "__import__",
        "compile",
        "input",
        "breakpoint",
        "__loader__",
        "__spec__",
    )
    and not name.startswith("__")
}
# Re-add essentials that start with __ so class definitions and name lookups work
_SAFE_BUILTINS["__build_class__"] = _builtins_mod.__build_class__
_SAFE_BUILTINS["__name__"] = "__sandbox__"


def _make_restricted_globals(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a restricted globals dict for exec."""
    g: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
    if extra:
        g.update(extra)
    return g


def execute_python(code: str, config: ExecutionConfig) -> ExecutionResult:
    """Safely execute Python code in a restricted namespace."""
    start = time.monotonic()

    # Size guard: reject empty or pathologically large inputs before sanitize/compile.
    if not code or len(code) > _MAX_EXEC_LEN:
        elapsed = time.monotonic() - start
        return ExecutionResult(
            code=code,
            stdout="",
            stderr="",
            return_value="",
            success=False,
            execution_time=elapsed,
            error=(
                "empty code"
                if not code
                else f"code too large ({len(code)} chars > {_MAX_EXEC_LEN})"
            ),
        )

    # Sanitize first
    is_safe, reason = sanitize_code(code, config.allowed_modules)
    if not is_safe:
        elapsed = time.monotonic() - start
        return ExecutionResult(
            code=code,
            stdout="",
            stderr="",
            return_value="",
            success=False,
            execution_time=elapsed,
            error=f"Sanitization failed: {reason}",
        )

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    exc_holder: list[Exception] = []

    restricted_globals = _make_restricted_globals()

    def _run() -> None:
        try:
            compiled = compile(code, "<sandbox>", "exec")
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                exec(compiled, restricted_globals)  # noqa: S102
        except Exception as exc:
            exc_holder.append(exc)

    proc = multiprocessing.Process(target=_run, daemon=True)
    proc.start()
    proc.join(timeout=config.timeout_seconds)

    elapsed = time.monotonic() - start

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=1.0)
        return ExecutionResult(
            code=code,
            stdout=stdout_buf.getvalue()[: config.max_output_len],
            stderr=stderr_buf.getvalue()[: config.max_output_len],
            return_value="",
            success=False,
            execution_time=elapsed,
            error=f"Execution timed out after {config.timeout_seconds}s",
        )

    raw_stdout = stdout_buf.getvalue()[: config.max_output_len]
    raw_stderr = stderr_buf.getvalue()[: config.max_output_len]

    if exc_holder:
        exc = exc_holder[0]
        return ExecutionResult(
            code=code,
            stdout=raw_stdout,
            stderr=raw_stderr,
            return_value="",
            success=False,
            execution_time=elapsed,
            error=f"{type(exc).__name__}: {exc}",
        )

    return ExecutionResult(
        code=code,
        stdout=raw_stdout,
        stderr=raw_stderr,
        return_value="",
        success=True,
        execution_time=elapsed,
        error=None,
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_code_result(result: ExecutionResult) -> str:
    """Format an ExecutionResult for injection back into LLM context."""
    if result.success:
        return f"Code executed successfully:\n{result.stdout}"
    return f"Code error: {result.error}\n{result.stderr}"


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class CodeInterpreterSession:
    """Maintains state across multiple code executions via a persistent namespace."""

    def __init__(self, config: ExecutionConfig) -> None:
        self._config = config
        self._namespace: dict[str, Any] = _make_restricted_globals()
        self._history: list[ExecutionResult] = []

    def execute(self, code: str) -> ExecutionResult:
        """Sanitize and execute code in the persistent namespace."""
        start = time.monotonic()

        if not code or len(code) > _MAX_EXEC_LEN:
            elapsed = time.monotonic() - start
            result = ExecutionResult(
                code=code,
                stdout="",
                stderr="",
                return_value="",
                success=False,
                execution_time=elapsed,
                error=(
                    "empty code"
                    if not code
                    else f"code too large ({len(code)} chars > {_MAX_EXEC_LEN})"
                ),
            )
            self._history.append(result)
            return result

        is_safe, reason = sanitize_code(code, self._config.allowed_modules)
        if not is_safe:
            elapsed = time.monotonic() - start
            result = ExecutionResult(
                code=code,
                stdout="",
                stderr="",
                return_value="",
                success=False,
                execution_time=elapsed,
                error=f"Sanitization failed: {reason}",
            )
            self._history.append(result)
            return result

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        exc_holder: list[Exception] = []
        namespace = self._namespace

        def _run() -> None:
            try:
                compiled = compile(code, "<session>", "exec")
                with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                    # nosec B102 -- sandboxed run; input validated via sanitize_code() and _MAX_EXEC_LEN; globals come from _make_restricted_globals().
                    exec(compiled, namespace)  # noqa: S102  # nosec B102
            except Exception as exc:
                exc_holder.append(exc)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=self._config.timeout_seconds)

        elapsed = time.monotonic() - start
        raw_stdout = stdout_buf.getvalue()[: self._config.max_output_len]
        raw_stderr = stderr_buf.getvalue()[: self._config.max_output_len]

        if thread.is_alive():
            result = ExecutionResult(
                code=code,
                stdout=raw_stdout,
                stderr=raw_stderr,
                return_value="",
                success=False,
                execution_time=elapsed,
                error=f"Execution timed out after {self._config.timeout_seconds}s",
            )
        elif exc_holder:
            exc = exc_holder[0]
            result = ExecutionResult(
                code=code,
                stdout=raw_stdout,
                stderr=raw_stderr,
                return_value="",
                success=False,
                execution_time=elapsed,
                error=f"{type(exc).__name__}: {exc}",
            )
        else:
            result = ExecutionResult(
                code=code,
                stdout=raw_stdout,
                stderr=raw_stderr,
                return_value="",
                success=True,
                execution_time=elapsed,
                error=None,
            )

        self._history.append(result)
        return result

    def reset(self) -> None:
        """Clear the persistent namespace and execution history."""
        self._namespace = _make_restricted_globals()
        self._history = []

    def history(self) -> list[ExecutionResult]:
        """Return the list of all past ExecutionResults."""
        return list(self._history)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class CodeGenerationEvaluator:
    """Evaluate LLM-generated code against a suite of test cases."""

    def __init__(self, config: ExecutionConfig) -> None:
        self._config = config

    def evaluate(
        self,
        code: str,
        test_cases: list[tuple[str, str]],
    ) -> dict[str, float]:
        """Run test_cases against code and return pass metrics.

        Each test case is (input_code_to_run, expected_output).
        The executed snippet is: code + newline + input_code.
        A test passes when the stripped stdout equals the stripped expected output.
        """
        n_passed = 0
        n_total = len(test_cases)

        for input_code, expected_output in test_cases:
            full_code = code + "\n" + input_code
            result = execute_python(full_code, self._config)
            if result.success and result.stdout.strip() == expected_output.strip():
                n_passed += 1

        pass_rate = n_passed / n_total if n_total > 0 else 0.0
        return {
            "pass_rate": pass_rate,
            "n_passed": n_passed,
            "n_total": n_total,
        }
