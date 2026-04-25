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

import contextlib
import io
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

# Patterns that are unconditionally blocked
_BLOCKED_PATTERNS: list[tuple[str, str]] = [
    (r"\bimport\s+os\b", "import of 'os' is not allowed"),
    (r"\bimport\s+sys\b", "import of 'sys' is not allowed"),
    (r"\bsubprocess\b", "use of 'subprocess' is not allowed"),
    (r"\b__import__\s*\(", "use of '__import__' is not allowed"),
    (r"\bexec\s*\(", "use of 'exec' is not allowed"),
    (r"\beval\s*\(", "use of 'eval' is not allowed"),
    (r"\bopen\s*\(", "use of 'open' is not allowed"),
    (r"\bfile\s*\(", "use of 'file' is not allowed"),
    (r"\bsocket\b", "use of 'socket' is not allowed"),
]


def sanitize_code(code: str, allowed_modules: list[str]) -> tuple[bool, str]:
    """Check code for dangerous patterns.

    Returns (is_safe, reason). reason is "" when safe, else explains the block.
    """
    # Check unconditionally blocked patterns
    for pattern, reason in _BLOCKED_PATTERNS:
        if re.search(pattern, code):
            return False, reason

    # Check all import statements against the allowed list
    import_pattern = re.compile(
        r"^\s*(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.MULTILINE
    )
    for m in import_pattern.finditer(code):
        module = m.group(1)
        if module not in allowed_modules:
            return False, f"import of '{module}' is not in the allowed modules list"

    return True, ""


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

import builtins as _builtins_mod

_SAFE_BUILTINS = {
    name: getattr(_builtins_mod, name)
    for name in dir(_builtins_mod)
    if name not in ("open", "exec", "eval", "__import__", "compile", "input",
                    "breakpoint", "__loader__", "__spec__")
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
                # nosec B102 -- sandboxed run; input validated via sanitize_code() and _MAX_EXEC_LEN; globals come from _make_restricted_globals().
                exec(compiled, restricted_globals)  # noqa: S102  # nosec B102
        except Exception as exc:
            exc_holder.append(exc)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=config.timeout_seconds)

    elapsed = time.monotonic() - start

    if thread.is_alive():
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
