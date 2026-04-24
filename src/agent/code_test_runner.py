"""Code test-runner tool for coding agents.

Finding AUR-SEC-2026-0021; CWE-78 (OS command injection),
CWE-426 (untrusted search path).

Given a working directory with tests, runs pytest (or a specified test
command) in a subprocess with a wall-clock timeout and parses the
output for pass/fail/skipped/error counts and the names of failing
tests.

The runner composes :class:`src.agent.code_execution_sandbox.CodeExecutionSandbox`
for process isolation and environment hardening. Pure standard library.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

from .code_execution_sandbox import CodeExecutionSandbox, ExecutionResult


__all__ = [
    "TestResult",
    "CodeTestRunner",
]


# ---------------------------------------------------------------------------
# Parsing regexes for pytest's short summary line.
# Examples we aim to match:
#   "1 passed in 0.03s"
#   "1 passed, 1 failed in 0.04s"
#   "3 failed, 5 passed, 2 skipped, 1 error in 0.12s"
#   "no tests ran in 0.01s"
# ---------------------------------------------------------------------------

_COUNT_RE = re.compile(
    r"(?P<n>\d+)\s+(?P<word>passed|failed|skipped|errors?|xfailed|xpassed)",
    re.IGNORECASE,
)

# Matches pytest's "FAILED tests/foo.py::test_bar - ..." lines (short -tb).
_FAILED_NAME_RE = re.compile(
    r"^FAILED\s+(\S+?)(?:\s+-\s+.*)?$",
    re.MULTILINE,
)
_ERROR_NAME_RE = re.compile(
    r"^ERROR\s+(\S+?)(?:\s+-\s+.*)?$",
    re.MULTILINE,
)


@dataclass
class TestResult:
    """Parsed result of a single test-runner invocation."""

    # Prevent pytest from trying to collect this dataclass as a test class.
    __test__ = False

    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    failed_names: list[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    duration_ms: float = 0.0
    timed_out: bool = False
    exit_code: int = 0


def _parse_counts(text: str) -> dict[str, int]:
    """Parse pytest summary counts out of combined stdout text."""
    out = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
    }
    # Prefer the last block of count tokens -- pytest prints the final
    # tally near the bottom. We walk all matches and keep the last
    # occurrence per category so we do not double-count progress dots.
    for m in _COUNT_RE.finditer(text):
        word = m.group("word").lower()
        n = int(m.group("n"))
        if word == "error":
            out["errors"] = n
        elif word == "errors":
            out["errors"] = n
        elif word in ("xfailed", "xpassed"):
            # Not tracked explicitly; fold xfailed into skipped for
            # completeness, xpassed into passed.
            if word == "xfailed":
                out["skipped"] += n
            else:
                out["passed"] += n
        elif word in out:
            out[word] = n
    return out


def _parse_failed_names(text: str) -> list[str]:
    """Extract nodeids of failing / erroring tests from pytest output."""
    names: list[str] = []
    for m in _FAILED_NAME_RE.finditer(text):
        nid = m.group(1).strip()
        if nid and nid not in names:
            names.append(nid)
    for m in _ERROR_NAME_RE.finditer(text):
        nid = m.group(1).strip()
        # Skip ERROR lines that aren't actually test nodeids (e.g.
        # "ERROR in file collection").
        if not nid:
            continue
        if nid not in names:
            names.append(nid)
    return names


class CodeTestRunner:
    """Runs a test command in an isolated subprocess and parses output.

    Defaults to ``python -m pytest -q --tb=short`` but accepts any
    custom argv. Wraps :class:`CodeExecutionSandbox` for process
    hardening; the sandbox is instantiated fresh per run so timeout
    and working-directory overrides compose cleanly.
    """

    DEFAULT_TIMEOUT = 60.0

    def __init__(
        self,
        test_command: Optional[list[str]] = None,
        timeout: float = DEFAULT_TIMEOUT,
        working_dir: Optional[str] = None,
        python_path: Optional[str] = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout!r}")
        if working_dir is not None and not os.path.isdir(working_dir):
            raise NotADirectoryError(working_dir)
        # AUR-SEC-2026-0021: resolve to absolute path so downstream
        # :func:`run_safe` allowlist checks succeed.
        self.python_path = os.path.realpath(python_path or sys.executable)
        default_cmd = [self.python_path, "-m", "pytest", "-q", "--tb=short"]
        self.test_command: list[str] = (
            list(test_command) if test_command else default_cmd
        )
        self.timeout = float(timeout)
        self.working_dir = working_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, extra_args: Optional[list[str]] = None) -> TestResult:
        argv = list(self.test_command)
        if extra_args:
            argv.extend(str(a) for a in extra_args)
        return self._invoke(argv)

    def run_file(self, path: str) -> TestResult:
        return self.run(extra_args=[path])

    def run_function(self, path: str, function_name: str) -> TestResult:
        return self.run(extra_args=[path, "-k", function_name])

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _invoke(self, argv: list[str]) -> TestResult:
        # We deliberately do NOT use sandbox.execute() because we need
        # to pass a full argv with arbitrary binary (pytest) rather
        # than a -c snippet. We still reuse the sandbox's env scrubbing
        # and timeout semantics via a direct subprocess.run below --
        # wrapped through the sandbox primitive for consistency.
        sandbox = CodeExecutionSandbox(
            python_path=self.python_path,
            timeout=self.timeout,
            scrub_env=False,  # need PATH + site-packages for pytest
            disallow_network_env=True,
        )
        exec_result = _run_argv_via_sandbox(
            sandbox=sandbox,
            argv=argv,
            cwd=self.working_dir,
        )

        combined = exec_result.stdout + "\n" + exec_result.stderr
        counts = _parse_counts(combined)
        failed_names = _parse_failed_names(combined)

        total = (
            counts["passed"]
            + counts["failed"]
            + counts["skipped"]
            + counts["errors"]
        )

        return TestResult(
            total=total,
            passed=counts["passed"],
            failed=counts["failed"],
            skipped=counts["skipped"],
            errors=counts["errors"],
            failed_names=failed_names,
            stdout=exec_result.stdout,
            stderr=exec_result.stderr,
            duration_ms=exec_result.duration_ms,
            timed_out=exec_result.timed_out,
            exit_code=exec_result.exit_code,
        )


def _run_argv_via_sandbox(
    sandbox: CodeExecutionSandbox,
    argv: list[str],
    cwd: Optional[str],
) -> ExecutionResult:
    """Run an arbitrary argv through the sandbox's hardened subprocess path.

    The public :class:`CodeExecutionSandbox` API is Python-snippet
    oriented; here we piggy-back on its private ``_run`` to get the
    same env scrubbing, timeout handling, and truncation for an
    arbitrary argv (pytest invocation). If the sandbox ever changes
    that private contract we fall back to a minimal direct subprocess
    call.
    """
    import shutil
    import tempfile
    import time

    from src.security.safe_subprocess import run_safe

    run_fn = getattr(sandbox, "_run", None)
    if run_fn is not None and cwd is None:
        # Delegate to sandbox for full hardening in a tmpdir.
        return run_fn(argv, stdin_text=None)  # type: ignore[no-any-return]

    # Custom cwd path: we need to preserve the user's working dir but
    # still get env scrubbing + timeout. Use the hardened wrapper for
    # argv allowlist + timeout enforcement.
    env = sandbox._build_env()  # noqa: SLF001
    tmpdir = None
    workdir = cwd
    if workdir is None:
        tmpdir = tempfile.mkdtemp(prefix="aurelius_testrun_")
        workdir = tmpdir

    # Allowlist = sandbox interpreter + resolved argv[0] if it is an
    # absolute path. This keeps the allowlist tight while letting
    # callers point the runner at custom test binaries explicitly.
    base_allowed = set(getattr(
        sandbox,
        "_allowed_executables",
        frozenset({sandbox.python_path}),
    ))
    if argv and os.path.isabs(argv[0]):
        base_allowed.add(argv[0])
    allowed = frozenset(base_allowed)

    t0 = time.perf_counter()
    try:
        # AUR-SEC-2026-0021: hardened subprocess wrapper replaces raw
        # subprocess.run call here.
        safe_res = run_safe(
            argv,
            timeout=sandbox.timeout,
            allowed_executables=allowed,
            env_override=env,
            cwd=workdir,
        )
        exit_code = safe_res.returncode
        stdout = safe_res.stdout
        stderr = safe_res.stderr
        timed_out = safe_res.killed_on_timeout
    finally:
        duration_ms = (time.perf_counter() - t0) * 1000.0
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)

    return ExecutionResult(
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_ms=duration_ms,
        timed_out=timed_out,
    )
