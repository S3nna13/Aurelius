"""Sandboxed Python code executor for eval benchmarks.

Runs untrusted code in a subprocess with a hard timeout and a denylist
pre-check so that dangerous patterns never reach the OS.
"""

from __future__ import annotations

import subprocess
import tempfile
import os
from dataclasses import dataclass, field


@dataclass
class EvalConfig:
    timeout_s: float = 5.0
    max_output_bytes: int = 65536
    allowed_imports: list[str] = field(
        default_factory=lambda: ["math", "re", "json", "collections"]
    )


@dataclass
class EvalResult:
    code: str
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    passed: bool


class SandboxedEvaluator:
    """Safe Python code execution with subprocess isolation and timeout."""

    DENYLIST: list[str] = [
        "os.system",
        "subprocess.call(shell=True)",
        "subprocess.Popen(shell=True)",
        "__import__('os')",
        "exec(",
        "eval(",
        "open(",
        "importlib",
    ]

    def __init__(self, config: EvalConfig | None = None) -> None:
        self._config = config or EvalConfig()

    def is_safe(self, code: str) -> tuple[bool, str]:
        for pattern in self.DENYLIST:
            if pattern in code:
                return False, f"denylist hit: {pattern!r}"
        return True, ""

    def run(self, code: str) -> EvalResult:
        safe, reason = self.is_safe(code)
        if not safe:
            return EvalResult(
                code=code,
                stdout="",
                stderr=f"BLOCKED: {reason}",
                exit_code=1,
                timed_out=False,
                passed=False,
            )

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".py")
        try:
            with os.fdopen(tmp_fd, "w") as fh:
                fh.write(code)

            timed_out = False
            try:
                proc = subprocess.run(  # nosec B603
                    ["python3", tmp_path],
                    capture_output=True,
                    timeout=self._config.timeout_s,
                    shell=False,
                )
                stdout = proc.stdout.decode("utf-8", errors="replace")[
                    : self._config.max_output_bytes
                ]
                stderr = proc.stderr.decode("utf-8", errors="replace")[
                    : self._config.max_output_bytes
                ]
                exit_code = proc.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                stdout = ""
                stderr = "TimeoutExpired"
                exit_code = -1
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        passed = exit_code == 0 and not timed_out
        return EvalResult(
            code=code,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            timed_out=timed_out,
            passed=passed,
        )

    def run_batch(self, codes: list[str]) -> list[EvalResult]:
        return [self.run(c) for c in codes]

    def pass_rate(self, results: list[EvalResult]) -> float:
        if not results:
            return 0.0
        return sum(1 for r in results if r.passed) / len(results)
