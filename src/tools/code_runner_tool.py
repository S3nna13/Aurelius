from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field


@dataclass
class CodeRunnerConfig:
    timeout_s: float = 5.0
    max_output_bytes: int = 65536
    allowed_modules: list[str] = field(
        default_factory=lambda: [
            "math",
            "json",
            "re",
            "collections",
            "itertools",
            "functools",
            "string",
            "datetime",
        ]
    )


@dataclass(frozen=True)
class CodeRunnerResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False


class CodeRunnerTool:
    def __init__(self, config: CodeRunnerConfig | None = None) -> None:
        self.config = config or CodeRunnerConfig()

    def run(self, code: str) -> CodeRunnerResult:
        if not self.is_safe(code):
            return CodeRunnerResult(
                stdout="", stderr="code rejected: unsafe pattern detected", exit_code=1
            )
        try:
            proc = subprocess.run(  # noqa: S603
                [sys.executable, "-c", code],
                capture_output=True,
                timeout=self.config.timeout_s,
                text=True,
            )
            limit = self.config.max_output_bytes
            return CodeRunnerResult(
                stdout=proc.stdout[:limit],
                stderr=proc.stderr[:limit],
                exit_code=proc.returncode,
            )
        except subprocess.TimeoutExpired:
            return CodeRunnerResult(stdout="", stderr="timeout", exit_code=-1, timed_out=True)

    def is_safe(self, code: str) -> bool:
        forbidden = [
            "import os",
            "import sys",
            "subprocess",
            "__import__",
            "open(",
            "eval(",
            "exec(",
        ]
        return not any(tok in code for tok in forbidden)


CODE_RUNNER_REGISTRY: dict[str, type[CodeRunnerTool]] = {"default": CodeRunnerTool}
