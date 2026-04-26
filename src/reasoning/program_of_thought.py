"""Program-of-Thought: Chen et al. 2022 'Program of Thoughts Prompting'."""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class PoTConfig:
    language: str = "python"
    timeout_s: float = 5.0
    max_output_len: int = 1024


@dataclass(frozen=True)
class PoTResult:
    code: str
    output: str
    answer: str
    success: bool


class ProgramOfThought:
    def __init__(self, config: PoTConfig | None = None) -> None:
        self.config = config or PoTConfig()

    def extract_code(self, text: str) -> str:
        m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        first = text.strip().splitlines()[0] if text.strip() else ""
        if any(first.startswith(kw) for kw in ("def ", "import ", "print(", "answer")):
            return text.strip()
        return text

    def execute_code(self, code: str) -> tuple[str, bool]:
        try:
            result = subprocess.run(  # noqa: S603
                [sys.executable, "-c", code],
                capture_output=True,
                timeout=self.config.timeout_s,
                text=True,
            )
            if result.returncode == 0:
                return result.stdout[: self.config.max_output_len], True
            return result.stderr[: self.config.max_output_len], False
        except subprocess.TimeoutExpired:
            return "timeout", False

    def extract_answer_from_output(self, output: str) -> str:
        for line in output.splitlines():
            if "answer" in line.lower() and "=" in line:
                parts = line.split("=", 1)
                if len(parts) == 2:
                    candidate = parts[1].strip()
                    if candidate:
                        return candidate
        nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", output)
        if nums:
            return nums[-1]
        return output.strip()

    def run(self, question: str, generate_fn: Callable[[str], str]) -> PoTResult:
        text = generate_fn(question)
        code = self.extract_code(text)
        output, success = self.execute_code(code)
        answer = self.extract_answer_from_output(output) if success else output
        return PoTResult(code=code, output=output, answer=answer, success=success)


POT_REGISTRY: dict[str, type[ProgramOfThought]] = {"default": ProgramOfThought}
