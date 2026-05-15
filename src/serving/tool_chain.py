"""Multi-step tool chain execution for Aurelius."""

import json
import re
from dataclasses import dataclass

from src.serving.tool_executor import ToolExecutor

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


@dataclass
class ChainStep:
    tool_name: str
    args: dict
    result: str | None = None
    success: bool = False


class ToolChain:
    """Parse, execute, and format sequences of tool calls embedded in text."""

    def __init__(self, executor: ToolExecutor) -> None:
        self._executor = executor

    def parse_chain(self, text: str) -> list[ChainStep]:
        """Extract every ``<tool_call>…</tool_call>`` block and return ChainSteps."""
        steps: list[ChainStep] = []
        for match in _TOOL_CALL_RE.finditer(text):
            raw = match.group(1).strip()
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue
            tool_name = parsed.get("name", "")
            args = parsed.get("args", {})
            steps.append(ChainStep(tool_name=tool_name, args=args))
        return steps

    def execute_chain(self, steps: list[ChainStep]) -> list[ChainStep]:
        """Execute each step in order, setting *result* and *success* in place."""
        for step in steps:
            tool_result = self._executor.execute(step.tool_name, step.args)
            step.result = tool_result.output if tool_result.success else tool_result.error
            step.success = tool_result.success
        return steps

    def format_results(self, steps: list[ChainStep]) -> str:
        """Return a multi-line string with one ``[tool_name]: result`` line per step."""
        lines = []
        for step in steps:
            result_text = step.result if step.result is not None else ""
            lines.append(f"[{step.tool_name}]: {result_text}")
        return "\n".join(lines) + ("\n" if lines else "")

    def run(self, text: str) -> tuple[list[ChainStep], str]:
        """Parse, execute, and format tool calls found in *text*."""
        steps = self.parse_chain(text)
        steps = self.execute_chain(steps)
        formatted = self.format_results(steps)
        return steps, formatted
