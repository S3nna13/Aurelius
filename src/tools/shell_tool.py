"""Shell execution tool with deny-list and output capture."""
from __future__ import annotations

import subprocess

from .tool_registry import ToolResult, ToolSpec, TOOL_REGISTRY

SHELL_DENY_PATTERNS: frozenset[str] = frozenset([
    "rm -rf",
    ":(){ :|:& };:",
    "dd if=",
    "mkfs",
    "shutdown",
    "reboot",
    "chmod 777 /",
    "wget|curl.*|sh",
    "eval $(",
    "> /dev/sda",
])


class ShellTool:
    def __init__(
        self,
        timeout_seconds: int = 10,
        deny_patterns: frozenset[str] = SHELL_DENY_PATTERNS,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.deny_patterns = deny_patterns

    def is_denied(self, command: str) -> bool:
        """Check if any deny pattern is a substring of command."""
        return any(pattern in command for pattern in self.deny_patterns)

    def run(self, command: str) -> ToolResult:
        """Run a shell command and return ToolResult."""
        for pattern in self.deny_patterns:
            if pattern in command:
                return ToolResult(
                    tool_name="shell",
                    success=False,
                    output="",
                    error=f"command denied: {pattern}",
                )
        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            return ToolResult(
                tool_name="shell",
                success=proc.returncode == 0,
                output=proc.stdout[:2000],
                error=proc.stderr[:500],
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_name="shell",
                success=False,
                output="",
                error=f"command timed out after {self.timeout_seconds}s",
            )
        except Exception as e:
            return ToolResult(tool_name="shell", success=False, output="", error=str(e))

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="shell",
            description="Execute a shell command",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                },
            },
            required=["command"],
        )


SHELL_TOOL = ShellTool()
TOOL_REGISTRY.register(SHELL_TOOL.spec(), lambda command: SHELL_TOOL.run(command))
