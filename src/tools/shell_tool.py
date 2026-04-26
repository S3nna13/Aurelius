"""Shell execution tool with allow-list and output capture.

Security note — hardened subprocess execution:
    ShellTool uses ``shell=False`` with ``shlex.split()`` and an explicit
    **allow-list** of safe commands.  Pipelines, redirections, and subshells
    are rejected.  Changed after security review AUR-SEC-2026-0028 / CWE-78.
"""
from __future__ import annotations

import shlex
import subprocess

from .tool_registry import ToolResult, ToolSpec, TOOL_REGISTRY

# Commands that are allowed to run.  Arguments are passed as a list (no shell
# interpolation), so glob expansion and variable substitution do not occur.
SHELL_ALLOWLIST: frozenset[str] = frozenset([
    # File inspection
    "ls", "cat", "head", "tail", "less", "more", "file", "wc",
    # Search
    "grep", "rg", "find",
    # VCS
    "git",
    # Python tooling
    "python", "python3", "pytest", "pip", "uv",
    # Build
    "make", "cmake",
    # System info
    "uname", "df", "du", "ps", "top", "htop", "free", "uptime",
    # Networking (read-only)
    "ping", "curl", "wget",
    # Text processing
    "sed", "awk", "cut", "sort", "uniq", "xargs",
    # Compression
    "tar", "gzip", "gunzip", "zip", "unzip",
    # Misc
    "echo", "printf", "date", "time", "which", "whoami", "id",
])

# Characters that indicate shell metacharacters (pipelines, redirections,
# subshells, variable expansion, etc.).  Backslash is intentionally omitted
# because ``shlex.split()`` handles escapes safely under ``shell=False``.
_SHELL_META_CHARS: frozenset[str] = frozenset("|&;<>$`\"\n\r")


class ShellTool:
    def __init__(
        self,
        timeout_seconds: int = 10,
        allowlist: frozenset[str] = SHELL_ALLOWLIST,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.allowlist = allowlist

    def _validate(self, command: str) -> tuple[list[str] | None, str]:
        """Parse and validate *command*.  Returns (argv, error_msg).

        *error_msg* is empty on success.  On failure *argv* is None.
        """
        if not command or not isinstance(command, str):
            return None, "command must be a non-empty string"

        # Reject obvious shell metacharacters before parsing
        for ch in command:
            if ch in _SHELL_META_CHARS:
                return None, (
                    f"command contains disallowed shell metacharacter: {ch!r}. "
                    "Pipelines, redirections, subshells, and variable expansion are not permitted."
                )

        try:
            argv = shlex.split(command)
        except ValueError as exc:
            return None, f"invalid shell command: {exc}"

        if not argv:
            return None, "command parsed to empty argument list"

        # Resolve simple path prefixes like /bin/ls -> ls
        base_cmd = argv[0]
        if "/" in base_cmd:
            base_cmd = base_cmd.split("/")[-1]

        if base_cmd not in self.allowlist:
            return None, (
                f"command {base_cmd!r} is not in the allowlist. "
                f"Allowed commands: {', '.join(sorted(self.allowlist))}"
            )

        return argv, ""

    def run(self, command: str) -> ToolResult:
        """Run a shell command and return ToolResult."""
        argv, error = self._validate(command)
        if argv is None:
            return ToolResult(
                tool_name="shell",
                success=False,
                output="",
                error=error,
            )
        try:
            proc = subprocess.run(
                argv,
                shell=False,
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
            description="Execute a shell command (allow-list only, no shell metacharacters)",
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
