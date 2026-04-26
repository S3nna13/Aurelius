"""Shell execution tool with allow-list and deny-pattern validation.

Safety is enforced through:
  1. shlex parsing (no shell interpretation of metacharacters)
  2. A base-command allow-list
  3. Regex deny patterns matched against the full command string
  4. A hard timeout

Pipelines, command substitution, and chaining are rejected because
``shell=False`` cannot safely support them.
"""

from __future__ import annotations

import re
import shlex
import subprocess

from .tool_registry import TOOL_REGISTRY, ToolResult, ToolSpec

#: Backward-compatible deny patterns (substring-based).  These are exposed for
#: tests and introspection but the runtime enforcement uses the stricter regex
#: and allow-list checks below.
SHELL_DENY_PATTERNS: frozenset[str] = frozenset({
    "rm -rf",
    ":(){ :|:& };:",
    "dd if=",
    "mkfs",
    "shutdown",
    "reboot",
    "chmod 777 /",
    "eval $(",
    "> /dev/sda",
})

#: Base commands considered generally safe (read-only or project-local).
_ALLOWLIST: frozenset[str] = frozenset({
    "ls", "pwd", "echo", "cat", "head", "tail", "wc", "sort", "uniq",
    "cut", "tr", "diff", "git", "grep", "rg", "find", "fd", "uv",
    "ruff", "mypy", "black", "isort", "which", "whoami", "date",
    "true", "false", "stat", "file", "tree", "mkdir", "touch",
    "cp", "mv", "rm", "ln", "chmod", "chown", "sleep", "yes",
    "seq", "printf",
})

#: Commands that are frequently abused for arbitrary code execution.
#: Any command whose base name matches one of these is rejected.
_DENYLIST: frozenset[str] = frozenset({
    "dd", "mkfs", "mkswap", "fdisk", "parted", "shred", "wipefs",
    "shutdown", "reboot", "halt", "poweroff", "kexec", "umount",
    "chattr", "python", "python3", "pip", "pip3", "node", "npm",
    "npx", "yarn", "pnpm", "make", "cmake", "ninja", "env",
    "bash", "sh", "zsh", "fish", "dash", "ksh", "csh",
    "wget", "curl", "nc", "netcat", "telnet", "ssh", "scp",
    "su", "sudo", "doas", "pkexec",
})

#: Regex patterns matched against the *full* command string. Any hit
#: marks the command as forbidden.
_DENY_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*r|-rf|-fr)"
            r"\s+(/|~|\$HOME|/\*)"
        ),
        "rm -rf against root or home",
    ),
    (re.compile(r"\bsudo\b"), "sudo escalation"),
    (re.compile(r"\bchmod\s+-?R?\s*777\b"), "chmod 777"),
    (re.compile(r"\bchmod\s+-?R?\s*666\b"), "chmod 666"),
    (re.compile(r">\s*/dev/(sd[a-z]|nvme|hd[a-z]|disk)"), "redirect to raw disk"),
    (re.compile(r">\s*/etc/passwd\b"), "overwrite /etc/passwd"),
    (re.compile(r">\s*/etc/shadow\b"), "overwrite /etc/shadow"),
    (re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"), "fork bomb"),
)


class ShellTool:
    def __init__(
        self,
        timeout_seconds: int = 10,
        allowlist: frozenset[str] | None = None,
        denylist: frozenset[str] | None = None,
        deny_patterns: frozenset[str] | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self._allowlist = allowlist if allowlist is not None else _ALLOWLIST
        self._denylist = denylist if denylist is not None else _DENYLIST
        self._deny_patterns = deny_patterns if deny_patterns is not None else SHELL_DENY_PATTERNS

    def is_denied(self, command: str) -> bool:
        """Backward-compatible substring deny-list check.

        Returns *True* when any pattern in ``self._deny_patterns`` is a
        substring of *command*.
        """
        return any(pattern in command for pattern in self._deny_patterns)

    def _validate(self, command: str) -> tuple[bool, str]:
        """Return (ok, error_message)."""
        command = command.strip()
        if not command:
            return False, "empty command"

        # Backward-compatible deny-pattern check
        matched = next((p for p in self._deny_patterns if p in command), None)
        if matched is not None:
            return False, f"command matches deny-list pattern: {matched}"

        # Reject shell metacharacters that imply piping, chaining, or substitution
        if any(ch in command for ch in "|;&`$(){}<>\n\r"):
            return (
                False,
                "shell metacharacters are not allowed (no pipes, chaining, or substitution)",
            )

        try:
            tokens = shlex.split(command)
        except ValueError as exc:
            return False, f"invalid shell syntax: {exc}"

        if not tokens:
            return False, "empty command after parsing"

        base = tokens[0]
        base_name = base.split("/")[-1]

        if base_name in self._denylist:
            return False, f"command '{base_name}' is on the deny-list"

        if base_name not in self._allowlist:
            return False, f"command '{base_name}' is not in the allow-list"

        for pat, reason in _DENY_PATTERNS:
            if pat.search(command):
                return False, f"forbidden pattern: {reason}"

        return True, ""

    def run(self, command: str) -> ToolResult:
        """Run a shell command and return ToolResult."""
        ok, error = self._validate(command)
        if not ok:
            return ToolResult(
                tool_name="shell",
                success=False,
                output="",
                error=error,
            )

        args = shlex.split(command)
        try:
            proc = subprocess.run(  # noqa: S603
                args,
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
            description="Run an allowed shell command with output capture.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            "The shell command to run (single command, no pipes or chaining)."
                        ),
                    }
                },
            },
            required=["command"],
        )


SHELL_TOOL = ShellTool()
TOOL_REGISTRY.register(SHELL_TOOL.spec(), handler=SHELL_TOOL.run)
