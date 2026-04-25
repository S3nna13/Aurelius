from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    tool: str
    target: str | None = None
    stdout: str = ""
    stderr: str = ""
    return_code: int = -1
    error: str | None = None


class SecurityToolManager:
    def __init__(self, tool_paths: dict[str, str] | None = None) -> None:
        self._tool_paths: dict[str, str] = {}
        if tool_paths:
            self._tool_paths.update(tool_paths)
        self._discover_tools()

    def _discover_tools(self) -> None:
        tools = ["nmap", "nikto", "sqlmap", "gobuster", "ffuf", "wpscan", "hydra", "john", "aircrack-ng"]
        for tool in tools:
            try:
                result = subprocess.run(
                    ["which", tool],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    path = result.stdout.strip()
                    if path:
                        self._tool_paths[tool] = path
            except Exception:
                pass

    def is_available(self, tool: str) -> bool:
        return tool in self._tool_paths

    def available_tools(self) -> list[str]:
        return list(self._tool_paths.keys())

    def run_nmap(self, target: str, flags: str = "-sV -sC") -> ToolResult:
        return self._run("nmap", target, flags.split())

    def run_nikto(self, target: str) -> ToolResult:
        return self._run("nikto", target, ["-h", target])

    def run_sqlmap(self, target: str, data: str | None = None) -> ToolResult:
        args = ["-u", target, "--batch"]
        if data:
            args += ["--data", data]
        return self._run("sqlmap", target, args)

    def run_gobuster(self, target: str, wordlist: str = "/usr/share/wordlists/dirb/common.txt") -> ToolResult:
        return self._run("gobuster", target, ["dir", "-u", target, "-w", wordlist])

    def run_ffuf(self, target: str, wordlist: str = "/usr/share/wordlists/dirb/common.txt") -> ToolResult:
        return self._run("ffuf", target, ["-u", f"{target}/FUZZ", "-w", wordlist])

    def _run(self, tool: str, target: str | None, args: list[str]) -> ToolResult:
        if tool not in self._tool_paths:
            return ToolResult(tool=tool, target=target, error=f"{tool} not found")
        try:
            cmd = [self._tool_paths[tool]] + args
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            return ToolResult(
                tool=tool,
                target=target,
                stdout=result.stdout[:10000],
                stderr=result.stderr[:5000],
                return_code=result.returncode,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(tool=tool, target=target, error="timed out after 300s")
        except Exception as e:
            return ToolResult(tool=tool, target=target, error=str(e))
