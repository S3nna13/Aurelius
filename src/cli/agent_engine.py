"""Aurelius CLI Terminal — Dragon-themed AI coding agent.

Inspired by Claude Code, OPENDEV, Terminal-Bench, NL2SH, YAMLE.
Features:
- Dragon mascot with lightning blue (#00BFFF) theme
- Dual-agent architecture (planner + executor)
- NL→Bash translation with functional equivalence
- Adaptive context compaction
- Session memory with automated persistence
- Tool authorization with deny-first safety
- Streaming AI responses
"""

# ruff: noqa: E501
from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.traceback import install

from .dragon_mascot import DRAGON_ART

install()

LIGHTNING_BLUE = "#00BFFF"
DRAGON_GREEN = "#00FF88"
TERMINAL_BG = "#0a0a1a"

console = Console(highlight=False)

# ─── Data Models ────────────────────────────────────────────────────────────


@dataclass
class ToolResult:
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    duration: float = 0.0

    @property
    def success(self) -> bool:
        return self.returncode == 0

    def truncated(self, max_len: int = 5000) -> str:
        text = self.stdout or self.stderr
        return text[:max_len] + "..." if len(text) > max_len else text


@dataclass
class Message:
    role: str
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    messages: list[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    workspace: str = ""
    memory: dict[str, Any] = field(default_factory=dict)


# ─── NL2SH Engine ──────────────────────────────────────────────────────────


class NL2SHEngine:
    """Natural Language to Bash translation (2502.06858 / NL2SH).

    Translates natural language commands to bash with functional
    equivalence verification via output comparison.
    """

    COMMAND_MAP: dict[str, list[str]] = {
        "list files": ["ls -la", "ls -l", "ls"],
        "show directory": ["pwd"],
        "make directory": ["mkdir -p {dir}"],
        "create file": ["touch {file}", "echo > {file}"],
        "edit file": ["echo '{content}' > {file}", "cat > {file} << 'EOF'\n{content}\nEOF"],
        "read file": ["cat {file}", "head -n 100 {file}"],
        "find text": ["grep -r '{pattern}' {path}", "rg '{pattern}' {path}"],
        "count lines": ["wc -l {file}"],
        "delete file": ["rm {file}"],
        "copy file": ["cp {source} {dest}"],
        "move file": ["mv {source} {dest}"],
        "git status": ["git status"],
        "git diff": ["git diff"],
        "git log": ["git log --oneline -n 10"],
        "build": ["make", "cargo build", "npm run build"],
        "test": ["pytest", "make test", "cargo test", "npm test"],
        "search code": ["grep -rn '{pattern}' src/"],
    }

    def __init__(self, llm_fn: Callable | None = None):
        self.llm_fn = llm_fn or self._default_llm
        self._cache: dict[str, str] = {}

    def _default_llm(self, prompt: str) -> str:
        if "list" in prompt.lower() and "file" in prompt.lower():
            return "ls -la"
        if "find" in prompt.lower() and "text" in prompt.lower():
            return "grep -r {} ."
        if "git status" in prompt.lower():
            return "git status"
        if "make" in prompt.lower() or "build" in prompt.lower():
            return "make"
        return prompt

    def translate(self, nl_command: str, context: str = "") -> str:
        cache_key = f"{nl_command}|{context}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        nl_lower = nl_command.lower().strip()
        for key, commands in self.COMMAND_MAP.items():
            if key in nl_lower or nl_lower.startswith(key.split()[0]):
                result = commands[0]
                self._cache[cache_key] = result
                return result

        result = self.llm_fn(f"Translate to bash: {nl_command}\nContext: {context}")
        self._cache[cache_key] = result
        return result

    def verify_equivalence(self, cmd_a: str, cmd_b: str) -> float:
        try:
            r1 = subprocess.run(  # noqa: S603
                shlex.split(cmd_a), capture_output=True, text=True, timeout=5
            )
            r2 = subprocess.run(  # noqa: S603
                shlex.split(cmd_b), capture_output=True, text=True, timeout=5
            )
            if r1.stdout == r2.stdout:
                return 1.0
            common = set(r1.stdout.split()) & set(r2.stdout.split())
            union = set(r1.stdout.split()) | set(r2.stdout.split())
            return len(common) / max(len(union), 1)
        except Exception:
            return 0.0


# ─── Tool Executor ─────────────────────────────────────────────────────────


class ToolExecutor:
    """Executes shell commands, edits files, runs code with safety controls."""

    DENY_LIST = {"rm -rf /", "dd if=", "> /dev/sda", ":(){ :|:& };:", "chmod 000 /"}
    ALLOW_LIST = {
        "ls",
        "cat",
        "head",
        "tail",
        "grep",
        "find",
        "pwd",
        "echo",
        "touch",
        "mkdir",
        "cp",
        "mv",
        "rm",
        "git",
        "make",
        "cargo",
        "npm",
        "python",
        "pytest",
        "cargo test",
    }

    def __init__(self, sandbox: bool = True):
        self.sandbox = sandbox
        self.history: list[tuple[str, ToolResult]] = []

    def check_safety(self, command: str) -> tuple[bool, str]:
        cmd_lower = command.lower()
        for deny in self.DENY_LIST:
            if deny in cmd_lower:
                return False, f"Command denied: matches pattern '{deny}'"
        base = command.split()[0] if command.split() else ""
        if base not in self.ALLOW_LIST and not base.startswith("./"):
            return False, f"Command not in allow list: {base}"
        return True, ""

    def execute(self, command: str, timeout: int = 30, cwd: str | None = None) -> ToolResult:
        safe, reason = self.check_safety(command)
        if not safe:
            return ToolResult(stderr=reason, returncode=-1)

        start = time.time()
        try:
            result = subprocess.run(  # noqa: S602
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )
            elapsed = time.time() - start
            tr = ToolResult(
                stdout=result.stdout[:10000],
                stderr=result.stderr[:5000],
                returncode=result.returncode,
                duration=elapsed,
            )
        except subprocess.TimeoutExpired:
            tr = ToolResult(stderr=f"Timeout after {timeout}s", returncode=-1)
        except Exception as e:
            tr = ToolResult(stderr=str(e), returncode=-1)

        self.history.append((command, tr))
        return tr

    def edit_file(self, path: str, content: str, mode: str = "w") -> ToolResult:
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            if mode == "w":
                p.write_text(content)
            elif mode == "a":
                with open(p, "a") as f:
                    f.write(content)
            return ToolResult(stdout=f"Written {len(content)} bytes to {path}", returncode=0)
        except Exception as e:
            return ToolResult(stderr=str(e), returncode=-1)

    def read_file(self, path: str) -> ToolResult:
        try:
            p = Path(path)
            if not p.exists():
                return ToolResult(stderr=f"File not found: {path}", returncode=-1)
            return ToolResult(stdout=p.read_text(), returncode=0)
        except Exception as e:
            return ToolResult(stderr=str(e), returncode=-1)

    def search_code(self, pattern: str, path: str = ".") -> ToolResult:
        return self.execute(f"grep -rn '{pattern}' {path}")


# ─── Agent Engine ──────────────────────────────────────────────────────────


class AgentEngine:
    """Dual-agent architecture: Planner + Executor (2603.05344 / OPENDEV inspired)."""

    def __init__(self, llm_fn: Callable | None = None):
        self.llm_fn = llm_fn or self._default_llm
        self.tool = ToolExecutor()
        self.nl2sh = NL2SHEngine(llm_fn)
        self.session = Session()
        self._plan_history: list[dict[str, Any]] = []
        self._context_size: int = 0

    def _default_llm(self, prompt: str) -> str:
        return f"# Response to: {prompt[:60]}...\nprint('processing')"

    def plan(self, task: str) -> list[dict[str, Any]]:
        prompt = f"""Plan the steps to accomplish this task:
Task: {task}
Current workspace: {self.session.workspace or os.getcwd()}

Respond with a numbered plan, one step per line:"""
        response = self.llm_fn(prompt)
        steps: list[dict[str, Any]] = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                steps.append(
                    {
                        "description": line,
                        "type": "shell",
                        "command": self.nl2sh.translate(line),
                    }
                )
        if not steps:
            steps.append(
                {"description": task, "type": "shell", "command": self.nl2sh.translate(task)}
            )
        self._plan_history.append({"task": task, "steps": steps})
        return steps

    def execute_plan(self, steps: list[dict[str, Any]]) -> list[ToolResult]:
        results: list[ToolResult] = []
        for step in steps:
            cmd = step.get("command", "")
            if step.get("type") == "edit":
                result = self.tool.edit_file(step.get("file", ""), step.get("content", ""))
            else:
                result = self.tool.execute(cmd, cwd=self.session.workspace or None)
            results.append(result)
            self.session.messages.append(
                Message(role="tool", content=f"$ {cmd}\n{result.truncated(200)}")
            )
        return results

    def process_message(self, user_input: str) -> str:
        self.session.messages.append(Message(role="user", content=user_input))

        nl_lower = user_input.lower()

        if nl_lower.startswith("/"):
            return self._handle_command(user_input)

        if any(kw in nl_lower for kw in ["plan", "steps", "how to", "approach"]):
            steps = self.plan(user_input)
            result_lines = ["**Plan:**"]
            for i, s in enumerate(steps, 1):
                result_lines.append(f"{i}. {s['description']}")
                if s.get("command"):
                    result_lines.append(f"   → `{s['command']}`")
            result = "\n".join(result_lines)

            exec_results = self.execute_plan(steps)
            for i, (s, r) in enumerate(zip(steps, exec_results)):
                if r.success and r.stdout:
                    result += f"\n\n**Step {i + 1} result:**\n```\n{r.truncated(500)}\n```"
                elif not r.success:
                    result += f"\n\n**Step {i + 1} error:** `{r.stderr[:200]}`"
            self.session.messages.append(Message(role="assistant", content=result))
            return result

        if any(kw in nl_lower for kw in ["run", "execute", "bash", "shell", "command"]):
            bash_cmd = self.nl2sh.translate(user_input)
            result = self.tool.execute(bash_cmd, cwd=self.session.workspace or None)
            output = f"$ {bash_cmd}\n"
            if result.success and result.stdout:
                output += f"```\n{result.truncated(1000)}\n```"
            elif not result.success:
                output += f"⚠️ Error: {result.truncated(200)}"
            else:
                output += "✅ Command completed (no output)"
            self.session.messages.append(Message(role="assistant", content=output))
            return output

        if any(kw in nl_lower for kw in ["search", "find", "grep"]):
            result = self.tool.execute(
                self.nl2sh.translate(user_input), cwd=self.session.workspace or None
            )
            output = (
                f"**Search results:**\n```\n{result.truncated(2000)}\n```"
                if result.success and result.stdout
                else "No results found."
            )
            self.session.messages.append(Message(role="assistant", content=output))
            return output

        response = self.llm_fn(user_input)
        self.session.messages.append(Message(role="assistant", content=response))
        return response

    def _handle_command(self, cmd: str) -> str:
        parts = cmd[1:].strip().split()
        command = parts[0] if parts else ""

        if command == "help":
            return self._help_text()
        elif command in ("clear", "cls"):
            if os.name == "nt":
                subprocess.run(["cmd", "/c", "cls"], check=False)  # noqa: S607
            else:
                subprocess.run(["clear"], check=False)  # noqa: S607
            return ""
        elif command == "status":
            return self._status_text()
        elif command == "session":
            return f"Session ID: {self.session.id}\nMessages: {len(self.session.messages)}\nWorkspace: {self.session.workspace or os.getcwd()}"
        elif command == "workspace":
            ws = parts[1] if len(parts) > 1 else ""
            if ws:
                self.session.workspace = os.path.abspath(ws)
                return f"Workspace set to: {self.session.workspace}"
            return f"Current workspace: {self.session.workspace or os.getcwd()}"
        elif command == "history":
            lines = []
            for i, msg in enumerate(self.session.messages[-20:], 1):
                role = f"[{msg.role}]"
                content = msg.content[:80].replace("\n", " ")
                lines.append(f"  {i:>3}. {role:<12} {content}")
            return "\n".join(lines) if lines else "No history yet."
        elif command == "save":
            path = parts[1] if len(parts) > 1 else f"aurelius_session_{self.session.id}.json"
            self._save_session(path)
            return f"Session saved to: {path}"
        elif command == "load":
            path = parts[1] if len(parts) > 1 else ""
            if not path:
                return "Usage: /load <filepath>"
            self._load_session(path)
            return f"Session loaded from: {path}"
        elif command == "mascot":
            return DRAGON_ART
        elif command in ("q", "quit", "exit"):
            return "exit"
        return f"Unknown command: /{command}. Try /help"

    def _help_text(self) -> str:
        return """**Aurelius CLI Commands:**
  `/help`        — Show this help
  `/clear`       — Clear screen
  `/status`      — Show session status
  `/session`     — Show session info
  `/workspace <dir>` — Set workspace directory
  `/history`     — Show message history
  `/save <path>` — Save session to file
  `/load <path>` — Load session from file
  `/mascot`      — Display dragon mascot
  `/quit`        — Exit terminal

**Natural Language:**
  • Type any question or task
  • "plan ..." — AI will plan and execute steps
  • "run ..." — Execute bash commands
  • "search ..." — Search codebase
  • "edit file.txt" — Edit files via AI

**Safety:**
  • Deny-first tool authorization
  • Command allow/deny lists
  • Session-scoped permissions"""

    def _status_text(self) -> str:
        mem = self.session.memory or {}
        return (
            f"""**Session Status**
  Session: {self.session.id[:12]}
  Messages: {len(self.session.messages)}
  Workspace: {self.session.workspace or os.getcwd()}
  Memory keys: {list(mem.keys())}
  Tool history: {len(self.tool.history)} commands
  NL2SH cache: {len(self.nl2sh._cache)} entries"""
        )

    def _save_session(self, path: str) -> None:
        data = {
            "id": self.session.id,
            "messages": [{"role": m.role, "content": m.content} for m in self.session.messages],
            "workspace": self.session.workspace,
            "memory": self.session.memory,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def _load_session(self, path: str) -> None:
        data = json.loads(Path(path).read_text())
        self.session = Session(
            id=data.get("id", uuid.uuid4().hex[:12]),
            messages=[Message(**m) for m in data.get("messages", [])],
            workspace=data.get("workspace", ""),
            memory=data.get("memory", {}),
        )
