"""Aurelius Agent Router — dispatches tasks to the right agent mode.

Routes tasks based on intent classification. All powered by Aurelius.
No external models. Modes: coding, research, security, debug, architect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentMode:
    id: str
    name: str
    description: str
    system_prompt: str
    tools: list[str] = field(default_factory=list)
    temperature: float = 0.7


BUILTIN_MODES: dict[str, AgentMode] = {
    "coding": AgentMode(
        "coding", "Coding Agent",
        "Write, edit, and refactor code across the repository",
        "You are Aurelius, an expert software engineer. Write clean, tested, production-quality code.",
        ["read_file", "write_file", "edit_file", "run_command", "search_code", "run_tests", "git_ops"],
        0.3,
    ),
    "research": AgentMode(
        "research", "Research Agent",
        "Deep analysis of codebases, architectures, and documentation",
        "You are Aurelius, a senior architect. Analyze systems deeply and provide comprehensive insights.",
        ["read_file", "search_code", "grep", "list_directory", "git_log"],
        0.5,
    ),
    "security": AgentMode(
        "security", "Security Agent",
        "Security audit, vulnerability detection, and threat modeling",
        "You are Aurelius, a security expert. Audit code for vulnerabilities following OWASP and CWE guidelines.",
        ["read_file", "search_code", "run_tests", "check_dependencies"],
        0.3,
    ),
    "debug": AgentMode(
        "debug", "Debug Agent",
        "Find and fix bugs with systematic root cause analysis",
        "You are Aurelius, a debugging expert. Systematically isolate, diagnose, and fix issues.",
        ["read_file", "search_code", "run_tests", "run_command", "inspect_variable"],
        0.4,
    ),
    "architect": AgentMode(
        "architect", "Architect Agent",
        "Design system architecture and plan large-scale changes",
        "You are Aurelius, a system architect. Design clean, scalable architectures with clear boundaries.",
        ["read_file", "list_directory", "search_code", "analyze_dependencies"],
        0.6,
    ),
}


class AgentRouter:
    """Routes tasks to the appropriate agent mode based on intent."""

    def __init__(self):
        self.modes = BUILTIN_MODES

    def classify(self, task: str) -> AgentMode:
        task_lower = task.lower()
        if any(kw in task_lower for kw in ["security", "vulnerability", "cve", "threat", "audit", "attack"]):
            return self.modes["security"]
        if any(kw in task_lower for kw in ["debug", "bug", "fix", "error", "crash", "issue", "failing"]):
            return self.modes["debug"]
        if any(kw in task_lower for kw in ["design", "architecture", "plan", "structure", "component", "system"]):
            return self.modes["architect"]
        if any(kw in task_lower for kw in ["research", "analysis", "analyze", "understand", "explain", "document"]):
            return self.modes["research"]
        return self.modes["coding"]

    def route(self, task: str) -> tuple[AgentMode, str]:
        mode = self.classify(task)
        return mode, mode.system_prompt

    def list_modes(self) -> list[dict[str, Any]]:
        return [
            {"id": m.id, "name": m.name, "description": m.description, "tools": len(m.tools)}
            for m in self.modes.values()
        ]
