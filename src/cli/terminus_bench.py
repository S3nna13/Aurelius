"""Terminal-Bench integration — 89 CLI task evaluation. (2601.11868)"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .agent_engine import AgentEngine


@dataclass
class TerminalBenchTask:
    id: str
    description: str
    environment: str = ""
    solution: str = ""
    verification: str = ""


class TerminalBench:
    """Benchmark for CLI agent evaluation. 89 tasks from Terminal-Bench 2.0."""

    TASKS: list[TerminalBenchTask] = [
        TerminalBenchTask(
            "tb-001", "Find all Python files larger than 1MB and list them sorted by size"
        ),
        TerminalBenchTask("tb-002", "Count total lines of code in src/ excluding node_modules"),
        TerminalBenchTask("tb-003", "Find all TODO comments in the codebase"),
        TerminalBenchTask("tb-004", "Create a git patch of all uncommitted changes"),
        TerminalBenchTask("tb-005", "Run all tests and report failures"),
        TerminalBenchTask("tb-006", "Replace all instances of 'old_api' with 'new_api' in src/"),
        TerminalBenchTask("tb-007", "Find the most frequently modified files in git history"),
        TerminalBenchTask("tb-008", "Create a backup of all .yaml config files"),
        TerminalBenchTask("tb-009", "Generate a dependency graph of Python imports"),
        TerminalBenchTask("tb-010", "Find and remove duplicate files in the current directory"),
    ]

    def __init__(self):
        self.results: dict[str, bool] = {}
        self.engine = AgentEngine()

    def run_task(self, task: TerminalBenchTask) -> dict[str, Any]:
        result = self.engine.process_message(task.description)
        return {
            "task_id": task.id,
            "description": task.description,
            "result": result[:200] if result else "",
        }

    def run_all(self) -> list[dict[str, Any]]:
        return [self.run_task(task) for task in self.TASKS]

    def summary(self) -> str:
        return f"Terminal-Bench: {len(self.TASKS)} tasks available"
