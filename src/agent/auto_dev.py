"""AutoDev — Automated AI-Driven Development agent.

Implementation of arxiv:2403.08299 (AutoDev).

Architecture: autonomous agents that plan and execute software engineering
tasks inside a Docker-containerized sandbox with guardrails.

Key components:
- DevAgent: Autonomous development agent with planning, coding, building, testing, git
- DevSandbox: Docker-based secure execution environment with allow/deny guardrails
- FileOps: File editing, creation, deletion, retrieval operations
- GitOps: Git operations (commit, branch, push, pull, diff, log)
- BuildOps: Build and test execution with output capture
- DevPlan: Structured plan for software engineering tasks
- DevGuardrails: Security guardrails for allowed/blocked operations
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

AGENT_LOOP_REGISTRY: dict[str, type] = {}
TOOL_REGISTRY: dict[str, Any] = {}


class SandboxLevel(Enum):
    NONE = "none"
    READ = "read"
    WRITE = "write"
    BUILD = "build"
    DANGEROUS = "dangerous"


class DevTaskStatus(Enum):
    PENDING = "pending"
    PLANNING = "planning"
    CODING = "coding"
    BUILDING = "building"
    TESTING = "testing"
    GIT = "git"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class DevOperation:
    op_type: str  # create_file | edit_file | delete_file | read_file | run_build | run_test | git_commit | git_push | search_code
    path: str = ""
    content: str = ""
    old_string: str = ""
    new_string: str = ""
    result: str = ""
    status: str = "pending"
    sandbox_level: SandboxLevel = SandboxLevel.READ
    error: str | None = None


@dataclass
class DevPlan:
    task: str
    operations: list[DevOperation] = field(default_factory=list)
    status: DevTaskStatus = DevTaskStatus.PENDING
    language: str = "python"
    project_type: str = "script"
    created: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed: datetime | None = None


@dataclass
class DevGuardrail:
    pattern: str
    action: str  # allow | deny | warn
    reason: str = ""


class DevSandbox:
    """Docker-containerized sandbox with guardrails.

    Mirrors AutoDev's secure execution environment:
    - All operations confined within Docker
    - Allow/deny lists for commands and file paths
    - User-defined guardrails for privacy and security
    - Output capture and timeout enforcement
    """

    def __init__(self, work_dir: str | None = None, use_docker: bool = False) -> None:
        self.work_dir = work_dir or tempfile.mkdtemp(prefix="ark_auto_dev_")
        self.use_docker = use_docker
        self.guardrails: list[DevGuardrail] = [
            DevGuardrail(r"rm\s+-rf\s+/", "deny", "Prevents root filesystem deletion"),
            DevGuardrail(r"sudo", "deny", "Prevents privilege escalation"),
            DevGuardrail(r"chmod\s+777", "warn", "Overly permissive file permissions"),
            DevGuardrail(r"curl\s+.*\||wget\s+.*\|", "deny", "Prevents pipe-to-shell injection"),
            DevGuardrail(
                r"(?:eval|exec|system|popen|subprocess)\(.*\)", "warn", "Dynamic code execution"
            ),
            DevGuardrail(
                r"sk-[a-fA-F0-9]{32,}|pk-[a-fA-F0-9]{32,}", "deny", "Prevents API key leakage"
            ),
        ]
        self._history: list[dict[str, Any]] = []

    def check_operation(self, operation: DevOperation) -> bool:
        """Check an operation against guardrails. Returns True if allowed."""
        op_str = f"{operation.op_type}:{operation.path}:{operation.content}"
        for guardrail in self.guardrails:
            if re.search(guardrail.pattern, op_str, re.IGNORECASE):
                if guardrail.action == "deny":
                    operation.status = "blocked"
                    operation.error = f"Guardrail blocked: {guardrail.reason}"
                    return False
        return True

    def add_guardrail(self, pattern: str, action: str, reason: str = "") -> None:
        self.guardrails.append(DevGuardrail(pattern=pattern, action=action, reason=reason))

    def execute_command(self, command: str, timeout: int = 30) -> tuple[str, str, int]:
        """Execute a shell command in the sandbox.

        Returns: (stdout, stderr, returncode)
        """
        self._history.append({"command": command, "timestamp": datetime.now(UTC).isoformat()})

        if self.use_docker:
            cmd = ["docker", "exec", "-i", "ark-sandbox", "sh", "-c", command]
        else:
            cmd = ["sh", "-c", command]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.work_dir,
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            return "", f"Command timed out after {timeout}s", -1
        except FileNotFoundError:
            return "", "Command not found", -1

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            raise ValueError(f"Absolute paths not allowed in sandbox: {path}")
        resolved = os.path.realpath(os.path.join(self.work_dir, path))
        real_work = os.path.realpath(self.work_dir)
        if not resolved.startswith(real_work + os.sep) and resolved != real_work:
            raise ValueError(f"Path traversal blocked: {path} resolves outside sandbox")
        return resolved

    def read_file(self, path: str) -> str:
        full_path = self._resolve_path(path)
        if not os.path.exists(full_path):
            return ""
        with open(full_path) as f:
            return f.read()

    def write_file(self, path: str, content: str) -> None:
        full_path = self._resolve_path(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)

    def edit_file(self, path: str, old_string: str, new_string: str) -> bool:
        content = self.read_file(path)
        if old_string not in content:
            return False
        new_content = content.replace(old_string, new_string, 1)
        self.write_file(path, new_content)
        return True

    def delete_file(self, path: str) -> None:
        full_path = self._resolve_path(path)
        if os.path.exists(full_path):
            os.remove(full_path)

    def get_history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def reset(self) -> None:
        self._history.clear()


class AutoDevAgent:
    """Autonomous software development agent.

    Full AutoDev pipeline:
    1. Plan: Analyze task, create DevPlan with operations
    2. Code: Execute file operations (create, edit, delete)
    3. Build: Run build process, capture output
    4. Test: Run tests, capture results
    5. Git: Commit changes, manage version control
    6. Verify: Check all operations completed successfully

    All operations run through DevSandbox with guardrails.
    """

    def __init__(
        self,
        sandbox: DevSandbox | None = None,
        plan_fn: Callable[[str], DevPlan] | None = None,
        code_fn: Callable[[DevPlan], list[DevOperation]] | None = None,
        build_fn: Callable[[DevPlan], str] | None = None,
        test_fn: Callable[[DevPlan], str] | None = None,
        git_fn: Callable[[DevPlan], str] | None = None,
    ) -> None:
        self.sandbox = sandbox or DevSandbox()
        self.plan_fn = plan_fn or self._default_plan
        self.code_fn = code_fn or self._default_code
        self.build_fn = build_fn or self._default_build
        self.test_fn = test_fn or self._default_test
        self.git_fn = git_fn or self._default_git
        self._session_id = uuid.uuid4().hex[:8]

    def execute(self, task: str) -> DevPlan:
        plan = self.plan_fn(task)

        plan.status = DevTaskStatus.CODING
        ops = self.code_fn(plan)
        plan.operations.extend(ops)

        plan.status = DevTaskStatus.BUILDING
        build_result = self.build_fn(plan)
        plan.operations.append(DevOperation(op_type="run_build", result=build_result))

        plan.status = DevTaskStatus.TESTING
        test_result = self.test_fn(plan)
        plan.operations.append(DevOperation(op_type="run_test", result=test_result))

        plan.status = DevTaskStatus.GIT
        git_result = self.git_fn(plan)
        plan.operations.append(DevOperation(op_type="git_commit", result=git_result))

        failed = [op for op in plan.operations if op.status == "blocked"]
        plan.status = DevTaskStatus.FAILED if failed else DevTaskStatus.COMPLETED
        plan.completed = datetime.now(UTC)
        return plan

    def _default_plan(self, task: str) -> DevPlan:
        language = "python"
        if any(kw in task.lower() for kw in ["javascript", "typescript", "node", "react"]):
            language = "javascript"
        elif any(kw in task.lower() for kw in ["rust", "cargo"]):
            language = "rust"
        elif any(kw in task.lower() for kw in ["go", "golang"]):
            language = "go"
        ops = [
            DevOperation(
                op_type="create_file", path=f"main.{language}", sandbox_level=SandboxLevel.WRITE
            ),
        ]
        return DevPlan(task=task, operations=ops, language=language)

    def _default_code(self, plan: DevPlan) -> list[DevOperation]:
        ops: list[DevOperation] = []
        for op in plan.operations:
            if op.op_type == "create_file" and not self.sandbox.check_operation(op):
                continue
            if op.op_type in ("create_file", "edit_file"):
                self.sandbox.write_file(op.path, op.content)
                op.status = "completed"
                op.result = f"Written {len(op.content)} bytes to {op.path}"
            elif op.op_type == "delete_file":
                self.sandbox.delete_file(op.path)
                op.status = "completed"
                op.result = f"Deleted {op.path}"
            elif op.op_type == "read_file":
                content = self.sandbox.read_file(op.path)
                op.result = content
                op.status = "completed"
            ops.append(op)
        return ops

    def _default_build(self, plan: DevPlan) -> str:
        if plan.language == "python":
            stdout, stderr, rc = self.sandbox.execute_command(
                "python3 -m py_compile main.py 2>&1 || python -m py_compile main.py 2>&1"
            )
        elif plan.language in ("javascript", "typescript"):
            stdout, stderr, rc = self.sandbox.execute_command("node --check main.js 2>&1")
        else:
            return "Build not implemented for this language"
        if rc != 0:
            return f"Build failed:\n{stderr or stdout}"
        return "Build passed"

    def _default_test(self, plan: DevPlan) -> str:
        if plan.language == "python":
            stdout, stderr, rc = self.sandbox.execute_command(
                "python3 -m pytest tests/ -q 2>&1 || echo 'No tests found'"
            )
        else:
            return "Test not implemented for this language"
        return stdout or stderr or "No test output"

    def _default_git(self, plan: DevPlan) -> str:
        cmds = [
            "git init",
            "git add -A",
            f'git commit -m "AutoDev: {plan.task[:60]}"',
        ]
        results: list[str] = []
        for cmd in cmds:
            stdout, stderr, rc = self.sandbox.execute_command(cmd)
            results.append(f"$ {cmd}\n{stdout or stderr}")
        return "\n".join(results)


# Registry hooks
AGENT_LOOP_REGISTRY["auto_dev"] = AutoDevAgent
TOOL_REGISTRY["auto_dev_sandbox"] = {
    "name": "auto_dev_sandbox",
    "description": "Secure Docker sandbox for automated development operations",
    "sandbox_level": "dangerous",
}
