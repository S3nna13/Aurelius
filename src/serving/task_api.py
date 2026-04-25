"""Async task lifecycle REST contracts (AGI TRD pattern).

Provides dataclasses for task creation, result tracking, and a thread-safe
in-memory TaskStore. Production deployments would replace the store with a
database backend.
"""

import uuid
import time
from dataclasses import dataclass, field
from typing import Literal, Any

RiskProfile = Literal["research", "balanced", "strict"]
TaskStatus = Literal["pending", "running", "completed", "failed", "cancelled"]


@dataclass
class Artifact:
    artifact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    artifact_type: str = "text"
    content: str = ""
    mime_type: str = "text/plain"


@dataclass
class TaskCitation:
    source: str
    text: str
    url: str = ""
    confidence: float = 1.0


@dataclass
class TaskMetrics:
    steps: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    tool_calls: int = 0
    safety_warnings: int = 0


@dataclass
class CreateTaskRequest:
    goal: str
    user_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    constraints: dict = field(default_factory=dict)
    priority: int = 5
    risk_profile: RiskProfile = "balanced"
    tool_allowlist: list[str] = field(default_factory=list)
    max_steps: int = 50
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)


@dataclass
class TaskResult:
    task_id: str
    status: TaskStatus
    summary: str = ""
    artifacts: list[Artifact] = field(default_factory=list)
    citations: list[TaskCitation] = field(default_factory=list)
    metrics: TaskMetrics = field(default_factory=TaskMetrics)
    error: str | None = None
    completed_at: float | None = None


@dataclass
class APIError:
    code: str
    message: str
    retry_after_seconds: float = 0.0
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "retry_after_seconds": self.retry_after_seconds,
                "trace_id": self.trace_id,
            }
        }


class TaskStore:
    """In-memory task registry. Production would use a database."""

    def __init__(self):
        self._tasks: dict[str, CreateTaskRequest] = {}
        self._results: dict[str, TaskResult] = {}

    def create(self, request: CreateTaskRequest) -> TaskResult:
        self._tasks[request.task_id] = request
        result = TaskResult(task_id=request.task_id, status="pending")
        self._results[request.task_id] = result
        return result

    def get_result(self, task_id: str) -> TaskResult | None:
        return self._results.get(task_id)

    def update_status(self, task_id: str, status: TaskStatus, **kwargs) -> bool:
        if task_id not in self._results:
            return False
        result = self._results[task_id]
        result.status = status
        for k, v in kwargs.items():
            if hasattr(result, k):
                setattr(result, k, v)
        if status in ("completed", "failed", "cancelled"):
            result.completed_at = time.time()
        return True

    def list_tasks(self, user_id: str | None = None) -> list[TaskResult]:
        results = list(self._results.values())
        if user_id:
            task_ids = {tid for tid, req in self._tasks.items() if req.user_id == user_id}
            results = [r for r in results if r.task_id in task_ids]
        return results


TASK_STORE = TaskStore()

# Register in the shared SERVING_REGISTRY if available.
try:
    from src.serving import SERVING_REGISTRY  # type: ignore
    SERVING_REGISTRY["task_api"] = TASK_STORE
except ImportError:
    pass
