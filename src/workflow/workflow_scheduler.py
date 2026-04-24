import heapq
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable


class WorkflowPriority(IntEnum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


@dataclass
class WorkflowJob:
    name: str
    fn: Callable
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    priority: WorkflowPriority = WorkflowPriority.NORMAL
    max_retries: int = 0
    tags: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.monotonic)


@dataclass(frozen=True)
class JobResult:
    job_id: str
    name: str
    success: bool
    result: Any
    duration_ms: float
    attempts: int
    error: str = ""


class WorkflowScheduler:
    def __init__(self) -> None:
        self._heap: list[tuple[int, float, str]] = []
        self._jobs: dict[str, WorkflowJob] = {}
        self._cancelled: set[str] = set()

    def submit(
        self,
        name: str,
        fn: Callable,
        priority: WorkflowPriority = WorkflowPriority.NORMAL,
        max_retries: int = 0,
        tags: list[str] | None = None,
    ) -> str:
        job = WorkflowJob(
            name=name,
            fn=fn,
            priority=priority,
            max_retries=max_retries,
            tags=list(tags or []),
        )
        self._jobs[job.job_id] = job
        heapq.heappush(self._heap, (priority.value, job.created_at, job.job_id))
        return job.job_id

    def _pop_next(self) -> WorkflowJob | None:
        while self._heap:
            _, _, job_id = heapq.heappop(self._heap)
            if job_id in self._cancelled:
                self._cancelled.discard(job_id)
                self._jobs.pop(job_id, None)
                continue
            job = self._jobs.pop(job_id, None)
            if job is not None:
                return job
        return None

    def run_next(self) -> JobResult | None:
        job = self._pop_next()
        if job is None:
            return None

        attempts = 0
        last_error = ""
        start = time.monotonic()
        output: Any = None
        success = False
        total_attempts = job.max_retries + 1

        for _ in range(total_attempts):
            attempts += 1
            try:
                output = job.fn()
                success = True
                last_error = ""
                break
            except Exception as exc:
                last_error = str(exc)

        duration = (time.monotonic() - start) * 1000.0
        return JobResult(
            job_id=job.job_id,
            name=job.name,
            success=success,
            result=output,
            duration_ms=duration,
            attempts=attempts,
            error=last_error,
        )

    def run_all(self) -> list[JobResult]:
        results: list[JobResult] = []
        while True:
            r = self.run_next()
            if r is None:
                break
            results.append(r)
        return results

    def queue_size(self) -> int:
        return sum(1 for _, _, jid in self._heap if jid not in self._cancelled)

    def pending_by_priority(self) -> dict[str, int]:
        counts = {p.name: 0 for p in WorkflowPriority}
        for _, _, jid in self._heap:
            if jid in self._cancelled:
                continue
            job = self._jobs.get(jid)
            if job is not None:
                counts[job.priority.name] += 1
        return counts

    def cancel(self, job_id: str) -> bool:
        if job_id not in self._jobs or job_id in self._cancelled:
            return False
        self._cancelled.add(job_id)
        return True


WORKFLOW_SCHEDULER_REGISTRY: dict[str, type[WorkflowScheduler]] = {"default": WorkflowScheduler}
