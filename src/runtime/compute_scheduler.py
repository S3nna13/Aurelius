from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class Priority(int, Enum):
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class ComputeJob:
    job_id: str
    priority: Priority
    estimated_flops: float
    submitted_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: str = "pending"
    metadata: dict = field(default_factory=dict)


class ComputeScheduler:
    """Priority-based compute job scheduler with FLOP budgeting."""

    def __init__(self, max_concurrent: int = 4, flop_budget_per_s: float = 1e12) -> None:
        self._max_concurrent = max_concurrent
        self._flop_budget_per_s = flop_budget_per_s
        self._jobs: Dict[str, ComputeJob] = {}

    def submit(
        self,
        priority: Priority,
        estimated_flops: float,
        metadata: Optional[dict] = None,
    ) -> ComputeJob:
        job = ComputeJob(
            job_id=str(uuid.uuid4()),
            priority=priority,
            estimated_flops=estimated_flops,
            submitted_at=time.monotonic(),
            metadata=metadata or {},
        )
        self._jobs[job.job_id] = job
        return job

    def next_job(self) -> Optional[ComputeJob]:
        pending = [j for j in self._jobs.values() if j.status == "pending"]
        if not pending:
            return None
        pending.sort(key=lambda j: (j.priority, j.submitted_at))
        return pending[0]

    def start(self, job_id: str) -> ComputeJob:
        if job_id not in self._jobs:
            raise KeyError(f"Unknown job: {job_id}")
        running = [j for j in self._jobs.values() if j.status == "running"]
        if len(running) >= self._max_concurrent:
            raise RuntimeError(
                f"max_concurrent ({self._max_concurrent}) running jobs reached"
            )
        job = self._jobs[job_id]
        job.started_at = time.monotonic()
        job.status = "running"
        return job

    def complete(self, job_id: str) -> ComputeJob:
        if job_id not in self._jobs:
            raise KeyError(f"Unknown job: {job_id}")
        job = self._jobs[job_id]
        job.completed_at = time.monotonic()
        job.status = "done"
        return job

    def cancel(self, job_id: str) -> ComputeJob:
        if job_id not in self._jobs:
            raise KeyError(f"Unknown job: {job_id}")
        job = self._jobs[job_id]
        job.status = "cancelled"
        return job

    def running_jobs(self) -> List[ComputeJob]:
        return [j for j in self._jobs.values() if j.status == "running"]

    def pending_jobs(self) -> List[ComputeJob]:
        return [j for j in self._jobs.values() if j.status == "pending"]

    def stats(self) -> dict:
        all_jobs = list(self._jobs.values())
        return {
            "pending": sum(1 for j in all_jobs if j.status == "pending"),
            "running": sum(1 for j in all_jobs if j.status == "running"),
            "done": sum(1 for j in all_jobs if j.status == "done"),
            "cancelled": sum(1 for j in all_jobs if j.status == "cancelled"),
            "total_flops_scheduled": sum(j.estimated_flops for j in all_jobs),
        }
