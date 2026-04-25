from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass
class ElasticConfig:
    min_workers: int = 1
    max_workers: int = 8
    rdzv_backend: str = "c10d"
    max_restarts: int = 3
    monitor_interval_s: float = 5.0


class WorkerState(str, Enum):
    INIT = "INIT"
    RUNNING = "RUNNING"
    FAILED = "FAILED"
    SUCCEEDED = "SUCCEEDED"
    STOPPED = "STOPPED"


@dataclass
class WorkerInfo:
    worker_id: int
    state: WorkerState = WorkerState.INIT
    restart_count: int = 0


class ElasticCoordinator:
    def __init__(self, config: ElasticConfig | None = None) -> None:
        self.config = config or ElasticConfig()
        self.workers: dict[int, WorkerInfo] = {}

    def register_worker(self, worker_id: int) -> WorkerInfo:
        info = WorkerInfo(worker_id=worker_id)
        self.workers[worker_id] = info
        return info

    def mark_failed(self, worker_id: int) -> bool:
        info = self.workers[worker_id]
        info.state = WorkerState.FAILED
        if info.restart_count < self.config.max_restarts:
            info.restart_count += 1
            info.state = WorkerState.INIT
            return True
        return False

    def mark_succeeded(self, worker_id: int) -> None:
        self.workers[worker_id].state = WorkerState.SUCCEEDED

    def active_workers(self) -> list[int]:
        return [
            wid
            for wid, info in self.workers.items()
            if info.state in (WorkerState.RUNNING, WorkerState.INIT)
        ]

    def should_restart(self) -> bool:
        return any(
            info.state == WorkerState.FAILED and info.restart_count < self.config.max_restarts
            for info in self.workers.values()
        )

    def world_size(self) -> int:
        return sum(
            1
            for info in self.workers.values()
            if info.state not in (WorkerState.FAILED, WorkerState.STOPPED)
        )

    def is_elastic_valid(self) -> bool:
        ws = self.world_size()
        return self.config.min_workers <= ws <= self.config.max_workers

    def reset(self) -> None:
        self.workers.clear()


ELASTIC_REGISTRY: dict[str, type[ElasticCoordinator]] = {"default": ElasticCoordinator}
