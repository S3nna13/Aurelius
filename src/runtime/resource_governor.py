from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum


@dataclass
class ResourceLimit:
    max_memory_mb: float = 8192.0
    max_cpu_percent: float = 80.0
    max_concurrent_requests: int = 100
    max_queue_depth: int = 1000


@dataclass(frozen=True)
class ResourceSnapshot:
    timestamp_s: float
    memory_mb: float
    cpu_percent: float
    active_requests: int
    queue_depth: int


class GovernorDecision(Enum):
    ALLOW = "allow"
    THROTTLE = "throttle"
    REJECT = "reject"


class ResourceGovernor:
    def __init__(self, limits: ResourceLimit | None = None):
        self.limits = limits or ResourceLimit()
        self._history: deque[ResourceSnapshot] = deque(maxlen=1000)
        self._latest: ResourceSnapshot | None = None

    def update(
        self,
        memory_mb: float,
        cpu_percent: float,
        active_requests: int,
        queue_depth: int,
    ) -> None:
        snap = ResourceSnapshot(
            timestamp_s=time.perf_counter(),
            memory_mb=memory_mb,
            cpu_percent=cpu_percent,
            active_requests=active_requests,
            queue_depth=queue_depth,
        )
        self._latest = snap
        self._history.append(snap)

    def decide(self) -> GovernorDecision:
        if self._latest is None:
            return GovernorDecision.ALLOW
        s = self._latest
        if s.memory_mb > self.limits.max_memory_mb or s.queue_depth > self.limits.max_queue_depth:
            return GovernorDecision.REJECT
        if (
            s.cpu_percent > self.limits.max_cpu_percent
            or s.active_requests > self.limits.max_concurrent_requests
        ):
            return GovernorDecision.THROTTLE
        return GovernorDecision.ALLOW

    def headroom(self) -> dict:
        if self._latest is None:
            return {
                "memory": 100.0,
                "cpu": 100.0,
                "active_requests": 100.0,
                "queue_depth": 100.0,
            }
        s = self._latest
        lim = self.limits
        return {
            "memory": max(0.0, (lim.max_memory_mb - s.memory_mb) / lim.max_memory_mb * 100.0),
            "cpu": max(0.0, (lim.max_cpu_percent - s.cpu_percent) / lim.max_cpu_percent * 100.0),
            "active_requests": max(
                0.0,
                (lim.max_concurrent_requests - s.active_requests)
                / lim.max_concurrent_requests
                * 100.0,
            ),
            "queue_depth": max(
                0.0,
                (lim.max_queue_depth - s.queue_depth) / lim.max_queue_depth * 100.0,
            ),
        }

    def is_overloaded(self) -> bool:
        return self.decide() != GovernorDecision.ALLOW

    def history(self, n: int = 10) -> list[ResourceSnapshot]:
        if n <= 0:
            return []
        items = list(self._history)
        return items[-n:]


RESOURCE_GOVERNOR_REGISTRY: dict[str, type[ResourceGovernor]] = {"default": ResourceGovernor}
