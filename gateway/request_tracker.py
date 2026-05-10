"""In-flight request tracker for serving observability."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class RequestInfo:
    id: str = ""
    method: str = ""
    path: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    status: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def duration_ms(self) -> float:
        if self.end_time == 0.0:
            return 0.0
        return (self.end_time - self.start_time) * 1000


@dataclass
class RequestTracker:
    _active: dict[str, RequestInfo] = field(default_factory=dict, repr=False)
    _completed: list[RequestInfo] = field(default_factory=list, repr=False)
    max_completed: int = 1000

    def start(self, method: str = "GET", path: str = "/") -> str:
        req_id = uuid4().hex[:12]
        info = RequestInfo(id=req_id, method=method, path=path, start_time=time.monotonic())
        self._active[req_id] = info
        return req_id

    def finish(self, req_id: str, status: int = 200, **meta: Any) -> None:
        info = self._active.pop(req_id, None)
        if info is None:
            return
        info.end_time = time.monotonic()
        info.status = status
        info.metadata.update(meta)
        self._completed.append(info)
        if len(self._completed) > self.max_completed:
            self._completed = self._completed[-self.max_completed :]

    def active_count(self) -> int:
        return len(self._active)

    def active_requests(self) -> list[RequestInfo]:
        return list(self._active.values())

    def completed(self) -> list[RequestInfo]:
        return list(self._completed)

    def latency_stats(self) -> dict[str, float]:
        if not self._completed:
            return {"avg_ms": 0.0, "p50_ms": 0.0, "p99_ms": 0.0}
        durations = sorted(r.duration_ms() for r in self._completed)
        n = len(durations)
        return {
            "avg_ms": sum(durations) / n,
            "p50_ms": durations[n // 2],
            "p99_ms": durations[int(n * 0.99)],
        }


REQUEST_TRACKER = RequestTracker()
