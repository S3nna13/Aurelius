import heapq
import time
from dataclasses import dataclass, field
from enum import IntEnum


class QueuePriority(IntEnum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


@dataclass
class QueuedRequest:
    request_id: str
    priority: QueuePriority = QueuePriority.NORMAL
    payload: dict = field(default_factory=dict)
    enqueue_time: float = field(default_factory=time.monotonic)
    deadline: float | None = None


class RequestQueue:
    def __init__(self, maxsize: int = 1000):
        self.maxsize = maxsize
        self._heap: list[tuple] = []

    def enqueue(self, req: QueuedRequest) -> bool:
        if len(self._heap) >= self.maxsize:
            return False
        heapq.heappush(self._heap, (req.priority, req.enqueue_time, req))
        return True

    def dequeue(self) -> QueuedRequest | None:
        if not self._heap:
            return None
        _, _, req = heapq.heappop(self._heap)
        return req

    def peek(self) -> QueuedRequest | None:
        if not self._heap:
            return None
        return self._heap[0][2]

    def drop_expired(self) -> int:
        now = time.monotonic()
        before = len(self._heap)
        self._heap = [
            entry for entry in self._heap
            if entry[2].deadline is None or entry[2].deadline > now
        ]
        heapq.heapify(self._heap)
        return before - len(self._heap)

    def size(self) -> int:
        return len(self._heap)

    def is_empty(self) -> bool:
        return len(self._heap) == 0


REQUEST_QUEUE_REGISTRY: dict[str, type[RequestQueue]] = {"default": RequestQueue}
