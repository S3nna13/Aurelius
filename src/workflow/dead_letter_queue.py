"""Workflow dead-letter queue for failed step handling."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class DeadLetteredStep:
    """A workflow step that failed and was moved to DLQ."""

    step_id: str
    workflow_id: str
    error: str
    payload: dict
    failed_at: str = ""
    retry_count: int = 0

    def __post_init__(self) -> None:
        if not self.failed_at:
            self.failed_at = datetime.now(UTC).isoformat()


@dataclass
class DeadLetterQueue:
    """Holds failed workflow steps for later inspection/retry."""

    _queue: list[DeadLetteredStep] = field(default_factory=list, repr=False)

    def enqueue(self, step: DeadLetteredStep) -> None:
        self._queue.append(step)

    def requeue(self, workflow_id: str) -> list[DeadLetteredStep]:
        requeued = [s for s in self._queue if s.workflow_id == workflow_id]
        self._queue = [s for s in self._queue if s.workflow_id != workflow_id]
        for s in requeued:
            s.retry_count += 1
        return requeued

    def pending(self) -> int:
        return len(self._queue)

    def clear(self) -> None:
        self._queue.clear()


DEAD_LETTER_QUEUE = DeadLetterQueue()
