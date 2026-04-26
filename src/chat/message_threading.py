"""Message threading: reply chains, branching, thread merging."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import StrEnum


class ThreadStatus(StrEnum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    ARCHIVED = "archived"


@dataclass
class Thread:
    parent_id: str | None = None
    status: ThreadStatus = ThreadStatus.ACTIVE
    message_ids: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    thread_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


class MessageThreader:
    def __init__(self) -> None:
        self._threads: dict[str, Thread] = {}

    def create_thread(self, parent_id: str | None = None, **metadata) -> Thread:
        t = Thread(parent_id=parent_id, metadata=dict(metadata))
        self._threads[t.thread_id] = t
        return t

    def add_message(self, thread_id: str, message_id: str) -> bool:
        thread = self._threads.get(thread_id)
        if thread is None:
            return False
        thread.message_ids.append(message_id)
        return True

    def reply_to(self, parent_thread_id: str, message_id: str) -> Thread:
        child = Thread(parent_id=parent_thread_id)
        child.message_ids.append(message_id)
        self._threads[child.thread_id] = child
        return child

    def get_thread(self, thread_id: str) -> Thread | None:
        return self._threads.get(thread_id)

    def children(self, thread_id: str) -> list[Thread]:
        return [t for t in self._threads.values() if t.parent_id == thread_id]

    def resolve(self, thread_id: str) -> bool:
        thread = self._threads.get(thread_id)
        if thread is None:
            return False
        thread.status = ThreadStatus.RESOLVED
        return True

    def archive(self, thread_id: str) -> bool:
        thread = self._threads.get(thread_id)
        if thread is None:
            return False
        thread.status = ThreadStatus.ARCHIVED
        return True

    def active_threads(self) -> list[Thread]:
        return [t for t in self._threads.values() if t.status == ThreadStatus.ACTIVE]

    def message_count(self) -> int:
        return sum(len(t.message_ids) for t in self._threads.values())
