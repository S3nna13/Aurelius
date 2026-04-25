"""Priority-based memory eviction policy."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class Priority(Enum):
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


@dataclass
class MemoryItem:
    key: str
    value: str
    priority: Priority = Priority.LOW
    access_count: int = 0


@dataclass
class PriorityMemoryStore:
    """Memory store with priority-based eviction."""

    max_size: int = 1000
    _items: dict[str, MemoryItem] = field(default_factory=dict, repr=False)

    def put(self, item: MemoryItem) -> None:
        if len(self._items) >= self.max_size and item.key not in self._items:
            self._evict_lowest_priority()
        self._items[item.key] = item

    def get(self, key: str) -> str | None:
        item = self._items.get(key)
        if item is None:
            return None
        item.access_count += 1
        return item.value

    def _evict_lowest_priority(self) -> None:
        sorted_items = sorted(
            self._items.values(),
            key=lambda x: (x.priority.value, x.access_count),
        )
        evict = sorted_items[0]
        del self._items[evict.key]

    def size(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        self._items.clear()


PRIORITY_MEMORY = PriorityMemoryStore()