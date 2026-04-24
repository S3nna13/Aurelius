"""Memory writer module for the Aurelius agent surface.

Writes agent observations and decisions to persistent in-process memory,
supporting filtered recall and lightweight analytics.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MemoryType(Enum):
    OBSERVATION = "observation"
    DECISION    = "decision"
    REFLECTION  = "reflection"
    FACT        = "fact"
    PLAN        = "plan"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MemoryRecord:
    """A single immutable entry in the agent memory store."""
    record_id:   str
    memory_type: MemoryType
    content:     str
    importance:  float
    timestamp:   float
    tags:        list[str] = field(default_factory=list)

    def __new__(cls, *args, **kwargs):  # noqa: D102 – handled via __init__ below
        return object.__new__(cls)

    @classmethod
    def create(
        cls,
        memory_type: MemoryType,
        content: str,
        importance: float,
        timestamp: float,
        tags: list[str] | None = None,
        record_id: str | None = None,
    ) -> "MemoryRecord":
        """Factory that auto-generates record_id when not provided."""
        rid  = record_id if record_id is not None else uuid.uuid4().hex[:10]
        tgs  = tags if tags is not None else []
        return cls(
            record_id=rid,
            memory_type=memory_type,
            content=content,
            importance=importance,
            timestamp=timestamp,
            tags=tgs,
        )


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class MemoryWriter:
    """Write and recall agent memories with tag/type/importance filtering."""

    def __init__(self, max_records: int = 5000) -> None:
        self._max_records: int = max_records
        self._records: list[MemoryRecord] = []

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def write(
        self,
        memory_type: MemoryType,
        content: str,
        importance: float = 0.5,
        tags: list[str] | None = None,
    ) -> MemoryRecord:
        """Create and store a new MemoryRecord.

        Raises ValueError when max_records has been reached.
        """
        if len(self._records) >= self._max_records:
            raise ValueError(
                f"Memory capacity reached ({self._max_records} records). "
                "Call forget() to free space before writing."
            )
        record = MemoryRecord.create(
            memory_type=memory_type,
            content=content,
            importance=importance,
            timestamp=time.monotonic(),
            tags=tags,
        )
        self._records.append(record)
        return record

    def write_observation(self, content: str, **kwargs: Any) -> MemoryRecord:
        """Convenience wrapper: write a OBSERVATION-typed record."""
        return self.write(MemoryType.OBSERVATION, content, **kwargs)

    def write_decision(self, content: str, **kwargs: Any) -> MemoryRecord:
        """Convenience wrapper: write a DECISION-typed record."""
        return self.write(MemoryType.DECISION, content, **kwargs)

    # ------------------------------------------------------------------
    # Recall & forget
    # ------------------------------------------------------------------

    def recall(
        self,
        tags: list[str] | None = None,
        memory_type: MemoryType | None = None,
        min_importance: float = 0.0,
    ) -> list[MemoryRecord]:
        """Return filtered records sorted by importance descending.

        All provided criteria are combined with AND logic.
        """
        results: list[MemoryRecord] = []
        for rec in self._records:
            if memory_type is not None and rec.memory_type is not memory_type:
                continue
            if rec.importance < min_importance:
                continue
            if tags is not None:
                if not all(t in rec.tags for t in tags):
                    continue
            results.append(rec)
        results.sort(key=lambda r: r.importance, reverse=True)
        return results

    def forget(self, record_id: str) -> bool:
        """Remove the record with the given id. Returns True if found."""
        for i, rec in enumerate(self._records):
            if rec.record_id == record_id:
                del self._records[i]
                return True
        return False

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return summary statistics about stored memories."""
        by_type: dict[str, int] = {mt.name: 0 for mt in MemoryType}
        total_importance = 0.0
        for rec in self._records:
            by_type[rec.memory_type.name] += 1
            total_importance += rec.importance
        mean_importance = total_importance / len(self._records) if self._records else 0.0
        return {
            "total": len(self._records),
            "by_type": by_type,
            "mean_importance": mean_importance,
        }

    def __len__(self) -> int:
        return len(self._records)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MEMORY_WRITER_REGISTRY: dict[str, Any] = {
    "default": MemoryWriter,
}
