"""Immutable append-only audit trail for security events.

Each event is assigned a short UUID hex prefix as its identifier.  The trail
supports filtered queries, JSONL export, integrity hashing, and bounded
capacity.  Pure stdlib — no third-party dependencies.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# AuditEvent
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrailAuditEvent:
    """A single immutable audit event recorded in an :class:`AuditTrail`."""

    actor: str
    action: str
    resource: str
    outcome: str
    timestamp: float
    metadata: Dict
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "outcome": self.outcome,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


# Keep public name as AuditEvent per spec, but alias to avoid collision
# with the existing audit_logger.AuditEvent when both are imported
AuditEvent = TrailAuditEvent


# ---------------------------------------------------------------------------
# AuditTrail
# ---------------------------------------------------------------------------

class AuditTrail:
    """Append-only, bounded audit trail for security events."""

    def __init__(self, max_events: int = 10_000) -> None:
        self._max_events = max_events
        self._events: List[TrailAuditEvent] = []

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def log(
        self,
        actor: str,
        action: str,
        resource: str,
        outcome: str = "success",
        metadata: Optional[Dict] = None,
    ) -> TrailAuditEvent:
        """Record a new audit event and return it."""
        event = TrailAuditEvent(
            actor=actor,
            action=action,
            resource=resource,
            outcome=outcome,
            timestamp=time.time(),
            metadata=metadata if metadata is not None else {},
        )
        if len(self._events) >= self._max_events:
            # Drop the oldest event (FIFO eviction)
            self._events.pop(0)
        self._events.append(event)
        return event

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def query(
        self,
        actor: Optional[str] = None,
        action: Optional[str] = None,
        outcome: Optional[str] = None,
    ) -> List[TrailAuditEvent]:
        """Filter events by any combination of *actor*, *action*, *outcome*.

        A ``None`` value acts as a wildcard (matches any value).
        """
        result = []
        for ev in self._events:
            if actor is not None and ev.actor != actor:
                continue
            if action is not None and ev.action != action:
                continue
            if outcome is not None and ev.outcome != outcome:
                continue
            result.append(ev)
        return result

    def since(self, timestamp: float) -> List[TrailAuditEvent]:
        """Return all events with ``timestamp >= *timestamp*``."""
        return [ev for ev in self._events if ev.timestamp >= timestamp]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_jsonl(self, path: str) -> int:
        """Write each event as a JSON line to *path*.  Returns the count written."""
        count = 0
        with open(path, "w", encoding="utf-8") as fh:
            for ev in self._events:
                fh.write(json.dumps(ev.to_dict()) + "\n")
                count += 1
        return count

    # ------------------------------------------------------------------
    # Integrity
    # ------------------------------------------------------------------

    def integrity_hash(self) -> str:
        """Return a SHA-256 hex digest of all event IDs in insertion order."""
        concatenated = "".join(ev.event_id for ev in self._events)
        return hashlib.sha256(concatenated.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._events)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

AUDIT_TRAIL_REGISTRY: dict = {
    "default": AuditTrail,
}
