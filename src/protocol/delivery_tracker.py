"""Delivery tracker for protocol-level message reliability.

Tracks send/ack state with monotonic timeouts and retry budgets.
Fail closed: unacknowledged messages after max retries are surfaced
as failed deliveries, not silently dropped.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DeliveryStatus(Enum):
    PENDING = "pending"
    ACKED = "acked"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class DeliveryRecord:
    message_id: str
    recipient: str
    status: DeliveryStatus = DeliveryStatus.PENDING
    attempts: int = 0
    created_at: float = field(default_factory=time.monotonic)
    last_attempt_at: float | None = None
    acked_at: float | None = None
    payload_preview: str = ""


@dataclass
class DeliveryConfig:
    max_retries: int = 3
    timeout_seconds: float = 30.0
    backoff_base_seconds: float = 1.0


class DeliveryTracker:
    """Tracks message delivery lifecycle and exposes retry decisions.

    This is a pure state tracker; actual transport retries are the
    responsibility of the caller.
    """

    def __init__(self, config: DeliveryConfig | None = None) -> None:
        self._config = config or DeliveryConfig()
        self._records: dict[str, DeliveryRecord] = {}

    def register(
        self,
        message_id: str,
        recipient: str,
        payload_preview: str = "",
    ) -> DeliveryRecord:
        """Record a new outbound message as pending."""
        record = DeliveryRecord(
            message_id=message_id,
            recipient=recipient,
            payload_preview=payload_preview[:200],
        )
        self._records[message_id] = record
        return record

    def attempt(self, message_id: str) -> DeliveryRecord | None:
        """Increment attempt counter and timestamp. Returns record or None."""
        record = self._records.get(message_id)
        if record is None:
            return None
        record.attempts += 1
        record.last_attempt_at = time.monotonic()
        return record

    def ack(self, message_id: str) -> bool:
        """Mark *message_id* as successfully acknowledged."""
        record = self._records.get(message_id)
        if record is None:
            return False
        record.status = DeliveryStatus.ACKED
        record.acked_at = time.monotonic()
        return True

    def is_expired(self, message_id: str) -> bool:
        """True if the record has exceeded timeout without ack."""
        record = self._records.get(message_id)
        if record is None or record.status != DeliveryStatus.PENDING:
            return False
        elapsed = time.monotonic() - (record.last_attempt_at or record.created_at)
        return elapsed > self._config.timeout_seconds

    def should_retry(self, message_id: str) -> bool:
        """True if the message is pending, expired, and under retry budget."""
        record = self._records.get(message_id)
        if record is None:
            return False
        if record.status != DeliveryStatus.PENDING:
            return False
        if record.attempts > self._config.max_retries:
            return False
        return self.is_expired(message_id)

    def mark_failed(self, message_id: str) -> bool:
        """Force a record to FAILED. Returns True if it existed."""
        record = self._records.get(message_id)
        if record is None:
            return False
        record.status = DeliveryStatus.FAILED
        return True

    def mark_expired(self, message_id: str) -> bool:
        """Force a record to EXPIRED. Returns True if it existed."""
        record = self._records.get(message_id)
        if record is None:
            return False
        record.status = DeliveryStatus.EXPIRED
        return True

    def get_record(self, message_id: str) -> DeliveryRecord | None:
        return self._records.get(message_id)

    def pending_ids(self) -> list[str]:
        """Return message_ids currently in PENDING status."""
        return [
            mid
            for mid, r in self._records.items()
            if r.status == DeliveryStatus.PENDING
        ]

    def failed_ids(self) -> list[str]:
        """Return message_ids in FAILED or EXPIRED status."""
        return [
            mid
            for mid, r in self._records.items()
            if r.status in (DeliveryStatus.FAILED, DeliveryStatus.EXPIRED)
        ]

    def stats(self) -> dict[str, int]:
        """Aggregate counts by status."""
        counts: dict[str, int] = {
            "pending": 0,
            "acked": 0,
            "failed": 0,
            "expired": 0,
        }
        for r in self._records.values():
            counts[r.status.value] = counts.get(r.status.value, 0) + 1
        return counts

    def reset(self) -> None:
        """Clear all records. Useful in testing."""
        self._records.clear()


# Module-level registry
DELIVERY_TRACKER_REGISTRY: dict[str, DeliveryTracker] = {}
DEFAULT_DELIVERY_TRACKER = DeliveryTracker()
DELIVERY_TRACKER_REGISTRY["default"] = DEFAULT_DELIVERY_TRACKER
