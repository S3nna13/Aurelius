"""Structured security audit log with secret redaction.

Records security-relevant events (policy violations, auth failures, sandbox
escapes, etc.) as structured JSON to Python's logging system under the logger
name ``aurelius.security.audit``. Events are also retained in a bounded
in-memory ring buffer for diagnostic ``recent_events`` queries.

Pattern inspired by the CERBERUS Layer4 detection/monitoring taxonomy
(Heavens_Gate) where every defensive surface emits a uniform, machine-parseable
event envelope that downstream SOC stages can consume.

Pure stdlib.
"""

from __future__ import annotations

import enum
import json
import logging
import re
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Deque, List


AUDIT_LOGGER_NAME = "aurelius.security.audit"

_MAX_BUFFER = 10_000


class AuditLevel(str, enum.Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AuditCategory(str, enum.Enum):
    AUTH = "AUTH"
    POLICY = "POLICY"
    SANDBOX = "SANDBOX"
    NETWORK = "NETWORK"
    DATA_ACCESS = "DATA_ACCESS"
    TOOL_CALL = "TOOL_CALL"
    ADMIN = "ADMIN"


_REDACT_PATTERNS = [
    re.compile(r"Bearer\s+\S+"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)password[=:]\S+"),
    re.compile(r"(?i)api[_-]?key[=:]\S+"),
]

_REDACTED = "[REDACTED]"


def _redact_text(text: str) -> str:
    """Scrub common secret patterns. Never raises."""
    try:
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        for pat in _REDACT_PATTERNS:
            text = pat.sub(_REDACTED, text)
        return text
    except Exception:
        return _REDACTED


@dataclass(frozen=True)
class AuditEvent:
    event_id: str
    category: AuditCategory
    level: AuditLevel
    principal: str
    action: str
    target: str
    outcome: str
    detail: str = ""
    timestamp: float = field(default_factory=time.time)

    @staticmethod
    def create(
        category: AuditCategory,
        level: AuditLevel,
        principal: str,
        action: str,
        target: str,
        outcome: str,
        detail: str = "",
    ) -> "AuditEvent":
        return AuditEvent(
            event_id=str(uuid.uuid4()),
            category=category,
            level=level,
            principal=_redact_text(principal),
            action=action,
            target=target,
            outcome=outcome,
            detail=_redact_text(detail),
            timestamp=time.time(),
        )

    def to_json(self) -> str:
        payload = {
            "event_id": self.event_id,
            "category": self.category.value,
            "level": self.level.value,
            "principal": self.principal,
            "action": self.action,
            "target": self.target,
            "outcome": self.outcome,
            "detail": self.detail,
            "timestamp": self.timestamp,
        }
        return json.dumps(payload, sort_keys=True)


class AuditLogger:
    """In-process structured audit sink with bounded retention."""

    def __init__(self, buffer_size: int = _MAX_BUFFER) -> None:
        self._buffer_size = int(buffer_size)
        self._buffer: Deque[AuditEvent] = deque(maxlen=self._buffer_size)
        self._logger = logging.getLogger(AUDIT_LOGGER_NAME)

    def _redact(self, text: str) -> str:
        return _redact_text(text)

    def log(self, event: AuditEvent) -> None:
        # Re-scrub principal/detail as a belt-and-braces guard in case the
        # event was hand-constructed rather than via ``AuditEvent.create``.
        safe = AuditEvent(
            event_id=event.event_id,
            category=event.category,
            level=event.level,
            principal=self._redact(event.principal),
            action=event.action,
            target=event.target,
            outcome=event.outcome,
            detail=self._redact(event.detail),
            timestamp=event.timestamp,
        )
        self._buffer.append(safe)
        py_level = {
            AuditLevel.INFO: logging.INFO,
            AuditLevel.WARNING: logging.WARNING,
            AuditLevel.CRITICAL: logging.CRITICAL,
        }.get(safe.level, logging.INFO)
        try:
            self._logger.log(py_level, safe.to_json())
        except Exception:
            # Logging must never propagate failures from the audit path.
            pass

    def recent_events(self, n: int = 100) -> List[AuditEvent]:
        if n <= 0:
            return []
        if n >= len(self._buffer):
            return list(self._buffer)
        return list(self._buffer)[-n:]

    def clear(self) -> None:
        self._buffer.clear()


AUDIT_LOGGER = AuditLogger()


__all__ = [
    "AuditLevel",
    "AuditCategory",
    "AuditEvent",
    "AuditLogger",
    "AUDIT_LOGGER",
    "AUDIT_LOGGER_NAME",
]
