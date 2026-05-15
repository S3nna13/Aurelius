"""Aurelius v2 Skill Telemetry — usage tracking, success rates, latency statistics."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class TelemetryEvent:
    """A single skill execution event."""
    success: bool
    runtime_ms: int = 0
    tools_used: list[str] = field(default_factory=list)
    permissions_used: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    error: str = ""


@dataclass
class SkillTelemetry:
    """Telemetry for a single skill."""
    skill_id: str
    use_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_runtime_ms: int = 0
    tools_used: dict[str, int] = field(default_factory=dict)
    permissions_used: dict[str, int] = field(default_factory=dict)
    user_approval_count: int = 0
    rollback_count: int = 0
    last_used: str = ""
    daies_pass: bool = True
    events: list[TelemetryEvent] = field(default_factory=list)

    def record(self, event: TelemetryEvent) -> None:
        """Record a telemetry event."""
        self.use_count += 1
        if event.success:
            self.success_count += 1
        else:
            self.failure_count += 1
        self.total_runtime_ms += event.runtime_ms
        self.last_used = event.timestamp
        for t in event.tools_used:
            self.tools_used[t] = self.tools_used.get(t, 0) + 1
        for p in event.permissions_used:
            self.permissions_used[p] = self.permissions_used.get(p, 0) + 1
        self.events.append(event)
        # Keep only last 1000 events
        if len(self.events) > 1000:
            self.events = self.events[-1000:]

    @property
    def success_rate(self) -> float:
        if self.use_count == 0:
            return 0.0
        return self.success_count / self.use_count

    @property
    def average_runtime_ms(self) -> float:
        if self.use_count == 0:
            return 0.0
        return self.total_runtime_ms / self.use_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "use_count": self.use_count,
            "success_rate": round(self.success_rate, 4),
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "average_runtime_ms": round(self.average_runtime_ms, 2),
            "tools_used": dict(self.tools_used),
            "permissions_used": dict(self.permissions_used),
            "user_approval_count": self.user_approval_count,
            "rollback_count": self.rollback_count,
            "last_used": self.last_used,
            "daies_pass": self.daies_pass,
        }
