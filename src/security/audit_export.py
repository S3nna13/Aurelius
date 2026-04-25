"""Audit log exporter for compliance reporting."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AuditEntry:
    timestamp: str
    actor: str
    action: str
    resource: str
    result: str  # success, failure, denied
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "result": self.result,
            "details": self.details,
        }


@dataclass
class AuditExporter:
    entries: list[AuditEntry] = field(default_factory=list, repr=False)

    def record(self, entry: AuditEntry) -> None:
        self.entries.append(entry)

    def export_json(self, path: str) -> None:
        data = [e.to_dict() for e in self.entries]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def export_csv(self, path: str) -> None:
        import csv
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "actor", "action", "resource", "result"])
            writer.writeheader()
            for e in self.entries:
                writer.writerow({"timestamp": e.timestamp, "actor": e.actor,
                                 "action": e.action, "resource": e.resource, "result": e.result})

    def filter(self, actor: str | None = None, action: str | None = None,
               result: str | None = None) -> list[AuditEntry]:
        filtered = self.entries[:]
        if actor:
            filtered = [e for e in filtered if e.actor == actor]
        if action:
            filtered = [e for e in filtered if e.action == action]
        if result:
            filtered = [e for e in filtered if e.result == result]
        return filtered

    def clear(self) -> None:
        self.entries.clear()


AUDIT_EXPORTER = AuditExporter()