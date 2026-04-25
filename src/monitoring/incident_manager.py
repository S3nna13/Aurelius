"""Incident manager for monitoring and alerting lifecycle."""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class IncidentError(ValueError):
    """Raised on incident management errors."""


@dataclass
class Incident:
    id: str
    title: str
    severity: str
    service: str
    status: str = "open"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    acknowledged_at: str | None = None
    resolved_at: str | None = None
    runbook_url: str | None = None
    notes: list[str] = field(default_factory=list)


class IncidentManager:
    """Track incidents from open → acknowledged → resolved."""

    SEVERITIES = ("low", "medium", "high", "critical")
    STATUSES = ("open", "acknowledged", "resolved")

    def __init__(self, persistence_path: str | None = None) -> None:
        self._incidents: dict[str, Incident] = {}
        self._lock = threading.RLock()
        self._persistence_path = Path(persistence_path) if persistence_path else None
        if self._persistence_path and self._persistence_path.exists():
            self._load()

    def _validate_id(self, incident_id: str) -> None:
        if not isinstance(incident_id, str) or not re.fullmatch(r"[A-Za-z0-9]{1,64}", incident_id):
            raise IncidentError("incident_id must be alphanumeric, 1-64 chars")

    def _validate_severity(self, severity: str) -> None:
        if severity not in self.SEVERITIES:
            raise IncidentError(f"severity must be one of {self.SEVERITIES}")

    def _persist(self) -> None:
        if not self._persistence_path:
            return
        data = []
        for inc in self._incidents.values():
            data.append(
                {
                    "id": inc.id,
                    "title": inc.title,
                    "severity": inc.severity,
                    "service": inc.service,
                    "status": inc.status,
                    "created_at": inc.created_at,
                    "acknowledged_at": inc.acknowledged_at,
                    "resolved_at": inc.resolved_at,
                    "runbook_url": inc.runbook_url,
                    "notes": inc.notes,
                }
            )
        self._persistence_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load(self) -> None:
        raw = json.loads(self._persistence_path.read_text(encoding="utf-8"))
        for obj in raw:
            inc = Incident(
                id=obj["id"],
                title=obj["title"],
                severity=obj["severity"],
                service=obj["service"],
                status=obj["status"],
                created_at=obj["created_at"],
                acknowledged_at=obj.get("acknowledged_at"),
                resolved_at=obj.get("resolved_at"),
                runbook_url=obj.get("runbook_url"),
                notes=list(obj.get("notes", [])),
            )
            self._incidents[inc.id] = inc

    def create(
        self, title: str, severity: str, service: str, runbook_url: str | None = None
    ) -> str:
        """Create a new incident and return its ID."""
        self._validate_severity(severity)
        incident_id = os.urandom(4).hex()
        with self._lock:
            self._incidents[incident_id] = Incident(
                id=incident_id,
                title=str(title),
                severity=severity,
                service=str(service),
                runbook_url=runbook_url,
            )
            self._persist()
        return incident_id

    def acknowledge(self, incident_id: str, notes: str | None = None) -> None:
        """Acknowledge an open incident."""
        self._validate_id(incident_id)
        with self._lock:
            inc = self._incidents.get(incident_id)
            if not inc:
                raise IncidentError(f"incident not found: {incident_id}")
            if inc.status != "open":
                raise IncidentError(f"cannot acknowledge incident in status {inc.status}")
            inc.status = "acknowledged"
            inc.acknowledged_at = datetime.now(timezone.utc).isoformat()
            if notes:
                inc.notes.append(str(notes))
            self._persist()

    def resolve(self, incident_id: str, notes: str | None = None) -> None:
        """Resolve an acknowledged incident."""
        self._validate_id(incident_id)
        with self._lock:
            inc = self._incidents.get(incident_id)
            if not inc:
                raise IncidentError(f"incident not found: {incident_id}")
            if inc.status != "acknowledged":
                raise IncidentError(f"cannot resolve incident in status {inc.status}")
            inc.status = "resolved"
            inc.resolved_at = datetime.now(timezone.utc).isoformat()
            if notes:
                inc.notes.append(str(notes))
            self._persist()

    def escalate(self, incident_id: str) -> bool:
        """Bump severity up one level. Returns False if already critical."""
        self._validate_id(incident_id)
        with self._lock:
            inc = self._incidents.get(incident_id)
            if not inc:
                raise IncidentError(f"incident not found: {incident_id}")
            idx = self.SEVERITIES.index(inc.severity)
            if idx + 1 >= len(self.SEVERITIES):
                return False
            inc.severity = self.SEVERITIES[idx + 1]
            self._persist()
        return True

    def list_open(self) -> list[Incident]:
        """Return all open incidents."""
        with self._lock:
            return [inc for inc in self._incidents.values() if inc.status == "open"]

    def list_by_service(self, service: str) -> list[Incident]:
        """Return incidents filtered by service."""
        with self._lock:
            return [inc for inc in self._incidents.values() if inc.service == service]

    def get(self, incident_id: str) -> Incident:
        """Get a single incident by ID."""
        self._validate_id(incident_id)
        with self._lock:
            inc = self._incidents.get(incident_id)
            if not inc:
                raise IncidentError(f"incident not found: {incident_id}")
            return inc
