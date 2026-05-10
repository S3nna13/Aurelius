"""Cron job model — scheduled agent tasks with delivery and skill attachment.

Inspired by Hermes Agent's cron system: supports one-shot, interval, and cron
expression schedules, with configurable delivery targets and skill bindings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SCHEDULE_KINDS = ("at", "every", "cron")
_DELIVERY_TARGETS = ("origin", "none", "webhook")


@dataclass
class CronJob:
    id: str
    schedule_kind: str  # at, every, cron
    schedule_expr: str  # ISO 8601, interval like "30m", cron like "0 9 * * *"
    message: str
    agent: str = "main"
    skills: list[str] = field(default_factory=list)
    delivery: str = "origin"  # origin, none, webhook
    webhook_url: str = ""
    model: str = ""
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schedule_kind not in _SCHEDULE_KINDS:
            raise ValueError(
                f"schedule_kind must be one of {_SCHEDULE_KINDS}, got {self.schedule_kind!r}"
            )
        if self.delivery not in _DELIVERY_TARGETS:
            raise ValueError(
                f"delivery must be one of {_DELIVERY_TARGETS}, got {self.delivery!r}"
            )


class CronJobStore:
    """Persistent JSON-backed cron job store."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else Path.home() / ".aurelius" / "cron" / "jobs.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, CronJob] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            for item in raw:
                job = CronJob(**item)
                self._jobs[job.id] = job

    def _save(self) -> None:
        self._path.write_text(
            json.dumps(
                [vars(j) for j in self._jobs.values()],
                indent=2,
                sort_keys=True,
                default=str,
            ),
            encoding="utf-8",
        )

    def add(self, job: CronJob) -> None:
        self._jobs[job.id] = job
        self._save()

    def get(self, job_id: str) -> CronJob | None:
        return self._jobs.get(job_id)

    def remove(self, job_id: str) -> bool:
        removed = self._jobs.pop(job_id, None)
        if removed:
            self._save()
        return removed is not None

    def toggle(self, job_id: str) -> bool | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        job.enabled = not job.enabled
        self._save()
        return job.enabled

    def list(self) -> list[CronJob]:
        return list(self._jobs.values())

    @property
    def stats(self) -> dict[str, Any]:
        total = len(self._jobs)
        enabled = sum(1 for j in self._jobs.values() if j.enabled)
        return {
            "total": total,
            "enabled": enabled,
            "disabled": total - enabled,
            "path": str(self._path),
        }


__all__ = [
    "CronJob",
    "CronJobStore",
]
