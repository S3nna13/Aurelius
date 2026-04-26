"""Data provenance ledger for Aurelius training samples."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

SourceType = Literal[
    "internal_logs",
    "approved_corpus",
    "synthetic_trace",
    "web_scrape",
    "human_annotated",
]
LicenseTier = Literal["open", "restricted", "commercial", "unknown"]
SplitType = Literal["train", "eval", "test"]
TaskType = Literal["reasoning", "code", "tool_calling", "long_context", "general", "alignment"]

_ALLOWED_SOURCES = {
    "internal_logs",
    "approved_corpus",
    "synthetic_trace",
    "web_scrape",
    "human_annotated",
}
_ALLOWED_LICENSES = {"open", "restricted", "commercial", "unknown"}
_ALLOWED_SPLITS = {"train", "eval", "test"}
_ALLOWED_TASKS = {
    "reasoning",
    "code",
    "tool_calling",
    "long_context",
    "general",
    "alignment",
}


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class ProvenanceRecord:
    """Required provenance metadata for a training sample."""

    source_type: SourceType
    license_tier: LicenseTier
    origin_model: str
    task_type: TaskType
    split: SplitType
    sample_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: datetime = field(default_factory=_utc_now)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source_type": self.source_type,
            "license_tier": self.license_tier,
            "origin_model": self.origin_model,
            "timestamp": self.timestamp.isoformat(),
            "task_type": self.task_type,
            "split": self.split,
            "notes": self.notes,
        }


class ProvenanceValidator:
    """Validate provenance records for schema and license compliance."""

    def __init__(self, allow_restricted: bool = False) -> None:
        self._allow_restricted = allow_restricted

    def validate(self, record: ProvenanceRecord) -> bool:
        try:
            self._check(record)
        except ValueError:
            return False
        return True

    def validate_batch(
        self, records: list[ProvenanceRecord]
    ) -> tuple[list[ProvenanceRecord], list[str]]:
        valid: list[ProvenanceRecord] = []
        errors: list[str] = []
        for rec in records:
            try:
                self._check(rec)
                valid.append(rec)
            except ValueError as exc:
                errors.append(f"{rec.sample_id}: {exc}")
        return valid, errors

    def _check(self, record: ProvenanceRecord) -> None:
        if not record.sample_id:
            raise ValueError("sample_id must be non-empty")
        if record.source_type not in _ALLOWED_SOURCES:
            raise ValueError(f"invalid source_type: {record.source_type}")
        if record.license_tier not in _ALLOWED_LICENSES:
            raise ValueError(f"invalid license_tier: {record.license_tier}")
        if not record.origin_model:
            raise ValueError("origin_model must be non-empty")
        if record.task_type not in _ALLOWED_TASKS:
            raise ValueError(f"invalid task_type: {record.task_type}")
        if record.split not in _ALLOWED_SPLITS:
            raise ValueError(f"invalid split: {record.split}")
        if record.license_tier == "unknown":
            raise ValueError("license_tier 'unknown' forbids usage")
        if record.license_tier == "restricted" and not self._allow_restricted:
            raise ValueError("restricted license requires explicit allowance")


class ProvenanceLedger:
    """In-memory ledger of provenance records with filtering and export."""

    def __init__(self) -> None:
        self._records: dict[str, ProvenanceRecord] = {}

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, sample_id: object) -> bool:
        return isinstance(sample_id, str) and sample_id in self._records

    def add(self, record: ProvenanceRecord) -> None:
        if record.sample_id in self._records:
            raise ValueError(f"duplicate sample_id: {record.sample_id}")
        self._records[record.sample_id] = record

    def get(self, sample_id: str) -> ProvenanceRecord | None:
        return self._records.get(sample_id)

    def all_records(self) -> list[ProvenanceRecord]:
        return list(self._records.values())

    def filter_by(
        self,
        source_type: SourceType | None = None,
        split: SplitType | None = None,
        task_type: TaskType | None = None,
    ) -> list[ProvenanceRecord]:
        out: list[ProvenanceRecord] = []
        for rec in self._records.values():
            if source_type is not None and rec.source_type != source_type:
                continue
            if split is not None and rec.split != split:
                continue
            if task_type is not None and rec.task_type != task_type:
                continue
            out.append(rec)
        return out

    def stats(self) -> dict[str, dict[str, int]]:
        source_counts: dict[str, int] = {}
        split_counts: dict[str, int] = {}
        task_counts: dict[str, int] = {}
        for rec in self._records.values():
            source_counts[rec.source_type] = source_counts.get(rec.source_type, 0) + 1
            split_counts[rec.split] = split_counts.get(rec.split, 0) + 1
            task_counts[rec.task_type] = task_counts.get(rec.task_type, 0) + 1
        return {
            "source_type": source_counts,
            "split": split_counts,
            "task_type": task_counts,
            "total": {"count": len(self._records)},
        }

    def export_jsonl(self, path: str) -> int:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            for rec in self._records.values():
                fh.write(json.dumps(rec.to_dict()) + "\n")
        return len(self._records)

    def clone_with_notes(self, sample_id: str, notes: str) -> ProvenanceRecord:
        original = self._records[sample_id]
        updated = replace(original, notes=notes)
        self._records[sample_id] = updated
        return updated


DATA_PROVENANCE_REGISTRY = {"default": ProvenanceLedger}
