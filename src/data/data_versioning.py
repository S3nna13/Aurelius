"""Data versioning: dataset snapshots, diff tracking, lineage graph."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


@dataclass
class DatasetVersion:
    name: str
    n_samples: int
    version_id: str = field(default_factory=_new_id)
    created_at: str = field(default_factory=_now_iso)
    parent_id: str | None = None
    description: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class DataDiff:
    from_version: str
    to_version: str
    added: int
    removed: int
    modified: int


class DataVersionRegistry:
    def __init__(self) -> None:
        self._versions: dict[str, DatasetVersion] = {}

    def create_version(
        self,
        name: str,
        n_samples: int,
        parent_id: str | None = None,
        description: str = "",
        **metadata,
    ) -> DatasetVersion:
        v = DatasetVersion(
            name=name,
            n_samples=n_samples,
            parent_id=parent_id,
            description=description,
            metadata=dict(metadata),
        )
        self._versions[v.version_id] = v
        return v

    def get(self, version_id: str) -> DatasetVersion | None:
        return self._versions.get(version_id)

    def diff(self, from_id: str, to_id: str) -> DataDiff | None:
        from_v = self._versions.get(from_id)
        to_v = self._versions.get(to_id)
        if from_v is None or to_v is None:
            return None
        added = max(0, to_v.n_samples - from_v.n_samples)
        removed = max(0, from_v.n_samples - to_v.n_samples)
        modified = min(from_v.n_samples, to_v.n_samples) // 10
        return DataDiff(
            from_version=from_id,
            to_version=to_id,
            added=added,
            removed=removed,
            modified=modified,
        )

    def lineage(self, version_id: str) -> list[DatasetVersion]:
        """Walk parent_id chain from version to root; return list from root to version."""
        chain: list[DatasetVersion] = []
        current_id: str | None = version_id
        while current_id is not None:
            v = self._versions.get(current_id)
            if v is None:
                break
            chain.append(v)
            current_id = v.parent_id
        chain.reverse()
        return chain

    def list_versions(self) -> list[DatasetVersion]:
        """All versions sorted by created_at."""
        return sorted(self._versions.values(), key=lambda v: v.created_at)

    def delete(self, version_id: str) -> bool:
        if version_id in self._versions:
            del self._versions[version_id]
            return True
        return False


DATA_VERSION_REGISTRY = DataVersionRegistry()
