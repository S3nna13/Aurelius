"""Rollback manager with persistent manifest snapshots."""
from __future__ import annotations

import json
import os
from pathlib import Path


class RollbackManager:
    def __init__(self, artifact_dir: str, max_revisions: int = 5) -> None:
        if max_revisions < 1:
            raise ValueError("max_revisions must be >= 1.")
        self._artifact_dir = Path(artifact_dir)
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        self._max_revisions = max_revisions

    def _validate_version(self, version: str) -> None:
        if not isinstance(version, str) or not version:
            raise ValueError("version must be a non-empty string.")
        if ".." in version:
            raise ValueError("version must not contain path traversal characters.")
        if len(version) > 64:
            raise ValueError("version must not exceed 64 characters.")

    def snapshot(self, version: str, manifest: dict) -> str:
        self._validate_version(version)
        version_dir = self._artifact_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = version_dir / "snapshot.json"
        snapshot_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return str(snapshot_path)

    def list_revisions(self) -> list[str]:
        revisions: list[tuple[str, float]] = []
        for entry in self._artifact_dir.iterdir():
            if not entry.is_dir():
                continue
            snapshot_path = entry / "snapshot.json"
            if not snapshot_path.exists():
                continue
            try:
                mtime = os.stat(snapshot_path).st_mtime
            except OSError:
                continue
            revisions.append((entry.name, mtime))
        revisions.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in revisions]

    def get_manifest(self, version: str) -> dict:
        self._validate_version(version)
        snapshot_path = self._artifact_dir / version / "snapshot.json"
        if not snapshot_path.exists():
            raise FileNotFoundError(f"Snapshot not found for version '{version}'.")
        return json.loads(snapshot_path.read_text(encoding="utf-8"))

    def rollback(self, target_version: str) -> dict:
        self._validate_version(target_version)
        snapshot_path = self._artifact_dir / target_version / "snapshot.json"
        if not snapshot_path.exists():
            raise FileNotFoundError(f"Snapshot not found for version '{target_version}'.")
        return json.loads(snapshot_path.read_text(encoding="utf-8"))

    def prune_old_revisions(self) -> None:
        revisions = self.list_revisions()
        if len(revisions) <= self._max_revisions:
            return
        for old_version in revisions[self._max_revisions :]:
            old_dir = self._artifact_dir / old_version
            if old_dir.exists():
                snapshot = old_dir / "snapshot.json"
                if snapshot.exists():
                    snapshot.unlink()
                os.rmdir(old_dir)


ROLLBACK_MANAGER_REGISTRY: dict[str, type[RollbackManager]] = {"default": RollbackManager}
