"""Dataset versioning — track every dataset version, transformation, and mix.

Each dataset gets a content-addressed hash (based on its data).
Every transformation creates a new version.
The version manifest tracks the full provenance chain.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DatasetVersion:
    name: str
    version: str
    hash: str
    created: str
    tokens: int
    samples: int
    parent: str | None = None
    transform: str = ""
    parameters: dict[str, Any] | None = None


class DatasetVersioner:
    """Content-addressed dataset versioning."""

    def __init__(self, manifest_dir: str | Path = "data/manifests"):
        self.manifest_dir = Path(manifest_dir)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self._versions: dict[str, list[DatasetVersion]] = {}

    def hash_content(self, texts: list[str]) -> str:
        combined = "".join(texts[:1000]).encode()
        return hashlib.sha256(combined).hexdigest()[:16]

    def version(self, name: str, texts: list[str], transform: str = "raw", parent: str | None = None, parameters: dict[str, Any] | None = None) -> DatasetVersion:
        content_hash = self.hash_content(texts)
        ver_num = len(self._versions.get(name, [])) + 1
        version_str = f"v{ver_num}.{content_hash[:8]}"

        dv = DatasetVersion(
            name=name, version=version_str, hash=content_hash,
            created=datetime.now(timezone.utc).isoformat(),
            tokens=sum(len(t.split()) for t in texts),
            samples=len(texts),
            parent=parent, transform=transform, parameters=parameters,
        )

        self._versions.setdefault(name, []).append(dv)
        self._save_manifest(dv)
        return dv

    def get_history(self, name: str) -> list[DatasetVersion]:
        return self._versions.get(name, [])

    def get_latest(self, name: str) -> DatasetVersion | None:
        versions = self._versions.get(name, [])
        return versions[-1] if versions else None

    def _save_manifest(self, dv: DatasetVersion) -> None:
        path = self.manifest_dir / f"{dv.name}_{dv.version}.json"
        with open(path, "w") as f:
            json.dump({
                "name": dv.name, "version": dv.version, "hash": dv.hash,
                "created": dv.created, "tokens": dv.tokens, "samples": dv.samples,
                "parent": dv.parent, "transform": dv.transform, "parameters": dv.parameters,
            }, f, indent=2)

    def load_all(self) -> None:
        for path in self.manifest_dir.glob("*.json"):
            data = json.loads(path.read_text())
            dv = DatasetVersion(**data)
            self._versions.setdefault(dv.name, []).append(dv)
        self._versions = {k: sorted(v, key=lambda x: x.version) for k, v in self._versions.items()}

    def summary(self) -> dict[str, Any]:
        return {
            name: {
                "versions": len(vs),
                "latest": vs[-1].version if vs else None,
                "total_tokens": sum(v.tokens for v in vs),
                "total_samples": sum(v.samples for v in vs),
            }
            for name, vs in self._versions.items()
        }
