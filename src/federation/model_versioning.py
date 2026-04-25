"""Model versioning registry for federated learning rounds."""
from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class ModelVersion:
    version_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    round_number: int = 0
    param_hash: str = ""
    created_at: float = field(default_factory=time.monotonic)
    metadata: dict = field(default_factory=dict)


def _hash_params(params: dict[str, torch.Tensor]) -> str:
    """Return a SHA-256 hex digest of concatenated parameter bytes."""
    h = hashlib.sha256(usedforsecurity=False)
    for key in sorted(params.keys()):
        h.update(key.encode())
        h.update(params[key].cpu().numpy().tobytes())
    return h.hexdigest()


class ModelVersionRegistry:
    """Tracks model versions across federated learning rounds."""

    def __init__(self) -> None:
        self._versions: dict[str, ModelVersion] = {}  # version_id -> ModelVersion
        self._round_index: dict[int, str] = {}         # round_number -> version_id

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        round_number: int,
        params: dict[str, torch.Tensor],
        metadata: Optional[dict] = None,
    ) -> ModelVersion:
        """Register a new model version for *round_number*."""
        version = ModelVersion(
            round_number=round_number,
            param_hash=_hash_params(params),
            created_at=time.monotonic(),
            metadata=metadata or {},
        )
        self._versions[version.version_id] = version
        self._round_index[round_number] = version.version_id
        return version

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(self, version_id: str) -> Optional[ModelVersion]:
        """Return the version with *version_id*, or None."""
        return self._versions.get(version_id)

    def get_by_round(self, round_number: int) -> Optional[ModelVersion]:
        """Return the version registered for *round_number*, or None."""
        vid = self._round_index.get(round_number)
        return self._versions.get(vid) if vid else None

    def list_versions(self) -> list[ModelVersion]:
        """Return all versions sorted by round_number ascending."""
        return sorted(self._versions.values(), key=lambda v: v.round_number)

    # ------------------------------------------------------------------
    # Diff
    # ------------------------------------------------------------------

    def diff_rounds(self, r1: int, r2: int) -> dict:
        """Compare param names present in two rounds.

        Returns a dict with keys:
            added   – param names in r2 but not r1
            removed – param names in r1 but not r2
            changed – param names present in both (always empty in stub
                      since we only store hashes, not raw params)
        """
        v1 = self.get_by_round(r1)
        v2 = self.get_by_round(r2)

        names1: set[str] = set(v1.metadata.get("param_names", [])) if v1 else set()
        names2: set[str] = set(v2.metadata.get("param_names", [])) if v2 else set()

        return {
            "added": sorted(names2 - names1),
            "removed": sorted(names1 - names2),
            "changed": sorted(names1 & names2),
        }


# Module-level singleton
_MODEL_VERSION_REGISTRY = ModelVersionRegistry()

try:
    from src.federation.federated_learning import (  # type: ignore[import]
        FEDERATION_REGISTRY as _REG,
    )
    _REG["model_versioning"] = _MODEL_VERSION_REGISTRY
except Exception:
    pass

FEDERATION_REGISTRY: dict[str, object] = {"model_versioning": _MODEL_VERSION_REGISTRY}
