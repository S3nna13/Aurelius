"""
rollback_manager.py — Manages deployment rollback operations.
Aurelius LLM Project — stdlib only.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RollbackReason(Enum):
    MANUAL = "manual"
    AUTO_HEALTH_FAIL = "auto_health_fail"
    AUTO_ERROR_RATE = "auto_error_rate"
    AUTO_LATENCY = "auto_latency"
    CANARY_FAIL = "canary_fail"


@dataclass(frozen=True)
class DeploymentSnapshot:
    snapshot_id: str
    version: str
    config: dict
    created_at: float


class RollbackManager:
    """Tracks deployment snapshots and manages rollback operations."""

    def __init__(self, max_snapshots: int = 10) -> None:
        if max_snapshots < 1:
            raise ValueError("max_snapshots must be >= 1.")
        self._max_snapshots = max_snapshots
        # deque ordered oldest→newest
        self._snapshots: deque[DeploymentSnapshot] = deque()
        self._index: dict[str, DeploymentSnapshot] = {}

    def snapshot(self, version: str, config: dict) -> DeploymentSnapshot:
        """
        Create and store a snapshot of the current deployment state.

        Auto-assigns snapshot_id (uuid4 hex[:8]) and created_at (time.monotonic()).
        Evicts the oldest snapshot when the maximum capacity is reached.
        """
        snap = DeploymentSnapshot(
            snapshot_id=uuid.uuid4().hex[:8],
            version=version,
            config=dict(config),
            created_at=time.monotonic(),
        )

        if len(self._snapshots) >= self._max_snapshots:
            oldest = self._snapshots.popleft()
            self._index.pop(oldest.snapshot_id, None)

        self._snapshots.append(snap)
        self._index[snap.snapshot_id] = snap
        return snap

    def rollback_to(self, snapshot_id: str, reason: RollbackReason) -> dict:
        """
        Record a rollback to the identified snapshot.

        Returns a dict with snapshot_id, version, reason, rolled_back_at.
        Raises KeyError if the snapshot_id is not found.
        """
        snap = self._index.get(snapshot_id)
        if snap is None:
            raise KeyError(f"Snapshot '{snapshot_id}' not found.")
        return {
            "snapshot_id": snap.snapshot_id,
            "version": snap.version,
            "reason": reason.value,
            "rolled_back_at": time.monotonic(),
        }

    def latest(self) -> Optional[DeploymentSnapshot]:
        """Return the most recently created snapshot, or None if empty."""
        if not self._snapshots:
            return None
        return self._snapshots[-1]

    def history(self) -> list:
        """Return all snapshots, newest first."""
        return list(reversed(self._snapshots))

    def __len__(self) -> int:
        return len(self._snapshots)


# ---------------------------------------------------------------------------
# Module registry
# ---------------------------------------------------------------------------
ROLLBACK_MANAGER_REGISTRY: dict = {"default": RollbackManager}

REGISTRY = ROLLBACK_MANAGER_REGISTRY
