"""Checkpoint metadata migration registry and orchestrator.

When ``checkpoint_format_version`` bumps (major or minor), checkpoints saved
under the old schema need to be upgraded. This module provides a registry of
:class:`MigrationStep` objects keyed by ``(from_version, to_version)`` along
with a BFS-based path finder and the :class:`CheckpointMigrator` orchestrator.

Design notes
------------
* Apply functions must be **pure** — they receive a state dict and return a
  new dict; they must not mutate the input.
* Path resolution uses breadth-first search over registered edges so the
  shortest chain of semver-adjacent steps is preferred.
* Same-version migration is a no-op (returns a shallow copy of the state).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple


class MigrationError(Exception):
    """Raised when a checkpoint migration cannot be resolved or applied."""


@dataclass(frozen=True)
class MigrationStep:
    """A single registered migration step between two adjacent versions."""

    from_version: str
    to_version: str
    description: str
    apply: Callable[[Dict], Dict]


MIGRATION_REGISTRY: Dict[Tuple[str, str], MigrationStep] = {}


def register_migration(step: MigrationStep) -> None:
    """Register a migration step; duplicate ``(from, to)`` keys raise."""
    key = (step.from_version, step.to_version)
    if key in MIGRATION_REGISTRY:
        raise MigrationError(
            f"migration already registered: {step.from_version} -> {step.to_version}"
        )
    MIGRATION_REGISTRY[key] = step


def get_migration_path(from_version: str, to_version: str) -> List[MigrationStep]:
    """Return shortest sequence of steps from ``from_version`` to ``to_version``.

    Uses BFS over the registry's directed edges. Raises :class:`MigrationError`
    if no path exists. Same-version input returns an empty list.
    """
    if from_version == to_version:
        return []

    # Collect versions known to the registry.
    known: set[str] = set()
    for src, dst in MIGRATION_REGISTRY:
        known.add(src)
        known.add(dst)
    if from_version not in known:
        raise MigrationError(f"unknown source version: {from_version!r}")
    if to_version not in known:
        raise MigrationError(f"unknown destination version: {to_version!r}")

    # BFS: track predecessor edges so we can reconstruct the path.
    predecessor: Dict[str, Tuple[str, MigrationStep]] = {}
    queue: deque[str] = deque([from_version])
    visited: set[str] = {from_version}
    while queue:
        current = queue.popleft()
        if current == to_version:
            break
        for (src, dst), step in MIGRATION_REGISTRY.items():
            if src != current or dst in visited:
                continue
            visited.add(dst)
            predecessor[dst] = (current, step)
            queue.append(dst)

    if to_version not in predecessor:
        raise MigrationError(
            f"no migration path from {from_version!r} to {to_version!r}"
        )

    # Reconstruct path by walking predecessors back to from_version.
    path: List[MigrationStep] = []
    cursor = to_version
    while cursor != from_version:
        prev, step = predecessor[cursor]
        path.append(step)
        cursor = prev
    path.reverse()
    return path


class CheckpointMigrator:
    """Apply a sequence of registered migrations to a checkpoint state dict."""

    def migrate(
        self, state: Dict, from_version: str, to_version: str
    ) -> Dict:
        path = get_migration_path(from_version, to_version)
        if not path:
            # Same-version no-op: shallow copy to honor the no-mutate contract.
            result = dict(state)
            result["checkpoint_format_version"] = to_version
            return result
        current = state
        for step in path:
            current = step.apply(current)
        final_version = current.get("checkpoint_format_version")
        if final_version != to_version:
            raise MigrationError(
                f"post-migration version mismatch: expected {to_version!r}, "
                f"got {final_version!r}"
            )
        return current

    def dry_run(self, from_version: str, to_version: str) -> List[str]:
        path = get_migration_path(from_version, to_version)
        return [step.description for step in path]

    def can_migrate(self, from_version: str, to_version: str) -> bool:
        try:
            get_migration_path(from_version, to_version)
        except MigrationError:
            return False
        return True


# ---------------------------------------------------------------------------
# Seed migrations (illustrative; no real production bump yet).
# ---------------------------------------------------------------------------


def _apply_1_0_0_to_1_1_0(state: Dict) -> Dict:
    """Add optional ``training_recipe`` field (defaults to ``None``)."""
    new_state = dict(state)
    if "training_recipe" not in new_state:
        new_state["training_recipe"] = None
    new_state["checkpoint_format_version"] = "1.1.0"
    return new_state


def _apply_1_1_0_to_2_0_0(state: Dict) -> Dict:
    """Rename ``tokenizer_name`` -> ``tokenizer_id`` (major bump)."""
    new_state = dict(state)
    if "tokenizer_name" in new_state:
        new_state["tokenizer_id"] = new_state["tokenizer_name"]
        del new_state["tokenizer_name"]
    new_state["checkpoint_format_version"] = "2.0.0"
    return new_state


register_migration(
    MigrationStep(
        from_version="1.0.0",
        to_version="1.1.0",
        description="add optional 'training_recipe' field",
        apply=_apply_1_0_0_to_1_1_0,
    )
)
register_migration(
    MigrationStep(
        from_version="1.1.0",
        to_version="2.0.0",
        description="rename 'tokenizer_name' to 'tokenizer_id'",
        apply=_apply_1_1_0_to_2_0_0,
    )
)


__all__ = [
    "CheckpointMigrator",
    "MIGRATION_REGISTRY",
    "MigrationError",
    "MigrationStep",
    "get_migration_path",
    "register_migration",
]
