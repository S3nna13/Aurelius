"""Skill version registry and manager for Aurelius.

Tracks immutable skill versions with checksums, supports semver-like
version strings, and provides loud validation on invalid inputs.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

__all__ = [
    "SkillVersionError",
    "SkillVersion",
    "SkillVersionManager",
    "SKILL_VERSION_REGISTRY",
]

_VERSION_RE = re.compile(r"^[0-9]+(\.[0-9]+)*$")


class SkillVersionError(Exception):
    """Raised when a skill version operation fails validation."""


@dataclass(frozen=True)
class SkillVersion:
    """Immutable record of a registered skill version."""

    skill_id: str
    version: str
    checksum: str
    registered_at: float
    metadata: dict[str, Any]


class SkillVersionManager:
    """Manages registration and lookup of skill versions."""

    def __init__(self) -> None:
        self._registry: dict[str, dict[str, SkillVersion]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_version(version: str) -> None:
        if not isinstance(version, str) or not version.strip():
            raise SkillVersionError("version must be a non-empty string")
        if not _VERSION_RE.match(version):
            raise SkillVersionError(
                f"version {version!r} is invalid; "
                "expected an integer or dot-separated numeric string (e.g., '1.2.3')"
            )

    @staticmethod
    def _validate_skill_id(skill_id: str) -> None:
        if not isinstance(skill_id, str) or not skill_id.strip():
            raise SkillVersionError("skill_id must be a non-empty string")

    @staticmethod
    def _validate_checksum(checksum: str) -> None:
        if not isinstance(checksum, str) or not checksum.strip():
            raise SkillVersionError("checksum must be a non-empty string")

    @staticmethod
    def _parse_version(version: str) -> tuple[int, ...]:
        """Convert a version string into a tuple of integers for comparison."""
        return tuple(int(part) for part in version.split("."))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        skill_id: str,
        version: str,
        checksum: str,
        metadata: dict[str, Any] | None = None,
    ) -> SkillVersion:
        """Register a new skill version.

        Raises :class:`SkillVersionError` on invalid inputs or duplicate
        registrations.
        """
        self._validate_skill_id(skill_id)
        self._validate_version(version)
        self._validate_checksum(checksum)

        if skill_id not in self._registry:
            self._registry[skill_id] = {}

        if version in self._registry[skill_id]:
            raise SkillVersionError(
                f"version {version!r} for skill {skill_id!r} is already registered"
            )

        entry = SkillVersion(
            skill_id=skill_id,
            version=version,
            checksum=checksum,
            registered_at=time.time(),
            metadata=dict(metadata) if metadata is not None else {},
        )
        self._registry[skill_id][version] = entry
        return entry

    def get_latest(self, skill_id: str) -> SkillVersion | None:
        """Return the latest registered version for *skill_id*, or ``None``."""
        self._validate_skill_id(skill_id)
        versions = self._registry.get(skill_id, {})
        if not versions:
            return None
        latest_version = max(versions.keys(), key=self._parse_version)
        return versions[latest_version]

    def get_version(self, skill_id: str, version: str) -> SkillVersion | None:
        """Return a specific version record, or ``None``."""
        self._validate_skill_id(skill_id)
        self._validate_version(version)
        return self._registry.get(skill_id, {}).get(version)

    def list_versions(self, skill_id: str) -> list[str]:
        """Return all registered versions for *skill_id* in ascending order."""
        self._validate_skill_id(skill_id)
        versions = self._registry.get(skill_id, {})
        return sorted(versions.keys(), key=self._parse_version)

    def list_all(self) -> list[str]:
        """Return all registered skill IDs."""
        return list(self._registry.keys())

    def validate_checksum(self, skill_id: str, version: str, checksum: str) -> bool:
        """Return ``True`` if the stored checksum matches *checksum*."""
        self._validate_skill_id(skill_id)
        self._validate_version(version)
        self._validate_checksum(checksum)
        entry = self._registry.get(skill_id, {}).get(version)
        if entry is None:
            return False
        return entry.checksum == checksum


# Module-level singletons
DEFAULT_SKILL_VERSION_MANAGER = SkillVersionManager()

SKILL_VERSION_REGISTRY: dict[str, SkillVersionManager] = {
    "default": DEFAULT_SKILL_VERSION_MANAGER,
}
