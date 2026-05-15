"""Aurelius v2 Skill Registry — discovery, lookup, listing, loading of native skills."""

from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.skills.manifest import (
    RiskLevel,
    SkillExecutionMode,
    SkillManifest,
    SkillPermission,
    SkillStatus,
)
from src.skills.permissions import (
    PermissionCheck,
    PermissionContext,
    PermissionGate,
)
from src.skills.telemetry import SkillTelemetry, TelemetryEvent


@dataclass
class SkillEntry:
    """A registered skill with its manifest, loaded state, and telemetry."""

    manifest: SkillManifest
    loaded: bool = False
    module: Any = None
    telemetry: SkillTelemetry | None = None
    _permission_checks: list[PermissionCheck] = field(default_factory=list)


class SkillRegistry:
    """Central registry for all Aurelius native skills.

    Responsibilities:
    - Discover skills from builtin/ directory
    - Load/unload skills on demand
    - Search and filter by category, tag, or query
    - Enforce permissions before execution
    - Track telemetry
    """

    def __init__(self, base_path: str | None = None) -> None:
        self._skills: dict[str, SkillEntry] = {}
        self._permission_gate = PermissionGate()
        self._base_path = base_path or self._default_base_path()

    def register(self, manifest: SkillManifest) -> None:
        """Register a skill manifest."""
        entry = SkillEntry(manifest=manifest)
        self._skills[manifest.id] = entry

    def discover_from_path(self, path: str | None = None) -> int:
        """Discover and register all skills under a directory."""
        search_path = path or self._base_path
        count = 0
        builtin_path = Path(search_path) / "builtin"
        if not builtin_path.exists():
            return 0

        for category_dir in sorted(builtin_path.iterdir()):
            if not category_dir.is_dir():
                continue
            for skill_file in category_dir.glob("skill.json"):
                try:
                    data = json.loads(skill_file.read_text())
                    perm_list = data.get("permissions", [])
                    perms = [
                        SkillPermission(name=p["name"], scope=p.get("scope", "limited"))
                        for p in perm_list
                    ]
                    mode_list = data.get("supported_modes", [])
                    modes = [SkillExecutionMode(m) for m in mode_list]
                    manifest = SkillManifest(
                        id=data["id"], name=data["name"],
                        version=data["version"], category=data["category"],
                        summary=data["summary"],
                        description=data.get("description", ""),
                        permissions=perms,
                        risk_level=RiskLevel(data.get("risk_level", "low")),
                        required_tools=data.get("required_tools", []),
                        entrypoint=data.get("entrypoint", ""),
                        daies_tests=data.get("daies_tests", []),
                        tags=data.get("tags", []),
                        status=SkillStatus(data.get("status", "stable")),
                        supported_modes=modes,
                        max_runtime_seconds=data.get("max_runtime_seconds", 300),
                        metadata=data.get("metadata", {}),
                    )
                    errs = manifest.validate()
                    if not errs:
                        self.register(manifest)
                        count += 1
                except Exception:
                    # Skip malformed manifests
                    pass
        return count

    def get(self, skill_id: str) -> SkillEntry | None:
        """Get a skill by ID."""
        return self._skills.get(skill_id)

    def list_skills(
        self, category: str | None = None,
    ) -> list[SkillEntry]:
        """List all registered skills, optionally filtered by category."""
        skills = list(self._skills.values())
        if category:
            return [s for s in skills if s.manifest.category == category]
        return skills

    def search(self, query: str) -> list[SkillEntry]:
        """Search skills by name, summary, tags, or ID."""
        ql = query.lower()
        results = []
        for entry in self._skills.values():
            m = entry.manifest
            searchable = (
                f"{m.id} {m.name} {m.summary}"
                f" {' '.join(m.tags)}"
            ).lower()
            if ql in searchable:
                results.append(entry)
        return results

    def categories(self) -> list[str]:
        """Return sorted list of unique skill categories."""
        cats = {s.manifest.category for s in self._skills.values()}
        return sorted(cats)

    def load_skill(self, skill_id: str) -> bool:
        """Load a skill module. Returns True on success."""
        entry = self._skills.get(skill_id)
        if not entry:
            return False
        if not entry.manifest.entrypoint:
            return False
        try:
            module_path, func_name = entry.manifest.entrypoint.split(":")
            module = importlib.import_module(module_path)
            entry.module = getattr(module, func_name)
            entry.loaded = True
            entry.telemetry = SkillTelemetry(skill_id=skill_id)
            return True
        except Exception:
            return False

    def unload_skill(self, skill_id: str) -> bool:
        """Unload a skill to free memory."""
        entry = self._skills.get(skill_id)
        if not entry:
            return False
        entry.loaded = False
        entry.module = None
        return True

    def check_permissions(
        self, skill_id: str,
        context: PermissionContext | None = None,
    ) -> list[PermissionCheck]:
        """Check permissions for a skill."""
        entry = self._skills.get(skill_id)
        if not entry:
            reason = f"Skill {skill_id} not registered"
            return [PermissionCheck(
                permission="skill_not_found",
                grant="denied", reason=reason,
            )]
        checks = self._permission_gate.check(entry.manifest, context)
        entry._permission_checks = checks
        return checks

    def record_use(
        self, skill_id: str,
        success: bool = True, runtime_ms: int = 0,
    ) -> None:
        """Record skill usage telemetry."""
        entry = self._skills.get(skill_id)
        if entry and entry.telemetry:
            event = TelemetryEvent(success=success, runtime_ms=runtime_ms)
            entry.telemetry.record(event)

    def stats(self) -> dict[str, Any]:
        """Return registry statistics."""
        return {
            "total_skills": len(self._skills),
            "loaded_skills": sum(
                1 for s in self._skills.values() if s.loaded
            ),
            "categories": self.categories(),
            "by_category": {
                cat: sum(
                    1 for s in self._skills.values()
                    if s.manifest.category == cat
                )
                for cat in self.categories()
            },
            "by_status": {
                status.value: sum(
                    1 for s in self._skills.values()
                    if s.manifest.status == status
                )
                for status in SkillStatus
            },
            "by_risk": {
                risk.value: sum(
                    1 for s in self._skills.values()
                    if s.manifest.risk_level == risk
                )
                for risk in RiskLevel
            },
        }

    @staticmethod
    def _default_base_path() -> str:
        return os.path.dirname(__file__)
