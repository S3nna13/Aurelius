"""Aurelius v2 Skill Curator — lifecycle management: enable, disable, deprecate."""

from __future__ import annotations

from src.skills.manifest import SkillStatus
from src.skills.registry import SkillRegistry


class SkillCurator:
    """Manages skill lifecycle: enable, disable, deprecate, quarantine.

    Provides operations to:
    - Enable/disable skills at runtime
    - Mark skills as stable/experimental/deprecated/unsafe
    - Generate skill audit reports
    - Find unused or failing skills
    """

    def __init__(self, registry: SkillRegistry) -> None:
        self.registry = registry

    def disable(self, skill_id: str, reason: str = "") -> bool:
        """Disable a skill. It remains registered but won't be loaded."""
        entry = self.registry.get(skill_id)
        if not entry:
            return False
        entry.manifest.status = SkillStatus.UNSAFE
        entry.manifest.metadata["disabled_reason"] = reason
        self.registry.unload_skill(skill_id)
        return True

    def enable(self, skill_id: str) -> bool:
        """Re-enable a previously disabled skill."""
        entry = self.registry.get(skill_id)
        if not entry:
            return False
        entry.manifest.status = SkillStatus.STABLE
        entry.manifest.metadata.pop("disabled_reason", None)
        return True

    def deprecate(self, skill_id: str, replacement: str = "") -> bool:
        """Mark a skill as deprecated."""
        entry = self.registry.get(skill_id)
        if not entry:
            return False
        entry.manifest.status = SkillStatus.DEPRECATED
        if replacement:
            entry.manifest.metadata["replaced_by"] = replacement
        return True

    def audit(self) -> dict:
        """Generate a full skill audit report."""
        stats = self.registry.stats()
        skills = self.registry.list_skills()
        failing: list[str] = []
        unused: list[str] = []

        for entry in skills:
            if entry.telemetry and entry.telemetry.success_rate < 0.5:
                failing.append(entry.manifest.id)
            if entry.telemetry and entry.telemetry.use_count == 0:
                unused.append(entry.manifest.id)

        return {
            "stats": stats,
            "failing_skills": failing,
            "unused_skills": unused,
            "total_skills": stats["total_skills"],
            "categories": stats["by_category"],
        }
