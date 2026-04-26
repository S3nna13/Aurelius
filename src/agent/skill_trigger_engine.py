"""Skill trigger engine — bridges skill catalog triggers with execution.

Matches incoming text against registered skill triggers and optionally
executes matched skills via the skill executor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.agent.skill_executor import (
    DEFAULT_SKILL_EXECUTOR,
    ExecutionResult,
    SkillContext,
    SkillExecutionError,
    SkillExecutor,
)
from src.mcp.skill_catalog import SkillCatalogError, SkillMetadata


class TriggerEngineError(Exception):
    """Raised for invalid trigger engine operations."""


@dataclass
class MatchedSkill:
    """A skill that matched a trigger pattern against input text."""

    skill_id: str
    name: str
    trigger_pattern: str
    confidence: float = 1.0


@dataclass
class TriggerResult:
    """Result of trigger matching and optional execution."""

    matches: list[MatchedSkill] = field(default_factory=list)
    executed: list[ExecutionResult] = field(default_factory=list)
    text: str = ""


@dataclass
class SkillTriggerEngine:
    """Matches text against skill triggers and executes matched skills."""

    skill_catalog: Any | None = None
    executor: SkillExecutor | None = None

    def __post_init__(self) -> None:
        if self.executor is None:
            self.executor = DEFAULT_SKILL_EXECUTOR
        self._skills: dict[str, SkillMetadata] = {}

    def match(self, text: str) -> TriggerResult:
        """Find skills whose triggers appear in *text*.

        Deduplicates by ``skill_id``.  If a ``skill_catalog`` is attached
        and exposes ``find_by_trigger``, it is queried first; internal
        overrides are then merged.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}")

        seen: dict[str, MatchedSkill] = {}

        # Query catalog if available
        if self.skill_catalog is not None and hasattr(self.skill_catalog, "find_by_trigger"):
            try:
                catalog_matches = self.skill_catalog.find_by_trigger(text)
                for skill in catalog_matches:
                    for trigger in skill.triggers:
                        seen.setdefault(
                            skill.skill_id,
                            MatchedSkill(
                                skill_id=skill.skill_id,
                                name=skill.name,
                                trigger_pattern=trigger.pattern,
                                confidence=1.0,
                            ),
                        )
            except SkillCatalogError:
                pass

        # Merge internal skills
        for skill in self._skills.values():
            for trigger in skill.triggers:
                if trigger.pattern.lower() in text.lower():
                    seen.setdefault(
                        skill.skill_id,
                        MatchedSkill(
                            skill_id=skill.skill_id,
                            name=skill.name,
                            trigger_pattern=trigger.pattern,
                            confidence=1.0,
                        ),
                    )

        return TriggerResult(matches=list(seen.values()), text=text)

    def match_and_execute(
        self,
        text: str,
        context: SkillContext | None = None,
    ) -> TriggerResult:
        """Match *text* against triggers and execute each matched skill."""
        trigger_result = self.match(text)
        executed: list[ExecutionResult] = []

        for matched in trigger_result.matches:
            instructions = self._resolve_instructions(matched.skill_id)
            try:
                result = self.executor.execute(matched.skill_id, instructions, context)
            except SkillExecutionError as exc:
                result = ExecutionResult(
                    output=str(exc),
                    success=False,
                    duration_ms=0.0,
                    metadata={"error": str(exc)},
                )
            executed.append(result)

        trigger_result.executed = executed
        return trigger_result

    def add_skill(self, skill_metadata: SkillMetadata) -> None:
        """Register *skill_metadata* into the engine's internal store."""
        self._skills[skill_metadata.skill_id] = skill_metadata
        if self.skill_catalog is not None and hasattr(self.skill_catalog, "register"):
            try:
                self.skill_catalog.register(skill_metadata)
            except SkillCatalogError:
                pass

    def remove_skill(self, skill_id: str) -> None:
        """Remove *skill_id* from the internal store."""
        if skill_id not in self._skills:
            raise TriggerEngineError(f"skill not found: {skill_id!r}")
        del self._skills[skill_id]

    def _resolve_instructions(self, skill_id: str) -> str:
        """Resolve instructions for *skill_id*."""
        if skill_id in self._skills:
            return self._skills[skill_id].description
        if self.skill_catalog is not None and hasattr(self.skill_catalog, "get"):
            try:
                skill = self.skill_catalog.get(skill_id)
                if hasattr(skill, "description"):
                    return skill.description
            except SkillCatalogError:
                pass
        return f"Skill: {skill_id}"


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

DEFAULT_TRIGGER_ENGINE: SkillTriggerEngine = SkillTriggerEngine()

TRIGGER_ENGINE_REGISTRY: dict[str, SkillTriggerEngine] = {
    "default": DEFAULT_TRIGGER_ENGINE,
}


__all__ = [
    "DEFAULT_TRIGGER_ENGINE",
    "TRIGGER_ENGINE_REGISTRY",
    "MatchedSkill",
    "SkillTriggerEngine",
    "TriggerEngineError",
    "TriggerResult",
]
