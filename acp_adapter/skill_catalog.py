"""Aurelius MCP skill catalog and invocation registry.

Provides skill discovery, invocation tracking, and versioning for the Aurelius
MCP surface.  All logic uses only stdlib — no external dependencies.

Inspired by Anthropic Claude Code skills system (MIT), Goose extension/MCP
(Apache-2.0), clean-room reimplementation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Semantic version pattern: X.Y.Z (integers only, no pre-release suffix)
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SkillCatalogError(Exception):
    """Raised for invalid skill operations (bad metadata, unknown id, etc.)."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class SkillTrigger:
    """A trigger pattern that activates a skill via text matching."""

    pattern: str
    description: str
    examples: list[str] = field(default_factory=list)


@dataclass
class SkillMetadata:
    """Descriptor for a single Aurelius skill."""

    skill_id: str
    name: str
    version: str
    description: str
    triggers: list[SkillTrigger]
    tags: list[str]
    author: str = "aurelius"
    min_tool_version: str = "1.0.0"
    enabled: bool = True


@dataclass
class SkillInvocationRecord:
    """Record of a single skill invocation."""

    skill_id: str
    invoked_at: float
    args: dict
    result: dict | None = None
    error: str | None = None
    duration_ms: float | None = None


# ---------------------------------------------------------------------------
# Skill catalog
# ---------------------------------------------------------------------------


class SkillCatalog:
    """Registry of :class:`SkillMetadata` objects with invocation history.

    Supports register, unregister, trigger-based and tag-based discovery,
    invocation recording, and serialization.
    """

    def __init__(self) -> None:
        self._skills: dict[str, SkillMetadata] = {}
        self._invocations: list[SkillInvocationRecord] = []

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(skill: SkillMetadata) -> None:
        if not isinstance(skill.skill_id, str) or not skill.skill_id.strip():
            raise SkillCatalogError(f"skill_id must be a non-empty string, got {skill.skill_id!r}")
        if not _SEMVER_RE.match(skill.version):
            raise SkillCatalogError(
                f"version must match 'X.Y.Z' (semver integers), got {skill.version!r}"
            )

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def register(self, skill: SkillMetadata) -> None:
        """Validate and register *skill*.

        Raises :class:`SkillCatalogError` if validation fails.
        """
        self._validate(skill)
        self._skills[skill.skill_id] = skill

    def unregister(self, skill_id: str) -> None:
        """Remove the skill with *skill_id*.

        Raises :class:`SkillCatalogError` if not found.
        """
        if skill_id not in self._skills:
            raise SkillCatalogError(f"skill not found: {skill_id!r}")
        del self._skills[skill_id]

    def get(self, skill_id: str) -> SkillMetadata:
        """Return the metadata for *skill_id*.

        Raises :class:`SkillCatalogError` if not found.
        """
        if skill_id not in self._skills:
            raise SkillCatalogError(f"skill not found: {skill_id!r}")
        return self._skills[skill_id]

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def find_by_trigger(self, text: str) -> list[SkillMetadata]:
        """Return skills whose any trigger.pattern appears as a substring in *text*.

        Matching is case-insensitive.
        """
        text_lower = text.lower()
        results: list[SkillMetadata] = []
        for skill in self._skills.values():
            for trigger in skill.triggers:
                if trigger.pattern.lower() in text_lower:
                    results.append(skill)
                    break  # avoid duplicating the same skill
        return results

    def find_by_tag(self, tag: str) -> list[SkillMetadata]:
        """Return all skills that carry *tag* (case-sensitive)."""
        return [s for s in self._skills.values() if tag in s.tags]

    def list_skills(self, enabled_only: bool = True) -> list[SkillMetadata]:
        """Return all registered skills, optionally filtered to enabled ones."""
        skills = list(self._skills.values())
        if enabled_only:
            skills = [s for s in skills if s.enabled]
        return skills

    # ------------------------------------------------------------------
    # Invocation history
    # ------------------------------------------------------------------

    def record_invocation(self, record: SkillInvocationRecord) -> None:
        """Append *record* to the invocation history."""
        self._invocations.append(record)

    def invocation_history(self, skill_id: str | None = None) -> list[SkillInvocationRecord]:
        """Return invocation records, optionally filtered to *skill_id*."""
        if skill_id is None:
            return list(self._invocations)
        return [r for r in self._invocations if r.skill_id == skill_id]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serializable snapshot of skills and invocations."""
        return {
            "skills": {
                sid: {
                    "skill_id": s.skill_id,
                    "name": s.name,
                    "version": s.version,
                    "description": s.description,
                    "triggers": [
                        {
                            "pattern": t.pattern,
                            "description": t.description,
                            "examples": list(t.examples),
                        }
                        for t in s.triggers
                    ],
                    "tags": list(s.tags),
                    "author": s.author,
                    "min_tool_version": s.min_tool_version,
                    "enabled": s.enabled,
                }
                for sid, s in self._skills.items()
            },
            "invocations": [
                {
                    "skill_id": r.skill_id,
                    "invoked_at": r.invoked_at,
                    "args": r.args,
                    "result": r.result,
                    "error": r.error,
                    "duration_ms": r.duration_ms,
                }
                for r in self._invocations
            ],
        }


# ---------------------------------------------------------------------------
# Named catalog registry and singleton
# ---------------------------------------------------------------------------

#: Named collection of :class:`SkillCatalog` instances.
SKILL_CATALOG_REGISTRY: dict[str, SkillCatalog] = {}

#: Default singleton catalog pre-seeded with stub skills.
DEFAULT_SKILL_CATALOG: SkillCatalog = SkillCatalog()

# Seed with 3 built-in stub skills
DEFAULT_SKILL_CATALOG.register(
    SkillMetadata(
        skill_id="code-review",
        name="Code Review",
        version="1.0.0",
        description="Performs automated code review and suggests improvements.",
        triggers=[
            SkillTrigger(
                pattern="review",
                description="Triggered when the user asks for a code review.",
                examples=["review this code", "please review my PR"],
            )
        ],
        tags=["code", "review", "quality"],
    )
)

DEFAULT_SKILL_CATALOG.register(
    SkillMetadata(
        skill_id="test-generator",
        name="Test Generator",
        version="1.0.0",
        description="Generates unit and integration tests for existing code.",
        triggers=[
            SkillTrigger(
                pattern="generate test",
                description="Triggered when the user wants tests generated.",
                examples=["generate tests for this function", "generate test suite"],
            ),
            SkillTrigger(
                pattern="write test",
                description="Triggered when the user asks to write tests.",
                examples=["write tests for my class"],
            ),
        ],
        tags=["test", "generation", "quality"],
    )
)

DEFAULT_SKILL_CATALOG.register(
    SkillMetadata(
        skill_id="doc-writer",
        name="Documentation Writer",
        version="1.0.0",
        description="Generates documentation, docstrings, and README files.",
        triggers=[
            SkillTrigger(
                pattern="document",
                description="Triggered when the user needs documentation.",
                examples=["document this module", "add documentation"],
            ),
            SkillTrigger(
                pattern="docstring",
                description="Triggered when the user needs docstrings.",
                examples=["add docstrings to my functions"],
            ),
        ],
        tags=["docs", "documentation", "generation"],
    )
)

# Register default catalog under the named registry
SKILL_CATALOG_REGISTRY["default"] = DEFAULT_SKILL_CATALOG


__all__ = [
    "DEFAULT_SKILL_CATALOG",
    "SKILL_CATALOG_REGISTRY",
    "SkillCatalog",
    "SkillCatalogError",
    "SkillInvocationRecord",
    "SkillMetadata",
    "SkillTrigger",
]
