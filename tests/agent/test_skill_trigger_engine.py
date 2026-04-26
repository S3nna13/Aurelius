"""Tests for src.agent.skill_trigger_engine."""

from __future__ import annotations

import pytest

from src.agent.skill_executor import (
    ExecutionResult,
    SkillContext,
    SkillExecutionError,
    SkillExecutor,
)
from src.agent.skill_trigger_engine import (
    TRIGGER_ENGINE_REGISTRY,
    MatchedSkill,
    SkillTriggerEngine,
    TriggerEngineError,
)
from src.mcp.skill_catalog import SkillCatalogError, SkillMetadata, SkillTrigger


class FakeCatalog:
    """Minimal fake catalog with find_by_trigger and get."""

    def __init__(self, skills: list[SkillMetadata] | None = None) -> None:
        self._skills: dict[str, SkillMetadata] = {}
        if skills:
            for s in skills:
                self._skills[s.skill_id] = s

    def find_by_trigger(self, text: str) -> list[SkillMetadata]:
        results: list[SkillMetadata] = []
        for skill in self._skills.values():
            for trigger in skill.triggers:
                if trigger.pattern.lower() in text.lower():
                    results.append(skill)
                    break
        return results

    def get(self, skill_id: str) -> SkillMetadata:
        if skill_id not in self._skills:
            raise SkillCatalogError(f"not found: {skill_id}")
        return self._skills[skill_id]

    def register(self, skill: SkillMetadata) -> None:
        self._skills[skill.skill_id] = skill


class EchoExecutor(SkillExecutor):
    """Executor that echoes skill_id + instructions."""

    def execute(
        self,
        skill_id: str,
        instructions: str,
        context: SkillContext | None = None,
    ) -> ExecutionResult:
        return ExecutionResult(
            output=f"executed {skill_id}: {instructions}",
            success=True,
            duration_ms=1.0,
        )


class FailExecutor(SkillExecutor):
    """Executor that always raises SkillExecutionError."""

    def execute(
        self,
        skill_id: str,
        instructions: str,
        context: SkillContext | None = None,
    ) -> ExecutionResult:
        raise SkillExecutionError("execution failed")


class TestTriggerEngineError:
    def test_is_exception(self) -> None:
        with pytest.raises(TriggerEngineError):
            raise TriggerEngineError("test")


class TestMatchedSkill:
    def test_fields(self) -> None:
        ms = MatchedSkill(
            skill_id="code-review",
            name="Code Review",
            trigger_pattern="review",
        )
        assert ms.confidence == 1.0


class TestMatch:
    def test_empty_catalog_returns_empty(self) -> None:
        engine = SkillTriggerEngine()
        result = engine.match("please review this code")
        assert result.matches == []

    def test_match_single_skill(self) -> None:
        catalog = FakeCatalog(
            [
                SkillMetadata(
                    skill_id="code-review",
                    name="Code Review",
                    version="1.0.0",
                    description="Reviews code",
                    triggers=[SkillTrigger("review", "code review")],
                    tags=["code"],
                )
            ]
        )
        engine = SkillTriggerEngine(skill_catalog=catalog)
        result = engine.match("please review this code")
        assert len(result.matches) == 1
        assert result.matches[0].skill_id == "code-review"

    def test_match_multiple_skills(self) -> None:
        catalog = FakeCatalog(
            [
                SkillMetadata(
                    skill_id="code-review",
                    name="Code Review",
                    version="1.0.0",
                    description="Reviews code",
                    triggers=[SkillTrigger("review", "")],
                    tags=["code"],
                ),
                SkillMetadata(
                    skill_id="test-gen",
                    name="Test Generator",
                    version="1.0.0",
                    description="Generates tests",
                    triggers=[SkillTrigger("test", "")],
                    tags=["test"],
                ),
            ]
        )
        engine = SkillTriggerEngine(skill_catalog=catalog)
        result = engine.match("review and test this code")
        assert len(result.matches) == 2

    def test_deduplication(self) -> None:
        catalog = FakeCatalog(
            [
                SkillMetadata(
                    skill_id="code-review",
                    name="Code Review",
                    version="1.0.0",
                    description="Reviews code",
                    triggers=[
                        SkillTrigger("review", ""),
                        SkillTrigger("code review", ""),
                    ],
                    tags=["code"],
                )
            ]
        )
        engine = SkillTriggerEngine(skill_catalog=catalog)
        result = engine.match("code review please")
        assert len(result.matches) == 1

    def test_case_insensitive(self) -> None:
        catalog = FakeCatalog(
            [
                SkillMetadata(
                    skill_id="code-review",
                    name="Code Review",
                    version="1.0.0",
                    description="Reviews code",
                    triggers=[SkillTrigger("REVIEW", "")],
                    tags=["code"],
                )
            ]
        )
        engine = SkillTriggerEngine(skill_catalog=catalog)
        result = engine.match("please Review this")
        assert len(result.matches) == 1

    def test_non_string_text_raises(self) -> None:
        engine = SkillTriggerEngine()
        with pytest.raises(TypeError, match="str"):
            engine.match(123)  # type: ignore[arg-type]

    def test_internal_skills_merged(self) -> None:
        engine = SkillTriggerEngine()
        engine.add_skill(
            SkillMetadata(
                skill_id="internal",
                name="Internal",
                version="1.0.0",
                description="Internal skill",
                triggers=[SkillTrigger("internal", "")],
                tags=["internal"],
            )
        )
        result = engine.match("run internal process")
        assert len(result.matches) == 1
        assert result.matches[0].skill_id == "internal"


class TestMatchAndExecute:
    def test_executes_matched_skills(self) -> None:
        catalog = FakeCatalog(
            [
                SkillMetadata(
                    skill_id="code-review",
                    name="Code Review",
                    version="1.0.0",
                    description="Review the code",
                    triggers=[SkillTrigger("review", "")],
                    tags=["code"],
                )
            ]
        )
        engine = SkillTriggerEngine(skill_catalog=catalog, executor=EchoExecutor())
        result = engine.match_and_execute("please review this")
        assert len(result.executed) == 1
        assert "code-review" in result.executed[0].output
        assert result.executed[0].success is True

    def test_execution_error_caught(self) -> None:
        catalog = FakeCatalog(
            [
                SkillMetadata(
                    skill_id="code-review",
                    name="Code Review",
                    version="1.0.0",
                    description="Review",
                    triggers=[SkillTrigger("review", "")],
                    tags=["code"],
                )
            ]
        )
        engine = SkillTriggerEngine(skill_catalog=catalog, executor=FailExecutor())
        result = engine.match_and_execute("review")
        assert len(result.executed) == 1
        assert result.executed[0].success is False
        assert "execution failed" in result.executed[0].output

    def test_no_matches_no_execution(self) -> None:
        engine = SkillTriggerEngine(executor=EchoExecutor())
        result = engine.match_and_execute("hello world")
        assert result.matches == []
        assert result.executed == []


class TestAddRemoveSkill:
    def test_add_skill(self) -> None:
        engine = SkillTriggerEngine()
        skill = SkillMetadata(
            skill_id="s1",
            name="Skill One",
            version="1.0.0",
            description="Desc",
            triggers=[SkillTrigger("one", "")],
            tags=["tag"],
        )
        engine.add_skill(skill)
        assert "s1" in engine._skills

    def test_add_skill_to_catalog(self) -> None:
        catalog = FakeCatalog()
        engine = SkillTriggerEngine(skill_catalog=catalog)
        skill = SkillMetadata(
            skill_id="s1",
            name="Skill One",
            version="1.0.0",
            description="Desc",
            triggers=[SkillTrigger("one", "")],
            tags=["tag"],
        )
        engine.add_skill(skill)
        assert catalog.get("s1").skill_id == "s1"

    def test_remove_skill(self) -> None:
        engine = SkillTriggerEngine()
        skill = SkillMetadata(
            skill_id="s1",
            name="Skill One",
            version="1.0.0",
            description="Desc",
            triggers=[SkillTrigger("one", "")],
            tags=["tag"],
        )
        engine.add_skill(skill)
        engine.remove_skill("s1")
        assert "s1" not in engine._skills

    def test_remove_unknown_raises(self) -> None:
        engine = SkillTriggerEngine()
        with pytest.raises(TriggerEngineError, match="not found"):
            engine.remove_skill("unknown")


class TestResolveInstructions:
    def test_from_internal_skills(self) -> None:
        engine = SkillTriggerEngine()
        engine.add_skill(
            SkillMetadata(
                skill_id="s1",
                name="Skill One",
                version="1.0.0",
                description="Internal desc",
                triggers=[SkillTrigger("one", "")],
                tags=["tag"],
            )
        )
        assert engine._resolve_instructions("s1") == "Internal desc"

    def test_fallback(self) -> None:
        engine = SkillTriggerEngine()
        assert engine._resolve_instructions("s1") == "Skill: s1"


class TestRegistry:
    def test_default_singleton_exists(self) -> None:
        assert "default" in TRIGGER_ENGINE_REGISTRY

    def test_custom_engine_in_registry(self) -> None:
        custom = SkillTriggerEngine()
        TRIGGER_ENGINE_REGISTRY["custom"] = custom
        assert TRIGGER_ENGINE_REGISTRY["custom"] is custom
        del TRIGGER_ENGINE_REGISTRY["custom"]
