"""Unit tests for src/agent/skill_executor.py."""

from __future__ import annotations

import pytest

from src.agent.skill_executor import (
    DEFAULT_SKILL_EXECUTOR,
    SKILL_EXECUTOR_REGISTRY,
    ExecutionResult,
    SkillContext,
    SkillExecutionError,
    SkillExecutor,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh() -> SkillExecutor:
    """Return a fresh executor to avoid test pollution."""
    return SkillExecutor()


# ---------------------------------------------------------------------------
# 1. test_basic_variable_substitution
# ---------------------------------------------------------------------------


def test_basic_variable_substitution():
    exe = _fresh()
    ctx = SkillContext(variables={"name": "Aurelius", "mode": "code"})
    result = exe.execute("test", "Hello {name}, mode is {mode}", ctx)
    assert result.output == "Hello Aurelius, mode is code"
    assert result.success is True


# ---------------------------------------------------------------------------
# 2. test_missing_variables_left_as_is
# ---------------------------------------------------------------------------


def test_missing_variables_left_as_is():
    exe = _fresh()
    ctx = SkillContext(variables={"known": "value"})
    result = exe.execute("test", "{known} and {unknown}", ctx)
    assert result.output == "value and {unknown}"


# ---------------------------------------------------------------------------
# 3. test_memory_placeholder
# ---------------------------------------------------------------------------


def test_memory_placeholder():
    exe = _fresh()
    ctx = SkillContext(memory=["line one", "line two", "line three"])
    result = exe.execute("test", "History:\n{memory}\nEnd", ctx)
    assert result.output == "History:\nline one\nline two\nline three\nEnd"


# ---------------------------------------------------------------------------
# 4. test_tools_placeholder
# ---------------------------------------------------------------------------


def test_tools_placeholder():
    exe = _fresh()
    ctx = SkillContext(available_tools=["bash", "read_file", "write_file"])
    result = exe.execute("test", "Tools: {tools}", ctx)
    assert result.output == "Tools: bash, read_file, write_file"


# ---------------------------------------------------------------------------
# 5. test_empty_context_uses_defaults
# ---------------------------------------------------------------------------


def test_empty_context_uses_defaults():
    exe = _fresh()
    result = exe.execute("test", "plain text")
    assert result.output == "plain text"
    assert result.success is True


# ---------------------------------------------------------------------------
# 6. test_empty_skill_id_raises
# ---------------------------------------------------------------------------


def test_empty_skill_id_raises():
    exe = _fresh()
    with pytest.raises(
        SkillExecutionError, match="skill_id must be a non-empty string"
    ):
        exe.execute("", "instructions")


# ---------------------------------------------------------------------------
# 7. test_whitespace_only_skill_id_raises
# ---------------------------------------------------------------------------


def test_whitespace_only_skill_id_raises():
    exe = _fresh()
    with pytest.raises(
        SkillExecutionError, match="skill_id must be a non-empty string"
    ):
        exe.execute("   ", "instructions")


# ---------------------------------------------------------------------------
# 8. test_empty_instructions_raises
# ---------------------------------------------------------------------------


def test_empty_instructions_raises():
    exe = _fresh()
    with pytest.raises(
        SkillExecutionError, match="instructions must be a non-empty string"
    ):
        exe.execute("test", "")


# ---------------------------------------------------------------------------
# 9. test_whitespace_only_instructions_raises
# ---------------------------------------------------------------------------


def test_whitespace_only_instructions_raises():
    exe = _fresh()
    with pytest.raises(
        SkillExecutionError, match="instructions must be a non-empty string"
    ):
        exe.execute("test", "   ")


# ---------------------------------------------------------------------------
# 10. test_too_long_instructions_raises
# ---------------------------------------------------------------------------


def test_too_long_instructions_raises():
    exe = _fresh()
    long_text = "x" * 100_001
    with pytest.raises(SkillExecutionError, match="instructions exceed max length"):
        exe.execute("test", long_text)


# ---------------------------------------------------------------------------
# 11. test_duration_is_non_negative
# ---------------------------------------------------------------------------


def test_duration_is_non_negative():
    exe = _fresh()
    result = exe.execute("test", "some instructions")
    assert isinstance(result, ExecutionResult)
    assert result.duration_ms >= 0.0


# ---------------------------------------------------------------------------
# 12. test_default_singleton_exists
# ---------------------------------------------------------------------------


def test_default_singleton_exists():
    assert isinstance(DEFAULT_SKILL_EXECUTOR, SkillExecutor)


# ---------------------------------------------------------------------------
# 13. test_registry_singleton_exists
# ---------------------------------------------------------------------------


def test_registry_singleton_exists():
    assert "default" in SKILL_EXECUTOR_REGISTRY
    assert SKILL_EXECUTOR_REGISTRY["default"] is DEFAULT_SKILL_EXECUTOR


# ---------------------------------------------------------------------------
# 14. test_custom_executor_in_registry
# ---------------------------------------------------------------------------


def test_custom_executor_in_registry():
    custom = SkillExecutor()
    SKILL_EXECUTOR_REGISTRY["custom"] = custom
    assert SKILL_EXECUTOR_REGISTRY["custom"] is custom
    # Clean up to avoid pollution
    del SKILL_EXECUTOR_REGISTRY["custom"]


# ---------------------------------------------------------------------------
# 15. test_multiple_variables
# ---------------------------------------------------------------------------


def test_multiple_variables():
    exe = _fresh()
    ctx = SkillContext(
        variables={"a": "1", "b": "2", "c": "3"},
    )
    result = exe.execute("test", "{a}+{b}+{c}={a}{b}{c}", ctx)
    assert result.output == "1+2+3=123"


# ---------------------------------------------------------------------------
# 16. test_no_placeholders_passthrough
# ---------------------------------------------------------------------------


def test_no_placeholders_passthrough():
    exe = _fresh()
    text = "Just a normal string without any placeholders."
    result = exe.execute("test", text)
    assert result.output == text


# ---------------------------------------------------------------------------
# 17. test_memory_and_tools_with_no_items
# ---------------------------------------------------------------------------


def test_memory_and_tools_with_no_items():
    exe = _fresh()
    ctx = SkillContext(memory=[], available_tools=[])
    result = exe.execute("test", "m={memory}|t={tools}", ctx)
    assert result.output == "m=|t="


# ---------------------------------------------------------------------------
# 18. test_mixed_placeholders
# ---------------------------------------------------------------------------


def test_mixed_placeholders():
    exe = _fresh()
    ctx = SkillContext(
        variables={"user": "Marcus"},
        memory=["Thought 1", "Thought 2"],
        available_tools=["search"],
    )
    result = exe.execute("test", "{user}\n{memory}\n{tools}", ctx)
    assert result.output == "Marcus\nThought 1\nThought 2\nsearch"


# ---------------------------------------------------------------------------
# 19. test_curly_braces_without_placeholder_ignored
# ---------------------------------------------------------------------------


def test_curly_braces_without_placeholder_ignored():
    exe = _fresh()
    text = 'JSON: {"key": "value"} and {not_closed'
    result = exe.execute("test", text)
    assert result.output == text


# ---------------------------------------------------------------------------
# 20. test_context_metadata_defaults
# ---------------------------------------------------------------------------


def test_context_metadata_defaults():
    ctx = SkillContext()
    assert ctx.variables == {}
    assert ctx.memory == []
    assert ctx.available_tools == []
    assert ctx.metadata == {}
