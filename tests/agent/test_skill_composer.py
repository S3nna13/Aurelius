"""Tests for src.agent.skill_composer."""

from __future__ import annotations

import re

import pytest

from src.agent.skill_composer import (
    SKILL_COMPOSER_REGISTRY,
    CompositionStep,
    SkillComposer,
    SkillCompositionError,
)
from src.agent.skill_executor import (
    ExecutionResult,
    SkillContext,
    SkillExecutionError,
    SkillExecutor,
)

_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


class FakeExecutor(SkillExecutor):
    """Executor that echoes instructions prefixed with skill_id,
    with simple variable substitution.
    """

    def execute(
        self,
        skill_id: str,
        instructions: str,
        context: SkillContext | None = None,
    ) -> ExecutionResult:
        ctx = context if context is not None else SkillContext()

        def _replacer(match: re.Match[str]) -> str:
            key = match.group(1)
            if key == "memory":
                return "\n".join(ctx.memory)
            if key == "tools":
                return ", ".join(ctx.available_tools)
            return ctx.variables.get(key, match.group(0))

        rendered = _PLACEHOLDER_RE.sub(_replacer, instructions)
        output = f"[{skill_id}] {rendered}"
        return ExecutionResult(
            output=output,
            success=True,
            duration_ms=1.0,
        )


class FailingExecutor(SkillExecutor):
    """Executor that fails every other call."""

    def __init__(self) -> None:
        self._call_count = 0

    def execute(
        self,
        skill_id: str,
        instructions: str,
        context: SkillContext | None = None,
    ) -> ExecutionResult:
        self._call_count += 1
        if self._call_count % 2 == 0:
            return ExecutionResult(
                output="fail",
                success=False,
                duration_ms=1.0,
            )
        return ExecutionResult(
            output=f"[{skill_id}] {instructions}",
            success=True,
            duration_ms=1.0,
        )


class RaisingExecutor(SkillExecutor):
    """Executor that raises SkillExecutionError."""

    def execute(
        self,
        skill_id: str,
        instructions: str,
        context: SkillContext | None = None,
    ) -> ExecutionResult:
        raise SkillExecutionError("boom")


class TestSkillCompositionError:
    def test_is_exception(self) -> None:
        with pytest.raises(SkillCompositionError):
            raise SkillCompositionError("test")


class TestCompositionStep:
    def test_defaults(self) -> None:
        step = CompositionStep(skill_id="s1", instructions="do x")
        assert step.skill_id == "s1"
        assert step.instructions == "do x"
        assert step.output_key == "output"

    def test_custom_output_key(self) -> None:
        step = CompositionStep(skill_id="s1", instructions="do x", output_key="result")
        assert step.output_key == "result"


class TestSkillComposerInit:
    def test_default_executor(self) -> None:
        composer = SkillComposer()
        assert composer.executor is not None

    def test_custom_executor(self) -> None:
        fake = FakeExecutor()
        composer = SkillComposer(executor=fake)
        assert composer.executor is fake


class TestExecutePipeline:
    def test_single_step(self) -> None:
        composer = SkillComposer(executor=FakeExecutor())
        result = composer.execute_pipeline([CompositionStep(skill_id="s1", instructions="hello")])
        assert result.overall_success is True
        assert "output" in result.outputs
        assert result.outputs["output"].output == "[s1] hello"

    def test_multi_step_context_passing(self) -> None:
        composer = SkillComposer(executor=FakeExecutor())
        steps = [
            CompositionStep(skill_id="s1", instructions="step1", output_key="step1_out"),
            CompositionStep(skill_id="s2", instructions="{step1_out}", output_key="step2_out"),
        ]
        result = composer.execute_pipeline(steps)
        assert result.overall_success is True
        assert result.outputs["step1_out"].output == "[s1] step1"
        assert result.outputs["step2_out"].output == "[s2] [s1] step1"

    def test_fail_fast(self) -> None:
        composer = SkillComposer(executor=FailingExecutor())
        steps = [
            CompositionStep(skill_id="s1", instructions="a", output_key="r1"),
            CompositionStep(skill_id="s2", instructions="b", output_key="r2"),
            CompositionStep(skill_id="s3", instructions="c", output_key="r3"),
        ]
        result = composer.execute_pipeline(steps)
        assert result.overall_success is False
        # Only first step succeeded; second failed, pipeline stopped
        assert len(result.outputs) == 2
        assert result.outputs["r1"].success is True
        assert result.outputs["r2"].success is False

    def test_skill_execution_error_caught(self) -> None:
        composer = SkillComposer(executor=RaisingExecutor())
        result = composer.execute_pipeline([CompositionStep(skill_id="s1", instructions="x")])
        assert result.overall_success is False
        assert result.outputs["output"].success is False
        assert "boom" in result.outputs["output"].output

    def test_empty_steps_raises(self) -> None:
        composer = SkillComposer(executor=FakeExecutor())
        with pytest.raises(SkillCompositionError, match="empty"):
            composer.execute_pipeline([])

    def test_non_list_steps_raises(self) -> None:
        composer = SkillComposer(executor=FakeExecutor())
        with pytest.raises(TypeError, match="list"):
            composer.execute_pipeline("not a list")  # type: ignore[arg-type]

    def test_duration_non_negative(self) -> None:
        composer = SkillComposer(executor=FakeExecutor())
        result = composer.execute_pipeline([CompositionStep(skill_id="s1", instructions="hello")])
        assert result.total_duration_ms >= 0

    def test_same_output_key_overwrites(self) -> None:
        composer = SkillComposer(executor=FakeExecutor())
        steps = [
            CompositionStep(skill_id="s1", instructions="first", output_key="out"),
            CompositionStep(skill_id="s2", instructions="second", output_key="out"),
        ]
        result = composer.execute_pipeline(steps)
        assert result.overall_success is True
        assert result.outputs["out"].output == "[s2] second"

    def test_with_initial_context(self) -> None:
        composer = SkillComposer(executor=FakeExecutor())
        context = SkillContext(variables={"prefix": "hello"})
        result = composer.execute_pipeline(
            [CompositionStep(skill_id="s1", instructions="{prefix} world")],
            initial_context=context,
        )
        assert result.outputs["output"].output == "[s1] hello world"


class TestExecuteParallel:
    def test_parallel_independent(self) -> None:
        composer = SkillComposer(executor=FakeExecutor())
        steps = [
            CompositionStep(skill_id="s1", instructions="a", output_key="r1"),
            CompositionStep(skill_id="s2", instructions="b", output_key="r2"),
        ]
        result = composer.execute_parallel(steps)
        assert result.overall_success is True
        assert result.outputs["r1"].output == "[s1] a"
        assert result.outputs["r2"].output == "[s2] b"

    def test_parallel_failure(self) -> None:
        composer = SkillComposer(executor=FailingExecutor())
        steps = [
            CompositionStep(skill_id="s1", instructions="a", output_key="r1"),
            CompositionStep(skill_id="s2", instructions="b", output_key="r2"),
        ]
        result = composer.execute_parallel(steps)
        assert result.overall_success is False
        # Both run independently; first succeeds, second fails
        assert result.outputs["r1"].success is True
        assert result.outputs["r2"].success is False

    def test_empty_steps_raises(self) -> None:
        composer = SkillComposer(executor=FakeExecutor())
        with pytest.raises(SkillCompositionError, match="empty"):
            composer.execute_parallel([])

    def test_non_list_steps_raises(self) -> None:
        composer = SkillComposer(executor=FakeExecutor())
        with pytest.raises(TypeError, match="list"):
            composer.execute_parallel(123)  # type: ignore[arg-type]


class TestRegistry:
    def test_default_singleton_exists(self) -> None:
        assert "default" in SKILL_COMPOSER_REGISTRY

    def test_custom_composer_in_registry(self) -> None:
        custom = SkillComposer(executor=FakeExecutor())
        SKILL_COMPOSER_REGISTRY["custom"] = custom
        assert SKILL_COMPOSER_REGISTRY["custom"] is custom
        del SKILL_COMPOSER_REGISTRY["custom"]
