"""Tests for src/agent/plan_observe_reflect.py."""

from __future__ import annotations

import pytest

from src.agent.plan_observe_reflect import (
    PLAN_OBSERVE_REFLECT_REGISTRY,
    LoopResult,
    LoopStep,
    PlanObserveReflect,
    StepStatus,
    _parse_steps,
)


# ---------------------------------------------------------------- _parse_steps

def test_parse_steps_dot_format():
    text = "1. first\n2. second\n3. third"
    assert _parse_steps(text, 10) == ["first", "second", "third"]


def test_parse_steps_paren_format():
    text = "1) first\n2) second"
    assert _parse_steps(text, 10) == ["first", "second"]


def test_parse_steps_mixed_numbering():
    text = "1. alpha\n2) beta"
    assert _parse_steps(text, 10) == ["alpha", "beta"]


def test_parse_steps_caps_at_max():
    text = "1. a\n2. b\n3. c\n4. d\n5. e"
    assert _parse_steps(text, 3) == ["a", "b", "c"]


def test_parse_steps_fallback_non_empty_lines():
    text = "do thing one\ndo thing two"
    assert _parse_steps(text, 10) == ["do thing one", "do thing two"]


def test_parse_steps_fallback_skips_comments():
    text = "# a comment\nreal line"
    assert _parse_steps(text, 10) == ["real line"]


def test_parse_steps_empty_text_returns_empty():
    assert _parse_steps("", 5) == []


def test_parse_steps_whitespace_only_returns_empty():
    assert _parse_steps("   \n\n  ", 5) == []


def test_parse_steps_ignores_preamble_before_numbered():
    text = "Here is the plan:\n1. do a\n2. do b"
    assert _parse_steps(text, 10) == ["do a", "do b"]


def test_parse_steps_preserves_extra_spacing_stripped():
    text = "1.   spaced out"
    assert _parse_steps(text, 10) == ["spaced out"]


# -------------------------------------------------------------------- LoopStep

def test_loopstep_is_frozen():
    step = LoopStep(index=1, description="x")
    with pytest.raises(Exception):
        step.description = "y"  # type: ignore[misc]


def test_loopstep_defaults():
    step = LoopStep(index=1, description="x")
    assert step.status == StepStatus.PENDING
    assert step.observation == ""
    assert step.error == ""


# ------------------------------------------------------------------- LoopResult

def test_loopresult_is_frozen():
    r = LoopResult(task="t", steps=[], reflection="", succeeded=True, total_steps=0)
    with pytest.raises(Exception):
        r.task = "other"  # type: ignore[misc]


def test_loopresult_fields():
    r = LoopResult(task="t", steps=[], reflection="done", succeeded=True, total_steps=0)
    assert r.task == "t"
    assert r.reflection == "done"
    assert r.succeeded is True
    assert r.total_steps == 0


# --------------------------------------------------------- PlanObserveReflect

def test_registry_default_is_class():
    assert PLAN_OBSERVE_REFLECT_REGISTRY["default"] is PlanObserveReflect


def test_plan_parses_numbered():
    p = PlanObserveReflect()
    steps = p.plan("task", planner=lambda t: "1. alpha\n2. beta")
    assert steps == ["alpha", "beta"]


def test_plan_fallback_when_no_numbered():
    p = PlanObserveReflect()
    steps = p.plan("my task", planner=lambda t: "")
    assert steps == ["my task"]


def test_act_passes_step_to_actor():
    p = PlanObserveReflect()
    captured = []
    obs = p.act("do thing", actor=lambda s: (captured.append(s) or "ok"))
    assert captured == ["do thing"]
    assert obs == "ok"


def test_act_handles_none_actor_return():
    p = PlanObserveReflect()
    obs = p.act("x", actor=lambda s: None)
    assert obs == ""


def test_reflect_summary_contains_task_and_steps():
    p = PlanObserveReflect()
    received = {}

    def reflector(summary: str) -> str:
        received["summary"] = summary
        return "reflection"

    steps = [LoopStep(index=1, description="do a", status=StepStatus.COMPLETED, observation="ok")]
    out = p.reflect("the task", steps, reflector)
    assert out == "reflection"
    assert "the task" in received["summary"]
    assert "do a" in received["summary"]
    assert "ok" in received["summary"]


def test_run_full_happy_path():
    p = PlanObserveReflect()
    planner = lambda t: "1. step A\n2. step B"
    actor = lambda s: f"observed:{s}"
    reflector = lambda summary: "all good"
    result = p.run("my task", planner, actor, reflector, max_steps=5)
    assert result.total_steps == 2
    assert result.succeeded is True
    assert result.reflection == "all good"
    assert result.steps[0].description == "step A"
    assert result.steps[0].observation == "observed:step A"
    assert result.steps[0].status == StepStatus.COMPLETED


def test_run_caps_steps_at_max():
    p = PlanObserveReflect()
    planner = lambda t: "1. a\n2. b\n3. c\n4. d\n5. e\n6. f"
    result = p.run("t", planner, lambda s: "ok", lambda s: "r", max_steps=3)
    assert result.total_steps == 3


def test_run_fallback_when_planner_returns_empty():
    p = PlanObserveReflect()
    result = p.run("lone task", lambda t: "", lambda s: "ok", lambda s: "r", max_steps=5)
    assert result.total_steps == 1
    assert result.steps[0].description == "lone task"


def test_run_step_failure_marks_failed():
    p = PlanObserveReflect()

    def actor(step: str) -> str:
        if "bad" in step:
            raise RuntimeError("boom")
        return "ok"

    planner = lambda t: "1. good step\n2. bad step\n3. good again"
    result = p.run("t", planner, actor, lambda s: "r", max_steps=5)
    assert result.total_steps == 3
    assert result.steps[0].status == StepStatus.COMPLETED
    assert result.steps[1].status == StepStatus.FAILED
    assert "boom" in result.steps[1].error
    assert result.steps[2].status == StepStatus.COMPLETED
    assert result.succeeded is False


def test_run_continues_after_failure():
    p = PlanObserveReflect()
    seen = []
    planner = lambda t: "1. a\n2. b\n3. c"

    def actor(step: str) -> str:
        seen.append(step)
        if step == "b":
            raise ValueError("nope")
        return "ok"

    p.run("t", planner, actor, lambda s: "", max_steps=5)
    assert seen == ["a", "b", "c"]


def test_run_reflection_called_with_summary():
    p = PlanObserveReflect()
    captured = {}

    def reflector(summary: str) -> str:
        captured["s"] = summary
        return "done"

    p.run("root task", lambda t: "1. x", lambda s: "ok", reflector, max_steps=5)
    assert "root task" in captured["s"]
    assert "x" in captured["s"]


def test_run_reflection_value_in_result():
    p = PlanObserveReflect()
    result = p.run("t", lambda t: "1. x", lambda s: "ok", lambda s: "REFL", max_steps=5)
    assert result.reflection == "REFL"


def test_run_total_steps_matches_steps_list():
    p = PlanObserveReflect()
    result = p.run("t", lambda t: "1. a\n2. b", lambda s: "ok", lambda s: "", max_steps=5)
    assert result.total_steps == len(result.steps)


def test_run_task_field_preserved():
    p = PlanObserveReflect()
    result = p.run("the original", lambda t: "1. x", lambda s: "ok", lambda s: "", max_steps=5)
    assert result.task == "the original"


def test_run_all_failed_not_succeeded():
    p = PlanObserveReflect()

    def actor(s: str) -> str:
        raise RuntimeError("x")

    result = p.run("t", lambda t: "1. a\n2. b", actor, lambda s: "", max_steps=5)
    assert result.succeeded is False
    assert all(s.status == StepStatus.FAILED for s in result.steps)
