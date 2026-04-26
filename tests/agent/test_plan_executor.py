"""Tests for src/agent/plan_executor.py."""

from src.agent.plan_executor import (
    PLAN_EXECUTOR_REGISTRY,
    PlanExecutor,
    PlanStep,
    StepStatus,
)

# ---------------------------------------------------------------------------
# StepStatus enum values (5 tests)
# ---------------------------------------------------------------------------


def test_status_pending_value():
    assert StepStatus.PENDING == "pending"


def test_status_running_value():
    assert StepStatus.RUNNING == "running"


def test_status_done_value():
    assert StepStatus.DONE == "done"


def test_status_failed_value():
    assert StepStatus.FAILED == "failed"


def test_status_skipped_value():
    assert StepStatus.SKIPPED == "skipped"


# ---------------------------------------------------------------------------
# PlanStep defaults
# ---------------------------------------------------------------------------


def test_planstep_defaults():
    s = PlanStep(id="a", description="do something")
    assert s.depends_on == []
    assert s.status == StepStatus.PENDING
    assert s.result == ""


def test_planstep_id_stored():
    s = PlanStep(id="xyz", description="desc")
    assert s.id == "xyz"


def test_planstep_description_stored():
    s = PlanStep(id="a", description="hello")
    assert s.description == "hello"


def test_planstep_custom_status():
    s = PlanStep(id="a", description="d", status=StepStatus.RUNNING)
    assert s.status == StepStatus.RUNNING


def test_planstep_depends_on_stored():
    s = PlanStep(id="b", description="d", depends_on=["a"])
    assert s.depends_on == ["a"]


def test_planstep_depends_on_independent():
    # mutable default is independent between instances
    s1 = PlanStep(id="a", description="a")
    s2 = PlanStep(id="b", description="b")
    s1.depends_on.append("x")
    assert s2.depends_on == []


# ---------------------------------------------------------------------------
# PlanExecutor.add_step + ready_steps
# ---------------------------------------------------------------------------


def test_add_step_single():
    pe = PlanExecutor()
    step = PlanStep(id="1", description="first")
    pe.add_step(step)
    assert pe.ready_steps() == [step]


def test_add_step_multiple():
    pe = PlanExecutor()
    pe.add_step(PlanStep(id="a", description="A"))
    pe.add_step(PlanStep(id="b", description="B"))
    ids = {s.id for s in pe.ready_steps()}
    assert ids == {"a", "b"}


def test_ready_steps_no_steps():
    pe = PlanExecutor()
    assert pe.ready_steps() == []


def test_ready_steps_running_not_ready():
    pe = PlanExecutor()
    step = PlanStep(id="a", description="A", status=StepStatus.RUNNING)
    pe.add_step(step)
    assert pe.ready_steps() == []


def test_ready_steps_done_not_ready():
    pe = PlanExecutor()
    step = PlanStep(id="a", description="A", status=StepStatus.DONE)
    pe.add_step(step)
    assert pe.ready_steps() == []


# ---------------------------------------------------------------------------
# ready_steps: dependency resolution
# ---------------------------------------------------------------------------


def test_ready_steps_unresolved_dep_not_ready():
    pe = PlanExecutor()
    pe.add_step(PlanStep(id="a", description="A"))
    pe.add_step(PlanStep(id="b", description="B", depends_on=["a"]))
    # "b" should not be ready because "a" is still PENDING
    ready_ids = {s.id for s in pe.ready_steps()}
    assert "b" not in ready_ids


def test_ready_steps_resolved_dep_ready():
    pe = PlanExecutor()
    pe.add_step(PlanStep(id="a", description="A", status=StepStatus.DONE))
    pe.add_step(PlanStep(id="b", description="B", depends_on=["a"]))
    ready_ids = {s.id for s in pe.ready_steps()}
    assert "b" in ready_ids


def test_ready_steps_partial_dep_not_ready():
    pe = PlanExecutor()
    pe.add_step(PlanStep(id="a", description="A", status=StepStatus.DONE))
    pe.add_step(PlanStep(id="b", description="B"))
    pe.add_step(PlanStep(id="c", description="C", depends_on=["a", "b"]))
    ready_ids = {s.id for s in pe.ready_steps()}
    assert "c" not in ready_ids


def test_ready_steps_all_deps_done():
    pe = PlanExecutor()
    pe.add_step(PlanStep(id="a", description="A", status=StepStatus.DONE))
    pe.add_step(PlanStep(id="b", description="B", status=StepStatus.DONE))
    pe.add_step(PlanStep(id="c", description="C", depends_on=["a", "b"]))
    ready_ids = {s.id for s in pe.ready_steps()}
    assert "c" in ready_ids


# ---------------------------------------------------------------------------
# mark_done
# ---------------------------------------------------------------------------


def test_mark_done_returns_true_valid():
    pe = PlanExecutor()
    pe.add_step(PlanStep(id="a", description="A"))
    assert pe.mark_done("a") is True


def test_mark_done_returns_false_unknown():
    pe = PlanExecutor()
    assert pe.mark_done("nonexistent") is False


def test_mark_done_sets_status():
    pe = PlanExecutor()
    pe.add_step(PlanStep(id="a", description="A"))
    pe.mark_done("a")
    assert pe._steps["a"].status == StepStatus.DONE


def test_mark_done_stores_result():
    pe = PlanExecutor()
    pe.add_step(PlanStep(id="a", description="A"))
    pe.mark_done("a", result="output text")
    assert pe._steps["a"].result == "output text"


def test_mark_done_empty_result_default():
    pe = PlanExecutor()
    pe.add_step(PlanStep(id="a", description="A"))
    pe.mark_done("a")
    assert pe._steps["a"].result == ""


# ---------------------------------------------------------------------------
# mark_failed
# ---------------------------------------------------------------------------


def test_mark_failed_returns_true_valid():
    pe = PlanExecutor()
    pe.add_step(PlanStep(id="a", description="A"))
    assert pe.mark_failed("a") is True


def test_mark_failed_returns_false_unknown():
    pe = PlanExecutor()
    assert pe.mark_failed("nope") is False


def test_mark_failed_sets_status():
    pe = PlanExecutor()
    pe.add_step(PlanStep(id="a", description="A"))
    pe.mark_failed("a", result="err")
    assert pe._steps["a"].status == StepStatus.FAILED


def test_mark_failed_stores_result():
    pe = PlanExecutor()
    pe.add_step(PlanStep(id="a", description="A"))
    pe.mark_failed("a", result="boom")
    assert pe._steps["a"].result == "boom"


# ---------------------------------------------------------------------------
# skip_dependents
# ---------------------------------------------------------------------------


def test_skip_dependents_direct():
    pe = PlanExecutor()
    pe.add_step(PlanStep(id="a", description="A"))
    pe.add_step(PlanStep(id="b", description="B", depends_on=["a"]))
    skipped = pe.skip_dependents("a")
    assert "b" in skipped
    assert pe._steps["b"].status == StepStatus.SKIPPED


def test_skip_dependents_transitive():
    pe = PlanExecutor()
    pe.add_step(PlanStep(id="a", description="A"))
    pe.add_step(PlanStep(id="b", description="B", depends_on=["a"]))
    pe.add_step(PlanStep(id="c", description="C", depends_on=["b"]))
    skipped = pe.skip_dependents("a")
    assert set(skipped) == {"b", "c"}
    assert pe._steps["c"].status == StepStatus.SKIPPED


def test_skip_dependents_no_dependents():
    pe = PlanExecutor()
    pe.add_step(PlanStep(id="a", description="A"))
    skipped = pe.skip_dependents("a")
    assert skipped == []


def test_skip_dependents_does_not_skip_unrelated():
    pe = PlanExecutor()
    pe.add_step(PlanStep(id="a", description="A"))
    pe.add_step(PlanStep(id="b", description="B", depends_on=["a"]))
    pe.add_step(PlanStep(id="c", description="C"))
    pe.skip_dependents("a")
    assert pe._steps["c"].status == StepStatus.PENDING


# ---------------------------------------------------------------------------
# is_complete
# ---------------------------------------------------------------------------


def test_is_complete_all_done():
    pe = PlanExecutor()
    pe.add_step(PlanStep(id="a", description="A", status=StepStatus.DONE))
    assert pe.is_complete() is True


def test_is_complete_mixed_terminal():
    pe = PlanExecutor()
    pe.add_step(PlanStep(id="a", description="A", status=StepStatus.DONE))
    pe.add_step(PlanStep(id="b", description="B", status=StepStatus.FAILED))
    pe.add_step(PlanStep(id="c", description="C", status=StepStatus.SKIPPED))
    assert pe.is_complete() is True


def test_is_complete_pending_not_complete():
    pe = PlanExecutor()
    pe.add_step(PlanStep(id="a", description="A", status=StepStatus.DONE))
    pe.add_step(PlanStep(id="b", description="B"))
    assert pe.is_complete() is False


def test_is_complete_running_not_complete():
    pe = PlanExecutor()
    pe.add_step(PlanStep(id="a", description="A", status=StepStatus.RUNNING))
    assert pe.is_complete() is False


def test_is_complete_empty():
    pe = PlanExecutor()
    assert pe.is_complete() is True


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------


def test_summary_all_pending():
    pe = PlanExecutor()
    pe.add_step(PlanStep(id="a", description="A"))
    pe.add_step(PlanStep(id="b", description="B"))
    s = pe.summary()
    assert s["total"] == 2
    assert s["pending"] == 2
    assert s["done"] == 0


def test_summary_mixed():
    pe = PlanExecutor()
    pe.add_step(PlanStep(id="a", description="A", status=StepStatus.DONE))
    pe.add_step(PlanStep(id="b", description="B", status=StepStatus.FAILED))
    pe.add_step(PlanStep(id="c", description="C", status=StepStatus.SKIPPED))
    pe.add_step(PlanStep(id="d", description="D"))
    s = pe.summary()
    assert s == {"total": 4, "done": 1, "failed": 1, "skipped": 1, "pending": 1}


def test_summary_empty():
    pe = PlanExecutor()
    s = pe.summary()
    assert s["total"] == 0


def test_summary_has_all_keys():
    pe = PlanExecutor()
    s = pe.summary()
    assert set(s.keys()) == {"total", "done", "failed", "skipped", "pending"}


# ---------------------------------------------------------------------------
# PLAN_EXECUTOR_REGISTRY
# ---------------------------------------------------------------------------


def test_registry_has_default_key():
    assert "default" in PLAN_EXECUTOR_REGISTRY


def test_registry_default_is_plan_executor():
    assert PLAN_EXECUTOR_REGISTRY["default"] is PlanExecutor


# ---------------------------------------------------------------------------
# Constructor with initial steps
# ---------------------------------------------------------------------------


def test_constructor_with_steps():
    steps = [PlanStep(id="a", description="A"), PlanStep(id="b", description="B")]
    pe = PlanExecutor(steps=steps)
    assert len(pe._steps) == 2


def test_constructor_no_steps():
    pe = PlanExecutor()
    assert pe._steps == {}
