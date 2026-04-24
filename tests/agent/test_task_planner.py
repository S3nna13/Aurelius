"""
Tests for src/agent/task_planner.py  (≥28 tests)
"""

import pytest
from src.agent.task_planner import (
    SubTask,
    TaskPlan,
    TaskPlanner,
    TASK_PLANNER_REGISTRY,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def planner():
    return TaskPlanner()


# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------

def test_registry_key_exists():
    assert "default" in TASK_PLANNER_REGISTRY


def test_registry_value_is_task_planner():
    assert TASK_PLANNER_REGISTRY["default"] is TaskPlanner


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

def test_max_depth_stored():
    tp = TaskPlanner(max_depth=3)
    assert tp.max_depth == 3


def test_max_subtasks_stored():
    tp = TaskPlanner(max_subtasks=10)
    assert tp.max_subtasks == 10


def test_default_max_depth():
    tp = TaskPlanner()
    assert tp.max_depth == 5


def test_default_max_subtasks():
    tp = TaskPlanner()
    assert tp.max_subtasks == 20


# ---------------------------------------------------------------------------
# plan() — keyword routing
# ---------------------------------------------------------------------------

def test_plan_research_has_3_subtasks(planner):
    plan = planner.plan("research climate change")
    assert len(plan.subtasks) == 3


def test_plan_find_has_3_subtasks(planner):
    plan = planner.plan("find the best approach")
    assert len(plan.subtasks) == 3


def test_plan_write_has_4_subtasks(planner):
    plan = planner.plan("write a report")
    assert len(plan.subtasks) == 4


def test_plan_draft_has_4_subtasks(planner):
    plan = planner.plan("draft the proposal")
    assert len(plan.subtasks) == 4


def test_plan_fix_has_4_subtasks(planner):
    plan = planner.plan("fix the memory leak")
    assert len(plan.subtasks) == 4


def test_plan_debug_has_4_subtasks(planner):
    plan = planner.plan("debug the crash")
    assert len(plan.subtasks) == 4


def test_plan_analyze_has_4_subtasks(planner):
    plan = planner.plan("analyze the logs")
    assert len(plan.subtasks) == 4


def test_plan_evaluate_has_4_subtasks(planner):
    plan = planner.plan("evaluate performance")
    assert len(plan.subtasks) == 4


def test_plan_default_has_3_subtasks(planner):
    plan = planner.plan("do something generic")
    assert len(plan.subtasks) == 3


def test_plan_goal_stored(planner):
    plan = planner.plan("research X")
    assert plan.goal == "research X"


def test_plan_created_at_is_float(planner):
    plan = planner.plan("research X")
    assert isinstance(plan.created_at, float)


# ---------------------------------------------------------------------------
# plan() — sequential dependencies
# ---------------------------------------------------------------------------

def test_first_subtask_no_deps(planner):
    plan = planner.plan("write report")
    assert plan.subtasks[0].depends_on == []


def test_second_subtask_depends_on_first(planner):
    plan = planner.plan("write report")
    assert plan.subtasks[1].depends_on == ["t1"]


def test_third_subtask_depends_on_second(planner):
    plan = planner.plan("write report")
    assert plan.subtasks[2].depends_on == ["t2"]


def test_task_ids_sequential(planner):
    plan = planner.plan("write report")
    ids = [st.task_id for st in plan.subtasks]
    assert ids == ["t1", "t2", "t3", "t4"]


# ---------------------------------------------------------------------------
# mark_complete()
# ---------------------------------------------------------------------------

def test_mark_complete_changes_status(planner):
    plan = planner.plan("research X")
    planner.mark_complete(plan, "t1")
    assert plan.subtasks[0].status == "completed"


def test_mark_complete_stores_result(planner):
    plan = planner.plan("research X")
    planner.mark_complete(plan, "t1", result="done!")
    assert plan.subtasks[0].result == "done!"


def test_mark_complete_only_targets_given_id(planner):
    plan = planner.plan("research X")
    planner.mark_complete(plan, "t1")
    assert plan.subtasks[1].status == "pending"


# ---------------------------------------------------------------------------
# next_ready()
# ---------------------------------------------------------------------------

def test_next_ready_first_task_no_deps(planner):
    plan = planner.plan("write report")
    ready = planner.next_ready(plan)
    assert len(ready) == 1
    assert ready[0].task_id == "t1"


def test_next_ready_after_completing_dep(planner):
    plan = planner.plan("write report")
    planner.mark_complete(plan, "t1")
    ready = planner.next_ready(plan)
    assert any(st.task_id == "t2" for st in ready)


def test_next_ready_no_tasks_when_all_blocked(planner):
    plan = planner.plan("write report")
    # t2 depends on t1 which is still pending, so only t1 is ready
    ready = planner.next_ready(plan)
    task_ids = [st.task_id for st in ready]
    assert "t2" not in task_ids


# ---------------------------------------------------------------------------
# is_done()
# ---------------------------------------------------------------------------

def test_is_done_all_completed(planner):
    plan = planner.plan("research X")
    for st in plan.subtasks:
        planner.mark_complete(plan, st.task_id)
    assert planner.is_done(plan) is True


def test_is_done_partial_false(planner):
    plan = planner.plan("research X")
    planner.mark_complete(plan, "t1")
    assert planner.is_done(plan) is False


# ---------------------------------------------------------------------------
# progress()
# ---------------------------------------------------------------------------

def test_progress_initial(planner):
    plan = planner.plan("research X")
    p = planner.progress(plan)
    assert p["total"] == 3
    assert p["completed"] == 0
    assert p["pct"] == 0.0


def test_progress_pct_after_completion(planner):
    plan = planner.plan("research X")
    planner.mark_complete(plan, "t1")
    p = planner.progress(plan)
    assert p["completed"] == 1
    assert abs(p["pct"] - 100.0 / 3) < 0.01
