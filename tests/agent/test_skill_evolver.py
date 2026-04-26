"""Tests for src.agent.skill_evolver."""

from __future__ import annotations

import pytest

from src.agent.skill_evolver import (
    DEFAULT_SKILL_EVOLVER,
    SKILL_EVOLVER_REGISTRY,
    CrystallizedSkill,
    SkillEvolutionError,
    SkillEvolver,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def evolver():
    return SkillEvolver()


SIMPLE_TASK = {
    "task_name": "greet_user",
    "description": "Greet a user by name",
    "steps": [
        {"action": "fetch username from context", "success": True},
        {"action": "say Hello {name}", "success": True},
        {"action": "log greeting", "success": True},
    ],
    "parameters": {"name": "Alice"},
    "context": {"user_id": "u1"},
}


# ---------------------------------------------------------------------------
# Crystallize
# ---------------------------------------------------------------------------


def test_crystallize_basic(evolver):
    skill = evolver.crystallize(SIMPLE_TASK)
    assert isinstance(skill, CrystallizedSkill)
    assert skill.name == "greet_user"
    assert len(skill.steps) == 3


def test_crystallize_filters_failed_steps(evolver):
    record = {
        "task_name": "mixed",
        "steps": [
            {"action": "step1", "success": True},
            {"action": "step2", "success": False},
            {"action": "step3", "success": True},
        ],
    }
    skill = evolver.crystallize(record)
    assert skill.steps == ["step1", "step3"]


def test_crystallize_parameter_templating(evolver):
    skill = evolver.crystallize(SIMPLE_TASK)
    assert any("{name}" in s for s in skill.steps)


def test_crystallize_preconditions_from_context(evolver):
    skill = evolver.crystallize(SIMPLE_TASK)
    assert "user_id" in skill.preconditions


def test_crystallize_unnamed_task(evolver):
    skill = evolver.crystallize({})
    assert skill.name == "unnamed_task"


def test_crystallize_parent_task_link(evolver):
    parent = evolver.crystallize({"task_name": "parent_task", "steps": []})
    child = evolver.crystallize(
        {"task_name": "child_task", "steps": [], "parent_task": "parent_task"}
    )
    assert child.parent_skill_id == parent.skill_id


def test_crystallize_no_parent_match(evolver):
    skill = evolver.crystallize(
        {"task_name": "orphan", "steps": [], "parent_task": "nonexistent"}
    )
    assert skill.parent_skill_id is None


# ---------------------------------------------------------------------------
# Match
# ---------------------------------------------------------------------------


def test_match_by_name(evolver):
    evolver.crystallize({"task_name": "deploy_app", "steps": []})
    results = evolver.match("deploy")
    assert len(results) == 1
    assert results[0].name == "deploy_app"


def test_match_by_description(evolver):
    evolver.crystallize({"task_name": "x", "description": "run tests", "steps": []})
    results = evolver.match("tests")
    assert len(results) == 1


def test_match_by_step(evolver):
    evolver.crystallize(
        {"task_name": "x", "steps": [{"action": "git push", "success": True}]}
    )
    results = evolver.match("git")
    assert len(results) == 1


def test_match_no_results(evolver):
    evolver.crystallize({"task_name": "x", "steps": []})
    assert evolver.match("yzabc") == []


def test_match_case_insensitive(evolver):
    evolver.crystallize({"task_name": "UPPER", "steps": []})
    assert len(evolver.match("upper")) == 1


# ---------------------------------------------------------------------------
# Refine
# ---------------------------------------------------------------------------


def test_refine_add_preconditions(evolver):
    skill = evolver.crystallize(SIMPLE_TASK)
    updated = evolver.refine(skill.skill_id, {"add_preconditions": ["api_key"]})
    assert "api_key" in updated.preconditions


def test_refine_replace_steps(evolver):
    skill = evolver.crystallize(SIMPLE_TASK)
    updated = evolver.refine(
        skill.skill_id, {"replace_steps": ["new_step_1", "new_step_2"]}
    )
    assert updated.steps == ["new_step_1", "new_step_2"]


def test_refine_add_steps(evolver):
    skill = evolver.crystallize(SIMPLE_TASK)
    updated = evolver.refine(skill.skill_id, {"add_steps": ["cleanup"]})
    assert "cleanup" in updated.steps


def test_refine_update_parameters(evolver):
    skill = evolver.crystallize(SIMPLE_TASK)
    updated = evolver.refine(skill.skill_id, {"update_parameters": {"greeting": "Hi"}})
    assert updated.parameters["greeting"] == "Hi"


def test_refine_increment_counts(evolver):
    skill = evolver.crystallize(SIMPLE_TASK)
    evolver.refine(skill.skill_id, {"increment_success": 3})
    evolver.refine(skill.skill_id, {"increment_failure": 1})
    assert skill.success_count == 3
    assert skill.failure_count == 1


def test_refine_updates_last_used(evolver):
    skill = evolver.crystallize(SIMPLE_TASK)
    assert skill.last_used is None
    evolver.refine(skill.skill_id, {})
    assert skill.last_used is not None


def test_refine_unknown_skill(evolver):
    with pytest.raises(SkillEvolutionError):
        evolver.refine("bad_id", {})


# ---------------------------------------------------------------------------
# Skill tree
# ---------------------------------------------------------------------------


def test_get_skill_tree_single_root(evolver):
    root = evolver.crystallize({"task_name": "root", "steps": []})
    child = evolver.crystallize(
        {"task_name": "child", "steps": [], "parent_task": "root"}
    )
    tree = evolver.get_skill_tree(root.skill_id)
    assert tree["name"] == "root"
    assert len(tree["children"]) == 1
    assert tree["children"][0]["name"] == "child"


def test_get_skill_tree_forest(evolver):
    evolver.crystallize({"task_name": "a", "steps": []})
    evolver.crystallize({"task_name": "b", "steps": []})
    forest = evolver.get_skill_tree()
    assert "forest" in forest
    assert len(forest["forest"]) == 2


def test_get_skill_tree_unknown_root(evolver):
    with pytest.raises(SkillEvolutionError):
        evolver.get_skill_tree("nope")


def test_get_skill_tree_deep_hierarchy(evolver):
    r = evolver.crystallize({"task_name": "r", "steps": []})
    c1 = evolver.crystallize({"task_name": "c1", "steps": [], "parent_task": "r"})
    c2 = evolver.crystallize({"task_name": "c2", "steps": [], "parent_task": "c1"})
    tree = evolver.get_skill_tree(r.skill_id)
    assert tree["children"][0]["children"][0]["name"] == "c2"


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def test_stats_empty(evolver):
    assert evolver.stats() == {
        "total_skills": 0,
        "avg_success_rate": 0.0,
        "deepest_tree_depth": 0,
    }


def test_stats_with_skills(evolver):
    s1 = evolver.crystallize(SIMPLE_TASK)
    evolver.refine(s1.skill_id, {"increment_success": 4, "increment_failure": 1})
    stats = evolver.stats()
    assert stats["total_skills"] == 1
    assert stats["avg_success_rate"] == 0.8
    assert stats["deepest_tree_depth"] == 1


def test_stats_deep_tree(evolver):
    r = evolver.crystallize({"task_name": "r", "steps": []})
    c = evolver.crystallize({"task_name": "c", "steps": [], "parent_task": "r"})
    gc = evolver.crystallize({"task_name": "gc", "steps": [], "parent_task": "c"})
    stats = evolver.stats()
    assert stats["deepest_tree_depth"] == 3


# ---------------------------------------------------------------------------
# Forget
# ---------------------------------------------------------------------------


def test_forget_removes_skill(evolver):
    skill = evolver.crystallize(SIMPLE_TASK)
    assert evolver.forget(skill.skill_id) is True
    assert evolver.list_skills() == []


def test_forget_orphans_children(evolver):
    root = evolver.crystallize({"task_name": "root", "steps": []})
    child = evolver.crystallize(
        {"task_name": "child", "steps": [], "parent_task": "root"}
    )
    evolver.forget(root.skill_id)
    assert child.parent_skill_id is None


def test_forget_unknown_returns_false(evolver):
    assert evolver.forget("nope") is False


# ---------------------------------------------------------------------------
# List skills
# ---------------------------------------------------------------------------


def test_list_skills(evolver):
    evolver.crystallize({"task_name": "a", "steps": []})
    evolver.crystallize({"task_name": "b", "steps": []})
    assert len(evolver.list_skills()) == 2


# ---------------------------------------------------------------------------
# Singleton / registry
# ---------------------------------------------------------------------------


def test_default_singleton():
    assert isinstance(DEFAULT_SKILL_EVOLVER, SkillEvolver)


def test_registry_contains_default():
    assert "default" in SKILL_EVOLVER_REGISTRY
    assert SKILL_EVOLVER_REGISTRY["default"] is DEFAULT_SKILL_EVOLVER
