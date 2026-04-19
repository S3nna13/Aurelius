"""Integration tests for the Tree-of-Thoughts planner surface.

Ensures the ``beam_plan`` entry is wired into ``AGENT_LOOP_REGISTRY``
without disturbing existing entries (notably ``react``), and that a
planner built from the registry produces a usable execution queue end
to end.
"""

from __future__ import annotations

from src.agent import AGENT_LOOP_REGISTRY, BeamPlanner, PlanNode


def _echo_generate(task, path):
    # Each call emits exactly two deterministic proposals echoing the
    # current depth and task, mimicking a toy language-model planner.
    last = path[-1]
    return [
        {
            "description": f"{task}|d{last.depth + 1}|branch-{i}",
            "expected_tool": "noop" if i == 0 else "echo",
            "expected_effect": f"observed d{last.depth + 1}-{i}",
        }
        for i in range(2)
    ]


def _constant_scorer(node):
    return 1.0


def test_registry_has_beam_plan_and_react():
    assert "beam_plan" in AGENT_LOOP_REGISTRY
    assert AGENT_LOOP_REGISTRY["beam_plan"] is BeamPlanner
    # Sibling entry must remain intact.
    assert "react" in AGENT_LOOP_REGISTRY


def test_end_to_end_tiny_plan_from_registry():
    PlannerCls = AGENT_LOOP_REGISTRY["beam_plan"]
    planner = PlannerCls(
        generate_fn=_echo_generate,
        scorer_fn=_constant_scorer,
        beam_width=2,
        max_depth=2,
    )
    root = planner.plan("ship-it")
    assert isinstance(root, PlanNode)
    path = planner.best_path(root)
    queue = planner.to_execution_queue(path)
    # max_depth=2 -> two executable steps.
    assert len(queue) == 2
    assert queue[0]["depth"] == 1
    assert queue[1]["depth"] == 2
    assert all("ship-it" in step["description"] for step in queue)
    # Every queued step carries the ReActLoop-compatible fields.
    for step in queue:
        assert set(step.keys()) >= {
            "description",
            "expected_tool",
            "expected_effect",
            "plan_id",
            "depth",
            "score",
        }


def test_other_registry_entries_untouched():
    # The planner registration must not shadow or remove any sibling.
    for key in ("react", "safe_dispatch", "beam_plan"):
        assert key in AGENT_LOOP_REGISTRY
