"""Unit tests for the Tree-of-Thoughts :class:`BeamPlanner`."""

from __future__ import annotations

import math
import random

import pytest

from src.agent.agent_planner import BeamPlanner, PlanNode

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def make_fixed_generate(k: int = 3):
    """Return a generator emitting ``k`` valid proposals each call."""

    def _gen(task, path):
        last = path[-1]
        return [
            {
                "description": f"d{last.depth + 1}-{i}",
                "expected_tool": "echo" if i % 2 == 0 else None,
                "expected_effect": f"eff-{last.depth + 1}-{i}",
            }
            for i in range(k)
        ]

    return _gen


def constant_scorer(value: float = 1.0):
    def _s(node):
        return value

    return _s


def index_scorer(node):
    """Score based on the trailing digit of the description (deterministic)."""
    return float(node.description.rsplit("-", 1)[-1])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_expand_produces_beam_width_children():
    planner = BeamPlanner(
        generate_fn=make_fixed_generate(k=10),
        scorer_fn=constant_scorer(1.0),
        beam_width=4,
        max_depth=3,
    )
    root = PlanNode(
        id="r",
        parent_id=None,
        depth=0,
        description="root",
        expected_tool=None,
        expected_effect="",
    )
    kids = planner.expand(root, task="t")
    assert len(kids) == 4
    assert all(c.parent_id == "r" for c in kids)
    assert all(c.depth == 1 for c in kids)


def test_plan_terminates_at_max_depth():
    planner = BeamPlanner(
        generate_fn=make_fixed_generate(k=2),
        scorer_fn=constant_scorer(1.0),
        beam_width=2,
        max_depth=3,
    )
    root = planner.plan("task")
    # Walk to deepest node.
    deepest = 0
    stack = [root]
    while stack:
        n = stack.pop()
        deepest = max(deepest, n.depth)
        stack.extend(n.children)
    assert deepest == 3


def test_best_path_returns_highest_scoring_chain():
    planner = BeamPlanner(
        generate_fn=make_fixed_generate(k=3),
        scorer_fn=index_scorer,
        beam_width=3,
        max_depth=2,
    )
    root = planner.plan("task")
    path = planner.best_path(root)
    assert path[0] is root
    # With index_scorer, index=2 wins at each depth.
    assert all(p.description.endswith("-2") for p in path[1:])


def test_to_execution_queue_yields_ordered_dicts():
    planner = BeamPlanner(
        generate_fn=make_fixed_generate(k=2),
        scorer_fn=constant_scorer(1.0),
        beam_width=2,
        max_depth=2,
    )
    root = planner.plan("task")
    path = planner.best_path(root)
    queue = planner.to_execution_queue(path)
    assert isinstance(queue, list)
    assert all(isinstance(q, dict) for q in queue)
    # Root is skipped.
    assert len(queue) == len(path) - 1
    assert [q["depth"] for q in queue] == list(range(1, len(path)))
    assert {"description", "expected_tool", "expected_effect", "plan_id"} <= queue[0].keys()


def test_malformed_generate_output_is_dropped():
    def bad_gen(task, path):
        return [
            {"description": "ok", "expected_tool": None, "expected_effect": "e"},
            "not a dict",
            {"description": "missing-fields"},
            {"description": 42, "expected_tool": None, "expected_effect": "e"},
        ]

    planner = BeamPlanner(
        generate_fn=bad_gen,
        scorer_fn=constant_scorer(1.0),
        beam_width=4,
        max_depth=1,
    )
    root = PlanNode(
        id="r",
        parent_id=None,
        depth=0,
        description="root",
        expected_tool=None,
        expected_effect="",
    )
    kids = planner.expand(root, "task")
    assert len(kids) == 1
    assert kids[0].description == "ok"


def test_scorer_returning_nan_is_clamped():
    def nan_scorer(node):
        return float("nan")

    planner = BeamPlanner(
        generate_fn=make_fixed_generate(k=2),
        scorer_fn=nan_scorer,
        beam_width=2,
        max_depth=1,
    )
    root = planner.plan("task")
    for child in root.children:
        assert child.score == -math.inf


def test_empty_children_stops_planner_early():
    calls = {"n": 0}

    def gen(task, path):
        calls["n"] += 1
        # Expand once at depth 0, then refuse.
        if path[-1].depth == 0:
            return [{"description": "d1", "expected_tool": None, "expected_effect": ""}]
        return []

    planner = BeamPlanner(
        generate_fn=gen,
        scorer_fn=constant_scorer(1.0),
        beam_width=2,
        max_depth=5,
    )
    root = planner.plan("task")
    assert len(root.children) == 1
    assert root.children[0].children == []
    # Depth-1 expansion tried but returned nothing; the planner must stop.
    assert calls["n"] == 2


def test_beam_width_one_is_greedy():
    planner = BeamPlanner(
        generate_fn=make_fixed_generate(k=4),
        scorer_fn=index_scorer,
        beam_width=1,
        max_depth=3,
    )
    root = planner.plan("t")
    # At each depth only one node should survive.
    depth_counts: dict[int, int] = {}
    stack = [root]
    while stack:
        n = stack.pop()
        depth_counts[n.depth] = depth_counts.get(n.depth, 0) + 1
        stack.extend(n.children)
    assert depth_counts[0] == 1
    for d in range(1, 4):
        assert depth_counts[d] == 1


def test_determinism_with_pseudo_random_generate_fn():
    def make_gen(seed):
        rng = random.Random(seed)

        def _g(task, path):
            return [
                {
                    "description": f"d-{rng.randint(0, 9999)}",
                    "expected_tool": None,
                    "expected_effect": "e",
                }
                for _ in range(3)
            ]

        return _g

    def run(seed):
        planner = BeamPlanner(
            generate_fn=make_gen(seed),
            scorer_fn=index_scorer,
            beam_width=2,
            max_depth=2,
        )
        root = planner.plan("task")
        return [n.description for n in planner.best_path(root)]

    # Same seed -> identical plan descriptions (ignoring uuid ids).
    assert run(42) == run(42)


def test_plan_node_ids_are_unique():
    planner = BeamPlanner(
        generate_fn=make_fixed_generate(k=3),
        scorer_fn=constant_scorer(1.0),
        beam_width=3,
        max_depth=3,
    )
    root = planner.plan("task")
    ids = []
    stack = [root]
    while stack:
        n = stack.pop()
        ids.append(n.id)
        stack.extend(n.children)
    assert len(ids) == len(set(ids))


def test_parent_ids_link_to_real_nodes():
    planner = BeamPlanner(
        generate_fn=make_fixed_generate(k=2),
        scorer_fn=constant_scorer(1.0),
        beam_width=2,
        max_depth=3,
    )
    root = planner.plan("task")
    nodes: dict = {}
    stack = [root]
    while stack:
        n = stack.pop()
        nodes[n.id] = n
        stack.extend(n.children)
    for nid, node in nodes.items():
        if node.parent_id is None:
            assert node is root
        else:
            assert node.parent_id in nodes
            assert nodes[node.parent_id].depth == node.depth - 1


def test_zero_depth_returns_root_with_no_expansions():
    expansions = {"n": 0}

    def gen(task, path):
        expansions["n"] += 1
        return []

    planner = BeamPlanner(
        generate_fn=gen,
        scorer_fn=constant_scorer(1.0),
        beam_width=4,
        max_depth=0,
    )
    root = planner.plan("task")
    assert root.parent_id is None
    assert root.depth == 0
    assert root.children == []
    assert expansions["n"] == 0
    # best_path on a lonely root yields [root], and execution queue is empty.
    assert planner.best_path(root) == [root]
    assert planner.to_execution_queue([root]) == []


def test_invalid_constructor_args_raise():
    with pytest.raises(TypeError):
        BeamPlanner(generate_fn="x", scorer_fn=constant_scorer(1.0))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        BeamPlanner(generate_fn=make_fixed_generate(), scorer_fn="x")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        BeamPlanner(
            generate_fn=make_fixed_generate(),
            scorer_fn=constant_scorer(1.0),
            beam_width=0,
        )
    with pytest.raises(ValueError):
        BeamPlanner(
            generate_fn=make_fixed_generate(),
            scorer_fn=constant_scorer(1.0),
            max_depth=-1,
        )


def test_scorer_raising_is_clamped_not_fatal():
    def boom(node):
        raise RuntimeError("nope")

    planner = BeamPlanner(
        generate_fn=make_fixed_generate(k=2),
        scorer_fn=boom,
        beam_width=2,
        max_depth=1,
    )
    root = planner.plan("task")
    assert len(root.children) == 2
    assert all(c.score == -math.inf for c in root.children)
