"""Unit tests for taubench_scorer.py.

Covers:
 1.  TauBenchTask dataclass creation with all fields
 2.  TauBenchTrajectory dataclass creation
 3.  score_trajectory: successful trajectory → success=True, correct metrics
 4.  score_trajectory: failed trajectory → success=False
 5.  score_trajectory: partial completion → partial_credit > 0
 6.  compute_tool_accuracy: exact match → 1.0
 7.  compute_tool_accuracy: no overlap → 0.0
 8.  compute_tool_accuracy: partial overlap → between 0 and 1
 9.  score_batch: aggregates correctly over multiple tasks
10.  check_success: respects different success criteria types
11.  TauBenchDataset.get_sample_tasks: returns valid tasks
12.  Adversarial: empty trajectory → no crash
13.  Adversarial: trajectory with no tool calls → scored correctly
14.  efficiency_score: agent that completes in fewer turns scores higher
15.  compute_tool_accuracy: both empty → 1.0
16.  partial_credit: full completion → 1.0
17.  check_success: exact_tool_sequence criterion
18.  score_batch: empty inputs → zeros
"""

from __future__ import annotations

import pytest

from src.eval.taubench_scorer import (
    TauBenchDataset,
    TauBenchScorer,
    TauBenchTask,
    TauBenchTrajectory,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_task(
    task_id: str = "t-001",
    expected_actions: list[str] | None = None,
    success_criteria: dict | None = None,
    domain: str = "coding",
    max_turns: int = 30,
) -> TauBenchTask:
    return TauBenchTask(
        task_id=task_id,
        instruction="Do the thing.",
        tools=[{"name": "search_code"}, {"name": "edit_file"}],
        expected_actions=expected_actions
        if expected_actions is not None
        else ["search_code", "edit_file"],
        success_criteria=success_criteria
        if success_criteria is not None
        else {
            "required_tool_calls": ["search_code", "edit_file"],
            "required_state": {"done": True},
        },
        domain=domain,
        max_turns=max_turns,
    )


def _make_trajectory(
    task_id: str = "t-001",
    tool_calls: list[str] | None = None,
    final_state: dict | None = None,
    success: bool = True,
    num_turns: int | None = None,
) -> TauBenchTrajectory:
    calls = tool_calls if tool_calls is not None else ["search_code", "edit_file"]
    turns = [
        {
            "role": "assistant",
            "content": f"Calling {c}",
            "tool_calls": [c],
            "tool_results": [{"status": "ok"}],
        }
        for c in calls
    ]
    return TauBenchTrajectory(
        task_id=task_id,
        turns=turns,
        final_state=final_state if final_state is not None else {"done": True},
        success=success,
        num_turns=num_turns if num_turns is not None else len(turns),
    )


SCORER = TauBenchScorer()


# ---------------------------------------------------------------------------
# 1. TauBenchTask dataclass creation with all fields
# ---------------------------------------------------------------------------


def test_task_dataclass_creation():
    task = TauBenchTask(
        task_id="task-123",
        instruction="Find and fix the bug.",
        tools=[{"name": "search_code"}, {"name": "edit_file"}],
        expected_actions=["search_code", "edit_file", "run_tests"],
        success_criteria={"required_tool_calls": ["edit_file"]},
        domain="coding",
        max_turns=20,
    )
    assert task.task_id == "task-123"
    assert task.instruction == "Find and fix the bug."
    assert len(task.tools) == 2
    assert task.expected_actions == ["search_code", "edit_file", "run_tests"]
    assert task.domain == "coding"
    assert task.max_turns == 20


# ---------------------------------------------------------------------------
# 2. TauBenchTrajectory dataclass creation
# ---------------------------------------------------------------------------


def test_trajectory_dataclass_creation():
    traj = TauBenchTrajectory(
        task_id="task-123",
        turns=[
            {
                "role": "assistant",
                "content": "Calling search_code",
                "tool_calls": ["search_code"],
                "tool_results": [{"status": "ok"}],
            }
        ],
        final_state={"result": "found"},
        success=True,
        num_turns=1,
    )
    assert traj.task_id == "task-123"
    assert len(traj.turns) == 1
    assert traj.final_state == {"result": "found"}
    assert traj.success is True
    assert traj.num_turns == 1


# ---------------------------------------------------------------------------
# 3. score_trajectory: successful trajectory → success=True, correct metrics
# ---------------------------------------------------------------------------


def test_score_trajectory_success():
    task = _make_task()
    traj = _make_trajectory(final_state={"done": True}, success=True)
    result = SCORER.score_trajectory(task, traj)

    assert result["success"] is True
    assert result["turns"] == 2
    assert result["tool_accuracy"] == pytest.approx(1.0)
    assert result["partial_credit"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 4. score_trajectory: failed trajectory → success=False
# ---------------------------------------------------------------------------


def test_score_trajectory_failure():
    task = _make_task()
    traj = _make_trajectory(
        tool_calls=["other_tool"],
        final_state={"done": False},
        success=False,
    )
    result = SCORER.score_trajectory(task, traj)
    assert result["success"] is False


# ---------------------------------------------------------------------------
# 5. score_trajectory: partial completion → partial_credit > 0
# ---------------------------------------------------------------------------


def test_score_trajectory_partial():
    task = _make_task(
        expected_actions=["search_code", "edit_file", "run_tests"],
        success_criteria={"required_tool_calls": ["search_code", "edit_file", "run_tests"]},
    )
    # Only completed the first expected action
    traj = _make_trajectory(
        tool_calls=["search_code"],
        final_state={},
        success=False,
    )
    result = SCORER.score_trajectory(task, traj)
    assert result["success"] is False
    assert 0.0 < result["partial_credit"] < 1.0


# ---------------------------------------------------------------------------
# 6. compute_tool_accuracy: exact match → 1.0
# ---------------------------------------------------------------------------


def test_tool_accuracy_exact_match():
    acc = SCORER.compute_tool_accuracy(
        ["search_code", "edit_file"],
        ["search_code", "edit_file"],
    )
    assert acc == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 7. compute_tool_accuracy: no overlap → 0.0
# ---------------------------------------------------------------------------


def test_tool_accuracy_no_overlap():
    acc = SCORER.compute_tool_accuracy(
        ["search_code", "edit_file"],
        ["run_tests", "deploy"],
    )
    assert acc == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 8. compute_tool_accuracy: partial overlap → between 0 and 1
# ---------------------------------------------------------------------------


def test_tool_accuracy_partial_overlap():
    acc = SCORER.compute_tool_accuracy(
        ["search_code", "edit_file", "run_tests"],
        ["search_code", "deploy"],
    )
    assert 0.0 < acc < 1.0


# ---------------------------------------------------------------------------
# 9. score_batch: aggregates correctly over multiple tasks
# ---------------------------------------------------------------------------


def test_score_batch_aggregation():
    tasks = [_make_task(task_id=f"t-{i:03d}") for i in range(4)]
    # First 3 succeed, last fails
    trajectories = [
        _make_trajectory(task_id=f"t-{i:03d}", final_state={"done": True}, success=True)
        for i in range(3)
    ] + [
        _make_trajectory(
            task_id="t-003",
            tool_calls=["other"],
            final_state={"done": False},
            success=False,
            num_turns=5,
        )
    ]
    agg = SCORER.score_batch(tasks, trajectories)
    assert agg["n_tasks"] == 4
    assert agg["success_rate"] == pytest.approx(0.75)
    assert agg["avg_turns"] > 0
    assert "tool_accuracy" in agg
    assert "efficiency_score" in agg
    assert "domain_breakdown" in agg
    assert len(agg["per_task"]) == 4


# ---------------------------------------------------------------------------
# 10. check_success: respects different success criteria types
# ---------------------------------------------------------------------------


def test_check_success_required_tool_calls():
    task = _make_task(success_criteria={"required_tool_calls": ["search_code"]})
    traj = _make_trajectory(tool_calls=["search_code", "edit_file"])
    assert SCORER.check_success(task, traj) is True

    traj_fail = _make_trajectory(tool_calls=["edit_file"])
    assert SCORER.check_success(task, traj_fail) is False


def test_check_success_required_state():
    task = _make_task(success_criteria={"required_state": {"done": True, "count": 3}})
    traj_pass = _make_trajectory(final_state={"done": True, "count": 3, "extra": "x"})
    traj_fail = _make_trajectory(final_state={"done": True, "count": 2})
    assert SCORER.check_success(task, traj_pass) is True
    assert SCORER.check_success(task, traj_fail) is False


def test_check_success_exact_sequence():
    task = _make_task(
        expected_actions=["search_code", "edit_file"],
        success_criteria={"exact_tool_sequence": ["search_code", "edit_file"]},
    )
    traj_pass = _make_trajectory(tool_calls=["search_code", "edit_file"])
    traj_fail = _make_trajectory(tool_calls=["edit_file", "search_code"])  # wrong order
    assert SCORER.check_success(task, traj_pass) is True
    assert SCORER.check_success(task, traj_fail) is False


def test_check_success_empty_criteria_falls_back_to_flag():
    task = _make_task(success_criteria={})
    traj_true = _make_trajectory(success=True)
    traj_false = _make_trajectory(success=False)
    assert SCORER.check_success(task, traj_true) is True
    assert SCORER.check_success(task, traj_false) is False


# ---------------------------------------------------------------------------
# 11. TauBenchDataset.get_sample_tasks: returns valid tasks
# ---------------------------------------------------------------------------


def test_dataset_get_sample_tasks_coding():
    tasks = TauBenchDataset.get_sample_tasks(domain="coding", n=5)
    assert len(tasks) == 5
    for task in tasks:
        assert isinstance(task, TauBenchTask)
        assert task.domain == "coding"
        assert task.task_id.startswith("coding-")
        assert len(task.expected_actions) >= 1
        assert isinstance(task.tools, list)
        assert isinstance(task.success_criteria, dict)


def test_dataset_get_sample_tasks_retail():
    tasks = TauBenchDataset.get_sample_tasks(domain="retail", n=3)
    assert len(tasks) == 3
    for task in tasks:
        assert task.domain == "retail"


def test_dataset_get_sample_tasks_unknown_domain():
    # Should not crash; falls back to coding
    tasks = TauBenchDataset.get_sample_tasks(domain="unknown_xyz", n=2)
    assert len(tasks) == 2


# ---------------------------------------------------------------------------
# 12. Adversarial: empty trajectory → no crash
# ---------------------------------------------------------------------------


def test_empty_trajectory_no_crash():
    task = _make_task()
    traj = TauBenchTrajectory(
        task_id="t-001",
        turns=[],
        final_state={},
        success=False,
        num_turns=0,
    )
    result = SCORER.score_trajectory(task, traj)
    assert isinstance(result, dict)
    assert result["success"] is False
    assert result["turns"] == 0
    assert 0.0 <= result["tool_accuracy"] <= 1.0
    assert 0.0 <= result["partial_credit"] <= 1.0


# ---------------------------------------------------------------------------
# 13. Adversarial: trajectory with no tool calls → scored correctly
# ---------------------------------------------------------------------------


def test_trajectory_no_tool_calls():
    task = _make_task(
        success_criteria={"required_tool_calls": ["search_code"]},
    )
    traj = TauBenchTrajectory(
        task_id="t-001",
        turns=[
            {
                "role": "assistant",
                "content": "I will think about this.",
                "tool_calls": [],
                "tool_results": [],
            },
        ],
        final_state={},
        success=False,
        num_turns=1,
    )
    result = SCORER.score_trajectory(task, traj)
    assert result["success"] is False
    assert result["tool_accuracy"] == pytest.approx(0.0)
    assert result["partial_credit"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 14. efficiency_score: agent that completes in fewer turns scores higher
# ---------------------------------------------------------------------------


def test_efficiency_score_fewer_turns_is_better():
    task = _make_task()

    # Efficient: 2 turns
    fast_traj = _make_trajectory(final_state={"done": True}, success=True, num_turns=2)
    # Slow: 20 turns
    slow_traj = _make_trajectory(final_state={"done": True}, success=True, num_turns=20)

    fast_batch = SCORER.score_batch([task], [fast_traj])
    slow_batch = SCORER.score_batch([task], [slow_traj])

    assert fast_batch["efficiency_score"] > slow_batch["efficiency_score"]


# ---------------------------------------------------------------------------
# 15. compute_tool_accuracy: both empty → 1.0
# ---------------------------------------------------------------------------


def test_tool_accuracy_both_empty():
    acc = SCORER.compute_tool_accuracy([], [])
    assert acc == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 16. partial_credit: full completion → 1.0
# ---------------------------------------------------------------------------


def test_partial_credit_full():
    task = _make_task(expected_actions=["search_code", "edit_file"])
    traj = _make_trajectory(tool_calls=["search_code", "edit_file"])
    pc = SCORER.partial_credit(task, traj)
    assert pc == pytest.approx(1.0)


def test_partial_credit_zero():
    task = _make_task(expected_actions=["search_code", "edit_file"])
    traj = _make_trajectory(tool_calls=["run_tests", "deploy"])
    pc = SCORER.partial_credit(task, traj)
    assert pc == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 17. check_success: combined criteria (AND logic)
# ---------------------------------------------------------------------------


def test_check_success_combined_criteria():
    task = _make_task(
        success_criteria={
            "required_tool_calls": ["search_code"],
            "required_state": {"done": True},
        }
    )
    # Both satisfied
    traj_both = _make_trajectory(
        tool_calls=["search_code", "edit_file"],
        final_state={"done": True},
    )
    # Only tool call satisfied
    traj_tool_only = _make_trajectory(
        tool_calls=["search_code"],
        final_state={"done": False},
    )
    assert SCORER.check_success(task, traj_both) is True
    assert SCORER.check_success(task, traj_tool_only) is False


# ---------------------------------------------------------------------------
# 18. score_batch: empty inputs → zeros
# ---------------------------------------------------------------------------


def test_score_batch_empty():
    result = SCORER.score_batch([], [])
    assert result["success_rate"] == pytest.approx(0.0)
    assert result["avg_turns"] == pytest.approx(0.0)
    assert result["tool_accuracy"] == pytest.approx(0.0)
    assert result["efficiency_score"] == pytest.approx(0.0)
    assert result["n_tasks"] == 0
    assert result["per_task"] == []
    assert result["domain_breakdown"] == {}


# ---------------------------------------------------------------------------
# Bonus: dataset make_successful_trajectory helper
# ---------------------------------------------------------------------------


def test_dataset_make_successful_trajectory():
    task = TauBenchDataset.get_sample_tasks(domain="coding", n=1)[0]
    traj = TauBenchDataset.make_successful_trajectory(task)
    assert traj.task_id == task.task_id
    assert traj.success is True
    result = SCORER.score_trajectory(task, traj)
    assert result["success"] is True
    assert result["partial_credit"] == pytest.approx(1.0)
