"""Unit tests for src.agent.agent_swarm (Kimi K2.5 §3.2, arXiv:2602.02276).

12 tests covering CriticalPathAnalyzer, AgentSwarm.dispatch, SubAgentResult
fields, and edge / adversarial cases.
"""

from __future__ import annotations

import pytest

from src.agent.agent_swarm import AgentSwarm, CriticalPathAnalyzer, SubAgentResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_subagent_fn(steps: int = 5, status: str = "completed"):
    """Return a mock subagent_fn that always succeeds."""

    def fn(task, max_steps: int) -> SubAgentResult:
        return SubAgentResult(
            task_id=task["id"] if isinstance(task, dict) else task,
            result=f"result_for_{task}",
            steps_used=min(steps, max_steps),
            status=status,
        )

    return fn


# ---------------------------------------------------------------------------
# CriticalPathAnalyzer tests
# ---------------------------------------------------------------------------


def test_critical_path_single_stage_three_subagents():
    """Single stage: main=2, subagents=[10, 8, 12] → critical = 2 + 12 = 14."""
    cpa = CriticalPathAnalyzer()
    result = cpa.compute(main_steps_per_stage=[2], sub_steps_per_stage=[[10, 8, 12]])
    assert result == 14


def test_critical_path_two_stages_sum():
    """Two stages: (main=3, subs=[5,7]) + (main=2, subs=[4]) = (3+7) + (2+4) = 16."""
    cpa = CriticalPathAnalyzer()
    result = cpa.compute(
        main_steps_per_stage=[3, 2],
        sub_steps_per_stage=[[5, 7], [4]],
    )
    assert result == 16


def test_speedup_greater_than_one_when_parallel():
    """Parallel speedup > 1.0 when subagents run concurrently.

    Serial: 2 + 10 + 2 + 7 = 21
    Critical path: (2 + 10) + (2 + 7) = 21  — but with 3 subagents in
    first stage the serial count for that stage would be 2 + 10 + 8 + 12 = 32
    whereas critical is 2 + 12 = 14.
    """
    cpa = CriticalPathAnalyzer()
    serial = 2 + 10 + 8 + 12  # sequential execution
    critical = cpa.compute(main_steps_per_stage=[2], sub_steps_per_stage=[[10, 8, 12]])
    speedup = cpa.speedup(serial, critical)
    assert speedup > 1.0


def test_speedup_equals_one_for_single_sequential_subagent():
    """One subagent, no overlap: critical == serial → speedup == 1.0."""
    cpa = CriticalPathAnalyzer()
    serial = 3 + 7  # main steps + one subagent
    critical = cpa.compute(main_steps_per_stage=[3], sub_steps_per_stage=[[7]])
    assert critical == serial
    assert cpa.speedup(serial, critical) == pytest.approx(1.0)


def test_critical_path_empty_subagent_list_per_stage():
    """Stages with no subagents: critical_steps equals sum of main steps only."""
    cpa = CriticalPathAnalyzer()
    result = cpa.compute(
        main_steps_per_stage=[4, 6],
        sub_steps_per_stage=[[], []],
    )
    assert result == 10  # 4 + 0 + 6 + 0


def test_speedup_no_division_by_zero_when_critical_zero():
    """critical_steps=0 must not raise ZeroDivisionError; speedup = serial / 1."""
    cpa = CriticalPathAnalyzer()
    # main=0, no subagents → critical = 0
    critical = cpa.compute(main_steps_per_stage=[0], sub_steps_per_stage=[[]])
    assert critical == 0
    # speedup clamps denominator to 1
    speedup = cpa.speedup(serial_steps=10, critical_steps=critical)
    assert speedup == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# AgentSwarm.dispatch tests
# ---------------------------------------------------------------------------


def test_dispatch_returns_n_results_for_n_tasks():
    """N tasks → exactly N SubAgentResults."""
    swarm = AgentSwarm()
    tasks = [{"id": i} for i in range(5)]
    fn = _make_subagent_fn()
    results = swarm.dispatch(tasks, fn)
    assert len(results) == 5


def test_dispatch_task_id_assigned_per_result():
    """task_id on each result matches the dict 'id' passed in."""
    swarm = AgentSwarm()
    tasks = [{"id": 10}, {"id": 20}, {"id": 30}]
    fn = _make_subagent_fn()
    results = swarm.dispatch(tasks, fn)
    ids = [r.task_id for r in results]
    assert ids == [10, 20, 30]


def test_dispatch_status_completed_for_successful_mock():
    """Mock subagent returns 'completed' status."""
    swarm = AgentSwarm()
    tasks = [{"id": 0}]
    fn = _make_subagent_fn(status="completed")
    results = swarm.dispatch(tasks, fn)
    assert results[0].status == "completed"


def test_dispatch_propagates_subagent_exception():
    """If subagent_fn raises, dispatch must NOT swallow it."""

    def failing_fn(task, max_steps):
        raise RuntimeError("subagent exploded")

    swarm = AgentSwarm()
    with pytest.raises(RuntimeError, match="subagent exploded"):
        swarm.dispatch([{"id": 1}], failing_fn)


def test_orchestrator_max_steps_stored_correctly():
    """Custom orchestrator_max_steps is preserved on the instance."""
    swarm = AgentSwarm(orchestrator_max_steps=42, subagent_max_steps=200)
    assert swarm.orchestrator_max_steps == 42
    assert swarm.subagent_max_steps == 200


def test_subagent_result_fields():
    """SubAgentResult exposes task_id, result, steps_used, status."""
    r = SubAgentResult(task_id=7, result="payload", steps_used=12, status="truncated")
    assert r.task_id == 7
    assert r.result == "payload"
    assert r.steps_used == 12
    assert r.status == "truncated"


def test_critical_path_multi_stage_mixed_subagents():
    """Three stages, mixed subagent counts — verify formula step by step."""
    # Stage 0: main=1, subs=[5, 3]  → 1 + 5 = 6
    # Stage 1: main=2, subs=[]      → 2 + 0 = 2
    # Stage 2: main=4, subs=[8]     → 4 + 8 = 12
    # total = 20
    cpa = CriticalPathAnalyzer()
    result = cpa.compute(
        main_steps_per_stage=[1, 2, 4],
        sub_steps_per_stage=[[5, 3], [], [8]],
    )
    assert result == 20
