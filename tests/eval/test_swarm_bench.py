"""Unit tests for src/eval/swarm_bench.py — 12+ tests covering SwarmBench."""
from __future__ import annotations

import pytest

from src.eval.swarm_bench import SwarmBench, SwarmBenchResult, SwarmTask
from src.agent.agent_swarm import AgentSwarm, CriticalPathAnalyzer, SubAgentResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tasks(n: int, n_subtasks: int = 3) -> list[SwarmTask]:
    return [
        SwarmTask(task_id=i, description=f"task {i}", n_subtasks=n_subtasks)
        for i in range(n)
    ]


def mock_subagent_fn(task: SwarmTask, max_steps: int) -> SubAgentResult:
    return SubAgentResult(
        task_id=task.task_id, result="done", steps_used=5, status="completed"
    )


def failing_subagent_fn(task: SwarmTask, max_steps: int) -> SubAgentResult:
    return SubAgentResult(
        task_id=task.task_id, result=None, steps_used=3, status="error"
    )


def half_failing_subagent_fn(task: SwarmTask, max_steps: int) -> SubAgentResult:
    """Completes even-indexed tasks, errors on odd-indexed."""
    if task.task_id % 2 == 0:
        return SubAgentResult(
            task_id=task.task_id, result="done", steps_used=5, status="completed"
        )
    return SubAgentResult(
        task_id=task.task_id, result=None, steps_used=3, status="error"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_completion_rate_all_pass():
    """All tasks completed → completion_rate == 1.0."""
    bench = SwarmBench()
    tasks = make_tasks(5)
    result = bench.run(tasks, mock_subagent_fn)
    assert result.completion_rate == 1.0


def test_completion_rate_all_fail():
    """All tasks error → completion_rate == 0.0."""
    bench = SwarmBench()
    tasks = make_tasks(5)
    result = bench.run(tasks, failing_subagent_fn)
    assert result.completion_rate == 0.0


def test_completion_rate_half():
    """Half completed → completion_rate == 0.5."""
    bench = SwarmBench()
    tasks = make_tasks(4)  # task_ids 0,1,2,3 → 0,2 complete; 1,3 fail
    result = bench.run(tasks, half_failing_subagent_fn)
    assert result.completion_rate == pytest.approx(0.5)


def test_n_tasks_n_completed_match():
    """n_tasks == len(input); n_completed matches actual successes."""
    bench = SwarmBench()
    tasks = make_tasks(6)
    result = bench.run(tasks, half_failing_subagent_fn)
    assert result.n_tasks == 6
    # task_ids 0,2,4 are even → 3 completed
    assert result.n_completed == 3


def test_speedup_greater_than_one():
    """Multiple parallel tasks should yield speedup >= 1.0."""
    bench = SwarmBench()
    tasks = make_tasks(4)
    result = bench.run(tasks, mock_subagent_fn)
    assert result.speedup >= 1.0


def test_speedup_serial():
    """Single task: serial == critical, so speedup == 1.0."""
    bench = SwarmBench()
    tasks = make_tasks(1)
    # With one task, serial = main + sub = critical → speedup == 1.0
    result = bench.run(tasks, mock_subagent_fn)
    assert result.speedup == pytest.approx(1.0)


def test_critical_steps_positive():
    """critical_steps > 0 for non-empty task list."""
    bench = SwarmBench()
    tasks = make_tasks(3)
    result = bench.run(tasks, mock_subagent_fn)
    assert result.critical_steps > 0


def test_empty_tasks():
    """run([]) returns zero-state SwarmBenchResult without crashing."""
    bench = SwarmBench()
    result = bench.run([], mock_subagent_fn)
    assert result.completion_rate == 0.0
    assert result.critical_steps == 0
    assert result.speedup == pytest.approx(1.0)
    assert result.n_tasks == 0
    assert result.n_completed == 0


def test_result_type():
    """run() must return a SwarmBenchResult instance."""
    bench = SwarmBench()
    tasks = make_tasks(2)
    result = bench.run(tasks, mock_subagent_fn)
    assert isinstance(result, SwarmBenchResult)


def test_determinism():
    """Same tasks + same fn → identical results on two consecutive runs."""
    bench = SwarmBench()
    tasks = make_tasks(4)
    r1 = bench.run(tasks, mock_subagent_fn)
    r2 = bench.run(tasks, mock_subagent_fn)
    assert r1.completion_rate == r2.completion_rate
    assert r1.critical_steps == r2.critical_steps
    assert r1.speedup == pytest.approx(r2.speedup)
    assert r1.n_completed == r2.n_completed


def test_n_tasks_field():
    """result.n_tasks always equals len(input_tasks)."""
    bench = SwarmBench()
    for n in [1, 3, 7, 10]:
        tasks = make_tasks(n)
        result = bench.run(tasks, mock_subagent_fn)
        assert result.n_tasks == n


def test_parallelism_ratio_nonnegative():
    """parallelism_ratio >= 0 for any valid task list."""
    bench = SwarmBench()
    tasks = make_tasks(5, n_subtasks=4)
    result = bench.run(tasks, mock_subagent_fn)
    assert result.parallelism_ratio >= 0.0


def test_custom_main_steps_accepted():
    """Providing main_steps_per_task overrides the default orchestrator budget."""
    bench = SwarmBench()
    tasks = make_tasks(3)
    result = bench.run(tasks, mock_subagent_fn, main_steps_per_task=[2, 2, 2])
    assert result.critical_steps > 0


def test_failing_tasks_still_count_in_n_tasks():
    """Even when all fail, n_tasks is populated correctly."""
    bench = SwarmBench()
    tasks = make_tasks(8)
    result = bench.run(tasks, failing_subagent_fn)
    assert result.n_tasks == 8
    assert result.n_completed == 0


def test_swarm_bench_accepts_custom_swarm():
    """SwarmBench accepts an externally-constructed AgentSwarm."""
    custom_swarm = AgentSwarm(orchestrator_max_steps=5, subagent_max_steps=20)
    bench = SwarmBench(swarm=custom_swarm)
    tasks = make_tasks(2)
    result = bench.run(tasks, mock_subagent_fn)
    assert isinstance(result, SwarmBenchResult)


def test_swarm_bench_accepts_custom_analyzer():
    """SwarmBench accepts an externally-constructed CriticalPathAnalyzer."""
    bench = SwarmBench(analyzer=CriticalPathAnalyzer())
    tasks = make_tasks(3)
    result = bench.run(tasks, mock_subagent_fn)
    assert result.speedup >= 1.0
