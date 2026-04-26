"""Integration tests for SwarmBench wired into BENCHMARK_REGISTRY."""

from __future__ import annotations

from src.agent.agent_swarm import CriticalPathAnalyzer, SubAgentResult
from src.eval import BENCHMARK_REGISTRY
from src.eval.swarm_bench import SwarmBenchResult, SwarmTask

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tasks(n: int) -> list[SwarmTask]:
    return [SwarmTask(task_id=i, description=f"integration task {i}") for i in range(n)]


def _mock_subagent_fn(task: SwarmTask, max_steps: int) -> SubAgentResult:
    return SubAgentResult(task_id=task.task_id, result="done", steps_used=5, status="completed")


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_swarm_bench_in_benchmark_registry():
    """'swarm_bench' key must be present in BENCHMARK_REGISTRY."""
    assert "swarm_bench" in BENCHMARK_REGISTRY


def test_construct_from_registry_and_run():
    """Construct SwarmBench via registry, run 5 tasks, completion_rate in [0,1]."""
    bench_cls = BENCHMARK_REGISTRY["swarm_bench"]
    bench = bench_cls()
    tasks = _make_tasks(5)
    result = bench.run(tasks, _mock_subagent_fn)
    assert isinstance(result, SwarmBenchResult)
    assert 0.0 <= result.completion_rate <= 1.0


def test_critical_path_analyzer_importable():
    """CriticalPathAnalyzer from agent_swarm (cycle-124) imports successfully."""
    analyzer = CriticalPathAnalyzer()
    # Smoke-test the compute method directly
    critical = analyzer.compute([10, 5], [[3, 7], [2]])
    assert critical == (10 + 7) + (5 + 2)


def test_existing_registry_keys_present():
    """Regression guard: pre-existing BENCHMARK_REGISTRY entries remain intact."""
    for key in ("niah", "ruler", "humaneval", "mbpp"):
        assert key in BENCHMARK_REGISTRY, f"Regression: '{key}' missing from BENCHMARK_REGISTRY"
