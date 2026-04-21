"""Agent Swarm Benchmark — evaluates AgentSwarm on BrowseComp-style tasks.
Measures completion_rate, critical_steps, speedup, parallelism_ratio.
Based on Kimi K2.5 §3.2 results: single-agent 60.6% → swarm 78.4%.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from src.agent.agent_swarm import AgentSwarm, CriticalPathAnalyzer, SubAgentResult


@dataclass
class SwarmTask:
    task_id: int
    description: str
    n_subtasks: int = 3        # expected parallel subagents
    expected_steps: int = 10   # per-subagent step budget


@dataclass
class SwarmBenchResult:
    completion_rate: float
    critical_steps: int
    speedup: float
    parallelism_ratio: float
    n_tasks: int
    n_completed: int


class SwarmBench:
    def __init__(
        self,
        swarm: AgentSwarm | None = None,
        analyzer: CriticalPathAnalyzer | None = None,
    ):
        self.swarm = swarm or AgentSwarm()
        self.analyzer = analyzer or CriticalPathAnalyzer()

    def run(
        self,
        tasks: list[SwarmTask],
        subagent_fn: Callable[[Any, int], SubAgentResult],
        main_steps_per_task: list[int] | None = None,
    ) -> SwarmBenchResult:
        if not tasks:
            return SwarmBenchResult(
                completion_rate=0.0,
                critical_steps=0,
                speedup=1.0,
                parallelism_ratio=0.0,
                n_tasks=0,
                n_completed=0,
            )

        results = self.swarm.dispatch(tasks, subagent_fn)
        n_completed = sum(1 for r in results if r.status == "completed")
        completion_rate = n_completed / len(tasks)

        # Compute critical path
        main_steps = main_steps_per_task or [self.swarm.orchestrator_max_steps] * len(tasks)
        sub_steps = [[r.steps_used] for r in results]
        critical = self.analyzer.compute(main_steps, sub_steps)
        serial = sum(m + s[0] for m, s in zip(main_steps, sub_steps))
        speedup = self.analyzer.speedup(serial, critical)

        par_ratio = sum(t.n_subtasks for t in tasks) / max(len(tasks), 1)
        par_ratio /= max(self.swarm.subagent_max_steps, 1)

        return SwarmBenchResult(
            completion_rate=completion_rate,
            critical_steps=critical,
            speedup=speedup,
            parallelism_ratio=par_ratio,
            n_tasks=len(tasks),
            n_completed=n_completed,
        )
