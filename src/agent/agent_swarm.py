"""Agent Swarm — orchestrator + frozen subagents (Kimi K2.5 §3.2, arXiv:2602.02276).

Orchestrator is trainable via PARL reward; subagents are frozen from
intermediate checkpoints. Reported 3–4.5× latency reduction vs. single-agent
on BrowseComp (60.6% → 78.4% accuracy).

Design principles
-----------------
* Orchestrator dynamically spawns and schedules subagents.
* Subagent weights are **frozen** during swarm RL — only the orchestrator
  is updated. This decouples credit assignment from the subagent policy and
  eliminates ambiguity in multi-agent reward attribution.
* Critical-path analysis quantifies the theoretical speedup:

      critical_steps = Σ_t [ S_main(t) + max_i(S_sub,i(t)) ]

  where t is the stage index, S_main is the number of orchestrator steps in
  that stage, and S_sub,i is the number of steps taken by the i-th subagent
  running in parallel during that stage.

      speedup = serial_steps / critical_steps

No heavyweight imports — pure standard library only at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class SubAgentResult:
    """Result produced by a single frozen subagent.

    Attributes
    ----------
    task_id:
        Integer identifier that ties this result back to its input task.
    result:
        Arbitrary payload returned by the subagent function.
    steps_used:
        Number of generation / tool-call steps consumed.
    status:
        One of ``"completed"``, ``"truncated"``, or ``"error"``.
    """

    task_id: int
    result: Any
    steps_used: int
    status: str  # "completed" | "truncated" | "error"


# ---------------------------------------------------------------------------
# Critical-path analysis
# ---------------------------------------------------------------------------


@dataclass
class CriticalPathAnalyzer:
    """Compute theoretical parallel speedup as described in Kimi K2.5 §3.2.

    The formula treats each *stage* as one orchestrator phase followed by a
    set of concurrently-executing subagents.  Within a stage the latency is:

        L(t) = S_main(t) + max_i(S_sub,i(t))

    so the total critical-path length is the sum across all stages.

    The class is a plain ``@dataclass`` (no ``__init__`` arguments) so it can
    be instantiated as ``CriticalPathAnalyzer()`` with zero configuration.
    """

    def compute(
        self,
        main_steps_per_stage: List[int],
        sub_steps_per_stage: List[List[int]],
    ) -> int:
        """Return the total critical-path step count.

        Parameters
        ----------
        main_steps_per_stage:
            Orchestrator steps for each stage, e.g. ``[2, 3, 1]``.
        sub_steps_per_stage:
            For each stage, a list of step counts for the concurrently-running
            subagents, e.g. ``[[10, 8, 12], [5], []]``.  An empty list for a
            stage means no subagents were spawned in that stage.

        Returns
        -------
        int
            Total critical-path length (sum of per-stage latencies).
        """
        total = 0
        for t, main_s in enumerate(main_steps_per_stage):
            parallel: List[int] = (
                sub_steps_per_stage[t] if t < len(sub_steps_per_stage) else []
            )
            total += main_s + (max(parallel) if parallel else 0)
        return total

    def speedup(self, serial_steps: int, critical_steps: int) -> float:
        """Return the parallel speedup ratio.

        Parameters
        ----------
        serial_steps:
            Total steps if the same tasks were executed sequentially by a
            single agent (i.e. sum of all subagent steps plus all orchestrator
            steps).
        critical_steps:
            Value returned by :meth:`compute`.

        Returns
        -------
        float
            ``serial_steps / critical_steps``.  Clamped to avoid division by
            zero: if ``critical_steps <= 0`` it is treated as 1.
        """
        return serial_steps / max(critical_steps, 1)


# ---------------------------------------------------------------------------
# Orchestrator / swarm controller
# ---------------------------------------------------------------------------


@dataclass
class AgentSwarm:
    """Orchestrator that dispatches tasks to frozen subagents.

    The orchestrator itself is trainable via a PARL reward signal; the
    subagent functions passed to :meth:`dispatch` are assumed to be frozen
    (their parameters are not updated during swarm RL training).

    Attributes
    ----------
    orchestrator_max_steps:
        Step budget for the orchestrator's own reasoning loop.
    subagent_max_steps:
        Step budget forwarded to every spawned subagent.
    """

    orchestrator_max_steps: int = 15
    subagent_max_steps: int = 100

    def dispatch(
        self,
        tasks: List[Any],
        subagent_fn: Callable[[Any, int], SubAgentResult],
    ) -> List[SubAgentResult]:
        """Dispatch *tasks* to the frozen *subagent_fn* and collect results.

        Parameters
        ----------
        tasks:
            Sequence of task descriptors.  Each element is passed verbatim to
            *subagent_fn*.
        subagent_fn:
            Callable with signature ``(task, max_steps) -> SubAgentResult``.
            Must be frozen (not modified) during swarm training.  Any
            exception raised by the callable propagates to the caller; there
            is no silent error swallowing.

        Returns
        -------
        List[SubAgentResult]
            One result per task, in the same order as *tasks*.

        Raises
        ------
        Exception
            Any exception raised by *subagent_fn* is re-raised immediately,
            preserving the original traceback.
        """
        results: List[SubAgentResult] = []
        for task in tasks:
            # Exceptions intentionally propagate — no silent swallowing.
            result = subagent_fn(task, self.subagent_max_steps)
            results.append(result)
        return results
