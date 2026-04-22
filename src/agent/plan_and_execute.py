"""Plan-and-Execute agent for Aurelius.

Implements the Plan-and-Solve Prompting paradigm from Wang et al. 2023
("Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning
by Large Language Models", arXiv:2305.04091).

The core idea separates planning from execution:

1. **Plan phase**: Generate a full ordered list of subtasks up front.
2. **Execute phase**: Execute each subtask sequentially with its own executor.
3. **Re-plan phase**: When a subtask fails, generate a fresh plan for the
   remaining work, informed by what was already completed.

This is in contrast to ReAct (interleaved reasoning + acting), which decides
the next action after every observation.  Plan-and-Execute is better suited to
tasks where the full problem structure is knowable in advance.

Design choices
--------------
* Pure Python standard library: ``dataclasses``, ``typing``.
  No torch, no external ML libraries.
* ``PlanAndExecuteAgent`` receives all model-dependent functionality via
  callables so the agent itself is fully model-agnostic.
* The agent is registered in ``AGENT_LOOP_REGISTRY["plan_and_execute"]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PlanConfig:
    """Hyper-parameters for a :class:`PlanAndExecuteAgent` run.

    Attributes
    ----------
    max_replans:
        Maximum number of times to re-plan on failure.  When this limit is
        reached, the agent stops even if there are still pending steps.
    max_steps:
        Maximum total execution steps across all plans (including re-plans).
        Acts as an absolute upper bound on agent work.
    replan_on_failure:
        If ``True``, trigger a re-plan whenever a step fails.  If ``False``,
        failed steps are recorded but the agent continues with the next step.
    task_timeout_steps:
        Number of steps that can be spent on a single task before it is
        considered failed (reserved for future per-step retry logic; not
        enforced in the base implementation).
    """

    max_replans: int = 3
    max_steps: int = 20
    replan_on_failure: bool = True
    task_timeout_steps: int = 5


# ---------------------------------------------------------------------------
# Plan data model
# ---------------------------------------------------------------------------


@dataclass
class PlanStep:
    """A single step inside an :class:`ExecutionPlan`.

    Attributes
    ----------
    step_idx:
        Zero-based index of this step within its parent plan.
    task:
        Plain-text description of the subtask to be executed.
    result:
        Execution result string, or ``None`` if not yet executed.
    success:
        ``True`` once the step has been executed successfully.
    n_attempts:
        Number of execution attempts made for this step.
    is_final:
        ``True`` for the last step in the plan.  Set by
        :meth:`ExecutionPlan.__post_init__`.
    """

    step_idx: int
    task: str
    result: Optional[str] = None
    success: bool = False
    n_attempts: int = 0
    is_final: bool = False


@dataclass
class ExecutionPlan:
    """An ordered sequence of :class:`PlanStep` objects.

    Attributes
    ----------
    steps:
        Ordered list of steps to execute.
    created_at_attempt:
        The re-plan iteration index during which this plan was created
        (0 = original plan, 1 = first re-plan, etc.).
    """

    steps: List[PlanStep]
    created_at_attempt: int = 0

    def __post_init__(self) -> None:
        # Mark the final step so downstream code can detect it without
        # needing to know the length.
        if self.steps:
            self.steps[-1].is_final = True

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def n_steps(self) -> int:
        """Total number of steps in the plan."""
        return len(self.steps)

    @property
    def completed_steps(self) -> List[PlanStep]:
        """Steps that have been executed (``result is not None``)."""
        return [s for s in self.steps if s.result is not None]

    @property
    def pending_steps(self) -> List[PlanStep]:
        """Steps that have not yet been executed (``result is None``)."""
        return [s for s in self.steps if s.result is None]


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class PlanExecuteResult:
    """Complete result of a :meth:`PlanAndExecuteAgent.run` invocation.

    Attributes
    ----------
    final_answer:
        The result string of the last successfully executed step, or the
        result of the last executed step if no step succeeded.
    plans:
        All :class:`ExecutionPlan` instances generated during the run
        (original + any re-plans), in creation order.
    total_steps_executed:
        Total number of :meth:`~PlanAndExecuteAgent.execute_step` calls made.
    n_replans:
        Number of re-planning events that occurred (excludes the initial plan).
    success:
        ``True`` if the run completed without exceeding ``max_replans`` and
        the final plan was fully executed with all steps succeeding.
    """

    final_answer: str
    plans: List[ExecutionPlan]
    total_steps_executed: int
    n_replans: int
    success: bool


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class PlanAndExecuteAgent:
    """Plan-and-Execute agent following Wang et al. 2023.

    The agent is fully model-agnostic: it delegates all text-generation
    work to callables provided by the caller at run time.

    Parameters
    ----------
    config:
        :class:`PlanConfig` controlling loop behaviour.  Defaults to
        :class:`PlanConfig` with its own defaults if ``None``.
    """

    def __init__(self, config: Optional[PlanConfig] = None) -> None:
        self.config: PlanConfig = config if config is not None else PlanConfig()

    # ------------------------------------------------------------------
    # Plan
    # ------------------------------------------------------------------

    def plan(
        self,
        task: str,
        context: str,
        plan_fn: Callable[[str, str], List[str]],
    ) -> ExecutionPlan:
        """Generate an :class:`ExecutionPlan` from a task description.

        Parameters
        ----------
        task:
            High-level task description to plan for.
        context:
            Additional context (e.g. background knowledge, constraints).
        plan_fn:
            ``(task, context) -> list[subtask_str]`` callable that produces
            an ordered list of subtask strings.

        Returns
        -------
        ExecutionPlan
            A plan whose steps are indexed from 0.
        """
        subtasks: List[str] = plan_fn(task, context)
        steps = [
            PlanStep(step_idx=i, task=subtask)
            for i, subtask in enumerate(subtasks)
        ]
        return ExecutionPlan(steps=steps, created_at_attempt=0)

    # ------------------------------------------------------------------
    # Execute a single step
    # ------------------------------------------------------------------

    def execute_step(
        self,
        step: PlanStep,
        context: str,
        execute_fn: Callable[[str, str], Tuple[str, bool]],
    ) -> PlanStep:
        """Execute a single :class:`PlanStep` in place and return it.

        Parameters
        ----------
        step:
            The step to execute.  Its ``result``, ``success``, and
            ``n_attempts`` attributes are updated in place.
        context:
            Execution context string (e.g. accumulated results so far).
        execute_fn:
            ``(task, context) -> (result_str, success_bool)`` callable.

        Returns
        -------
        PlanStep
            The same *step* object with updated attributes.
        """
        result, success = execute_fn(step.task, context)
        step.result = result
        step.success = success
        step.n_attempts += 1
        return step

    # ------------------------------------------------------------------
    # Re-plan
    # ------------------------------------------------------------------

    def replan(
        self,
        original_task: str,
        completed: List[PlanStep],
        failed_step: PlanStep,
        plan_fn: Callable[[str, str], List[str]],
        replan_attempt: int = 0,
    ) -> ExecutionPlan:
        """Generate a new :class:`ExecutionPlan` given partial completion.

        The new plan addresses the *original_task* but its context includes
        a summary of already-completed steps so the planner can skip
        redundant work.

        Parameters
        ----------
        original_task:
            The top-level task that was being solved.
        completed:
            Steps that have been successfully executed so far.
        failed_step:
            The step that failed and triggered this re-plan.
        plan_fn:
            Same ``(task, context) -> list[subtask_str]`` callable used for
            the original plan.
        replan_attempt:
            The re-plan iteration index (used to tag the new plan).

        Returns
        -------
        ExecutionPlan
            A fresh plan tagged with ``created_at_attempt=replan_attempt``.
        """
        # Build a context string that summarises what has been done.
        completed_summary_parts: List[str] = []
        for s in completed:
            completed_summary_parts.append(
                f"[Step {s.step_idx}] {s.task}: {s.result}"
            )
        completed_summary = "\n".join(completed_summary_parts) if completed_summary_parts else "None"

        context = (
            f"Original task: {original_task}\n"
            f"Completed steps:\n{completed_summary}\n"
            f"Failed step: {failed_step.task}\n"
            "Generate a new plan for the remaining work."
        )

        subtasks: List[str] = plan_fn(original_task, context)
        steps = [
            PlanStep(step_idx=i, task=subtask)
            for i, subtask in enumerate(subtasks)
        ]
        return ExecutionPlan(steps=steps, created_at_attempt=replan_attempt)

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(
        self,
        task: str,
        plan_fn: Callable[[str, str], List[str]],
        execute_fn: Callable[[str, str], Tuple[str, bool]],
    ) -> PlanExecuteResult:
        """Execute the full Plan-and-Execute loop.

        Algorithm
        ---------
        1. Call :meth:`plan` to get the initial :class:`ExecutionPlan`.
        2. Iterate through the plan's steps in order:
           a. Build a running context from previously completed results.
           b. Call :meth:`execute_step`.
           c. If the step fails and ``config.replan_on_failure`` is ``True``,
              call :meth:`replan` and switch to the new plan (unless
              ``config.max_replans`` has been reached).
        3. Stop when the current plan is exhausted, ``max_replans`` is
           exceeded, or ``max_steps`` total steps have been executed.

        Parameters
        ----------
        task:
            High-level task description.
        plan_fn:
            ``(task, context) -> list[subtask_str]`` callable.
        execute_fn:
            ``(task, context) -> (result_str, success_bool)`` callable.

        Returns
        -------
        PlanExecuteResult
        """
        all_plans: List[ExecutionPlan] = []
        n_replans: int = 0
        total_steps_executed: int = 0
        completed_steps: List[PlanStep] = []
        last_result: str = ""

        # Initial plan.
        current_plan = self.plan(task, "", plan_fn)
        all_plans.append(current_plan)

        while True:
            # Execute each pending step in the current plan.
            plan_done = False
            for step in current_plan.pending_steps:
                if total_steps_executed >= self.config.max_steps:
                    plan_done = True
                    break

                # Build running context from completed results.
                context_parts: List[str] = [f"Task: {task}"]
                for cs in completed_steps:
                    context_parts.append(f"[Done] {cs.task}: {cs.result}")
                context = "\n".join(context_parts)

                self.execute_step(step, context, execute_fn)
                total_steps_executed += 1

                if step.success:
                    completed_steps.append(step)
                    last_result = step.result or ""
                else:
                    # Step failed.
                    if (
                        self.config.replan_on_failure
                        and n_replans < self.config.max_replans
                    ):
                        n_replans += 1
                        current_plan = self.replan(
                            original_task=task,
                            completed=completed_steps,
                            failed_step=step,
                            plan_fn=plan_fn,
                            replan_attempt=n_replans,
                        )
                        all_plans.append(current_plan)
                        # Break the inner for-loop to restart with new plan.
                        break
                    else:
                        # No more replans allowed — record and continue.
                        last_result = step.result or ""
            else:
                # Inner loop exhausted without break — current plan is done.
                plan_done = True

            if plan_done:
                break

            # If we broke out of the inner loop due to a re-plan or step
            # limit, check whether we should stop.
            if total_steps_executed >= self.config.max_steps:
                break

        # Determine overall success: all plans completed, no pending steps in
        # the final plan, and every completed step succeeded.
        final_plan = all_plans[-1]
        all_completed = len(final_plan.pending_steps) == 0
        all_succeeded = all(s.success for s in final_plan.completed_steps)
        success = all_completed and all_succeeded and len(final_plan.steps) > 0

        return PlanExecuteResult(
            final_answer=last_result,
            plans=all_plans,
            total_steps_executed=total_steps_executed,
            n_replans=n_replans,
            success=success,
        )

    # ------------------------------------------------------------------
    # Post-run helpers
    # ------------------------------------------------------------------

    def statistics(self, result: PlanExecuteResult) -> dict[str, float]:
        """Compute summary statistics for a completed run.

        Returns
        -------
        dict with keys:
            * ``success_rate`` — 1.0 if ``result.success`` else 0.0
            * ``n_replans`` — number of re-planning events
            * ``total_steps`` — total execution steps taken
            * ``steps_per_plan`` — average steps per plan
            * ``plan_completion_rate`` — fraction of the final plan's steps
              that were completed (have a non-``None`` result)
        """
        n_plans = len(result.plans)
        steps_per_plan = (
            result.total_steps_executed / n_plans if n_plans > 0 else 0.0
        )
        final_plan = result.plans[-1] if result.plans else None
        if final_plan is not None and final_plan.n_steps > 0:
            plan_completion_rate = len(final_plan.completed_steps) / final_plan.n_steps
        else:
            plan_completion_rate = 0.0

        return {
            "success_rate": 1.0 if result.success else 0.0,
            "n_replans": float(result.n_replans),
            "total_steps": float(result.total_steps_executed),
            "steps_per_plan": float(steps_per_plan),
            "plan_completion_rate": float(plan_completion_rate),
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from src.agent import AGENT_LOOP_REGISTRY  # noqa: E402

AGENT_LOOP_REGISTRY["plan_and_execute"] = PlanAndExecuteAgent


__all__ = [
    "ExecutionPlan",
    "PlanAndExecuteAgent",
    "PlanConfig",
    "PlanExecuteResult",
    "PlanStep",
]
