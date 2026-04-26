"""Absolute Zero: Self-play RL with no human-labeled data (Zhao et al. 2025).

The model proposes its own verifiable tasks, solves them, and learns from the
outcome. Three task types are supported:
  - deduction:  given a program, predict the output.
  - abduction:  given input + output, infer the program.
  - induction:  given examples, generalise the pattern.

Components:
    AbsoluteZeroConfig  -- hyper-parameters for the trainer.
    AbsoluteZeroTask    -- a single self-proposed task.
    AbsoluteZeroRollout -- a task + solution + reward.
    AbsoluteZeroTrainer -- orchestrates propose → solve → verify → PG.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AbsoluteZeroConfig:
    """Hyper-parameters for Absolute Zero self-play RL.

    Args:
        task_types: List of task type names to cycle through.
        temperature_propose: Sampling temperature used by the task proposer.
        temperature_solve: Sampling temperature used by the solver.
        n_propose_candidates: How many task candidates the proposer generates
            per task type per call.
        n_solve_attempts: How many solution attempts are made per task.
        reward_correct: Reward given for a correct solution.
        reward_incorrect: Reward given for an incorrect solution.
        leakage_penalty: Additional reward (negative) for a trivially easy task
            where the answer is directly readable from the task description.
    """

    task_types: list[str] = field(default_factory=lambda: ["deduction", "abduction", "induction"])
    temperature_propose: float = 0.9
    temperature_solve: float = 0.7
    n_propose_candidates: int = 4
    n_solve_attempts: int = 2
    reward_correct: float = 1.0
    reward_incorrect: float = 0.0
    leakage_penalty: float = -0.5


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AbsoluteZeroTask:
    """A single self-proposed verifiable task.

    Args:
        task_id: Unique, sequentially assigned identifier.
        task_type: One of "deduction", "abduction", "induction".
        task_tokens: Tokenised task description.
        answer_tokens: Ground-truth answer tokens (used for verification).
        proposed_by: Who proposed the task (default "model").
    """

    task_id: int
    task_type: str
    task_tokens: list[int]
    answer_tokens: list[int]
    proposed_by: str = "model"


@dataclass
class AbsoluteZeroRollout:
    """A task paired with the model's solution attempt and associated reward.

    Args:
        task: The task that was attempted.
        solution_tokens: Tokens produced by the solver.
        is_correct: Whether the solution matched the ground-truth answer.
        reward: Final scalar reward (after any penalties).
        leakage_detected: True when the answer was found verbatim in the task.
    """

    task: AbsoluteZeroTask
    solution_tokens: list[int]
    is_correct: bool
    reward: float
    leakage_detected: bool = False


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class AbsoluteZeroTrainer:
    """Orchestrates the Absolute Zero self-play RL loop.

    The life-cycle per iteration is:
        1. propose_tasks      — model generates candidate tasks.
        2. solve_tasks        — model attempts to answer each task.
        3. apply_leakage_penalty — penalise trivially easy tasks.
        4. compute_policy_gradient — REINFORCE update.

    Args:
        config: Trainer hyper-parameters (uses defaults if not provided).
    """

    def __init__(self, config: AbsoluteZeroConfig | None = None) -> None:
        self.config = config if config is not None else AbsoluteZeroConfig()
        self._next_task_id: int = 0

    # ------------------------------------------------------------------
    # Task proposal
    # ------------------------------------------------------------------

    def propose_tasks(
        self,
        propose_fn: Callable[[str, int], list[list[int]]],
        n: int | None = None,
    ) -> list[AbsoluteZeroTask]:
        """Generate candidate tasks for every configured task type.

        For each task type, ``propose_fn(task_type, n_candidates)`` is called
        and is expected to return a list of token sequences.  Each returned
        sequence is treated as *both* the task description and (by convention)
        its own ground-truth answer — callers may override ``answer_tokens``
        after the fact if a separate answer is available.

        The first half of the returned tokens are used as ``task_tokens`` and
        the second half as ``answer_tokens`` to ensure they are distinct (for
        leakage detection purposes the two halves are structurally separate).

        Args:
            propose_fn: Callable ``(task_type, n_candidates) → list[list[int]]``.
                Returns one token sequence per candidate.
            n: Override for ``config.n_propose_candidates``.

        Returns:
            List of :class:`AbsoluteZeroTask` objects with sequential IDs.
        """
        n_candidates = n if n is not None else self.config.n_propose_candidates
        tasks: list[AbsoluteZeroTask] = []

        for task_type in self.config.task_types:
            token_seqs: list[list[int]] = propose_fn(task_type, n_candidates)
            for tokens in token_seqs:
                # Split token sequence: first half → task, second half → answer.
                # If the sequence has an odd length, the extra token goes to the
                # task half.
                mid = (len(tokens) + 1) // 2
                task_tokens = tokens[:mid]
                answer_tokens = tokens[mid:]
                task = AbsoluteZeroTask(
                    task_id=self._next_task_id,
                    task_type=task_type,
                    task_tokens=task_tokens,
                    answer_tokens=answer_tokens,
                )
                tasks.append(task)
                self._next_task_id += 1

        return tasks

    # ------------------------------------------------------------------
    # Solving
    # ------------------------------------------------------------------

    def solve_tasks(
        self,
        tasks: list[AbsoluteZeroTask],
        solve_fn: Callable[[list[int]], list[int]],
    ) -> list[AbsoluteZeroRollout]:
        """Attempt to solve each task and assign base rewards.

        ``solve_fn(task_tokens) → solution_tokens`` is called once per task.
        Correctness is determined by exact token-sequence match against
        ``task.answer_tokens``.

        Args:
            tasks: Tasks to solve.
            solve_fn: Callable that maps task tokens → solution tokens.

        Returns:
            List of :class:`AbsoluteZeroRollout` objects (one per task).
        """
        rollouts: list[AbsoluteZeroRollout] = []
        for task in tasks:
            solution_tokens = solve_fn(task.task_tokens)
            is_correct = solution_tokens == task.answer_tokens
            reward = self.config.reward_correct if is_correct else self.config.reward_incorrect
            rollouts.append(
                AbsoluteZeroRollout(
                    task=task,
                    solution_tokens=solution_tokens,
                    is_correct=is_correct,
                    reward=reward,
                )
            )
        return rollouts

    # ------------------------------------------------------------------
    # Leakage detection
    # ------------------------------------------------------------------

    def detect_leakage(self, task: AbsoluteZeroTask) -> bool:
        """Return True if ``answer_tokens`` is a contiguous sub-sequence of
        ``task_tokens``.

        A task is considered trivial (leaking its answer) when the ground-truth
        answer can be read verbatim from the task description.

        Args:
            task: Task to inspect.

        Returns:
            ``True`` if the answer is embedded in the task description.
        """
        answer = task.answer_tokens
        task_tok = task.task_tokens
        if not answer:
            return False
        n = len(answer)
        return any(task_tok[i : i + n] == answer for i in range(len(task_tok) - n + 1))

    def apply_leakage_penalty(
        self, rollouts: list[AbsoluteZeroRollout]
    ) -> list[AbsoluteZeroRollout]:
        """Apply ``config.leakage_penalty`` to any rollout whose task leaks its answer.

        Args:
            rollouts: Rollouts to inspect and (possibly) modify in place.

        Returns:
            The same list with rewards and ``leakage_detected`` flags updated.
        """
        for rollout in rollouts:
            if self.detect_leakage(rollout.task):
                rollout.reward += self.config.leakage_penalty
                rollout.leakage_detected = True
        return rollouts

    # ------------------------------------------------------------------
    # Policy gradient
    # ------------------------------------------------------------------

    def compute_policy_gradient(
        self,
        rollouts: list[AbsoluteZeroRollout],
        log_probs: Tensor,
    ) -> Tensor:
        """Compute a REINFORCE policy-gradient loss.

        Loss = ``-mean(reward_i * log_prob_i)`` over all rollouts.

        Positive rewards on correct rollouts produce a *negative* loss, which
        when minimised will increase the log-probability of correct solutions —
        the standard REINFORCE update.

        Args:
            rollouts: Completed rollouts (provides rewards).
            log_probs: 1-D tensor of shape ``[n_rollouts]`` — the log-
                probability of each rollout's solution under the current policy.

        Returns:
            Scalar loss tensor.
        """
        rewards = torch.tensor(
            [r.reward for r in rollouts],
            dtype=log_probs.dtype,
            device=log_probs.device,
        )
        loss = -(rewards * log_probs).mean()
        return loss

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def statistics(self, rollouts: list[AbsoluteZeroRollout]) -> dict:
        """Compute summary statistics over a batch of rollouts.

        Args:
            rollouts: Completed rollouts to summarise.

        Returns:
            Dictionary with keys:
                - ``"accuracy"`` (float): fraction of correct rollouts.
                - ``"leakage_rate"`` (float): fraction of rollouts with leakage.
                - ``"mean_reward"`` (float): mean reward across rollouts.
                - ``"by_type"`` (dict): per-task-type ``{"accuracy": float, "n": int}``.
        """
        if not rollouts:
            return {
                "accuracy": 0.0,
                "leakage_rate": 0.0,
                "mean_reward": 0.0,
                "by_type": {},
            }

        n_total = len(rollouts)
        n_correct = sum(r.is_correct for r in rollouts)
        n_leaked = sum(r.leakage_detected for r in rollouts)
        mean_reward = sum(r.reward for r in rollouts) / n_total

        # Per-type breakdown
        by_type: dict[str, dict] = {}
        for rollout in rollouts:
            tt = rollout.task.task_type
            if tt not in by_type:
                by_type[tt] = {"correct": 0, "n": 0}
            by_type[tt]["n"] += 1
            if rollout.is_correct:
                by_type[tt]["correct"] += 1

        by_type_out = {
            tt: {"accuracy": v["correct"] / v["n"], "n": v["n"]} for tt, v in by_type.items()
        }

        return {
            "accuracy": n_correct / n_total,
            "leakage_rate": n_leaked / n_total,
            "mean_reward": mean_reward,
            "by_type": by_type_out,
        }
