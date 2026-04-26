"""
SWE-RL: Software Engineering RL Training (2025).

Train agents on software engineering tasks (bug fixes, feature additions)
using test suite execution as the reward signal. The model generates patches;
a verifier runs tests (simulated as a callable) and returns pass/fail counts.

Classes:
    SWETask          — task spec: repo context, issue, test cases, difficulty
    SWEPatch         — generated patch with token usage metadata
    SWEResult        — evaluation result with reward
    SWERLConfig      — hyperparameters
    SWERLTrainer     — reward computation, patch evaluation, policy loss
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SWETask:
    """Specification for a software engineering RL task."""

    task_id: str
    repo_context: str  # condensed repo context (docstrings, signatures)
    issue_description: str
    test_cases: list[str]  # test identifiers (simulated)
    difficulty: str = "medium"  # "easy", "medium", "hard"
    metadata: dict = field(default_factory=dict)


@dataclass
class SWEPatch:
    """A generated patch produced by the model."""

    task_id: str
    patch_text: str  # generated patch
    tokens_used: int
    attempt_idx: int = 0


@dataclass
class SWEResult:
    """Evaluation result for a single patch against a task."""

    patch: SWEPatch
    tests_passed: int
    tests_total: int
    reward: float
    resolved: bool  # all tests pass


@dataclass
class SWERLConfig:
    """Hyperparameters for SWERLTrainer."""

    max_patch_tokens: int = 4096
    n_attempts_per_task: int = 4  # generate N patches, take best
    pass_rate_reward: bool = True  # reward = tests_passed/tests_total, not just 0/1
    resolved_bonus: float = 0.5  # extra reward for fully resolved
    partial_credit: bool = True
    difficulty_weights: dict = field(
        default_factory=lambda: {"easy": 0.5, "medium": 1.0, "hard": 2.0}
    )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class SWERLTrainer:
    """
    Software Engineering RL Trainer.

    Implements reward computation, patch evaluation, best-of-N selection,
    REINFORCE policy gradient loss, and aggregate statistics.

    Usage::

        config = SWERLConfig()
        trainer = SWERLTrainer(config)

        task = SWETask("t0", "...", "Fix bug", ["test_a", "test_b"])
        patch = SWEPatch("t0", "--- a/foo.py\\n+++ b/foo.py\\n...", tokens_used=128)

        def verifier(patch_text, test_cases):
            return (len(test_cases), len(test_cases))   # all pass

        result = trainer.evaluate_patch(patch, task, verifier)
        print(result.reward, result.resolved)
    """

    def __init__(self, config: SWERLConfig | None = None) -> None:
        self.config = config if config is not None else SWERLConfig()

    # ------------------------------------------------------------------
    # Core reward computation
    # ------------------------------------------------------------------

    def compute_reward(self, result: SWEResult, task: SWETask | None = None) -> float:
        """Compute the scalar reward for a SWEResult.

        Algorithm:
          - base = tests_passed / tests_total   (if pass_rate_reward)
                   1.0 if resolved else 0.0     (if not pass_rate_reward)
          - base += resolved_bonus              (if resolved)
          - reward = base * difficulty_weight

        Args:
            result: The evaluation result containing pass/fail counts.
            task:   Optional SWETask for difficulty weighting. If None,
                    defaults to "medium" weight.

        Returns:
            Float reward value.
        """
        cfg = self.config
        tests_passed = result.tests_passed
        tests_total = result.tests_total

        if tests_total == 0:
            base = 0.0
        elif cfg.pass_rate_reward:
            base = tests_passed / tests_total
        else:
            base = 1.0 if result.resolved else 0.0

        if result.resolved:
            base += cfg.resolved_bonus

        # Difficulty weight
        if task is not None:
            difficulty = task.difficulty
        else:
            difficulty = "medium"
        weight = cfg.difficulty_weights.get(difficulty, 1.0)

        return base * weight

    # ------------------------------------------------------------------
    # Patch evaluation
    # ------------------------------------------------------------------

    def evaluate_patch(
        self,
        patch: SWEPatch,
        task: SWETask,
        verifier_fn: Callable[[str, list[str]], tuple[int, int]],
    ) -> SWEResult:
        """Evaluate a single patch against a task using a verifier function.

        Args:
            patch:       The patch to evaluate.
            task:        The task the patch is targeting.
            verifier_fn: Callable(patch_text, test_cases) → (tests_passed, tests_total).

        Returns:
            SWEResult with reward and resolved flag populated.
        """
        tests_passed, tests_total = verifier_fn(patch.patch_text, task.test_cases)
        resolved = tests_passed == tests_total and tests_total > 0
        # Build a partial result first (reward=0.0 placeholder) so compute_reward
        # can inspect the resolved flag.
        partial = SWEResult(
            patch=patch,
            tests_passed=tests_passed,
            tests_total=tests_total,
            reward=0.0,
            resolved=resolved,
        )
        reward = self.compute_reward(partial, task)
        return SWEResult(
            patch=patch,
            tests_passed=tests_passed,
            tests_total=tests_total,
            reward=reward,
            resolved=resolved,
        )

    # ------------------------------------------------------------------
    # Best-of-N selection
    # ------------------------------------------------------------------

    def best_of_n(
        self,
        patches: list[SWEPatch],
        task: SWETask,
        verifier_fn: Callable[[str, list[str]], tuple[int, int]],
    ) -> SWEResult:
        """Evaluate all patches and return the one with the highest reward.

        If multiple patches tie, the first maximum-reward patch is returned.
        If *patches* is empty, raises ValueError.

        Args:
            patches:     List of candidate patches to evaluate.
            task:        The task each patch targets.
            verifier_fn: Callable(patch_text, test_cases) → (tests_passed, tests_total).

        Returns:
            The SWEResult with the highest reward.

        Raises:
            ValueError: If *patches* is empty.
        """
        if not patches:
            raise ValueError("patches must not be empty.")

        results = [self.evaluate_patch(p, task, verifier_fn) for p in patches]
        return max(results, key=lambda r: r.reward)

    # ------------------------------------------------------------------
    # Policy gradient loss
    # ------------------------------------------------------------------

    def compute_policy_loss(self, log_probs: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        """REINFORCE policy gradient loss.

        loss = -mean(reward * log_prob)

        Args:
            log_probs: Tensor of shape [B] — log-probabilities of selected actions.
            rewards:   Tensor of shape [B] — scalar rewards per action.

        Returns:
            Scalar loss tensor (differentiable w.r.t. log_probs).
        """
        if log_probs.shape != rewards.shape:
            raise ValueError(
                f"log_probs and rewards must have the same shape, "
                f"got {log_probs.shape} vs {rewards.shape}"
            )
        return -(rewards * log_probs).mean()

    # ------------------------------------------------------------------
    # Aggregate statistics
    # ------------------------------------------------------------------

    def statistics(self, results: list[SWEResult]) -> dict:
        """Compute aggregate statistics over a list of evaluation results.

        Args:
            results: List of SWEResult objects (may be from different tasks/difficulties).

        Returns:
            dict with keys:
                - ``resolve_rate``:   fraction of results where resolved=True
                - ``mean_reward``:    mean reward across all results
                - ``mean_pass_rate``: mean (tests_passed / tests_total) across all results
                - ``by_difficulty``:  dict mapping difficulty string to
                                      {"resolve_rate": float, "n": int}
        """
        if not results:
            return {
                "resolve_rate": 0.0,
                "mean_reward": 0.0,
                "mean_pass_rate": 0.0,
                "by_difficulty": {},
            }

        n = len(results)
        resolve_rate = sum(1 for r in results if r.resolved) / n
        mean_reward = sum(r.reward for r in results) / n

        pass_rates = []
        for r in results:
            if r.tests_total > 0:
                pass_rates.append(r.tests_passed / r.tests_total)
            else:
                pass_rates.append(0.0)
        mean_pass_rate = sum(pass_rates) / n

        # Group by difficulty (stored in patch.task_id is not difficulty,
        # so we store difficulty on the task — but SWEResult only holds patch+counts).
        # We expose by_difficulty grouped by the difficulty tag baked into metadata
        # if available; otherwise we fall back to a single "unknown" bucket.
        # Convention: callers may set result.patch.task_id to carry difficulty
        # via metadata, but the clean API is to call statistics() with results
        # produced by evaluate_patch() which knows the task.  Since SWEResult
        # does not directly store difficulty, we extract it from
        # result.patch via a side-channel dict if the caller attached one,
        # or expose a separate helper.  For maximum test-compatibility we
        # support a "_difficulty" attribute optionally set on SWEResult.
        by_difficulty: dict[str, dict] = {}
        for r in results:
            diff = getattr(r, "_difficulty", None) or "unknown"
            if diff not in by_difficulty:
                by_difficulty[diff] = {"resolved": 0, "n": 0}
            by_difficulty[diff]["n"] += 1
            if r.resolved:
                by_difficulty[diff]["resolved"] += 1

        by_difficulty_out = {
            diff: {
                "resolve_rate": v["resolved"] / v["n"] if v["n"] > 0 else 0.0,
                "n": v["n"],
            }
            for diff, v in by_difficulty.items()
        }

        return {
            "resolve_rate": resolve_rate,
            "mean_reward": mean_reward,
            "mean_pass_rate": mean_pass_rate,
            "by_difficulty": by_difficulty_out,
        }

    def evaluate_patch_with_difficulty(
        self,
        patch: SWEPatch,
        task: SWETask,
        verifier_fn: Callable[[str, list[str]], tuple[int, int]],
    ) -> SWEResult:
        """Like evaluate_patch but tags the result with task difficulty.

        This allows statistics() to correctly group results by difficulty.

        Args:
            patch:       The patch to evaluate.
            task:        The SWETask (difficulty is read from task.difficulty).
            verifier_fn: Callable(patch_text, test_cases) → (tests_passed, tests_total).

        Returns:
            SWEResult with a ``_difficulty`` attribute set.
        """
        result = self.evaluate_patch(patch, task, verifier_fn)
        result._difficulty = task.difficulty  # type: ignore[attr-defined]
        return result
