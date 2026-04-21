"""
Unit tests for src/training/swe_rl.py — SWE-RL Trainer.

10–16 tests covering: config defaults, reward computation, patch evaluation,
best-of-N selection, policy gradient loss, and aggregate statistics.
"""

from __future__ import annotations

import math
import pytest
import torch

from src.training.swe_rl import (
    SWEPatch,
    SWEResult,
    SWERLConfig,
    SWERLTrainer,
    SWETask,
)


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

def _make_task(difficulty: str = "medium", n_tests: int = 5) -> SWETask:
    return SWETask(
        task_id="task_0",
        repo_context="def foo(): pass",
        issue_description="Fix the bug in foo()",
        test_cases=[f"test_{i}" for i in range(n_tests)],
        difficulty=difficulty,
    )


def _make_patch(task_id: str = "task_0", attempt: int = 0) -> SWEPatch:
    return SWEPatch(
        task_id=task_id,
        patch_text="--- a/foo.py\n+++ b/foo.py\n-pass\n+return 42",
        tokens_used=64,
        attempt_idx=attempt,
    )


def _make_result(
    passed: int,
    total: int,
    resolved: bool,
    reward: float = 0.0,
) -> SWEResult:
    return SWEResult(
        patch=_make_patch(),
        tests_passed=passed,
        tests_total=total,
        reward=reward,
        resolved=resolved,
    )


def _verifier_all_pass(patch_text: str, test_cases: list[str]) -> tuple[int, int]:
    return (len(test_cases), len(test_cases))


def _verifier_none_pass(patch_text: str, test_cases: list[str]) -> tuple[int, int]:
    return (0, len(test_cases))


def _verifier_partial(patch_text: str, test_cases: list[str]) -> tuple[int, int]:
    # Always return 3 out of total
    return (3, len(test_cases))


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    """SWERLConfig defaults: n_attempts=4, resolved_bonus=0.5."""
    cfg = SWERLConfig()
    assert cfg.n_attempts_per_task == 4
    assert cfg.resolved_bonus == 0.5
    assert cfg.max_patch_tokens == 4096
    assert cfg.pass_rate_reward is True
    assert cfg.partial_credit is True
    assert cfg.difficulty_weights == {"easy": 0.5, "medium": 1.0, "hard": 2.0}


# ---------------------------------------------------------------------------
# 2. test_compute_reward_resolved
# ---------------------------------------------------------------------------

def test_compute_reward_resolved():
    """All tests pass → base=1.0 + bonus=0.5 = 1.5 × medium_weight(1.0) = 1.5."""
    trainer = SWERLTrainer()
    task = _make_task("medium", n_tests=5)
    result = _make_result(5, 5, resolved=True)
    reward = trainer.compute_reward(result, task)
    assert math.isclose(reward, 1.5, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# 3. test_compute_reward_partial
# ---------------------------------------------------------------------------

def test_compute_reward_partial():
    """3/5 tests pass, not resolved → base=0.6, no bonus, medium weight → 0.6."""
    trainer = SWERLTrainer()
    task = _make_task("medium", n_tests=5)
    result = _make_result(3, 5, resolved=False)
    reward = trainer.compute_reward(result, task)
    assert math.isclose(reward, 0.6, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# 4. test_compute_reward_zero
# ---------------------------------------------------------------------------

def test_compute_reward_zero():
    """0/5 tests pass → reward = 0.0."""
    trainer = SWERLTrainer()
    task = _make_task("medium", n_tests=5)
    result = _make_result(0, 5, resolved=False)
    reward = trainer.compute_reward(result, task)
    assert math.isclose(reward, 0.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# 5. test_compute_reward_difficulty_weight
# ---------------------------------------------------------------------------

def test_compute_reward_difficulty_weight():
    """Hard task reward > medium task reward for same pass rate."""
    trainer = SWERLTrainer()
    result = _make_result(5, 5, resolved=True)

    hard_task = _make_task("hard", n_tests=5)
    medium_task = _make_task("medium", n_tests=5)

    reward_hard = trainer.compute_reward(result, hard_task)
    reward_medium = trainer.compute_reward(result, medium_task)

    assert reward_hard > reward_medium


# ---------------------------------------------------------------------------
# 6. test_compute_reward_no_pass_rate
# ---------------------------------------------------------------------------

def test_compute_reward_no_pass_rate():
    """pass_rate_reward=False → binary: 1.0 if resolved, 0.0 otherwise."""
    cfg = SWERLConfig(pass_rate_reward=False, resolved_bonus=0.0)
    trainer = SWERLTrainer(cfg)
    task = _make_task("medium", n_tests=5)

    # Partial pass, not resolved → 0.0
    result_partial = _make_result(3, 5, resolved=False)
    assert math.isclose(trainer.compute_reward(result_partial, task), 0.0, abs_tol=1e-9)

    # Fully resolved → 1.0
    result_resolved = _make_result(5, 5, resolved=True)
    assert math.isclose(trainer.compute_reward(result_resolved, task), 1.0, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# 7. test_evaluate_patch_correct
# ---------------------------------------------------------------------------

def test_evaluate_patch_correct():
    """verifier returns (5, 5) → resolved=True, reward > 1.0 (includes bonus)."""
    trainer = SWERLTrainer()
    task = _make_task("medium", n_tests=5)
    patch = _make_patch()

    result = trainer.evaluate_patch(patch, task, _verifier_all_pass)

    assert result.resolved is True
    assert result.tests_passed == 5
    assert result.tests_total == 5
    assert result.reward > 1.0   # base(1.0) + bonus(0.5) × weight(1.0) = 1.5


# ---------------------------------------------------------------------------
# 8. test_evaluate_patch_partial
# ---------------------------------------------------------------------------

def test_evaluate_patch_partial():
    """verifier returns (3, 5) → resolved=False, reward ~= 0.6."""
    trainer = SWERLTrainer()
    task = _make_task("medium", n_tests=5)
    patch = _make_patch()

    result = trainer.evaluate_patch(patch, task, _verifier_partial)

    assert result.resolved is False
    assert result.tests_passed == 3
    assert result.tests_total == 5
    assert math.isclose(result.reward, 0.6, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# 9. test_best_of_n_picks_best
# ---------------------------------------------------------------------------

def test_best_of_n_picks_best():
    """best_of_n returns the patch with the highest reward."""
    trainer = SWERLTrainer()
    task = _make_task("medium", n_tests=5)

    call_count = [0]

    def verifier(patch_text: str, test_cases: list[str]) -> tuple[int, int]:
        idx = call_count[0]
        call_count[0] += 1
        # patch 0 → 1/5, patch 1 → 3/5, patch 2 → 5/5 (best)
        passes = [1, 3, 5]
        return (passes[idx], len(test_cases))

    patches = [_make_patch(attempt=i) for i in range(3)]
    best = trainer.best_of_n(patches, task, verifier)

    assert best.tests_passed == 5
    assert best.resolved is True


# ---------------------------------------------------------------------------
# 10. test_best_of_n_all_zero
# ---------------------------------------------------------------------------

def test_best_of_n_all_zero():
    """All patches have 0 reward → function returns without crashing."""
    trainer = SWERLTrainer()
    task = _make_task("medium", n_tests=5)
    patches = [_make_patch(attempt=i) for i in range(4)]
    result = trainer.best_of_n(patches, task, _verifier_none_pass)
    assert result is not None
    assert result.reward == 0.0
    assert result.resolved is False


# ---------------------------------------------------------------------------
# 11. test_policy_loss_scalar
# ---------------------------------------------------------------------------

def test_policy_loss_scalar():
    """compute_policy_loss returns a scalar tensor."""
    trainer = SWERLTrainer()
    log_probs = torch.log(torch.tensor([0.9, 0.5, 0.1]))
    rewards = torch.tensor([1.5, 0.6, 0.0])
    loss = trainer.compute_policy_loss(log_probs, rewards)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0   # scalar


# ---------------------------------------------------------------------------
# 12. test_policy_loss_gradient
# ---------------------------------------------------------------------------

def test_policy_loss_gradient():
    """Backward pass through compute_policy_loss does not raise."""
    trainer = SWERLTrainer()
    # Use a leaf tensor directly (already in log-space) so .grad is populated.
    log_probs = torch.tensor([-0.22, -0.92, -1.61], requires_grad=True)
    rewards = torch.tensor([1.0, 0.5, 0.0])
    loss = trainer.compute_policy_loss(log_probs, rewards)
    loss.backward()

    assert log_probs.grad is not None
    assert log_probs.grad.shape == log_probs.shape


# ---------------------------------------------------------------------------
# 13. test_statistics_keys
# ---------------------------------------------------------------------------

def test_statistics_keys():
    """statistics() returns dict with the four required keys."""
    trainer = SWERLTrainer()
    results = [
        _make_result(5, 5, resolved=True, reward=1.5),
        _make_result(3, 5, resolved=False, reward=0.6),
    ]
    stats = trainer.statistics(results)

    assert "resolve_rate" in stats
    assert "mean_reward" in stats
    assert "mean_pass_rate" in stats
    assert "by_difficulty" in stats


# ---------------------------------------------------------------------------
# 14. test_statistics_resolve_rate
# ---------------------------------------------------------------------------

def test_statistics_resolve_rate():
    """All resolved → resolve_rate = 1.0."""
    trainer = SWERLTrainer()
    results = [
        _make_result(5, 5, resolved=True, reward=1.5),
        _make_result(5, 5, resolved=True, reward=1.5),
        _make_result(5, 5, resolved=True, reward=1.5),
    ]
    stats = trainer.statistics(results)
    assert math.isclose(stats["resolve_rate"], 1.0, rel_tol=1e-6)
    assert math.isclose(stats["mean_pass_rate"], 1.0, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# 15. test_statistics_by_difficulty
# ---------------------------------------------------------------------------

def test_statistics_by_difficulty():
    """statistics() groups results by _difficulty attribute correctly."""
    trainer = SWERLTrainer()
    task_easy = _make_task("easy", n_tests=5)
    task_hard = _make_task("hard", n_tests=5)

    patch = _make_patch()

    # Easy task: resolved
    r_easy = trainer.evaluate_patch_with_difficulty(patch, task_easy, _verifier_all_pass)
    # Hard task: not resolved
    r_hard = trainer.evaluate_patch_with_difficulty(patch, task_hard, _verifier_none_pass)

    stats = trainer.statistics([r_easy, r_hard])

    by_diff = stats["by_difficulty"]
    assert "easy" in by_diff
    assert "hard" in by_diff
    assert by_diff["easy"]["n"] == 1
    assert by_diff["hard"]["n"] == 1
    assert math.isclose(by_diff["easy"]["resolve_rate"], 1.0, rel_tol=1e-6)
    assert math.isclose(by_diff["hard"]["resolve_rate"], 0.0, abs_tol=1e-9)
