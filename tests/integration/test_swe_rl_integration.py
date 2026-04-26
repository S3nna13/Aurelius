"""
Integration test for SWERLTrainer.

Verifies end-to-end flow:
  - 3 tasks (easy / medium / hard) with distinct mock verifiers
  - evaluate_patch_with_difficulty + best_of_n
  - statistics() with correct grouping
  - compute_policy_loss backward pass works
  - TRAINING_REGISTRY["swe_rl"] is wired to SWERLTrainer
"""

from __future__ import annotations

import math

import pytest
import torch

from src.training import TRAINING_REGISTRY
from src.training.swe_rl import (
    SWEPatch,
    SWERLConfig,
    SWERLTrainer,
    SWETask,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def trainer() -> SWERLTrainer:
    cfg = SWERLConfig(
        n_attempts_per_task=3,
        pass_rate_reward=True,
        resolved_bonus=0.5,
        difficulty_weights={"easy": 0.5, "medium": 1.0, "hard": 2.0},
    )
    return SWERLTrainer(cfg)


@pytest.fixture
def tasks() -> list[SWETask]:
    return [
        SWETask(
            task_id="easy_task",
            repo_context="def add(a, b): pass",
            issue_description="Fix add() to return sum.",
            test_cases=["test_add_1", "test_add_2", "test_add_3", "test_add_4", "test_add_5"],
            difficulty="easy",
        ),
        SWETask(
            task_id="medium_task",
            repo_context="def sort(lst): pass",
            issue_description="Fix sort() to return sorted list.",
            test_cases=["test_sort_1", "test_sort_2", "test_sort_3", "test_sort_4", "test_sort_5"],
            difficulty="medium",
        ),
        SWETask(
            task_id="hard_task",
            repo_context="def parse(src): pass",
            issue_description="Fix parser to handle edge cases.",
            test_cases=[
                "test_parse_1",
                "test_parse_2",
                "test_parse_3",
                "test_parse_4",
                "test_parse_5",
            ],
            difficulty="hard",
        ),
    ]


def _make_verifier(passes: int):
    """Return a verifier that always returns (passes, total)."""

    def verifier(patch_text: str, test_cases: list[str]) -> tuple[int, int]:
        return (passes, len(test_cases))

    return verifier


# Verifiers: easy → all pass (5/5), medium → partial (3/5), hard → none (0/5)
verifier_easy = _make_verifier(5)
verifier_medium = _make_verifier(3)
verifier_hard = _make_verifier(0)


# ---------------------------------------------------------------------------
# Helper: build one patch per task
# ---------------------------------------------------------------------------


def _patch_for(task: SWETask, attempt: int = 0) -> SWEPatch:
    return SWEPatch(
        task_id=task.task_id,
        patch_text=f"--- fix for {task.task_id} ---",
        tokens_used=128,
        attempt_idx=attempt,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_integration_evaluate_easy_task(trainer, tasks):
    """Easy task with all-pass verifier → resolved=True, correct reward."""
    task = tasks[0]  # easy
    patch = _patch_for(task)
    result = trainer.evaluate_patch_with_difficulty(patch, task, verifier_easy)

    assert result.resolved is True
    assert result.tests_passed == 5
    assert result.tests_total == 5
    # base(1.0) + bonus(0.5) × weight(0.5) = 0.75
    assert math.isclose(result.reward, 0.75, rel_tol=1e-6)
    assert getattr(result, "_difficulty", None) == "easy"


def test_integration_evaluate_medium_task(trainer, tasks):
    """Medium task with 3/5 verifier → resolved=False, reward = 0.6."""
    task = tasks[1]  # medium
    patch = _patch_for(task)
    result = trainer.evaluate_patch_with_difficulty(patch, task, verifier_medium)

    assert result.resolved is False
    assert result.tests_passed == 3
    assert result.tests_total == 5
    # base(0.6) × weight(1.0) = 0.6
    assert math.isclose(result.reward, 0.6, rel_tol=1e-6)
    assert getattr(result, "_difficulty", None) == "medium"


def test_integration_evaluate_hard_task(trainer, tasks):
    """Hard task with 0/5 verifier → resolved=False, reward = 0.0."""
    task = tasks[2]  # hard
    patch = _patch_for(task)
    result = trainer.evaluate_patch_with_difficulty(patch, task, verifier_hard)

    assert result.resolved is False
    assert result.tests_passed == 0
    assert result.reward == 0.0
    assert getattr(result, "_difficulty", None) == "hard"


def test_integration_best_of_n(trainer, tasks):
    """best_of_n over 3 patches with different verifiers picks highest reward."""
    task = tasks[1]  # medium

    call_idx = [0]
    pass_counts = [0, 2, 5]  # attempt 0→0, 1→2, 2→5

    def staged_verifier(patch_text: str, test_cases: list[str]) -> tuple[int, int]:
        p = pass_counts[call_idx[0]]
        call_idx[0] += 1
        return (p, len(test_cases))

    patches = [_patch_for(task, attempt=i) for i in range(3)]
    best = trainer.best_of_n(patches, task, staged_verifier)

    assert best.tests_passed == 5
    assert best.resolved is True


def test_integration_statistics_keys_and_values(trainer, tasks):
    """statistics() over all three task results has correct structure and values."""
    results = []
    verifiers = [verifier_easy, verifier_medium, verifier_hard]
    for task, verifier in zip(tasks, verifiers):
        patch = _patch_for(task)
        result = trainer.evaluate_patch_with_difficulty(patch, task, verifier)
        results.append(result)

    stats = trainer.statistics(results)

    # Keys
    assert set(stats.keys()) >= {"resolve_rate", "mean_reward", "mean_pass_rate", "by_difficulty"}

    # resolve_rate: 1 out of 3 resolved
    assert math.isclose(stats["resolve_rate"], 1.0 / 3.0, rel_tol=1e-5)

    # mean_pass_rate: (5/5 + 3/5 + 0/5) / 3 = 8/15
    expected_mpr = (1.0 + 0.6 + 0.0) / 3.0
    assert math.isclose(stats["mean_pass_rate"], expected_mpr, rel_tol=1e-5)

    # by_difficulty grouping
    by_diff = stats["by_difficulty"]
    assert "easy" in by_diff and by_diff["easy"]["n"] == 1
    assert "medium" in by_diff and by_diff["medium"]["n"] == 1
    assert "hard" in by_diff and by_diff["hard"]["n"] == 1
    assert math.isclose(by_diff["easy"]["resolve_rate"], 1.0, rel_tol=1e-6)
    assert math.isclose(by_diff["medium"]["resolve_rate"], 0.0, abs_tol=1e-9)
    assert math.isclose(by_diff["hard"]["resolve_rate"], 0.0, abs_tol=1e-9)


def test_integration_policy_loss_backward(trainer):
    """compute_policy_loss backward pass runs cleanly end-to-end."""
    # Leaf tensor in log-space so .grad is populated after backward().
    log_probs = torch.tensor([-0.105, -0.511, -2.303], requires_grad=True)
    rewards = torch.tensor([0.75, 0.6, 0.0])

    loss = trainer.compute_policy_loss(log_probs, rewards)
    assert loss.ndim == 0  # scalar

    loss.backward()
    assert log_probs.grad is not None
    assert not torch.isnan(log_probs.grad).any()


def test_integration_registry_wired():
    """TRAINING_REGISTRY['swe_rl'] must be SWERLTrainer."""
    assert "swe_rl" in TRAINING_REGISTRY, (
        "TRAINING_REGISTRY missing 'swe_rl' key — ensure __init__.py registers it."
    )
    assert TRAINING_REGISTRY["swe_rl"] is SWERLTrainer


def test_integration_registry_instantiable():
    """The registry entry can be instantiated with default config."""
    cls = TRAINING_REGISTRY["swe_rl"]
    trainer = cls()
    task = SWETask(
        task_id="reg_task",
        repo_context="...",
        issue_description="...",
        test_cases=["t0", "t1"],
    )
    patch = SWEPatch(task_id="reg_task", patch_text="fix", tokens_used=10)
    result = trainer.evaluate_patch(patch, task, lambda pt, tc: (len(tc), len(tc)))
    assert result.resolved is True
