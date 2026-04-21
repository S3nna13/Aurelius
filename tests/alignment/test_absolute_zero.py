"""Unit tests for src/alignment/absolute_zero.py — 15 tests."""
from __future__ import annotations

import torch
import pytest

from src.alignment.absolute_zero import (
    AbsoluteZeroConfig,
    AbsoluteZeroTask,
    AbsoluteZeroRollout,
    AbsoluteZeroTrainer,
)
from src.alignment import ALIGNMENT_REGISTRY


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_propose_fn(tokens_per_candidate: list[int] | None = None):
    """Return a propose_fn that emits deterministic token sequences.

    tokens_per_candidate: list of token-sequence lengths to emit for each
    candidate.  If None a simple default is used.
    """
    counter = [0]

    def propose_fn(task_type: str, n: int) -> list[list[int]]:
        seqs = []
        for _ in range(n):
            length = 6 if tokens_per_candidate is None else tokens_per_candidate[counter[0] % len(tokens_per_candidate)]
            seq = list(range(counter[0] * 100, counter[0] * 100 + length))
            counter[0] += 1
            seqs.append(seq)
        return seqs

    return propose_fn


def _make_task(
    task_type: str = "deduction",
    task_tokens: list[int] | None = None,
    answer_tokens: list[int] | None = None,
    task_id: int = 0,
) -> AbsoluteZeroTask:
    return AbsoluteZeroTask(
        task_id=task_id,
        task_type=task_type,
        task_tokens=task_tokens if task_tokens is not None else [1, 2, 3, 4],
        answer_tokens=answer_tokens if answer_tokens is not None else [10, 11],
    )


# ---------------------------------------------------------------------------
# Test 1 — config defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = AbsoluteZeroConfig()
    assert cfg.task_types == ["deduction", "abduction", "induction"]
    assert cfg.reward_correct == 1.0
    assert cfg.reward_incorrect == 0.0
    assert cfg.leakage_penalty == -0.5
    assert cfg.temperature_propose == 0.9
    assert cfg.temperature_solve == 0.7


# ---------------------------------------------------------------------------
# Test 2 — propose_tasks count
# ---------------------------------------------------------------------------

def test_propose_tasks_count():
    cfg = AbsoluteZeroConfig(n_propose_candidates=2)
    trainer = AbsoluteZeroTrainer(cfg)
    tasks = trainer.propose_tasks(_make_propose_fn())
    # 3 task types × 2 candidates = 6
    assert len(tasks) == 6


# ---------------------------------------------------------------------------
# Test 3 — propose_tasks sequential IDs
# ---------------------------------------------------------------------------

def test_propose_tasks_ids():
    trainer = AbsoluteZeroTrainer(AbsoluteZeroConfig(n_propose_candidates=2))
    tasks = trainer.propose_tasks(_make_propose_fn())
    ids = [t.task_id for t in tasks]
    assert ids == list(range(len(tasks)))


# ---------------------------------------------------------------------------
# Test 4 — each task_type is represented
# ---------------------------------------------------------------------------

def test_propose_tasks_types():
    trainer = AbsoluteZeroTrainer(AbsoluteZeroConfig(n_propose_candidates=1))
    tasks = trainer.propose_tasks(_make_propose_fn())
    types_found = {t.task_type for t in tasks}
    assert types_found == {"deduction", "abduction", "induction"}


# ---------------------------------------------------------------------------
# Test 5 — solve_tasks: correct solution
# ---------------------------------------------------------------------------

def test_solve_correct():
    trainer = AbsoluteZeroTrainer()
    task = _make_task(task_tokens=[1, 2, 3], answer_tokens=[7, 8])
    solve_fn = lambda toks: [7, 8]  # always returns the right answer
    rollouts = trainer.solve_tasks([task], solve_fn)
    assert len(rollouts) == 1
    r = rollouts[0]
    assert r.is_correct is True
    assert r.reward == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test 6 — solve_tasks: incorrect solution
# ---------------------------------------------------------------------------

def test_solve_incorrect():
    trainer = AbsoluteZeroTrainer()
    task = _make_task(task_tokens=[1, 2, 3], answer_tokens=[7, 8])
    solve_fn = lambda toks: [99, 100]  # always wrong
    rollouts = trainer.solve_tasks([task], solve_fn)
    r = rollouts[0]
    assert r.is_correct is False
    assert r.reward == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 7 — detect_leakage: True when answer is sub-sequence of task_tokens
# ---------------------------------------------------------------------------

def test_detect_leakage_true():
    trainer = AbsoluteZeroTrainer()
    task = _make_task(task_tokens=[1, 2, 3, 4, 5], answer_tokens=[3, 4])
    assert trainer.detect_leakage(task) is True


# ---------------------------------------------------------------------------
# Test 8 — detect_leakage: False when answer is not in task_tokens
# ---------------------------------------------------------------------------

def test_detect_leakage_false():
    trainer = AbsoluteZeroTrainer()
    task = _make_task(task_tokens=[1, 2, 3, 4, 5], answer_tokens=[10, 11])
    assert trainer.detect_leakage(task) is False


# ---------------------------------------------------------------------------
# Test 9 — apply_leakage_penalty
# ---------------------------------------------------------------------------

def test_apply_leakage_penalty():
    cfg = AbsoluteZeroConfig(leakage_penalty=-0.5)
    trainer = AbsoluteZeroTrainer(cfg)

    # Leaked task (answer [3,4] is inside task_tokens)
    leaked_task = _make_task(task_tokens=[1, 2, 3, 4, 5], answer_tokens=[3, 4])
    # Clean task
    clean_task = _make_task(task_tokens=[1, 2, 3, 4, 5], answer_tokens=[10, 11], task_id=1)

    rollouts = [
        AbsoluteZeroRollout(task=leaked_task, solution_tokens=[3, 4], is_correct=True, reward=1.0),
        AbsoluteZeroRollout(task=clean_task, solution_tokens=[10, 11], is_correct=True, reward=1.0),
    ]
    updated = trainer.apply_leakage_penalty(rollouts)

    assert updated[0].leakage_detected is True
    assert updated[0].reward == pytest.approx(1.0 + (-0.5))
    assert updated[1].leakage_detected is False
    assert updated[1].reward == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test 10 — compute_policy_gradient: returns scalar tensor
# ---------------------------------------------------------------------------

def test_compute_pg_shape():
    trainer = AbsoluteZeroTrainer()
    task = _make_task()
    rollouts = [
        AbsoluteZeroRollout(task=task, solution_tokens=[], is_correct=True, reward=1.0),
        AbsoluteZeroRollout(task=task, solution_tokens=[], is_correct=False, reward=0.0),
    ]
    log_probs = torch.tensor([-0.5, -1.0])
    loss = trainer.compute_policy_gradient(rollouts, log_probs)
    assert loss.shape == torch.Size([])  # scalar


# ---------------------------------------------------------------------------
# Test 11 — compute_policy_gradient: positive rewards → positive gradient signal
# ---------------------------------------------------------------------------

def test_compute_pg_correct_lower_loss():
    """REINFORCE: loss = -mean(reward * log_prob).

    With log_probs < 0 and reward=1.0 the loss is positive (−(1.0 × negative) > 0).
    With reward=0.0 the loss is exactly 0.
    Minimising the positive loss pushes log_probs toward 0 (higher probability),
    which is the correct gradient signal for rewarded actions.

    This test verifies that:
      - correct rollouts (reward=1.0) → loss > 0 (gradient signal exists).
      - wrong rollouts  (reward=0.0) → loss == 0 (no gradient signal).
    """
    trainer = AbsoluteZeroTrainer()
    task = _make_task()

    log_probs = torch.tensor([-0.3, -0.3])

    # All correct (reward=1.0) → loss = -mean(1.0 * -0.3) = 0.3 > 0
    rollouts_correct = [
        AbsoluteZeroRollout(task=task, solution_tokens=[], is_correct=True, reward=1.0),
        AbsoluteZeroRollout(task=task, solution_tokens=[], is_correct=True, reward=1.0),
    ]
    loss_correct = trainer.compute_policy_gradient(rollouts_correct, log_probs)

    # All incorrect (reward=0.0) → loss = 0
    rollouts_wrong = [
        AbsoluteZeroRollout(task=task, solution_tokens=[], is_correct=False, reward=0.0),
        AbsoluteZeroRollout(task=task, solution_tokens=[], is_correct=False, reward=0.0),
    ]
    loss_wrong = trainer.compute_policy_gradient(rollouts_wrong, log_probs)

    # Correct rollouts produce positive loss (gradient exists); wrong produce zero
    assert loss_correct.item() > 0.0
    assert loss_wrong.item() == pytest.approx(0.0)
    # And the correct-rollout loss is strictly greater (more gradient signal)
    assert loss_correct.item() > loss_wrong.item()


# ---------------------------------------------------------------------------
# Test 12 — statistics: expected keys present
# ---------------------------------------------------------------------------

def test_statistics_keys():
    trainer = AbsoluteZeroTrainer()
    task = _make_task()
    rollouts = [
        AbsoluteZeroRollout(task=task, solution_tokens=[], is_correct=True, reward=1.0),
    ]
    stats = trainer.statistics(rollouts)
    assert "accuracy" in stats
    assert "leakage_rate" in stats
    assert "mean_reward" in stats
    assert "by_type" in stats


# ---------------------------------------------------------------------------
# Test 13 — statistics: all correct → accuracy == 1.0
# ---------------------------------------------------------------------------

def test_statistics_accuracy():
    trainer = AbsoluteZeroTrainer()
    task = _make_task()
    rollouts = [
        AbsoluteZeroRollout(task=task, solution_tokens=[], is_correct=True, reward=1.0),
        AbsoluteZeroRollout(task=task, solution_tokens=[], is_correct=True, reward=1.0),
    ]
    stats = trainer.statistics(rollouts)
    assert stats["accuracy"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test 14 — statistics: by_type breakdown
# ---------------------------------------------------------------------------

def test_statistics_by_type():
    trainer = AbsoluteZeroTrainer()
    tasks = [
        _make_task(task_type="deduction", task_id=0),
        _make_task(task_type="abduction", task_id=1),
        _make_task(task_type="induction", task_id=2),
    ]
    rollouts = [
        AbsoluteZeroRollout(task=tasks[0], solution_tokens=[], is_correct=True, reward=1.0),
        AbsoluteZeroRollout(task=tasks[1], solution_tokens=[], is_correct=False, reward=0.0),
        AbsoluteZeroRollout(task=tasks[2], solution_tokens=[], is_correct=True, reward=1.0),
    ]
    stats = trainer.statistics(rollouts)
    bt = stats["by_type"]
    # All three types present
    assert set(bt.keys()) == {"deduction", "abduction", "induction"}
    # Each has "accuracy" and "n"
    for v in bt.values():
        assert "accuracy" in v
        assert "n" in v
    # Spot-check values
    assert bt["deduction"]["accuracy"] == pytest.approx(1.0)
    assert bt["abduction"]["accuracy"] == pytest.approx(0.0)
    assert bt["induction"]["n"] == 1


# ---------------------------------------------------------------------------
# Test 15 — registry
# ---------------------------------------------------------------------------

def test_registry():
    assert "absolute_zero" in ALIGNMENT_REGISTRY
    assert ALIGNMENT_REGISTRY["absolute_zero"] is AbsoluteZeroTrainer
