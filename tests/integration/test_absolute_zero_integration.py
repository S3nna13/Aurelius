"""Integration test for the Absolute Zero self-play RL pipeline.

Exercises the full propose → solve → leakage_detection → compute_pg flow
with mock functions.  The solver is correct 50 % of the time (deterministic
alternation), so the measured accuracy should be exactly 0.5 for an even
number of tasks.
"""

from __future__ import annotations

import pytest
import torch

from src.alignment import ALIGNMENT_REGISTRY
from src.alignment.absolute_zero import (
    AbsoluteZeroConfig,
    AbsoluteZeroTrainer,
)

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _build_propose_fn():
    """Returns a propose_fn that emits non-leaking token sequences.

    task_tokens  = [type_offset, type_offset+1, type_offset+2]
    answer_tokens = [type_offset+10, type_offset+11]
    where type_offset is unique per (task_type, candidate) pair to ensure no
    leakage.
    """
    call_counter = [0]

    def propose_fn(task_type: str, n: int) -> list[list[int]]:
        seqs = []
        for i in range(n):
            base = call_counter[0] * 1000 + i * 10
            # 6 tokens: first 3 → task, last 3 → answer (split by trainer)
            # We design so answer (last 3) doesn't appear inside first 3.
            tokens = [base, base + 1, base + 2, base + 50, base + 51, base + 52]
            seqs.append(tokens)
        call_counter[0] += 1
        return seqs

    return propose_fn


def _build_solve_fn(tasks_ref: list):
    """Alternating solver: even-index tasks get the correct answer, odd get wrong."""
    idx = [0]

    def solve_fn(task_tokens: list[int]) -> list[int]:
        # We resolve which task this is by looking up via the shared list
        current_idx = idx[0]
        idx[0] += 1
        task = tasks_ref[current_idx]
        if current_idx % 2 == 0:
            return list(task.answer_tokens)  # correct
        else:
            return [9999, 9998]  # wrong

    return solve_fn


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


def test_full_pipeline_integration():
    """Full pipeline: propose → solve → leakage → PG."""
    cfg = AbsoluteZeroConfig(
        n_propose_candidates=2,
        leakage_penalty=-0.5,
        reward_correct=1.0,
        reward_incorrect=0.0,
    )
    trainer = AbsoluteZeroTrainer(cfg)

    # --- 1. Propose tasks -------------------------------------------------
    propose_fn = _build_propose_fn()
    tasks = trainer.propose_tasks(propose_fn)

    # 3 types × 2 candidates = 6 tasks
    assert len(tasks) == 6

    # Confirm sequential IDs
    assert [t.task_id for t in tasks] == list(range(6))

    # All three task types present
    task_types = {t.task_type for t in tasks}
    assert task_types == {"deduction", "abduction", "induction"}

    # --- 2. Solve tasks ---------------------------------------------------
    solve_fn = _build_solve_fn(tasks)
    rollouts = trainer.solve_tasks(tasks, solve_fn)

    assert len(rollouts) == 6

    # --- 3. Leakage detection + penalty -----------------------------------
    rollouts = trainer.apply_leakage_penalty(rollouts)

    # None of the mock tasks should leak (by design of _build_propose_fn)
    for r in rollouts:
        assert r.leakage_detected is False

    # --- 4. Statistics ----------------------------------------------------
    stats = trainer.statistics(rollouts)

    # Alternating correct/incorrect → exactly 3/6 correct = 0.5
    assert stats["accuracy"] == pytest.approx(0.5)
    assert stats["leakage_rate"] == pytest.approx(0.0)
    assert "mean_reward" in stats
    assert "by_type" in stats

    for tt, v in stats["by_type"].items():
        assert "accuracy" in v
        assert "n" in v
        assert v["n"] == 2  # 2 candidates per type

    # --- 5. Policy gradient -----------------------------------------------
    log_probs = torch.full((len(rollouts),), -0.5)
    loss = trainer.compute_policy_gradient(rollouts, log_probs)

    # Must be a scalar tensor
    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)

    # --- 6. Registry wired correctly -------------------------------------
    assert ALIGNMENT_REGISTRY["absolute_zero"] is AbsoluteZeroTrainer
    trainer_from_registry = ALIGNMENT_REGISTRY["absolute_zero"]()
    assert isinstance(trainer_from_registry, AbsoluteZeroTrainer)
