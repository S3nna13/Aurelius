"""Tests for the Self-Refine iterative refinement implementation."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.alignment.self_refine import (
    SelfRefineConfig,
    SelfRefineStep,
    SelfRefineTrainer,
    compute_refinement_gain,
)


# ---------------------------------------------------------------------------
# MockModel (as specified)
# ---------------------------------------------------------------------------

class MockModel(nn.Module):
    def __init__(self, vocab_size: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, 16)
        self.proj = nn.Linear(16, vocab_size)

    def forward(self, input_ids, **kwargs):
        x = self.embed(input_ids).mean(1)
        logits = self.proj(x)
        return (None, logits.unsqueeze(1).expand(-1, input_ids.shape[1], -1), None)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

VOCAB = 64


def _ids(n: int, start: int = 1) -> torch.Tensor:
    """1-D LongTensor of n sequential token ids, clamped to VOCAB."""
    return torch.arange(start, start + n, dtype=torch.long) % VOCAB


def _make_reward_fn(base: float = 0.5, delta: float = 0.0):
    """Return a simple deterministic reward function.

    If delta is non-zero the function increases its return value on each call,
    simulating improvement.
    """
    state = {"calls": 0}

    def reward_fn(ids: torch.Tensor) -> float:
        r = base + state["calls"] * delta
        state["calls"] += 1
        return r

    return reward_fn


@pytest.fixture
def model():
    torch.manual_seed(42)
    return MockModel(vocab_size=VOCAB)


@pytest.fixture
def trainer(model):
    return SelfRefineTrainer(
        model=model,
        reward_fn=_make_reward_fn(base=0.5),
        n_refine_steps=3,
        temperature=0.7,
        stop_if_improved=True,
    )


def _make_trainer(model, reward_fn, n_refine_steps=3, stop_if_improved=True):
    return SelfRefineTrainer(
        model=model,
        reward_fn=reward_fn,
        n_refine_steps=n_refine_steps,
        temperature=0.7,
        stop_if_improved=stop_if_improved,
    )


# ---------------------------------------------------------------------------
# Test 1: SelfRefineConfig defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    """SelfRefineConfig should have the correct default values."""
    cfg = SelfRefineConfig()
    assert cfg.n_refine_steps == 3
    assert cfg.temperature == 0.7
    assert cfg.stop_if_improved is True
    assert cfg.max_new_tokens == 64
    assert cfg.min_improvement == 0.0


# ---------------------------------------------------------------------------
# Test 2: generate_with_ids returns tensor of correct length
# ---------------------------------------------------------------------------

def test_generate_with_ids_length(trainer):
    """generate_with_ids should return exactly max_new tokens."""
    prompt_ids = _ids(4, start=1)
    max_new = 8
    out = trainer.generate_with_ids(prompt_ids, max_new=max_new, temperature=1.0)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (max_new,), f"Expected {max_new} tokens, got {out.shape}"
    assert out.dtype == torch.long


# ---------------------------------------------------------------------------
# Test 3: SelfRefineStep has all required fields
# ---------------------------------------------------------------------------

def test_self_refine_step_fields():
    """SelfRefineStep dataclass should have all required attributes."""
    step = SelfRefineStep(
        initial_ids=torch.tensor([1, 2, 3]),
        critique_ids=torch.tensor([4, 5]),
        refined_ids=torch.tensor([6, 7, 8]),
        reward_before=0.3,
        reward_after=0.7,
        improvement=0.4,
    )
    assert hasattr(step, "initial_ids")
    assert hasattr(step, "critique_ids")
    assert hasattr(step, "refined_ids")
    assert hasattr(step, "reward_before")
    assert hasattr(step, "reward_after")
    assert hasattr(step, "improvement")


# ---------------------------------------------------------------------------
# Test 4: run_refinement_loop returns list of SelfRefineStep
# ---------------------------------------------------------------------------

def test_run_refinement_loop_returns_steps(model):
    """run_refinement_loop should return a list of SelfRefineStep objects."""
    trainer = _make_trainer(model, _make_reward_fn(0.5), n_refine_steps=3,
                            stop_if_improved=False)
    steps = trainer.run_refinement_loop(
        prompt_ids=_ids(4, start=1),
        initial_response_ids=_ids(6, start=5),
        critique_prompt_ids=_ids(3, start=20),
        refine_prompt_ids=_ids(3, start=30),
    )
    assert isinstance(steps, list)
    assert len(steps) > 0
    for step in steps:
        assert isinstance(step, SelfRefineStep)


# ---------------------------------------------------------------------------
# Test 5: Number of steps <= n_refine_steps
# ---------------------------------------------------------------------------

def test_run_refinement_loop_max_steps(model):
    """The loop must never exceed n_refine_steps iterations."""
    n = 3
    trainer = _make_trainer(model, _make_reward_fn(0.5), n_refine_steps=n,
                            stop_if_improved=False)
    steps = trainer.run_refinement_loop(
        prompt_ids=_ids(4),
        initial_response_ids=_ids(6),
        critique_prompt_ids=_ids(3, start=20),
        refine_prompt_ids=_ids(3, start=30),
    )
    assert len(steps) <= n, f"Expected at most {n} steps, got {len(steps)}"


# ---------------------------------------------------------------------------
# Test 6: stop_if_improved=False always runs all n_refine_steps
# ---------------------------------------------------------------------------

def test_stop_if_improved_false_runs_all_steps(model):
    """With stop_if_improved=False the loop must always run n_refine_steps."""
    n = 3
    # Reward increases each call to simulate consistent improvement
    trainer = _make_trainer(
        model,
        _make_reward_fn(base=0.0, delta=0.2),
        n_refine_steps=n,
        stop_if_improved=False,
    )
    steps = trainer.run_refinement_loop(
        prompt_ids=_ids(4),
        initial_response_ids=_ids(6),
        critique_prompt_ids=_ids(3, start=20),
        refine_prompt_ids=_ids(3, start=30),
    )
    assert len(steps) == n, (
        f"Expected exactly {n} steps with stop_if_improved=False, got {len(steps)}"
    )


# ---------------------------------------------------------------------------
# Test 7: create_training_pairs filters steps with improvement <= 0
# ---------------------------------------------------------------------------

def test_create_training_pairs_filters_non_improvements(trainer):
    """create_training_pairs must only include steps where improvement > 0."""
    steps = [
        SelfRefineStep(
            initial_ids=_ids(4), critique_ids=_ids(3), refined_ids=_ids(4),
            reward_before=0.5, reward_after=0.8, improvement=0.3,
        ),
        SelfRefineStep(
            initial_ids=_ids(4), critique_ids=_ids(3), refined_ids=_ids(4),
            reward_before=0.8, reward_after=0.8, improvement=0.0,
        ),
        SelfRefineStep(
            initial_ids=_ids(4), critique_ids=_ids(3), refined_ids=_ids(4),
            reward_before=0.8, reward_after=0.6, improvement=-0.2,
        ),
    ]
    pairs = trainer.create_training_pairs(steps)
    assert len(pairs) == 1, f"Expected 1 pair, got {len(pairs)}"
    assert pairs[0]["improvement"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Test 8: compute_refinement_gain success_rate in [0, 1]
# ---------------------------------------------------------------------------

def test_compute_refinement_gain_success_rate_range():
    """success_rate must always be in [0, 1]."""
    steps = [
        SelfRefineStep(_ids(4), _ids(3), _ids(4), 0.3, 0.7, 0.4),
        SelfRefineStep(_ids(4), _ids(3), _ids(4), 0.7, 0.5, -0.2),
        SelfRefineStep(_ids(4), _ids(3), _ids(4), 0.5, 0.5, 0.0),
    ]
    result = compute_refinement_gain(steps)
    assert 0.0 <= result["success_rate"] <= 1.0, (
        f"success_rate out of range: {result['success_rate']}"
    )


# ---------------------------------------------------------------------------
# Test 9: compute_refinement_gain mean_improvement correct
# ---------------------------------------------------------------------------

def test_compute_refinement_gain_mean_improvement():
    """mean_improvement should equal the arithmetic mean of step improvements."""
    improvements = [0.4, -0.2, 0.0, 0.6]
    steps = [
        SelfRefineStep(_ids(4), _ids(3), _ids(4), 0.0, 0.0 + v, v)
        for v in improvements
    ]
    result = compute_refinement_gain(steps)
    expected_mean = sum(improvements) / len(improvements)
    assert result["mean_improvement"] == pytest.approx(expected_mean, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 10: generate_critique and refine return non-empty tensors
# ---------------------------------------------------------------------------

def test_generate_critique_and_refine_nonempty(trainer):
    """generate_critique and refine must both return non-empty 1-D tensors."""
    prompt_ids = _ids(4, start=1)
    response_ids = _ids(6, start=5)
    critique_prompt_ids = _ids(3, start=20)
    refine_prompt_ids = _ids(3, start=30)

    critique_ids = trainer.generate_critique(
        prompt_ids, response_ids, critique_prompt_ids
    )
    assert critique_ids.ndim == 1
    assert critique_ids.numel() > 0, "critique_ids must be non-empty"

    refined_ids = trainer.refine(
        prompt_ids, response_ids, critique_ids, refine_prompt_ids
    )
    assert refined_ids.ndim == 1
    assert refined_ids.numel() > 0, "refined_ids must be non-empty"


# ---------------------------------------------------------------------------
# Test 11: SelfRefineStep.improvement == reward_after - reward_before
# ---------------------------------------------------------------------------

def test_self_refine_step_improvement_equals_delta(model):
    """Each SelfRefineStep.improvement must equal reward_after - reward_before."""
    # Reward: returns increasing values 0.2, 0.4, 0.6, 0.8, ...
    trainer = _make_trainer(
        model,
        _make_reward_fn(base=0.2, delta=0.2),
        n_refine_steps=3,
        stop_if_improved=False,
    )
    steps = trainer.run_refinement_loop(
        prompt_ids=_ids(4),
        initial_response_ids=_ids(6),
        critique_prompt_ids=_ids(3, start=20),
        refine_prompt_ids=_ids(3, start=30),
    )
    for step in steps:
        expected = step.reward_after - step.reward_before
        assert step.improvement == pytest.approx(expected, abs=1e-6), (
            f"improvement {step.improvement} != reward_after - reward_before "
            f"({step.reward_after} - {step.reward_before} = {expected})"
        )
