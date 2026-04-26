"""Tests for Chain-of-Thought Consistency training module.

Tiny model: nn.Embedding(16, 8) + nn.Linear(8, 16).
n_chains=3, max_reasoning=4, max_answer=2, seq_len=4.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.training.cot_consistency import (
    AnswerExtractor,
    ChainOfThoughtSampler,
    ConsistencyReward,
    CoTConsistencyLoss,
    STaRTrainer,
)

# ---------------------------------------------------------------------------
# Tiny model fixture
# ---------------------------------------------------------------------------

VOCAB_SIZE = 16
D_MODEL = 8


class TinyLM(nn.Module):
    """Minimal language model: embed -> linear -> logits."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.proj = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T)  ->  (B, T, V)
        return self.proj(self.embed(x))


@pytest.fixture()
def tiny_model() -> TinyLM:
    torch.manual_seed(0)
    return TinyLM()


@pytest.fixture()
def sampler(tiny_model: TinyLM) -> ChainOfThoughtSampler:
    return ChainOfThoughtSampler(tiny_model, n_chains=3, temperature=0.8)


@pytest.fixture()
def input_ids() -> torch.Tensor:
    return torch.tensor([1, 3, 5, 7], dtype=torch.long)


# ---------------------------------------------------------------------------
# ChainOfThoughtSampler tests
# ---------------------------------------------------------------------------


def test_sampler_returns_n_chains_chains(sampler, input_ids):
    """ChainOfThoughtSampler.sample_chains returns exactly n_chains chains."""
    chains, log_probs = sampler.sample_chains(
        input_ids, max_reasoning_tokens=4, max_answer_tokens=2
    )
    assert len(chains) == 3
    assert len(log_probs) == 3


def test_sampler_returns_n_chains_log_probs(sampler, input_ids):
    """ChainOfThoughtSampler.sample_chains returns n_chains log_probs."""
    chains, log_probs = sampler.sample_chains(
        input_ids, max_reasoning_tokens=4, max_answer_tokens=2
    )
    assert len(log_probs) == 3


def test_sampler_tokens_in_valid_vocab_range(sampler, input_ids):
    """All generated tokens must be within [0, VOCAB_SIZE)."""
    chains, _ = sampler.sample_chains(input_ids, max_reasoning_tokens=4, max_answer_tokens=2)
    for chain in chains:
        assert chain.numel() > 0, "chain should not be empty"
        assert int(chain.min()) >= 0
        assert int(chain.max()) < VOCAB_SIZE


def test_sampler_log_probs_are_negative(sampler, input_ids):
    """Per-chain sum log-probs should be negative (log of probability < 1)."""
    _, log_probs = sampler.sample_chains(input_ids, max_reasoning_tokens=4, max_answer_tokens=2)
    for lp in log_probs:
        assert lp < 0.0, f"log_prob should be negative, got {lp}"


# ---------------------------------------------------------------------------
# AnswerExtractor tests
# ---------------------------------------------------------------------------


def test_answer_extractor_split_concat_equals_original():
    """reasoning + [sep] + answer concatenated = chain (when sep present)."""
    extractor = AnswerExtractor(answer_separator_id=2)
    # chain: tokens before sep + sep + tokens after sep
    chain = torch.tensor([5, 6, 2, 7, 8], dtype=torch.long)
    reasoning, answer = extractor.extract(chain)
    # reasoning should be [5, 6], answer should be [7, 8]
    assert reasoning.tolist() == [5, 6]
    assert answer.tolist() == [7, 8]


def test_answer_extractor_no_separator_fallback():
    """When sep not found: reasoning = chain[:-1], answer = chain[-1:]."""
    extractor = AnswerExtractor(answer_separator_id=2)
    chain = torch.tensor([5, 6, 7, 8], dtype=torch.long)  # no sep=2
    reasoning, answer = extractor.extract(chain)
    assert reasoning.tolist() == [5, 6, 7]
    assert answer.tolist() == [8]


def test_majority_vote_returns_most_common():
    """majority_vote should return the token sequence appearing most often."""
    extractor = AnswerExtractor()
    a = torch.tensor([1, 2])
    b = torch.tensor([1, 2])
    c = torch.tensor([3, 4])
    result = extractor.majority_vote([a, b, c])
    assert result.tolist() == [1, 2]


def test_majority_vote_tie_returns_one_of_tied():
    """When tied, majority_vote returns one of the tied answer sequences."""
    extractor = AnswerExtractor()
    a = torch.tensor([1])
    b = torch.tensor([2])
    result = extractor.majority_vote([a, b])
    assert result.tolist() in ([1], [2])


# ---------------------------------------------------------------------------
# ConsistencyReward tests
# ---------------------------------------------------------------------------


def test_consistency_reward_returns_n_chains_rewards():
    """ConsistencyReward.compute returns a list of length n_chains."""
    reward_fn = ConsistencyReward()
    chains = [
        torch.tensor([1, 2, 3]),
        torch.tensor([1, 2, 4]),
        torch.tensor([1, 2, 3]),
    ]
    rewards = reward_fn.compute(chains, [0.0] * 3)
    assert len(rewards) == 3


def test_consistency_reward_agreeing_higher_than_disagreeing():
    """Chains agreeing with majority get higher reward than the lone dissenter."""
    # Force separator to a token not in chains so last token = answer
    reward_fn = ConsistencyReward(length_penalty=0.0)  # disable length for clarity
    # majority answer = last token = 9 (appears twice)
    chains = [
        torch.tensor([5, 6, 9]),  # answer = 9 (majority)
        torch.tensor([5, 6, 9]),  # answer = 9 (majority)
        torch.tensor([5, 6, 7]),  # answer = 7 (minority)
    ]
    rewards = reward_fn.compute(chains, [0.0] * 3)
    # Majority chains (idx 0,1) reward > minority chain (idx 2) reward
    assert rewards[0] > rewards[2]
    assert rewards[1] > rewards[2]


def test_consistency_reward_all_agree_gives_all_ones():
    """When all chains agree, every reward should be 1.0."""
    reward_fn = ConsistencyReward()
    chains = [
        torch.tensor([1, 2, 9]),
        torch.tensor([3, 4, 9]),
        torch.tensor([5, 6, 9]),
    ]
    rewards = reward_fn.compute(chains, [0.0] * 3)
    assert rewards == [1.0, 1.0, 1.0]


# ---------------------------------------------------------------------------
# CoTConsistencyLoss tests
# ---------------------------------------------------------------------------


def test_cot_loss_is_scalar_finite_and_grad_flows():
    """CoTConsistencyLoss returns a finite scalar with gradients flowing."""
    loss_fn = CoTConsistencyLoss(baseline="mean")
    log_probs = torch.tensor([-1.0, -2.0, -3.0], requires_grad=True)
    rewards = torch.tensor([1.0, 0.0, 1.0])
    loss = loss_fn(log_probs, rewards)

    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)

    loss.backward()
    assert log_probs.grad is not None
    assert torch.isfinite(log_probs.grad).all()


def test_cot_loss_baseline_none_equals_raw_reinforce():
    """With baseline='none', loss = -mean(log_probs * rewards)."""
    loss_fn = CoTConsistencyLoss(baseline="none")
    log_probs = torch.tensor([-1.0, -2.0, -3.0], requires_grad=True)
    rewards = torch.tensor([1.0, 0.5, 0.0])
    loss = loss_fn(log_probs, rewards)
    expected = -(log_probs.detach() * rewards).mean()
    assert torch.isclose(loss, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# STaRTrainer tests
# ---------------------------------------------------------------------------


def test_star_rationalize_step_returns_valid_fields(tiny_model, input_ids):
    """rationalize_step returns dict with n_correct in [0, n_chains] and loss scalar."""
    opt = torch.optim.SGD(tiny_model.parameters(), lr=1e-3)
    trainer = STaRTrainer(tiny_model, opt, n_chains=3)
    correct_answer = torch.tensor([5], dtype=torch.long)
    result = trainer.rationalize_step(input_ids, correct_answer)

    assert "n_correct" in result and "n_total" in result and "loss" in result
    assert 0 <= result["n_correct"] <= 3
    assert result["n_total"] == 3
    # loss is either 0 (int) or a scalar tensor
    if isinstance(result["loss"], torch.Tensor):
        assert result["loss"].shape == torch.Size([1]) or result["loss"].ndim == 0


def test_star_rationalize_step_no_correct_chains_loss_zero(tiny_model, input_ids):
    """When no chain produces the correct answer, rationalize_step returns loss=0."""
    opt = torch.optim.SGD(tiny_model.parameters(), lr=1e-3)
    trainer = STaRTrainer(tiny_model, opt, n_chains=3)

    # Use an impossible answer (length 3, values that never occur at end of chain)
    # by patching the extractor to never match
    correct_answer = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long)

    # Run multiple times; statistically very unlikely to match
    for _ in range(5):
        result = trainer.rationalize_step(input_ids, correct_answer)
        if result["n_correct"] == 0:
            assert result["loss"] == 0
            return
    # If by chance some matched, just verify n_correct in valid range
    assert result["n_correct"] <= 3


def test_star_consistency_step_mean_reward_in_range(tiny_model, input_ids):
    """consistency_step returns mean_reward in [0, 1] and a scalar loss."""
    opt = torch.optim.SGD(tiny_model.parameters(), lr=1e-3)
    trainer = STaRTrainer(tiny_model, opt, n_chains=3)
    result = trainer.consistency_step(input_ids)

    # mean_reward is always 1.0 when all agree, or mixture otherwise
    assert 0.0 <= result["mean_reward"] <= 1.0 + 1e-6  # allow floating point
    assert isinstance(result["consistency_loss"], torch.Tensor)
    assert result["consistency_loss"].ndim == 0


def test_star_full_cycle_runs_without_error(tiny_model, input_ids):
    """Running both rationalize_step and consistency_step completes without error."""
    opt = torch.optim.SGD(tiny_model.parameters(), lr=1e-3)
    trainer = STaRTrainer(tiny_model, opt, n_chains=3)
    correct_answer = torch.tensor([7], dtype=torch.long)

    r_result = trainer.rationalize_step(input_ids, correct_answer)
    c_result = trainer.consistency_step(input_ids)

    assert "n_correct" in r_result
    assert "consistency_loss" in c_result
