"""Tests for Chain-of-Hindsight (CoH) training implementation."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.alignment.chain_of_hindsight import (
    CoHConfig,
    CoHTrainer,
    create_hindsight_dataset,
    rank_responses,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

class TinyLM(nn.Module):
    """Minimal language model stub for testing.

    Returns (None, logits, None) to match Aurelius model convention.
    """

    def __init__(self, vocab_size: int = 256, d_model: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor):
        x = self.embed(input_ids)        # (B, seq_len, d_model)
        logits = self.proj(x)            # (B, seq_len, vocab_size)
        return None, logits, None


def _ids(n: int, start: int = 10) -> torch.Tensor:
    """Create a 1-D LongTensor of sequential token ids for testing."""
    return torch.arange(start, start + n, dtype=torch.long)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return TinyLM(vocab_size=256, d_model=32)


@pytest.fixture
def trainer(model):
    return CoHTrainer(model=model, tokenizer_pad_id=0, coh_weight=1.0)


# ---------------------------------------------------------------------------
# Test 1: CoHConfig defaults
# ---------------------------------------------------------------------------

def test_cohconfig_defaults():
    """CoHConfig should have the specified default values."""
    cfg = CoHConfig()
    assert cfg.coh_weight == 1.0
    assert cfg.feedback_type == "scalar"
    assert cfg.max_bad_response_len == 256
    assert cfg.max_good_response_len == 512
    assert cfg.min_reward_gap == 0.1


# ---------------------------------------------------------------------------
# Test 2: build_coh_sequence correct shape
# ---------------------------------------------------------------------------

def test_build_coh_sequence_shape(trainer):
    """build_coh_sequence should concatenate all parts into one sequence."""
    prompt_ids = _ids(5, start=1)
    bad_ids = _ids(8, start=10)
    good_ids = _ids(6, start=20)

    input_ids, labels = trainer.build_coh_sequence(prompt_ids, bad_ids, good_ids)

    # Without explicit feedback_ids, one separator token is inserted
    expected_len = 5 + 8 + 1 + 6
    assert input_ids.shape == (expected_len,), (
        f"Expected length {expected_len}, got {input_ids.shape}"
    )
    assert labels.shape == input_ids.shape


def test_build_coh_sequence_with_feedback(trainer):
    """build_coh_sequence should use provided feedback_ids."""
    prompt_ids = _ids(4, start=1)
    bad_ids = _ids(3, start=10)
    good_ids = _ids(5, start=20)
    feedback_ids = torch.tensor([100, 101, 102], dtype=torch.long)

    input_ids, labels = trainer.build_coh_sequence(
        prompt_ids, bad_ids, good_ids, feedback_ids=feedback_ids
    )

    expected_len = 4 + 3 + 3 + 5
    assert input_ids.shape == (expected_len,)


# ---------------------------------------------------------------------------
# Test 3: Labels mask only allows loss on good_response portion
# ---------------------------------------------------------------------------

def test_labels_mask_good_response_only(trainer):
    """Labels should be -100 everywhere except the good_response region."""
    prompt_ids = _ids(4, start=1)   # positions 0-3
    bad_ids = _ids(3, start=10)     # positions 4-6
    # With default separator: position 7
    good_ids = _ids(5, start=20)    # positions 8-12

    input_ids, labels = trainer.build_coh_sequence(prompt_ids, bad_ids, good_ids)

    # Everything before good_response should be masked
    prefix_len = len(prompt_ids) + len(bad_ids) + 1  # +1 for separator
    assert (labels[:prefix_len] == -100).all(), (
        "Labels in prompt+bad_response+feedback should all be -100"
    )

    # good_response portion should equal good_ids
    assert (labels[prefix_len:] == good_ids).all(), (
        "Labels in good_response region should match good_ids"
    )


# ---------------------------------------------------------------------------
# Test 4: compute_coh_loss returns scalar tensor
# ---------------------------------------------------------------------------

def test_compute_coh_loss_scalar(trainer):
    """compute_coh_loss should return a 0-dimensional tensor."""
    vocab_size = 256
    seq_len = 12
    batch = 2
    logits = torch.randn(batch, seq_len, vocab_size)
    labels = torch.full((batch, seq_len), -100, dtype=torch.long)
    # Mark last 4 tokens as valid
    labels[:, -4:] = torch.randint(0, vocab_size, (batch, 4))

    loss = trainer.compute_coh_loss(logits, labels)
    assert loss.dim() == 0, "Loss should be a scalar (0-dim) tensor"
    assert loss.item() > 0, "Loss should be positive"


# ---------------------------------------------------------------------------
# Test 5: build_hindsight_feedback high score → "good" message
# ---------------------------------------------------------------------------

def test_build_hindsight_feedback_high(trainer):
    """Scores >= threshold should return the 'good' feedback string."""
    fb = trainer.build_hindsight_feedback(reward_score=0.8, threshold=0.5)
    assert isinstance(fb, str)
    assert "good" in fb.lower()


# ---------------------------------------------------------------------------
# Test 6: build_hindsight_feedback low score → "improved" message
# ---------------------------------------------------------------------------

def test_build_hindsight_feedback_low(trainer):
    """Scores < threshold should return the 'improved' feedback string."""
    fb = trainer.build_hindsight_feedback(reward_score=0.3, threshold=0.5)
    assert isinstance(fb, str)
    assert "improved" in fb.lower()
    # The two messages must be different
    fb_high = trainer.build_hindsight_feedback(reward_score=0.8, threshold=0.5)
    assert fb != fb_high


# ---------------------------------------------------------------------------
# Test 7: rank_responses orders by reward ascending
# ---------------------------------------------------------------------------

def test_rank_responses_ascending():
    """rank_responses should return pairs sorted by reward ascending."""
    responses = ["c", "a", "b"]
    rewards = [0.9, 0.1, 0.5]
    ranked = rank_responses(responses, rewards)
    assert len(ranked) == 3
    # Scores should be ascending
    scores = [r for _, r in ranked]
    assert scores == sorted(scores), f"Expected ascending order, got {scores}"
    # Worst first
    assert ranked[0] == ("a", 0.1)
    assert ranked[-1] == ("c", 0.9)


# ---------------------------------------------------------------------------
# Test 8: train_step returns dict with required keys
# ---------------------------------------------------------------------------

def test_train_step_keys(trainer):
    """train_step should return a dict containing loss, n_tokens, mean_reward_gap."""
    torch.manual_seed(1)
    prompt_ids = _ids(4, start=1)
    responses = [_ids(6, start=10), _ids(6, start=20)]
    rewards = [0.2, 0.8]

    result = trainer.train_step(prompt_ids, responses, rewards)

    assert "loss" in result, "Missing 'loss' key"
    assert "n_tokens" in result, "Missing 'n_tokens' key"
    assert "mean_reward_gap" in result, "Missing 'mean_reward_gap' key"


def test_train_step_reward_gap(trainer):
    """mean_reward_gap should equal best_reward - worst_reward."""
    prompt_ids = _ids(4, start=1)
    responses = [_ids(5, start=10), _ids(5, start=20), _ids(5, start=30)]
    rewards = [0.1, 0.5, 0.9]

    result = trainer.train_step(prompt_ids, responses, rewards)
    assert abs(result["mean_reward_gap"] - 0.8) < 1e-5


# ---------------------------------------------------------------------------
# Test 9: create_hindsight_dataset filters pairs below min_reward_gap
# ---------------------------------------------------------------------------

def test_create_hindsight_dataset_filters_small_gap():
    """Examples with reward_gap < min_reward_gap should be excluded."""
    config = CoHConfig(min_reward_gap=0.5)

    prompts = ["Prompt A", "Prompt B"]
    responses_list = [
        ["bad A", "good A"],  # gap = 0.8 — keep
        ["bad B", "good B"],  # gap = 0.3 — filter out
    ]
    rewards_list = [
        [0.1, 0.9],   # gap = 0.8
        [0.6, 0.9],   # gap = 0.3
    ]

    examples = create_hindsight_dataset(prompts, responses_list, rewards_list, config)
    assert len(examples) == 1, (
        f"Expected 1 example after filtering, got {len(examples)}"
    )
    assert examples[0]["reward_gap"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Test 10: create_hindsight_dataset returns correct number of examples
# ---------------------------------------------------------------------------

def test_create_hindsight_dataset_count():
    """All examples above min_reward_gap should be returned."""
    config = CoHConfig(min_reward_gap=0.1)

    prompts = ["P1", "P2", "P3"]
    responses_list = [
        ["r1_bad", "r1_good"],
        ["r2_bad", "r2_good"],
        ["r3_bad", "r3_good"],
    ]
    rewards_list = [
        [0.0, 0.5],   # gap = 0.5 — keep
        [0.2, 0.8],   # gap = 0.6 — keep
        [0.3, 0.35],  # gap = 0.05 — filter (< 0.1)
    ]

    examples = create_hindsight_dataset(prompts, responses_list, rewards_list, config)
    assert len(examples) == 2, (
        f"Expected 2 examples, got {len(examples)}"
    )
    # Verify returned keys
    for ex in examples:
        assert "input_ids" in ex
        assert "labels" in ex
        assert "reward_gap" in ex


# ---------------------------------------------------------------------------
# Test 11: Gradient flows through coh_loss (backward works)
# ---------------------------------------------------------------------------

def test_gradient_flows_through_coh_loss(trainer):
    """Calling .backward() on coh_loss should not error and produce gradients."""
    vocab_size = 256
    seq_len = 10
    logits = torch.randn(1, seq_len, vocab_size, requires_grad=True)
    labels = torch.full((1, seq_len), -100, dtype=torch.long)
    labels[0, -3:] = torch.randint(0, vocab_size, (3,))

    loss = trainer.compute_coh_loss(logits, labels)
    loss.backward()

    assert logits.grad is not None, "Gradient should flow back to logits"
    assert logits.grad.shape == logits.shape


# ---------------------------------------------------------------------------
# Test 12: No training example when all rewards identical (gap = 0)
# ---------------------------------------------------------------------------

def test_no_example_when_gap_zero():
    """When all rewards are identical (gap = 0), no training example should be created."""
    config = CoHConfig(min_reward_gap=0.1)

    prompts = ["Identical prompt"]
    responses_list = [["response A", "response B", "response C"]]
    rewards_list = [[0.5, 0.5, 0.5]]   # gap = 0.0

    examples = create_hindsight_dataset(prompts, responses_list, rewards_list, config)
    assert len(examples) == 0, (
        "No examples should be created when reward gap is 0"
    )
