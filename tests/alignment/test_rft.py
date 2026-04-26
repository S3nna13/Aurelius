"""Tests for src/alignment/rft.py — Reinforced Fine-Tuning."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.alignment.rft import (
    RFTDataset,
    RFTSampler,
    RFTTrainer,
    compute_reward_stats,
)

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class MockModel(nn.Module):
    """Tiny model that returns random logits — no real transformer needed."""

    VOCAB = 32

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(self.VOCAB, 8)
        self.head = nn.Linear(8, self.VOCAB)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask=None,
        labels: torch.Tensor | None = None,
        past_key_values=None,
    ):
        x = self.embed(input_ids)  # (B, S, 8)
        logits = self.head(x)  # (B, S, VOCAB)

        loss = None
        if labels is not None:
            # Shift labels to align with next-token prediction; ignore -100
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.VOCAB),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return loss, logits, []  # kv = empty list


class MockRewardModel:
    """Deterministic reward based on response length (for testability)."""

    def __call__(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> float:
        return float(response_ids.shape[1] % 10) / 10.0


def _make_sampler(k: int = 4, reward_threshold: float = 0.5, max_new_tokens: int = 5):
    model = MockModel()
    reward = MockRewardModel()
    return RFTSampler(
        model=model,
        reward_model=reward,
        k=k,
        temperature=1.0,
        top_p=0.9,
        reward_threshold=reward_threshold,
        max_new_tokens=max_new_tokens,
    )


def _prompt(length: int = 3) -> torch.Tensor:
    """Return a (1, length) prompt tensor."""
    return torch.randint(0, MockModel.VOCAB, (1, length))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_sample_and_score_count():
    """sample_and_score returns exactly k=4 scored completions."""
    sampler = _make_sampler(k=4)
    scored = sampler.sample_and_score(_prompt())
    assert len(scored) == 4, f"Expected 4, got {len(scored)}"


def test_sample_and_score_sorted():
    """sample_and_score returns completions sorted by reward descending."""
    sampler = _make_sampler(k=6)
    scored = sampler.sample_and_score(_prompt())
    rewards = [d["reward"] for d in scored]
    assert rewards == sorted(rewards, reverse=True), f"Rewards not sorted descending: {rewards}"


def test_filter_completions():
    """filter_completions keeps only completions with reward >= 0.5."""
    sampler = _make_sampler(reward_threshold=0.5)
    # Build a hand-crafted scored list
    scored = [
        {"response_ids": torch.zeros(1, 1, dtype=torch.long), "reward": 0.9},
        {"response_ids": torch.zeros(1, 1, dtype=torch.long), "reward": 0.5},
        {"response_ids": torch.zeros(1, 1, dtype=torch.long), "reward": 0.3},
        {"response_ids": torch.zeros(1, 1, dtype=torch.long), "reward": 0.1},
    ]
    kept = sampler.filter_completions(scored)
    assert len(kept) == 2, f"Expected 2 kept, got {len(kept)}"
    assert all(d["reward"] >= 0.5 for d in kept)


def test_filter_all_rejected():
    """filter_completions returns empty list when all rewards < threshold."""
    sampler = _make_sampler(reward_threshold=0.9)
    scored = [
        {"response_ids": torch.zeros(1, 1, dtype=torch.long), "reward": 0.1},
        {"response_ids": torch.zeros(1, 1, dtype=torch.long), "reward": 0.2},
    ]
    kept = sampler.filter_completions(scored)
    assert kept == [], f"Expected empty list, got {kept}"


def test_rft_dataset_add_examples():
    """After adding accepted completions, len(dataset) == n_accepted."""
    prompt_ids = _prompt(3)
    accepted = [
        {"response_ids": torch.randint(0, MockModel.VOCAB, (1, 4)), "reward": 0.8},
        {"response_ids": torch.randint(0, MockModel.VOCAB, (1, 6)), "reward": 0.6},
    ]
    dataset = RFTDataset()
    dataset.add_examples(prompt_ids, accepted)
    assert len(dataset) == 2, f"Expected 2, got {len(dataset)}"


def test_rft_dataset_labels_prompt_masked():
    """Prompt positions in labels must all be -100."""
    prompt_len = 3
    response_len = 5
    prompt_ids = _prompt(prompt_len)
    accepted = [
        {"response_ids": torch.randint(0, MockModel.VOCAB, (1, response_len)), "reward": 0.7},
    ]
    dataset = RFTDataset()
    dataset.add_examples(prompt_ids, accepted)

    example = dataset[0]
    labels = example["labels"]

    assert labels[:prompt_len].tolist() == [-100] * prompt_len, (
        f"Prompt positions should be -100, got {labels[:prompt_len].tolist()}"
    )
    # Response positions should NOT be -100
    assert all(v != -100 for v in labels[prompt_len:].tolist()), (
        f"Response positions should not be -100, got {labels[prompt_len:].tolist()}"
    )


def test_rft_trainer_step():
    """train_step on a valid batch returns a float loss."""
    model = MockModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    trainer = RFTTrainer(model, optimizer)

    seq_len = 8
    input_ids = torch.randint(0, MockModel.VOCAB, (2, seq_len))
    labels = input_ids.clone()
    labels[:, :3] = -100  # mask first 3 positions as prompt

    loss = trainer.train_step(input_ids, labels)
    assert isinstance(loss, float), f"Expected float, got {type(loss)}"
    assert loss > 0.0, f"Expected positive loss, got {loss}"


def test_reward_stats():
    """compute_reward_stats returns dict with all required keys."""
    scored = [
        {"response_ids": torch.zeros(1, 1, dtype=torch.long), "reward": 0.2},
        {"response_ids": torch.zeros(1, 1, dtype=torch.long), "reward": 0.8},
        {"response_ids": torch.zeros(1, 1, dtype=torch.long), "reward": 0.5},
    ]
    stats = compute_reward_stats(scored)

    required_keys = {"mean_reward", "max_reward", "min_reward", "acceptance_rate"}
    assert required_keys <= set(stats.keys()), f"Missing keys: {required_keys - set(stats.keys())}"

    assert abs(stats["mean_reward"] - 0.5) < 1e-5, stats["mean_reward"]
    assert abs(stats["max_reward"] - 0.8) < 1e-5, stats["max_reward"]
    assert abs(stats["min_reward"] - 0.2) < 1e-5, stats["min_reward"]
    # All rewards > 0, so acceptance_rate == 1.0
    assert abs(stats["acceptance_rate"] - 1.0) < 1e-5, stats["acceptance_rate"]


def test_rft_loop_returns_losses():
    """rft_loop returns a non-empty list of losses when completions are accepted."""
    model = MockModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    trainer = RFTTrainer(model, optimizer)

    # Use threshold 0.0 so that every completion is accepted
    sampler = _make_sampler(k=3, reward_threshold=0.0, max_new_tokens=4)
    prompts = [_prompt(3) for _ in range(2)]

    losses = trainer.rft_loop(sampler, prompts, n_epochs=1)
    assert isinstance(losses, list), "Expected list of losses"
    assert len(losses) > 0, "Expected at least one loss entry"
    assert all(isinstance(line, float) for line in losses), "All losses should be floats"


def test_rft_loop_no_accepted():
    """rft_loop returns empty list gracefully when threshold rejects everything."""
    model = MockModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    trainer = RFTTrainer(model, optimizer)

    # threshold=999 ensures nothing is ever accepted
    sampler = _make_sampler(k=3, reward_threshold=999.0, max_new_tokens=4)
    prompts = [_prompt(3) for _ in range(2)]

    losses = trainer.rft_loop(sampler, prompts, n_epochs=1)
    assert losses == [], f"Expected empty list, got {losses}"
