"""Integration test for STILL3Trainer.

Verifies:
- TRAINING_REGISTRY["still3"] is correctly wired.
- filter_and_prepare discards uniform-reward groups and retains varied ones.
- total_loss is computed on kept groups and supports .backward().
- Metrics dict is well-formed.
"""
from __future__ import annotations

import pytest
import torch

from src.training import TRAINING_REGISTRY
from src.training.still3_trainer import STILL3Config, STILL3Trainer


def test_still3_registry_wired():
    """TRAINING_REGISTRY must expose 'still3' mapped to STILL3Trainer."""
    assert "still3" in TRAINING_REGISTRY
    assert TRAINING_REGISTRY["still3"] is STILL3Trainer


def test_full_pipeline_filter_and_backward():
    """End-to-end: build groups, filter, compute loss, run backward pass."""
    config = STILL3Config(
        min_std_threshold=0.05,
        entropy_coeff=0.01,
        normalize_rewards=True,
    )
    trainer = STILL3Trainer(config)

    # Build 4 groups: 2 uniform (will be filtered), 2 varied (will be kept).
    # Each group has a "logits" field so we can compute total_loss later.
    B, T, V = 2, 6, 20  # batch=2 (one per kept group), seq_len=6, vocab=20

    groups = [
        {
            "rewards": [0.9, 0.9, 0.9, 0.9],   # uniform → filtered
            "tag": "uniform_easy",
        },
        {
            "rewards": [0.0, 1.0, 0.2, 0.8],   # varied → kept
            "tag": "varied_a",
        },
        {
            "rewards": [0.1, 0.1, 0.1, 0.1],   # uniform → filtered
            "tag": "uniform_hard",
        },
        {
            "rewards": [0.3, 0.9, 0.6, 0.4],   # varied → kept
            "tag": "varied_b",
        },
    ]

    # --- Step 1: filter and prepare ---
    kept = trainer.filter_and_prepare(groups)

    assert len(kept) == 2, f"Expected 2 kept groups, got {len(kept)}"
    kept_tags = {g["tag"] for g in kept}
    assert kept_tags == {"varied_a", "varied_b"}

    # After normalisation each group's rewards should have mean ≈ 0
    for g in kept:
        mean = sum(g["rewards"]) / len(g["rewards"])
        assert abs(mean) < 1e-5, (
            f"Group '{g['tag']}' rewards not normalised; mean={mean}"
        )

    # --- Step 2: compute total_loss on kept groups ---
    # Simulate: one scalar log_prob and one reward per kept group.
    torch.manual_seed(42)
    logits = torch.randn(len(kept), T, V, requires_grad=True)
    log_probs = torch.tensor(
        [-1.2, -0.8], dtype=torch.float32, requires_grad=True
    )
    # Use mean normalised reward per group as the advantage signal
    rewards = torch.tensor(
        [sum(g["rewards"]) / len(g["rewards"]) for g in kept],
        dtype=torch.float32,
    )

    total, metrics = trainer.total_loss(logits, log_probs, rewards)

    # --- Step 3: verify metrics ---
    assert isinstance(total, torch.Tensor)
    assert total.shape == torch.Size([])
    assert set(metrics.keys()) == {"policy_loss", "entropy_bonus", "total"}
    assert abs(metrics["total"] - total.item()) < 1e-6

    # --- Step 4: backward pass ---
    total.backward()
    assert logits.grad is not None, "logits.grad should be populated after backward"
    assert log_probs.grad is not None, "log_probs.grad should be populated after backward"
    assert logits.grad.shape == logits.shape
    assert log_probs.grad.shape == log_probs.shape


def test_registry_instantiation_with_custom_config():
    """Instantiate via registry with a custom config and verify it works."""
    TrainerClass = TRAINING_REGISTRY["still3"]
    config = STILL3Config(min_std_threshold=0.1, entropy_coeff=0.05)
    trainer = TrainerClass(config)

    assert trainer.config.min_std_threshold == 0.1
    assert trainer.config.entropy_coeff == 0.05

    # Quick smoke test: filtering
    groups = [
        [0.0, 0.0, 0.0],   # uniform → filtered
        [0.0, 0.5, 1.0],   # varied → kept
    ]
    result = trainer.filter_by_std(groups)
    assert len(result) == 1
    assert result[0] == [0.0, 0.5, 1.0]
