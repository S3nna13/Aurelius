"""Integration tests for Toggle token-efficient RL registry wiring."""
import torch
import pytest
from src.alignment import ALIGNMENT_REGISTRY


def test_toggle_in_registry():
    """'toggle' key must be present in ALIGNMENT_REGISTRY."""
    assert "toggle" in ALIGNMENT_REGISTRY, (
        f"'toggle' not found in ALIGNMENT_REGISTRY. Keys: {list(ALIGNMENT_REGISTRY.keys())}"
    )


def test_toggle_construct_and_call_from_registry():
    """Construct ToggleReward from registry, call it, verify output shape."""
    ToggleReward = ALIGNMENT_REGISTRY["toggle"]
    reward_fn = ToggleReward(lambda_threshold=0.8, token_budget=2048)

    mean_reward = torch.tensor([1.0, 2.0, 3.0, 4.0])

    # Phase 0 within budget/threshold
    result = reward_fn(mean_reward, accuracy=0.9, tokens_used=512, phase=0)
    assert result.shape == mean_reward.shape, (
        f"Shape mismatch: expected {mean_reward.shape}, got {result.shape}"
    )
    assert torch.allclose(result, mean_reward), (
        f"Expected mean_reward passthrough, got {result}"
    )

    # Phase 1 always passes
    result_p1 = reward_fn(mean_reward, accuracy=0.0, tokens_used=99999, phase=1)
    assert result_p1.shape == mean_reward.shape
    assert torch.allclose(result_p1, mean_reward)


def test_existing_registry_keys_still_present():
    """Regression guard: pre-existing registry keys must not be removed."""
    assert "prm" in ALIGNMENT_REGISTRY, "'prm' missing from registry (regression)"
    assert "adversarial_code_battle" in ALIGNMENT_REGISTRY, (
        "'adversarial_code_battle' missing from registry (regression)"
    )
    assert "constitution_dimensions" in ALIGNMENT_REGISTRY, (
        "'constitution_dimensions' missing from registry (regression)"
    )
