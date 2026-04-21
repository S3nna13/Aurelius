"""Unit tests for Toggle token-efficient RL (Kimi K2.5 §3.4, arXiv:2602.02276)."""
import pytest
import torch
from src.alignment.toggle import ToggleReward


@pytest.fixture
def toggle():
    return ToggleReward(lambda_threshold=0.8, token_budget=2048)


def test_phase0_below_threshold(toggle):
    """Phase 0: accuracy below threshold -> reward = 0."""
    mean_reward = torch.tensor([1.0, 2.0, 3.0])
    result = toggle(mean_reward, accuracy=0.5, tokens_used=100, phase=0)
    assert torch.all(result == 0.0), f"Expected zeros, got {result}"


def test_phase0_above_threshold_within_budget(toggle):
    """Phase 0: accuracy >= threshold and tokens <= budget -> reward = mean_reward."""
    mean_reward = torch.tensor([1.0, 2.0, 3.0])
    result = toggle(mean_reward, accuracy=0.9, tokens_used=1000, phase=0)
    assert torch.allclose(result, mean_reward), f"Expected {mean_reward}, got {result}"


def test_phase0_over_budget(toggle):
    """Phase 0: accuracy meets threshold but tokens exceed budget -> reward = 0."""
    mean_reward = torch.tensor([1.0, 2.0, 3.0])
    result = toggle(mean_reward, accuracy=0.95, tokens_used=4096, phase=0)
    assert torch.all(result == 0.0), f"Expected zeros, got {result}"


def test_phase1_always_passes_reward(toggle):
    """Phase 1: reward always equals mean_reward regardless of accuracy/tokens."""
    mean_reward = torch.tensor([1.5, -0.5, 2.0])
    # Low accuracy and high tokens — should still pass through in phase 1
    result = toggle(mean_reward, accuracy=0.1, tokens_used=99999, phase=1)
    assert torch.allclose(result, mean_reward), f"Expected {mean_reward}, got {result}"


def test_batch_shape(toggle):
    """Input tensor of shape [4] produces output of shape [4]."""
    mean_reward = torch.randn(4)
    result = toggle(mean_reward, accuracy=0.9, tokens_used=512, phase=0)
    assert result.shape == (4,), f"Expected shape (4,), got {result.shape}"


def test_no_nan_on_zero_inputs_phase0(toggle):
    """No NaN values when mean_reward is all zeros in phase 0."""
    mean_reward = torch.zeros(3)
    result = toggle(mean_reward, accuracy=0.9, tokens_used=512, phase=0)
    assert not torch.isnan(result).any(), "Got NaN in phase 0 zero-input result"


def test_no_nan_on_zero_inputs_phase1(toggle):
    """No NaN values when mean_reward is all zeros in phase 1."""
    mean_reward = torch.zeros(3)
    result = toggle(mean_reward, accuracy=0.0, tokens_used=99999, phase=1)
    assert not torch.isnan(result).any(), "Got NaN in phase 1 zero-input result"


def test_exactly_at_threshold(toggle):
    """Phase 0: accuracy exactly at lambda_threshold -> reward given (inclusive)."""
    mean_reward = torch.tensor([2.0, 3.0])
    result = toggle(mean_reward, accuracy=0.8, tokens_used=512, phase=0)
    assert torch.allclose(result, mean_reward), (
        f"Expected reward at exact threshold, got {result}"
    )


def test_exactly_at_budget(toggle):
    """Phase 0: tokens_used exactly at token_budget -> reward given (inclusive)."""
    mean_reward = torch.tensor([2.0, 3.0])
    result = toggle(mean_reward, accuracy=0.9, tokens_used=2048, phase=0)
    assert torch.allclose(result, mean_reward), (
        f"Expected reward at exact budget, got {result}"
    )


def test_phase1_ignores_accuracy(toggle):
    """Phase 1: accuracy=0.0, tokens=9999 -> still returns mean_reward."""
    mean_reward = torch.tensor([5.0])
    result = toggle(mean_reward, accuracy=0.0, tokens_used=9999, phase=1)
    assert torch.allclose(result, mean_reward), (
        f"Expected {mean_reward} in phase 1, got {result}"
    )


def test_determinism(toggle):
    """Same call produces identical result (deterministic)."""
    mean_reward = torch.tensor([1.0, 2.0, 3.0])
    result_a = toggle(mean_reward, accuracy=0.9, tokens_used=512, phase=0)
    result_b = toggle(mean_reward, accuracy=0.9, tokens_used=512, phase=0)
    assert torch.allclose(result_a, result_b), (
        f"Non-deterministic: {result_a} vs {result_b}"
    )
