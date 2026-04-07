"""Tests for GRPO alignment config and reward functions."""
import pytest
from src.alignment.grpo import (
    GRPORunConfig,
    extract_answer,
    gsm8k_reward,
    format_reward,
    combined_reward,
)


def test_grpo_config_defaults():
    cfg = GRPORunConfig()
    assert cfg.num_generations == 8
    assert cfg.learning_rate == 1e-6
    assert cfg.beta1 if hasattr(cfg, 'beta1') else True  # no beta in GRPO


def test_extract_answer_gsm8k_format():
    text = "So we get 3 + 4 = 7. #### 7"
    assert extract_answer(text) == "7"


def test_extract_answer_the_answer_is():
    text = "The answer is 42."
    assert extract_answer(text) == "42"


def test_extract_answer_not_found():
    text = "I'm not sure how to solve this."
    assert extract_answer(text) is None


def test_extract_answer_with_commas():
    text = "#### 1,234"
    assert extract_answer(text) == "1234"


def test_gsm8k_reward_correct():
    completions = ["Step 1: add. #### 7"]
    ground_truths = ["#### 7"]
    rewards = gsm8k_reward(completions, ground_truths)
    assert rewards == [1.0]


def test_gsm8k_reward_incorrect():
    completions = ["#### 5"]
    ground_truths = ["#### 7"]
    rewards = gsm8k_reward(completions, ground_truths)
    assert rewards == [0.0]


def test_gsm8k_reward_batch():
    completions = ["#### 3", "#### 10", "I don't know"]
    ground_truths = ["#### 3", "#### 7", "#### 5"]
    rewards = gsm8k_reward(completions, ground_truths)
    assert rewards[0] == 1.0   # correct
    assert rewards[1] == 0.0   # wrong
    assert rewards[2] == 0.0   # no answer found


def test_format_reward_step_marker():
    completions = ["Step 1: 2 + 2 = 4. The answer is 4."]
    rewards = format_reward(completions)
    assert rewards[0] > 0.0


def test_format_reward_no_structure():
    completions = ["I have no idea."]
    rewards = format_reward(completions)
    assert rewards[0] == 0.0


def test_combined_reward_correct_gets_high_score():
    completions = ["Step 1. #### 7"]
    ground_truths = ["#### 7"]
    rewards = combined_reward(completions, ground_truths)
    assert rewards[0] >= 1.0  # at least correctness reward


def test_combined_reward_wrong_gets_low():
    completions = ["#### 999"]
    ground_truths = ["#### 1"]
    rewards = combined_reward(completions, ground_truths)
    assert rewards[0] < 1.0
