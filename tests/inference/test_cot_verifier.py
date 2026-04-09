"""Tests for src/inference/cot_verifier.py"""

import pytest
import torch

from src.inference.cot_verifier import (
    BestOfNVerifier,
    ChainOfThoughtVerifier,
    StepVerifier,
    VerifierConfig,
    compute_process_reward,
    parse_chain_of_thought,
    verify_answer,
)

torch.manual_seed(0)

D_MODEL = 32
TOKENIZE_FN = lambda text: [ord(c) % 256 for c in text]
SAMPLE_COT = "Step 1: compute 2+2\nStep 2: result is 4\nAnswer: 4"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_config():
    return VerifierConfig()


@pytest.fixture
def step_verifier():
    torch.manual_seed(0)
    return StepVerifier(d_model=D_MODEL)


@pytest.fixture
def cot_verifier(step_verifier, default_config):
    return ChainOfThoughtVerifier(
        step_verifier=step_verifier,
        tokenize_fn=TOKENIZE_FN,
        config=default_config,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_verifier_config_defaults():
    cfg = VerifierConfig()
    assert cfg.max_steps == 10
    assert cfg.step_delimiter == "\n"
    assert cfg.answer_marker == "Answer:"
    assert cfg.confidence_threshold == 0.5


def test_parse_cot_steps_count(default_config):
    steps, _ = parse_chain_of_thought(SAMPLE_COT, default_config)
    assert len(steps) == 2


def test_parse_cot_answer_extracted(default_config):
    _, answer = parse_chain_of_thought(SAMPLE_COT, default_config)
    assert answer == "4"


def test_parse_cot_no_answer(default_config):
    text = "Step 1: think\nStep 2: think more"
    steps, answer = parse_chain_of_thought(text, default_config)
    assert answer == ""
    assert len(steps) == 2


def test_verify_answer_exact_match():
    score = verify_answer("42", "42", exact_match=True)
    assert score == 1.0


def test_verify_answer_no_match():
    score = verify_answer("apple orange", "banana grape", exact_match=False)
    assert score == 0.0


def test_verify_answer_partial():
    score = verify_answer("the quick brown fox", "the slow brown dog", exact_match=False)
    assert 0.0 < score < 1.0


def test_step_verifier_output_shape(step_verifier):
    torch.manual_seed(0)
    B, T = 4, 16
    token_ids = torch.randint(0, 256, (B, T))
    out = step_verifier(token_ids)
    assert out.shape == (B,)


def test_step_verifier_gradient_flow(step_verifier):
    torch.manual_seed(0)
    B, T = 2, 8
    token_ids = torch.randint(0, 256, (B, T))
    step_verifier.train()
    out = step_verifier(token_ids)
    loss = out.sum()
    loss.backward()
    # Check that scorer weight has gradient
    assert step_verifier.scorer.weight.grad is not None
    assert step_verifier.scorer.weight.grad.abs().sum() > 0


def test_cot_verifier_chain_keys(cot_verifier):
    result = cot_verifier.verify_chain(SAMPLE_COT, "4")
    assert "answer_score" in result
    assert "mean_step_score" in result
    assert "min_step_score" in result
    assert "n_steps" in result


def test_compute_process_reward_length():
    steps = ["step 1", "step 2", "step 3"]
    rewards = compute_process_reward(steps, final_reward=1.0)
    assert len(rewards) == len(steps)


def test_compute_process_reward_last_equals_final():
    steps = ["step 1", "step 2", "step 3"]
    final_reward = 0.8
    rewards = compute_process_reward(steps, final_reward=final_reward)
    assert rewards[-1] == pytest.approx(final_reward)
