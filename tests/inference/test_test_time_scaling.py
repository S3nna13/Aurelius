"""Tests for test-time compute scaling: majority voting, reward-guided search, MCTS."""
from __future__ import annotations

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.inference.test_time_scaling import (
    ScalingConfig,
    nucleus_sample,
    generate_samples,
    majority_vote,
    self_consistency_vote,
    StepLevelScorer,
    BeamMCTS,
    TestTimeScaler,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256

@pytest.fixture(scope="module")
def small_cfg():
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=VOCAB_SIZE,
        max_seq_len=512,
    )


@pytest.fixture(scope="module")
def small_model(small_cfg):
    torch.manual_seed(42)
    model = AureliusTransformer(small_cfg)
    model.train(False)
    return model


@pytest.fixture(scope="module")
def fast_config():
    return ScalingConfig(n_samples=2, max_new_tokens=4, temperature=0.8, top_p=0.95)


@pytest.fixture(scope="module")
def prompt_tensor(small_cfg):
    torch.manual_seed(0)
    return torch.randint(0, small_cfg.vocab_size, (1, 4))


def dummy_encode(text: str) -> list[int]:
    """Simple byte-level tokenizer for testing."""
    return [b % VOCAB_SIZE for b in text.encode("utf-8")] or [0]


def dummy_decode(token_ids: list[int]) -> str:
    """Decode token ids back to a string (best-effort)."""
    return str(token_ids)


# ---------------------------------------------------------------------------
# 1. test_scaling_config_defaults
# ---------------------------------------------------------------------------

def test_scaling_config_defaults():
    cfg = ScalingConfig()
    assert cfg.n_samples == 8
    assert cfg.max_new_tokens == 64
    assert cfg.temperature == 0.8
    assert cfg.top_p == 0.95
    assert cfg.voting_method == "majority"
    assert cfg.use_step_level is False


# ---------------------------------------------------------------------------
# 2. test_nucleus_sample_valid_token
# ---------------------------------------------------------------------------

def test_nucleus_sample_valid_token():
    V = VOCAB_SIZE
    torch.manual_seed(1)
    logits = torch.randn(V)
    token_id = nucleus_sample(logits, top_p=0.9, temperature=1.0)
    assert isinstance(token_id, int)
    assert 0 <= token_id < V


# ---------------------------------------------------------------------------
# 3. test_nucleus_sample_top1_deterministic
# ---------------------------------------------------------------------------

def test_nucleus_sample_top1_deterministic():
    """With top_p=1.0 and temperature near-zero the argmax token is always chosen."""
    V = VOCAB_SIZE
    logits = torch.zeros(V)
    best_token = 42
    logits[best_token] = 100.0   # overwhelmingly highest

    results = {nucleus_sample(logits, top_p=1.0, temperature=0.01) for _ in range(10)}
    assert results == {best_token}


# ---------------------------------------------------------------------------
# 4. test_generate_samples_count
# ---------------------------------------------------------------------------

def test_generate_samples_count(small_model, prompt_tensor, fast_config):
    samples = generate_samples(small_model, prompt_tensor, fast_config)
    assert len(samples) == fast_config.n_samples


# ---------------------------------------------------------------------------
# 5. test_generate_samples_nonempty
# ---------------------------------------------------------------------------

def test_generate_samples_nonempty(small_model, prompt_tensor, fast_config):
    samples = generate_samples(small_model, prompt_tensor, fast_config)
    for s in samples:
        assert len(s) > 0


# ---------------------------------------------------------------------------
# 6. test_majority_vote_most_common
# ---------------------------------------------------------------------------

def test_majority_vote_most_common():
    answers = ["apple", "banana", "apple", "apple", "banana"]
    assert majority_vote(answers) == "apple"


# ---------------------------------------------------------------------------
# 7. test_majority_vote_tie_first
# ---------------------------------------------------------------------------

def test_majority_vote_tie_first():
    # "cat" appears first but both appear once — should return first encountered
    answers = ["cat", "dog"]
    result = majority_vote(answers)
    assert result == "cat"


# ---------------------------------------------------------------------------
# 8. test_self_consistency_probabilities_sum
# ---------------------------------------------------------------------------

def test_self_consistency_probabilities_sum():
    answers = ["yes", "no", "yes", "maybe", "no", "yes"]
    probs = self_consistency_vote(answers, normalize=True)
    assert abs(sum(probs.values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 9. test_step_level_scorer_returns_float
# ---------------------------------------------------------------------------

def test_step_level_scorer_returns_float(small_model):
    scorer = StepLevelScorer(small_model, dummy_encode)
    score = scorer.score_step("Hello world", "Next step")
    assert isinstance(score, float)


# ---------------------------------------------------------------------------
# 10. test_test_time_scaler_generate_and_vote_returns_str
# ---------------------------------------------------------------------------

def test_test_time_scaler_generate_and_vote_returns_str(small_model, fast_config):
    scaler = TestTimeScaler(
        model=small_model,
        tokenizer_encode=dummy_encode,
        tokenizer_decode=dummy_decode,
        config=fast_config,
    )
    best, info = scaler.generate_and_vote("Hello")
    assert isinstance(best, str)


# ---------------------------------------------------------------------------
# 11. test_test_time_scaler_vote_dict_keys
# ---------------------------------------------------------------------------

def test_test_time_scaler_vote_dict_keys(small_model, fast_config):
    scaler = TestTimeScaler(
        model=small_model,
        tokenizer_encode=dummy_encode,
        tokenizer_decode=dummy_decode,
        config=fast_config,
    )
    best, info = scaler.generate_and_vote("Hello")
    assert "samples" in info
    assert "votes" in info
    assert isinstance(info["samples"], list)
    assert len(info["samples"]) == fast_config.n_samples


# ---------------------------------------------------------------------------
# 12. test_scale_compute_override_n_samples
# ---------------------------------------------------------------------------

def test_scale_compute_override_n_samples(small_model):
    config = ScalingConfig(n_samples=2, max_new_tokens=4)
    scaler = TestTimeScaler(
        model=small_model,
        tokenizer_encode=dummy_encode,
        tokenizer_decode=dummy_decode,
        config=config,
    )
    # Override n_samples to 1
    result = scaler.scale_compute("Test prompt", n_samples=1)
    assert isinstance(result, str)
    # Ensure original config is restored
    assert config.n_samples == 2
