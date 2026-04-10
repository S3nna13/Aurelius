"""Tests for RLAIF (Reinforcement Learning from AI Feedback) module."""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.alignment.rlaif import (
    RLAIFConfig,
    generate_response,
    ai_judge_score,
    rank_responses,
    preference_pair_from_rankings,
    RLAIFTrainer,
    _sequence_log_probs,
)


# ---------------------------------------------------------------------------
# Tiny model fixture
# ---------------------------------------------------------------------------

def _make_tiny_model() -> nn.Module:
    from src.model.config import AureliusConfig
    from src.model.transformer import AureliusTransformer

    torch.manual_seed(42)
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )
    model = AureliusTransformer(cfg)
    model.eval()
    return model


@pytest.fixture(scope="module")
def tiny_model() -> nn.Module:
    return _make_tiny_model()


@pytest.fixture(scope="module")
def judge_model() -> nn.Module:
    """Separate judge model instance."""
    return _make_tiny_model()


@pytest.fixture
def prompt_ids() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, 256, (1, 8))


# ===================================================================
# 1. RLAIFConfig tests
# ===================================================================

def test_config_defaults():
    """RLAIFConfig should have correct default values."""
    cfg = RLAIFConfig()
    assert cfg.n_samples == 4
    assert cfg.beta == 0.1
    assert cfg.ai_judge_temperature == 0.7
    assert cfg.max_response_tokens == 64
    assert cfg.reward_scale == 1.0


def test_config_custom_values():
    """RLAIFConfig should accept custom values."""
    cfg = RLAIFConfig(n_samples=8, beta=0.2, ai_judge_temperature=0.5,
                      max_response_tokens=32, reward_scale=2.0)
    assert cfg.n_samples == 8
    assert cfg.beta == 0.2
    assert cfg.ai_judge_temperature == 0.5
    assert cfg.max_response_tokens == 32
    assert cfg.reward_scale == 2.0


# ===================================================================
# 2. generate_response tests
# ===================================================================

def test_generate_response_shape(tiny_model, prompt_ids):
    """generate_response should return (1, max_tokens) tensor."""
    max_tokens = 6
    result = generate_response(tiny_model, prompt_ids, max_tokens=max_tokens)
    assert result.shape == (1, max_tokens)


def test_generate_response_dtype(tiny_model, prompt_ids):
    """generate_response should return long (integer) token ids."""
    result = generate_response(tiny_model, prompt_ids, max_tokens=4)
    assert result.dtype == torch.long


def test_generate_response_in_vocab_range(tiny_model, prompt_ids):
    """Generated tokens should be within [0, vocab_size)."""
    result = generate_response(tiny_model, prompt_ids, max_tokens=8)
    assert result.min() >= 0
    assert result.max() < 256


def test_generate_response_temperature(tiny_model, prompt_ids):
    """Different temperatures should produce valid outputs."""
    for temp in [0.5, 1.0, 1.5]:
        result = generate_response(tiny_model, prompt_ids, max_tokens=4, temperature=temp)
        assert result.shape == (1, 4)


# ===================================================================
# 3. ai_judge_score tests
# ===================================================================

def test_ai_judge_score_returns_float(tiny_model, prompt_ids):
    """ai_judge_score should return a float."""
    response_ids = torch.randint(0, 256, (1, 6))
    score = ai_judge_score(tiny_model, prompt_ids, response_ids)
    assert isinstance(score, float)


def test_ai_judge_score_negative(tiny_model, prompt_ids):
    """Log-probability scores should typically be negative."""
    response_ids = torch.randint(0, 256, (1, 10))
    score = ai_judge_score(tiny_model, prompt_ids, response_ids)
    # Log probs are <= 0; for a random model with vocab 256, mean should be negative.
    assert score <= 0.0


def test_ai_judge_score_empty_response(tiny_model, prompt_ids):
    """Empty response should return 0.0."""
    empty_response = torch.zeros((1, 0), dtype=torch.long)
    score = ai_judge_score(tiny_model, prompt_ids, empty_response)
    assert score == 0.0


# ===================================================================
# 4. rank_responses tests
# ===================================================================

def test_rank_responses_sorted(tiny_model, prompt_ids):
    """rank_responses should return responses sorted by score descending."""
    responses = [torch.randint(0, 256, (1, 6)) for _ in range(3)]
    rankings = rank_responses(tiny_model, prompt_ids, responses)
    scores = [s for _, s in rankings]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1]


def test_rank_responses_length(tiny_model, prompt_ids):
    """rank_responses should return same number of entries as input."""
    n = 5
    responses = [torch.randint(0, 256, (1, 4)) for _ in range(n)]
    rankings = rank_responses(tiny_model, prompt_ids, responses)
    assert len(rankings) == n


def test_rank_responses_tuple_structure(tiny_model, prompt_ids):
    """Each ranking entry should be (tensor, float)."""
    responses = [torch.randint(0, 256, (1, 4)) for _ in range(2)]
    rankings = rank_responses(tiny_model, prompt_ids, responses)
    for resp, score in rankings:
        assert isinstance(resp, torch.Tensor)
        assert isinstance(score, float)


# ===================================================================
# 5. preference_pair_from_rankings tests
# ===================================================================

def test_preference_pair_best_worst():
    """preference_pair_from_rankings should pick best and worst."""
    r1 = torch.tensor([[10, 20]])
    r2 = torch.tensor([[30, 40]])
    r3 = torch.tensor([[50, 60]])
    rankings = [(r1, 0.9), (r2, 0.5), (r3, 0.1)]
    chosen, rejected = preference_pair_from_rankings(rankings)
    assert torch.equal(chosen, r1)
    assert torch.equal(rejected, r3)


def test_preference_pair_two_items():
    """With two items, chosen=best, rejected=worst."""
    best = torch.tensor([[1, 2, 3]])
    worst = torch.tensor([[4, 5, 6]])
    rankings = [(best, 1.0), (worst, -1.0)]
    chosen, rejected = preference_pair_from_rankings(rankings)
    assert torch.equal(chosen, best)
    assert torch.equal(rejected, worst)


# ===================================================================
# 6. _sequence_log_probs tests
# ===================================================================

def test_sequence_log_probs_is_scalar(tiny_model, prompt_ids):
    """_sequence_log_probs should return a scalar tensor."""
    response_ids = torch.randint(0, 256, (1, 4))
    lp = _sequence_log_probs(tiny_model, prompt_ids, response_ids)
    assert lp.dim() == 0  # scalar


def test_sequence_log_probs_negative(tiny_model, prompt_ids):
    """Sum of log-probs should be negative for non-trivial sequences."""
    response_ids = torch.randint(0, 256, (1, 8))
    lp = _sequence_log_probs(tiny_model, prompt_ids, response_ids)
    assert lp.item() <= 0.0


# ===================================================================
# 7. RLAIFTrainer tests
# ===================================================================

def test_trainer_init(tiny_model, judge_model):
    """RLAIFTrainer should initialise without error."""
    cfg = RLAIFConfig(n_samples=2, max_response_tokens=4)
    opt = torch.optim.SGD(tiny_model.parameters(), lr=1e-4)
    trainer = RLAIFTrainer(tiny_model, judge_model, cfg, opt)
    assert trainer.policy_model is tiny_model
    assert trainer.judge_model is judge_model


def test_trainer_train_step_returns_dict(tiny_model, judge_model, prompt_ids):
    """train_step should return dict with required keys."""
    cfg = RLAIFConfig(n_samples=2, max_response_tokens=4)
    opt = torch.optim.SGD(tiny_model.parameters(), lr=1e-4)
    trainer = RLAIFTrainer(tiny_model, judge_model, cfg, opt)
    result = trainer.train_step(prompt_ids)
    assert isinstance(result, dict)
    assert "loss" in result
    assert "mean_score" in result
    assert "best_score" in result
    assert "n_samples" in result


def test_trainer_loss_is_float(tiny_model, judge_model, prompt_ids):
    """train_step loss should be a Python float."""
    cfg = RLAIFConfig(n_samples=2, max_response_tokens=4)
    opt = torch.optim.SGD(tiny_model.parameters(), lr=1e-4)
    trainer = RLAIFTrainer(tiny_model, judge_model, cfg, opt)
    result = trainer.train_step(prompt_ids)
    assert isinstance(result["loss"], float)


def test_trainer_n_samples_matches_config(tiny_model, judge_model, prompt_ids):
    """n_samples in result should match config."""
    cfg = RLAIFConfig(n_samples=3, max_response_tokens=4)
    opt = torch.optim.SGD(tiny_model.parameters(), lr=1e-4)
    trainer = RLAIFTrainer(tiny_model, judge_model, cfg, opt)
    result = trainer.train_step(prompt_ids)
    assert result["n_samples"] == 3


def test_trainer_best_score_gte_mean(tiny_model, judge_model, prompt_ids):
    """best_score should be >= mean_score (both scaled)."""
    cfg = RLAIFConfig(n_samples=3, max_response_tokens=4, reward_scale=1.0)
    opt = torch.optim.SGD(tiny_model.parameters(), lr=1e-4)
    trainer = RLAIFTrainer(tiny_model, judge_model, cfg, opt)
    result = trainer.train_step(prompt_ids)
    assert result["best_score"] >= result["mean_score"]


def test_trainer_reward_scale(tiny_model, judge_model, prompt_ids):
    """reward_scale should multiplicatively affect scores."""
    cfg1 = RLAIFConfig(n_samples=2, max_response_tokens=4, reward_scale=1.0)
    cfg2 = RLAIFConfig(n_samples=2, max_response_tokens=4, reward_scale=2.0)
    opt = torch.optim.SGD(tiny_model.parameters(), lr=0.0)  # lr=0 so model doesn't change

    torch.manual_seed(99)
    trainer1 = RLAIFTrainer(tiny_model, judge_model, cfg1, opt)
    r1 = trainer1.train_step(prompt_ids)

    torch.manual_seed(99)
    trainer2 = RLAIFTrainer(tiny_model, judge_model, cfg2, opt)
    r2 = trainer2.train_step(prompt_ids)

    # With same seed, the scores from scale=2 should be ~2x scale=1
    assert abs(r2["mean_score"] - 2.0 * r1["mean_score"]) < 1e-4


def test_trainer_updates_parameters(prompt_ids):
    """train_step should update policy model parameters."""
    model = _make_tiny_model()
    judge = _make_tiny_model()
    cfg = RLAIFConfig(n_samples=2, max_response_tokens=4)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    trainer = RLAIFTrainer(model, judge, cfg, opt)

    # Snapshot first parameter
    first_param = next(model.parameters())
    before = first_param.clone()

    trainer.train_step(prompt_ids)

    # Parameters should have changed
    assert not torch.equal(first_param, before), "Parameters should be updated after train_step"
