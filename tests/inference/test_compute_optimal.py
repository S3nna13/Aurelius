"""Tests for inference-time compute scaling (arXiv:2408.03314)."""

import math

import pytest
import torch

from src.inference.compute_optimal import (
    ComputeOptimalConfig,
    ComputeOptimalResult,
    SelectionStrategy,
    compute_optimal_generate,
    generate_n_samples,
    score_with_model,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_cfg():
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=32,
    )


@pytest.fixture(scope="module")
def small_model(small_cfg):
    torch.manual_seed(42)
    model = AureliusTransformer(small_cfg)
    model.eval()
    return model


@pytest.fixture
def prompt(small_cfg):
    torch.manual_seed(0)
    return torch.randint(0, small_cfg.vocab_size, (1, 4))


@pytest.fixture
def fast_cfg():
    """Config with n_samples=2 and short generation for speed."""
    return ComputeOptimalConfig(
        n_samples=2,
        strategy=SelectionStrategy.BEST_OF_N,
        max_new_tokens=4,
        temperature=1.0,
        top_p=0.9,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = ComputeOptimalConfig()
    assert cfg.n_samples == 8
    assert cfg.strategy == SelectionStrategy.BEST_OF_N


def test_score_with_model_returns_float(small_model, small_cfg):
    prompt_ids = torch.randint(0, small_cfg.vocab_size, (4,))
    response_ids = torch.randint(0, small_cfg.vocab_size, (3,))
    score = score_with_model(small_model, prompt_ids, response_ids)
    assert isinstance(score, float)


def test_score_with_model_finite(small_model, small_cfg):
    prompt_ids = torch.randint(0, small_cfg.vocab_size, (4,))
    response_ids = torch.randint(0, small_cfg.vocab_size, (3,))
    score = score_with_model(small_model, prompt_ids, response_ids)
    assert math.isfinite(score)


def test_generate_n_samples_count(small_model, prompt, fast_cfg):
    samples = generate_n_samples(small_model, prompt, fast_cfg)
    assert len(samples) == fast_cfg.n_samples


def test_generate_n_samples_tensor_type(small_model, prompt, fast_cfg):
    samples = generate_n_samples(small_model, prompt, fast_cfg)
    for s in samples:
        assert isinstance(s, torch.Tensor)


def test_compute_optimal_best_of_n(small_model, prompt, fast_cfg):
    result = compute_optimal_generate(small_model, prompt, fast_cfg)
    assert isinstance(result, ComputeOptimalResult)
    assert result.selected_score == max(result.all_scores)


def test_compute_optimal_majority_vote(small_model, prompt, small_cfg):
    cfg = ComputeOptimalConfig(
        n_samples=2,
        strategy=SelectionStrategy.MAJORITY_VOTE,
        max_new_tokens=4,
    )
    result = compute_optimal_generate(small_model, prompt, cfg)
    assert isinstance(result, ComputeOptimalResult)
    assert result.strategy == SelectionStrategy.MAJORITY_VOTE
    assert result.selected_ids is not None


def test_compute_optimal_with_verifier(small_model, prompt, fast_cfg):
    call_count = {"n": 0}

    def custom_verifier(prompt_ids, response_ids):
        call_count["n"] += 1
        return 0.5  # constant score

    result = compute_optimal_generate(small_model, prompt, fast_cfg, verifier=custom_verifier)
    assert isinstance(result, ComputeOptimalResult)
    # Verifier must have been called once per sample
    assert call_count["n"] == fast_cfg.n_samples


def test_result_score_mean(small_model, prompt, fast_cfg):
    result = compute_optimal_generate(small_model, prompt, fast_cfg)
    expected = sum(result.all_scores) / len(result.all_scores)
    assert abs(result.score_mean - expected) < 1e-9


def test_result_n_samples(small_model, prompt, fast_cfg):
    result = compute_optimal_generate(small_model, prompt, fast_cfg)
    assert result.n_samples_generated == fast_cfg.n_samples
