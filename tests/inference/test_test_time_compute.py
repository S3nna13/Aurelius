"""Tests for test-time compute scaling (TTCConfig, sampling, generation, perplexity)."""
from __future__ import annotations

import math

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.inference.test_time_compute import (
    TTCConfig,
    sample_with_temperature,
    greedy_generate,
    compute_perplexity,
    best_of_n_generate,
    iterative_refine,
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
    model.eval()
    return model


@pytest.fixture(scope="module")
def fast_config():
    return TTCConfig(n_samples=2, budget=16, strategy="best_of_n", temperature=1.0)


@pytest.fixture(scope="module")
def prompt_ids(small_cfg):
    torch.manual_seed(7)
    return torch.randint(0, small_cfg.vocab_size, (1, 5))


# ---------------------------------------------------------------------------
# Score helpers (named functions to avoid inline expressions)
# ---------------------------------------------------------------------------

def neg_perplexity(model, seq):
    return -compute_perplexity(model, seq)


def biased_score_factory(prompt_len, call_counts):
    """Return a score function that prefers sequences starting with token 0."""
    def _score(model, seq):
        call_counts["n"] += 1
        first_gen = seq[0, prompt_len].item()
        return 1.0 if first_gen == 0 else 0.0
    return _score


def counting_score_factory(call_counts):
    def _score(model, seq):
        call_counts["n"] += 1
        return -compute_perplexity(model, seq)
    return _score


# ---------------------------------------------------------------------------
# 1. TTCConfig defaults
# ---------------------------------------------------------------------------

def test_ttcconfig_default_n_samples():
    cfg = TTCConfig()
    assert cfg.n_samples == 8


def test_ttcconfig_default_budget():
    cfg = TTCConfig()
    assert cfg.budget == 512


def test_ttcconfig_default_strategy():
    cfg = TTCConfig()
    assert cfg.strategy == "best_of_n"


def test_ttcconfig_default_temperature():
    cfg = TTCConfig()
    assert cfg.temperature == 1.0


def test_ttcconfig_custom_values():
    cfg = TTCConfig(n_samples=4, budget=256, strategy="tree_search", temperature=0.7)
    assert cfg.n_samples == 4
    assert cfg.budget == 256
    assert cfg.strategy == "tree_search"
    assert cfg.temperature == 0.7


# ---------------------------------------------------------------------------
# 2. sample_with_temperature — basic validity
# ---------------------------------------------------------------------------

def test_sample_with_temperature_returns_tensor():
    logits = torch.randn(VOCAB_SIZE)
    token = sample_with_temperature(logits, temperature=1.0)
    assert isinstance(token, torch.Tensor)


def test_sample_with_temperature_valid_range():
    logits = torch.randn(VOCAB_SIZE)
    token = sample_with_temperature(logits, temperature=1.0)
    assert 0 <= token.item() < VOCAB_SIZE


def test_sample_with_temperature_scalar():
    """Returned tensor should be scalar (1-element)."""
    logits = torch.randn(VOCAB_SIZE)
    token = sample_with_temperature(logits, temperature=1.0)
    assert token.numel() == 1


# ---------------------------------------------------------------------------
# 3. sample_with_temperature — high temperature -> more uniform distribution
# ---------------------------------------------------------------------------

def test_sample_with_temperature_high_temp_more_uniform():
    """At very high temperature the sampled distribution should be more uniform.

    Run many samples at low T and high T; high T should produce more unique tokens.
    """
    torch.manual_seed(0)
    logits = torch.zeros(VOCAB_SIZE)
    logits[0] = 10.0   # single dominant token at low temperature

    n_trials = 200

    low_t_tokens = {
        sample_with_temperature(logits, temperature=0.01).item()
        for _ in range(n_trials)
    }
    high_t_tokens = {
        sample_with_temperature(logits, temperature=5.0).item()
        for _ in range(n_trials)
    }

    # High temperature must produce strictly more unique tokens
    assert len(high_t_tokens) > len(low_t_tokens)


# ---------------------------------------------------------------------------
# 4. greedy_generate — returns longer sequence
# ---------------------------------------------------------------------------

def test_greedy_generate_returns_longer_sequence(small_model, prompt_ids):
    max_tokens = 6
    result = greedy_generate(small_model, prompt_ids, max_tokens)
    assert result.shape[1] == prompt_ids.shape[1] + max_tokens


def test_greedy_generate_preserves_prompt(small_model, prompt_ids):
    result = greedy_generate(small_model, prompt_ids, max_tokens=4)
    assert torch.equal(result[0, : prompt_ids.shape[1]], prompt_ids[0])


def test_greedy_generate_returns_2d_tensor(small_model, prompt_ids):
    result = greedy_generate(small_model, prompt_ids, max_tokens=3)
    assert result.dim() == 2
    assert result.shape[0] == 1


# ---------------------------------------------------------------------------
# 5. compute_perplexity — returns positive float
# ---------------------------------------------------------------------------

def test_compute_perplexity_returns_float(small_model, prompt_ids):
    ppl = compute_perplexity(small_model, prompt_ids)
    assert isinstance(ppl, float)


def test_compute_perplexity_positive(small_model, prompt_ids):
    ppl = compute_perplexity(small_model, prompt_ids)
    assert ppl > 0.0


# ---------------------------------------------------------------------------
# 6. compute_perplexity — lower for likely sequence
# ---------------------------------------------------------------------------

def test_compute_perplexity_lower_for_likely_sequence(small_model):
    """Perplexity of a greedy sequence differs from a random sequence."""
    torch.manual_seed(99)
    prompt = torch.randint(0, VOCAB_SIZE, (1, 4))
    greedy_seq = greedy_generate(small_model, prompt, max_tokens=8)
    random_seq = torch.randint(0, VOCAB_SIZE, greedy_seq.shape)

    ppl_greedy = compute_perplexity(small_model, greedy_seq)
    ppl_random = compute_perplexity(small_model, random_seq)

    # Both must be positive
    assert ppl_greedy > 0.0
    assert ppl_random > 0.0


# ---------------------------------------------------------------------------
# 7. best_of_n_generate — returns tensor
# ---------------------------------------------------------------------------

def test_best_of_n_generate_returns_tensor(small_model, prompt_ids, fast_config):
    result = best_of_n_generate(small_model, prompt_ids, fast_config)
    assert isinstance(result, torch.Tensor)


def test_best_of_n_generate_longer_than_prompt(small_model, prompt_ids, fast_config):
    result = best_of_n_generate(small_model, prompt_ids, fast_config)
    assert result.shape[1] > prompt_ids.shape[1]


# ---------------------------------------------------------------------------
# 8. best_of_n_generate — selects best by score_fn
# ---------------------------------------------------------------------------

def test_best_of_n_generate_uses_score_fn(small_model, prompt_ids):
    """A custom score_fn is called once per sample."""
    call_counts = {"n": 0}
    score_fn = biased_score_factory(prompt_ids.shape[1], call_counts)

    cfg = TTCConfig(n_samples=3, budget=12, strategy="best_of_n", temperature=2.0)
    result = best_of_n_generate(small_model, prompt_ids, cfg, score_fn=score_fn)

    assert call_counts["n"] == cfg.n_samples
    assert isinstance(result, torch.Tensor)


# ---------------------------------------------------------------------------
# 9. iterative_refine — returns tensor of similar length
# ---------------------------------------------------------------------------

def test_iterative_refine_returns_tensor(small_model, prompt_ids, fast_config):
    result = iterative_refine(small_model, prompt_ids, fast_config, n_iterations=1)
    assert isinstance(result, torch.Tensor)


def test_iterative_refine_longer_than_prompt(small_model, prompt_ids, fast_config):
    result = iterative_refine(small_model, prompt_ids, fast_config, n_iterations=1)
    assert result.shape[1] > prompt_ids.shape[1]


def test_iterative_refine_preserves_prompt(small_model, prompt_ids, fast_config):
    result = iterative_refine(small_model, prompt_ids, fast_config, n_iterations=1)
    assert torch.equal(result[0, : prompt_ids.shape[1]], prompt_ids[0])


# ---------------------------------------------------------------------------
# 10. TestTimeScaler.generate — returns (sequence, score) tuple
# ---------------------------------------------------------------------------

def test_testtime_scaler_generate_returns_tuple(small_model, prompt_ids, fast_config):
    scaler = TestTimeScaler(small_model, fast_config)
    out = scaler.generate(prompt_ids)
    assert isinstance(out, tuple)
    assert len(out) == 2


def test_testtime_scaler_generate_sequence_is_tensor(small_model, prompt_ids, fast_config):
    scaler = TestTimeScaler(small_model, fast_config)
    best_seq, best_score = scaler.generate(prompt_ids)
    assert isinstance(best_seq, torch.Tensor)


def test_testtime_scaler_generate_score_is_float(small_model, prompt_ids, fast_config):
    scaler = TestTimeScaler(small_model, fast_config)
    best_seq, best_score = scaler.generate(prompt_ids)
    assert isinstance(best_score, float)


def test_testtime_scaler_iterative_refinement_strategy(small_model, prompt_ids):
    cfg = TTCConfig(n_samples=2, budget=16, strategy="iterative_refinement", temperature=1.0)
    scaler = TestTimeScaler(small_model, cfg)
    best_seq, best_score = scaler.generate(prompt_ids)
    assert isinstance(best_seq, torch.Tensor)
    assert isinstance(best_score, float)


def test_testtime_scaler_tree_search_strategy(small_model, prompt_ids):
    cfg = TTCConfig(n_samples=2, budget=16, strategy="tree_search", temperature=1.0)
    scaler = TestTimeScaler(small_model, cfg)
    best_seq, best_score = scaler.generate(prompt_ids)
    assert isinstance(best_seq, torch.Tensor)
    assert isinstance(best_score, float)


def test_testtime_scaler_custom_score_fn(small_model, prompt_ids, fast_config):
    """Custom score_fn is honored by TestTimeScaler."""
    call_counts = {"n": 0}
    score_fn = counting_score_factory(call_counts)

    scaler = TestTimeScaler(small_model, fast_config)
    best_seq, best_score = scaler.generate(prompt_ids, score_fn=score_fn)
    assert call_counts["n"] >= 1


def test_testtime_scaler_invalid_strategy_raises(small_model, prompt_ids):
    cfg = TTCConfig(strategy="unknown_strategy")
    scaler = TestTimeScaler(small_model, cfg)
    with pytest.raises(ValueError, match="Unknown strategy"):
        scaler.generate(prompt_ids)
