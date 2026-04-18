"""Tests for src/inference/nucleus_plus.py"""
import pytest
import torch
import torch.nn.functional as F

from src.inference.nucleus_plus import (
    NucleusConfig,
    apply_temperature,
    apply_repetition_penalty,
    apply_min_p,
    apply_surprise_upweight,
    nucleus_plus_sample,
    NucleusDecoder,
)

VOCAB_SIZE = 64
EOS = 2


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _uniform_logits(vocab: int = VOCAB_SIZE) -> torch.Tensor:
    return torch.zeros(vocab)


def _mock_model(ids: torch.Tensor) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(ids.shape[0], ids.shape[1], VOCAB_SIZE)


def _prompt(length: int = 4) -> torch.Tensor:
    return torch.randint(0, VOCAB_SIZE, (1, length))


# ---------------------------------------------------------------------------
# Test 1: temperature > 1 flattens the distribution
# ---------------------------------------------------------------------------

def test_temperature_flattens_distribution():
    """Higher temperature should produce a more uniform distribution."""
    torch.manual_seed(0)
    logits = torch.randn(VOCAB_SIZE)
    p_cold = F.softmax(apply_temperature(logits, 0.5), dim=-1)
    p_hot = F.softmax(apply_temperature(logits, 2.0), dim=-1)
    # Entropy of hot distribution should be larger
    eps = 1e-9
    entropy_cold = -(p_cold * (p_cold + eps).log()).sum().item()
    entropy_hot = -(p_hot * (p_hot + eps).log()).sum().item()
    assert entropy_hot > entropy_cold


# ---------------------------------------------------------------------------
# Test 2: repetition penalty reduces score of repeated tokens
# ---------------------------------------------------------------------------

def test_repetition_penalty_reduces_repeated_tokens():
    """Tokens in prev_ids should have reduced logits after penalty."""
    logits = torch.ones(VOCAB_SIZE) * 2.0
    prev_ids = [5, 10, 15]
    penalized = apply_repetition_penalty(logits.clone(), prev_ids, penalty=2.0)
    for tok in prev_ids:
        assert penalized[tok].item() < logits[tok].item(), (
            f"Token {tok} logit should decrease under repetition penalty"
        )
    # Non-penalized tokens should be unchanged
    assert penalized[0].item() == logits[0].item()


# ---------------------------------------------------------------------------
# Test 3: min_p filters low-probability tokens
# ---------------------------------------------------------------------------

def test_min_p_filters_low_prob_tokens():
    """Tokens with prob below min_p * max_prob should be zeroed."""
    # Create a distribution with one dominant token
    probs = torch.full((VOCAB_SIZE,), 0.001)
    probs[0] = 0.9
    probs = probs / probs.sum()

    filtered = apply_min_p(probs, min_p=0.1)
    # Only token 0 should survive (0.001 << 0.1 * 0.9 = 0.09)
    assert filtered[0].item() > 0.0
    for i in range(1, VOCAB_SIZE):
        assert filtered[i].item() == 0.0, f"Token {i} should be zeroed by min_p"
    # Should sum to 1
    assert abs(filtered.sum().item() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Test 4: surprise_alpha=0 is identity
# ---------------------------------------------------------------------------

def test_surprise_alpha_zero_is_identity():
    """apply_surprise_upweight with alpha=0 should return probs unchanged."""
    probs = F.softmax(torch.randn(VOCAB_SIZE), dim=-1)
    out = apply_surprise_upweight(probs, alpha=0.0)
    assert torch.allclose(out, probs)


# ---------------------------------------------------------------------------
# Test 5: nucleus_plus_sample returns int in valid range
# ---------------------------------------------------------------------------

def test_nucleus_plus_sample_returns_valid_int():
    """nucleus_plus_sample must return a Python int in [0, vocab_size)."""
    torch.manual_seed(1)
    logits = torch.randn(VOCAB_SIZE)
    config = NucleusConfig()
    token = nucleus_plus_sample(logits, [], config)
    assert isinstance(token, int)
    assert 0 <= token < VOCAB_SIZE


# ---------------------------------------------------------------------------
# Test 6: temperature=0.01 is near-greedy (always picks argmax)
# ---------------------------------------------------------------------------

def test_very_low_temperature_is_near_greedy():
    """With temperature close to 0, the sampler should almost always pick the argmax."""
    torch.manual_seed(99)
    logits = torch.randn(VOCAB_SIZE)
    greedy_token = int(logits.argmax().item())
    config = NucleusConfig(temperature=0.01, top_p=1.0, min_p=0.0, repetition_penalty=1.0)
    # With very low temp, multinomial should almost certainly pick the top token
    for seed in range(10):
        torch.manual_seed(seed)
        token = nucleus_plus_sample(logits.clone(), [], config)
        assert token == greedy_token, (
            f"seed={seed}: expected greedy token {greedy_token}, got {token}"
        )


# ---------------------------------------------------------------------------
# Test 7: config defaults work without arguments
# ---------------------------------------------------------------------------

def test_config_defaults():
    """NucleusConfig() should have the correct default field values."""
    cfg = NucleusConfig()
    assert cfg.temperature == 1.0
    assert cfg.top_p == 0.9
    assert cfg.min_p == 0.02
    assert cfg.repetition_penalty == 1.1
    assert cfg.surprise_alpha == 0.0


# ---------------------------------------------------------------------------
# Test 8: NucleusDecoder.generate returns a list of ints
# ---------------------------------------------------------------------------

def test_generate_returns_list_of_ints():
    """generate() must return a list of Python ints."""
    torch.manual_seed(7)
    decoder = NucleusDecoder(NucleusConfig(top_p=1.0, min_p=0.0, repetition_penalty=1.0))
    tokens = decoder.generate(_mock_model, _prompt(), max_new_tokens=5, eos_id=EOS)
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    for t in tokens:
        assert isinstance(t, int)
        assert 0 <= t < VOCAB_SIZE


# ---------------------------------------------------------------------------
# Test 9: eos_id stops generation early
# ---------------------------------------------------------------------------

def test_eos_stops_generation():
    """generate() should stop as soon as eos_id is emitted."""
    eos = 3
    # Model that always emits eos_id strongly
    def eos_model(ids: torch.Tensor) -> torch.Tensor:
        logits = torch.full((1, ids.shape[1], VOCAB_SIZE), -1e9)
        logits[:, :, eos] = 10.0
        return logits

    decoder = NucleusDecoder(NucleusConfig(top_p=1.0, min_p=0.0, repetition_penalty=1.0,
                                           temperature=1.0))
    tokens = decoder.generate(eos_model, _prompt(), max_new_tokens=20, eos_id=eos)
    # Should stop after 1 token (the EOS itself)
    assert len(tokens) == 1
    assert tokens[0] == eos


# ---------------------------------------------------------------------------
# Test 10: batch of 1 works (input_ids shape (1, T))
# ---------------------------------------------------------------------------

def test_batch_of_one_works():
    """generate() should handle input_ids with batch dim = 1."""
    torch.manual_seed(5)
    decoder = NucleusDecoder(NucleusConfig(top_p=1.0, min_p=0.0, repetition_penalty=1.0))
    input_ids = torch.randint(0, VOCAB_SIZE, (1, 6))  # shape (1, 6)
    tokens = decoder.generate(_mock_model, input_ids, max_new_tokens=4, eos_id=EOS)
    assert isinstance(tokens, list)
    assert len(tokens) > 0
