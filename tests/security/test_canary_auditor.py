"""Tests for the canary-based memorization auditor."""

from __future__ import annotations

import math

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.security.canary_auditor import CanaryAuditor

# ---------------------------------------------------------------------------
# Shared tiny config and fixtures
# ---------------------------------------------------------------------------

TINY_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=4,
    n_kv_heads=2,
    head_dim=16,
    d_ff=128,
    vocab_size=256,
    max_seq_len=64,
)

CANARY_LEN = 16
SEED_A = 42
SEED_B = 99


@pytest.fixture(scope="module")
def model() -> AureliusTransformer:
    torch.manual_seed(SEED_A)
    m = AureliusTransformer(TINY_CFG)
    m.eval()
    return m


@pytest.fixture(scope="module")
def auditor(model) -> CanaryAuditor:
    return CanaryAuditor(model, vocab_size=TINY_CFG.vocab_size, canary_len=CANARY_LEN)


@pytest.fixture(scope="module")
def canary_a(auditor) -> torch.LongTensor:
    return auditor.generate_canary(SEED_A)


@pytest.fixture(scope="module")
def canary_b(auditor) -> torch.LongTensor:
    return auditor.generate_canary(SEED_B)


# ---------------------------------------------------------------------------
# 1. generate_canary returns LongTensor of shape (1, canary_len)
# ---------------------------------------------------------------------------

def test_generate_canary_shape(auditor):
    canary = auditor.generate_canary(SEED_A)
    assert canary.shape == (1, CANARY_LEN)


def test_generate_canary_dtype(auditor):
    canary = auditor.generate_canary(SEED_A)
    assert canary.dtype == torch.long


# ---------------------------------------------------------------------------
# 2. All tokens are in [0, vocab_size)
# ---------------------------------------------------------------------------

def test_generate_canary_token_range(auditor):
    canary = auditor.generate_canary(SEED_A)
    assert canary.min().item() >= 0
    assert canary.max().item() < TINY_CFG.vocab_size


# ---------------------------------------------------------------------------
# 3. Same seed produces identical canary
# ---------------------------------------------------------------------------

def test_generate_canary_deterministic(auditor):
    c1 = auditor.generate_canary(SEED_A)
    c2 = auditor.generate_canary(SEED_A)
    assert torch.equal(c1, c2)


# ---------------------------------------------------------------------------
# 4. Different seeds produce different canaries
# ---------------------------------------------------------------------------

def test_generate_canary_different_seeds(canary_a, canary_b):
    assert not torch.equal(canary_a, canary_b)


# ---------------------------------------------------------------------------
# 5. canary_loss returns float
# ---------------------------------------------------------------------------

def test_canary_loss_returns_float(auditor, canary_a):
    loss = auditor.canary_loss(canary_a)
    assert isinstance(loss, float)


# ---------------------------------------------------------------------------
# 6. canary_loss is finite and non-negative
# ---------------------------------------------------------------------------

def test_canary_loss_finite_nonneg(auditor, canary_a):
    loss = auditor.canary_loss(canary_a)
    assert math.isfinite(loss)
    assert loss >= 0.0


# ---------------------------------------------------------------------------
# 7. perplexity returns float >= 1.0
# ---------------------------------------------------------------------------

def test_perplexity_returns_float(auditor, canary_a):
    ppl = auditor.perplexity(canary_a)
    assert isinstance(ppl, float)


def test_perplexity_at_least_one(auditor, canary_a):
    ppl = auditor.perplexity(canary_a)
    assert ppl >= 1.0


# ---------------------------------------------------------------------------
# 8. perplexity == exp(canary_loss)
# ---------------------------------------------------------------------------

def test_perplexity_equals_exp_loss(auditor, canary_a):
    loss = auditor.canary_loss(canary_a)
    ppl = auditor.perplexity(canary_a)
    assert math.isclose(ppl, math.exp(loss), rel_tol=1e-6)


# ---------------------------------------------------------------------------
# 9. exposure returns float
# ---------------------------------------------------------------------------

def test_exposure_returns_float(auditor, canary_a):
    result = auditor.exposure(canary_a, n_random_trials=10, seed=SEED_A)
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# 10. exposure is non-negative
# ---------------------------------------------------------------------------

def test_exposure_nonneg(auditor, canary_a):
    result = auditor.exposure(canary_a, n_random_trials=10, seed=SEED_A)
    assert result >= 0.0


# ---------------------------------------------------------------------------
# 11. is_memorized returns bool
# ---------------------------------------------------------------------------

def test_is_memorized_returns_bool(auditor, canary_a):
    result = auditor.is_memorized(canary_a, threshold_perplexity=1e9)
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# 12. High threshold → is_memorized True
# ---------------------------------------------------------------------------

def test_is_memorized_high_threshold(auditor, canary_a):
    assert auditor.is_memorized(canary_a, threshold_perplexity=1e12) is True


# ---------------------------------------------------------------------------
# 13. Low threshold → is_memorized False
# ---------------------------------------------------------------------------

def test_is_memorized_low_threshold(auditor, canary_a):
    assert auditor.is_memorized(canary_a, threshold_perplexity=1.0) is False


# ---------------------------------------------------------------------------
# 14. exposure with n_random_trials=10 runs without error
# ---------------------------------------------------------------------------

def test_exposure_ten_trials_no_error(auditor, canary_a):
    result = auditor.exposure(canary_a, n_random_trials=10, seed=7)
    assert math.isfinite(result)
