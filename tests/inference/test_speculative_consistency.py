"""Tests for src/inference/speculative_consistency.py

12 tests covering config defaults, greedy decode, speculative decode,
consistency checking, acceptance stats, distribution testing, threshold
calibration and the ConsistencyReport dataclass.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.inference.speculative_consistency import (
    SpeculativeConsistencyConfig,
    ConsistencyReport,
    greedy_decode,
    speculative_decode_simple,
    SpeculativeConsistencyChecker,
)

# ---------------------------------------------------------------------------
# Mock models
# ---------------------------------------------------------------------------

class TargetModel(nn.Module):
    def __init__(self, vocab_size: int = 256) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, 16)

    def forward(self, input_ids, **kwargs):
        B, T = input_ids.shape
        logits = torch.zeros(B, T, self.vocab_size)
        logits[:, :, 1] = 10.0  # always predicts token 1
        return (None, logits, None)


class DraftModel(nn.Module):
    def __init__(self, vocab_size: int = 256, agreement: float = 1.0) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.agreement = agreement
        self.embed = nn.Embedding(vocab_size, 16)

    def forward(self, input_ids, **kwargs):
        B, T = input_ids.shape
        logits = torch.zeros(B, T, self.vocab_size)
        logits[:, :, 1] = 10.0 * self.agreement          # controlled agreement
        logits[:, :, 2] = 10.0 * (1 - self.agreement)
        return (None, logits, None)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VOCAB = 256
PROMPT = torch.tensor([[5, 6, 7, 8]])  # (1, 4)


# ---------------------------------------------------------------------------
# Test 1: SpeculativeConsistencyConfig defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = SpeculativeConsistencyConfig()
    assert cfg.n_test_samples == 100
    assert cfg.max_new_tokens == 20
    assert cfg.acceptance_threshold == 0.0
    assert cfg.target_acceptance_rate == 0.8


# ---------------------------------------------------------------------------
# Test 2: greedy_decode returns longer sequence than input
# ---------------------------------------------------------------------------

def test_greedy_decode_longer_than_input():
    model = TargetModel(VOCAB)
    output = greedy_decode(model, PROMPT, max_new_tokens=5)
    # output shape should be (B, max_new_tokens)
    assert output.shape[1] > 0
    assert output.shape[1] == 5


# ---------------------------------------------------------------------------
# Test 3: greedy_decode is deterministic
# ---------------------------------------------------------------------------

def test_greedy_decode_deterministic():
    model = TargetModel(VOCAB)
    out1 = greedy_decode(model, PROMPT, max_new_tokens=5)
    out2 = greedy_decode(model, PROMPT, max_new_tokens=5)
    assert torch.equal(out1, out2)


# ---------------------------------------------------------------------------
# Test 4: speculative_decode_simple returns valid sequence
# ---------------------------------------------------------------------------

def test_speculative_decode_simple_valid():
    target = TargetModel(VOCAB)
    draft = DraftModel(VOCAB, agreement=1.0)
    out = speculative_decode_simple(target, draft, PROMPT, n_draft=3, max_new=10)
    assert out.ndim == 2
    assert out.shape[0] == PROMPT.shape[0]
    assert out.shape[1] == 10


# ---------------------------------------------------------------------------
# Test 5: With identical draft + target (agreement=1.0), verify_single is True
# ---------------------------------------------------------------------------

def test_verify_single_consistent():
    target = TargetModel(VOCAB)
    draft = DraftModel(VOCAB, agreement=1.0)
    checker = SpeculativeConsistencyChecker(target, draft)
    result = checker.verify_single(PROMPT, max_new_tokens=5)
    assert result is True


# ---------------------------------------------------------------------------
# Test 6: compute_acceptance_stats returns dict with required keys
# ---------------------------------------------------------------------------

def test_acceptance_stats_keys():
    target = TargetModel(VOCAB)
    draft = DraftModel(VOCAB, agreement=1.0)
    checker = SpeculativeConsistencyChecker(target, draft)
    stats = checker.compute_acceptance_stats(PROMPT, n_samples=5)
    assert "mean_acceptance_rate" in stats
    assert "std_acceptance_rate" in stats
    assert "min_acceptance" in stats
    assert "max_acceptance" in stats


# ---------------------------------------------------------------------------
# Test 7: Mean acceptance rate is in [0, 1]
# ---------------------------------------------------------------------------

def test_acceptance_rate_range():
    target = TargetModel(VOCAB)
    draft = DraftModel(VOCAB, agreement=1.0)
    checker = SpeculativeConsistencyChecker(target, draft)
    stats = checker.compute_acceptance_stats(PROMPT, n_samples=5)
    assert 0.0 <= stats["mean_acceptance_rate"] <= 1.0


# ---------------------------------------------------------------------------
# Test 8: distribution_test returns dict with 'kl_divergence' key
# ---------------------------------------------------------------------------

def test_distribution_test_keys():
    target = TargetModel(VOCAB)
    draft = DraftModel(VOCAB, agreement=1.0)
    checker = SpeculativeConsistencyChecker(target, draft)
    result = checker.distribution_test(PROMPT, n_samples=5, position=0)
    assert "kl_divergence" in result
    assert "tvd" in result
    assert "is_consistent" in result


# ---------------------------------------------------------------------------
# Test 9: kl_divergence >= 0
# ---------------------------------------------------------------------------

def test_kl_divergence_nonnegative():
    target = TargetModel(VOCAB)
    draft = DraftModel(VOCAB, agreement=1.0)
    checker = SpeculativeConsistencyChecker(target, draft)
    result = checker.distribution_test(PROMPT, n_samples=5, position=0)
    assert result["kl_divergence"] >= 0.0


# ---------------------------------------------------------------------------
# Test 10: calibrate_threshold returns float in [0, 1]
# ---------------------------------------------------------------------------

def test_calibrate_threshold_range():
    target = TargetModel(VOCAB)
    draft = DraftModel(VOCAB, agreement=1.0)
    checker = SpeculativeConsistencyChecker(target, draft)
    threshold = checker.calibrate_threshold([PROMPT], target_acceptance=0.8)
    assert isinstance(threshold, float)
    assert 0.0 <= threshold <= 1.0


# ---------------------------------------------------------------------------
# Test 11: ConsistencyReport has all required fields
# ---------------------------------------------------------------------------

def test_consistency_report_fields():
    report = ConsistencyReport(
        n_tests=10,
        n_consistent=9,
        consistency_rate=0.9,
        mean_acceptance_rate=0.85,
        kl_divergence=0.01,
    )
    assert report.n_tests == 10
    assert report.n_consistent == 9
    assert report.consistency_rate == 0.9
    assert report.mean_acceptance_rate == 0.85
    assert report.kl_divergence == 0.01


# ---------------------------------------------------------------------------
# Test 12: greedy_decode with max_new_tokens=5 returns exactly 5 new tokens
# ---------------------------------------------------------------------------

def test_greedy_decode_exact_length():
    model = TargetModel(VOCAB)
    output = greedy_decode(model, PROMPT, max_new_tokens=5)
    assert output.shape == (PROMPT.shape[0], 5)
