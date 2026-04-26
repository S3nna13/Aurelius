"""Tests for src/data/token_frequency_analysis.py.

Uses vocab_size=50 and small token sequences to keep the suite fast.
Pure PyTorch only — no scipy, numpy, or sklearn.
"""

from __future__ import annotations

import torch

from src.data.token_frequency_analysis import (
    DistributionShiftDetector,
    TokenFrequencyCounter,
    VocabularyStats,
    ZipfAnalyzer,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

VOCAB_SIZE = 50


def make_counter(token_ids_list: list[int] | None = None) -> TokenFrequencyCounter:
    """Return a TokenFrequencyCounter updated with token_ids_list (or a default)."""
    counter = TokenFrequencyCounter(vocab_size=VOCAB_SIZE)
    if token_ids_list is None:
        # Tokens 0-9, token 0 appears most often (10x), token 1 five times, etc.
        token_ids_list = [0] * 10 + [1] * 5 + [2] * 3 + [3] * 2 + [4] * 1
    ids = torch.tensor(token_ids_list, dtype=torch.long)
    counter.update(ids)
    return counter


# ---------------------------------------------------------------------------
# TokenFrequencyCounter tests
# ---------------------------------------------------------------------------


def test_update_increments_total_tokens():
    """update() should add the number of tokens to total_tokens."""
    counter = TokenFrequencyCounter(vocab_size=VOCAB_SIZE)
    ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    counter.update(ids)
    assert counter.total_tokens == 4


def test_counts_sum_equals_total_tokens():
    """Sum of all counts must equal total_tokens."""
    counter = make_counter()
    assert counter.counts.sum().item() == counter.total_tokens


def test_frequency_sums_to_one():
    """frequency() must sum to approximately 1.0."""
    counter = make_counter()
    freq_sum = counter.frequency().sum().item()
    assert abs(freq_sum - 1.0) < 1e-5


def test_top_k_tokens_sorted_descending():
    """top_k_tokens() counts must be in non-increasing order."""
    counter = make_counter()
    _, counts = counter.top_k_tokens(k=5)
    for i in range(len(counts) - 1):
        assert counts[i].item() >= counts[i + 1].item()


def test_top_k_tokens_ids_in_vocab_range():
    """All returned token ids must be valid vocab indices."""
    counter = make_counter()
    token_ids, _ = counter.top_k_tokens(k=5)
    assert token_ids.min().item() >= 0
    assert token_ids.max().item() < VOCAB_SIZE


def test_rare_tokens_below_threshold():
    """rare_tokens() should return only tokens with count < min_count."""
    counter = make_counter()
    # token 4 appears once, tokens 5-49 appear zero times — all < 2
    rare = counter.rare_tokens(min_count=2)
    for tid in rare.tolist():
        assert counter.counts[tid].item() < 2


# ---------------------------------------------------------------------------
# ZipfAnalyzer tests
# ---------------------------------------------------------------------------


def test_fit_zipf_returns_expected_keys():
    """fit_zipf() must return 'slope', 'intercept', and 'r_squared'."""
    analyzer = ZipfAnalyzer()
    counts = torch.arange(VOCAB_SIZE, 0, -1, dtype=torch.float32)
    result = analyzer.fit_zipf(counts)
    assert set(result.keys()) == {"slope", "intercept", "r_squared"}


def test_fit_zipf_r_squared_in_range():
    """r_squared must lie in [0, 1]."""
    analyzer = ZipfAnalyzer()
    counts = torch.arange(VOCAB_SIZE, 0, -1, dtype=torch.float32)
    result = analyzer.fit_zipf(counts)
    assert 0.0 <= result["r_squared"] <= 1.0


def test_zipf_divergence_non_negative():
    """zipf_divergence() must be >= 0 for any distribution."""
    analyzer = ZipfAnalyzer()
    # Uniform distribution — likely deviates from Zipf
    counts = torch.ones(VOCAB_SIZE, dtype=torch.float32)
    div = analyzer.zipf_divergence(counts)
    assert div >= 0.0


# ---------------------------------------------------------------------------
# DistributionShiftDetector tests
# ---------------------------------------------------------------------------


def test_kl_divergence_non_negative():
    """KL divergence is always >= 0."""
    detector = DistributionShiftDetector()
    freq_a = torch.rand(VOCAB_SIZE)
    freq_b = torch.rand(VOCAB_SIZE)
    kl = detector.kl_divergence(freq_a, freq_b)
    assert kl >= 0.0


def test_kl_divergence_zero_for_identical():
    """KL(P || P) should be approximately 0."""
    detector = DistributionShiftDetector()
    freq = torch.rand(VOCAB_SIZE)
    kl = detector.kl_divergence(freq, freq)
    assert abs(kl) < 1e-4


def test_top_k_shift_returns_gained_and_lost():
    """top_k_shift() must return dict with 'gained' and 'lost' keys."""
    detector = DistributionShiftDetector()
    freq_a = torch.rand(VOCAB_SIZE)
    freq_b = torch.rand(VOCAB_SIZE)
    result = detector.top_k_shift(freq_a, freq_b, k=10)
    assert "gained" in result
    assert "lost" in result
    assert isinstance(result["gained"], torch.Tensor)
    assert isinstance(result["lost"], torch.Tensor)


def test_coverage_in_range():
    """coverage() must return a value in [0, 1]."""
    detector = DistributionShiftDetector()
    freq_a = torch.rand(VOCAB_SIZE)
    freq_b = torch.rand(VOCAB_SIZE)
    cov = detector.coverage(freq_a, freq_b)
    assert 0.0 <= cov <= 1.0


# ---------------------------------------------------------------------------
# VocabularyStats tests
# ---------------------------------------------------------------------------


def test_vocab_stats_returns_expected_keys():
    """compute() must return all six expected stat keys."""
    stats = VocabularyStats()
    counter = make_counter()
    result = stats.compute(counter)
    expected = {
        "vocab_coverage",
        "hapax_legomena",
        "mean_freq",
        "median_freq",
        "max_freq",
        "entropy",
    }
    assert set(result.keys()) == expected


def test_entropy_non_negative():
    """Shannon entropy of any distribution is >= 0."""
    stats = VocabularyStats()
    counter = make_counter()
    result = stats.compute(counter)
    assert result["entropy"] >= 0.0
