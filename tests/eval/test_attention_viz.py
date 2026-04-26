"""Tests for src/eval/attention_viz.py — numerical attention pattern analysis."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from aurelius.eval.attention_viz import (
    AttentionConcentration,
    HeadDiversityAnalyzer,
    RecencyBiasAnalyzer,
    SinkTokenDetector,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

B, H, T = 2, 4, 8


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def uniform_attn():
    """Perfectly uniform attention: (B, H, T, T), each row sums to 1."""
    attn = torch.full((B, H, T, T), 1.0 / T)
    return attn


@pytest.fixture
def concentrated_attn():
    """Concentrated attention: all weight on first key for every query."""
    attn = torch.zeros(B, H, T, T)
    attn[..., 0] = 1.0
    return attn


@pytest.fixture
def random_attn():
    """Normalized random attention weights."""
    torch.manual_seed(42)
    raw = torch.rand(B, H, T, T)
    return F.softmax(raw, dim=-1)


# ---------------------------------------------------------------------------
# AttentionConcentration
# ---------------------------------------------------------------------------


class TestAttentionConcentration:
    def setup_method(self):
        self.ac = AttentionConcentration()

    # Test 1: gini_coefficient output shape
    def test_gini_shape(self, random_attn):
        result = self.ac.gini_coefficient(random_attn)
        assert result.shape == (B, H, T), f"Expected ({B}, {H}, {T}), got {result.shape}"

    # Test 2: Gini = 0 for uniform attention
    def test_gini_zero_for_uniform(self, uniform_attn):
        result = self.ac.gini_coefficient(uniform_attn)
        # For uniform distribution, Gini should be ~0
        assert result.abs().max().item() < 1e-4, (
            f"Gini should be ~0 for uniform, got max={result.abs().max().item()}"
        )

    # Test 3: Gini > 0 for concentrated attention
    def test_gini_positive_for_concentrated(self, concentrated_attn):
        result = self.ac.gini_coefficient(concentrated_attn)
        assert result.min().item() > 0, (
            f"Gini should be positive for concentrated attention, got min={result.min().item()}"
        )

    # Test 4: top_k_fraction output shape
    def test_top_k_fraction_shape(self, random_attn):
        result = self.ac.top_k_fraction(random_attn, k=3)
        assert result.shape == (B, H, T), f"Expected ({B}, {H}, {T}), got {result.shape}"

    # Test 5: top_k_fraction in [0, 1]
    def test_top_k_fraction_range(self, random_attn):
        result = self.ac.top_k_fraction(random_attn, k=5)
        assert result.min().item() >= 0.0, f"top_k_fraction < 0: {result.min().item()}"
        assert result.max().item() <= 1.0 + 1e-6, f"top_k_fraction > 1: {result.max().item()}"

    # Test 6: entropy output shape
    def test_entropy_shape(self, random_attn):
        result = self.ac.entropy(random_attn)
        assert result.shape == (B, H, T), f"Expected ({B}, {H}, {T}), got {result.shape}"

    # Test 7: entropy = log(T_k) for uniform attention (maximum entropy)
    def test_entropy_max_for_uniform(self, uniform_attn):
        result = self.ac.entropy(uniform_attn)
        expected = math.log(T)
        assert abs(result.mean().item() - expected) < 1e-4, (
            f"Expected entropy ~{expected:.4f}, got {result.mean().item():.4f}"
        )


# ---------------------------------------------------------------------------
# RecencyBiasAnalyzer
# ---------------------------------------------------------------------------


class TestRecencyBiasAnalyzer:
    def setup_method(self):
        self.rba = RecencyBiasAnalyzer()

    # Test 8: relative_position_weights shape (T,)
    def test_relative_position_weights_shape(self, random_attn):
        result = self.rba.relative_position_weights(random_attn)
        assert result.shape == (T,), f"Expected ({T},), got {result.shape}"

    # Test 9: recency_score in [0, 1]
    def test_recency_score_range(self, random_attn):
        score = self.rba.recency_score(random_attn, window=3)
        assert 0.0 <= score <= 1.0 + 1e-6, f"recency_score out of [0,1]: {score}"


# ---------------------------------------------------------------------------
# SinkTokenDetector
# ---------------------------------------------------------------------------


class TestSinkTokenDetector:
    def setup_method(self):
        self.detector = SinkTokenDetector(sink_threshold=0.1)

    # Test 10: detect_sinks returns bool tensor of shape (T_k,)
    def test_detect_sinks_shape_and_dtype(self, random_attn):
        result = self.detector.detect_sinks(random_attn)
        assert result.shape == (T,), f"Expected ({T},), got {result.shape}"
        assert result.dtype == torch.bool, f"Expected bool dtype, got {result.dtype}"

    # Test 11: sink_positions returns list of ints
    def test_sink_positions_list_of_ints(self, concentrated_attn):
        # concentrated_attn puts all weight on position 0 → should be a sink
        positions = self.detector.sink_positions(concentrated_attn)
        assert isinstance(positions, list), f"Expected list, got {type(positions)}"
        for p in positions:
            assert isinstance(p, int), f"Expected int elements, got {type(p)}"
        # Position 0 must be detected as a sink
        assert 0 in positions, f"Position 0 should be a sink, got {positions}"

    # Test 12: sink_attention_fraction in [0, 1]
    def test_sink_attention_fraction_range(self, random_attn):
        frac = self.detector.sink_attention_fraction(random_attn)
        assert 0.0 <= frac <= 1.0 + 1e-6, f"sink_attention_fraction out of [0,1]: {frac}"


# ---------------------------------------------------------------------------
# HeadDiversityAnalyzer
# ---------------------------------------------------------------------------


class TestHeadDiversityAnalyzer:
    def setup_method(self):
        self.hda = HeadDiversityAnalyzer()

    # Test 13: inter_head_kl shape (H, H)
    def test_inter_head_kl_shape(self, random_attn):
        result = self.hda.inter_head_kl(random_attn)
        assert result.shape == (H, H), f"Expected ({H}, {H}), got {result.shape}"

    # Test 14: KL diagonal is ~0 (a head vs itself)
    def test_inter_head_kl_diagonal_zero(self, random_attn):
        result = self.hda.inter_head_kl(random_attn)
        diag = result.diagonal()
        assert diag.abs().max().item() < 1e-4, (
            f"KL diagonal should be ~0, got max={diag.abs().max().item()}"
        )

    # Test 15: effective_heads returns positive int
    def test_effective_heads_positive_int(self, random_attn):
        count = self.hda.effective_heads(random_attn, threshold=0.0)
        assert isinstance(count, int), f"Expected int, got {type(count)}"
        assert count >= 1, f"effective_heads should be >= 1, got {count}"
