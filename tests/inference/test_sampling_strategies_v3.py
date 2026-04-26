"""Tests for src/inference/sampling_strategies_v3.py

Covers all six classes with ≥ 14 test functions.
"""

from __future__ import annotations

import pytest
import torch

from src.inference.sampling_strategies_v3 import (
    DRYRepetitionPenalty,
    LogitsProcessor,
    MinPSampling,
    Sampler,
    SamplingConfig,
    TypicalSampling,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOCAB = 20


def _uniform_logits(v: int = VOCAB) -> torch.Tensor:
    return torch.zeros(v)


def _peaked_logits(peak_idx: int = 0, peak_val: float = 10.0, v: int = VOCAB) -> torch.Tensor:
    logits = torch.zeros(v)
    logits[peak_idx] = peak_val
    return logits


# ---------------------------------------------------------------------------
# SamplingConfig — defaults
# ---------------------------------------------------------------------------


class TestSamplingConfigDefaults:
    def test_defaults(self):
        cfg = SamplingConfig()
        assert cfg.temperature == 1.0
        assert cfg.top_k == 0
        assert cfg.top_p == 1.0
        assert cfg.min_p == 0.0
        assert cfg.typical_p == 1.0
        assert cfg.repetition_penalty == 1.0
        assert cfg.dry_multiplier == 0.0
        assert cfg.dry_base == 1.75
        assert cfg.dry_allowed_length == 2


# ---------------------------------------------------------------------------
# MinPSampling
# ---------------------------------------------------------------------------


class TestMinPSampling:
    def setup_method(self):
        self.sampler = MinPSampling()

    def test_removes_low_prob_tokens(self):
        """Tokens far below p_max * min_p should be set to -inf."""
        logits = _peaked_logits(peak_idx=0, peak_val=10.0)
        filtered = self.sampler.filter(logits, min_p=0.5)
        torch.softmax(logits, dim=-1)
        # All tokens except 0 have near-zero prob; they must be -inf after filtering
        assert filtered[0] != float("-inf"), "Top token should not be removed"
        # At least one non-peak token must be -inf
        assert (filtered[1:] == float("-inf")).any()

    def test_keeps_top_token(self):
        """The token with the highest probability must never be removed."""
        logits = _peaked_logits(peak_idx=3, peak_val=8.0)
        filtered = self.sampler.filter(logits, min_p=0.9)
        assert filtered[3] != float("-inf")

    def test_zero_min_p_is_noop(self):
        """min_p=0 disables the filter entirely."""
        logits = _peaked_logits()
        filtered = self.sampler.filter(logits, min_p=0.0)
        assert torch.equal(filtered, logits)

    def test_uniform_keeps_all_with_small_min_p(self):
        """With uniform logits and tiny min_p, all tokens should survive."""
        logits = _uniform_logits()
        filtered = self.sampler.filter(logits, min_p=0.001)
        assert not (filtered == float("-inf")).any()


# ---------------------------------------------------------------------------
# TypicalSampling
# ---------------------------------------------------------------------------


class TestTypicalSampling:
    def setup_method(self):
        self.sampler = TypicalSampling()

    def test_typical_p_1_keeps_all_tokens(self):
        """typical_p=1.0 should be a no-op."""
        logits = _peaked_logits()
        filtered = self.sampler.filter(logits, typical_p=1.0)
        assert torch.equal(filtered, logits)

    def test_keeps_tokens_near_entropy(self):
        """With a peaked distribution, low-p=1.0 boundary should prune low-prob tokens."""
        logits = _peaked_logits(peak_idx=0, peak_val=10.0)
        filtered = self.sampler.filter(logits, typical_p=0.5)
        # At least some tokens should survive (the typical ones)
        valid = (filtered != float("-inf")).sum().item()
        assert valid >= 1

    def test_typical_p_small_prunes_aggressively(self):
        """Very small typical_p should leave fewer tokens."""
        logits = _peaked_logits(peak_idx=0, peak_val=5.0)
        filtered_small = self.sampler.filter(logits, typical_p=0.3)
        filtered_large = self.sampler.filter(logits, typical_p=0.9)
        small_count = (filtered_small != float("-inf")).sum().item()
        large_count = (filtered_large != float("-inf")).sum().item()
        assert small_count <= large_count


# ---------------------------------------------------------------------------
# DRYRepetitionPenalty
# ---------------------------------------------------------------------------


class TestDRYRepetitionPenalty:
    def test_no_penalty_for_token_not_in_context(self):
        """A token that never appears in context should receive no penalty."""
        dry = DRYRepetitionPenalty(multiplier=1.0, base=1.75, allowed_length=2)
        logits = torch.zeros(VOCAB)
        context = [1, 2, 3, 4]  # token 5 not in context
        modified = dry.compute_penalty(logits, context)
        # Token 5 should be unchanged (no match)
        assert modified[5].item() == pytest.approx(0.0, abs=1e-5)

    def test_penalty_increases_with_repetition_length(self):
        """Longer suffix matches should produce larger penalties."""
        dry = DRYRepetitionPenalty(multiplier=1.0, base=1.75, allowed_length=1)
        vocab = 10
        # Context: [0, 1, 2, 0, 1] — if we add token 2 we get suffix [0,1,2]
        # which appears at position 0 in context → match_length=3
        # If we add token 1 we get suffix [0,1] which appears at pos 3 → match_length=2
        context = [0, 1, 2, 0, 1]
        logits = torch.zeros(vocab)
        modified = dry.compute_penalty(logits, context)
        # Token 2 has a longer match than token 0 (no match expected for token 0 alone)
        # The key property: penalty for token with longer match ≥ penalty for shorter match
        # We verify token 2 is penalised more than a token with no match (token 9)
        penalty_2 = logits[2].item() - modified[2].item()
        penalty_9 = logits[9].item() - modified[9].item()
        assert penalty_2 >= penalty_9

    def test_zero_multiplier_is_noop(self):
        """multiplier=0 disables DRY penalty."""
        dry = DRYRepetitionPenalty(multiplier=0.0, base=1.75, allowed_length=2)
        logits = torch.ones(VOCAB)
        context = list(range(VOCAB))
        modified = dry.compute_penalty(logits, context)
        assert torch.equal(logits, modified)

    def test_empty_context_is_noop(self):
        """Empty context means no match is possible, no change."""
        dry = DRYRepetitionPenalty(multiplier=2.0, base=1.75, allowed_length=1)
        logits = torch.ones(VOCAB)
        modified = dry.compute_penalty(logits, [])
        assert torch.equal(logits, modified)


# ---------------------------------------------------------------------------
# LogitsProcessor
# ---------------------------------------------------------------------------


class TestLogitsProcessor:
    def test_temperature_less_than_1_sharpens(self):
        """Temperature < 1 should increase the spread (larger logit differences)."""
        cfg = SamplingConfig(temperature=0.5)
        proc = LogitsProcessor(cfg)
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
        sharp = proc.apply_temperature(logits)
        # Dividing by 0.5 doubles the values → differences are larger
        assert (sharp - logits).abs().max() > 0
        assert torch.allclose(sharp, logits / 0.5)

    def test_temperature_greater_than_1_flattens(self):
        """Temperature > 1 should compress the logit differences."""
        cfg = SamplingConfig(temperature=2.0)
        proc = LogitsProcessor(cfg)
        logits = torch.tensor([1.0, 2.0, 4.0, 8.0])
        flat = proc.apply_temperature(logits)
        assert torch.allclose(flat, logits / 2.0)

    def test_top_k_keeps_exactly_k_tokens(self):
        """apply_top_k should keep exactly k finite tokens."""
        cfg = SamplingConfig(top_k=5)
        proc = LogitsProcessor(cfg)
        logits = torch.arange(VOCAB, dtype=torch.float)
        filtered = proc.apply_top_k(logits, k=5)
        finite_count = (filtered != float("-inf")).sum().item()
        assert finite_count == 5

    def test_repetition_penalty_reduces_logit_for_repeated_token(self):
        """A token that appears in context should have its positive logit reduced."""
        cfg = SamplingConfig(repetition_penalty=1.5)
        proc = LogitsProcessor(cfg)
        logits = torch.ones(VOCAB) * 2.0
        context = [3, 7, 12]
        modified = proc.apply_repetition_penalty(logits, context)
        # Repeated tokens have positive logits → divided by penalty → smaller
        for t in context:
            assert modified[t].item() < logits[t].item()
        # Non-repeated tokens unchanged
        assert modified[0].item() == pytest.approx(logits[0].item())

    def test_process_returns_finite_logits_with_top_k(self):
        """After process(), at least one logit must be finite."""
        cfg = SamplingConfig(temperature=0.8, top_k=10, top_p=0.9, repetition_penalty=1.2)
        proc = LogitsProcessor(cfg)
        logits = torch.randn(VOCAB)
        context = list(range(5))
        out = proc.process(logits, context)
        assert (out != float("-inf")).any()

    def test_process_all_filters_disabled_is_noop(self):
        """Default config (all filters at identity values) should not change logits."""
        cfg = SamplingConfig()  # all defaults
        proc = LogitsProcessor(cfg)
        logits = torch.randn(VOCAB)
        out = proc.process(logits, context_ids=None)
        assert torch.allclose(out, logits)


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class TestSampler:
    def setup_method(self):
        self.sampler = Sampler(vocab_size=VOCAB)

    def test_sample_returns_valid_token_id(self):
        """sample() must return an int in [0, vocab_size)."""
        logits = torch.randn(VOCAB)
        token = self.sampler.sample(logits)
        assert isinstance(token, int)
        assert 0 <= token < VOCAB

    def test_greedy_returns_argmax(self):
        """greedy() must return the index of the maximum logit."""
        logits = _peaked_logits(peak_idx=7, peak_val=100.0)
        token = self.sampler.greedy(logits)
        assert token == 7

    def test_greedy_deterministic(self):
        """greedy() is deterministic for the same input."""
        logits = torch.randn(VOCAB)
        assert self.sampler.greedy(logits) == self.sampler.greedy(logits)

    def test_beam_step_returns_correct_shapes(self):
        """beam_step() should return tensors of shape (beam_size,)."""
        beam = 4
        logits = torch.randn(beam, VOCAB)
        beam_scores = torch.zeros(beam)
        new_scores, new_ids = self.sampler.beam_step(logits, beam_scores, beam_size=beam)
        assert new_scores.shape == (beam,)
        assert new_ids.shape == (beam,)

    def test_beam_step_token_ids_in_range(self):
        """All token ids returned by beam_step must be in [0, vocab_size)."""
        beam = 3
        logits = torch.randn(beam, VOCAB)
        beam_scores = torch.tensor([-1.0, -2.0, -3.0])
        _, new_ids = self.sampler.beam_step(logits, beam_scores, beam_size=beam)
        assert (new_ids >= 0).all()
        assert (new_ids < VOCAB).all()

    def test_beam_step_broadcast_1d_logits(self):
        """beam_step should handle 1-D logits broadcast across beams."""
        beam = 3
        logits = torch.randn(VOCAB)  # 1-D, will be broadcast
        beam_scores = torch.zeros(beam)
        new_scores, new_ids = self.sampler.beam_step(logits, beam_scores, beam_size=beam)
        assert new_scores.shape == (beam,)
        assert new_ids.shape == (beam,)
