"""Tests for src/inference/speculative_sampler.py (~50 tests)."""

from __future__ import annotations

import math
import pytest

from src.inference.speculative_sampler import (
    DraftToken,
    SpeculativeConfig,
    SpeculativeSampler,
    VerificationResult,
)


# ---------------------------------------------------------------------------
# SpeculativeConfig defaults
# ---------------------------------------------------------------------------

class TestSpeculativeConfigDefaults:
    def test_n_draft_default(self):
        cfg = SpeculativeConfig()
        assert cfg.n_draft == 5

    def test_temperature_default(self):
        cfg = SpeculativeConfig()
        assert cfg.temperature == 1.0

    def test_top_p_default(self):
        cfg = SpeculativeConfig()
        assert cfg.top_p == 1.0

    def test_min_acceptance_rate_default(self):
        cfg = SpeculativeConfig()
        assert cfg.min_acceptance_rate == 0.5

    def test_custom_n_draft(self):
        cfg = SpeculativeConfig(n_draft=10)
        assert cfg.n_draft == 10

    def test_custom_temperature(self):
        cfg = SpeculativeConfig(temperature=0.7)
        assert cfg.temperature == pytest.approx(0.7)

    def test_custom_top_p(self):
        cfg = SpeculativeConfig(top_p=0.9)
        assert cfg.top_p == pytest.approx(0.9)

    def test_custom_min_acceptance_rate(self):
        cfg = SpeculativeConfig(min_acceptance_rate=0.8)
        assert cfg.min_acceptance_rate == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# DraftToken fields
# ---------------------------------------------------------------------------

class TestDraftTokenFields:
    def test_token_id_field(self):
        dt = DraftToken(token_id=42, logprob=-0.5, position=0)
        assert dt.token_id == 42

    def test_logprob_field(self):
        dt = DraftToken(token_id=1, logprob=-1.2, position=3)
        assert dt.logprob == pytest.approx(-1.2)

    def test_position_field(self):
        dt = DraftToken(token_id=0, logprob=-0.1, position=7)
        assert dt.position == 7

    def test_draft_token_is_dataclass(self):
        import dataclasses
        assert dataclasses.is_dataclass(DraftToken)


# ---------------------------------------------------------------------------
# VerificationResult fields
# ---------------------------------------------------------------------------

class TestVerificationResultFields:
    def test_accepted_tokens_field(self):
        vr = VerificationResult(accepted_tokens=[1, 2, 3], rejection_point=None, acceptance_rate=1.0)
        assert vr.accepted_tokens == [1, 2, 3]

    def test_rejection_point_none(self):
        vr = VerificationResult(accepted_tokens=[], rejection_point=None, acceptance_rate=0.0)
        assert vr.rejection_point is None

    def test_rejection_point_set(self):
        vr = VerificationResult(accepted_tokens=[1], rejection_point=2, acceptance_rate=0.5)
        assert vr.rejection_point == 2

    def test_acceptance_rate_field(self):
        vr = VerificationResult(accepted_tokens=[1, 2], rejection_point=None, acceptance_rate=0.8)
        assert vr.acceptance_rate == pytest.approx(0.8)

    def test_verification_result_is_dataclass(self):
        import dataclasses
        assert dataclasses.is_dataclass(VerificationResult)


# ---------------------------------------------------------------------------
# SpeculativeSampler construction
# ---------------------------------------------------------------------------

class TestSpeculativeSamplerConstruction:
    def test_default_config_created_when_none(self):
        s = SpeculativeSampler()
        assert isinstance(s.config, SpeculativeConfig)

    def test_custom_config_used(self):
        cfg = SpeculativeConfig(n_draft=3)
        s = SpeculativeSampler(config=cfg)
        assert s.config.n_draft == 3

    def test_none_config_uses_defaults(self):
        s = SpeculativeSampler(config=None)
        assert s.config.n_draft == 5


# ---------------------------------------------------------------------------
# SpeculativeSampler.sample_draft
# ---------------------------------------------------------------------------

class TestSampleDraft:
    def _sampler(self):
        return SpeculativeSampler()

    def test_returns_list(self):
        s = self._sampler()
        result = s.sample_draft([[1.0, 2.0, 3.0]])
        assert isinstance(result, list)

    def test_returns_draft_tokens(self):
        s = self._sampler()
        result = s.sample_draft([[1.0, 0.0, 2.0]])
        assert all(isinstance(t, DraftToken) for t in result)

    def test_length_matches_sequence_length(self):
        s = self._sampler()
        logits_seq = [[1.0, 2.0], [0.5, 1.5], [3.0, 0.0]]
        result = s.sample_draft(logits_seq)
        assert len(result) == 3

    def test_single_position(self):
        s = self._sampler()
        result = s.sample_draft([[0.1, 0.9, 0.5]])
        assert len(result) == 1

    def test_empty_sequence(self):
        s = self._sampler()
        result = s.sample_draft([])
        assert result == []

    def test_token_id_is_argmax(self):
        s = self._sampler()
        logits = [1.0, 5.0, 2.0]  # argmax = 1
        result = s.sample_draft([logits])
        assert result[0].token_id == 1

    def test_token_id_is_argmax_first(self):
        s = self._sampler()
        logits = [9.0, 1.0, 2.0]  # argmax = 0
        result = s.sample_draft([logits])
        assert result[0].token_id == 0

    def test_token_id_valid_index(self):
        s = self._sampler()
        logits_seq = [[1.0, 2.0, 3.0, 4.0, 5.0] for _ in range(4)]
        result = s.sample_draft(logits_seq)
        vocab_size = len(logits_seq[0])
        assert all(0 <= t.token_id < vocab_size for t in result)

    def test_positions_are_sequential(self):
        s = self._sampler()
        result = s.sample_draft([[1.0, 2.0]] * 5)
        for i, t in enumerate(result):
            assert t.position == i

    def test_logprob_is_negative(self):
        # log-softmax is always <= 0
        s = self._sampler()
        result = s.sample_draft([[1.0, 2.0, 3.0]])
        assert result[0].logprob <= 0.0

    def test_temperature_scaling_does_not_change_argmax(self):
        s_low = SpeculativeSampler(SpeculativeConfig(temperature=0.1))
        s_high = SpeculativeSampler(SpeculativeConfig(temperature=10.0))
        logits = [[1.0, 5.0, 2.0]]
        assert s_low.sample_draft(logits)[0].token_id == s_high.sample_draft(logits)[0].token_id


# ---------------------------------------------------------------------------
# SpeculativeSampler.verify
# ---------------------------------------------------------------------------

class TestVerify:
    def _sampler(self):
        return SpeculativeSampler()

    def _draft_tokens(self, n, logprob=-0.5):
        return [DraftToken(token_id=i, logprob=logprob, position=i) for i in range(n)]

    def test_empty_draft_tokens(self):
        s = self._sampler()
        result = s.verify([], [])
        assert result.accepted_tokens == []
        assert result.rejection_point is None
        assert result.acceptance_rate == 0.0

    def test_all_accepted_when_target_equals_draft(self):
        s = self._sampler()
        drafts = self._draft_tokens(3, logprob=-0.5)
        # target_lp == draft_lp → acceptance_prob = min(1, exp(0)) = 1.0 >= 0.5 → accepted
        target_lps = [-0.5, -0.5, -0.5]
        result = s.verify(drafts, target_lps)
        assert result.rejection_point is None
        assert len(result.accepted_tokens) == 3

    def test_all_accepted_returns_none_rejection_point(self):
        s = self._sampler()
        drafts = self._draft_tokens(2, logprob=-0.3)
        result = s.verify(drafts, [-0.3, -0.3])
        assert result.rejection_point is None

    def test_all_rejected_when_target_much_lower(self):
        s = self._sampler()
        drafts = self._draft_tokens(3, logprob=-0.1)
        # target_lp = -100 → exp(-100 - (-0.1)) ≈ 0 < 0.5 → rejected
        target_lps = [-100.0, -100.0, -100.0]
        result = s.verify(drafts, target_lps)
        assert result.rejection_point == 0
        assert result.accepted_tokens == []

    def test_acceptance_rate_all_rejected(self):
        s = self._sampler()
        drafts = self._draft_tokens(4, logprob=-0.1)
        result = s.verify(drafts, [-100.0] * 4)
        assert result.acceptance_rate == pytest.approx(0.0)

    def test_acceptance_rate_all_accepted(self):
        s = self._sampler()
        drafts = self._draft_tokens(4, logprob=-0.5)
        result = s.verify(drafts, [-0.5] * 4)
        assert result.acceptance_rate == pytest.approx(1.0)

    def test_acceptance_rate_in_range(self):
        s = self._sampler()
        drafts = self._draft_tokens(5, logprob=-0.5)
        target_lps = [-0.5, -0.5, -100.0, -100.0, -100.0]
        result = s.verify(drafts, target_lps)
        assert 0.0 <= result.acceptance_rate <= 1.0

    def test_rejection_stops_at_first_failure(self):
        s = self._sampler()
        drafts = self._draft_tokens(5, logprob=-0.5)
        # positions 0,1 accepted; position 2 rejected
        target_lps = [-0.5, -0.5, -100.0, -0.5, -0.5]
        result = s.verify(drafts, target_lps)
        assert result.rejection_point == 2
        assert len(result.accepted_tokens) == 2

    def test_returns_verification_result(self):
        s = self._sampler()
        result = s.verify(self._draft_tokens(2), [-0.5, -0.5])
        assert isinstance(result, VerificationResult)


# ---------------------------------------------------------------------------
# SpeculativeSampler.efficiency_gain
# ---------------------------------------------------------------------------

class TestEfficiencyGain:
    def _sampler(self):
        return SpeculativeSampler()

    def test_empty_results(self):
        s = self._sampler()
        assert s.efficiency_gain([]) == pytest.approx(0.0)

    def test_all_accepted(self):
        s = self._sampler()
        results = [
            VerificationResult(accepted_tokens=[1, 2, 3], rejection_point=None, acceptance_rate=1.0),
            VerificationResult(accepted_tokens=[4, 5, 6], rejection_point=None, acceptance_rate=1.0),
        ]
        assert s.efficiency_gain(results) == pytest.approx(1.0)

    def test_all_rejected(self):
        s = self._sampler()
        results = [
            VerificationResult(accepted_tokens=[], rejection_point=0, acceptance_rate=0.0),
        ]
        assert s.efficiency_gain(results) == pytest.approx(0.0)

    def test_partial_acceptance(self):
        s = self._sampler()
        results = [
            VerificationResult(accepted_tokens=[1], rejection_point=None, acceptance_rate=0.5),
            VerificationResult(accepted_tokens=[2, 3], rejection_point=None, acceptance_rate=0.5),
        ]
        assert s.efficiency_gain(results) == pytest.approx(0.5)

    def test_mixed_acceptance(self):
        s = self._sampler()
        results = [
            VerificationResult(accepted_tokens=[1, 2], rejection_point=None, acceptance_rate=1.0),
            VerificationResult(accepted_tokens=[], rejection_point=0, acceptance_rate=0.0),
        ]
        assert s.efficiency_gain(results) == pytest.approx(0.5)

    def test_single_result(self):
        s = self._sampler()
        results = [VerificationResult(accepted_tokens=[1], rejection_point=None, acceptance_rate=0.75)]
        assert s.efficiency_gain(results) == pytest.approx(0.75)
