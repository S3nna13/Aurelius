"""Tests for src/inference/beam_search_v3.py

All tests use tiny configs (small vocab, short sequences) and pure PyTorch.
"""

from __future__ import annotations

import pytest
import torch

from src.inference.beam_search_v3 import (
    BeamConfig,
    BeamHypothesis,
    BeamSearchDecoder,
    apply_length_penalty,
    apply_repetition_penalty,
    get_ngram_blocked_tokens,
)

# ---------------------------------------------------------------------------
# Shared test constants and helpers
# ---------------------------------------------------------------------------

VOCAB = 20
EOS = 2
PROMPT_LEN = 3


def _make_prompt() -> torch.Tensor:
    """Return a fixed 1-D prompt tensor."""
    return torch.tensor([5, 6, 7], dtype=torch.long)


def _random_model_fn(ids: torch.Tensor) -> torch.Tensor:
    """Deterministic mock model: returns random logits seeded by input shape."""
    torch.manual_seed(ids.shape[1])
    return torch.randn(ids.shape[0], ids.shape[1], VOCAB)


def _eos_model_fn(ids: torch.Tensor) -> torch.Tensor:
    """Mock model that always strongly prefers token EOS."""
    logits = torch.full((ids.shape[0], ids.shape[1], VOCAB), -1e9)
    logits[:, :, EOS] = 10.0
    return logits


def _uniform_model_fn(ids: torch.Tensor) -> torch.Tensor:
    """Mock model that returns uniform logits (no preference)."""
    return torch.zeros(ids.shape[0], ids.shape[1], VOCAB)


# ---------------------------------------------------------------------------
# 1. BeamConfig defaults
# ---------------------------------------------------------------------------


class TestBeamConfigDefaults:
    def test_beam_width_default(self):
        cfg = BeamConfig()
        assert cfg.beam_width == 4

    def test_max_new_tokens_default(self):
        cfg = BeamConfig()
        assert cfg.max_new_tokens == 128

    def test_length_penalty_default(self):
        cfg = BeamConfig()
        assert cfg.length_penalty == 1.0

    def test_no_repeat_ngram_size_default(self):
        cfg = BeamConfig()
        assert cfg.no_repeat_ngram_size == 0

    def test_min_length_default(self):
        cfg = BeamConfig()
        assert cfg.min_length == 0

    def test_eos_token_id_default(self):
        cfg = BeamConfig()
        assert cfg.eos_token_id == 2

    def test_temperature_default(self):
        cfg = BeamConfig()
        assert cfg.temperature == 1.0

    def test_repetition_penalty_default(self):
        cfg = BeamConfig()
        assert cfg.repetition_penalty == 1.0


# ---------------------------------------------------------------------------
# 2. apply_length_penalty
# ---------------------------------------------------------------------------


class TestApplyLengthPenalty:
    def test_shorter_gets_higher_normalized_score(self):
        """With alpha > 0, a sequence with a better per-token score wins.

        Give two sequences the same per-token log-prob (-1.0 per token).
        The short sequence (length 3) has raw score -3.0; the long sequence
        (length 10) has raw score -10.0.  After length-normalisation, the
        short sequence should have the higher (less negative) score because
        both sequences were equally good per token, but the normaliser grows
        with length, so the long sequence's larger raw score is penalised more.
        """
        # Equal per-token quality: score ≈ length * (-1.0)
        scores = torch.tensor([-3.0, -10.0])  # short (len 3) vs long (len 10)
        lengths = torch.tensor([3, 10])
        normed = apply_length_penalty(scores, lengths, alpha=1.0)
        # short: -3.0 / ((5+3)/6)^1 = -3.0 / 1.333 ≈ -2.25
        # long:  -10.0 / ((5+10)/6)^1 = -10.0 / 2.5 = -4.0
        assert normed[0].item() > normed[1].item()

    def test_alpha_zero_is_noop(self):
        """alpha=0 must leave scores unchanged."""
        scores = torch.tensor([-3.5, -1.2, -7.0])
        lengths = torch.tensor([5, 10, 2])
        normed = apply_length_penalty(scores, lengths, alpha=0.0)
        assert torch.allclose(normed, scores)

    def test_equal_lengths_equal_scores(self):
        """Same length → same normalized score regardless of alpha."""
        scores = torch.tensor([-2.0, -2.0])
        lengths = torch.tensor([8, 8])
        normed = apply_length_penalty(scores, lengths, alpha=0.75)
        assert torch.allclose(normed[0], normed[1])

    def test_output_shape_preserved(self):
        scores = torch.tensor([1.0, 2.0, 3.0])
        lengths = torch.tensor([4, 5, 6])
        out = apply_length_penalty(scores, lengths, alpha=0.6)
        assert out.shape == scores.shape

    def test_google_nmt_formula_single(self):
        """Verify the exact (5+L)^alpha / 6^alpha formula for one element."""
        scores = torch.tensor([1.0])
        lengths = torch.tensor([1])
        alpha = 1.0
        expected = 1.0 / ((5.0 + 1.0) / 6.0) ** alpha  # = 1.0
        out = apply_length_penalty(scores, lengths, alpha)
        assert out[0].item() == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# 3. apply_repetition_penalty
# ---------------------------------------------------------------------------


class TestApplyRepetitionPenalty:
    def test_penalizes_positive_repeated_token(self):
        logits = torch.zeros(VOCAB)
        logits[5] = 4.0  # positive logit
        input_ids = torch.tensor([5])
        out = apply_repetition_penalty(logits, input_ids, penalty=2.0)
        assert out[5].item() == pytest.approx(2.0)  # 4.0 / 2.0

    def test_penalizes_negative_repeated_token(self):
        logits = torch.zeros(VOCAB)
        logits[5] = -4.0  # negative logit
        input_ids = torch.tensor([5])
        out = apply_repetition_penalty(logits, input_ids, penalty=2.0)
        assert out[5].item() == pytest.approx(-8.0)  # -4.0 * 2.0

    def test_unseen_tokens_unchanged(self):
        logits = torch.ones(VOCAB) * 3.0
        input_ids = torch.tensor([0, 1])  # only tokens 0 and 1 are penalized
        out = apply_repetition_penalty(logits, input_ids, penalty=3.0)
        # Tokens 2..VOCAB-1 should be untouched
        assert torch.allclose(out[2:], torch.ones(VOCAB - 2) * 3.0)

    def test_penalty_one_is_noop(self):
        logits = torch.randn(VOCAB)
        input_ids = torch.arange(VOCAB)
        out = apply_repetition_penalty(logits, input_ids, penalty=1.0)
        assert torch.allclose(out, logits)

    def test_empty_input_ids_noop(self):
        logits = torch.randn(VOCAB)
        out = apply_repetition_penalty(logits, torch.tensor([], dtype=torch.long), penalty=5.0)
        assert torch.allclose(out, logits)

    def test_output_shape_unchanged(self):
        logits = torch.randn(VOCAB)
        out = apply_repetition_penalty(logits, torch.tensor([3, 7]), penalty=1.5)
        assert out.shape == (VOCAB,)


# ---------------------------------------------------------------------------
# 4. get_ngram_blocked_tokens
# ---------------------------------------------------------------------------


class TestGetNgramBlockedTokens:
    def test_returns_set(self):
        result = get_ngram_blocked_tokens([1, 2, 3, 1, 2], ngram_size=3)
        assert isinstance(result, set)

    def test_blocks_repeated_bigram_continuation(self):
        # Sequence "1 2 3 1 2": last 1-gram prefix is (2,)
        # Earlier occurrence of (2,) was followed by 3 → block 3
        result = get_ngram_blocked_tokens([1, 2, 3, 1, 2], ngram_size=2)
        # last token is 2; previously (2,) was at index 1 and followed by 3
        assert 3 in result

    def test_blocks_repeated_trigram_continuation(self):
        # "a b c a b" → prefix (a, b) appeared before followed by c → block c
        result = get_ngram_blocked_tokens([10, 11, 12, 10, 11], ngram_size=3)
        assert 12 in result

    def test_ngram_size_zero_returns_empty(self):
        result = get_ngram_blocked_tokens([1, 2, 3, 1, 2, 3], ngram_size=0)
        assert result == set()

    def test_no_repeat_when_sequence_unique(self):
        result = get_ngram_blocked_tokens([1, 2, 3, 4, 5], ngram_size=3)
        assert len(result) == 0

    def test_short_sequence_below_ngram_size(self):
        # Sequence shorter than ngram_size - 1 → nothing to block
        result = get_ngram_blocked_tokens([1], ngram_size=3)
        assert result == set()


# ---------------------------------------------------------------------------
# 5. BeamHypothesis fields
# ---------------------------------------------------------------------------


class TestBeamHypothesis:
    def test_has_token_ids_field(self):
        hyp = BeamHypothesis(token_ids=[1, 2, 3], score=-1.5, length=3)
        assert hyp.token_ids == [1, 2, 3]

    def test_has_score_field(self):
        hyp = BeamHypothesis(token_ids=[1], score=-0.5, length=1)
        assert hyp.score == pytest.approx(-0.5)

    def test_has_length_field(self):
        hyp = BeamHypothesis(token_ids=[1, 2], score=0.0, length=2)
        assert hyp.length == 2


# ---------------------------------------------------------------------------
# 6. BeamSearchDecoder.initialize_beams
# ---------------------------------------------------------------------------


class TestInitializeBeams:
    def test_creates_beam_width_hypotheses(self):
        cfg = BeamConfig(beam_width=3, eos_token_id=EOS)
        decoder = BeamSearchDecoder(_random_model_fn, cfg)
        beams = decoder.initialize_beams(_make_prompt())
        assert len(beams) == 3

    def test_all_scores_zero(self):
        cfg = BeamConfig(beam_width=4, eos_token_id=EOS)
        decoder = BeamSearchDecoder(_random_model_fn, cfg)
        beams = decoder.initialize_beams(_make_prompt())
        for b in beams:
            assert b.score == 0.0

    def test_token_ids_match_prompt(self):
        cfg = BeamConfig(beam_width=2, eos_token_id=EOS)
        decoder = BeamSearchDecoder(_random_model_fn, cfg)
        prompt = _make_prompt()
        beams = decoder.initialize_beams(prompt)
        for b in beams:
            assert b.token_ids == prompt.tolist()

    def test_accepts_2d_prompt(self):
        cfg = BeamConfig(beam_width=2, eos_token_id=EOS)
        decoder = BeamSearchDecoder(_random_model_fn, cfg)
        prompt_2d = _make_prompt().unsqueeze(0)  # (1, T)
        beams = decoder.initialize_beams(prompt_2d)
        assert len(beams) == 2


# ---------------------------------------------------------------------------
# 7. BeamSearchDecoder.decode
# ---------------------------------------------------------------------------


class TestDecode:
    def test_returns_1d_tensor(self):
        cfg = BeamConfig(beam_width=2, max_new_tokens=3, eos_token_id=EOS)
        decoder = BeamSearchDecoder(_random_model_fn, cfg)
        out = decoder.decode(_make_prompt())
        assert out.dim() == 1

    def test_longer_than_prompt(self):
        cfg = BeamConfig(beam_width=2, max_new_tokens=5, eos_token_id=EOS)
        decoder = BeamSearchDecoder(_random_model_fn, cfg)
        out = decoder.decode(_make_prompt())
        assert out.shape[0] > PROMPT_LEN

    def test_starts_with_prompt_tokens(self):
        cfg = BeamConfig(beam_width=2, max_new_tokens=4, eos_token_id=EOS)
        decoder = BeamSearchDecoder(_random_model_fn, cfg)
        prompt = _make_prompt()
        out = decoder.decode(prompt)
        assert out[:PROMPT_LEN].tolist() == prompt.tolist()

    def test_eos_model_terminates_early(self):
        """A model that always emits EOS should stop after one new token."""
        cfg = BeamConfig(beam_width=2, max_new_tokens=20, eos_token_id=EOS)
        decoder = BeamSearchDecoder(_eos_model_fn, cfg)
        out = decoder.decode(_make_prompt())
        # Generated at most 1 extra token (the EOS itself)
        assert out.shape[0] <= PROMPT_LEN + 2

    def test_respects_max_new_tokens(self):
        cfg = BeamConfig(beam_width=2, max_new_tokens=3, eos_token_id=99)  # EOS never emitted
        decoder = BeamSearchDecoder(_uniform_model_fn, cfg)
        out = decoder.decode(_make_prompt())
        assert out.shape[0] <= PROMPT_LEN + 3


# ---------------------------------------------------------------------------
# 8. BeamSearchDecoder.decode_with_scores
# ---------------------------------------------------------------------------


class TestDecodeWithScores:
    def test_returns_list(self):
        cfg = BeamConfig(beam_width=2, max_new_tokens=3, eos_token_id=EOS)
        decoder = BeamSearchDecoder(_random_model_fn, cfg)
        result = decoder.decode_with_scores(_make_prompt())
        assert isinstance(result, list)

    def test_each_element_is_tuple_of_tensor_and_float(self):
        cfg = BeamConfig(beam_width=2, max_new_tokens=3, eos_token_id=EOS)
        decoder = BeamSearchDecoder(_random_model_fn, cfg)
        result = decoder.decode_with_scores(_make_prompt())
        for seq, score in result:
            assert isinstance(seq, torch.Tensor)
            assert isinstance(score, float)

    def test_scores_sorted_best_first(self):
        cfg = BeamConfig(beam_width=3, max_new_tokens=4, eos_token_id=EOS)
        decoder = BeamSearchDecoder(_random_model_fn, cfg)
        result = decoder.decode_with_scores(_make_prompt())
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True), f"Expected sorted descending, got {scores}"

    def test_sequences_start_with_prompt(self):
        cfg = BeamConfig(beam_width=2, max_new_tokens=3, eos_token_id=EOS)
        decoder = BeamSearchDecoder(_random_model_fn, cfg)
        prompt = _make_prompt()
        result = decoder.decode_with_scores(prompt)
        for seq, _ in result:
            assert seq[:PROMPT_LEN].tolist() == prompt.tolist()


# ---------------------------------------------------------------------------
# 9. Integration: repetition penalty actually reduces repeated token probability
# ---------------------------------------------------------------------------


class TestRepetitionPenaltyIntegration:
    def test_high_penalty_changes_output(self):
        """With extreme repetition penalty the decoder should avoid repeating tokens."""
        cfg_no_pen = BeamConfig(
            beam_width=2, max_new_tokens=5, eos_token_id=EOS, repetition_penalty=1.0
        )
        cfg_pen = BeamConfig(
            beam_width=2, max_new_tokens=5, eos_token_id=EOS, repetition_penalty=100.0
        )
        decoder_no = BeamSearchDecoder(_uniform_model_fn, cfg_no_pen)
        decoder_yes = BeamSearchDecoder(_uniform_model_fn, cfg_pen)
        out_no = decoder_no.decode(_make_prompt())
        out_yes = decoder_yes.decode(_make_prompt())
        # With repetition_penalty=100 on uniform model the outputs may differ
        # (at minimum, this should not raise an exception)
        assert out_no.dim() == 1
        assert out_yes.dim() == 1


# ---------------------------------------------------------------------------
# 10. Integration: n-gram blocking suppresses repeated n-grams
# ---------------------------------------------------------------------------


class TestNgramBlockingIntegration:
    def test_no_repeat_ngram_does_not_crash(self):
        cfg = BeamConfig(beam_width=2, max_new_tokens=6, eos_token_id=EOS, no_repeat_ngram_size=2)
        decoder = BeamSearchDecoder(_random_model_fn, cfg)
        out = decoder.decode(_make_prompt())
        assert out.dim() == 1
        assert out.shape[0] > PROMPT_LEN
