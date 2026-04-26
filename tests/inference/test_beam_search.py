"""Tests for beam search decoding (src/inference/beam_search.py)."""

from __future__ import annotations

import math

import pytest
import torch

from src.inference.beam_search import (
    Beam,
    BeamSearchConfig,
    BeamSearchDecoder,
    apply_no_repeat_ngram,
    apply_repetition_penalty,
    beam_search_step,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_model():
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )
    torch.manual_seed(42)
    model = AureliusTransformer(cfg)
    model.eval()
    return model


@pytest.fixture
def default_config():
    return BeamSearchConfig()


# ---------------------------------------------------------------------------
# 1. BeamSearchConfig defaults
# ---------------------------------------------------------------------------


def test_beam_search_config_defaults():
    cfg = BeamSearchConfig()
    assert cfg.beam_width == 4
    assert cfg.max_new_tokens == 50
    assert cfg.length_penalty == 1.0
    assert cfg.repetition_penalty == 1.0
    assert cfg.no_repeat_ngram_size == 0
    assert cfg.early_stopping is True


def test_beam_search_config_custom():
    cfg = BeamSearchConfig(beam_width=8, max_new_tokens=20, length_penalty=0.6)
    assert cfg.beam_width == 8
    assert cfg.max_new_tokens == 20
    assert cfg.length_penalty == 0.6


# ---------------------------------------------------------------------------
# 2. Beam score computation with length penalty
# ---------------------------------------------------------------------------


def test_beam_score_default_length_penalty():
    """With length_penalty=1.0 score = log_prob / len."""
    beam = Beam(token_ids=[1, 2, 3], log_prob=-3.0, _length_penalty=1.0)
    assert math.isclose(beam.score, -3.0 / 3.0, rel_tol=1e-6)


def test_beam_score_zero_length_penalty():
    """With length_penalty=0.0 score = log_prob / 1 = log_prob."""
    beam = Beam(token_ids=[1, 2, 3, 4, 5], log_prob=-5.0, _length_penalty=0.0)
    assert math.isclose(beam.score, -5.0, rel_tol=1e-6)


def test_beam_score_large_length_penalty():
    """With length_penalty=2.0 score = log_prob / len^2."""
    beam = Beam(token_ids=[1, 2, 3], log_prob=-9.0, _length_penalty=2.0)
    expected = -9.0 / (3**2)
    assert math.isclose(beam.score, expected, rel_tol=1e-6)


def test_beam_score_empty_token_ids():
    """Empty token list should not raise ZeroDivisionError."""
    beam = Beam(token_ids=[], log_prob=-1.0, _length_penalty=1.0)
    assert math.isfinite(beam.score)


def test_beam_score_length_penalty_ordering():
    """Longer beams should be penalised more with length_penalty=1.0."""
    short = Beam(token_ids=[1, 2], log_prob=-2.0, _length_penalty=1.0)
    long_ = Beam(token_ids=[1, 2, 3, 4], log_prob=-4.0, _length_penalty=1.0)
    # Both have the same avg log-prob per token; equal scores
    assert math.isclose(short.score, long_.score, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# 3. apply_repetition_penalty
# ---------------------------------------------------------------------------


def test_repetition_penalty_no_op_when_one():
    logits = torch.tensor([1.0, 2.0, 3.0])
    out = apply_repetition_penalty(logits, [0, 1], penalty=1.0)
    assert torch.allclose(out, logits)


def test_repetition_penalty_divides_positive():
    logits = torch.tensor([2.0, 4.0, 6.0])
    out = apply_repetition_penalty(logits, [1], penalty=2.0)
    # Token 1 has positive logit 4.0 → 4.0/2 = 2.0
    assert math.isclose(out[1].item(), 2.0, rel_tol=1e-6)
    assert math.isclose(out[0].item(), 2.0, rel_tol=1e-6)  # unchanged
    assert math.isclose(out[2].item(), 6.0, rel_tol=1e-6)  # unchanged


def test_repetition_penalty_multiplies_negative():
    logits = torch.tensor([-2.0, -4.0, 1.0])
    out = apply_repetition_penalty(logits, [1], penalty=2.0)
    # Token 1 has negative logit -4.0 → -4.0*2 = -8.0
    assert math.isclose(out[1].item(), -8.0, rel_tol=1e-6)


def test_repetition_penalty_empty_token_ids():
    logits = torch.tensor([1.0, 2.0, 3.0])
    out = apply_repetition_penalty(logits, [], penalty=2.0)
    assert torch.allclose(out, logits)


def test_repetition_penalty_does_not_mutate_input():
    logits = torch.tensor([1.0, 2.0, 3.0])
    original = logits.clone()
    apply_repetition_penalty(logits, [0], penalty=2.0)
    assert torch.allclose(logits, original)


# ---------------------------------------------------------------------------
# 4. apply_no_repeat_ngram
# ---------------------------------------------------------------------------


def test_no_repeat_ngram_zero_is_noop():
    logits = torch.zeros(10)
    out = apply_no_repeat_ngram(logits, [1, 2, 3], n=0)
    assert torch.allclose(out, logits)


def test_no_repeat_ngram_blocks_continuation():
    """If [1, 2] appeared before and current suffix is [1], ban token 2."""
    logits = torch.zeros(10)
    token_ids = [1, 2, 3, 1]  # bigram (1,2) already seen; last token is 1
    out = apply_no_repeat_ngram(logits, token_ids, n=2)
    assert out[2].item() == float("-inf")
    # Token 3 was also preceded by 2, but current prefix is 1 — token 3 not banned
    assert out[3].item() == 0.0


def test_no_repeat_ngram_short_sequence_noop():
    """Sequence shorter than n-1 should not modify logits."""
    logits = torch.zeros(10)
    out = apply_no_repeat_ngram(logits, [1], n=3)
    assert torch.allclose(out, logits)


def test_no_repeat_ngram_does_not_mutate_input():
    logits = torch.zeros(10)
    original = logits.clone()
    apply_no_repeat_ngram(logits, [1, 2, 1], n=2)
    assert torch.allclose(logits, original)


# ---------------------------------------------------------------------------
# 5. beam_search_step
# ---------------------------------------------------------------------------


def test_beam_search_step_returns_beam_width(small_model):
    cfg = BeamSearchConfig(beam_width=3, max_new_tokens=5)
    beams = [Beam(token_ids=[0, 1, 2], log_prob=0.0, _length_penalty=1.0)]
    new_beams = beam_search_step(small_model, beams, vocab_size=256, eos_token_id=None, config=cfg)
    assert len(new_beams) == 3


def test_beam_search_step_sorted_by_score(small_model):
    cfg = BeamSearchConfig(beam_width=4)
    beams = [Beam(token_ids=[5, 10], log_prob=0.0, _length_penalty=1.0)]
    new_beams = beam_search_step(small_model, beams, vocab_size=256, eos_token_id=None, config=cfg)
    scores = [b.score for b in new_beams]
    assert scores == sorted(scores, reverse=True)


def test_beam_search_step_eos_marks_done(small_model):
    """A beam that produces EOS should be marked is_done=True."""
    cfg = BeamSearchConfig(beam_width=256)  # wide beam to guarantee EOS appears
    beams = [Beam(token_ids=[1, 2], log_prob=0.0, _length_penalty=1.0)]
    new_beams = beam_search_step(small_model, beams, vocab_size=256, eos_token_id=5, config=cfg)
    # At least one beam that ends with token 5 should be marked done
    done_beams = [b for b in new_beams if b.is_done]
    ending_5 = [b for b in new_beams if b.token_ids[-1] == 5]
    assert set(b.score for b in done_beams) == set(b.score for b in ending_5)


# ---------------------------------------------------------------------------
# 6. BeamSearchDecoder.generate
# ---------------------------------------------------------------------------


def test_generate_returns_tensor(small_model):
    decoder = BeamSearchDecoder(BeamSearchConfig(beam_width=2, max_new_tokens=5))
    input_ids = torch.randint(0, 256, (1, 4))
    out = decoder.generate(small_model, input_ids)
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.long


def test_generate_length_at_least_prompt_plus_one(small_model):
    prompt_len = 4
    decoder = BeamSearchDecoder(BeamSearchConfig(beam_width=2, max_new_tokens=5))
    input_ids = torch.randint(0, 256, (1, prompt_len))
    out = decoder.generate(small_model, input_ids)
    assert out.shape[0] >= prompt_len + 1


def test_generate_starts_with_prompt(small_model):
    """Output should start with the prompt tokens."""
    input_ids = torch.tensor([[10, 20, 30, 40]])
    decoder = BeamSearchDecoder(BeamSearchConfig(beam_width=2, max_new_tokens=3))
    out = decoder.generate(small_model, input_ids)
    assert out[:4].tolist() == [10, 20, 30, 40]


def test_generate_max_new_tokens_respected(small_model):
    max_new = 6
    prompt_len = 3
    decoder = BeamSearchDecoder(
        BeamSearchConfig(beam_width=2, max_new_tokens=max_new, early_stopping=False)
    )
    input_ids = torch.randint(0, 256, (1, prompt_len))
    out = decoder.generate(small_model, input_ids)
    assert out.shape[0] <= prompt_len + max_new


# ---------------------------------------------------------------------------
# 7. length_penalty=0 vs length_penalty=1.0 differences
# ---------------------------------------------------------------------------


def test_length_penalty_zero_vs_one_different_scores():
    """Beams with different lengths should rank differently under LP=0 vs LP=1."""
    short = Beam(token_ids=[1, 2], log_prob=-2.0, _length_penalty=1.0)
    long_ = Beam(token_ids=[1, 2, 3, 4, 5], log_prob=-3.0, _length_penalty=1.0)
    # With LP=1: short = -1.0, long = -0.6  → long wins
    assert long_.score > short.score

    short0 = Beam(token_ids=[1, 2], log_prob=-2.0, _length_penalty=0.0)
    long0 = Beam(token_ids=[1, 2, 3, 4, 5], log_prob=-3.0, _length_penalty=0.0)
    # With LP=0: short = -2.0, long = -3.0  → short wins
    assert short0.score > long0.score


# ---------------------------------------------------------------------------
# 8. Early stopping when all beams hit EOS
# ---------------------------------------------------------------------------


def test_early_stopping_all_eos(small_model):
    """With early_stopping=True and an EOS that every beam hits quickly,
    generate() should stop before max_new_tokens steps."""
    # We will monkey-patch the model to always return EOS (token 0) as the
    # highest-logit token so all beams complete on the first step.
    import torch.nn as nn

    class AlwaysEOSModel(nn.Module):
        def __init__(self, vocab_size=256):
            super().__init__()
            self.vocab_size = vocab_size

        def forward(self, input_ids):
            B, S = input_ids.shape
            logits = torch.full((B, S, self.vocab_size), -100.0)
            logits[:, :, 0] = 100.0  # EOS token = 0 is always top
            return None, logits, []

    eos_model = AlwaysEOSModel()
    decoder = BeamSearchDecoder(
        BeamSearchConfig(beam_width=2, max_new_tokens=50, early_stopping=True)
    )
    input_ids = torch.tensor([[5, 6, 7]])
    out = decoder.generate(eos_model, input_ids, eos_token_id=0)
    # Output should end with EOS and be short (stopped early)
    assert out[-1].item() == 0
    assert out.shape[0] < 3 + 50  # well under max


# ---------------------------------------------------------------------------
# 9. Single beam (beam_width=1) is greedy search
# ---------------------------------------------------------------------------


def test_single_beam_is_greedy(small_model):
    """beam_width=1 should always pick the argmax at each step."""
    torch.manual_seed(0)
    prompt = torch.tensor([[1, 2, 3]])
    cfg = BeamSearchConfig(beam_width=1, max_new_tokens=5, early_stopping=False)
    decoder = BeamSearchDecoder(cfg)
    out = decoder.generate(small_model, prompt)

    # Reproduce greedy manually
    tokens = [1, 2, 3]
    with torch.no_grad():
        for _ in range(5):
            ids = torch.tensor([tokens])
            _loss, logits, _ = small_model(ids)
            next_tok = int(logits[0, -1, :].argmax().item())
            tokens.append(next_tok)

    assert out.tolist() == tokens
