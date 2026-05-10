"""Tests for beam_search_v4 — BeamSearchConfig, Beam, BeamSearch, BeamSearchDecoder.

Import path: aurelius.inference.beam_search_v4
"""

from __future__ import annotations

import math

import torch
from src.inference.beam_search_v4 import (
    Beam,
    BeamSearch,
    BeamSearchConfig,
    BeamSearchDecoder,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

VOCAB_SIZE = 64
PROMPT_LEN = 3


def make_prompt(length: int = PROMPT_LEN) -> torch.LongTensor:
    """Return a (1, length) prompt tensor."""
    return torch.arange(1, length + 1, dtype=torch.long).unsqueeze(0)


def uniform_model_fn(input_ids: torch.LongTensor) -> torch.FloatTensor:
    """Model that returns uniform logits — every token equally likely."""
    B, T = input_ids.shape
    return torch.zeros(B, T, VOCAB_SIZE)


def deterministic_model_fn(input_ids: torch.LongTensor) -> torch.FloatTensor:
    """Model that always strongly prefers token 5."""
    B, T = input_ids.shape
    logits = torch.zeros(B, T, VOCAB_SIZE)
    logits[:, :, 5] = 100.0  # overwhelmingly prefer token 5
    return logits


def eos_always_model_fn(eos_id: int = 2):
    """Return a model_fn that always strongly prefers the EOS token."""

    def _fn(input_ids: torch.LongTensor) -> torch.FloatTensor:
        B, T = input_ids.shape
        logits = torch.full((B, T, VOCAB_SIZE), -100.0)
        logits[:, :, eos_id] = 100.0
        return logits

    return _fn


def make_config(**kwargs) -> BeamSearchConfig:
    defaults = dict(
        beam_size=4,
        max_new_tokens=10,
        length_penalty=1.0,
        eos_token_id=2,
        min_length=1,
    )
    defaults.update(kwargs)
    return BeamSearchConfig(**defaults)


# ---------------------------------------------------------------------------
# 1. BeamSearchConfig defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    """Config defaults should be sane."""
    cfg = BeamSearchConfig()
    assert cfg.beam_size == 4
    assert cfg.max_new_tokens == 50
    assert cfg.length_penalty == 1.0
    assert cfg.eos_token_id == 2
    assert cfg.min_length == 1


# ---------------------------------------------------------------------------
# 2. Beam.length
# ---------------------------------------------------------------------------


def test_beam_length_counts_tokens():
    b = Beam(token_ids=[10, 20, 30], score=-3.0)
    assert b.length == 3


def test_beam_length_empty():
    b = Beam(token_ids=[], score=0.0)
    assert b.length == 0


# ---------------------------------------------------------------------------
# 3. Beam.normalized_score
# ---------------------------------------------------------------------------


def test_beam_normalized_score_alpha_one():
    """normalized_score(alpha=1.0) == score / length."""
    b = Beam(token_ids=[1, 2, 3], score=-6.0)
    assert math.isclose(b.normalized_score(1.0), -6.0 / 3, rel_tol=1e-6)


def test_beam_normalized_score_alpha_zero():
    """normalized_score(alpha=0.0) == score (length^0 = 1 for any length)."""
    b = Beam(token_ids=[1, 2, 3, 4, 5], score=-10.0)
    assert math.isclose(b.normalized_score(0.0), -10.0, rel_tol=1e-6)


def test_beam_normalized_score_alpha_two():
    """normalized_score(alpha=2.0) == score / length^2."""
    b = Beam(token_ids=[1, 2, 3], score=-9.0)
    expected = -9.0 / (3**2)
    assert math.isclose(b.normalized_score(2.0), expected, rel_tol=1e-6)


def test_beam_normalized_score_empty_no_error():
    """Empty beam should not raise, returns raw score."""
    b = Beam(token_ids=[], score=-1.0)
    assert math.isfinite(b.normalized_score(1.0))
    assert math.isclose(b.normalized_score(1.0), -1.0, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# 4. Beam.extend
# ---------------------------------------------------------------------------


def test_beam_extend_appends_token():
    b = Beam(token_ids=[1, 2], score=-2.0)
    new_b = b.extend(token_id=7, log_prob=-1.5)
    assert new_b.token_ids == [1, 2, 7]


def test_beam_extend_updates_score():
    b = Beam(token_ids=[1, 2], score=-2.0)
    new_b = b.extend(token_id=7, log_prob=-1.5)
    assert math.isclose(new_b.score, -3.5, rel_tol=1e-6)


def test_beam_extend_does_not_mutate_original():
    b = Beam(token_ids=[1, 2], score=-2.0)
    _ = b.extend(token_id=7, log_prob=-0.5)
    assert b.token_ids == [1, 2]
    assert math.isclose(b.score, -2.0)


def test_beam_extend_returns_new_beam_instance():
    b = Beam(token_ids=[1], score=-1.0)
    new_b = b.extend(3, -0.5)
    assert new_b is not b


# ---------------------------------------------------------------------------
# 5. BeamSearch.search — structural properties
# ---------------------------------------------------------------------------


def test_search_returns_list_of_beams():
    cfg = make_config(beam_size=3, max_new_tokens=5)
    bs = BeamSearch(uniform_model_fn, VOCAB_SIZE, cfg)
    result = bs.search(make_prompt())
    assert isinstance(result, list)
    assert all(isinstance(b, Beam) for b in result)


def test_search_all_token_ids_in_vocab_range():
    cfg = make_config(beam_size=4, max_new_tokens=5)
    bs = BeamSearch(uniform_model_fn, VOCAB_SIZE, cfg)
    prompt = make_prompt()
    result = bs.search(prompt)
    for beam in result:
        for tid in beam.token_ids:
            assert 0 <= tid < VOCAB_SIZE, f"token {tid} out of vocab range"


def test_search_eos_always_finishes_after_one_token():
    """With a model that always predicts EOS, all beams finish after 1 generated token."""
    eos_id = 2
    cfg = make_config(beam_size=4, max_new_tokens=20, eos_token_id=eos_id, min_length=1)
    bs = BeamSearch(eos_always_model_fn(eos_id), VOCAB_SIZE, cfg)
    result = bs.search(make_prompt())
    # All finished beams should end with EOS.
    for beam in result:
        assert beam.token_ids[-1] == eos_id


def test_search_deterministic_model_top_beam_has_highest_score():
    """With deterministic model preferring token 5, top beam should have highest score."""
    cfg = make_config(beam_size=4, max_new_tokens=5)
    bs = BeamSearch(deterministic_model_fn, VOCAB_SIZE, cfg)
    result = bs.search(make_prompt())
    assert len(result) >= 1
    top_score = result[0].normalized_score(cfg.length_penalty)
    for beam in result[1:]:
        assert beam.normalized_score(cfg.length_penalty) <= top_score + 1e-9


def test_search_result_sorted_by_normalized_score():
    """Returned beams should be sorted descending by normalized_score."""
    cfg = make_config(beam_size=4, max_new_tokens=5)
    bs = BeamSearch(uniform_model_fn, VOCAB_SIZE, cfg)
    result = bs.search(make_prompt())
    scores = [b.normalized_score(cfg.length_penalty) for b in result]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# 6. length_penalty=0 makes all beams have equal length penalty (1^0 = 1)
# ---------------------------------------------------------------------------


def test_length_penalty_zero_all_equal_denominator():
    """With alpha=0, normalized_score == raw score for all lengths."""
    b_short = Beam(token_ids=[1, 2], score=-4.0)
    b_long = Beam(token_ids=[1, 2, 3, 4, 5, 6], score=-4.0)
    assert math.isclose(
        b_short.normalized_score(0.0),
        b_long.normalized_score(0.0),
        rel_tol=1e-6,
    )


# ---------------------------------------------------------------------------
# 7. beam_size=1 is equivalent to greedy search
# ---------------------------------------------------------------------------


def test_beam_size_one_is_greedy():
    """beam_size=1 should always pick the argmax at each step."""
    cfg = make_config(beam_size=1, max_new_tokens=5, eos_token_id=999)
    bs = BeamSearch(deterministic_model_fn, VOCAB_SIZE, cfg)
    result = bs.search(make_prompt())
    assert len(result) == 1
    # With deterministic model preferring token 5, all new tokens should be 5.
    prompt_len = PROMPT_LEN
    generated = result[0].token_ids[prompt_len:]
    assert all(t == 5 for t in generated)


# ---------------------------------------------------------------------------
# 8. BeamSearchDecoder.decode
# ---------------------------------------------------------------------------


def test_decoder_decode_returns_longtensor():
    cfg = make_config(beam_size=2, max_new_tokens=5)
    decoder = BeamSearchDecoder(uniform_model_fn, VOCAB_SIZE, cfg)
    out = decoder.decode(make_prompt())
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.long


def test_decoder_decode_length_le_max_new_tokens():
    max_new = 7
    cfg = make_config(beam_size=2, max_new_tokens=max_new, eos_token_id=999)
    decoder = BeamSearchDecoder(uniform_model_fn, VOCAB_SIZE, cfg)
    out = decoder.decode(make_prompt())
    assert out.shape[0] <= max_new


def test_decoder_decode_excludes_prompt():
    """decode() should return only the generated tokens, not the prompt."""
    cfg = make_config(beam_size=2, max_new_tokens=5, eos_token_id=999)
    decoder = BeamSearchDecoder(uniform_model_fn, VOCAB_SIZE, cfg)
    prompt = make_prompt(PROMPT_LEN)
    out = decoder.decode(prompt)
    # Output should NOT start with prompt tokens (1, 2, 3).
    assert out.tolist()[:PROMPT_LEN] != list(range(1, PROMPT_LEN + 1))


def test_decoder_decode_max_new_tokens_override():
    """Passing max_new_tokens to decode() overrides the config value."""
    cfg = make_config(beam_size=2, max_new_tokens=50, eos_token_id=999)
    decoder = BeamSearchDecoder(uniform_model_fn, VOCAB_SIZE, cfg)
    out = decoder.decode(make_prompt(), max_new_tokens=3)
    assert out.shape[0] <= 3
