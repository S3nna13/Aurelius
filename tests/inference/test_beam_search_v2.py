"""Tests for src/inference/beam_search_v2.py"""

import pytest
import torch

from src.inference.beam_search_v2 import (
    BeamSearch,
    BeamSearchConfig,
    get_ngram_blocked_tokens,
    normalize_score,
    top_p_filter,
)

VOCAB_SIZE = 32
BEAM = 2
PROMPT_LEN = 4


def _mock_model(ids: torch.Tensor) -> torch.Tensor:
    torch.manual_seed(7)
    return torch.randn(ids.shape[0], ids.shape[1], VOCAB_SIZE)


def _prompt() -> torch.Tensor:
    return torch.randint(0, VOCAB_SIZE, (1, PROMPT_LEN))


# ---------------------------------------------------------------------------
# BeamSearchConfig
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = BeamSearchConfig()
    assert cfg.beam_width == 4
    assert cfg.max_new_tokens == 50
    assert cfg.length_penalty == 1.0
    assert cfg.eos_token_id is None
    assert cfg.no_repeat_ngram_size == 0
    assert cfg.temperature == 1.0


# ---------------------------------------------------------------------------
# normalize_score
# ---------------------------------------------------------------------------


def test_normalize_score_lp_one():
    assert normalize_score(4.0, 4, 1.0) == pytest.approx(1.0)


def test_normalize_score_lp_zero_returns_raw():
    assert normalize_score(4.0, 4, 0.0) == pytest.approx(4.0)


def test_normalize_score_length_zero():
    assert normalize_score(3.0, 0, 1.0) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# get_ngram_blocked_tokens
# ---------------------------------------------------------------------------


def test_ngram_blocked_empty_when_size_zero():
    ids = torch.tensor([1, 2, 3, 1, 2])
    assert get_ngram_blocked_tokens(ids, 0) == set()


def test_ngram_blocked_finds_repeat():
    # sequence "1 2 3 1 2" — if prefix is (1, 2) → next should block 3
    ids = torch.tensor([1, 2, 3, 1, 2])
    blocked = get_ngram_blocked_tokens(ids, 3)
    assert 3 in blocked


def test_ngram_blocked_no_repeat():
    ids = torch.tensor([1, 2, 3, 4, 5])
    blocked = get_ngram_blocked_tokens(ids, 3)
    assert len(blocked) == 0


# ---------------------------------------------------------------------------
# top_p_filter
# ---------------------------------------------------------------------------


def test_top_p_filter_shape():
    logits = torch.randn(VOCAB_SIZE)
    out = top_p_filter(logits, 0.9)
    assert out.shape == (VOCAB_SIZE,)


def test_top_p_filter_p1_unchanged():
    logits = torch.randn(VOCAB_SIZE)
    out = top_p_filter(logits, 1.0)
    assert torch.allclose(out, logits)


def test_top_p_filter_p0_mostly_masked():
    logits = torch.randn(VOCAB_SIZE)
    out = top_p_filter(logits, 0.0)
    # All but the top token should be -inf
    finite = torch.isfinite(out).sum().item()
    assert finite <= 2  # at most the top token (possibly 1)


# ---------------------------------------------------------------------------
# BeamSearch.initialize_beams
# ---------------------------------------------------------------------------


def test_initialize_beams_count():
    cfg = BeamSearchConfig(beam_width=BEAM)
    bs = BeamSearch(_mock_model, cfg)
    beams = bs.initialize_beams(_prompt())
    assert len(beams) == BEAM


def test_initialize_beams_all_score_zero():
    cfg = BeamSearchConfig(beam_width=BEAM)
    bs = BeamSearch(_mock_model, cfg)
    beams = bs.initialize_beams(_prompt())
    for b in beams:
        assert b.score == 0.0


# ---------------------------------------------------------------------------
# BeamSearch.expand_beams
# ---------------------------------------------------------------------------


def test_expand_beams_returns_list():
    cfg = BeamSearchConfig(beam_width=BEAM, max_new_tokens=3)
    bs = BeamSearch(_mock_model, cfg)
    beams = bs.initialize_beams(_prompt())
    new_beams, done = bs.expand_beams(beams)
    assert isinstance(new_beams, list)
    assert len(new_beams) == BEAM


# ---------------------------------------------------------------------------
# BeamSearch.search
# ---------------------------------------------------------------------------


def test_search_output_is_1d():
    cfg = BeamSearchConfig(beam_width=BEAM, max_new_tokens=3)
    bs = BeamSearch(_mock_model, cfg)
    out = bs.search(_prompt())
    assert out.dim() == 1


def test_search_output_longer_than_prompt():
    cfg = BeamSearchConfig(beam_width=BEAM, max_new_tokens=5)
    bs = BeamSearch(_mock_model, cfg)
    out = bs.search(_prompt())
    assert out.shape[0] > PROMPT_LEN


def test_search_with_scores_sorted():
    cfg = BeamSearchConfig(beam_width=BEAM, max_new_tokens=3)
    bs = BeamSearch(_mock_model, cfg)
    hyps = bs.search_with_scores(_prompt())
    assert len(hyps) == BEAM
    scores = [normalize_score(h.score, h.token_ids.shape[0], cfg.length_penalty) for h in hyps]
    assert scores == sorted(scores, reverse=True)


def test_search_with_scores_all_have_scores():
    cfg = BeamSearchConfig(beam_width=BEAM, max_new_tokens=3)
    bs = BeamSearch(_mock_model, cfg)
    hyps = bs.search_with_scores(_prompt())
    for h in hyps:
        assert isinstance(h.score, float)


def test_eos_token_stops_beam():
    # Model that always outputs the same token (EOS)
    EOS = 5

    def eos_model(ids):
        logits = torch.full((ids.shape[0], ids.shape[1], VOCAB_SIZE), -1e9)
        logits[:, :, EOS] = 10.0
        return logits

    cfg = BeamSearchConfig(beam_width=2, max_new_tokens=10, eos_token_id=EOS)
    bs = BeamSearch(eos_model, cfg)
    out = bs.search(_prompt())
    # Should terminate quickly after generating EOS
    assert out.shape[0] <= PROMPT_LEN + 2  # at most 1-2 EOS tokens
