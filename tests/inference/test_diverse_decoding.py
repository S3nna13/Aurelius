"""Tests for diverse_decoding module."""

import pytest
import torch

from src.inference.diverse_decoding import (
    BeamHypothesis,
    BeamSearchDecoder,
    DiverseBeamConfig,
    DiverseBeamSearchDecoder,
    length_penalty_rerank,
    stochastic_beam_search,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_model():
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
    torch.manual_seed(0)
    model = AureliusTransformer(cfg)
    model.train(False)
    return model


@pytest.fixture
def default_config():
    return DiverseBeamConfig(
        num_beams=4,
        num_beam_groups=2,
        max_new_tokens=4,
    )


@pytest.fixture
def input_ids():
    torch.manual_seed(0)
    return torch.randint(0, 256, (1, 4))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_diverse_beam_config_defaults():
    """DiverseBeamConfig should have correct default values."""
    cfg = DiverseBeamConfig()
    assert cfg.num_beams == 4
    assert cfg.num_beam_groups == 2
    assert cfg.diversity_penalty == 0.5
    assert cfg.length_penalty == 1.0
    assert cfg.min_length == 0
    assert cfg.max_new_tokens == 20
    assert cfg.temperature == 1.0


def test_beam_hypothesis_normalized_score():
    """normalized_score should apply length penalty correctly."""
    hyp = BeamHypothesis(token_ids=[1, 2, 3], score=-3.0, length=3)
    # With length_penalty=1.0: -3.0 / 3^1.0 = -1.0
    assert hyp.normalized_score(1.0) == pytest.approx(-1.0)
    # With length_penalty=0.0: -3.0 / 3^0.0 = -3.0 / 1.0 = -3.0
    assert hyp.normalized_score(0.0) == pytest.approx(-3.0)
    # With length_penalty=2.0: -3.0 / 9.0
    assert hyp.normalized_score(2.0) == pytest.approx(-3.0 / 9.0)


def test_beam_search_decoder_returns_hypotheses(tiny_model, default_config, input_ids):
    """BeamSearchDecoder.generate() should return a list of BeamHypothesis."""
    decoder = BeamSearchDecoder(tiny_model, default_config)
    result = decoder.generate(input_ids)
    assert isinstance(result, list)
    assert all(isinstance(h, BeamHypothesis) for h in result)


def test_beam_search_decoder_count(tiny_model, default_config, input_ids):
    """BeamSearchDecoder.generate() should return exactly num_beams hypotheses."""
    decoder = BeamSearchDecoder(tiny_model, default_config)
    result = decoder.generate(input_ids)
    assert len(result) == default_config.num_beams


def test_beam_search_decoder_token_ids_nonempty(tiny_model, default_config, input_ids):
    """Each BeamHypothesis should have non-empty token_ids."""
    decoder = BeamSearchDecoder(tiny_model, default_config)
    result = decoder.generate(input_ids)
    for hyp in result:
        assert len(hyp.token_ids) > 0


def test_diverse_beam_search_returns_hypotheses(tiny_model, default_config, input_ids):
    """DiverseBeamSearchDecoder.generate() should return a list of BeamHypothesis."""
    decoder = DiverseBeamSearchDecoder(tiny_model, default_config)
    result = decoder.generate(input_ids)
    assert isinstance(result, list)
    assert all(isinstance(h, BeamHypothesis) for h in result)


def test_diverse_beam_search_count(tiny_model, default_config, input_ids):
    """DiverseBeamSearchDecoder should return num_beams total hypotheses."""
    decoder = DiverseBeamSearchDecoder(tiny_model, default_config)
    result = decoder.generate(input_ids)
    assert len(result) == default_config.num_beams


def test_apply_diversity_penalty_reduces_logits(tiny_model, default_config):
    """_apply_diversity_penalty should lower logits for penalized tokens."""
    decoder = DiverseBeamSearchDecoder(tiny_model, default_config)
    vocab_size = 256
    logits = torch.zeros(vocab_size)
    previous_tokens = {0, 5, 42}
    penalized = decoder._apply_diversity_penalty(logits, previous_tokens)

    for tok in previous_tokens:
        assert penalized[tok].item() < logits[tok].item(), (
            f"Token {tok} should be penalized but wasn't"
        )
    # Non-penalized token should remain unchanged
    assert penalized[100].item() == logits[100].item()


def test_stochastic_beam_search_different_runs(tiny_model, input_ids):
    """Stochastic beam search should produce different results across runs (most of the time)."""
    results = []
    for seed in range(5):
        torch.manual_seed(seed)
        hyps = stochastic_beam_search(
            tiny_model, input_ids, num_beams=4, max_new_tokens=4, temperature=1.0
        )
        results.append(tuple(hyps[0].token_ids))

    # At least 2 distinct results across 5 different seeds
    unique_results = set(results)
    assert len(unique_results) >= 2, (
        "Stochastic beam search produced identical results for all seeds"
    )


def test_length_penalty_rerank_sorted(tiny_model, default_config, input_ids):
    """length_penalty_rerank should return hypotheses sorted descending by normalized_score."""
    decoder = BeamSearchDecoder(tiny_model, default_config)
    hyps = decoder.generate(input_ids)
    reranked = length_penalty_rerank(hyps, length_penalty=1.0)

    scores = [h.normalized_score(1.0) for h in reranked]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], (
            f"Scores not sorted at position {i}: {scores[i]} < {scores[i + 1]}"
        )


def test_length_penalty_rerank_length_effect():
    """Higher length_penalty with positive scores should favor shorter sequences."""
    # With positive scores and length_penalty > 0:
    # normalized_score = score / (length^lp)
    # longer sequences get divided by a larger number, so they rank lower
    short_hyp = BeamHypothesis(token_ids=[1], score=4.0, length=1)
    long_hyp = BeamHypothesis(token_ids=[1, 2, 3, 4], score=4.0, length=4)

    # With length_penalty=0.0: both get score/1 = 4.0, so tied (order arbitrary)
    # With length_penalty=1.0: short=4/1=4.0, long=4/4=1.0 -> short ranks first
    reranked_high = length_penalty_rerank([long_hyp, short_hyp], length_penalty=1.0)
    assert reranked_high[0].length == 1, (
        "Shorter sequence should rank first when length_penalty=1.0 and scores are equal"
    )

    # Verify lp=0.0 makes them equal (within float precision, both score 4.0)
    reranked_zero = length_penalty_rerank([short_hyp, long_hyp], length_penalty=0.0)
    assert reranked_zero[0].normalized_score(0.0) == pytest.approx(
        reranked_zero[-1].normalized_score(0.0)
    ), "All hypotheses should have equal normalized score when length_penalty=0.0"
