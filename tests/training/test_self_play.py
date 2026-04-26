"""Tests for self-play DPO pair generation."""

import math

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.self_play import (
    SelfPlayConfig,
    SelfPlayPair,
    SelfPlayRollout,
    generate_candidates,
    make_dpo_pairs,
    score_response,
)


@pytest.fixture
def small_model():
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=128,
    )
    torch.manual_seed(42)
    model = AureliusTransformer(cfg)
    model.eval()
    return model


def test_score_response_returns_finite(small_model):
    prompt = torch.randint(0, 256, (8,))
    response = torch.randint(0, 256, (4,))
    score = score_response(small_model, prompt, response)
    assert isinstance(score, float)
    assert math.isfinite(score)
    assert score <= 0.0  # log-prob is always <= 0


def test_score_response_negative(small_model):
    """Score should be <= 0 (it's a mean log-probability)."""
    prompt = torch.randint(0, 256, (6,))
    response = torch.randint(0, 256, (6,))
    score = score_response(small_model, prompt, response)
    assert score <= 0.0


def test_generate_candidates_count(small_model):
    cfg = SelfPlayConfig(n_candidates=3, max_new_tokens=8)
    prompt = torch.randint(0, 256, (5,))
    rollouts = generate_candidates(small_model, prompt, cfg)
    assert len(rollouts) == 3
    for r in rollouts:
        assert isinstance(r, SelfPlayRollout)
        assert r.response_ids.ndim == 1
        assert len(r.response_ids) == 8


def test_generate_candidates_sorted(small_model):
    cfg = SelfPlayConfig(n_candidates=4, max_new_tokens=8)
    prompt = torch.randint(0, 256, (5,))
    rollouts = generate_candidates(small_model, prompt, cfg)
    scores = [r.score for r in rollouts]
    assert scores == sorted(scores, reverse=True)


def test_make_dpo_pairs_structure(small_model):
    cfg = SelfPlayConfig(n_candidates=4, max_new_tokens=8, keep_top_k=1, keep_bottom_k=1)
    prompts = [torch.randint(0, 256, (5,)) for _ in range(2)]
    pairs = make_dpo_pairs(small_model, prompts, cfg)
    assert len(pairs) >= 0  # may be empty if all scores equal
    for p in pairs:
        assert isinstance(p, SelfPlayPair)
        assert p.chosen_score >= p.rejected_score


def test_make_dpo_pairs_chosen_higher(small_model):
    """chosen_score must be >= rejected_score for every pair."""
    cfg = SelfPlayConfig(n_candidates=6, max_new_tokens=10, keep_top_k=2, keep_bottom_k=2)
    prompts = [torch.randint(0, 256, (6,)) for _ in range(3)]
    pairs = make_dpo_pairs(small_model, prompts, cfg)
    for p in pairs:
        assert p.chosen_score >= p.rejected_score - 1e-6
