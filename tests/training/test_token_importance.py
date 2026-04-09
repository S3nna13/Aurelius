"""Tests for src/training/token_importance.py"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.token_importance import (
    TokenImportanceConfig,
    TokenImportanceCurriculum,
    SelectiveTrainer,
    build_masked_labels,
    score_tokens_by_attention,
    score_tokens_by_loss,
    select_important_tokens,
)

B = 2
T = 16
VOCAB = 256


@pytest.fixture(scope="module")
def small_cfg():
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=VOCAB,
        max_seq_len=512,
    )


@pytest.fixture(scope="module")
def small_model(small_cfg):
    torch.manual_seed(42)
    model = AureliusTransformer(small_cfg)
    model.eval()
    return model


@pytest.fixture
def default_config():
    return TokenImportanceConfig()


@pytest.fixture
def input_ids():
    torch.manual_seed(0)
    return torch.randint(0, VOCAB, (B, T))


@pytest.fixture
def labels():
    torch.manual_seed(1)
    return torch.randint(0, VOCAB, (B, T))


def test_config_defaults():
    cfg = TokenImportanceConfig()
    assert cfg.scoring_method == "loss"
    assert cfg.top_k_fraction == 0.5
    assert cfg.min_tokens == 4
    assert cfg.smooth_alpha == 0.1
    assert cfg.update_freq == 100


def test_score_tokens_by_loss_shape(small_model, input_ids):
    scores = score_tokens_by_loss(small_model, input_ids)
    assert scores.shape == (B, T - 1)


def test_score_tokens_by_loss_nonnegative(small_model, input_ids):
    scores = score_tokens_by_loss(small_model, input_ids)
    assert (scores >= 0).all()


def test_score_tokens_by_attention_shape():
    torch.manual_seed(5)
    logits = torch.randn(B, T, VOCAB)
    entropy = score_tokens_by_attention(logits)
    assert entropy.shape == (B, T)


def test_score_tokens_by_attention_nonnegative():
    torch.manual_seed(6)
    logits = torch.randn(B, T, VOCAB)
    entropy = score_tokens_by_attention(logits)
    assert (entropy >= 0).all()


def test_select_important_tokens_shape(default_config):
    scores = torch.rand(B, T)
    mask = select_important_tokens(scores, default_config)
    assert mask.shape == (B, T)
    assert mask.dtype == torch.bool


def test_select_important_tokens_min_tokens():
    cfg = TokenImportanceConfig(top_k_fraction=0.01, min_tokens=4)
    scores = torch.rand(B, T)
    mask = select_important_tokens(scores, cfg)
    for b in range(B):
        n_selected = mask[b].sum().item()
        assert n_selected >= cfg.min_tokens


def test_select_important_tokens_top_k_fraction():
    cfg = TokenImportanceConfig(top_k_fraction=0.5, min_tokens=1)
    scores = torch.rand(B, T)
    mask = select_important_tokens(scores, cfg)
    expected_k = max(int(T * cfg.top_k_fraction), cfg.min_tokens)
    for b in range(B):
        n_selected = mask[b].sum().item()
        assert n_selected == expected_k


def test_build_masked_labels_minus100_where_false(labels):
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[:, :T // 2] = True
    masked = build_masked_labels(labels, mask)
    assert (masked[:, T // 2:] == -100).all()


def test_build_masked_labels_preserves_important(labels):
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[:, :T // 2] = True
    masked = build_masked_labels(labels, mask)
    assert (masked[:, :T // 2] == labels[:, :T // 2]).all()


def test_curriculum_fraction_at_step_0():
    cfg = TokenImportanceConfig(top_k_fraction=0.5)
    curriculum = TokenImportanceCurriculum(cfg, total_steps=1000)
    result = curriculum.get_fraction(0)
    assert abs(result - cfg.top_k_fraction) < 1e-6


def test_curriculum_fraction_at_total_steps():
    cfg = TokenImportanceConfig(top_k_fraction=0.5)
    total_steps = 500
    curriculum = TokenImportanceCurriculum(cfg, total_steps=total_steps)
    assert abs(curriculum.get_fraction(total_steps) - 1.0) < 1e-6
    assert abs(curriculum.get_fraction(total_steps + 100) - 1.0) < 1e-6


def test_curriculum_update_running_stats():
    cfg = TokenImportanceConfig(smooth_alpha=0.5)
    curriculum = TokenImportanceCurriculum(cfg, total_steps=100)
    initial_mean = curriculum.get_running_mean()
    assert initial_mean == 0.0

    scores = torch.ones(B, T) * 2.0
    curriculum.update_running_stats(scores)
    updated_mean = curriculum.get_running_mean()

    assert updated_mean != initial_mean
    expected = (1.0 - cfg.smooth_alpha) * initial_mean + cfg.smooth_alpha * 2.0
    assert abs(updated_mean - expected) < 1e-6


def test_selective_trainer_train_step(small_cfg, input_ids, labels):
    torch.manual_seed(99)
    model = AureliusTransformer(small_cfg)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    cfg = TokenImportanceConfig(top_k_fraction=0.5, min_tokens=4)

    trainer = SelectiveTrainer(model, optimizer, cfg)
    result = trainer.train_step(input_ids, labels)

    assert "loss" in result
    assert "n_active_tokens" in result
    assert "fraction_active" in result
    assert math.isfinite(result["loss"])
    assert result["n_active_tokens"] > 0
    assert 0.0 < result["fraction_active"] <= 1.0
