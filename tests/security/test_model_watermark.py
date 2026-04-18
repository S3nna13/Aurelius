"""Tests for src/security/model_watermark.py."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.security.model_watermark import (
    WatermarkConfig,
    WatermarkEmbedder,
    extract_signature,
    watermark_strength,
)

MODEL_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=4,
    n_kv_heads=2,
    head_dim=16,
    d_ff=128,
    vocab_size=256,
    max_seq_len=64,
)

WM_CFG = WatermarkConfig(
    n_triggers=4,
    trigger_length=8,
    target_class=0,
    lr=1e-3,
    embed_steps=50,
)


@pytest.fixture(scope="module")
def embedder() -> WatermarkEmbedder:
    return WatermarkEmbedder(WM_CFG)


@pytest.fixture(scope="module")
def triggers(embedder: WatermarkEmbedder) -> list:
    torch.manual_seed(7)
    return embedder.generate_triggers(MODEL_CFG.vocab_size)


def test_generate_triggers_length(embedder: WatermarkEmbedder) -> None:
    torch.manual_seed(0)
    trigs = embedder.generate_triggers(MODEL_CFG.vocab_size)
    assert len(trigs) == WM_CFG.n_triggers


def test_each_trigger_shape(embedder: WatermarkEmbedder) -> None:
    torch.manual_seed(1)
    trigs = embedder.generate_triggers(MODEL_CFG.vocab_size)
    for t in trigs:
        assert t.shape == (WM_CFG.trigger_length,)


def test_embed_runs_without_error(triggers: list) -> None:
    torch.manual_seed(10)
    m = AureliusTransformer(MODEL_CFG)
    embedder = WatermarkEmbedder(WM_CFG)
    result = embedder.embed(m, triggers)
    assert result is m


def test_verify_returns_float_in_range(triggers: list) -> None:
    torch.manual_seed(11)
    m = AureliusTransformer(MODEL_CFG)
    embedder = WatermarkEmbedder(WM_CFG)
    score = embedder.verify(m, triggers)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_extract_signature_returns_list_of_ints(triggers: list) -> None:
    torch.manual_seed(12)
    m = AureliusTransformer(MODEL_CFG)
    m.eval()
    sigs = extract_signature(m, triggers, WM_CFG.trigger_length)
    assert isinstance(sigs, list)
    assert all(isinstance(s, int) for s in sigs)


def test_after_embedding_verify_score_above_threshold(triggers: list) -> None:
    """After embedding, verify() should return > 0.5. Allow 3 retries."""
    embedder = WatermarkEmbedder(WatermarkConfig(
        n_triggers=4,
        trigger_length=8,
        target_class=0,
        lr=5e-3,
        embed_steps=100,
    ))
    passed = False
    for seed in [42, 123, 999]:
        torch.manual_seed(seed)
        m = AureliusTransformer(MODEL_CFG)
        torch.manual_seed(seed)
        local_triggers = embedder.generate_triggers(MODEL_CFG.vocab_size)
        embedder.embed(m, local_triggers)
        score = embedder.verify(m, local_triggers)
        if score > 0.5:
            passed = True
            break
    assert passed, "verify() did not exceed 0.5 after embedding across 3 seeds"


def test_watermark_strength_in_range(triggers: list) -> None:
    torch.manual_seed(13)
    m = AureliusTransformer(MODEL_CFG)
    m.eval()
    strength = watermark_strength(m, triggers)
    assert isinstance(strength, float)
    assert 0.0 <= strength <= 1.0


def test_trigger_length_respected(embedder: WatermarkEmbedder) -> None:
    cfg = WatermarkConfig(n_triggers=3, trigger_length=12)
    emb = WatermarkEmbedder(cfg)
    trigs = emb.generate_triggers(MODEL_CFG.vocab_size)
    for t in trigs:
        assert t.shape == (12,)


def test_config_defaults() -> None:
    cfg = WatermarkConfig()
    assert cfg.n_triggers == 10
    assert cfg.trigger_length == 8
    assert cfg.target_class == 0
    assert cfg.lr == 1e-3
    assert cfg.embed_steps == 50


def test_vocab_size_respected(embedder: WatermarkEmbedder) -> None:
    vocab = 50
    trigs = embedder.generate_triggers(vocab)
    for t in trigs:
        assert t.max().item() < vocab
        assert t.min().item() >= 0
