"""Tests for src/training/multilingual.py"""

from __future__ import annotations

import math
import random

import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.multilingual import (
    LanguageTaggedDataset,
    MultilingualConfig,
    MultilingualTrainer,
    compute_language_sampling_probs,
    cross_lingual_alignment_loss,
)

# ---------------------------------------------------------------------------
# Tiny model config for fast tests
# ---------------------------------------------------------------------------
VOCAB = 256
SEQ = 8
BATCH = 2


def make_model():
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=VOCAB,
        max_seq_len=512,
    )
    return AureliusTransformer(cfg)


def make_trainer(model=None):
    if model is None:
        model = make_model()
    cfg = MultilingualConfig()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    return MultilingualTrainer(model, cfg, opt)


# ---------------------------------------------------------------------------
# MultilingualConfig defaults
# ---------------------------------------------------------------------------


def test_multilingual_config_defaults():
    cfg = MultilingualConfig()
    assert cfg.languages == ["en", "fr", "de"]
    assert cfg.temperature == 5.0
    assert cfg.alpha == 0.3
    assert cfg.upsample_low_resource is True


# ---------------------------------------------------------------------------
# compute_language_sampling_probs
# ---------------------------------------------------------------------------


def test_sampling_probs_sum_to_one():
    counts = {"en": 10000, "fr": 2000, "de": 500}
    probs = compute_language_sampling_probs(counts, temperature=5.0, alpha=0.3)
    assert abs(sum(probs.values()) - 1.0) < 1e-6


def test_sampling_probs_high_resource_gets_higher_weight():
    counts = {"en": 100000, "sw": 100}
    probs = compute_language_sampling_probs(counts, temperature=5.0, alpha=0.0)
    assert probs["en"] > probs["sw"]


def test_sampling_probs_equal_counts_equal_probs():
    counts = {"en": 500, "fr": 500, "de": 500}
    probs = compute_language_sampling_probs(counts, temperature=5.0, alpha=0.3)
    values = list(probs.values())
    assert max(values) - min(values) < 1e-6


def test_sampling_probs_temperature_one_proportional():
    """At temperature=1.0 and alpha=0, probs should be proportional to counts."""
    counts = {"en": 400, "fr": 100}
    probs = compute_language_sampling_probs(counts, temperature=1.0, alpha=0.0)
    ratio = probs["en"] / probs["fr"]
    expected = 400 / 100
    assert abs(ratio - expected) < 1e-4


# ---------------------------------------------------------------------------
# cross_lingual_alignment_loss
# ---------------------------------------------------------------------------


def test_alignment_loss_returns_scalar():
    src = torch.randn(BATCH, 64)
    tgt = torch.randn(BATCH, 64)
    loss = cross_lingual_alignment_loss(src, tgt)
    assert loss.shape == ()


def test_alignment_loss_same_embeddings_near_zero():
    emb = torch.randn(BATCH, 64)
    loss = cross_lingual_alignment_loss(emb, emb)
    assert loss.item() < 1e-5


# ---------------------------------------------------------------------------
# LanguageTaggedDataset
# ---------------------------------------------------------------------------

SAMPLES = [
    {"lang": "en", "text": "Hello"},
    {"lang": "en", "text": "World"},
    {"lang": "fr", "text": "Bonjour"},
    {"lang": "de", "text": "Hallo"},
    {"lang": "de", "text": "Welt"},
]


def test_get_by_language_filters_correctly():
    ds = LanguageTaggedDataset(SAMPLES)
    en_samples = ds.get_by_language("en")
    assert len(en_samples) == 2
    assert all(s["lang"] == "en" for s in en_samples)


def test_sample_batch_returns_correct_size():
    ds = LanguageTaggedDataset(SAMPLES)
    probs = {"en": 0.5, "fr": 0.3, "de": 0.2}
    batch = ds.sample_batch(probs, batch_size=10)
    assert len(batch) == 10


def test_sample_batch_respects_language_probabilities():
    """With very skewed probs and many samples, dominant language appears most."""
    samples = [{"lang": "en"}] * 100 + [{"lang": "fr"}] * 100
    ds = LanguageTaggedDataset(samples)
    probs = {"en": 0.95, "fr": 0.05}
    rng = random.Random(42)
    batch = ds.sample_batch(probs, batch_size=200, rng=rng)
    en_count = sum(1 for s in batch if s["lang"] == "en")
    # Should be well above 50% even with stochasticity
    assert en_count > 100


# ---------------------------------------------------------------------------
# MultilingualTrainer.train_step
# ---------------------------------------------------------------------------


def test_train_step_returns_required_keys():
    trainer = make_trainer()
    batch_ids = [torch.randint(0, VOCAB, (SEQ,)) for _ in range(BATCH)]
    result = trainer.train_step(batch_ids)
    assert "loss" in result
    assert "n_languages" in result


def test_train_step_loss_is_finite():
    trainer = make_trainer()
    batch_ids = [torch.randint(0, VOCAB, (SEQ,)) for _ in range(BATCH)]
    result = trainer.train_step(batch_ids)
    assert math.isfinite(result["loss"])


# ---------------------------------------------------------------------------
# MultilingualTrainer.alignment_step
# ---------------------------------------------------------------------------


def test_alignment_step_returns_alignment_loss_key():
    trainer = make_trainer()
    src = torch.randint(0, VOCAB, (BATCH, SEQ))
    tgt = torch.randint(0, VOCAB, (BATCH, SEQ))
    result = trainer.alignment_step(src, tgt)
    assert "alignment_loss" in result


def test_alignment_step_loss_is_finite_float():
    trainer = make_trainer()
    src = torch.randint(0, VOCAB, (BATCH, SEQ))
    tgt = torch.randint(0, VOCAB, (BATCH, SEQ))
    result = trainer.alignment_step(src, tgt)
    assert isinstance(result["alignment_loss"], float)
    assert math.isfinite(result["alignment_loss"])
