"""Tests for src/data/multilingual.py.

Uses tiny configs (D_MODEL=8, small text corpora) to keep tests fast.
Covers ≥10 distinct test cases as required.
"""

from __future__ import annotations

import random

import pytest
import torch

from src.data.multilingual import (
    CrossLingualAligner,
    LanguageConfig,
    LanguageSampler,
    build_multilingual_batch,
    compute_language_accuracy,
    detect_language_heuristic,
    smooth_language_weights,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

D_MODEL = 8

TINY_DATA = {
    "en": ["hello world", "the quick brown fox", "good morning"],
    "fr": ["bonjour le monde", "je suis étudiant", "bonsoir"],
    "de": ["guten morgen", "ich heiße Klaus", "auf Wiedersehen"],
    "es": ["hola mundo", "buenos días", "¿cómo estás?"],
    "zh": ["你好世界", "我是学生", "早上好"],
}

LANG_TO_ID = {"en": 0, "fr": 1, "de": 2, "es": 3, "zh": 4}


def simple_tokenize(text: str) -> list[int]:
    """Tiny whitespace tokenizer: returns ASCII ordinal of first char per word."""
    return [ord(w[0]) % 256 for w in text.split() if w]


# ---------------------------------------------------------------------------
# 1. LanguageConfig defaults
# ---------------------------------------------------------------------------


def test_language_config_defaults():
    cfg = LanguageConfig()
    assert cfg.language_codes == ["en", "fr", "de", "es", "zh"]
    assert cfg.sampling_weights is None
    assert cfg.temperature == 0.3


# ---------------------------------------------------------------------------
# 2. smooth_language_weights sums to 1
# ---------------------------------------------------------------------------


def test_smooth_language_weights_sum_to_one():
    counts = {"en": 1000, "fr": 500, "de": 300, "es": 200, "zh": 100}
    weights = smooth_language_weights(counts, temperature=0.7)
    assert abs(sum(weights.values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 3. smooth_language_weights: higher-count language has higher weight
# ---------------------------------------------------------------------------


def test_smooth_language_weights_ordering():
    counts = {"en": 1000, "fr": 100}
    weights = smooth_language_weights(counts, temperature=0.5)
    assert weights["en"] > weights["fr"]


# ---------------------------------------------------------------------------
# 4. detect_language_heuristic — Chinese text
# ---------------------------------------------------------------------------


def test_detect_language_heuristic_chinese():
    assert detect_language_heuristic("你好世界") == "zh"


# ---------------------------------------------------------------------------
# 5. detect_language_heuristic — French (é)
# ---------------------------------------------------------------------------


def test_detect_language_heuristic_french():
    assert detect_language_heuristic("je suis très content") == "fr"


# ---------------------------------------------------------------------------
# 6. detect_language_heuristic — German (ü)
# ---------------------------------------------------------------------------


def test_detect_language_heuristic_german():
    assert detect_language_heuristic("schön und gemütlich") == "de"


# ---------------------------------------------------------------------------
# 7. detect_language_heuristic — default English
# ---------------------------------------------------------------------------


def test_detect_language_heuristic_english_default():
    assert detect_language_heuristic("hello world this is a test") == "en"


# ---------------------------------------------------------------------------
# 8. LanguageSampler.get_weights sums to 1
# ---------------------------------------------------------------------------


def test_language_sampler_get_weights_sum_to_one():
    cfg = LanguageConfig()
    sampler = LanguageSampler(TINY_DATA, cfg)
    weights = sampler.get_weights()
    assert abs(sum(weights.values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 9. sample_batch length equals batch_size
# ---------------------------------------------------------------------------


def test_sample_batch_length():
    cfg = LanguageConfig()
    sampler = LanguageSampler(TINY_DATA, cfg)
    rng = random.Random(42)
    batch = sampler.sample_batch(batch_size=16, rng=rng)
    assert len(batch) == 16


# ---------------------------------------------------------------------------
# 10. sample_batch returns valid language codes
# ---------------------------------------------------------------------------


def test_sample_batch_valid_language_codes():
    cfg = LanguageConfig()
    sampler = LanguageSampler(TINY_DATA, cfg)
    rng = random.Random(0)
    batch = sampler.sample_batch(batch_size=20, rng=rng)
    valid_langs = set(TINY_DATA.keys())
    for lang, _text in batch:
        assert lang in valid_langs, f"Unexpected language code: {lang!r}"


# ---------------------------------------------------------------------------
# 11. get_stats has all language keys
# ---------------------------------------------------------------------------


def test_get_stats_has_all_language_keys():
    cfg = LanguageConfig()
    sampler = LanguageSampler(TINY_DATA, cfg)
    stats = sampler.get_stats()
    for lang in TINY_DATA:
        assert lang in stats
        assert "n_samples" in stats[lang]
        assert "weight" in stats[lang]


# ---------------------------------------------------------------------------
# 12. build_multilingual_batch — token_ids shape
# ---------------------------------------------------------------------------


def test_build_multilingual_batch_token_ids_shape():
    samples = [("en", "hello world"), ("fr", "bonjour"), ("zh", "你好")]
    max_len = 10
    batch = build_multilingual_batch(
        samples, LANG_TO_ID, simple_tokenize, max_len=max_len, pad_id=0
    )
    assert batch.token_ids.shape == (3, max_len)
    assert batch.language_ids.shape == (3,)
    assert batch.attention_mask.shape == (3, max_len)
    assert len(batch.language_labels) == 3


# ---------------------------------------------------------------------------
# 13. attention_mask sum equals real token count
# ---------------------------------------------------------------------------


def test_build_multilingual_batch_attention_mask_real_tokens():
    text = "hello world test"
    samples = [("en", text)]
    max_len = 10
    batch = build_multilingual_batch(
        samples, LANG_TO_ID, simple_tokenize, max_len=max_len, pad_id=0
    )
    n_real_tokens = len(simple_tokenize(text))
    assert batch.attention_mask[0].sum().item() == n_real_tokens


# ---------------------------------------------------------------------------
# 14. compute_language_accuracy — perfect score
# ---------------------------------------------------------------------------


def test_compute_language_accuracy_perfect():
    true_ids = torch.tensor([0, 1, 2, 3, 4])
    pred_ids = torch.tensor([0, 1, 2, 3, 4])
    acc = compute_language_accuracy(pred_ids, true_ids)
    assert acc == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 15. compute_language_accuracy — partial score
# ---------------------------------------------------------------------------


def test_compute_language_accuracy_partial():
    true_ids = torch.tensor([0, 1, 2, 3])
    pred_ids = torch.tensor([0, 1, 9, 9])  # 2 correct out of 4
    acc = compute_language_accuracy(pred_ids, true_ids)
    assert acc == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 16. CrossLingualAligner.align_loss — returns scalar
# ---------------------------------------------------------------------------


def test_cross_lingual_aligner_align_loss_scalar():
    aligner = CrossLingualAligner(d_model=D_MODEL, n_languages=3)
    embs = torch.randn(6, D_MODEL)
    lang_ids = torch.tensor([0, 0, 1, 1, 2, 2])
    loss = aligner.align_loss(embs, lang_ids)
    assert loss.shape == torch.Size([])
    assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# 17. CrossLingualAligner.language_adversarial_loss — returns scalar
# ---------------------------------------------------------------------------


def test_cross_lingual_aligner_adversarial_loss_scalar():
    aligner = CrossLingualAligner(d_model=D_MODEL, n_languages=3)
    embs = torch.randn(6, D_MODEL)
    lang_ids = torch.tensor([0, 0, 1, 1, 2, 2])
    loss = aligner.language_adversarial_loss(embs, lang_ids)
    assert loss.shape == torch.Size([])
    assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# 18. align_loss is lower for identical embeddings per language
# ---------------------------------------------------------------------------


def test_align_loss_zero_for_identical_embeddings():
    aligner = CrossLingualAligner(d_model=D_MODEL, n_languages=2)
    # Identical embeddings within each language → zero pairwise distance
    emb_lang0 = torch.ones(3, D_MODEL)
    emb_lang1 = torch.zeros(3, D_MODEL)
    embs = torch.cat([emb_lang0, emb_lang1], dim=0)
    lang_ids = torch.tensor([0, 0, 0, 1, 1, 1])
    loss = aligner.align_loss(embs, lang_ids)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# 19. smooth_language_weights with temperature=1.0 is proportional to counts
# ---------------------------------------------------------------------------


def test_smooth_language_weights_temperature_one():
    counts = {"en": 3, "fr": 1}
    weights = smooth_language_weights(counts, temperature=1.0)
    # p(en) / p(fr) should equal 3/1 = 3.0
    ratio = weights["en"] / weights["fr"]
    assert ratio == pytest.approx(3.0, rel=1e-5)


# ---------------------------------------------------------------------------
# 20. LanguageSampler with explicit weights uses them (not count-derived)
# ---------------------------------------------------------------------------


def test_language_sampler_explicit_weights_normalised():
    explicit = {"en": 2.0, "fr": 2.0, "de": 2.0, "es": 2.0, "zh": 2.0}
    cfg = LanguageConfig(sampling_weights=explicit)
    sampler = LanguageSampler(TINY_DATA, cfg)
    weights = sampler.get_weights()
    # All equal and summing to 1
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    for w in weights.values():
        assert w == pytest.approx(0.2, rel=1e-5)
