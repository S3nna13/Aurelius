"""Tests for the semantic similarity defense module."""

from __future__ import annotations

import math

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.security.semantic_similarity_defense import SimilarityDefense

# ---------------------------------------------------------------------------
# Shared tiny config and fixtures
# ---------------------------------------------------------------------------

TINY_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=4,
    n_kv_heads=2,
    head_dim=16,
    d_ff=128,
    vocab_size=256,
    max_seq_len=64,
)

SEED = 42
B = 1
S = 8


@pytest.fixture(scope="module")
def model() -> AureliusTransformer:
    torch.manual_seed(SEED)
    m = AureliusTransformer(TINY_CFG)
    m.eval()
    return m


@pytest.fixture(scope="module")
def defense(model) -> SimilarityDefense:
    return SimilarityDefense(model)


@pytest.fixture(scope="module")
def reference_ids() -> torch.Tensor:
    torch.manual_seed(SEED)
    return torch.randint(0, 256, (B, S))


@pytest.fixture(scope="module")
def query_ids() -> torch.Tensor:
    torch.manual_seed(SEED + 1)
    return torch.randint(0, 256, (B, S))


# ---------------------------------------------------------------------------
# Test 1: SimilarityDefense instantiates
# ---------------------------------------------------------------------------


def test_instantiates(model):
    """SimilarityDefense instantiates without error."""
    sd = SimilarityDefense(model)
    assert sd is not None
    assert sd.similarity_threshold == 0.85
    assert sd.consistency_threshold == 0.7


# ---------------------------------------------------------------------------
# Test 2: embed returns 1-D Tensor of shape (d_model,)
# ---------------------------------------------------------------------------


def test_embed_shape(defense, reference_ids):
    """embed returns a 1-D tensor of shape (d_model,)."""
    emb = defense.embed(reference_ids)
    assert isinstance(emb, torch.Tensor)
    assert emb.dim() == 1
    assert emb.shape == (TINY_CFG.d_model,)


# ---------------------------------------------------------------------------
# Test 3: embed output is finite
# ---------------------------------------------------------------------------


def test_embed_finite(defense, reference_ids):
    """embed output contains only finite values."""
    emb = defense.embed(reference_ids)
    assert torch.isfinite(emb).all(), "embed contains non-finite values"


# ---------------------------------------------------------------------------
# Test 4: Same input -> cosine_similarity = 1.0
# ---------------------------------------------------------------------------


def test_cosine_similarity_same_input(defense, reference_ids):
    """cosine_similarity of an embedding with itself is 1.0."""
    emb = defense.embed(reference_ids)
    sim = defense.cosine_similarity(emb, emb)
    assert abs(sim - 1.0) < 1e-5, f"Expected 1.0, got {sim}"


# ---------------------------------------------------------------------------
# Test 5: cosine_similarity returns float in [-1, 1]
# ---------------------------------------------------------------------------


def test_cosine_similarity_range(defense, reference_ids, query_ids):
    """cosine_similarity returns a Python float in [-1, 1]."""
    emb_a = defense.embed(reference_ids)
    emb_b = defense.embed(query_ids)
    sim = defense.cosine_similarity(emb_a, emb_b)
    assert isinstance(sim, float)
    assert -1.0 <= sim <= 1.0


# ---------------------------------------------------------------------------
# Test 6: is_paraphrase_evasion returns bool
# ---------------------------------------------------------------------------


def test_is_paraphrase_evasion_returns_bool(defense, reference_ids, query_ids):
    """is_paraphrase_evasion returns a Python bool."""
    result = defense.is_paraphrase_evasion(reference_ids, query_ids)
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Test 7: Identical inputs -> is_paraphrase_evasion True
# ---------------------------------------------------------------------------


def test_is_paraphrase_evasion_identical_inputs(defense, reference_ids):
    """Identical inputs yield cosine similarity 1.0, which exceeds the threshold."""
    result = defense.is_paraphrase_evasion(reference_ids, reference_ids)
    assert result is True


# ---------------------------------------------------------------------------
# Test 8: Very different threshold -> is_paraphrase_evasion False
# ---------------------------------------------------------------------------


def test_is_paraphrase_evasion_high_threshold(model, reference_ids, query_ids):
    """With similarity_threshold=2.0 (impossible to exceed), evasion is never flagged."""
    sd = SimilarityDefense(model, similarity_threshold=2.0)
    result = sd.is_paraphrase_evasion(reference_ids, query_ids)
    assert result is False


# ---------------------------------------------------------------------------
# Test 9: output_consistency returns float in [0, 1]
# ---------------------------------------------------------------------------


def test_output_consistency_range(defense, model):
    """output_consistency returns a float in [0, 1]."""
    torch.manual_seed(SEED + 10)
    ids_a = torch.randint(0, 256, (B, S))
    ids_b = torch.randint(0, 256, (B, S))
    score = defense.output_consistency(model, [ids_a, ids_b])
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Test 10: Single-element list -> output_consistency = 1.0
# ---------------------------------------------------------------------------


def test_output_consistency_single_element(defense, model, reference_ids):
    """output_consistency returns 1.0 for a single-element list."""
    score = defense.output_consistency(model, [reference_ids])
    assert score == 1.0


# ---------------------------------------------------------------------------
# Test 11: is_consistent returns bool
# ---------------------------------------------------------------------------


def test_is_consistent_returns_bool(defense, model):
    """is_consistent returns a Python bool."""
    torch.manual_seed(SEED + 20)
    ids_a = torch.randint(0, 256, (B, S))
    ids_b = torch.randint(0, 256, (B, S))
    result = defense.is_consistent(model, [ids_a, ids_b])
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Test 12: scan returns dict with required keys
# ---------------------------------------------------------------------------


def test_scan_returns_dict_with_keys(defense, reference_ids, query_ids):
    """scan returns a dict containing 'similarity', 'is_evasion', 'similarity_threshold'."""
    result = defense.scan(reference_ids, query_ids)
    assert isinstance(result, dict)
    assert "similarity" in result
    assert "is_evasion" in result
    assert "similarity_threshold" in result


# ---------------------------------------------------------------------------
# Test 13: scan['similarity'] is in [-1, 1]
# ---------------------------------------------------------------------------


def test_scan_similarity_range(defense, reference_ids, query_ids):
    """scan['similarity'] is a float in [-1, 1]."""
    result = defense.scan(reference_ids, query_ids)
    sim = result["similarity"]
    assert isinstance(sim, float)
    assert -1.0 <= sim <= 1.0


# ---------------------------------------------------------------------------
# Test 14: scan['is_evasion'] matches manual threshold comparison
# ---------------------------------------------------------------------------


def test_scan_is_evasion_consistent_with_threshold(defense, reference_ids, query_ids):
    """scan['is_evasion'] is True iff similarity > similarity_threshold."""
    result = defense.scan(reference_ids, query_ids)
    expected = result["similarity"] > result["similarity_threshold"]
    assert result["is_evasion"] == expected
