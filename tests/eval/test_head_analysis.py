"""Tests for src/eval/head_analysis.py."""

from __future__ import annotations

import math

import pytest
import torch

from src.eval.head_analysis import (
    HeadAnalysisConfig,
    HeadImportanceAnalyzer,
    compute_head_agreement,
    compute_head_entropy,
    find_redundant_heads,
    head_specialization_score,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared small config
# ---------------------------------------------------------------------------

ACFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=2,
    n_kv_heads=2,
    head_dim=32,
    d_ff=128,
    vocab_size=256,
    max_seq_len=512,
)


def _uniform_attn(B: int, H: int, T: int) -> torch.Tensor:
    """Return uniform attention weights: all rows sum to 1."""
    return torch.full((B, H, T, T), 1.0 / T)


def _one_hot_attn(B: int, H: int, T: int) -> torch.Tensor:
    """Return attention weights where every query attends only to position 0."""
    w = torch.zeros(B, H, T, T)
    w[..., 0] = 1.0
    return w


# ---------------------------------------------------------------------------
# 1. HeadAnalysisConfig defaults
# ---------------------------------------------------------------------------


def test_head_analysis_config_defaults():
    cfg = HeadAnalysisConfig()
    assert cfg.n_layers == 2
    assert cfg.n_heads == 2
    assert cfg.redundancy_threshold == pytest.approx(0.9)
    assert cfg.importance_method == "gradient"


# ---------------------------------------------------------------------------
# 2. compute_head_entropy – output shape
# ---------------------------------------------------------------------------


def test_compute_head_entropy_shape():
    B, H, T = 3, 4, 8
    attn = _uniform_attn(B, H, T)
    out = compute_head_entropy(attn)
    assert out.shape == (B, H), f"Expected (B={B}, H={H}), got {out.shape}"


# ---------------------------------------------------------------------------
# 3. compute_head_entropy – uniform → maximum entropy
# ---------------------------------------------------------------------------


def test_compute_head_entropy_uniform_is_max():
    T = 16
    attn = _uniform_attn(1, 1, T)
    entropy = compute_head_entropy(attn)
    expected = math.log(T)
    assert entropy.item() == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# 4. compute_head_entropy – one-hot → near-zero entropy
# ---------------------------------------------------------------------------


def test_compute_head_entropy_one_hot_near_zero():
    T = 16
    attn = _one_hot_attn(1, 1, T)
    entropy = compute_head_entropy(attn)
    assert entropy.item() < 0.01


# ---------------------------------------------------------------------------
# 5. compute_head_agreement – identical → 1.0
# ---------------------------------------------------------------------------


def test_compute_head_agreement_identical():
    B, H, T = 2, 2, 6
    attn = _uniform_attn(B, H, T)
    sim = compute_head_agreement(attn, attn)
    assert sim == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# 6. compute_head_agreement – different → < 1.0
# ---------------------------------------------------------------------------


def test_compute_head_agreement_different_less_than_one():
    B, T = 2, 6
    attn_a = _uniform_attn(B, 1, T)
    attn_b = _one_hot_attn(B, 1, T)
    sim = compute_head_agreement(attn_a, attn_b)
    assert sim < 1.0


# ---------------------------------------------------------------------------
# 7. find_redundant_heads – identical heads → finds pair
# ---------------------------------------------------------------------------


def test_find_redundant_heads_identical_heads():
    B, T = 2, 6
    # Two heads with identical distributions → similarity = 1.0 > 0.9 threshold
    attn = _uniform_attn(B, 2, T)
    redundant = find_redundant_heads([attn], threshold=0.9)
    assert len(redundant) >= 1
    layer_idx, h_i, h_j, sim = redundant[0]
    assert layer_idx == 0
    assert {h_i, h_j} == {0, 1}
    assert sim == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# 8. find_redundant_heads – distinct heads → no redundancy at high threshold
# ---------------------------------------------------------------------------


def test_find_redundant_heads_distinct_heads_no_redundancy():
    B, T = 2, 6
    attn = torch.zeros(B, 2, T, T)
    # Head 0: attend to position 0; head 1: attend to position 5
    attn[:, 0, :, 0] = 1.0
    attn[:, 1, :, -1] = 1.0
    redundant = find_redundant_heads([attn], threshold=0.9)
    assert len(redundant) == 0


# ---------------------------------------------------------------------------
# 9. head_specialization_score – correct keys
# ---------------------------------------------------------------------------


def test_head_specialization_score_keys():
    T = 8
    attn = torch.eye(T) / 1.0  # identity: attends to self
    scores = head_specialization_score(attn)
    assert set(scores.keys()) == {"diagonal_focus", "local_focus", "uniform_score"}


# ---------------------------------------------------------------------------
# 10. head_specialization_score – identity matrix has high diagonal_focus
# ---------------------------------------------------------------------------


def test_head_specialization_score_diagonal_focus():
    T = 8
    attn = torch.eye(T)  # perfect diagonal attention
    scores = head_specialization_score(attn)
    # diagonal_focus should be 1.0 (mean of diagonal of identity = 1)
    assert scores["diagonal_focus"] == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# 11. HeadImportanceAnalyzer – rank_heads_by_entropy sorted ascending
# ---------------------------------------------------------------------------


def test_rank_heads_by_entropy_sorted():
    model = AureliusTransformer(ACFG)
    ha_cfg = HeadAnalysisConfig(n_layers=ACFG.n_layers, n_heads=ACFG.n_heads)
    analyzer = HeadImportanceAnalyzer(model, ha_cfg)

    input_ids = torch.randint(0, ACFG.vocab_size, (1, 16))
    ranked = analyzer.rank_heads_by_entropy(input_ids)

    # Should have n_layers * n_heads entries
    assert len(ranked) == ACFG.n_layers * ACFG.n_heads

    # Each entry is (layer_idx, head_idx, entropy_value)
    assert all(len(entry) == 3 for entry in ranked)

    # Should be sorted ascending by entropy
    entropies = [e for _, _, e in ranked]
    assert entropies == sorted(entropies)
