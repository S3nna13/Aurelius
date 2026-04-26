"""Tests for src/interpretability/counterfactual_explanation.py

Tiny model config: n_layers=2, d_model=64, n_heads=4, n_kv_heads=2,
head_dim=16, d_ff=128, vocab_size=256, max_seq_len=32

All sequences use T=8. Target token: 42.
"""

from __future__ import annotations

import pytest
import torch

from src.interpretability.counterfactual_explanation import (
    CounterfactualConfig,
    CounterfactualResult,
    beam_search_counterfactual,
    counterfactual_feature_importance,
    greedy_counterfactual,
    token_edit_distance,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

T = 8
VOCAB_SIZE = 256
TARGET_TOKEN = 42


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_config() -> AureliusConfig:
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=VOCAB_SIZE,
        max_seq_len=32,
    )


@pytest.fixture(scope="module")
def model(tiny_config: AureliusConfig) -> AureliusTransformer:
    torch.manual_seed(0)
    m = AureliusTransformer(tiny_config)
    m.eval()
    return m


@pytest.fixture(scope="module")
def input_ids() -> torch.Tensor:
    torch.manual_seed(7)
    ids = torch.randint(0, VOCAB_SIZE, (T,))
    # Replace any accidental TARGET_TOKENs to keep tests clean
    ids[ids == TARGET_TOKEN] = TARGET_TOKEN + 1
    return ids


@pytest.fixture(scope="module")
def cfg() -> CounterfactualConfig:
    return CounterfactualConfig(
        max_substitutions=3,
        n_candidates=8,
        beam_width=4,
        target_class=TARGET_TOKEN,
        temperature=1.0,
    )


# ---------------------------------------------------------------------------
# 1. token_edit_distance -- identical sequences -> 0
# ---------------------------------------------------------------------------


def test_edit_distance_identical():
    torch.manual_seed(1)
    ids = torch.randint(0, VOCAB_SIZE, (T,))
    assert token_edit_distance(ids, ids.clone()) == 0


# ---------------------------------------------------------------------------
# 2. token_edit_distance -- fully different -> T
# ---------------------------------------------------------------------------


def test_edit_distance_fully_different():
    ids_a = torch.zeros(T, dtype=torch.long)
    ids_b = torch.ones(T, dtype=torch.long)
    assert token_edit_distance(ids_a, ids_b) == T


# ---------------------------------------------------------------------------
# 3. token_edit_distance -- one substitution -> 1
# ---------------------------------------------------------------------------


def test_edit_distance_one_substitution():
    torch.manual_seed(2)
    ids_a = torch.randint(0, VOCAB_SIZE, (T,))
    ids_b = ids_a.clone()
    ids_b[3] = (ids_b[3].item() + 1) % VOCAB_SIZE
    assert token_edit_distance(ids_a, ids_b) == 1


# ---------------------------------------------------------------------------
# 4. CounterfactualConfig defaults are sane
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = CounterfactualConfig()
    assert cfg.max_substitutions == 3
    assert cfg.n_candidates == 8
    assert cfg.beam_width == 4
    assert cfg.target_class == 0
    assert cfg.temperature == 1.0


# ---------------------------------------------------------------------------
# 5. greedy_counterfactual returns CounterfactualResult
# ---------------------------------------------------------------------------


def test_greedy_returns_result(model, input_ids, cfg):
    torch.manual_seed(10)
    result = greedy_counterfactual(model, input_ids, TARGET_TOKEN, cfg)
    assert isinstance(result, CounterfactualResult)


# ---------------------------------------------------------------------------
# 6. greedy_counterfactual n_substitutions <= max_substitutions
# ---------------------------------------------------------------------------


def test_greedy_n_substitutions_bounded(model, input_ids, cfg):
    torch.manual_seed(11)
    result = greedy_counterfactual(model, input_ids, TARGET_TOKEN, cfg)
    assert result.n_substitutions <= cfg.max_substitutions


# ---------------------------------------------------------------------------
# 7. greedy_counterfactual changed_positions length matches n_substitutions
# ---------------------------------------------------------------------------


def test_greedy_changed_positions_length(model, input_ids, cfg):
    torch.manual_seed(12)
    result = greedy_counterfactual(model, input_ids, TARGET_TOKEN, cfg)
    assert len(result.changed_positions) == result.n_substitutions


# ---------------------------------------------------------------------------
# 8. greedy_counterfactual output shape: counterfactual_ids same length as input
# ---------------------------------------------------------------------------


def test_greedy_output_shape(model, input_ids, cfg):
    torch.manual_seed(13)
    result = greedy_counterfactual(model, input_ids, TARGET_TOKEN, cfg)
    assert result.counterfactual_ids.shape == input_ids.shape


# ---------------------------------------------------------------------------
# 9. beam_search_counterfactual returns CounterfactualResult
# ---------------------------------------------------------------------------


def test_beam_returns_result(model, input_ids, cfg):
    torch.manual_seed(20)
    result = beam_search_counterfactual(model, input_ids, TARGET_TOKEN, cfg)
    assert isinstance(result, CounterfactualResult)


# ---------------------------------------------------------------------------
# 10. counterfactual_feature_importance returns binary tensor of same length
# ---------------------------------------------------------------------------


def test_feature_importance_binary_same_length():
    torch.manual_seed(30)
    original = torch.randint(0, VOCAB_SIZE, (T,))
    cf = original.clone()
    cf[2] = (cf[2].item() + 1) % VOCAB_SIZE
    cf[5] = (cf[5].item() + 1) % VOCAB_SIZE

    mask = counterfactual_feature_importance(original, cf)
    assert mask.shape == (T,)
    unique_vals = set(mask.unique().tolist())
    assert unique_vals.issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# 11. Importance mask has 1s exactly at changed positions
# ---------------------------------------------------------------------------


def test_feature_importance_ones_at_changed_positions():
    torch.manual_seed(31)
    original = torch.randint(0, VOCAB_SIZE, (T,))
    cf = original.clone()
    changed = [1, 4, 6]
    for pos in changed:
        cf[pos] = (cf[pos].item() + 1) % VOCAB_SIZE

    mask = counterfactual_feature_importance(original, cf)

    for pos in range(T):
        if pos in changed:
            assert mask[pos].item() == 1.0, f"Expected 1 at changed position {pos}"
        else:
            assert mask[pos].item() == 0.0, f"Expected 0 at unchanged position {pos}"


# ---------------------------------------------------------------------------
# 12. beam_search n_substitutions <= max_substitutions
# ---------------------------------------------------------------------------


def test_beam_n_substitutions_bounded(model, input_ids, cfg):
    torch.manual_seed(21)
    result = beam_search_counterfactual(model, input_ids, TARGET_TOKEN, cfg)
    assert result.n_substitutions <= cfg.max_substitutions


# ---------------------------------------------------------------------------
# 13. greedy logits shapes are correct
# ---------------------------------------------------------------------------


def test_greedy_logits_shapes(model, input_ids, cfg):
    torch.manual_seed(14)
    result = greedy_counterfactual(model, input_ids, TARGET_TOKEN, cfg)
    assert result.original_logits.shape == (T, VOCAB_SIZE)
    assert result.new_logits.shape == (T, VOCAB_SIZE)
