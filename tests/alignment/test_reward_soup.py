"""Tests for reward_soup.py — Reward Model Soup (Rame et al. 2024)."""
from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from src.alignment.reward_model import RewardModel, RewardModelConfig
from src.alignment.reward_soup import (
    RewardSoupConfig,
    RewardSoup,
    weight_average_models,
    aggregate_rewards,
    evaluate_reward_diversity,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256
D_MODEL = 64
B = 2
T = 8


def make_tiny_cfg() -> AureliusConfig:
    return AureliusConfig(
        n_layers=2,
        d_model=D_MODEL,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=VOCAB_SIZE,
        max_seq_len=64,
    )


def make_reward_model(seed: int = 0) -> RewardModel:
    """Build a RewardModel backed by a tiny AureliusTransformer.

    AureliusTransformer.forward returns (loss, logits, pkv).
    RewardModel extracts out[1] = logits (B, T, vocab_size=256).
    RewardModelConfig(d_model=256) projects from vocab_size dim to scalar.
    """
    torch.manual_seed(seed)
    cfg = make_tiny_cfg()
    backbone = AureliusTransformer(cfg)

    def backbone_fn(ids: torch.Tensor) -> torch.Tensor:
        return backbone(ids)  # returns tuple; RewardModel handles it

    rm_cfg = RewardModelConfig(d_model=VOCAB_SIZE, dropout=0.0)
    return RewardModel(backbone_fn, rm_cfg)


def make_simple_model(d_in: int = 4, seed: int = 0) -> nn.Module:
    """Tiny Linear model for weight-averaging tests (no transformer)."""
    torch.manual_seed(seed)
    return nn.Linear(d_in, 1, bias=False)


def make_token_ids(seed: int = 42) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randint(0, VOCAB_SIZE, (B, T))


# ---------------------------------------------------------------------------
# 1. weight_average_models — single model yields identical weights
# ---------------------------------------------------------------------------

def test_weight_average_single_model_identical():
    torch.manual_seed(0)
    model = make_simple_model()
    soup = weight_average_models([model])

    for p_orig, p_soup in zip(model.parameters(), soup.parameters()):
        assert torch.allclose(p_orig, p_soup, atol=1e-6), (
            "Single-model soup must have identical weights"
        )


# ---------------------------------------------------------------------------
# 2. weight_average_models — two identical models → same weights
# ---------------------------------------------------------------------------

def test_weight_average_two_identical_models():
    torch.manual_seed(1)
    model = make_simple_model()
    model2 = copy.deepcopy(model)

    soup = weight_average_models([model, model2])

    for p_orig, p_soup in zip(model.parameters(), soup.parameters()):
        assert torch.allclose(p_orig, p_soup, atol=1e-6), (
            "Average of two identical models must equal the original"
        )


# ---------------------------------------------------------------------------
# 3. weight_average_models — different models → weights are average
# ---------------------------------------------------------------------------

def test_weight_average_different_models():
    torch.manual_seed(2)
    m1 = make_simple_model(seed=10)
    m2 = make_simple_model(seed=20)

    # Manually compute expected average
    expected = {}
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert n1 == n2
        expected[n1] = (p1 + p2) / 2.0

    soup = weight_average_models([m1, m2])

    for name, p_soup in soup.named_parameters():
        assert torch.allclose(p_soup, expected[name], atol=1e-6), (
            f"Parameter {name}: soup={p_soup}, expected={expected[name]}"
        )


# ---------------------------------------------------------------------------
# 4. aggregate_rewards mode="mean" computes correct mean
# ---------------------------------------------------------------------------

def test_aggregate_rewards_mean():
    torch.manual_seed(3)
    r1 = torch.tensor([1.0, 2.0, 3.0])
    r2 = torch.tensor([3.0, 4.0, 5.0])
    cfg = RewardSoupConfig(aggregation="mean")
    result = aggregate_rewards([r1, r2], cfg)
    expected = torch.tensor([2.0, 3.0, 4.0])
    assert torch.allclose(result, expected, atol=1e-6), (
        f"mean aggregation failed: {result} vs {expected}"
    )


# ---------------------------------------------------------------------------
# 5. aggregate_rewards mode="min" returns minimum
# ---------------------------------------------------------------------------

def test_aggregate_rewards_min():
    r1 = torch.tensor([1.0, 5.0, 3.0])
    r2 = torch.tensor([4.0, 2.0, 6.0])
    cfg = RewardSoupConfig(aggregation="min")
    result = aggregate_rewards([r1, r2], cfg)
    expected = torch.tensor([1.0, 2.0, 3.0])
    assert torch.allclose(result, expected, atol=1e-6), (
        f"min aggregation failed: {result} vs {expected}"
    )


# ---------------------------------------------------------------------------
# 6. aggregate_rewards mode="max" returns maximum
# ---------------------------------------------------------------------------

def test_aggregate_rewards_max():
    r1 = torch.tensor([1.0, 5.0, 3.0])
    r2 = torch.tensor([4.0, 2.0, 6.0])
    cfg = RewardSoupConfig(aggregation="max")
    result = aggregate_rewards([r1, r2], cfg)
    expected = torch.tensor([4.0, 5.0, 6.0])
    assert torch.allclose(result, expected, atol=1e-6), (
        f"max aggregation failed: {result} vs {expected}"
    )


# ---------------------------------------------------------------------------
# 7. aggregate_rewards mode="median" returns median
# ---------------------------------------------------------------------------

def test_aggregate_rewards_median():
    r1 = torch.tensor([1.0, 2.0, 3.0])
    r2 = torch.tensor([3.0, 4.0, 5.0])
    r3 = torch.tensor([2.0, 6.0, 4.0])
    cfg = RewardSoupConfig(aggregation="median")
    result = aggregate_rewards([r1, r2, r3], cfg)
    expected = torch.tensor([2.0, 4.0, 4.0])
    assert torch.allclose(result, expected, atol=1e-6), (
        f"median aggregation failed: {result} vs {expected}"
    )


# ---------------------------------------------------------------------------
# 8. RewardSoup constructs with list of models
# ---------------------------------------------------------------------------

def test_reward_soup_constructs():
    torch.manual_seed(8)
    m1 = make_reward_model(seed=0)
    m2 = make_reward_model(seed=1)
    soup = RewardSoup([m1, m2])
    assert len(soup.models) == 2
    assert isinstance(soup.config, RewardSoupConfig)


# ---------------------------------------------------------------------------
# 9. RewardSoup.score returns (N,) tensor
# ---------------------------------------------------------------------------

def test_reward_soup_score_shape():
    torch.manual_seed(9)
    m1 = make_reward_model(seed=0)
    m2 = make_reward_model(seed=1)
    soup = RewardSoup([m1, m2])

    ids = make_token_ids()
    result = soup.score(ids)
    assert result.shape == (B,), f"Expected shape ({B},), got {result.shape}"


# ---------------------------------------------------------------------------
# 10. RewardSoup.score returns finite values
# ---------------------------------------------------------------------------

def test_reward_soup_score_finite():
    torch.manual_seed(10)
    m1 = make_reward_model(seed=2)
    m2 = make_reward_model(seed=3)
    soup = RewardSoup([m1, m2])

    ids = make_token_ids()
    result = soup.score(ids)
    assert torch.isfinite(result).all(), f"score contains non-finite values: {result}"


# ---------------------------------------------------------------------------
# 11. RewardSoup.calibrate_weights returns weights summing to 1
# ---------------------------------------------------------------------------

def test_reward_soup_calibrate_weights_sum_to_one():
    torch.manual_seed(11)
    m1 = make_reward_model(seed=4)
    m2 = make_reward_model(seed=5)
    soup = RewardSoup([m1, m2])

    ids = make_token_ids()
    labels = torch.randint(0, 2, (B,)).float()
    weights = soup.calibrate_weights(ids, labels)

    assert len(weights) == 2, f"Expected 2 weights, got {len(weights)}"
    assert abs(sum(weights) - 1.0) < 1e-5, (
        f"Weights must sum to 1.0, got sum={sum(weights)}"
    )
    for w in weights:
        assert 0.0 <= w <= 1.0, f"Each weight must be in [0, 1], got {w}"


# ---------------------------------------------------------------------------
# 12. RewardSoup.distill_to_single returns nn.Module
# ---------------------------------------------------------------------------

def test_reward_soup_distill_to_single_returns_module():
    torch.manual_seed(12)
    m1 = make_reward_model(seed=6)
    m2 = make_reward_model(seed=7)
    soup = RewardSoup([m1, m2])

    distilled = soup.distill_to_single()
    assert isinstance(distilled, nn.Module), (
        f"distill_to_single must return nn.Module, got {type(distilled)}"
    )


# ---------------------------------------------------------------------------
# 13. evaluate_reward_diversity returns dict with required keys
# ---------------------------------------------------------------------------

def test_evaluate_reward_diversity_keys():
    torch.manual_seed(13)
    m1 = make_reward_model(seed=8)
    m2 = make_reward_model(seed=9)

    ids = make_token_ids()
    result = evaluate_reward_diversity([m1, m2], ids)

    required_keys = {"std_across_models", "max_disagreement", "mean_reward"}
    assert required_keys.issubset(set(result.keys())), (
        f"Missing keys: {required_keys - set(result.keys())}"
    )
    for key, val in result.items():
        assert isinstance(val, float), f"result[{key!r}] must be float, got {type(val)}"
