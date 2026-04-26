"""Tests for src/alignment/reward_soup_ensemble.py — 12 tests."""

from __future__ import annotations

import pytest
import torch

from src.alignment.reward_soup_ensemble import RewardSoupConfig, RewardSoupEnsemble


def _make_ensemble(mode: str = "mean", n: int = 3, weights=None) -> RewardSoupEnsemble:
    return RewardSoupEnsemble(RewardSoupConfig(n_models=n, aggregation=mode, weights=weights))


# ---------------------------------------------------------------------------
# RewardSoupConfig
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = RewardSoupConfig()
    assert cfg.n_models == 3
    assert cfg.aggregation == "mean"
    assert cfg.weights is None


def test_config_custom():
    cfg = RewardSoupConfig(n_models=5, aggregation="weighted", weights=[0.2] * 5)
    assert cfg.n_models == 5
    assert cfg.aggregation == "weighted"


# ---------------------------------------------------------------------------
# add_model_scores / aggregate — mean
# ---------------------------------------------------------------------------


def test_aggregate_mean_basic():
    ens = _make_ensemble("mean")
    ens.add_model_scores([1.0, 2.0])
    ens.add_model_scores([3.0, 4.0])
    result = ens.aggregate()
    assert result == pytest.approx([2.0, 3.0])


def test_aggregate_clears_buffer():
    ens = _make_ensemble("mean")
    ens.add_model_scores([1.0])
    ens.aggregate()
    assert ens._score_buffer == []


def test_aggregate_empty_buffer():
    ens = _make_ensemble("mean")
    assert ens.aggregate() == []


# ---------------------------------------------------------------------------
# aggregate — weighted
# ---------------------------------------------------------------------------


def test_aggregate_weighted():
    ens = _make_ensemble("weighted", n=2, weights=[0.8, 0.2])
    ens.add_model_scores([1.0, 0.0])
    ens.add_model_scores([0.0, 1.0])
    result = ens.aggregate()
    assert result == pytest.approx([0.8, 0.2])


def test_aggregate_weighted_default_equal():
    """Weighted with no explicit weights falls back to equal."""
    ens = _make_ensemble("weighted", n=2)
    ens.add_model_scores([2.0])
    ens.add_model_scores([4.0])
    result = ens.aggregate()
    assert result == pytest.approx([3.0])


# ---------------------------------------------------------------------------
# aggregate — min / max
# ---------------------------------------------------------------------------


def test_aggregate_min():
    ens = _make_ensemble("min")
    ens.add_model_scores([1.0, 5.0])
    ens.add_model_scores([3.0, 2.0])
    result = ens.aggregate()
    assert result == pytest.approx([1.0, 2.0])


def test_aggregate_max():
    ens = _make_ensemble("max")
    ens.add_model_scores([1.0, 5.0])
    ens.add_model_scores([3.0, 2.0])
    result = ens.aggregate()
    assert result == pytest.approx([3.0, 5.0])


def test_aggregate_unknown_mode_raises():
    ens = _make_ensemble("median")
    ens.add_model_scores([1.0])
    with pytest.raises(ValueError):
        ens.aggregate()


# ---------------------------------------------------------------------------
# interpolate_weights
# ---------------------------------------------------------------------------


def test_interpolate_weights_equal():
    ens = _make_ensemble()
    sd1 = {"w": torch.tensor([1.0, 2.0])}
    sd2 = {"w": torch.tensor([3.0, 4.0])}
    merged = ens.interpolate_weights([sd1, sd2])
    assert torch.allclose(merged["w"], torch.tensor([2.0, 3.0]))


def test_interpolate_weights_custom():
    ens = _make_ensemble()
    sd1 = {"w": torch.tensor([0.0, 0.0])}
    sd2 = {"w": torch.tensor([10.0, 10.0])}
    merged = ens.interpolate_weights([sd1, sd2], weights=[0.9, 0.1])
    assert torch.allclose(merged["w"], torch.tensor([1.0, 1.0]))


def test_interpolate_weights_empty_raises():
    ens = _make_ensemble()
    with pytest.raises(ValueError):
        ens.interpolate_weights([])


# ---------------------------------------------------------------------------
# score_batch
# ---------------------------------------------------------------------------


def test_score_batch_mean():
    ens = _make_ensemble("mean")
    result = ens.score_batch([[1.0, 2.0], [3.0, 4.0]])
    assert result == pytest.approx([2.0, 3.0])


def test_score_batch_clears_buffer():
    ens = _make_ensemble("mean")
    ens.score_batch([[1.0], [2.0]])
    assert ens._score_buffer == []
