"""Tests for model_pruner."""

from __future__ import annotations

import pytest

from src.compression.model_pruner import (
    MODEL_PRUNER_REGISTRY,
    ModelPruner,
    PruningConfig,
    PruningStats,
    PruningStrategy,
)

# --- enum ---


def test_strategy_magnitude():
    assert PruningStrategy.MAGNITUDE == "magnitude"


def test_strategy_structured():
    assert PruningStrategy.STRUCTURED == "structured"


def test_strategy_random():
    assert PruningStrategy.RANDOM == "random"


# --- config ---


def test_config_defaults():
    cfg = PruningConfig()
    assert cfg.strategy == PruningStrategy.MAGNITUDE
    assert cfg.sparsity == 0.5
    assert cfg.target_modules == []


def test_config_custom():
    cfg = PruningConfig(sparsity=0.8, target_modules=["a", "b"])
    assert cfg.sparsity == 0.8
    assert cfg.target_modules == ["a", "b"]


# --- stats ---


def test_stats_fields():
    s = PruningStats(
        module_name="m",
        original_params=10,
        remaining_params=5,
        sparsity_achieved=0.5,
    )
    assert s.module_name == "m"
    assert s.original_params == 10
    assert s.remaining_params == 5
    assert s.sparsity_achieved == 0.5


def test_stats_frozen():
    s = PruningStats("m", 10, 5, 0.5)
    with pytest.raises(Exception):
        s.module_name = "x"  # type: ignore[misc]


# --- compute_mask MAGNITUDE ---


def test_magnitude_mask_length():
    p = ModelPruner()
    mask = p.compute_mask([1.0, 2.0, 3.0, 4.0], 0.5)
    assert len(mask) == 4


def test_magnitude_keeps_largest():
    p = ModelPruner()
    weights = [0.1, 0.2, 0.9, 0.8]
    mask = p.compute_mask(weights, 0.5, PruningStrategy.MAGNITUDE)
    # should keep 0.9 and 0.8 indices
    assert mask[2] is True
    assert mask[3] is True
    assert mask[0] is False
    assert mask[1] is False


def test_magnitude_sparsity_fraction():
    p = ModelPruner()
    weights = [float(i + 1) for i in range(10)]
    mask = p.compute_mask(weights, 0.7)
    kept = sum(1 for m in mask if m)
    assert kept == 3


def test_magnitude_empty_weights():
    p = ModelPruner()
    assert p.compute_mask([], 0.5) == []


def test_magnitude_zero_sparsity_keeps_all():
    p = ModelPruner()
    mask = p.compute_mask([1.0, 2.0, 3.0], 0.0)
    assert all(mask)


def test_magnitude_full_sparsity_keeps_none():
    p = ModelPruner()
    mask = p.compute_mask([1.0, 2.0, 3.0], 1.0)
    assert not any(mask)


# --- RANDOM ---


def test_random_mask_correct_count():
    p = ModelPruner(seed=1)
    mask = p.compute_mask([1.0] * 10, 0.5, PruningStrategy.RANDOM)
    assert sum(1 for m in mask if m) == 5


def test_random_mask_reproducible():
    p1 = ModelPruner(seed=99)
    p2 = ModelPruner(seed=99)
    m1 = p1.compute_mask([1.0] * 20, 0.5, PruningStrategy.RANDOM)
    m2 = p2.compute_mask([1.0] * 20, 0.5, PruningStrategy.RANDOM)
    assert m1 == m2


# --- STRUCTURED ---


def test_structured_keeps_prefix():
    p = ModelPruner()
    mask = p.compute_mask([1.0] * 10, 0.5, PruningStrategy.STRUCTURED)
    assert mask[:5] == [True] * 5
    assert mask[5:] == [False] * 5


def test_structured_full_prune():
    p = ModelPruner()
    mask = p.compute_mask([1.0] * 6, 1.0, PruningStrategy.STRUCTURED)
    assert not any(mask)


# --- apply_mask ---


def test_apply_mask_zeroes():
    p = ModelPruner()
    out = p.apply_mask([1.0, 2.0, 3.0], [True, False, True])
    assert out == [1.0, 0.0, 3.0]


def test_apply_mask_length_mismatch():
    p = ModelPruner()
    with pytest.raises(ValueError):
        p.apply_mask([1.0, 2.0], [True])


def test_apply_mask_all_false():
    p = ModelPruner()
    assert p.apply_mask([1.0, 2.0], [False, False]) == [0.0, 0.0]


# --- prune ---


def test_prune_returns_stats():
    p = ModelPruner(PruningConfig(sparsity=0.5))
    pruned, stats = p.prune("layer1", [1.0, 2.0, 3.0, 4.0])
    assert isinstance(stats, PruningStats)
    assert stats.module_name == "layer1"
    assert stats.original_params == 4
    assert stats.remaining_params == 2


def test_prune_sparsity_achieved():
    p = ModelPruner(PruningConfig(sparsity=0.75))
    _, stats = p.prune("m", [1.0, 2.0, 3.0, 4.0])
    assert stats.sparsity_achieved == pytest.approx(0.75)


def test_prune_zeros_match_mask():
    p = ModelPruner(PruningConfig(sparsity=0.5))
    pruned, _ = p.prune("m", [0.1, 0.2, 0.9, 0.8])
    assert pruned[0] == 0.0
    assert pruned[1] == 0.0
    assert pruned[2] == 0.9
    assert pruned[3] == 0.8


# --- achieved_sparsity ---


def test_achieved_sparsity_all_zero():
    p = ModelPruner()
    assert p.achieved_sparsity([0.0, 0.0, 0.0]) == 1.0


def test_achieved_sparsity_none_zero():
    p = ModelPruner()
    assert p.achieved_sparsity([1.0, 2.0, 3.0]) == 0.0


def test_achieved_sparsity_half():
    p = ModelPruner()
    assert p.achieved_sparsity([0.0, 1.0, 0.0, 2.0]) == 0.5


def test_achieved_sparsity_empty():
    p = ModelPruner()
    assert p.achieved_sparsity([]) == 0.0


# --- can_prune_further ---


def test_can_prune_further_true():
    p = ModelPruner()
    assert p.can_prune_further(0.3, 0.5) is True


def test_can_prune_further_false():
    p = ModelPruner()
    assert p.can_prune_further(0.5, 0.5) is False


def test_can_prune_further_tolerance():
    p = ModelPruner()
    assert p.can_prune_further(0.49, 0.5, tolerance=0.05) is False


# --- registry ---


def test_registry_has_default():
    assert "default" in MODEL_PRUNER_REGISTRY


def test_registry_constructs():
    cls = MODEL_PRUNER_REGISTRY["default"]
    assert isinstance(cls(), ModelPruner)
