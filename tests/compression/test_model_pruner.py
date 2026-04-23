"""Tests for model pruner."""
from __future__ import annotations

import pytest
from src.compression.model_pruner import (
    PruningMethod,
    PruningMask,
    ModelPruner,
    PRUNING_REGISTRY,
)


# --- PruningMethod enum ---

def test_pruning_method_magnitude():
    assert PruningMethod.MAGNITUDE == "magnitude"

def test_pruning_method_random():
    assert PruningMethod.RANDOM == "random"

def test_pruning_method_structured_head():
    assert PruningMethod.STRUCTURED_HEAD == "structured_head"

def test_pruning_method_structured_layer():
    assert PruningMethod.STRUCTURED_LAYER == "structured_layer"


# --- PruningMask fields ---

def test_pruning_mask_fields():
    mask = PruningMask(
        layer_name="layer0",
        mask=[1.0, 0.0, 1.0],
        sparsity=0.333,
        method=PruningMethod.MAGNITUDE,
    )
    assert mask.layer_name == "layer0"
    assert mask.mask == [1.0, 0.0, 1.0]
    assert abs(mask.sparsity - 0.333) < 1e-9
    assert mask.method == PruningMethod.MAGNITUDE

def test_pruning_mask_is_dataclass():
    from dataclasses import fields as dc_fields
    field_names = {f.name for f in dc_fields(PruningMask)}
    assert "layer_name" in field_names
    assert "mask" in field_names
    assert "sparsity" in field_names
    assert "method" in field_names


# --- ModelPruner.compute_mask MAGNITUDE ---

def test_compute_mask_magnitude_correct_sparsity():
    pruner = ModelPruner(target_sparsity=0.5)
    weights = [1.0, 2.0, 3.0, 4.0]
    pm = pruner.compute_mask("layer0", weights, method=PruningMethod.MAGNITUDE)
    # 50% should be pruned → 2 zeros
    zero_count = sum(1 for m in pm.mask if m == 0.0)
    assert zero_count == 2

def test_compute_mask_magnitude_keeps_largest():
    pruner = ModelPruner(target_sparsity=0.5)
    weights = [0.1, 100.0, 0.2, 50.0]
    pm = pruner.compute_mask("layer0", weights, method=PruningMethod.MAGNITUDE)
    # 100.0 (idx 1) and 50.0 (idx 3) should be kept
    assert pm.mask[1] == 1.0
    assert pm.mask[3] == 1.0

def test_compute_mask_magnitude_only_01():
    pruner = ModelPruner(target_sparsity=0.5)
    weights = [1.0, 2.0, 3.0, 4.0]
    pm = pruner.compute_mask("layer0", weights, method=PruningMethod.MAGNITUDE)
    assert all(m in (0.0, 1.0) for m in pm.mask)

def test_compute_mask_magnitude_layer_name():
    pruner = ModelPruner(target_sparsity=0.5)
    pm = pruner.compute_mask("my_layer", [1.0, 2.0])
    assert pm.layer_name == "my_layer"

def test_compute_mask_magnitude_method_recorded():
    pruner = ModelPruner(target_sparsity=0.5)
    pm = pruner.compute_mask("layer", [1.0, 2.0, 3.0], method=PruningMethod.MAGNITUDE)
    assert pm.method == PruningMethod.MAGNITUDE


# --- Mask values are only 0.0 or 1.0 ---

def test_mask_values_random_only_01():
    pruner = ModelPruner(target_sparsity=0.5)
    pm = pruner.compute_mask("layer", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], method=PruningMethod.RANDOM)
    assert all(m in (0.0, 1.0) for m in pm.mask)

def test_mask_values_structured_head_only_01():
    pruner = ModelPruner(target_sparsity=0.5)
    pm = pruner.compute_mask("layer", [1.0, 2.0, 3.0, 4.0], method=PruningMethod.STRUCTURED_HEAD)
    assert all(m in (0.0, 1.0) for m in pm.mask)

def test_mask_values_structured_layer_keep_only_01():
    pruner = ModelPruner(target_sparsity=0.5)
    pm = pruner.compute_mask("layer", [1.0, 2.0, 3.0], method=PruningMethod.STRUCTURED_LAYER)
    assert all(m in (0.0, 1.0) for m in pm.mask)

def test_mask_values_structured_layer_prune_only_01():
    pruner = ModelPruner(target_sparsity=0.9)
    pm = pruner.compute_mask("layer", [1.0, 2.0, 3.0], method=PruningMethod.STRUCTURED_LAYER)
    assert all(m in (0.0, 1.0) for m in pm.mask)


# --- apply_mask ---

def test_apply_mask_zeros_pruned():
    pruner = ModelPruner()
    weights = [1.0, 2.0, 3.0, 4.0]
    mask = [1.0, 0.0, 1.0, 0.0]
    result = pruner.apply_mask(weights, mask)
    assert result[1] == 0.0
    assert result[3] == 0.0

def test_apply_mask_preserves_unpruned():
    pruner = ModelPruner()
    weights = [1.0, 2.0, 3.0, 4.0]
    mask = [1.0, 0.0, 1.0, 0.0]
    result = pruner.apply_mask(weights, mask)
    assert result[0] == 1.0
    assert result[2] == 3.0

def test_apply_mask_all_keep():
    pruner = ModelPruner()
    weights = [1.0, 2.0, 3.0]
    mask = [1.0, 1.0, 1.0]
    result = pruner.apply_mask(weights, mask)
    assert result == weights

def test_apply_mask_all_prune():
    pruner = ModelPruner()
    weights = [1.0, 2.0, 3.0]
    mask = [0.0, 0.0, 0.0]
    result = pruner.apply_mask(weights, mask)
    assert all(v == 0.0 for v in result)

def test_apply_mask_same_length():
    pruner = ModelPruner()
    weights = [1.0, 2.0, 3.0]
    mask = [1.0, 0.0, 1.0]
    assert len(pruner.apply_mask(weights, mask)) == 3


# --- sparsity_schedule ---

def test_sparsity_schedule_zero_at_warmup():
    pruner = ModelPruner()
    val = pruner.sparsity_schedule(current_step=0, total_steps=1000, final_sparsity=0.5, warmup_steps=100)
    assert val == 0.0

def test_sparsity_schedule_zero_before_warmup():
    pruner = ModelPruner()
    val = pruner.sparsity_schedule(current_step=50, total_steps=1000, final_sparsity=0.5, warmup_steps=100)
    assert val == 0.0

def test_sparsity_schedule_approaches_final():
    pruner = ModelPruner()
    val = pruner.sparsity_schedule(current_step=1000, total_steps=1000, final_sparsity=0.5, warmup_steps=100)
    assert abs(val - 0.5) < 1e-9

def test_sparsity_schedule_monotone():
    pruner = ModelPruner()
    vals = [
        pruner.sparsity_schedule(step, 1000, 0.5, 100)
        for step in [100, 200, 400, 700, 1000]
    ]
    for i in range(len(vals) - 1):
        assert vals[i] <= vals[i + 1]

def test_sparsity_schedule_cubic():
    pruner = ModelPruner()
    # At 50% progress past warmup, cubic gives 1-(0.5)^3=0.875
    val = pruner.sparsity_schedule(current_step=550, total_steps=1000, final_sparsity=0.5, warmup_steps=100)
    expected = 0.5 * (1 - (1 - 0.5) ** 3)
    assert abs(val - expected) < 1e-9


# --- global_sparsity ---

def test_global_sparsity_empty():
    pruner = ModelPruner()
    assert pruner.global_sparsity([]) == 0.0

def test_global_sparsity_single():
    pruner = ModelPruner()
    pm = PruningMask(layer_name="l0", mask=[0.0, 1.0], sparsity=0.5, method=PruningMethod.MAGNITUDE)
    assert abs(pruner.global_sparsity([pm]) - 0.5) < 1e-9

def test_global_sparsity_weighted_average():
    pruner = ModelPruner()
    pm1 = PruningMask(layer_name="l0", mask=[0.0] * 10, sparsity=1.0, method=PruningMethod.MAGNITUDE)
    pm2 = PruningMask(layer_name="l1", mask=[1.0] * 10, sparsity=0.0, method=PruningMethod.MAGNITUDE)
    result = pruner.global_sparsity([pm1, pm2])
    assert abs(result - 0.5) < 1e-9

def test_global_sparsity_different_sizes():
    pruner = ModelPruner()
    pm1 = PruningMask(layer_name="l0", mask=[0.0] * 4, sparsity=1.0, method=PruningMethod.MAGNITUDE)
    pm2 = PruningMask(layer_name="l1", mask=[1.0] * 2, sparsity=0.0, method=PruningMethod.MAGNITUDE)
    # weighted: (4*1.0 + 2*0.0)/6 = 4/6 = 0.667
    result = pruner.global_sparsity([pm1, pm2])
    assert abs(result - (4.0 / 6.0)) < 1e-9


# --- PRUNING_REGISTRY ---

def test_pruning_registry_has_default():
    assert "default" in PRUNING_REGISTRY

def test_pruning_registry_has_aggressive():
    assert "aggressive" in PRUNING_REGISTRY

def test_pruning_registry_has_light():
    assert "light" in PRUNING_REGISTRY

def test_pruning_registry_default_sparsity():
    assert PRUNING_REGISTRY["default"].target_sparsity == 0.5

def test_pruning_registry_aggressive_sparsity():
    assert PRUNING_REGISTRY["aggressive"].target_sparsity == 0.9

def test_pruning_registry_light_sparsity():
    assert PRUNING_REGISTRY["light"].target_sparsity == 0.2

def test_pruning_registry_values_are_pruners():
    for key, val in PRUNING_REGISTRY.items():
        assert isinstance(val, ModelPruner), f"{key} is not a ModelPruner"
