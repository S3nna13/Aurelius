"""Tests for src/training/gradient_checkpoint_manager.py."""

import pytest
from src.training.gradient_checkpoint_manager import (
    CheckpointConfig,
    CheckpointPolicy,
    GradientCheckpointManager,
)


# ---------------------------------------------------------------------------
# CheckpointPolicy enum
# ---------------------------------------------------------------------------

class TestCheckpointPolicy:
    def test_none_value(self):
        assert CheckpointPolicy.NONE == "none"

    def test_every_layer_value(self):
        assert CheckpointPolicy.EVERY_LAYER == "every_layer"

    def test_every_k_layers_value(self):
        assert CheckpointPolicy.EVERY_K_LAYERS == "every_k_layers"

    def test_custom_value(self):
        assert CheckpointPolicy.CUSTOM == "custom"

    def test_four_members(self):
        assert len(CheckpointPolicy) == 4

    def test_is_str(self):
        assert isinstance(CheckpointPolicy.NONE, str)


# ---------------------------------------------------------------------------
# CheckpointConfig dataclass
# ---------------------------------------------------------------------------

class TestCheckpointConfig:
    def test_default_policy(self):
        cfg = CheckpointConfig()
        assert cfg.policy == CheckpointPolicy.EVERY_K_LAYERS

    def test_default_k(self):
        cfg = CheckpointConfig()
        assert cfg.k == 2

    def test_default_memory_budget(self):
        cfg = CheckpointConfig()
        assert cfg.memory_budget_gb == 8.0

    def test_custom_policy(self):
        cfg = CheckpointConfig(policy=CheckpointPolicy.NONE)
        assert cfg.policy == CheckpointPolicy.NONE

    def test_custom_k(self):
        cfg = CheckpointConfig(k=4)
        assert cfg.k == 4

    def test_custom_memory_budget(self):
        cfg = CheckpointConfig(memory_budget_gb=16.0)
        assert cfg.memory_budget_gb == 16.0


# ---------------------------------------------------------------------------
# GradientCheckpointManager
# ---------------------------------------------------------------------------

class TestNonePolicy:
    def test_none_policy_returns_empty(self):
        cfg = CheckpointConfig(policy=CheckpointPolicy.NONE)
        mgr = GradientCheckpointManager(n_layers=12, config=cfg)
        assert mgr.checkpointed_layers() == []

    def test_none_policy_always_empty_regardless_of_n(self):
        cfg = CheckpointConfig(policy=CheckpointPolicy.NONE)
        mgr = GradientCheckpointManager(n_layers=0, config=cfg)
        assert mgr.checkpointed_layers() == []


class TestEveryLayerPolicy:
    def test_every_layer_returns_all_indices(self):
        cfg = CheckpointConfig(policy=CheckpointPolicy.EVERY_LAYER)
        mgr = GradientCheckpointManager(n_layers=4, config=cfg)
        assert mgr.checkpointed_layers() == [0, 1, 2, 3]

    def test_every_layer_length_equals_n_layers(self):
        cfg = CheckpointConfig(policy=CheckpointPolicy.EVERY_LAYER)
        mgr = GradientCheckpointManager(n_layers=10, config=cfg)
        assert len(mgr.checkpointed_layers()) == 10

    def test_every_layer_zero_layers(self):
        cfg = CheckpointConfig(policy=CheckpointPolicy.EVERY_LAYER)
        mgr = GradientCheckpointManager(n_layers=0, config=cfg)
        assert mgr.checkpointed_layers() == []


class TestEveryKLayersPolicy:
    def test_k2_correct_indices(self):
        cfg = CheckpointConfig(policy=CheckpointPolicy.EVERY_K_LAYERS, k=2)
        mgr = GradientCheckpointManager(n_layers=6, config=cfg)
        assert mgr.checkpointed_layers() == [0, 2, 4]

    def test_k3_correct_indices(self):
        cfg = CheckpointConfig(policy=CheckpointPolicy.EVERY_K_LAYERS, k=3)
        mgr = GradientCheckpointManager(n_layers=9, config=cfg)
        assert mgr.checkpointed_layers() == [0, 3, 6]

    def test_k1_same_as_every_layer(self):
        cfg = CheckpointConfig(policy=CheckpointPolicy.EVERY_K_LAYERS, k=1)
        mgr = GradientCheckpointManager(n_layers=5, config=cfg)
        assert mgr.checkpointed_layers() == [0, 1, 2, 3, 4]

    def test_k_larger_than_n(self):
        cfg = CheckpointConfig(policy=CheckpointPolicy.EVERY_K_LAYERS, k=10)
        mgr = GradientCheckpointManager(n_layers=5, config=cfg)
        assert mgr.checkpointed_layers() == [0]

    def test_default_config_k2(self):
        mgr = GradientCheckpointManager(n_layers=8)
        layers = mgr.checkpointed_layers()
        assert layers == [0, 2, 4, 6]


class TestCustomPolicy:
    def test_custom_returns_set_layers(self):
        cfg = CheckpointConfig(policy=CheckpointPolicy.CUSTOM)
        mgr = GradientCheckpointManager(n_layers=12, config=cfg)
        mgr.set_custom_layers([1, 3, 7])
        assert mgr.checkpointed_layers() == [1, 3, 7]

    def test_custom_before_set_returns_empty(self):
        cfg = CheckpointConfig(policy=CheckpointPolicy.CUSTOM)
        mgr = GradientCheckpointManager(n_layers=12, config=cfg)
        assert mgr.checkpointed_layers() == []

    def test_set_custom_layers_replaces_previous(self):
        cfg = CheckpointConfig(policy=CheckpointPolicy.CUSTOM)
        mgr = GradientCheckpointManager(n_layers=12, config=cfg)
        mgr.set_custom_layers([0, 1])
        mgr.set_custom_layers([5, 6, 7])
        assert mgr.checkpointed_layers() == [5, 6, 7]


class TestMemorySavings:
    def test_none_policy_zero_savings(self):
        cfg = CheckpointConfig(policy=CheckpointPolicy.NONE)
        mgr = GradientCheckpointManager(n_layers=12, config=cfg)
        assert mgr.memory_savings_estimate(n_layers=12) == 0.0

    def test_every_layer_savings(self):
        cfg = CheckpointConfig(policy=CheckpointPolicy.EVERY_LAYER)
        mgr = GradientCheckpointManager(n_layers=4, config=cfg)
        assert mgr.memory_savings_estimate(n_layers=4, layer_mem_gb=0.5) == 2.0

    def test_every_k_layers_savings(self):
        cfg = CheckpointConfig(policy=CheckpointPolicy.EVERY_K_LAYERS, k=2)
        mgr = GradientCheckpointManager(n_layers=6, config=cfg)
        # layers [0,2,4] = 3 checkpointed
        assert mgr.memory_savings_estimate(n_layers=6, layer_mem_gb=0.5) == 1.5

    def test_custom_savings(self):
        cfg = CheckpointConfig(policy=CheckpointPolicy.CUSTOM)
        mgr = GradientCheckpointManager(n_layers=12, config=cfg)
        mgr.set_custom_layers([1, 3, 5])
        assert mgr.memory_savings_estimate(n_layers=12, layer_mem_gb=1.0) == 3.0


class TestRecomputeCost:
    def test_zero_checkpointed(self):
        mgr = GradientCheckpointManager(n_layers=12)
        assert mgr.recompute_cost_estimate(0) == 0.0

    def test_cost_equals_n_times_base(self):
        mgr = GradientCheckpointManager(n_layers=12)
        assert mgr.recompute_cost_estimate(5, base_forward_cost=2.0) == 10.0

    def test_default_base_cost(self):
        mgr = GradientCheckpointManager(n_layers=12)
        assert mgr.recompute_cost_estimate(3) == 3.0
