"""Tests for src/training/param_group_builder.py"""

import torch.nn as nn

from src.training.param_group_builder import (
    PARAM_GROUP_BUILDER,
    ParamGroupBuilder,
    ParamGroupConfig,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 16)  # weight (2D) + bias (1D)
        self.norm = nn.LayerNorm(16)  # weight (1D) + bias (1D)
        self.embedding = nn.Embedding(100, 16)  # weight (2D) but name matches pattern

    def forward(self, x):
        return self.norm(self.linear(x))


class FrozenParamModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = nn.Linear(8, 8)
        self.trainable = nn.Linear(8, 8)

    def forward(self, x):
        return self.trainable(self.frozen(x))


# ---------------------------------------------------------------------------
# Tests: build()
# ---------------------------------------------------------------------------


def test_build_returns_two_groups():
    model = SimpleModel()
    cfg = ParamGroupConfig()
    builder = ParamGroupBuilder()
    groups = builder.build(model, cfg)
    assert len(groups) == 2


def test_build_decay_group_has_weight_decay():
    model = SimpleModel()
    cfg = ParamGroupConfig(weight_decay=0.1)
    builder = ParamGroupBuilder()
    groups = builder.build(model, cfg)
    assert groups[0]["weight_decay"] == 0.1


def test_build_no_decay_group_has_zero_weight_decay():
    model = SimpleModel()
    cfg = ParamGroupConfig(weight_decay=0.1)
    builder = ParamGroupBuilder()
    groups = builder.build(model, cfg)
    assert groups[1]["weight_decay"] == 0.0


def test_build_1d_params_go_to_no_decay():
    """Bias and LayerNorm weight are 1D → no-decay group."""
    model = nn.Linear(16, 16)
    cfg = ParamGroupConfig(no_decay_patterns=())  # disable pattern matching
    builder = ParamGroupBuilder()
    groups = builder.build(model, cfg)
    decay_shapes = [p.shape for p in groups[0]["params"]]
    no_decay_shapes = [p.shape for p in groups[1]["params"]]
    assert all(len(s) > 1 for s in decay_shapes)  # decay: multi-dim only
    assert all(len(s) == 1 for s in no_decay_shapes)  # no-decay: 1D only


def test_build_bias_in_name_goes_to_no_decay():
    model = SimpleModel()
    cfg = ParamGroupConfig()
    builder = ParamGroupBuilder()
    groups = builder.build(model, cfg)
    # Verify bias params are not in the decay group
    decay_ids = {id(p) for p in groups[0]["params"]}
    for name, param in model.named_parameters():
        if "bias" in name:
            assert id(param) not in decay_ids, f"{name} should be in no-decay"


def test_build_embedding_in_name_goes_to_no_decay():
    model = SimpleModel()
    cfg = ParamGroupConfig()
    builder = ParamGroupBuilder()
    groups = builder.build(model, cfg)
    no_decay_ids = {id(p) for p in groups[1]["params"]}
    for name, param in model.named_parameters():
        if "embedding" in name:
            assert id(param) in no_decay_ids, f"{name} should be in no-decay"


def test_build_frozen_params_excluded():
    model = FrozenParamModel()
    for p in model.frozen.parameters():
        p.requires_grad = False
    cfg = ParamGroupConfig()
    builder = ParamGroupBuilder()
    groups = builder.build(model, cfg)
    all_params = groups[0]["params"] + groups[1]["params"]
    frozen_ids = {id(p) for p in model.frozen.parameters()}
    for p in all_params:
        assert id(p) not in frozen_ids, "Frozen param must not appear in any group"


def test_build_all_trainable_params_covered():
    model = SimpleModel()
    cfg = ParamGroupConfig()
    builder = ParamGroupBuilder()
    groups = builder.build(model, cfg)
    covered_ids = {id(p) for p in groups[0]["params"]} | {id(p) for p in groups[1]["params"]}
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert id(param) in covered_ids, f"{name} missing from groups"


def test_build_custom_no_decay_patterns():
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
    cfg = ParamGroupConfig(no_decay_patterns=("weight",))
    builder = ParamGroupBuilder()
    groups = builder.build(model, cfg)
    # All 2D weight tensors whose name contains "weight" → no-decay
    no_decay_ids = {id(p) for p in groups[1]["params"]}
    for name, param in model.named_parameters():
        if "weight" in name:
            assert id(param) in no_decay_ids


# ---------------------------------------------------------------------------
# Tests: count_params()
# ---------------------------------------------------------------------------


def test_count_params_total_matches_sum():
    model = SimpleModel()
    builder = ParamGroupBuilder()
    counts = builder.count_params(model)
    expected = sum(p.numel() for p in model.parameters())
    assert counts["total"] == expected


def test_count_params_frozen_zero_when_all_trainable():
    model = SimpleModel()
    builder = ParamGroupBuilder()
    counts = builder.count_params(model)
    assert counts["frozen"] == 0
    assert counts["trainable"] == counts["total"]


def test_count_params_frozen_count():
    model = FrozenParamModel()
    frozen_count = sum(p.numel() for p in model.frozen.parameters())
    for p in model.frozen.parameters():
        p.requires_grad = False
    builder = ParamGroupBuilder()
    counts = builder.count_params(model)
    assert counts["frozen"] == frozen_count


# ---------------------------------------------------------------------------
# Tests: get_param_stats()
# ---------------------------------------------------------------------------


def test_get_param_stats_returns_dict():
    model = SimpleModel()
    builder = ParamGroupBuilder()
    stats = builder.get_param_stats(model)
    assert isinstance(stats, dict)


def test_get_param_stats_keys_are_param_names():
    model = SimpleModel()
    builder = ParamGroupBuilder()
    stats = builder.get_param_stats(model)
    expected_names = {name for name, _ in model.named_parameters()}
    assert set(stats.keys()) == expected_names


def test_get_param_stats_values_are_ints():
    model = SimpleModel()
    builder = ParamGroupBuilder()
    stats = builder.get_param_stats(model)
    for name, count in stats.items():
        assert isinstance(count, int), f"{name} count should be int"


# ---------------------------------------------------------------------------
# Tests: module-level singleton
# ---------------------------------------------------------------------------


def test_singleton_is_instance():
    assert isinstance(PARAM_GROUP_BUILDER, ParamGroupBuilder)
