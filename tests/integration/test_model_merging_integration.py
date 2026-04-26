"""Integration tests for weight-space model merging."""

from __future__ import annotations

import torch

from src.model import (
    MERGING_REGISTRY,
    AureliusConfig,
    MergeResult,
    MergeStrategy,
    ModelMerger,
    dare_merge,
    linear_merge,
    slerp_merge,
    ties_merge,
)
from src.runtime.feature_flags import FEATURE_FLAG_REGISTRY, FeatureFlag


def test_merging_registry_contains_all_strategies() -> None:
    assert set(MERGING_REGISTRY.keys()) >= {"linear", "slerp", "ties", "dare"}
    assert MERGING_REGISTRY["linear"] is linear_merge
    assert MERGING_REGISTRY["slerp"] is slerp_merge
    assert MERGING_REGISTRY["ties"] is ties_merge
    assert MERGING_REGISTRY["dare"] is dare_merge


def test_model_merging_config_flag_defaults_off() -> None:
    cfg = AureliusConfig()
    assert cfg.model_merging_enabled is False
    FEATURE_FLAG_REGISTRY.register(FeatureFlag(name="model.merging", enabled=True))
    cfg2 = AureliusConfig()
    assert cfg2.model_merging_enabled is True


def test_end_to_end_linear_via_modelmerger() -> None:
    # two pretend-checkpoints with matching keys and shapes
    a = {
        "embed.weight": torch.zeros(8, 4),
        "blocks.0.attn.q.weight": torch.ones(4, 4),
        "ln.weight": torch.ones(4),
    }
    b = {
        "embed.weight": torch.ones(8, 4),
        "blocks.0.attn.q.weight": torch.full((4, 4), 3.0),
        "ln.weight": torch.full((4,), 5.0),
    }
    merger = ModelMerger(MergeStrategy.LINEAR, weights=[1.0, 1.0])
    result = merger.merge([a, b], names=("ckpt_a", "ckpt_b"))
    assert isinstance(result, MergeResult)
    assert result.contributors == ("ckpt_a", "ckpt_b")
    assert torch.allclose(result.state_dict["embed.weight"], torch.full((8, 4), 0.5))
    assert torch.allclose(result.state_dict["blocks.0.attn.q.weight"], torch.full((4, 4), 2.0))
    assert torch.allclose(result.state_dict["ln.weight"], torch.full((4,), 3.0))
    assert set(result.state_dict.keys()) == set(a.keys())
