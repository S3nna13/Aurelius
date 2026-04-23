"""Integration tests for the head registry in ``src/model``."""

from __future__ import annotations

import torch
from torch import nn

import src.model as model_pkg
from src.model import AureliusConfig


def test_head_registry_exported_from_model_package() -> None:
    assert hasattr(model_pkg, "HEAD_REGISTRY")
    assert hasattr(model_pkg, "build_head")
    assert hasattr(model_pkg, "HeadKind")
    assert hasattr(model_pkg, "HeadSpec")
    assert "HEAD_REGISTRY" in model_pkg.__all__
    assert "build_head" in model_pkg.__all__


def test_seed_heads_reachable_via_package() -> None:
    assert "aurelius/default-lm" in model_pkg.HEAD_REGISTRY
    assert "aurelius/reward-v1" in model_pkg.HEAD_REGISTRY


def test_aurelius_config_default_unchanged() -> None:
    # Heads are variant-level, not backbone config booleans. Ensure no new
    # head-related attribute has leaked into the config surface.
    config = AureliusConfig()
    for name in vars(config):
        assert "head_registry" not in name.lower()
        assert "head_kind" not in name.lower()


def test_build_head_end_to_end() -> None:
    spec = model_pkg.get_head("aurelius/reward-v1")
    head = model_pkg.build_head(spec, d_model=16)
    assert isinstance(head, nn.Linear)
    x = torch.randn(1, 4, 16)
    y = head(x)
    assert y.shape == (1, 4, 1)
