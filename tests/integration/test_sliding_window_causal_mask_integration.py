"""Integration: sliding-window mask in longcontext registry."""

from __future__ import annotations

import torch

import src.longcontext as lc
from src.model.config import AureliusConfig


def test_registry():
    assert lc.LONGCONTEXT_STRATEGY_REGISTRY["swa_causal_mask"] is lc.SlidingWindowCausalMaskBuilder


def test_config_defaults():
    c = AureliusConfig()
    assert c.longcontext_sliding_window_causal_mask_enabled is False
    assert c.longcontext_sliding_window_size == 512


def test_prior_registry_entries():
    assert "ring_attention" in lc.LONGCONTEXT_STRATEGY_REGISTRY


def test_smoke_with_flag_enabled():
    cfg = AureliusConfig(longcontext_sliding_window_causal_mask_enabled=True)
    assert cfg.longcontext_sliding_window_causal_mask_enabled is True
    b = lc.SlidingWindowCausalMaskBuilder(window_size=min(8, cfg.longcontext_sliding_window_size))
    m = b.build(32, dtype=torch.float32)
    assert m.shape == (1, 1, 32, 32)
