"""
Tests for src/training/layer_wise_lr.py

All tests use tiny configs to stay fast and dependency-free.
Only pure PyTorch is required.
"""

from __future__ import annotations

import math
from typing import List

import pytest
import torch
import torch.nn as nn

from src.training.layer_wise_lr import (
    LLRDConfig,
    LayerWiseLRScheduler,
    assign_layer_params,
    build_llrd_param_groups,
    compute_layer_lrs,
    compute_lr_ratio_stats,
)


# ---------------------------------------------------------------------------
# Tiny helper model
# ---------------------------------------------------------------------------

class TinyTransformer(nn.Module):
    """Minimal model with embedding + 3 layers + head for testing."""

    def __init__(self, n_layers: int = 3, dim: int = 4) -> None:
        super().__init__()
        self.embedding = nn.Embedding(10, dim)
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])
        self.head = nn.Linear(dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        out = self.embedding(x)
        for layer in self.layers:
            out = layer(out)
        return self.head(out)


def _tiny_config(n_layers: int = 4, decay: float = 0.8) -> LLRDConfig:
    return LLRDConfig(
        base_lr=1e-3,
        decay_factor=decay,
        n_layers=n_layers,
        min_lr_ratio=0.01,
        embedding_lr_ratio=0.1,
    )


def _layer_patterns(n_layers: int) -> List[str]:
    return [f"layers.{i}" for i in range(n_layers)]


# ===========================================================================
# 1. LLRDConfig defaults
# ===========================================================================

def test_llrd_config_defaults() -> None:
    cfg = LLRDConfig()
    assert cfg.base_lr == pytest.approx(1e-3)
    assert cfg.decay_factor == pytest.approx(0.9)
    assert cfg.n_layers == 12
    assert cfg.min_lr_ratio == pytest.approx(0.01)
    assert cfg.embedding_lr_ratio == pytest.approx(0.1)


# ===========================================================================
# 2. compute_layer_lrs — length equals n_layers
# ===========================================================================

def test_compute_layer_lrs_length() -> None:
    for n in (1, 4, 8, 12):
        cfg = LLRDConfig(n_layers=n)
        lrs = compute_layer_lrs(cfg)
        assert len(lrs) == n, f"expected {n} lrs, got {len(lrs)}"


# ===========================================================================
# 3. compute_layer_lrs — layer 0 equals base_lr
# ===========================================================================

def test_compute_layer_lrs_layer0_equals_base_lr() -> None:
    cfg = _tiny_config()
    lrs = compute_layer_lrs(cfg)
    assert lrs[0] == pytest.approx(cfg.base_lr)


# ===========================================================================
# 4. compute_layer_lrs — strictly non-increasing (each lr <= previous)
# ===========================================================================

def test_compute_layer_lrs_non_increasing() -> None:
    cfg = _tiny_config(n_layers=6, decay=0.7)
    lrs = compute_layer_lrs(cfg)
    for i in range(1, len(lrs)):
        assert lrs[i] <= lrs[i - 1] + 1e-12, (
            f"lrs[{i}]={lrs[i]} > lrs[{i-1}]={lrs[i-1]}"
        )


# ===========================================================================
# 5. compute_layer_lrs — all >= min_lr_ratio * base_lr
# ===========================================================================

def test_compute_layer_lrs_min_lr_floor() -> None:
    # Use very small decay to force clamping for deep layers.
    cfg = LLRDConfig(base_lr=1.0, decay_factor=0.5, n_layers=20, min_lr_ratio=0.05)
    lrs = compute_layer_lrs(cfg)
    floor = cfg.base_lr * cfg.min_lr_ratio
    for i, lr in enumerate(lrs):
        assert lr >= floor - 1e-12, f"lrs[{i}]={lr} is below floor {floor}"


# ===========================================================================
# 6. assign_layer_params — groups length = n_layers + 1
# ===========================================================================

def test_assign_layer_params_groups_length() -> None:
    model = TinyTransformer(n_layers=3)
    named_params = list(model.named_parameters())
    patterns = _layer_patterns(3)
    groups = assign_layer_params(named_params, patterns)
    assert len(groups) == len(patterns) + 1


# ===========================================================================
# 7. assign_layer_params — params assigned to correct layer by pattern
# ===========================================================================

def test_assign_layer_params_correct_assignment() -> None:
    model = TinyTransformer(n_layers=3)
    named_params = list(model.named_parameters())
    patterns = _layer_patterns(3)
    groups = assign_layer_params(named_params, patterns)

    # layers.0.weight and layers.0.bias must be in group 0
    group0_names = {name for name, _ in groups[0]}
    assert "layers.0.weight" in group0_names
    assert "layers.0.bias" in group0_names

    # layers.2.weight must be in group 2
    group2_names = {name for name, _ in groups[2]}
    assert "layers.2.weight" in group2_names


# ===========================================================================
# 8. assign_layer_params — unmatched params go to last group
# ===========================================================================

def test_assign_layer_params_unmatched_go_to_last_group() -> None:
    model = TinyTransformer(n_layers=3)
    named_params = list(model.named_parameters())
    patterns = _layer_patterns(3)
    groups = assign_layer_params(named_params, patterns)

    # "embedding" and "head" params match none of the layer patterns
    last_group_names = {name for name, _ in groups[-1]}
    assert "embedding.weight" in last_group_names
    assert "head.weight" in last_group_names


# ===========================================================================
# 9. build_llrd_param_groups — correct number of groups
# ===========================================================================

def test_build_llrd_param_groups_num_groups() -> None:
    n_layers = 3
    model = TinyTransformer(n_layers=n_layers)
    cfg = LLRDConfig(n_layers=n_layers)
    patterns = _layer_patterns(n_layers)
    groups = build_llrd_param_groups(model, cfg, patterns)
    # n_layers groups + 1 "other" group
    assert len(groups) == n_layers + 1


# ===========================================================================
# 10. build_llrd_param_groups — all lrs are positive
# ===========================================================================

def test_build_llrd_param_groups_lrs_positive() -> None:
    n_layers = 3
    model = TinyTransformer(n_layers=n_layers)
    cfg = LLRDConfig(n_layers=n_layers, base_lr=1e-3, min_lr_ratio=0.01)
    patterns = _layer_patterns(n_layers)
    groups = build_llrd_param_groups(model, cfg, patterns)
    for i, group in enumerate(groups):
        assert group["lr"] > 0, f"group {i} lr={group['lr']} is not positive"


# ===========================================================================
# 11. LayerWiseLRScheduler.get_lrs — length matches optimizer param groups
# ===========================================================================

def test_scheduler_get_lrs_length() -> None:
    n_layers = 3
    model = TinyTransformer(n_layers=n_layers)
    cfg = LLRDConfig(n_layers=n_layers)
    patterns = _layer_patterns(n_layers)
    param_groups = build_llrd_param_groups(model, cfg, patterns)
    layer_lrs = compute_layer_lrs(cfg)

    optimizer = torch.optim.AdamW(param_groups)
    scheduler = LayerWiseLRScheduler(optimizer, cfg, layer_lrs)

    assert len(scheduler.get_lrs()) == len(optimizer.param_groups)


# ===========================================================================
# 12. LayerWiseLRScheduler.step — updates lrs (not all the same after step)
# ===========================================================================

def test_scheduler_step_updates_lrs() -> None:
    n_layers = 4
    model = TinyTransformer(n_layers=n_layers)
    cfg = LLRDConfig(n_layers=n_layers, base_lr=1e-3, decay_factor=0.8)
    patterns = _layer_patterns(n_layers)
    param_groups = build_llrd_param_groups(model, cfg, patterns)
    layer_lrs = compute_layer_lrs(cfg)

    optimizer = torch.optim.AdamW(param_groups)
    scheduler = LayerWiseLRScheduler(optimizer, cfg, layer_lrs)

    lrs_before = scheduler.get_lrs()
    # Call step in middle of training (no warmup), should apply cosine scaling
    scheduler.step(global_step=500, warmup_steps=0, total_steps=1000)
    lrs_after = scheduler.get_lrs()

    # At least some lrs should have changed from the initial values
    changed = any(
        abs(before - after) > 1e-15
        for before, after in zip(lrs_before, lrs_after)
    )
    assert changed, "No lr changed after scheduler.step()"


# ===========================================================================
# 13. compute_lr_ratio_stats — has all required keys
# ===========================================================================

def test_compute_lr_ratio_stats_keys() -> None:
    cfg = _tiny_config()
    lrs = compute_layer_lrs(cfg)
    stats = compute_lr_ratio_stats(lrs, cfg.base_lr)
    assert "min_ratio" in stats
    assert "max_ratio" in stats
    assert "mean_ratio" in stats


# ===========================================================================
# 14. compute_lr_ratio_stats — values are numerically correct
# ===========================================================================

def test_compute_lr_ratio_stats_values() -> None:
    base_lr = 1.0
    layer_lrs = [1.0, 0.8, 0.64]
    stats = compute_lr_ratio_stats(layer_lrs, base_lr)
    assert stats["max_ratio"] == pytest.approx(1.0)
    assert stats["min_ratio"] == pytest.approx(0.64)
    assert stats["mean_ratio"] == pytest.approx((1.0 + 0.8 + 0.64) / 3)


# ===========================================================================
# 15. LayerWiseLRScheduler — warmup phase scales lrs linearly
# ===========================================================================

def test_scheduler_warmup_linear_scale() -> None:
    n_layers = 3
    model = TinyTransformer(n_layers=n_layers)
    cfg = LLRDConfig(n_layers=n_layers, base_lr=1e-3)
    patterns = _layer_patterns(n_layers)
    param_groups = build_llrd_param_groups(model, cfg, patterns)
    layer_lrs = compute_layer_lrs(cfg)

    optimizer = torch.optim.AdamW(param_groups)
    scheduler = LayerWiseLRScheduler(optimizer, cfg, layer_lrs)

    warmup_steps = 100
    total_steps = 1000

    # At step 50 (halfway through warmup), scale should be 0.5
    scheduler.step(global_step=50, warmup_steps=warmup_steps, total_steps=total_steps)
    lrs_mid_warmup = scheduler.get_lrs()

    # At step 100 (end of warmup), scale should be 1.0 (cosine at progress=0)
    scheduler.step(global_step=100, warmup_steps=warmup_steps, total_steps=total_steps)
    lrs_end_warmup = scheduler.get_lrs()

    for lr_mid, lr_end, base in zip(lrs_mid_warmup, lrs_end_warmup, layer_lrs):
        expected_mid = base * (50.0 / 100.0)
        assert lr_mid == pytest.approx(expected_mid, rel=1e-5)
        # at step=warmup_steps, cosine progress=0 -> scale = 1.0
        assert lr_end == pytest.approx(base * 1.0, rel=1e-5)


# ===========================================================================
# 16. compute_layer_lrs — exact decay values before floor kicks in
# ===========================================================================

def test_compute_layer_lrs_exact_values() -> None:
    cfg = LLRDConfig(base_lr=1.0, decay_factor=0.5, n_layers=3, min_lr_ratio=0.0)
    lrs = compute_layer_lrs(cfg)
    assert lrs[0] == pytest.approx(1.0)
    assert lrs[1] == pytest.approx(0.5)
    assert lrs[2] == pytest.approx(0.25)


# ===========================================================================
# 17. build_llrd_param_groups — embedding group uses embedding_lr_ratio
# ===========================================================================

def test_build_llrd_param_groups_embedding_lr() -> None:
    n_layers = 3
    model = TinyTransformer(n_layers=n_layers)
    cfg = LLRDConfig(n_layers=n_layers, base_lr=1e-2, embedding_lr_ratio=0.05)
    patterns = _layer_patterns(n_layers)
    groups = build_llrd_param_groups(model, cfg, patterns)

    # Last group is the embedding/other group
    embedding_group = groups[-1]
    expected_lr = cfg.base_lr * cfg.embedding_lr_ratio
    assert embedding_group["lr"] == pytest.approx(expected_lr)
