"""Tests for src/alignment/lora_rank_allocation.py."""

from __future__ import annotations

import torch

from src.alignment.lora_rank_allocation import (
    LoRAAdapter,
    RankAllocationConfig,
    RankAllocator,
    allocate_ranks_by_sensitivity,
    allocate_ranks_uniform,
    compute_gradient_sensitivity,
    compute_singular_value_sensitivity,
    prune_low_rank_components,
)

IN_FEATURES = 64
OUT_FEATURES = 64
RANK = 8


# ---------------------------------------------------------------------------
# RankAllocationConfig
# ---------------------------------------------------------------------------


def test_rank_allocation_config_defaults():
    cfg = RankAllocationConfig()
    assert cfg.total_rank_budget == 64
    assert cfg.min_rank == 1
    assert cfg.max_rank == 16
    assert cfg.sensitivity_method == "gradient"
    assert cfg.prune_threshold == 0.01


# ---------------------------------------------------------------------------
# LoRAAdapter
# ---------------------------------------------------------------------------


def test_lora_adapter_forward_output_shape():
    adapter = LoRAAdapter(IN_FEATURES, OUT_FEATURES, RANK)
    x = torch.randn(4, IN_FEATURES)
    out = adapter(x)
    assert out.shape == (4, OUT_FEATURES)


def test_lora_adapter_effective_weight_shape():
    adapter = LoRAAdapter(IN_FEATURES, OUT_FEATURES, RANK)
    W = adapter.effective_weight()
    assert W.shape == (OUT_FEATURES, IN_FEATURES)


def test_lora_adapter_scaling():
    alpha = 2.0
    rank = 4
    adapter = LoRAAdapter(IN_FEATURES, OUT_FEATURES, rank, alpha=alpha)
    assert adapter.scaling == alpha / rank


# ---------------------------------------------------------------------------
# compute_gradient_sensitivity
# ---------------------------------------------------------------------------


def test_compute_gradient_sensitivity_returns_float():
    adapter = LoRAAdapter(IN_FEATURES, OUT_FEATURES, RANK)
    # No gradients yet
    sensitivity = compute_gradient_sensitivity(adapter)
    assert isinstance(sensitivity, float)
    assert sensitivity == 0.0


def test_compute_gradient_sensitivity_with_gradients():
    adapter = LoRAAdapter(IN_FEATURES, OUT_FEATURES, RANK)
    x = torch.randn(2, IN_FEATURES)
    out = adapter(x).sum()
    out.backward()
    sensitivity = compute_gradient_sensitivity(adapter)
    assert isinstance(sensitivity, float)
    assert sensitivity >= 0.0


# ---------------------------------------------------------------------------
# compute_singular_value_sensitivity
# ---------------------------------------------------------------------------


def test_compute_singular_value_sensitivity_in_range():
    adapter = LoRAAdapter(IN_FEATURES, OUT_FEATURES, RANK)
    # Give lora_B some non-zero values so the weight isn't trivially zero
    torch.nn.init.normal_(adapter.lora_B, std=0.1)
    s = compute_singular_value_sensitivity(adapter)
    assert isinstance(s, float)
    assert 0.0 <= s <= 1.0


def test_compute_singular_value_sensitivity_zero_weight():
    adapter = LoRAAdapter(IN_FEATURES, OUT_FEATURES, RANK)
    # lora_B is zero-initialized, so effective_weight is zero -> returns 0.0
    s = compute_singular_value_sensitivity(adapter)
    assert s == 0.0


# ---------------------------------------------------------------------------
# allocate_ranks_uniform
# ---------------------------------------------------------------------------


def test_allocate_ranks_uniform_respects_min_max():
    cfg = RankAllocationConfig(total_rank_budget=64, min_rank=2, max_rank=16)
    ranks = allocate_ranks_uniform(4, cfg)
    assert len(ranks) == 4
    for r in ranks:
        assert cfg.min_rank <= r <= cfg.max_rank


def test_allocate_ranks_uniform_total_within_budget():
    cfg = RankAllocationConfig(total_rank_budget=64, min_rank=1, max_rank=16)
    ranks = allocate_ranks_uniform(5, cfg)
    assert sum(ranks) <= cfg.total_rank_budget


def test_allocate_ranks_uniform_empty():
    cfg = RankAllocationConfig()
    ranks = allocate_ranks_uniform(0, cfg)
    assert ranks == []


# ---------------------------------------------------------------------------
# allocate_ranks_by_sensitivity
# ---------------------------------------------------------------------------


def test_allocate_ranks_by_sensitivity_total_within_budget():
    cfg = RankAllocationConfig(total_rank_budget=64, min_rank=1, max_rank=16)
    sensitivities = [0.5, 0.3, 0.1, 0.1]
    ranks = allocate_ranks_by_sensitivity(sensitivities, cfg)
    assert sum(ranks) <= cfg.total_rank_budget


def test_allocate_ranks_by_sensitivity_respects_min_rank():
    cfg = RankAllocationConfig(total_rank_budget=64, min_rank=2, max_rank=16)
    sensitivities = [0.0, 0.0, 0.0, 1.0]  # one dominant adapter
    ranks = allocate_ranks_by_sensitivity(sensitivities, cfg)
    assert len(ranks) == 4
    for r in ranks:
        assert r >= cfg.min_rank


def test_allocate_ranks_by_sensitivity_length():
    cfg = RankAllocationConfig(total_rank_budget=32, min_rank=1, max_rank=8)
    sensitivities = [1.0, 2.0, 3.0]
    ranks = allocate_ranks_by_sensitivity(sensitivities, cfg)
    assert len(ranks) == 3


# ---------------------------------------------------------------------------
# prune_low_rank_components
# ---------------------------------------------------------------------------


def test_prune_low_rank_components_returns_int():
    adapter = LoRAAdapter(IN_FEATURES, OUT_FEATURES, RANK)
    torch.nn.init.normal_(adapter.lora_A, std=0.5)
    torch.nn.init.normal_(adapter.lora_B, std=0.5)
    n_pruned = prune_low_rank_components(adapter, threshold=0.01)
    assert isinstance(n_pruned, int)
    assert n_pruned >= 0


def test_prune_low_rank_components_high_threshold_prunes_most():
    adapter = LoRAAdapter(IN_FEATURES, OUT_FEATURES, RANK)
    torch.nn.init.normal_(adapter.lora_A, std=0.1)
    torch.nn.init.normal_(adapter.lora_B, std=0.1)
    # With a very high threshold, most components should be pruned;
    # the SVD of (out_features x in_features) returns up to min(out, in) components.
    # After pruning, exactly 1 component is kept, so n_pruned = total_svd_components - 1.
    W = adapter.effective_weight().detach().float()
    _, S, _ = torch.linalg.svd(W, full_matrices=False)
    total_components = S.shape[0]
    n_pruned = prune_low_rank_components(adapter, threshold=1e6)
    # Should prune all but 1 (we always keep at least 1)
    assert n_pruned == total_components - 1


def test_prune_low_rank_components_updates_in_place():
    adapter = LoRAAdapter(IN_FEATURES, OUT_FEATURES, RANK)
    torch.nn.init.normal_(adapter.lora_A, std=0.5)
    torch.nn.init.normal_(adapter.lora_B, std=0.5)
    prune_low_rank_components(adapter, threshold=0.01)
    # lora_A and lora_B may have been resized; check they are still Parameters
    assert isinstance(adapter.lora_A, torch.nn.Parameter)
    assert isinstance(adapter.lora_B, torch.nn.Parameter)


# ---------------------------------------------------------------------------
# RankAllocator
# ---------------------------------------------------------------------------


def test_rank_allocator_register_and_compute_sensitivities():
    cfg = RankAllocationConfig(sensitivity_method="gradient")
    allocator = RankAllocator(cfg)
    adapter_a = LoRAAdapter(IN_FEATURES, OUT_FEATURES, RANK)
    adapter_b = LoRAAdapter(IN_FEATURES, OUT_FEATURES, RANK)
    allocator.register_adapter("layer_a", adapter_a)
    allocator.register_adapter("layer_b", adapter_b)

    sensitivities = allocator.compute_sensitivities()
    assert set(sensitivities.keys()) == {"layer_a", "layer_b"}
    for v in sensitivities.values():
        assert isinstance(v, float)


def test_rank_allocator_reallocate_returns_correct_keys():
    cfg = RankAllocationConfig(total_rank_budget=32, min_rank=1, max_rank=16)
    allocator = RankAllocator(cfg)
    for i in range(3):
        adapter = LoRAAdapter(IN_FEATURES, OUT_FEATURES, RANK)
        allocator.register_adapter(f"layer_{i}", adapter)

    plan = allocator.reallocate()
    assert set(plan.keys()) == {"layer_0", "layer_1", "layer_2"}
    for rank in plan.values():
        assert isinstance(rank, int)
        assert rank >= cfg.min_rank


def test_rank_allocator_reallocate_total_within_budget():
    cfg = RankAllocationConfig(total_rank_budget=32, min_rank=1, max_rank=16)
    allocator = RankAllocator(cfg)
    for i in range(4):
        adapter = LoRAAdapter(IN_FEATURES, OUT_FEATURES, RANK)
        allocator.register_adapter(f"layer_{i}", adapter)

    plan = allocator.reallocate()
    assert sum(plan.values()) <= cfg.total_rank_budget


def test_rank_allocator_singular_value_method():
    cfg = RankAllocationConfig(sensitivity_method="singular_value")
    allocator = RankAllocator(cfg)
    adapter = LoRAAdapter(IN_FEATURES, OUT_FEATURES, RANK)
    torch.nn.init.normal_(adapter.lora_B, std=0.1)
    allocator.register_adapter("layer", adapter)

    sensitivities = allocator.compute_sensitivities()
    assert "layer" in sensitivities
    assert 0.0 <= sensitivities["layer"] <= 1.0
