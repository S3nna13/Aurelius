"""Tests for tensor parallelism: column/row linear splitting, attention head sharding, and all-reduce simulation."""  # noqa: E501

from __future__ import annotations

import torch

from src.training.tensor_parallel import (
    TensorParallelConfig,
    TensorParallelLinear,
    column_parallel_linear,
    row_parallel_linear,
    simulated_all_reduce,
    split_attention_heads,
    split_weight_column,
    split_weight_row,
)

# Shared constants
B = 2
T = 8
IN_FEATURES = 64
OUT_FEATURES = 64
WORLD_SIZE = 2


# ---------------------------------------------------------------------------
# 1. TensorParallelConfig defaults
# ---------------------------------------------------------------------------


def test_tensor_parallel_config_defaults():
    cfg = TensorParallelConfig()
    assert cfg.world_size == 2
    assert cfg.rank == 0
    assert cfg.split_dim == "column"


# ---------------------------------------------------------------------------
# 2. split_weight_column output shapes
# ---------------------------------------------------------------------------


def test_split_weight_column_output_shapes():
    weight = torch.randn(OUT_FEATURES, IN_FEATURES)
    shards = split_weight_column(weight, WORLD_SIZE)
    assert len(shards) == WORLD_SIZE
    for shard in shards:
        assert shard.shape == (OUT_FEATURES // WORLD_SIZE, IN_FEATURES)


# ---------------------------------------------------------------------------
# 3. split_weight_column concat restores original
# ---------------------------------------------------------------------------


def test_split_weight_column_restores_original():
    torch.manual_seed(0)
    weight = torch.randn(OUT_FEATURES, IN_FEATURES)
    shards = split_weight_column(weight, WORLD_SIZE)
    restored = torch.cat(shards, dim=0)
    assert torch.allclose(weight, restored)


# ---------------------------------------------------------------------------
# 4. split_weight_row output shapes
# ---------------------------------------------------------------------------


def test_split_weight_row_output_shapes():
    weight = torch.randn(OUT_FEATURES, IN_FEATURES)
    shards = split_weight_row(weight, WORLD_SIZE)
    assert len(shards) == WORLD_SIZE
    for shard in shards:
        assert shard.shape == (OUT_FEATURES, IN_FEATURES // WORLD_SIZE)


# ---------------------------------------------------------------------------
# 5. split_weight_row concat restores original
# ---------------------------------------------------------------------------


def test_split_weight_row_restores_original():
    torch.manual_seed(1)
    weight = torch.randn(OUT_FEATURES, IN_FEATURES)
    shards = split_weight_row(weight, WORLD_SIZE)
    restored = torch.cat(shards, dim=1)
    assert torch.allclose(weight, restored)


# ---------------------------------------------------------------------------
# 6. column_parallel_linear output shape
# ---------------------------------------------------------------------------


def test_column_parallel_linear_output_shape():
    torch.manual_seed(2)
    x = torch.randn(B, T, IN_FEATURES)
    weight_shard = torch.randn(OUT_FEATURES // WORLD_SIZE, IN_FEATURES)
    bias_shard = torch.zeros(OUT_FEATURES // WORLD_SIZE)
    out = column_parallel_linear(x, weight_shard, bias_shard)
    assert out.shape == (B, T, OUT_FEATURES // WORLD_SIZE)


# ---------------------------------------------------------------------------
# 7. row_parallel_linear output shape
# ---------------------------------------------------------------------------


def test_row_parallel_linear_output_shape():
    torch.manual_seed(3)
    x_shard = torch.randn(B, T, IN_FEATURES // WORLD_SIZE)
    weight_shard = torch.randn(OUT_FEATURES, IN_FEATURES // WORLD_SIZE)
    out = row_parallel_linear(x_shard, weight_shard)
    assert out.shape == (B, T, OUT_FEATURES)


# ---------------------------------------------------------------------------
# 8. simulated_all_reduce sum of 2 partials
# ---------------------------------------------------------------------------


def test_simulated_all_reduce_sum():
    torch.manual_seed(4)
    p1 = torch.randn(B, T, OUT_FEATURES)
    p2 = torch.randn(B, T, OUT_FEATURES)
    result = simulated_all_reduce([p1, p2], op="sum")
    assert result.shape == (B, T, OUT_FEATURES)
    assert torch.allclose(result, p1 + p2)


# ---------------------------------------------------------------------------
# 9. simulated_all_reduce mean
# ---------------------------------------------------------------------------


def test_simulated_all_reduce_mean():
    torch.manual_seed(5)
    p1 = torch.randn(B, T, OUT_FEATURES)
    p2 = torch.randn(B, T, OUT_FEATURES)
    result = simulated_all_reduce([p1, p2], op="mean")
    assert result.shape == (B, T, OUT_FEATURES)
    assert torch.allclose(result, (p1 + p2) / 2)


# ---------------------------------------------------------------------------
# 10. split_attention_heads correct ranges
# ---------------------------------------------------------------------------


def test_split_attention_heads_correct_ranges():
    n_heads = 8
    ranges = split_attention_heads(n_heads, WORLD_SIZE)
    assert len(ranges) == WORLD_SIZE
    assert ranges[0] == range(0, 4)
    assert ranges[1] == range(4, 8)


# ---------------------------------------------------------------------------
# 11. split_attention_heads all heads covered
# ---------------------------------------------------------------------------


def test_split_attention_heads_all_heads_covered():
    n_heads = 8
    ranges = split_attention_heads(n_heads, WORLD_SIZE)
    all_heads = set()
    for r in ranges:
        all_heads.update(r)
    assert all_heads == set(range(n_heads))


# ---------------------------------------------------------------------------
# 12. TensorParallelLinear forward output shape
# ---------------------------------------------------------------------------


def test_tensor_parallel_linear_forward_output_shape():
    torch.manual_seed(6)
    cfg = TensorParallelConfig(world_size=WORLD_SIZE, rank=0, split_dim="column")
    layer = TensorParallelLinear(IN_FEATURES, OUT_FEATURES, cfg, bias=True)
    x = torch.randn(B, T, IN_FEATURES)
    out = layer(x)
    assert out.shape == (B, T, OUT_FEATURES // WORLD_SIZE)
