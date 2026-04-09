"""Tests for tensor parallelism utilities."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.training.tensor_parallel import (
    TPConfig,
    ColumnParallelLinear,
    RowParallelLinear,
    TensorParallelMLP,
    count_tp_parameters,
    SequenceParallelNorm,
    convert_to_tensor_parallel,
)

# Shared constants
B = 2
T = 8
D_MODEL = 64
D_FF = 128
TP_DEGREE = 2


@pytest.fixture
def tp_config():
    return TPConfig(tp_degree=TP_DEGREE)


@pytest.fixture
def x(tp_config):
    torch.manual_seed(42)
    return torch.randn(B, T, D_MODEL)


# ---------------------------------------------------------------------------
# 1. TPConfig defaults
# ---------------------------------------------------------------------------

def test_tp_config_defaults():
    cfg = TPConfig()
    assert cfg.tp_degree == 2
    assert cfg.sequence_parallel is False


# ---------------------------------------------------------------------------
# 2. ColumnParallelLinear — output shape
# ---------------------------------------------------------------------------

def test_column_parallel_output_shape(tp_config, x):
    layer = ColumnParallelLinear(D_MODEL, D_FF, tp_config)
    out = layer(x)
    assert out.shape == (B, T, D_FF)


# ---------------------------------------------------------------------------
# 3. ColumnParallelLinear — shard count
# ---------------------------------------------------------------------------

def test_column_parallel_shard_count(tp_config):
    layer = ColumnParallelLinear(D_MODEL, D_FF, tp_config)
    assert len(layer.shards) == TP_DEGREE


# ---------------------------------------------------------------------------
# 4. ColumnParallelLinear — merge_weights shape
# ---------------------------------------------------------------------------

def test_column_parallel_merge_shape(tp_config):
    layer = ColumnParallelLinear(D_MODEL, D_FF, tp_config)
    merged = layer.merge_weights()
    assert isinstance(merged, nn.Linear)
    assert merged.weight.shape == (D_FF, D_MODEL)


# ---------------------------------------------------------------------------
# 5. RowParallelLinear — output shape
# ---------------------------------------------------------------------------

def test_row_parallel_output_shape(tp_config, x):
    # x has D_MODEL features which is the in_features for RowParallel
    layer = RowParallelLinear(D_MODEL, D_FF, tp_config)
    out = layer(x)
    assert out.shape == (B, T, D_FF)


# ---------------------------------------------------------------------------
# 6. RowParallelLinear — shard count
# ---------------------------------------------------------------------------

def test_row_parallel_shard_count(tp_config):
    layer = RowParallelLinear(D_MODEL, D_FF, tp_config)
    assert len(layer.shards) == TP_DEGREE


# ---------------------------------------------------------------------------
# 7. RowParallelLinear — merge_weights shape
# ---------------------------------------------------------------------------

def test_row_parallel_merge_shape(tp_config):
    layer = RowParallelLinear(D_MODEL, D_FF, tp_config)
    merged = layer.merge_weights()
    assert isinstance(merged, nn.Linear)
    # merged: (D_FF, D_MODEL) — maps D_MODEL -> D_FF
    assert merged.weight.shape == (D_FF, D_MODEL)


# ---------------------------------------------------------------------------
# 8. TensorParallelMLP — output shape
# ---------------------------------------------------------------------------

def test_tp_mlp_output_shape(tp_config, x):
    mlp = TensorParallelMLP(D_MODEL, D_FF, tp_config)
    out = mlp(x)
    assert out.shape == (B, T, D_MODEL)


# ---------------------------------------------------------------------------
# 9. TensorParallelMLP — gradient flow
# ---------------------------------------------------------------------------

def test_tp_mlp_gradient_flow(tp_config, x):
    x_req = x.detach().requires_grad_(True)
    mlp = TensorParallelMLP(D_MODEL, D_FF, tp_config)
    out = mlp(x_req)
    loss = out.sum()
    loss.backward()
    assert x_req.grad is not None
    assert x_req.grad.shape == x_req.shape
    # Verify at least some gradients are non-zero
    assert x_req.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# 10. count_tp_parameters — correct keys
# ---------------------------------------------------------------------------

def test_count_tp_parameters_keys(tp_config):
    mlp = TensorParallelMLP(D_MODEL, D_FF, tp_config)
    counts = count_tp_parameters(mlp)
    assert "total" in counts
    assert "tp_sharded" in counts
    assert "non_sharded" in counts
    # Sanity: sharded + non_sharded == total
    assert counts["tp_sharded"] + counts["non_sharded"] == counts["total"]
    # All TP-MLP parameters should be in sharded layers
    assert counts["tp_sharded"] > 0
    assert counts["total"] > 0


# ---------------------------------------------------------------------------
# 11. SequenceParallelNorm — output shape
# ---------------------------------------------------------------------------

def test_sequence_parallel_norm_shape(tp_config):
    # T must be divisible by tp_degree
    assert T % TP_DEGREE == 0
    norm = SequenceParallelNorm(D_MODEL, tp_config)
    x = torch.randn(B, T, D_MODEL)
    out = norm(x)
    assert out.shape == (B, T, D_MODEL)


# ---------------------------------------------------------------------------
# 12. convert_to_tensor_parallel — model has ColumnParallelLinear after conversion
# ---------------------------------------------------------------------------

def test_convert_to_tensor_parallel(tp_config):
    # Build a simple model with Linear layers whose out_features are divisible by tp_degree
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(D_MODEL, D_FF)   # 64 -> 128, both divisible by 2
            self.fc2 = nn.Linear(D_FF, D_MODEL)   # 128 -> 64, both divisible by 2

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    model = SimpleModel()
    convert_to_tensor_parallel(model, tp_config)

    # Both linear layers should have been replaced
    assert isinstance(model.fc1, ColumnParallelLinear)
    assert isinstance(model.fc2, ColumnParallelLinear)
