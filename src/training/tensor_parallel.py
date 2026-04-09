"""Tensor parallelism utilities for training large models across multiple devices.

Implements column/row parallelism for Linear layers (Megatron-style).
Since single-GPU simulation is needed for tests, uses virtual device splitting.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class TPConfig:
    """Configuration for tensor parallelism."""

    tp_degree: int = 2
    sequence_parallel: bool = False


class ColumnParallelLinear(nn.Module):
    """Splits output features across tp_degree virtual shards (column-wise)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: TPConfig,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        self.shard_size = out_features // config.tp_degree
        self.shards = nn.ModuleList(
            [
                nn.Linear(in_features, self.shard_size, bias=bias)
                for _ in range(config.tp_degree)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run each shard on x independently, concatenate along last dim."""
        outputs = [shard(x) for shard in self.shards]
        return torch.cat(outputs, dim=-1)

    def merge_weights(self) -> nn.Linear:
        """Create a single nn.Linear merging all shards (for inference without parallelism)."""
        has_bias = self.shards[0].bias is not None
        merged = nn.Linear(self.in_features, self.out_features, bias=has_bias)
        with torch.no_grad():
            merged.weight.copy_(
                torch.cat([shard.weight for shard in self.shards], dim=0)
            )
            if has_bias:
                merged.bias.copy_(
                    torch.cat([shard.bias for shard in self.shards], dim=0)
                )
        return merged


class RowParallelLinear(nn.Module):
    """Splits input features across tp_degree virtual shards, then all-reduces (row-wise)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: TPConfig,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        self.shard_size = in_features // config.tp_degree
        # Each shard: (shard_size, out_features); only first shard gets bias to avoid duplication
        self.shards = nn.ModuleList(
            [
                nn.Linear(self.shard_size, out_features, bias=(bias and i == 0))
                for i in range(config.tp_degree)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Split x along last dim, run each shard, sum results (all-reduce simulation)."""
        chunks = torch.chunk(x, self.config.tp_degree, dim=-1)
        outputs = [shard(chunk) for shard, chunk in zip(self.shards, chunks)]
        return torch.stack(outputs, dim=0).sum(dim=0)

    def merge_weights(self) -> nn.Linear:
        """Create a single nn.Linear equivalent to all shards combined."""
        has_bias = self.shards[0].bias is not None
        merged = nn.Linear(self.in_features, self.out_features, bias=has_bias)
        with torch.no_grad():
            merged.weight.copy_(
                torch.cat([shard.weight for shard in self.shards], dim=1)
            )
            if has_bias:
                merged.bias.copy_(self.shards[0].bias)
        return merged


class TensorParallelMLP(nn.Module):
    """Megatron-style MLP: ColumnParallel -> GELU -> RowParallel."""

    def __init__(self, d_model: int, d_ff: int, config: TPConfig) -> None:
        super().__init__()
        self.fc1 = ColumnParallelLinear(d_model, d_ff, config)
        self.fc2 = RowParallelLinear(d_ff, d_model, config)

    def forward(self, x: Tensor) -> Tensor:
        """x -> fc1 -> GELU -> fc2."""
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


def count_tp_parameters(module: nn.Module) -> dict[str, int]:
    """Count total, shard, and non-shard parameters.

    Returns:
        dict with keys: "total", "tp_sharded", "non_sharded"
    """
    total = 0
    tp_sharded = 0
    non_sharded = 0

    for child_name, child in module.named_modules():
        if isinstance(child, (ColumnParallelLinear, RowParallelLinear)):
            for param in child.parameters():
                tp_sharded += param.numel()
        elif child is module:
            # Count direct parameters of the top-level module (not in submodules)
            for param in child.parameters(recurse=False):
                non_sharded += param.numel()

    # Walk all params for total
    for param in module.parameters():
        total += param.numel()

    # non_sharded = total - tp_sharded (params not owned by TP layers)
    non_sharded = total - tp_sharded

    return {"total": total, "tp_sharded": tp_sharded, "non_sharded": non_sharded}


class SequenceParallelNorm(nn.Module):
    """Simulates sequence-parallel LayerNorm (split sequence across devices, normalize locally).

    Note: for identical weights across shards, gives the same result as a single LayerNorm
    because LayerNorm operates over the last dimension (d_model), not the sequence dimension.
    """

    def __init__(self, d_model: int, config: TPConfig) -> None:
        super().__init__()
        self.config = config
        self.norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(config.tp_degree)]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Split x along seq dim, apply each norm to its shard, concatenate."""
        # x: (B, T, D)
        chunks = torch.chunk(x, self.config.tp_degree, dim=1)
        outputs = [norm(chunk) for norm, chunk in zip(self.norms, chunks)]
        return torch.cat(outputs, dim=1)


def convert_to_tensor_parallel(model: nn.Module, config: TPConfig) -> nn.Module:
    """Walk model, replace nn.Linear with ColumnParallelLinear (simple conversion).

    Only converts Linear layers with out_features divisible by tp_degree.
    Returns the modified model (in-place).
    """
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear):
            if module.out_features % config.tp_degree == 0:
                has_bias = module.bias is not None
                new_layer = ColumnParallelLinear(
                    module.in_features,
                    module.out_features,
                    config,
                    bias=has_bias,
                )
                # Copy weights into shards
                with torch.no_grad():
                    weight_chunks = torch.chunk(module.weight, config.tp_degree, dim=0)
                    for shard, w_chunk in zip(new_layer.shards, weight_chunks):
                        shard.weight.copy_(w_chunk)
                        if has_bias:
                            bias_chunks = torch.chunk(module.bias, config.tp_degree, dim=0)
                            for shard, b_chunk in zip(new_layer.shards, bias_chunks):
                                shard.bias.copy_(b_chunk)
                setattr(model, name, new_layer)
        else:
            convert_to_tensor_parallel(module, config)
    return model
