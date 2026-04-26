"""Tensor parallelism: column/row linear splitting, attention head sharding, and all-reduce simulation."""  # noqa: E501

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# New-spec dataclass and functional API
# ---------------------------------------------------------------------------


@dataclass
class TensorParallelConfig:
    """Configuration for tensor parallelism (new spec)."""

    world_size: int = 2
    rank: int = 0
    split_dim: str = "column"  # "column" | "row"


def split_weight_column(weight: Tensor, world_size: int) -> list[Tensor]:
    """Split weight along dim=0 (output dimension) into world_size chunks.

    Args:
        weight: ``(out_features, in_features)``
        world_size: Number of parallel workers.

    Returns:
        List of world_size tensors each ``(out_features // world_size, in_features)``.
    """
    return list(torch.chunk(weight, world_size, dim=0))


def split_weight_row(weight: Tensor, world_size: int) -> list[Tensor]:
    """Split weight along dim=1 (input dimension) into world_size chunks.

    Args:
        weight: ``(out_features, in_features)``
        world_size: Number of parallel workers.

    Returns:
        List of world_size tensors each ``(out_features, in_features // world_size)``.
    """
    return list(torch.chunk(weight, world_size, dim=1))


def column_parallel_linear(
    x: Tensor, weight_shard: Tensor, bias_shard: Tensor | None = None
) -> Tensor:
    """Linear forward with column-sharded weight.

    Args:
        x: ``(B, T, in_features)``
        weight_shard: ``(out_shard_size, in_features)``
        bias_shard: ``(out_shard_size,)`` or None.

    Returns:
        ``(B, T, out_shard_size)``
    """
    return F.linear(x, weight_shard, bias_shard)


def row_parallel_linear(
    x_shard: Tensor, weight_shard: Tensor, bias: Tensor | None = None
) -> Tensor:
    """Linear forward with row-sharded weight and sharded input.

    Args:
        x_shard: ``(B, T, in_shard_size)``
        weight_shard: ``(out_features, in_shard_size)``
        bias: ``(out_features,)`` or None.

    Returns:
        ``(B, T, out_features)`` — partial output; needs all-reduce to sum across ranks.
    """
    return x_shard @ weight_shard.T + (bias if bias is not None else 0)


def simulated_all_reduce(partial_outputs: list[Tensor], op: str = "sum") -> Tensor:
    """Simulate all-reduce by stacking and reducing partial outputs.

    Args:
        partial_outputs: List of ``(B, T, out_features)`` tensors from each rank.
        op: "sum" | "mean"

    Returns:
        ``(B, T, out_features)``
    """
    stacked = torch.stack(partial_outputs, dim=0)  # (world_size, B, T, out_features)
    if op == "sum":
        return stacked.sum(dim=0)
    elif op == "mean":
        return stacked.mean(dim=0)
    else:
        raise ValueError(f"Unknown op '{op}'. Expected 'sum' or 'mean'.")


def split_attention_heads(n_heads: int, world_size: int) -> list[range]:
    """Partition attention heads evenly across ranks.

    Args:
        n_heads: Total number of attention heads.
        world_size: Number of parallel workers.

    Returns:
        List of world_size range objects covering all heads.
        e.g., [range(0, 4), range(4, 8)] for n_heads=8, world_size=2.
    """
    heads_per_rank = n_heads // world_size
    result: list[range] = []
    for rank in range(world_size):
        start = rank * heads_per_rank
        # Last rank absorbs any remainder
        end = n_heads if rank == world_size - 1 else start + heads_per_rank
        result.append(range(start, end))
    return result


class TensorParallelLinear(nn.Module):
    """Column or row parallel linear layer controlled by TensorParallelConfig.

    For column parallel: splits output dim, each rank holds a weight shard of
    shape ``(out_features // world_size, in_features)``. Output is
    ``(B, T, out_features // world_size)`` — no all-reduce needed before the next layer.

    For row parallel: splits input dim; output is the full ``(B, T, out_features)``
    obtained by summing partial results via simulated_all_reduce.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: TensorParallelConfig,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config

        if config.split_dim == "column":
            shard_out = out_features // config.world_size
            self.weight_shard = nn.Parameter(torch.empty(shard_out, in_features))
            self.bias_shard = nn.Parameter(torch.zeros(shard_out)) if bias else None
            nn.init.kaiming_uniform_(self.weight_shard, a=0.01)
        elif config.split_dim == "row":
            shard_in = in_features // config.world_size
            self.weight_shard = nn.Parameter(torch.empty(out_features, shard_in))
            self.bias_shard = nn.Parameter(torch.zeros(out_features)) if bias else None
            nn.init.kaiming_uniform_(self.weight_shard, a=0.01)
        else:
            raise ValueError(f"Unknown split_dim '{config.split_dim}'.")

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Column parallel: ``x`` → ``(B, T, out_features // world_size)``
        Row parallel: ``x_shard`` → all-reduce → ``(B, T, out_features)``
        """
        if self.config.split_dim == "column":
            return column_parallel_linear(x, self.weight_shard, self.bias_shard)
        else:
            # Row parallel: x is already the correct shard of the input
            partial = row_parallel_linear(x, self.weight_shard, self.bias_shard)
            # Simulate all-reduce by returning partial (single-rank simulation)
            return partial


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
            [nn.Linear(in_features, self.shard_size, bias=bias) for _ in range(config.tp_degree)]
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
            merged.weight.copy_(torch.cat([shard.weight for shard in self.shards], dim=0))
            if has_bias:
                merged.bias.copy_(torch.cat([shard.bias for shard in self.shards], dim=0))
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
            merged.weight.copy_(torch.cat([shard.weight for shard in self.shards], dim=1))
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
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(config.tp_degree)])

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
