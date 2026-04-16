"""Tensor Parallelism utilities for single-GPU simulation.

Implements column-parallel, row-parallel linear layers and head-parallel
attention, simulating the split/gather logic used in multi-GPU tensor
parallelism without requiring torch.distributed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TPConfig:
    """Configuration for tensor-parallel layers."""
    n_partitions: int = 4
    d_model: int = 512
    gather_output: bool = True


# ---------------------------------------------------------------------------
# Weight splitting helpers
# ---------------------------------------------------------------------------

def split_tensor_column(weight: torch.Tensor, n_partitions: int) -> List[torch.Tensor]:
    """Split a (out_features, in_features) weight matrix along the output dim.

    Args:
        weight: 2-D tensor of shape (out_features, in_features).
        n_partitions: Number of partitions.

    Returns:
        List of n_partitions tensors, each of shape (out_features // n_partitions, in_features).
    """
    if weight.dim() != 2:
        raise ValueError(f"Expected 2-D weight, got shape {weight.shape}")
    out_features = weight.size(0)
    if out_features % n_partitions != 0:
        raise ValueError(
            f"out_features ({out_features}) must be divisible by n_partitions ({n_partitions})"
        )
    chunk_size = out_features // n_partitions
    return list(weight.split(chunk_size, dim=0))


def split_tensor_row(weight: torch.Tensor, n_partitions: int) -> List[torch.Tensor]:
    """Split a (out_features, in_features) weight matrix along the input dim.

    Args:
        weight: 2-D tensor of shape (out_features, in_features).
        n_partitions: Number of partitions.

    Returns:
        List of n_partitions tensors, each of shape (out_features, in_features // n_partitions).
    """
    if weight.dim() != 2:
        raise ValueError(f"Expected 2-D weight, got shape {weight.shape}")
    in_features = weight.size(1)
    if in_features % n_partitions != 0:
        raise ValueError(
            f"in_features ({in_features}) must be divisible by n_partitions ({n_partitions})"
        )
    chunk_size = in_features // n_partitions
    return list(weight.split(chunk_size, dim=1))


# ---------------------------------------------------------------------------
# ColumnParallelLinear
# ---------------------------------------------------------------------------

class ColumnParallelLinear(nn.Module):
    """Linear layer with weights split along the output (column) dimension.

    Each partition independently maps the full input to a slice of the output.
    With gather_output=True (default) the slices are concatenated, producing
    an output of shape (B, out_features) identical to a single nn.Linear.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_partitions: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if out_features % n_partitions != 0:
            raise ValueError(
                f"out_features ({out_features}) must be divisible by n_partitions ({n_partitions})"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.n_partitions = n_partitions
        self.partition_out = out_features // n_partitions

        self.partitions = nn.ModuleList(
            [
                nn.Linear(in_features, self.partition_out, bias=bias)
                for _ in range(n_partitions)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run each partition on the full input then concatenate.

        Args:
            x: (..., in_features)

        Returns:
            (..., out_features)
        """
        outputs = [p(x) for p in self.partitions]  # each: (..., partition_out)
        return torch.cat(outputs, dim=-1)


# ---------------------------------------------------------------------------
# RowParallelLinear
# ---------------------------------------------------------------------------

class RowParallelLinear(nn.Module):
    """Linear layer with weights split along the input (row) dimension.

    The input is sliced into n_partitions along the last dimension; each
    partition processes its slice and the results are summed (simulating the
    all-reduce across GPUs in true tensor parallelism).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_partitions: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if in_features % n_partitions != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by n_partitions ({n_partitions})"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.n_partitions = n_partitions
        self.partition_in = in_features // n_partitions

        # Only the first partition gets a bias term to avoid double-counting
        self.partitions = nn.ModuleList(
            [
                nn.Linear(self.partition_in, out_features, bias=(bias and i == 0))
                for i in range(n_partitions)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Split input along last dim, run each partition, sum results.

        Args:
            x: (..., in_features)

        Returns:
            (..., out_features)
        """
        chunks = x.split(self.partition_in, dim=-1)  # n_partitions tensors (..., partition_in)
        outputs = [p(chunk) for p, chunk in zip(self.partitions, chunks)]
        # all-reduce = sum across partitions
        result = outputs[0]
        for out in outputs[1:]:
            result = result + out
        return result


# ---------------------------------------------------------------------------
# HeadParallelAttention
# ---------------------------------------------------------------------------

class HeadParallelAttention(nn.Module):
    """Multi-head attention with heads partitioned across tensor-parallel ranks.

    Each partition handles n_heads // n_partitions heads independently using
    nn.MultiheadAttention.  Outputs are concatenated along the embedding
    dimension and projected back to d_model.
    """

    def __init__(self, d_model: int, n_heads: int, n_partitions: int) -> None:
        super().__init__()
        if n_heads % n_partitions != 0:
            raise ValueError(
                f"n_heads ({n_heads}) must be divisible by n_partitions ({n_partitions})"
            )
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_partitions = n_partitions
        self.heads_per_partition = n_heads // n_partitions

        # Each partition operates on a d_model-dimensional input but uses only
        # its fraction of the heads.  embed_dim for each partition is
        # d_model // n_partitions so that each head still has d_model // n_heads
        # dimensions (head_dim = d_model / n_heads, same as the full model).
        self.partition_dim = d_model // n_partitions
        self.partitions = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=self.partition_dim,
                    num_heads=self.heads_per_partition,
                    batch_first=True,
                )
                for _ in range(n_partitions)
            ]
        )

        # Input projection: map d_model -> partition_dim for each partition
        self.in_projs = nn.ModuleList(
            [nn.Linear(d_model, self.partition_dim, bias=False) for _ in range(n_partitions)]
        )

        # Output projection: gather n_partitions * partition_dim -> d_model
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run each partition's MHA, concatenate, project back.

        Args:
            x: (B, T, d_model)

        Returns:
            (B, T, d_model)
        """
        part_outs = []
        for in_proj, mha in zip(self.in_projs, self.partitions):
            xp = in_proj(x)  # (B, T, partition_dim)
            out, _ = mha(xp, xp, xp)  # (B, T, partition_dim)
            part_outs.append(out)

        # Concatenate along the embedding dimension: (B, T, d_model)
        concat = torch.cat(part_outs, dim=-1)
        return self.out_proj(concat)


# ---------------------------------------------------------------------------
# Equivalence verification
# ---------------------------------------------------------------------------

def verify_partition_equivalence(
    linear: nn.Linear,
    col_parallel: ColumnParallelLinear,
    x: torch.Tensor,
    tol: float = 1e-4,
) -> bool:
    """Copy weights from a single nn.Linear into a ColumnParallelLinear and
    verify that both produce identical outputs.

    Args:
        linear: Reference nn.Linear(in_features, out_features).
        col_parallel: ColumnParallelLinear with matching dimensions.
        x: Input tensor of shape (..., in_features).
        tol: Absolute tolerance for torch.allclose.

    Returns:
        True if outputs match within tol, False otherwise.
    """
    weight_chunks = split_tensor_column(linear.weight.data, col_parallel.n_partitions)

    has_bias = linear.bias is not None
    if has_bias:
        bias_chunks = linear.bias.data.split(col_parallel.partition_out, dim=0)

    with torch.no_grad():
        for i, part in enumerate(col_parallel.partitions):
            part.weight.data.copy_(weight_chunks[i])
            if has_bias and part.bias is not None:
                part.bias.data.copy_(bias_chunks[i])
            elif not has_bias and part.bias is not None:
                part.bias.data.zero_()

    with torch.no_grad():
        ref_out = linear(x)
        tp_out = col_parallel(x)

    return torch.allclose(ref_out, tp_out, atol=tol)
