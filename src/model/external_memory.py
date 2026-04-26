"""Differentiable external memory module (Neural Turing Machine / DNC style).

Provides content-based addressing for memory read/write operations,
a controller that interfaces with memory, and a transformer layer
augmented with external memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class MemoryConfig:
    """Configuration for differentiable external memory."""

    memory_size: int = 128
    memory_dim: int = 32
    n_read_heads: int = 2
    n_write_heads: int = 1
    controller_dim: int = 64


class ExternalMemory:
    """Differentiable memory bank with content-based addressing.

    Memory is stored as a (memory_size, memory_dim) tensor.
    Read and write operations use cosine similarity for soft addressing.
    """

    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self.memory: Tensor | None = None  # (memory_size, memory_dim)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _attention(self, query: Tensor) -> Tensor:
        """Compute soft attention weights over memory slots.

        Args:
            query: (B, memory_dim)

        Returns:
            weights: (B, memory_size) — softmax over cosine similarities
        """
        assert self.memory is not None, "Call reset() before using ExternalMemory."  # noqa: S101
        # Normalize query and memory for cosine similarity
        q_norm = F.normalize(query, p=2, dim=-1)  # (B, D)
        m_norm = F.normalize(self.memory, p=2, dim=-1)  # (M, D)
        # scores: (B, M)
        scores = torch.matmul(q_norm, m_norm.t())
        return F.softmax(scores, dim=-1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self, query: Tensor) -> Tensor:
        """Content-based read from memory.

        Args:
            query: (B, memory_dim)

        Returns:
            (B, memory_dim) — weighted average of memory slots
        """
        weights = self._attention(query)  # (B, M)
        return torch.matmul(weights, self.memory)  # (B, D)

    def write(self, key: Tensor, value: Tensor, strength: Tensor) -> None:
        """Soft write: update memory slots proportional to cosine similarity.

        Args:
            key:      (B, memory_dim) — used to compute write attention
            value:    (B, memory_dim) — content to write
            strength: (B, 1)          — scalar gate controlling write magnitude
        """
        assert self.memory is not None, "Call reset() before using ExternalMemory."  # noqa: S101
        weights = self._attention(key)  # (B, M)
        # Weighted write: accumulate value contributions across batch
        # write_signal: (M, D)  =  weights^T (M, B) @ (value * strength) (B, D)
        gated_value = value * strength  # (B, D)
        write_signal = torch.matmul(weights.t(), gated_value)  # (M, D)
        self.memory = self.memory + write_signal

    def reset(self, batch_size: int) -> None:
        """Initialize memory to small random values.

        Args:
            batch_size: kept for API consistency; memory is shared across batch.
        """
        self.memory = torch.randn(self.config.memory_size, self.config.memory_dim) * 0.01


class MemoryController(nn.Module):
    """Controller that projects token features to memory interface vectors.

    For each token position the controller:
    1. Projects x to query / key / value / strength.
    2. Reads from memory using each read head.
    3. Concatenates read vectors with x and projects back to d_model.
    """

    def __init__(self, d_model: int, config: MemoryConfig) -> None:
        super().__init__()
        self.config = config
        D = config.memory_dim
        H_r = config.n_read_heads

        # Projections for read heads
        self.query_proj = nn.Linear(d_model, H_r * D)

        # Projections for write head
        self.key_proj = nn.Linear(d_model, D)
        self.value_proj = nn.Linear(d_model, D)
        self.strength_proj = nn.Linear(d_model, 1)

        # Output projection: combine x + read vectors → d_model
        self.out_proj = nn.Linear(d_model + H_r * D, d_model)

    def forward(self, x: Tensor, memory: ExternalMemory) -> tuple[Tensor, Tensor]:
        """Apply memory read/write and project back to d_model.

        Args:
            x:      (B, T, d_model)
            memory: ExternalMemory (already reset)

        Returns:
            output:      (B, T, d_model)
            read_vector: (B, T, n_read_heads * memory_dim)
        """
        B, T, _ = x.shape
        config = self.config
        D = config.memory_dim
        H_r = config.n_read_heads

        # Flatten time into batch for per-token memory ops
        x_flat = x.reshape(B * T, -1)  # (B*T, d_model)

        # Write (using last token's projection to keep causal spirit; here we
        # write each token sequentially but for efficiency we do a batched write)
        key = self.key_proj(x_flat)  # (B*T, D)
        value = self.value_proj(x_flat)  # (B*T, D)
        strength = torch.sigmoid(self.strength_proj(x_flat))  # (B*T, 1)
        memory.write(key, value, strength)

        # Read
        queries = self.query_proj(x_flat)  # (B*T, H_r*D)
        queries = queries.view(B * T, H_r, D)  # (B*T, H_r, D)

        read_parts = []
        for h in range(H_r):
            r = memory.read(queries[:, h, :])  # (B*T, D)
            read_parts.append(r)
        read_vector_flat = torch.cat(read_parts, dim=-1)  # (B*T, H_r*D)

        # Combine
        combined = torch.cat([x_flat, read_vector_flat], dim=-1)  # (B*T, d_model+H_r*D)
        out_flat = self.out_proj(combined)  # (B*T, d_model)

        output = out_flat.view(B, T, -1)
        read_vector = read_vector_flat.view(B, T, H_r * D)

        return output, read_vector


class MemoryAugmentedLayer(nn.Module):
    """Transformer layer augmented with differentiable external memory.

    Applies the MemoryController and adds a residual connection so that
    the output shape matches the input shape (B, T, d_model).
    """

    def __init__(self, d_model: int, config: MemoryConfig) -> None:
        super().__init__()
        self.controller = MemoryController(d_model, config)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, memory: ExternalMemory) -> Tensor:
        """Apply memory-augmented controller with residual connection.

        Args:
            x:      (B, T, d_model)
            memory: ExternalMemory (already reset)

        Returns:
            (B, T, d_model)
        """
        out, _ = self.controller(self.norm(x), memory)
        return x + out
