"""Memory-augmented transformer: external memory banks for extended context (Memorizing Transformers)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class MemoryConfig:
    """Configuration for external memory bank."""

    memory_size: int = 512
    key_dim: int = 32
    value_dim: int = 64
    n_memory_heads: int = 2
    top_k: int = 32
    update_strategy: str = "fifo"  # "fifo" | "lru" | "random"


class ExternalMemoryBank(nn.Module):
    """Fixed-size key-value external memory bank."""

    def __init__(self, config: MemoryConfig) -> None:
        super().__init__()
        self.config = config

        # Buffers — not trained parameters
        self.register_buffer("keys", torch.randn(config.memory_size, config.key_dim))
        self.register_buffer("values", torch.zeros(config.memory_size, config.value_dim))
        self.register_buffer("_access_count", torch.zeros(config.memory_size))

        self._write_ptr: int = 0

    def read(self, queries: Tensor) -> tuple[Tensor, Tensor]:
        """Retrieve values from memory via top-k sparse softmax attention.

        Args:
            queries: (B, n_q, key_dim)

        Returns:
            retrieved_values: (B, n_q, value_dim)
            attention_weights: (B, n_q, memory_size)  — sparse (zeros outside top-k)
        """
        B, n_q, key_dim = queries.shape
        scale = math.sqrt(key_dim)

        # Snapshot keys to guard against in-place write() calls later in the same forward pass
        keys_snapshot = self.keys.detach().clone()
        # (B, n_q, memory_size)
        scores = torch.matmul(queries, keys_snapshot.T) / scale

        top_k = min(self.config.top_k, self.config.memory_size)

        # Top-k indices
        topk_scores, topk_indices = torch.topk(scores, top_k, dim=-1)  # (B, n_q, top_k)
        topk_weights = F.softmax(topk_scores, dim=-1)  # (B, n_q, top_k)

        # Scatter back into full (B, n_q, memory_size) weight tensor (non-inplace to preserve grad graph)
        full_weights = torch.zeros(B, n_q, self.config.memory_size, device=queries.device, dtype=queries.dtype)
        full_weights = full_weights.scatter(dim=-1, index=topk_indices, src=topk_weights)

        # Update access counts (for LRU) — use most recent batch's queries (no grad)
        with torch.no_grad():
            flat_indices = topk_indices.reshape(-1)
            self._access_count.index_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=torch.float))

        # Retrieve values: (B, n_q, memory_size) @ (memory_size, value_dim) → (B, n_q, value_dim)
        # Clone+detach to get an independent snapshot; write() will mutate self.values in-place
        # later in the same forward pass, which would corrupt the autograd saved tensor.
        values_snapshot = self.values.detach().clone()
        retrieved = torch.matmul(full_weights, values_snapshot)

        return retrieved, full_weights

    def write(self, new_keys: Tensor, new_values: Tensor) -> None:
        """Write N entries into memory.

        Args:
            new_keys:   (N, key_dim)
            new_values: (N, value_dim)
        """
        N = new_keys.shape[0]
        M = self.config.memory_size

        if self.config.update_strategy == "fifo":
            for i in range(N):
                slot = self._write_ptr % M
                self.keys[slot] = new_keys[i].detach()
                self.values[slot] = new_values[i].detach()
                self._write_ptr += 1

        elif self.config.update_strategy == "lru":
            # Replace N least-recently-accessed slots
            _, lru_indices = torch.topk(self._access_count, N, largest=False)
            for i, slot in enumerate(lru_indices):
                self.keys[slot] = new_keys[i].detach()
                self.values[slot] = new_values[i].detach()
                self._access_count[slot] = 0.0

        elif self.config.update_strategy == "random":
            indices = torch.randperm(M)[:N]
            for i, slot in enumerate(indices):
                self.keys[slot] = new_keys[i].detach()
                self.values[slot] = new_values[i].detach()

        else:
            raise ValueError(f"Unknown update_strategy: {self.config.update_strategy!r}")

    def reset(self) -> None:
        """Zero all memory contents."""
        self.keys.zero_()
        self.values.zero_()
        self._access_count.zero_()
        self._write_ptr = 0


class MemoryAttentionLayer(nn.Module):
    """Augments standard attention with memory-bank reads via a learnable gate."""

    def __init__(self, d_model: int, config: MemoryConfig) -> None:
        super().__init__()
        self.config = config
        self.d_model = d_model

        total_key_dim = config.key_dim * config.n_memory_heads
        self.q_proj = nn.Linear(d_model, total_key_dim)
        self.v_proj_mem = nn.Linear(config.value_dim, d_model)
        self.gate = nn.Parameter(torch.zeros(1))
        self.memory = ExternalMemoryBank(config)

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            hidden: (B, T, D)

        Returns:
            gated_output: (B, T, D)
            attention_weights: (B, T, memory_size)
        """
        B, T, D = hidden.shape
        cfg = self.config

        # Project queries: (B, T, n_heads * key_dim)
        q = self.q_proj(hidden)

        # Split heads and average across heads for single-key-dim query
        # Shape: (B, T, n_heads, key_dim) → average → (B, T, key_dim)
        q = q.view(B, T, cfg.n_memory_heads, cfg.key_dim).mean(dim=2)

        # Read from memory: (B, T, value_dim), (B, T, memory_size)
        retrieved, attn_weights = self.memory.read(q)

        # Project memory values to d_model
        mem_contribution = self.v_proj_mem(retrieved)  # (B, T, D)

        # Learnable gate: scalar in (0, 1)
        g = torch.sigmoid(self.gate)
        output = g * mem_contribution + (1.0 - g) * hidden

        # Write last hidden states to memory (detached; shape (T, D) → need key projection)
        # We write the query keys and project hidden to value_dim via v_proj_mem input space
        # Write using the last item in the batch to keep it simple (or mean)
        last_q = q[0].detach()  # (T, key_dim)
        last_h = hidden[0].detach()  # (T, D)
        # Project hidden → value_dim by slicing or using v_proj_mem weight pseudo-inverse;
        # simplest: use the first value_dim dims of hidden projected via linear with no grad
        with torch.no_grad():
            # (T, value_dim) — just take a linear slice for storage purposes
            v_to_store = last_h[:, : self.config.value_dim]
            # Pad if d_model < value_dim (shouldn't happen in practice)
            if v_to_store.shape[-1] < self.config.value_dim:
                pad = self.config.value_dim - v_to_store.shape[-1]
                v_to_store = F.pad(v_to_store, (0, pad))
        self.memory.write(last_q, v_to_store)

        return output, attn_weights


class MemorizingTransformerBlock(nn.Module):
    """TransformerBlock with local self-attention and external memory augmentation."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        config: MemoryConfig,
    ) -> None:
        super().__init__()
        self.local_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.memory_attn = MemoryAttentionLayer(d_model, config)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, D)

        Returns:
            (B, T, D)
        """
        # Local self-attention with pre-norm and residual
        normed = self.norm1(x)
        attn_out, _ = self.local_attn(normed, normed, normed)
        x = x + attn_out

        # Memory attention with pre-norm and residual
        mem_out, _ = self.memory_attn(self.norm2(x))
        x = x + mem_out

        # FFN with pre-norm and residual
        x = x + self.ffn(self.norm3(x))

        return x


def build_memory_augmented_model(
    d_model: int,
    n_layers: int,
    n_heads: int,
    d_ff: int,
    config: MemoryConfig,
) -> nn.ModuleList:
    """Build a stack of MemorizingTransformerBlocks.

    Returns:
        nn.ModuleList of n_layers MemorizingTransformerBlocks
    """
    return nn.ModuleList(
        [MemorizingTransformerBlock(d_model, n_heads, d_ff, config) for _ in range(n_layers)]
    )


class MemoryUtilizationTracker:
    """Track memory slot utilization statistics over time."""

    def __init__(self, memory: ExternalMemoryBank) -> None:
        self.memory = memory
        self._history: list[Tensor] = []

    def record_step(self, attention_weights: Tensor) -> None:
        """Record which memory slots were accessed.

        Args:
            attention_weights: (B, n_q, memory_size)
        """
        # Aggregate across batch and queries: which slots got any weight?
        accessed = (attention_weights > 0).float().sum(dim=(0, 1))  # (memory_size,)
        self._history.append(accessed.detach().cpu())

    def utilization_stats(self) -> dict[str, float]:
        """Compute utilization statistics over recorded history.

        Returns:
            dict with keys: mean_utilized_slots, coverage, entropy
        """
        if not self._history:
            return {"mean_utilized_slots": 0.0, "coverage": 0.0, "entropy": 0.0}

        # Stack: (steps, memory_size)
        history = torch.stack(self._history, dim=0)
        M = self.memory.config.memory_size

        # Mean number of slots accessed per step
        slots_per_step = (history > 0).float().sum(dim=-1)  # (steps,)
        mean_utilized = slots_per_step.mean().item()

        # Coverage: fraction of slots ever accessed
        ever_accessed = (history.sum(dim=0) > 0).float()
        coverage = ever_accessed.mean().item()

        # Entropy over slot-access distribution
        total_access = history.sum(dim=0)  # (memory_size,)
        total_sum = total_access.sum()
        if total_sum > 0:
            probs = total_access / total_sum
            # Clamp to avoid log(0)
            entropy = -(probs * (probs + 1e-10).log()).sum().item()
        else:
            entropy = 0.0

        return {
            "mean_utilized_slots": mean_utilized,
            "coverage": coverage,
            "entropy": entropy,
        }

    def reset(self) -> None:
        """Clear all recorded history."""
        self._history.clear()
