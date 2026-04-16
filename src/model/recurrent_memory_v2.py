"""Recurrent memory v2: GRU-style gated memory + MemoryAttention + RecurrentTransformerLayer.

V2 because recurrent_memory.py exists with a different (RMT segment-token) API.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class RecurrentConfig:
    d_model: int = 512
    n_memory_tokens: int = 16
    d_memory: int = 64
    chunk_size: int = 512
    state_decay: float = 0.9


class GatedMemoryUpdate(nn.Module):
    """GRU-style gated memory update: hidden + memory → new memory."""

    def __init__(self, d_model: int, d_memory: int) -> None:
        super().__init__()
        in_dim = d_model + d_memory
        self.W_z = nn.Linear(in_dim, d_memory, bias=True)
        self.W_r = nn.Linear(in_dim, d_memory, bias=True)
        self.W_c = nn.Linear(in_dim, d_memory, bias=True)

    def forward(self, hidden: Tensor, memory: Tensor) -> Tensor:
        """(B, d_model), (B, d_memory) → (B, d_memory)"""
        cat = torch.cat([hidden, memory], dim=-1)
        z = torch.sigmoid(self.W_z(cat))
        r = torch.sigmoid(self.W_r(cat))
        cat_r = torch.cat([hidden, r * memory], dim=-1)
        c = torch.tanh(self.W_c(cat_r))
        return (1 - z) * memory + z * c


class MemoryAttention(nn.Module):
    """Cross-attention from hidden states to learnable memory tokens."""

    def __init__(self, d_model: int, n_memory_tokens: int) -> None:
        super().__init__()
        self.memory = nn.Parameter(torch.randn(n_memory_tokens, d_model) * 0.02)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.scale = d_model ** -0.5

    def forward(self, hidden: Tensor, memory_state: Tensor) -> Tensor:
        """(B, T, d_model), (B, n_mem, d_model) → (B, T, d_model)"""
        B, T, _ = hidden.shape
        Q = self.q_proj(hidden)  # (B, T, d)
        K = self.k_proj(memory_state)  # (B, n_mem, d)
        V = self.v_proj(memory_state)  # (B, n_mem, d)
        attn = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) * self.scale, dim=-1)  # (B, T, n_mem)
        out = torch.bmm(attn, V)  # (B, T, d)
        return self.out_proj(out)


class RecurrentTransformerLayer(nn.Module):
    """One transformer layer augmented with GRU-style recurrent memory."""

    def __init__(self, config: RecurrentConfig) -> None:
        super().__init__()
        d = config.d_model
        n_mem = config.n_memory_tokens
        # Self-attention
        self.self_attn = nn.MultiheadAttention(d, num_heads=max(1, d // 16), batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        # Memory update (one GRU per memory slot via a shared module)
        self.mem_update = GatedMemoryUpdate(d, d)
        # Cross-attention to memory
        self.mem_attn = MemoryAttention(d, n_mem)
        self.ln2 = nn.LayerNorm(d)
        # FFN
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.ln3 = nn.LayerNorm(d)

    def forward(self, x: Tensor, memory: Tensor) -> Tuple[Tensor, Tensor]:
        """x (B,T,d), memory (B,n_mem,d) → output (B,T,d), new_memory (B,n_mem,d)"""
        # Self-attention
        x = x + self.self_attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        # Update memory from mean of x
        mean_h = x.mean(dim=1)  # (B, d)
        new_slots = []
        for slot_idx in range(memory.shape[1]):
            slot = memory[:, slot_idx, :]  # (B, d)
            new_slot = self.mem_update(mean_h, slot)
            new_slots.append(new_slot.unsqueeze(1))
        new_memory = torch.cat(new_slots, dim=1)  # (B, n_mem, d)
        # Cross-attend to memory
        x = x + self.mem_attn(self.ln2(x), new_memory)
        # FFN
        x = x + self.ffn(self.ln3(x))
        return x, new_memory


def init_memory(batch_size: int, n_memory: int, d_model: int) -> Tensor:
    return torch.zeros(batch_size, n_memory, d_model)


def apply_state_decay(memory: Tensor, decay: float) -> Tensor:
    return memory * decay
