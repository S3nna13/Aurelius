"""RocketKV — two-stage KV cache compression for long-context inference.

Stage 1 (prefill):  Coarse-grained permanent eviction using observation pooling
                    (SnapKV-style). Identifies critical KV positions.
Stage 2 (decode):   Fine-grained top-k sparse attention over retained positions.

Multi-turn variant maintains critical positions across conversation turns —
essential for Aurelius's ReAct agent loop.

Paper: arXiv:2502.14051 (ICML 2025, NVIDIA)
"""

from __future__ import annotations

import torch
from torch import Tensor


class RocketKVCache:
    """Two-stage KV cache manager combining SnapKV eviction + Quest-style sparse decode."""

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        max_positions: int = 2048,
        top_k_positions: int = 256,
        observation_window: int = 32,
    ):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_positions = max_positions
        self.top_k = top_k_positions
        self.obs_window = observation_window
        # Per-layer retained K, V and position mask
        self.retained_k: dict[int, Tensor] = {}
        self.retained_v: dict[int, Tensor] = {}

    def stage1_evict(self, layer: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Prefill: score positions by attention over last obs_window queries.

        k, v: (B, n_heads, S, head_dim)
        Returns retained (k, v) with top_k positions.
        """
        if k.size(2) <= self.top_k:
            self.retained_k[layer] = k
            self.retained_v[layer] = v
            return k, v

        # Use last obs_window key positions as query proxy
        query_proxy = k[:, :, -self.obs_window :, :]  # (B, H, W, D)
        # Attention scores from proxy over all positions
        scores = torch.matmul(query_proxy, k.transpose(-2, -1))  # (B, H, W, S)
        # Pool across query window and heads
        importance = scores.mean(dim=(1, 2))  # (B, S)
        # Always keep recent tokens + sink tokens
        sink_size = 4
        recent_size = min(64, self.top_k // 4)
        top_k_mid = self.top_k - sink_size - recent_size

        # Get top-k positions from the middle
        mid_scores = importance[:, sink_size:-recent_size]  # (B, S-sink-recent)
        _, mid_idx = mid_scores.topk(min(top_k_mid, mid_scores.size(1)), dim=-1)
        mid_idx = mid_idx + sink_size  # offset by sink

        # Combine: sink + mid top-k + recent
        S = k.size(2)
        sink_idx = torch.arange(sink_size, device=k.device).unsqueeze(0).expand(k.size(0), -1)
        recent_idx = (
            torch.arange(S - recent_size, S, device=k.device).unsqueeze(0).expand(k.size(0), -1)
        )
        all_idx = torch.cat([sink_idx, mid_idx, recent_idx], dim=1)  # (B, top_k)
        all_idx, _ = all_idx.sort(dim=1)

        # Gather retained positions (expand idx for heads and head_dim)
        idx_expanded = all_idx.unsqueeze(1).unsqueeze(-1).expand(-1, k.size(1), -1, k.size(-1))
        k_ret = k.gather(2, idx_expanded)
        v_ret = v.gather(2, idx_expanded)

        self.retained_k[layer] = k_ret
        self.retained_v[layer] = v_ret
        return k_ret, v_ret

    def stage2_get(self, layer: int) -> tuple[Tensor | None, Tensor | None]:
        """Decode: return retained K, V for sparse attention."""
        return self.retained_k.get(layer), self.retained_v.get(layer)

    def update_multiturn(self, layer: int, new_k: Tensor, new_v: Tensor):
        """Multi-turn: append new tokens and re-evict to maintain budget."""
        if layer in self.retained_k:
            k_full = torch.cat([self.retained_k[layer], new_k], dim=2)
            v_full = torch.cat([self.retained_v[layer], new_v], dim=2)
        else:
            k_full, v_full = new_k, new_v
        self.stage1_evict(layer, k_full, v_full)

    def clear(self):
        self.retained_k.clear()
        self.retained_v.clear()


__all__ = ["RocketKVCache"]
