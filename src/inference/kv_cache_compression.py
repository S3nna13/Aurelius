"""KV cache compression: H2O (Heavy-Hitter Oracle) and StreamingLLM-style eviction."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class KVCacheConfig:
    """Configuration for KV cache compression."""

    budget: int = 128
    n_sink_tokens: int = 4
    strategy: str = "h2o"  # "h2o" | "random" | "recent"
    accumulation_steps: int = 1


# ---------------------------------------------------------------------------
# Attention score accumulator
# ---------------------------------------------------------------------------


class AttentionScoreAccumulator:
    """Tracks cumulative attention scores per token to identify heavy hitters."""

    def __init__(self, budget: int, n_sink_tokens: int) -> None:
        self.budget = budget
        self.n_sink_tokens = n_sink_tokens
        self.scores: Tensor | None = None

    def update(self, attn_weights: Tensor) -> None:
        """Accumulate column-wise importance from attention weights.

        Args:
            attn_weights: shape (n_heads, T_q, T_k)
        """
        # Sum over heads and query dimension -> shape (T_k,)
        token_scores = attn_weights.sum(dim=(0, 1))  # (T_k,)
        if self.scores is None:
            self.scores = token_scores.clone()
        else:
            current_len = self.scores.shape[0]
            new_len = token_scores.shape[0]
            if new_len > current_len:
                # Extend with zeros for newly added tokens then accumulate
                padding = token_scores.new_zeros(new_len - current_len)
                self.scores = torch.cat([self.scores, padding], dim=0)
            self.scores[:new_len] += token_scores

    def get_heavy_hitters(self, n: int) -> Tensor:
        """Return indices of top-n tokens by score (excluding sink tokens).

        Returns:
            LongTensor shape (n,) sorted ascending.
        """
        if self.scores is None:
            return torch.zeros(0, dtype=torch.long)

        T = self.scores.shape[0]
        non_sink = self.scores[self.n_sink_tokens:]  # scores for non-sink positions
        n_available = non_sink.shape[0]
        n = min(n, n_available)

        if n <= 0:
            return torch.zeros(0, dtype=torch.long)

        # Indices in original score tensor
        _, top_local = torch.topk(non_sink, k=n, largest=True, sorted=False)
        top_global = top_local + self.n_sink_tokens
        top_global_sorted, _ = torch.sort(top_global)
        return top_global_sorted


# ---------------------------------------------------------------------------
# H2O KV Cache
# ---------------------------------------------------------------------------


class H2OKVCache:
    """KV cache with Heavy-Hitter Oracle eviction."""

    def __init__(
        self,
        config: KVCacheConfig,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
    ) -> None:
        self.config = config
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        self.keys: list[Tensor | None] = [None] * n_layers
        self.values: list[Tensor | None] = [None] * n_layers
        self.scores: list[AttentionScoreAccumulator] = [
            AttentionScoreAccumulator(config.budget, config.n_sink_tokens)
            for _ in range(n_layers)
        ]
        self.seq_len: int = 0
        self._step: int = 0

    def update(
        self,
        layer_idx: int,
        new_k: Tensor,
        new_v: Tensor,
        attn_weights: Tensor | None = None,
    ) -> None:
        """Append new K,V tokens and optionally evict old ones.

        Args:
            layer_idx: which transformer layer.
            new_k: shape (B, n_kv_heads, T_new, head_dim)
            new_v: shape (B, n_kv_heads, T_new, head_dim)
            attn_weights: optional (n_heads, T_q, T_k) for this layer.
        """
        # Append to cache
        if self.keys[layer_idx] is None:
            self.keys[layer_idx] = new_k
            self.values[layer_idx] = new_v
        else:
            self.keys[layer_idx] = torch.cat([self.keys[layer_idx], new_k], dim=2)
            self.values[layer_idx] = torch.cat([self.values[layer_idx], new_v], dim=2)

        # Update accumulator
        if attn_weights is not None:
            self.scores[layer_idx].update(attn_weights)

        # Update global seq_len using layer 0 as reference
        if layer_idx == 0:
            self.seq_len = self.keys[0].shape[2]

        # Evict if over budget
        cfg = self.config
        current_size = self.keys[layer_idx].shape[2]
        if current_size > cfg.budget:
            self._evict(layer_idx)

    def _evict(self, layer_idx: int) -> None:
        """Remove tokens to bring cache back within budget."""
        cfg = self.config
        k = self.keys[layer_idx]  # (B, H, T, D)
        v = self.values[layer_idx]

        T = k.shape[2]
        n_to_keep = cfg.budget - cfg.n_sink_tokens  # non-sink slots

        strategy = cfg.strategy

        if strategy == "h2o":
            acc = self.scores[layer_idx]
            hitter_indices = acc.get_heavy_hitters(n_to_keep)
            sink_indices = torch.arange(cfg.n_sink_tokens, device=k.device)
            keep = torch.cat([sink_indices, hitter_indices])
            keep = torch.unique(keep, sorted=True)
        elif strategy == "recent":
            sink_indices = torch.arange(cfg.n_sink_tokens, device=k.device)
            recent_start = max(cfg.n_sink_tokens, T - n_to_keep)
            recent_indices = torch.arange(recent_start, T, device=k.device)
            keep = torch.cat([sink_indices, recent_indices])
            keep = torch.unique(keep, sorted=True)
        elif strategy == "random":
            sink_indices = torch.arange(cfg.n_sink_tokens, device=k.device)
            non_sink_pool = torch.arange(cfg.n_sink_tokens, T, device=k.device)
            perm = torch.randperm(non_sink_pool.shape[0], generator=torch.Generator().manual_seed(42))
            chosen = non_sink_pool[perm[:n_to_keep]]
            keep = torch.cat([sink_indices, chosen])
            keep, _ = torch.sort(keep)
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

        self.keys[layer_idx] = k[:, :, keep, :]
        self.values[layer_idx] = v[:, :, keep, :]

        # Trim score tensor to match kept positions
        acc = self.scores[layer_idx]
        if acc.scores is not None and acc.scores.shape[0] == T:
            acc.scores = acc.scores[keep]

        # Update seq_len reference
        if layer_idx == 0:
            self.seq_len = self.keys[0].shape[2]

    def get(self, layer_idx: int) -> tuple[Tensor, Tensor]:
        """Return (keys, values) for given layer."""
        k = self.keys[layer_idx]
        v = self.values[layer_idx]
        if k is None:
            raise ValueError(f"Layer {layer_idx} has no cached tokens.")
        return k, v

    def size(self) -> int:
        """Current number of tokens in cache (layer 0 as reference)."""
        if self.keys[0] is None:
            return 0
        return self.keys[0].shape[2]

    def reset(self) -> None:
        """Clear all cached keys/values."""
        self.keys = [None] * self.n_layers
        self.values = [None] * self.n_layers
        self.scores = [
            AttentionScoreAccumulator(self.config.budget, self.config.n_sink_tokens)
            for _ in range(self.n_layers)
        ]
        self.seq_len = 0
        self._step = 0


# ---------------------------------------------------------------------------
# Pure compression function
# ---------------------------------------------------------------------------


def compress_kv_cache(
    keys: Tensor,
    values: Tensor,
    scores: Tensor,
    budget: int,
    n_sink: int,
) -> tuple[Tensor, Tensor]:
    """Select budget tokens from K,V using importance scores.

    Args:
        keys:   (B, H, T, D)
        values: (B, H, T, D)
        scores: (T,) importance score per token
        budget: total number of tokens to keep
        n_sink: number of initial tokens always kept

    Returns:
        (compressed_keys, compressed_values) each (B, H, budget, D) or smaller.
    """
    T = keys.shape[2]
    if T <= budget:
        return keys, values

    n_heavy = budget - n_sink
    sink_idx = torch.arange(n_sink, device=keys.device)

    non_sink_scores = scores[n_sink:]
    n_available = non_sink_scores.shape[0]
    n_heavy = min(n_heavy, n_available)

    if n_heavy > 0:
        _, top_local = torch.topk(non_sink_scores, k=n_heavy, largest=True, sorted=False)
        top_global = top_local + n_sink
        top_global_sorted, _ = torch.sort(top_global)
        keep = torch.cat([sink_idx, top_global_sorted])
    else:
        keep = sink_idx

    compressed_keys = keys[:, :, keep, :]
    compressed_values = values[:, :, keep, :]
    return compressed_keys, compressed_values


# ---------------------------------------------------------------------------
# Attention layer with H2O cache
# ---------------------------------------------------------------------------


class CachedAttentionLayer(nn.Module):
    """Multi-head attention layer that uses H2OKVCache for decode."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        cache_config: KVCacheConfig,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)

        self.cache: H2OKVCache = H2OKVCache(
            config=cache_config,
            n_layers=1,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
        )
        self.layer_idx: int = 0

    def forward(self, x: Tensor, use_cache: bool = False) -> Tensor:
        """Compute attention, optionally using KV cache.

        Args:
            x: (B, T, D)
            use_cache: if True, update and retrieve from H2OKVCache.

        Returns:
            output: (B, T, D)
        """
        B, T, D = x.shape

        # Projections
        q = self.q_proj(x)  # (B, T, n_heads * head_dim)
        k = self.k_proj(x)  # (B, T, n_kv_heads * head_dim)
        v = self.v_proj(x)

        # Reshape to (B, n_heads/n_kv_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if use_cache:
            self.cache.update(self.layer_idx, k, v)
            k, v = self.cache.get(self.layer_idx)

        # Expand k/v if grouped-query attention (n_kv_heads < n_heads)
        if self.n_kv_heads != self.n_heads:
            repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, T_q, T_k)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)  # (B, H, T_q, head_dim)

        # Merge heads and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        output = self.o_proj(attn_out)
        return output
