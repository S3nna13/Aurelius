"""
KV Cache Compression — H2O (Heavy-Hitter Oracle) and SnapKV
Pure PyTorch, no external ML libraries.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# AttentionScoreAccumulator
# ---------------------------------------------------------------------------


class AttentionScoreAccumulator:
    """Track per-token importance via accumulated attention scores."""

    def __init__(self, recent_window: int = 4) -> None:
        self.recent_window = recent_window
        self.accumulated: Tensor | None = None  # (B, H, T_kv)

    def update(self, attn_weights: Tensor) -> None:
        """
        Accumulate attention weights over the query dimension.

        Args:
            attn_weights: (B, H, T_q, T_kv) — attention probabilities
        """
        # Sum over query dim → (B, H, T_kv)
        incoming = attn_weights.sum(dim=2)  # (B, H, T_kv)

        if self.accumulated is None:
            self.accumulated = incoming.clone()
        else:
            current_len = self.accumulated.shape[-1]
            new_len = incoming.shape[-1]

            if new_len > current_len:
                # Sequence grew; pad existing with zeros then add
                pad_size = new_len - current_len
                self.accumulated = F.pad(self.accumulated, (0, pad_size))
                self.accumulated = self.accumulated + incoming
            elif new_len == current_len:
                self.accumulated = self.accumulated + incoming
            else:
                # Partial update (e.g. single new token attending to full cache)
                # incoming covers [0..new_len-1]; align and add
                self.accumulated[..., :new_len] = self.accumulated[..., :new_len] + incoming

    def importance_scores(self) -> Tensor:
        """
        Return normalised importance scores (sum-to-1 along T_kv).

        Returns:
            (B, H, T_kv) float tensor
        """
        if self.accumulated is None:
            raise RuntimeError("No data accumulated yet; call update() first.")

        total = self.accumulated.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        return self.accumulated / total

    def reset(self) -> None:
        """Clear accumulated state."""
        self.accumulated = None


# ---------------------------------------------------------------------------
# H2OEvictionPolicy
# ---------------------------------------------------------------------------


class H2OEvictionPolicy:
    """Heavy-Hitter Oracle eviction policy."""

    def __init__(self, budget: int, recent_window: int = 4) -> None:
        self.budget = budget
        self.recent_window = recent_window

    def select_keep_indices(self, accumulated_scores: Tensor, current_len: int) -> Tensor:
        """
        Select which KV positions to keep.

        Args:
            accumulated_scores: (B, H, T_kv)
            current_len: actual sequence length

        Returns:
            LongTensor of shape (B, budget) with indices to keep
            (or (B, current_len) when no eviction is needed)
        """
        B, H, T_kv = accumulated_scores.shape

        if current_len <= self.budget:
            # No eviction needed — return all indices (same for each batch/head)
            indices = torch.arange(current_len, device=accumulated_scores.device)
            return indices.unsqueeze(0).expand(B, -1)  # (B, current_len)

        recent_start = max(0, current_len - self.recent_window)
        recent_indices = torch.arange(
            recent_start, current_len, device=accumulated_scores.device
        )  # length ≤ recent_window

        heavy_budget = self.budget - recent_indices.shape[0]

        # Average over heads for a single importance vector per batch
        # shape: (B, T_kv)
        scores_avg = accumulated_scores.mean(dim=1)

        # Zero out recent positions so they don't compete for heavy-hitter slots
        scores_for_hh = scores_avg.clone()
        scores_for_hh[:, recent_start:] = -1.0

        # Pick top heavy_budget from earlier positions
        if heavy_budget > 0:
            _, top_idx = scores_for_hh[:, :recent_start].topk(
                min(heavy_budget, recent_start), dim=-1
            )  # (B, heavy_budget)
            top_idx_sorted, _ = top_idx.sort(dim=-1)
        else:
            top_idx_sorted = torch.zeros(B, 0, dtype=torch.long, device=accumulated_scores.device)

        # Combine: heavy hitters + recency tail
        recent_exp = recent_indices.unsqueeze(0).expand(B, -1)  # (B, recent_window)
        keep = torch.cat([top_idx_sorted, recent_exp], dim=-1)  # (B, budget)
        return keep

    def evict(
        self,
        keys: Tensor,
        values: Tensor,
        accumulated_scores: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Evict unimportant KV pairs.

        Args:
            keys:   (B, H, T, D_head)
            values: (B, H, T, D_head)
            accumulated_scores: (B, H, T)

        Returns:
            (keys_kept, values_kept) each (B, H, budget, D_head)
        """
        B, H, T, D = keys.shape
        keep_idx = self.select_keep_indices(accumulated_scores, T)  # (B, budget|T)
        budget_actual = keep_idx.shape[-1]

        # Expand indices for gathering: (B, 1, budget, 1) → (B, H, budget, D)
        idx = keep_idx.unsqueeze(1).unsqueeze(-1)  # (B, 1, budget_actual, 1)
        idx_k = idx.expand(B, H, budget_actual, D)
        idx_v = idx.expand(B, H, budget_actual, D)

        keys_kept = keys.gather(2, idx_k)
        values_kept = values.gather(2, idx_v)
        return keys_kept, values_kept


# ---------------------------------------------------------------------------
# SnapKVPolicy
# ---------------------------------------------------------------------------


class SnapKVPolicy:
    """SnapKV: cluster-based KV compression using an observation window."""

    def __init__(
        self,
        budget: int,
        observation_window: int = 4,
        pooling: str = "mean",
    ) -> None:
        if pooling not in ("mean", "max"):
            raise ValueError(f"pooling must be 'mean' or 'max', got '{pooling}'")
        self.budget = budget
        self.observation_window = observation_window
        self.pooling = pooling

    def compress(
        self,
        keys: Tensor,
        values: Tensor,
        query_window: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Compress KV cache using recent query attention as importance signal.

        Args:
            keys:         (B, H, T, D_head)
            values:       (B, H, T, D_head)
            query_window: (B, H, obs_window, D_head)

        Returns:
            (keys_compressed, values_compressed) each (B, H, budget, D_head)
        """
        B, H, T, D = keys.shape
        scale = math.sqrt(D)

        # Attention scores: (B, H, obs_window, T)
        attn_scores = torch.matmul(query_window, keys.transpose(-2, -1)) / scale
        attn_probs = F.softmax(attn_scores, dim=-1)  # (B, H, obs_window, T)

        # Pool over the obs_window dim to get per-kv importance: (B, H, T)
        if self.pooling == "mean":
            importance = attn_probs.mean(dim=2)
        else:  # max
            importance, _ = attn_probs.max(dim=2)

        budget = min(self.budget, T)

        # Average over heads for a single ranking per batch
        importance_avg = importance.mean(dim=1)  # (B, T)
        _, top_idx = importance_avg.topk(budget, dim=-1)  # (B, budget)
        top_idx_sorted, _ = top_idx.sort(dim=-1)  # (B, budget)

        # Gather: expand to (B, H, budget, D)
        idx = top_idx_sorted.unsqueeze(1).unsqueeze(-1).expand(B, H, budget, D)
        keys_compressed = keys.gather(2, idx)
        values_compressed = values.gather(2, idx)
        return keys_compressed, values_compressed


# ---------------------------------------------------------------------------
# CompressedKVCache
# ---------------------------------------------------------------------------


class CompressedKVCache:
    """Unified KV cache supporting H2O, SnapKV, and unlimited modes."""

    def __init__(
        self,
        policy: str = "h2o",
        budget: int = 32,
        **policy_kwargs,
    ) -> None:
        if policy not in ("h2o", "snapkv", "none"):
            raise ValueError(f"Unknown policy '{policy}'. Choose h2o/snapkv/none.")

        self.policy_name = policy
        self.budget = budget

        # Per-layer storage
        self._keys: dict[int, Tensor] = {}
        self._values: dict[int, Tensor] = {}

        # Build policy objects
        if policy == "h2o":
            recent_window = policy_kwargs.get("recent_window", 4)
            self._h2o = H2OEvictionPolicy(budget=budget, recent_window=recent_window)
            self._accumulator: dict[int, AttentionScoreAccumulator] = {}
            self._recent_window = recent_window
        elif policy == "snapkv":
            obs_window = policy_kwargs.get("observation_window", 4)
            pooling = policy_kwargs.get("pooling", "mean")
            self._snapkv = SnapKVPolicy(
                budget=budget,
                observation_window=obs_window,
                pooling=pooling,
            )

    def update(
        self,
        layer_idx: int,
        new_keys: Tensor,
        new_values: Tensor,
        attn_weights: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Append new KV pairs and compress if over budget.

        Args:
            layer_idx:   layer index
            new_keys:    (B, H, T_new, D_head)
            new_values:  (B, H, T_new, D_head)
            attn_weights: (B, H, T_q, T_kv) — required for h2o; used as
                          query proxy for snapkv

        Returns:
            (stored_keys, stored_values) after potential compression
        """
        if layer_idx in self._keys:
            keys = torch.cat([self._keys[layer_idx], new_keys], dim=2)
            values = torch.cat([self._values[layer_idx], new_values], dim=2)
        else:
            keys = new_keys
            values = new_values

        if self.policy_name == "none":
            self._keys[layer_idx] = keys
            self._values[layer_idx] = values
            return keys, values

        if self.policy_name == "h2o":
            # Ensure accumulator exists for this layer
            if layer_idx not in self._accumulator:
                self._accumulator[layer_idx] = AttentionScoreAccumulator(
                    recent_window=self._recent_window
                )
            acc = self._accumulator[layer_idx]

            if attn_weights is not None:
                acc.update(attn_weights)

            T = keys.shape[2]
            if T > self.budget:
                if acc.accumulated is not None:
                    # Pad/trim accumulated scores to match T
                    acc_scores = acc.accumulated
                    acc_T = acc_scores.shape[-1]
                    if acc_T < T:
                        acc_scores = F.pad(acc_scores, (0, T - acc_T))
                    elif acc_T > T:
                        acc_scores = acc_scores[..., :T]
                else:
                    # Uniform scores if no attention weights provided
                    B, H = keys.shape[:2]
                    acc_scores = torch.ones(B, H, T, device=keys.device)

                keys, values = self._h2o.evict(keys, values, acc_scores)
                # Reset accumulator scores to match compressed length
                acc.accumulated = None

        elif self.policy_name == "snapkv":
            T = keys.shape[2]
            if T > self.budget:
                if attn_weights is not None:
                    # Use attn_weights queries as the observation window
                    # attn_weights: (B, H, T_q, T_kv)
                    query_window = attn_weights.mean(dim=-1, keepdim=True)
                    # Build a synthetic query_window from new_keys as proxy
                    query_window = new_keys
                else:
                    # Use last obs_window keys as proxy queries
                    obs = self._snapkv.observation_window
                    query_window = keys[:, :, -obs:, :]

                keys, values = self._snapkv.compress(keys, values, query_window)

        self._keys[layer_idx] = keys
        self._values[layer_idx] = values
        return keys, values

    def cache_size(self, layer_idx: int) -> int:
        """Return current sequence length stored for layer_idx."""
        if layer_idx not in self._keys:
            return 0
        return self._keys[layer_idx].shape[2]

    def reset(self) -> None:
        """Clear all cached state."""
        self._keys.clear()
        self._values.clear()
        if self.policy_name == "h2o":
            self._accumulator.clear()


# ---------------------------------------------------------------------------
# KVCompressionAnalyzer
# ---------------------------------------------------------------------------


class KVCompressionAnalyzer:
    """Measure compression quality metrics."""

    def __init__(self) -> None:
        pass

    def memory_reduction(self, original_len: int, compressed_len: int) -> float:
        """
        Fraction of memory saved: 1 - compressed/original.

        Returns:
            float in [0, 1]; 0 means no reduction, 1 means fully compressed.
        """
        if original_len <= 0:
            raise ValueError("original_len must be > 0")
        ratio = compressed_len / original_len
        return float(1.0 - ratio)

    def attention_approximation_error(
        self,
        full_attn_output: Tensor,
        compressed_attn_output: Tensor,
    ) -> float:
        """
        Mean relative L2 error between full and compressed attention outputs.

        Returns:
            float ≥ 0; 0.0 for identical tensors.
        """
        diff = full_attn_output - compressed_attn_output
        norm_diff = diff.norm(dim=-1)
        norm_full = full_attn_output.norm(dim=-1).clamp(min=1e-9)
        return float((norm_diff / norm_full).mean().item())

    def perplexity_degradation(
        self,
        full_logprobs: Tensor,
        compressed_logprobs: Tensor,
    ) -> float:
        """
        KL divergence D_KL(full || compressed) as perplexity degradation proxy.

        Args:
            full_logprobs:       (*, V) log-probabilities
            compressed_logprobs: (*, V) log-probabilities

        Returns:
            float ≥ 0; 0.0 for identical distributions.
        """
        # KL(P || Q) = sum P * (log P - log Q)
        p = full_logprobs.exp()
        kl = (p * (full_logprobs - compressed_logprobs)).sum(dim=-1).mean()
        return float(kl.clamp(min=0.0).item())
