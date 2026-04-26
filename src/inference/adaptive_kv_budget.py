"""Adaptive per-layer KV cache budget allocation.

Allocates KV cache memory unevenly across transformer layers based on
observed attention pattern statistics. Layers with high attention entropy
(diffuse attention) get larger budgets; focused layers get smaller budgets.

Inspired by: AdaKV, DuoAttention (2024), per-layer KV compression research.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# AttentionPatternStats
# ---------------------------------------------------------------------------


class AttentionPatternStats:
    """Tracks attention entropy per layer to guide budget allocation.

    Args:
        n_layers: Number of transformer layers.
        n_heads:  Number of attention heads per layer.
    """

    def __init__(self, n_layers: int, n_heads: int) -> None:
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.entropy_history: list[list[float]] = [[] for _ in range(n_layers)]
        self.update_count: int = 0

    def update(self, layer_idx: int, attn_weights: Tensor) -> None:
        """Record entropy statistics for one layer from one forward pass.

        Args:
            layer_idx:    Index of the layer (0-based).
            attn_weights: Attention probabilities, shape (B, H, T_q, T_k).
        """
        # attn_weights: (B, H, T_q, T_k)
        # entropy per query position: H = -sum(p * log(p + eps), dim=-1)
        entropy = -(attn_weights * torch.log(attn_weights + 1e-10)).sum(dim=-1)
        # entropy shape: (B, H, T_q) → scalar mean
        mean_ent: float = entropy.mean().item()
        self.entropy_history[layer_idx].append(mean_ent)
        self.update_count += 1

    def mean_entropy(self, layer_idx: int) -> float:
        """Return mean entropy recorded for *layer_idx*, or 0.0 if no data."""
        history = self.entropy_history[layer_idx]
        if not history:
            return 0.0
        return sum(history) / len(history)

    def entropy_profile(self) -> list[float]:
        """Return per-layer mean entropy as a list of length *n_layers*."""
        return [self.mean_entropy(i) for i in range(self.n_layers)]


# ---------------------------------------------------------------------------
# KVBudgetAllocator
# ---------------------------------------------------------------------------


class KVBudgetAllocator:
    """Allocates a total KV token budget across transformer layers.

    Args:
        n_layers:           Number of transformer layers.
        total_budget:       Total number of KV token slots to distribute.
        min_budget_per_layer: Minimum tokens guaranteed to each layer.
    """

    def __init__(
        self,
        n_layers: int,
        total_budget: int,
        min_budget_per_layer: int = 16,
    ) -> None:
        self.n_layers = n_layers
        self.total_budget = total_budget
        self.min_budget_per_layer = min_budget_per_layer

    # ------------------------------------------------------------------
    # Allocation strategies
    # ------------------------------------------------------------------

    def allocate_uniform(self) -> list[int]:
        """Distribute budget equally; remainder added to the last layer."""
        base = self.total_budget // self.n_layers
        budgets = [base] * self.n_layers
        remainder = self.total_budget - base * self.n_layers
        budgets[-1] += remainder
        return budgets

    def allocate_by_entropy(self, entropy_profile: list[float]) -> list[int]:
        """Distribute budget proportionally to per-layer entropy (via softmax).

        Higher entropy → more budget.  Every layer is guaranteed at least
        *min_budget_per_layer* tokens.

        Algorithm:
            weights = softmax(entropy_profile)
            extra   = total_budget - n_layers * min_budget_per_layer
            budgets = round(weights * extra) + min_budget_per_layer
            adjust last element to hit exact total_budget

        Args:
            entropy_profile: List of length *n_layers* with per-layer entropies.

        Returns:
            List of *n_layers* positive ints that sum to *total_budget*.
        """
        n = self.n_layers
        min_b = self.min_budget_per_layer
        extra_pool = self.total_budget - n * min_b

        if extra_pool < 0:
            # min budgets alone exceed total; clamp uniformly
            base = max(self.total_budget // n, 1)
            budgets = [base] * n
            budgets[-1] += self.total_budget - base * n
            return budgets

        # softmax over entropy profile
        ep = torch.tensor(entropy_profile, dtype=torch.float64)
        # subtract max for numerical stability
        ep = ep - ep.max()
        weights = torch.exp(ep)
        weights = weights / weights.sum()

        # raw (possibly fractional) extra per layer
        raw_extra = (weights * extra_pool).tolist()

        # round to ints
        budgets = [min_b + int(math.floor(e)) for e in raw_extra]

        # distribute rounding remainder via largest-remainder method
        current_sum = sum(budgets)
        deficit = self.total_budget - current_sum
        # rank layers by fractional part, descending
        fracs = [(raw_extra[i] - math.floor(raw_extra[i]), i) for i in range(n)]
        fracs.sort(key=lambda x: -x[0])
        for _, idx in fracs[:deficit]:
            budgets[idx] += 1

        # exact sum enforcement via last-element adjustment
        diff = self.total_budget - sum(budgets)
        budgets[-1] += diff

        # guarantee minimum
        for i in range(n):
            if budgets[i] < min_b:
                budgets[i] = min_b
        # re-adjust last element if minimums pushed total over
        diff2 = self.total_budget - sum(budgets)
        budgets[-1] += diff2

        return budgets

    def allocate_by_layer_type(self, layer_types: list[str]) -> list[int]:
        """Assign budgets based on categorical layer type labels.

        Types and their relative weights:
            'full'  → base * 2
            'local' → base // 2
            'sink'  → min_budget_per_layer

        The raw amounts are then rescaled so the total equals *total_budget*.

        Args:
            layer_types: List of *n_layers* strings, each 'full', 'local', or 'sink'.

        Returns:
            List of *n_layers* positive ints approximately summing to *total_budget*.
        """
        base = self.total_budget // self.n_layers
        raw: list[int] = []
        for lt in layer_types:
            if lt == "full":
                raw.append(base * 2)
            elif lt == "local":
                raw.append(max(base // 2, 1))
            else:  # 'sink' or anything else
                raw.append(self.min_budget_per_layer)

        # rescale so total ≈ total_budget
        raw_sum = sum(raw)
        if raw_sum == 0:
            return self.allocate_uniform()

        scale = self.total_budget / raw_sum
        budgets = [max(int(round(r * scale)), self.min_budget_per_layer) for r in raw]

        # exact adjustment via last element
        diff = self.total_budget - sum(budgets)
        budgets[-1] = max(budgets[-1] + diff, self.min_budget_per_layer)

        return budgets


# ---------------------------------------------------------------------------
# LayerKVCache
# ---------------------------------------------------------------------------


class LayerKVCache:
    """Fixed-budget KV cache for a single transformer layer.

    Maintains a sliding window of the most recent *budget* tokens.

    Args:
        budget:  Maximum number of KV tokens to retain.
        n_heads: Number of attention heads.
        d_head:  Head dimensionality.
    """

    def __init__(self, budget: int, n_heads: int, d_head: int) -> None:
        self.budget = budget
        self.n_heads = n_heads
        self.d_head = d_head
        self.keys: Tensor | None = None
        self.values: Tensor | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        new_keys: Tensor,
        new_values: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Append new KV pairs and return the (truncated) full cache.

        Args:
            new_keys:   (B, H, T_new, d_head)
            new_values: (B, H, T_new, d_head)

        Returns:
            Tuple (keys, values), each shaped (B, H, min(total, budget), d_head).
        """
        if self.keys is None:
            self.keys = new_keys
            self.values = new_values
        else:
            self.keys = torch.cat([self.keys, new_keys], dim=2)
            self.values = torch.cat([self.values, new_values], dim=2)

        # Truncate to last `budget` tokens
        if self.keys.shape[2] > self.budget:
            self.keys = self.keys[:, :, -self.budget :, :]
            self.values = self.values[:, :, -self.budget :, :]

        return self.keys, self.values

    def size(self) -> int:
        """Return the current number of cached tokens (T dimension)."""
        if self.keys is None:
            return 0
        return self.keys.shape[2]

    def clear(self) -> None:
        """Reset the cache to empty."""
        self.keys = None
        self.values = None


# ---------------------------------------------------------------------------
# AdaptiveKVCacheManager
# ---------------------------------------------------------------------------


class AdaptiveKVCacheManager:
    """Manages per-layer KV caches with adaptive budget allocation.

    Starts with a uniform budget and supports rebalancing based on observed
    attention entropy statistics.

    Args:
        n_layers:     Number of transformer layers.
        n_heads:      Number of attention heads per layer.
        d_head:       Head dimensionality.
        total_budget: Total KV token slots to distribute across layers.
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        d_head: int,
        total_budget: int,
    ) -> None:
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_head
        self.total_budget = total_budget

        self.allocator = KVBudgetAllocator(n_layers, total_budget)
        self.budgets: list[int] = self.allocator.allocate_uniform()
        self.caches: list[LayerKVCache] = [LayerKVCache(b, n_heads, d_head) for b in self.budgets]
        self.stats = AttentionPatternStats(n_layers, n_heads)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_layer(
        self,
        layer_idx: int,
        keys: Tensor,
        values: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Append KV pairs to the specified layer's cache.

        Args:
            layer_idx: Which layer to update.
            keys:      (B, H, T_new, d_head)
            values:    (B, H, T_new, d_head)

        Returns:
            Tuple (keys, values) from that layer's cache after update.
        """
        return self.caches[layer_idx].update(keys, values)

    def rebalance(self) -> None:
        """Recompute layer budgets from entropy statistics and recreate caches.

        Existing cached content is discarded — call this between sequences.
        """
        profile = self.stats.entropy_profile()
        self.budgets = self.allocator.allocate_by_entropy(profile)
        for cache in self.caches:
            cache.clear()
        self.caches = [LayerKVCache(b, self.n_heads, self.d_head) for b in self.budgets]

    def total_cached_tokens(self) -> int:
        """Return total number of tokens currently held across all layer caches."""
        return sum(c.size() for c in self.caches)
