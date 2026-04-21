"""Adaptive KV Eviction — attention-score-based dynamic KV cache eviction.

Inspired by H2O (Zhang et al. 2023), PyramidKV (Cai et al. 2024), and
SnapKV (Li et al. 2024).  Token importance is measured by the cumulative
attention score the token has received across all previous decoding steps.
Tokens that attract little attention are evicted subject to a per-head,
per-layer budget, while the most-recent ``recent_window`` tokens are always
protected.

Pure PyTorch — no external ML libraries required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveKVConfig:
    """Hyper-parameters for AdaptiveKVEvictionManager."""

    n_layers: int = 24
    n_heads: int = 16
    head_dim: int = 128
    max_seq_len: int = 8192
    budget_ratio: float = 0.3       # keep top ``budget_ratio`` fraction
    min_budget: int = 64            # lower bound on kept tokens
    recent_window: int = 32         # always keep last N tokens (never evict)
    accumulate_steps: int = 1       # steps between eviction passes (reserved)


# ---------------------------------------------------------------------------
# Cache state
# ---------------------------------------------------------------------------

@dataclass
class KVCacheState:
    """Per-layer KV cache with accumulated attention statistics."""

    keys: Tensor                    # [n_heads, T, head_dim]
    values: Tensor                  # [n_heads, T, head_dim]
    attn_scores_acc: Tensor         # [n_heads, T]  accumulated attention
    kept_positions: List[int]       # original positions of kept tokens
    eviction_count: int = 0


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class AdaptiveKVEvictionManager:
    """Manages per-layer adaptive KV cache eviction.

    Usage::

        cfg   = AdaptiveKVConfig(n_heads=4, head_dim=16, budget_ratio=0.3)
        mgr   = AdaptiveKVEvictionManager(cfg)
        state = mgr.new_state(keys, values)     # initialise with first keys/values
        state = mgr.update_scores(state, attn)  # accumulate per-step attention
        if mgr.should_evict(state):
            state = mgr.evict(state)
    """

    def __init__(self, config: AdaptiveKVConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Budget computation
    # ------------------------------------------------------------------

    def compute_budget(self, current_len: int) -> int:
        """Return the target number of tokens to keep.

        budget = max(min_budget, floor(current_len * budget_ratio))
        The result is clamped to [min_budget, current_len].
        """
        cfg = self.config
        budget = max(cfg.min_budget, int(current_len * cfg.budget_ratio))
        return min(budget, current_len)

    # ------------------------------------------------------------------
    # Eviction gate
    # ------------------------------------------------------------------

    def should_evict(self, state: KVCacheState) -> bool:
        """Return True when the cache exceeds the computed budget."""
        current_len = len(state.kept_positions)
        return current_len > self.compute_budget(current_len)

    # ------------------------------------------------------------------
    # Score accumulation
    # ------------------------------------------------------------------

    def update_scores(
        self,
        state: KVCacheState,
        new_attn_weights: Tensor,  # [n_heads, T_current]
    ) -> KVCacheState:
        """Accumulate per-step attention weights into the running totals.

        ``new_attn_weights`` must have the same sequence length as the
        current cache (i.e. T_current == len(kept_positions)).
        """
        state.attn_scores_acc = state.attn_scores_acc + new_attn_weights
        return state

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def evict(self, state: KVCacheState) -> KVCacheState:
        """Evict low-importance tokens, returning a pruned KVCacheState.

        Algorithm:
        1. Compute budget ``k``.
        2. Protect the ``recent_window`` most-recently-seen tokens
           (last positions in ``kept_positions``).
        3. Among the older positions score them by ``attn_scores_acc``
           averaged across heads; keep the top ``(k - recent_window)``.
        4. Rebuild KV tensors and metadata.

        If the cache is already within budget the state is returned unchanged
        (eviction_count is NOT incremented in that case).
        """
        cfg = self.config
        T = len(state.kept_positions)
        k = self.compute_budget(T)

        if T <= k:
            # Nothing to evict — already within budget.
            return state

        # Split into "older" and "recent" halves.
        n_recent = min(cfg.recent_window, k)   # can't protect more than budget
        n_old    = T - n_recent                 # number of older positions
        n_keep_old = k - n_recent               # older positions we may keep

        # Indices into the *current* cache tensors (0 … T-1).
        old_indices    = list(range(n_old))          # older tokens
        recent_indices = list(range(n_old, T))        # always kept

        if n_keep_old <= 0 or n_old == 0:
            # Budget is tight — keep only recent tokens.
            keep_indices = recent_indices
        else:
            # Score older tokens: mean across heads → [n_old]
            old_scores = state.attn_scores_acc[:, :n_old].mean(dim=0)  # [n_old]
            # Select top-n_keep_old by score.
            n_keep_old = min(n_keep_old, n_old)
            topk_vals, topk_local = torch.topk(old_scores, n_keep_old)
            selected_old = topk_local.sort().values.tolist()
            keep_indices = selected_old + recent_indices

        # Rebuild tensors.
        idx_t = torch.tensor(keep_indices, dtype=torch.long,
                             device=state.keys.device)
        new_keys   = state.keys[:, idx_t, :]           # [n_heads, k', head_dim]
        new_values = state.values[:, idx_t, :]
        new_acc    = state.attn_scores_acc[:, idx_t]   # [n_heads, k']
        new_kept   = [state.kept_positions[i] for i in keep_indices]

        return KVCacheState(
            keys=new_keys,
            values=new_values,
            attn_scores_acc=new_acc,
            kept_positions=new_kept,
            eviction_count=state.eviction_count + 1,
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    def new_state(self, keys: Tensor, values: Tensor) -> KVCacheState:
        """Initialise a fresh KVCacheState from the given key/value tensors.

        Args:
            keys:   ``[n_heads, T, head_dim]``
            values: ``[n_heads, T, head_dim]``

        Returns:
            A :class:`KVCacheState` with zero accumulated scores and
            ``kept_positions = list(range(T))``.
        """
        n_heads, T, _ = keys.shape
        acc = torch.zeros(n_heads, T, dtype=keys.dtype, device=keys.device)
        return KVCacheState(
            keys=keys.clone(),
            values=values.clone(),
            attn_scores_acc=acc,
            kept_positions=list(range(T)),
            eviction_count=0,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def efficiency_stats(
        self,
        state: KVCacheState,
        original_len: int,
    ) -> dict:
        """Return a dictionary of efficiency metrics.

        Keys
        ----
        cache_size_ratio : float
            Fraction of original tokens still in cache.
        evictions : int
            Number of eviction passes performed so far.
        memory_saved : float
            Approximate bytes freed (float32, both K and V).
        """
        cfg = self.config
        kept = len(state.kept_positions)
        freed = original_len - kept
        # Each token occupies head_dim floats * n_heads heads * 2 (K+V) * 4 bytes
        bytes_per_token = cfg.head_dim * cfg.n_heads * 2 * 4
        return {
            "cache_size_ratio": kept / max(original_len, 1),
            "evictions": state.eviction_count,
            "memory_saved": float(freed * bytes_per_token),
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from src.inference import DECODER_REGISTRY  # noqa: E402

DECODER_REGISTRY["adaptive_kv_eviction"] = AdaptiveKVEvictionManager
