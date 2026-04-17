"""
H2O: Heavy-Hitter Oracle KV Cache Compression.

Implements the H2O eviction policy from:
    Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative Inference
    of Large Language Models", NeurIPS 2023 (arXiv:2306.14048).

Key idea (Section 3):
    Attention score distributions are highly skewed.  A small fraction of
    tokens — "heavy hitters" — receive the majority of attention mass.  H2O
    retains these heavy hitters in the KV cache and evicts "light" tokens,
    keeping cache size bounded at *k* tokens total.

Practical variant with local window:
    - Heavy-hitter budget:  k_hh = k - w   (top-scored tokens)
    - Recent-token budget:  w               (always keep the w most recent)
    - Total cache size:     k

Paper notation preserved in variable names:
    k   = max_size      (total cache budget)
    w   = recent_window (recent-token budget)
    a_t = attn_scores_to_new_token  (attention FROM new token t TO cache)
    s_j = score_j       (cumulative importance score for token j)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# H2OEvictionPolicy
# ---------------------------------------------------------------------------

class H2OEvictionPolicy:
    """Select the cache index to evict given per-token cumulative scores.

    Implements the greedy heavy-hitter eviction rule: always evict the token
    with the *lowest* cumulative importance score, subject to the constraint
    that tokens in the recent window (``recent_mask=True``) are never evicted.

    Args:
        None — stateless policy object.
    """

    def select_evict(self, scores: Tensor, recent_mask: Tensor) -> int:
        """Return the cache index of the token that should be evicted.

        Args:
            scores:      1-D float tensor of shape ``(cache_size,)`` — the
                         cumulative attention score ``s_j`` for each position
                         ``j`` currently in the cache.
            recent_mask: 1-D bool tensor of shape ``(cache_size,)`` — ``True``
                         for positions in the recent window that must NOT be
                         evicted.

        Returns:
            Integer index into ``scores`` of the token to evict.

        Raises:
            ValueError: if every position is protected by ``recent_mask``
                        (no evictable candidate exists).
        """
        if scores.ndim != 1:
            raise ValueError(
                f"scores must be 1-D, got shape {tuple(scores.shape)}"
            )
        if recent_mask.shape != scores.shape:
            raise ValueError(
                f"recent_mask shape {tuple(recent_mask.shape)} must match "
                f"scores shape {tuple(scores.shape)}"
            )

        # Mask protected (recent) positions with +inf so they are never chosen
        # as the argmin.
        masked_scores = scores.clone().float()
        masked_scores[recent_mask] = float("inf")

        if torch.all(recent_mask):
            raise ValueError(
                "All cache positions are in the recent window; "
                "no token can be evicted.  Ensure recent_window < max_size."
            )

        return int(masked_scores.argmin().item())


# ---------------------------------------------------------------------------
# H2OCache
# ---------------------------------------------------------------------------

class H2OCache:
    """Fixed-size KV cache that evicts light tokens via the H2O policy.

    During autoregressive generation, at each new token ``t``:
      1. Compute attention scores ``a_t ∈ R^{cache_len}`` (scores FROM ``t``
         TO each cached position).
      2. Accumulate importance: ``s_j += a_t[j]`` for all cached ``j``.
      3. Append the new key/value pair and its initial score (0).
      4. If ``|cache| > k``, evict the token with the lowest ``s_j`` that is
         not within the recent window of size ``w``.

    Total cache size stays at most ``k = max_size`` tokens after each step.

    Args:
        max_size (int):      ``k`` — total KV-pair budget (≥ 1).
        recent_window (int): ``w`` — number of most-recent tokens always kept
                             (0 ≤ w < max_size recommended; if w ≥ max_size
                             the cache is purely recency-based).
        eviction_policy:     Eviction policy instance.  Defaults to
                             :class:`H2OEvictionPolicy`.
    """

    def __init__(
        self,
        max_size: int = 256,
        recent_window: int = 32,
        eviction_policy: Optional[H2OEvictionPolicy] = None,
    ) -> None:
        if max_size < 1:
            raise ValueError(f"max_size must be ≥ 1, got {max_size}")
        if recent_window < 0:
            raise ValueError(f"recent_window must be ≥ 0, got {recent_window}")

        self._k: int = max_size            # k  in paper
        self._w: int = recent_window       # w  in paper
        self._policy: H2OEvictionPolicy = (
            eviction_policy if eviction_policy is not None
            else H2OEvictionPolicy()
        )

        # Stored KV tensors.  Shape: (B, n_heads, cache_len, head_dim).
        # None until the first call to update().
        self._keys:   Optional[Tensor] = None
        self._values: Optional[Tensor] = None

        # Cumulative importance scores s_j.
        # Shape: (B, n_heads, cache_len) — one per head per batch element.
        self._scores: Optional[Tensor] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        new_key:   Tensor,
        new_value: Tensor,
        attn_scores_to_new_token: Tensor,
    ) -> None:
        """Incorporate a new token into the cache.

        Implements Algorithm 1 from the H2O paper with the local-window
        variant:

        1. Update cumulative importance scores for existing cache entries
           using ``a_t`` (attention FROM the new token TO each cached token).
        2. Append the new key/value pair with initial score 0.
        3. If the cache exceeds ``k``, evict the lowest-scored non-recent token.

        Args:
            new_key:   ``(B, n_heads, 1, head_dim)`` — key for the new token.
            new_value: ``(B, n_heads, 1, head_dim)`` — value for the new token.
            attn_scores_to_new_token:
                ``(B, n_heads, cache_len)`` — attention scores ``a_t``
                FROM the new token TO each token currently in the cache.
                These are the pre-softmax *or* post-softmax scores; the paper
                uses post-softmax (attention probabilities).
                Pass an empty tensor (cache_len = 0) on the first call.

        Raises:
            ValueError: on shape mismatches.
        """
        # ---- shape validation ----------------------------------------
        if new_key.ndim != 4 or new_key.shape[2] != 1:
            raise ValueError(
                f"new_key must have shape (B, n_heads, 1, head_dim), "
                f"got {tuple(new_key.shape)}"
            )
        if new_value.shape != new_key.shape:
            raise ValueError(
                f"new_value shape {tuple(new_value.shape)} must match "
                f"new_key shape {tuple(new_key.shape)}"
            )

        B, H, _, D = new_key.shape

        if self._keys is None:
            # First token — initialise empty cache then append below.
            self._keys   = new_key.clone()
            self._values = new_value.clone()
            # score for this first token starts at 0
            self._scores = torch.zeros(B, H, 1, device=new_key.device, dtype=torch.float32)
            return

        cache_len = self._keys.shape[2]

        # ---- validate attn_scores shape --------------------------------
        if attn_scores_to_new_token.shape != (B, H, cache_len):
            raise ValueError(
                f"attn_scores_to_new_token must have shape "
                f"({B}, {H}, {cache_len}), "
                f"got {tuple(attn_scores_to_new_token.shape)}"
            )

        # ---- Step 2: accumulate importance scores (a_t[j] → s_j) ------
        # a_t shape: (B, n_heads, cache_len)
        a_t: Tensor = attn_scores_to_new_token.float()
        self._scores = self._scores + a_t  # broadcast over (B, H, cache_len)

        # ---- Step 3: append new token ----------------------------------
        self._keys   = torch.cat([self._keys,   new_key],   dim=2)
        self._values = torch.cat([self._values, new_value], dim=2)
        new_score    = torch.zeros(B, H, 1, device=new_key.device, dtype=torch.float32)
        self._scores = torch.cat([self._scores, new_score], dim=2)

        # ---- Step 4: evict if over budget ------------------------------
        new_cache_len = self._keys.shape[2]
        if new_cache_len > self._k:
            self._evict_one()

    def get_kv(self) -> Tuple[Tensor, Tensor]:
        """Return the current cached keys and values.

        Returns:
            Tuple ``(keys, values)``, each of shape
            ``(B, n_heads, current_size, head_dim)``.

        Raises:
            RuntimeError: if the cache is empty (no calls to ``update`` yet).
        """
        if self._keys is None:
            raise RuntimeError(
                "Cache is empty.  Call update() at least once before get_kv()."
            )
        return self._keys, self._values

    @property
    def size(self) -> int:
        """Current number of tokens stored in the cache."""
        if self._keys is None:
            return 0
        return int(self._keys.shape[2])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_one(self) -> None:
        """Evict the single lowest-scored non-recent token from the cache.

        The recent window of size ``w`` protects the *last* ``w`` positions
        in the cache (i.e. the most recently added tokens).
        Heavy hitters occupy the *first* ``k - w`` positions by score.
        """
        cache_len = self._keys.shape[2]  # currently k+1

        # Build recent_mask: protect the last w positions.
        recent_mask = torch.zeros(cache_len, dtype=torch.bool)
        if self._w > 0:
            recent_mask[-self._w:] = True

        # When w >= k, every non-new position is protected; just drop index 0.
        # (This handles the degenerate case without raising in the policy.)
        if self._w >= self._k:
            evict_idx = 0
        else:
            # Average scores across batch and heads for a single eviction
            # decision (B, H, cache_len) → (cache_len,)
            avg_scores: Tensor = self._scores.mean(dim=(0, 1))
            evict_idx = self._policy.select_evict(avg_scores, recent_mask)

        # Remove the evicted position along the sequence dimension (dim=2).
        keep = [i for i in range(cache_len) if i != evict_idx]
        keep_t = torch.tensor(keep, device=self._keys.device)

        self._keys   = self._keys.index_select(2, keep_t)
        self._values = self._values.index_select(2, keep_t)
        self._scores = self._scores.index_select(2, keep_t)
