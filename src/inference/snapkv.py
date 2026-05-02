"""
SnapKV: LLM Knows What You are Looking for Before Generation.

Implements the SnapKV algorithm (Li et al., 2024, arXiv:2404.14469).
SnapKV reduces KV-cache size by identifying which prompt KV pairs are
important for generation, based on attention patterns in a trailing
observation window of w tokens.

Variable notation follows the paper:
  w           — observation window size (last w tokens of the prompt)
  k           — max KV pairs to retain (max_capacity)
  A           — attention weight matrix, shape (B, n_heads, T_obs, T_seq)
  imp_j       — importance score for prompt position j
  K_snap      — set of selected (high-importance) indices
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# SnapKV Policy
# ---------------------------------------------------------------------------


class SnapKVPolicy:
    """Selects which KV-cache positions to retain using the SnapKV algorithm.

    Args:
        window_size:  Number of recent tokens forming the observation window
                      (w in the paper). Default 32.
        max_capacity: Maximum number of KV pairs to retain, including the
                      observation window tokens (k in the paper). Default 256.
        pool_method:  How to aggregate attention across heads before scoring.
                      'mean' (default) or 'max'.
    """

    def __init__(
        self,
        window_size: int = 32,
        max_capacity: int = 256,
        pool_method: Literal["mean", "max"] = "mean",
    ) -> None:
        if window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {window_size}")
        if max_capacity <= 0:
            raise ValueError(f"max_capacity must be > 0, got {max_capacity}")
        if pool_method not in ("mean", "max"):
            raise ValueError(f"pool_method must be 'mean' or 'max', got {pool_method!r}")

        self.window_size = window_size  # w
        self.max_capacity = max_capacity  # k
        self.pool_method = pool_method

    # ------------------------------------------------------------------
    # Core selection
    # ------------------------------------------------------------------

    def select_indices(
        self,
        attention_weights: Tensor,
        seq_len: int,
    ) -> Tensor:
        """Return sorted indices of KV positions to keep.

        The returned set is:  sorted(K_snap  ∪  obs_window_indices),
        with total length ≤ max_capacity.

        Args:
            attention_weights: Tensor of shape (B, n_heads, T_obs, T_seq).
                               Contains attention from the observation window
                               (last w rows) over all T_seq prompt positions.
                               Batch dimension B must be 1 here — importance
                               scores are derived from the first (and only)
                               sample; use compress() for batched inputs.
            seq_len:           Total sequence length T_seq.

        Returns:
            LongTensor of shape (k_kept,) with sorted indices,
            where k_kept ≤ max_capacity.
        """
        # attention_weights: (B, n_heads, T_obs, T_seq)
        if attention_weights.dim() != 4:
            raise ValueError(
                f"attention_weights must be 4-D (B, n_heads, T_obs, T_seq), "
                f"got shape {tuple(attention_weights.shape)}"
            )

        _B, _H, T_obs, T_seq = attention_weights.shape

        # Clamp observation window to actual sequence length
        w = min(self.window_size, seq_len)  # w in paper

        # Observation-window indices (always retained)
        obs_start = seq_len - w
        obs_indices = torch.arange(obs_start, seq_len, device=attention_weights.device)

        # If the full sequence fits within max_capacity, keep everything
        if seq_len <= self.max_capacity:
            return torch.arange(seq_len, device=attention_weights.device)

        # Budget for non-window positions after reserving obs window slots
        # k_budget = number of prompt-only positions to select via importance
        k_budget = self.max_capacity - w
        if k_budget <= 0:
            # window alone already fills the budget — return last max_capacity
            start = max(seq_len - self.max_capacity, 0)
            return torch.arange(start, seq_len, device=attention_weights.device)

        # ------------------------------------------------------------------
        # Compute per-position importance  imp_j  (Section 3, eq. 1)
        #
        # A has shape (B, n_heads, T_obs, T_seq).
        # We use the first sample in the batch for index selection.
        # imp_j = pool_over_heads( sum_{i in obs} A[i, j] )
        # ------------------------------------------------------------------
        A = attention_weights[0]  # (n_heads, T_obs, T_seq)

        # Sum over observation window dimension  → (n_heads, T_seq)
        imp_per_head = A.sum(dim=1)  # sum over T_obs

        # Pool across heads  → (T_seq,)
        if self.pool_method == "mean":
            imp = imp_per_head.mean(dim=0)  # imp_j
        else:  # 'max'
            imp = imp_per_head.max(dim=0).values

        # Zero out the observation-window region so it doesn't compete with
        # prompt positions (we always keep it anyway)
        imp_prompt = imp.clone()
        imp_prompt[obs_start:] = -float("inf")

        # Select top-k_budget prompt positions  → K_snap
        k_select = min(k_budget, obs_start)  # can't select more than exist
        if k_select <= 0:
            selected_indices = torch.empty(0, dtype=torch.long, device=attention_weights.device)
        else:
            _, topk_idx = torch.topk(imp_prompt[:obs_start], k=k_select, largest=True, sorted=False)
            selected_indices = topk_idx

        # Union: K_snap ∪ obs_window, then sort
        all_indices = torch.cat([selected_indices, obs_indices])
        kept, _ = all_indices.sort()
        return kept


# ---------------------------------------------------------------------------
# SnapKV Cache Compressor
# ---------------------------------------------------------------------------


class SnapKVCache:
    """Compresses keys and values using the SnapKV selection policy.

    Args:
        window_size:  Observation window size w. Default 32.
        max_capacity: Maximum KV pairs to keep k. Default 256.
        pool_method:  'mean' or 'max'. Default 'mean'.
    """

    def __init__(
        self,
        window_size: int = 32,
        max_capacity: int = 256,
        pool_method: Literal["mean", "max"] = "mean",
    ) -> None:
        self.policy = SnapKVPolicy(
            window_size=window_size,
            max_capacity=max_capacity,
            pool_method=pool_method,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress(
        self,
        keys: Tensor,
        values: Tensor,
        attention_weights: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply SnapKV compression to keys and values.

        Processes each item in the batch independently using the same
        index-selection logic, then pads to a uniform kept length.

        Args:
            keys:              (B, n_heads, T, head_dim)
            values:            (B, n_heads, T, head_dim)
            attention_weights: (B, n_heads, T_obs, T)  — attention from the
                               observation window over all T positions.

        Returns:
            Tuple (compressed_keys, compressed_values), each with shape
            (B, n_heads, k_kept, head_dim) where k_kept ≤ max_capacity.
        """
        if keys.shape != values.shape:
            raise ValueError(
                f"keys and values must have the same shape, "
                f"got {tuple(keys.shape)} vs {tuple(values.shape)}"
            )
        if keys.dim() != 4:
            raise ValueError(f"keys must be 4-D (B, n_heads, T, head_dim), got {keys.dim()}-D")
        if attention_weights.dim() != 4:
            raise ValueError(
                f"attention_weights must be 4-D (B, n_heads, T_obs, T), "
                f"got {attention_weights.dim()}-D"
            )

        B, n_heads, T, head_dim = keys.shape

        # Gather per-sample compressed tensors
        compressed_k_list = []
        compressed_v_list = []

        for b in range(B):
            # select_indices expects a (1, n_heads, T_obs, T) tensor
            attn_b = attention_weights[b : b + 1]  # (1, n_heads, T_obs, T)
            idx = self.policy.select_indices(attn_b, seq_len=T)  # (k_kept,)

            # Gather: (n_heads, k_kept, head_dim)
            idx_exp = idx.unsqueeze(0).unsqueeze(-1).expand(n_heads, -1, head_dim)
            ck = keys[b].gather(1, idx_exp)  # (n_heads, k_kept, head_dim)
            cv = values[b].gather(1, idx_exp)  # (n_heads, k_kept, head_dim)

            compressed_k_list.append(ck)
            compressed_v_list.append(cv)

        # Stack along batch dimension — all samples have the same k_kept because
        # select_indices returns at most max_capacity indices for all T.
        # (If batch items have different k_kept due to different T, we take min.)
        min_kept = min(ck.shape[1] for ck in compressed_k_list)
        compressed_keys = torch.stack([ck[:, :min_kept, :] for ck in compressed_k_list], dim=0)
        compressed_values = torch.stack([cv[:, :min_kept, :] for cv in compressed_v_list], dim=0)

        return compressed_keys, compressed_values
