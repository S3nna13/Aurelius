"""
PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling.

Implements the PyramidKV algorithm from:
    Cai et al., "PyramidKV: Dynamic KV Cache Compression based on Pyramidal
    Information Funneling", arXiv:2406.02069 (2024).

Key idea (Section 3):
    Lower transformer layers exhibit higher attention-pattern diversity (broad
    information gathering) while upper layers concentrate on fewer key tokens
    (information funneling).  PyramidKV exploits this by assigning a *larger*
    KV-cache budget b_l to lower layers and a *smaller* budget to upper layers,
    following a pyramid (monotonically decreasing) schedule.

Budget allocation schedules (Section 3.2):
  Pyramid:
    b_l = round( C * (L - l + 1) / sum_{i=1}^{L}(L - i + 1) )   l = 1..L

  Linear:
    b_l = max(b_min, round(b_max - (b_max - b_min) * (l - 1) / (L - 1)))

Within-layer selection:
    Attention-score-based top-k selection (same approach as SnapKV) using
    the observation window of the last `window_size` tokens.

Paper variable notation:
    L           — total number of transformer layers
    l           — 1-indexed layer index (l = 1 is the bottom layer)
    C           — total KV-cache budget (tokens) across all layers
    b_l         — per-layer budget (KV slots allocated to layer l)
    b_min       — minimum budget per layer (floor guard)
    b_max       — maximum budget for the first layer (linear schedule)
    w           — observation window size (last w tokens)
    A           — attention weight matrix, shape (B, n_heads, T_obs, T)
    imp_j       — importance score for prompt position j
"""

from __future__ import annotations

from typing import List, Literal, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# PyramidKVScheduler
# ---------------------------------------------------------------------------

class PyramidKVScheduler:
    """Computes per-layer KV-cache budgets following the PyramidKV schedule.

    Args:
        n_layers:      Total number of transformer layers  (L in the paper).
        total_budget:  Total KV-cache budget C shared across all layers.
        min_budget:    Minimum KV slots per layer  (b_min).  Default 4.
        schedule:      ``'pyramid'`` (default) or ``'linear'``.

    Raises:
        ValueError: on invalid arguments.
    """

    def __init__(
        self,
        n_layers: int,
        total_budget: int,
        min_budget: int = 4,
        schedule: Literal["pyramid", "linear"] = "pyramid",
    ) -> None:
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        if total_budget < 1:
            raise ValueError(f"total_budget must be >= 1, got {total_budget}")
        if min_budget < 1:
            raise ValueError(f"min_budget must be >= 1, got {min_budget}")
        if schedule not in ("pyramid", "linear"):
            raise ValueError(f"schedule must be 'pyramid' or 'linear', got {schedule!r}")

        self.n_layers = n_layers       # L
        self.total_budget = total_budget  # C
        self.min_budget = min_budget   # b_min
        self.schedule = schedule

        self._budgets: List[int] = self._compute_budgets()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def budget_for_layer(self, layer_idx: int) -> int:
        """Return the KV-cache budget allocated to *layer_idx*.

        Args:
            layer_idx: 0-indexed layer number (0 = first/bottom layer).

        Returns:
            Integer number of KV slots for this layer.

        Raises:
            IndexError: if ``layer_idx`` is out of range.
        """
        if not (0 <= layer_idx < self.n_layers):
            raise IndexError(
                f"layer_idx {layer_idx} out of range [0, {self.n_layers - 1}]"
            )
        return self._budgets[layer_idx]

    def all_budgets(self) -> List[int]:
        """Return list of budgets for all layers (index 0 = first/bottom layer).

        Returns:
            List of length ``n_layers`` with per-layer KV-slot counts.
        """
        return list(self._budgets)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_budgets(self) -> List[int]:
        """Compute and return all per-layer budgets."""
        L = self.n_layers
        C = self.total_budget
        b_min = self.min_budget

        # If the total budget is too small to give every layer min_budget,
        # assign min_budget to all layers gracefully (loudly cap b_min to C//L).
        if C < L * b_min:
            # Every layer gets at least b_min; total may exceed C but the
            # contract specifies graceful handling.
            return [b_min] * L

        if self.schedule == "pyramid":
            budgets = self._pyramid_schedule(L, C, b_min)
        else:
            budgets = self._linear_schedule(L, C, b_min)

        return budgets

    @staticmethod
    def _pyramid_schedule(L: int, C: int, b_min: int) -> List[int]:
        """Pyramid schedule: b_l ∝ (L - l + 1).

        Equation from Section 3.2:
            b_l = round( C * (L - l + 1) / sum_{i=1}^{L}(L - i + 1) )

        After rounding, residual tokens are distributed to the layers
        with the largest fractional parts (standard rounding correction).
        """
        # denominator: sum_{i=1}^{L}(L - i + 1) = sum_{k=1}^{L} k = L*(L+1)/2
        denom = L * (L + 1) // 2  # integer, exact

        # Compute exact (floating-point) allocations for l = 1..L
        # Layer 1 (l=1) is 0-indexed as layer_idx=0.
        exact: List[float] = []
        for l in range(1, L + 1):                # l is 1-indexed
            w_l = L - l + 1                      # weight for layer l
            exact.append(C * w_l / denom)

        # Round-and-redistribute to preserve sum == C
        budgets = [max(b_min, round(x)) for x in exact]
        _redistribute(budgets, C, b_min)
        return budgets

    @staticmethod
    def _linear_schedule(L: int, C: int, b_min: int) -> List[int]:
        """Linear schedule: budgets decrease uniformly from b_max to b_min.

        b_l = max(b_min, round(b_max - (b_max - b_min) * (l - 1) / (L - 1)))

        b_max is derived from the constraint sum b_l ≈ C.
        For a perfect arithmetic sequence from b_max down to b_min:
            sum = L * (b_max + b_min) / 2  =>  b_max = 2*C/L - b_min
        """
        if L == 1:
            return [C]

        # Derive b_max from budget constraint
        b_max_f = 2.0 * C / L - b_min
        b_max = max(b_min, round(b_max_f))

        budgets: List[int] = []
        for l in range(1, L + 1):               # l is 1-indexed
            raw = b_max - (b_max - b_min) * (l - 1) / (L - 1)
            budgets.append(max(b_min, round(raw)))

        _redistribute(budgets, C, b_min)
        return budgets


def _redistribute(budgets: List[int], C: int, b_min: int) -> None:
    """In-place adjustment so sum(budgets) == C while keeping each >= b_min.

    Adds/removes tokens greedily from the layer with the largest/smallest
    budget (excluding floors).  This is a standard rounding fixup.
    """
    diff = C - sum(budgets)
    if diff == 0:
        return

    n = len(budgets)
    if diff > 0:
        # Need to add `diff` tokens: give them to the layer with the most budget
        # (top of pyramid) to preserve the shape.
        indices = sorted(range(n), key=lambda i: -budgets[i])
        for idx in indices:
            if diff == 0:
                break
            budgets[idx] += 1
            diff -= 1
    else:
        # Need to remove abs(diff) tokens: take from the largest-budget layers
        # but never go below b_min.
        indices = sorted(range(n), key=lambda i: -budgets[i])
        diff = -diff  # now positive
        for idx in indices:
            if diff == 0:
                break
            removable = budgets[idx] - b_min
            remove = min(removable, diff)
            budgets[idx] -= remove
            diff -= remove


# ---------------------------------------------------------------------------
# PyramidKVCache
# ---------------------------------------------------------------------------

class PyramidKVCache:
    """Compresses per-layer KV caches using the PyramidKV budget schedule.

    The within-layer selection uses attention-score-based top-k (SnapKV-style):
      1. Pool attention weights from the observation window over heads.
      2. Select the top-``b_l`` positions by pooled importance.
      3. Always retain the most recent ``window_size`` tokens.

    Args:
        n_layers:      Total number of transformer layers  (L).
        total_budget:  Total KV-cache budget C across all layers.
        min_budget:    Minimum KV slots per layer  (b_min).  Default 4.
        window_size:   Observation window size  (w).  Default 16.
        schedule:      ``'pyramid'`` (default) or ``'linear'``.
    """

    def __init__(
        self,
        n_layers: int,
        total_budget: int,
        min_budget: int = 4,
        window_size: int = 16,
        schedule: Literal["pyramid", "linear"] = "pyramid",
    ) -> None:
        if window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {window_size}")

        self.scheduler = PyramidKVScheduler(
            n_layers=n_layers,
            total_budget=total_budget,
            min_budget=min_budget,
            schedule=schedule,
        )
        self.window_size = window_size  # w

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress(
        self,
        layer_idx: int,
        keys: Tensor,
        values: Tensor,
        attn_weights: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Compress the KV cache for a single layer using the PyramidKV budget.

        Args:
            layer_idx:    0-indexed layer number.
            keys:         (B, n_heads, T, head_dim)
            values:       (B, n_heads, T, head_dim)
            attn_weights: (B, n_heads, T_obs, T)  — attention from the
                          observation window over all T positions.

        Returns:
            Tuple ``(compressed_keys, compressed_values)``, each with shape
            ``(B, n_heads, k, head_dim)`` where ``k ≤ b_l``.

        Raises:
            ValueError: on shape mismatches or invalid arguments.
        """
        # ---- shape validation ----------------------------------------
        if keys.dim() != 4:
            raise ValueError(
                f"keys must be 4-D (B, n_heads, T, head_dim), got {keys.dim()}-D"
            )
        if keys.shape != values.shape:
            raise ValueError(
                f"keys and values must have the same shape, "
                f"got {tuple(keys.shape)} vs {tuple(values.shape)}"
            )
        if attn_weights.dim() != 4:
            raise ValueError(
                f"attn_weights must be 4-D (B, n_heads, T_obs, T), "
                f"got {attn_weights.dim()}-D"
            )

        B, n_heads, T, head_dim = keys.shape

        # b_l: KV budget for this layer
        b_l: int = self.scheduler.budget_for_layer(layer_idx)

        # If the full sequence fits in budget, keep everything
        if T <= b_l:
            return keys.clone(), values.clone()

        # ---- Per-sample index selection  --------------------------------
        compressed_k_list: List[Tensor] = []
        compressed_v_list: List[Tensor] = []

        for b in range(B):
            attn_b = attn_weights[b : b + 1]  # (1, n_heads, T_obs, T)
            idx = self._select_indices(attn_b, seq_len=T, budget=b_l)  # (k_kept,)

            # Gather: (n_heads, k_kept, head_dim)
            idx_exp = idx.unsqueeze(0).unsqueeze(-1).expand(n_heads, -1, head_dim)
            ck = keys[b].gather(1, idx_exp)    # (n_heads, k_kept, head_dim)
            cv = values[b].gather(1, idx_exp)  # (n_heads, k_kept, head_dim)

            compressed_k_list.append(ck)
            compressed_v_list.append(cv)

        # Stack — all samples have the same k_kept (same T, same budget)
        min_kept = min(ck.shape[1] for ck in compressed_k_list)
        compressed_keys = torch.stack(
            [ck[:, :min_kept, :] for ck in compressed_k_list], dim=0
        )
        compressed_values = torch.stack(
            [cv[:, :min_kept, :] for cv in compressed_v_list], dim=0
        )

        return compressed_keys, compressed_values

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_indices(
        self,
        attn_weights: Tensor,
        seq_len: int,
        budget: int,
    ) -> Tensor:
        """Select up to ``budget`` KV indices for one sequence.

        Strategy (SnapKV-style):
          - Always retain the last ``window_size`` positions (recent window).
          - Fill remaining budget with the highest-importance non-window positions
            scored by summing attention weights over the observation window and
            averaging across heads.

        Args:
            attn_weights: (1, n_heads, T_obs, T)
            seq_len:      T — total sequence length.
            budget:       b_l — maximum positions to retain.

        Returns:
            Sorted LongTensor of shape ``(k_kept,)`` with ``k_kept ≤ budget``.
        """
        device = attn_weights.device

        # Clamp window to actual sequence
        w = min(self.window_size, seq_len)   # w in paper
        obs_start = seq_len - w

        # Observation/recent window indices — always kept
        obs_indices = torch.arange(obs_start, seq_len, device=device)

        # If sequence fits in budget, keep all
        if seq_len <= budget:
            return torch.arange(seq_len, device=device)

        # Slots available for non-window (historical) positions
        k_hist = budget - w
        if k_hist <= 0:
            # Window alone fills or exceeds the budget; keep latest `budget` tokens
            start = max(seq_len - budget, 0)
            return torch.arange(start, seq_len, device=device)

        # ------------------------------------------------------------------
        # Importance score: imp_j = mean_over_heads( sum_over_T_obs( A[i,j] ) )
        # A: (1, n_heads, T_obs, T)  →  (n_heads, T)
        # ------------------------------------------------------------------
        A = attn_weights[0]               # (n_heads, T_obs, T)
        imp_per_head = A.sum(dim=1)        # (n_heads, T) — sum over T_obs
        imp: Tensor = imp_per_head.mean(dim=0)  # (T,) — mean over heads  → imp_j

        # Zero-out the recent window so it doesn't compete
        imp_hist = imp.clone()
        imp_hist[obs_start:] = -float("inf")

        # Select top-k historical positions
        k_select = min(k_hist, obs_start)
        if k_select <= 0:
            return obs_indices.sort().values

        _, topk_idx = torch.topk(
            imp_hist[:obs_start], k=k_select, largest=True, sorted=False
        )

        # Union: historical K_snap ∪ recent window
        all_idx = torch.cat([topk_idx, obs_indices])
        kept, _ = all_idx.sort()
        return kept
