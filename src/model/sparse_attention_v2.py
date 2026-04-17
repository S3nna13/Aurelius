"""
Sparse Attention Patterns for Aurelius LLM.

Implements Sliding Window (Longformer-style), Strided/Dilated, BigBird-style
(random + window + global), and Learned Sparse (top-k masking) attention.
All patterns maintain the same interface as standard attention.

Pure PyTorch only — no external attention libraries.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AttentionMaskBuilder:
    """Build sparse attention masks as boolean (T, T) tensors.

    All masks are causal: position i can only attend to positions <= i.
    True means "allowed to attend", False means "blocked".
    """

    def __init__(self, seq_len: int, device: str = "cpu") -> None:
        self.seq_len = seq_len
        self.device = device

    def causal_mask(self) -> Tensor:
        """Standard lower-triangular causal mask.

        Returns:
            (T, T) bool tensor — True where attention is allowed.
        """
        T = self.seq_len
        # tril gives lower-triangular, diagonal included → causal
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=self.device))
        return mask

    def sliding_window(self, window_size: int) -> Tensor:
        """Causal sliding-window mask: each query attends to at most
        the previous `window_size` tokens (inclusive of itself).

        Position i attends to j iff  i - window_size < j <= i
        (equivalently: j >= i - window_size + 1  AND  j <= i).

        Returns:
            (T, T) bool tensor.
        """
        T = self.seq_len
        rows = torch.arange(T, device=self.device).unsqueeze(1)   # (T, 1)
        cols = torch.arange(T, device=self.device).unsqueeze(0)   # (1, T)
        # causal: col <= row
        causal = cols <= rows
        # window: col >= row - window_size + 1
        in_window = cols >= (rows - window_size + 1)
        return causal & in_window

    def strided(self, stride: int, window_size: int = 1) -> Tensor:
        """Causal strided + local-window mask.

        Each query i attends to:
          - Local window: j in [i - window_size + 1, i]
          - Strided positions: every `stride`-th position j <= i

        Returns:
            (T, T) bool tensor.
        """
        T = self.seq_len
        rows = torch.arange(T, device=self.device).unsqueeze(1)   # (T, 1)
        cols = torch.arange(T, device=self.device).unsqueeze(0)   # (1, T)

        causal = cols <= rows
        local = cols >= (rows - window_size + 1)
        strided_pos = (cols % stride) == 0

        return causal & (local | strided_pos)

    def global_tokens(self, n_global: int, window_size: int = 2) -> Tensor:
        """BigBird-style global-tokens mask.

        Rules (all causal):
          1. First n_global tokens attend to ALL positions they can (full causal).
          2. ALL tokens attend to the first n_global tokens.
          3. ALL tokens also have a local sliding window of size window_size.

        Returns:
            (T, T) bool tensor.
        """
        T = self.seq_len
        rows = torch.arange(T, device=self.device).unsqueeze(1)   # (T, 1)
        cols = torch.arange(T, device=self.device).unsqueeze(0)   # (1, T)

        causal = cols <= rows

        # Rule 1: global rows attend everywhere (causal)
        is_global_row = rows < n_global          # (T, 1) broadcasts
        global_row_mask = is_global_row & causal

        # Rule 2: all rows can attend to global columns (causal)
        is_global_col = cols < n_global          # (1, T) broadcasts
        global_col_mask = is_global_col & causal

        # Rule 3: local window
        local = (cols >= (rows - window_size + 1)) & causal

        return global_row_mask | global_col_mask | local


# ---------------------------------------------------------------------------
# Helper: project to Q, K, V and split into heads
# ---------------------------------------------------------------------------

def _split_heads(x: Tensor, n_heads: int) -> Tensor:
    """(B, T, D) → (B, n_heads, T, head_dim)."""
    B, T, D = x.shape
    head_dim = D // n_heads
    x = x.view(B, T, n_heads, head_dim)
    return x.transpose(1, 2)  # (B, H, T, d_h)


def _merge_heads(x: Tensor) -> Tensor:
    """(B, n_heads, T, head_dim) → (B, T, D)."""
    B, H, T, d_h = x.shape
    x = x.transpose(1, 2).contiguous()
    return x.view(B, T, H * d_h)


def _masked_softmax(scores: Tensor, mask: Tensor) -> Tensor:
    """Apply boolean mask (True=keep) then softmax over last dim."""
    scores = scores.masked_fill(~mask, float("-inf"))
    return F.softmax(scores, dim=-1)


# ---------------------------------------------------------------------------
# SlidingWindowAttention
# ---------------------------------------------------------------------------

class SlidingWindowAttention(nn.Module):
    """Causal sliding-window attention (Longformer-style).

    Each query attends only to the previous `window_size` tokens.
    """

    def __init__(self, d_model: int, n_heads: int, window_size: int = 4) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            (B, T, D)
        """
        B, T, D = x.shape
        device = x.device

        qkv = self.qkv(x)                          # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)             # each (B, T, D)

        q = _split_heads(q, self.n_heads)           # (B, H, T, d_h)
        k = _split_heads(k, self.n_heads)
        v = _split_heads(v, self.n_heads)

        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, T, T)

        builder = AttentionMaskBuilder(T, device=str(device))
        mask = builder.sliding_window(self.window_size)         # (T, T)
        mask = mask.unsqueeze(0).unsqueeze(0)                   # (1, 1, T, T)

        attn = _masked_softmax(scores, mask)                    # (B, H, T, T)
        out = torch.matmul(attn, v)                             # (B, H, T, d_h)
        out = _merge_heads(out)                                 # (B, T, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# StridedAttention
# ---------------------------------------------------------------------------

class StridedAttention(nn.Module):
    """Dilated strided attention for long-range dependencies.

    Each query attends to strided positions + a local window.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        stride: int = 2,
        local_window: int = 2,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.stride = stride
        self.local_window = local_window
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            (B, T, D)
        """
        B, T, D = x.shape
        device = x.device

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = _split_heads(q, self.n_heads)
        k = _split_heads(k, self.n_heads)
        v = _split_heads(v, self.n_heads)

        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        builder = AttentionMaskBuilder(T, device=str(device))
        mask = builder.strided(self.stride, self.local_window)
        mask = mask.unsqueeze(0).unsqueeze(0)

        attn = _masked_softmax(scores, mask)
        out = torch.matmul(attn, v)
        out = _merge_heads(out)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# BigBirdAttention
# ---------------------------------------------------------------------------

class BigBirdAttention(nn.Module):
    """Combined random + window + global attention (BigBird-style).

    Pattern: global_tokens mask | random causal positions | local window.
    The random positions are seeded for reproducibility.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int = 2,
        n_global: int = 1,
        n_random: int = 2,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.n_global = n_global
        self.n_random = n_random
        self.seed = seed
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _build_mask(self, T: int, device: torch.device) -> Tensor:
        """Build the BigBird mask for sequence length T."""
        builder = AttentionMaskBuilder(T, device=str(device))
        # Base: global tokens + local window
        mask = builder.global_tokens(self.n_global, self.window_size)

        # Add n_random random causal positions per query (seeded)
        rng = torch.Generator(device=device)
        rng.manual_seed(self.seed)

        rows = torch.arange(T, device=device)
        cols = torch.arange(T, device=device)

        for i in range(T):
            # Candidate positions: causal (j <= i), not already attended to
            causal_pool = cols[cols <= i]
            n_cands = causal_pool.numel()
            if n_cands == 0:
                continue
            n_pick = min(self.n_random, n_cands)
            # Random sample without replacement from causal pool
            perm = torch.randperm(n_cands, generator=rng, device=device)[:n_pick]
            chosen = causal_pool[perm]
            mask[i, chosen] = True

        return mask

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            (B, T, D)
        """
        B, T, D = x.shape
        device = x.device

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = _split_heads(q, self.n_heads)
        k = _split_heads(k, self.n_heads)
        v = _split_heads(v, self.n_heads)

        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        mask = self._build_mask(T, device)
        mask = mask.unsqueeze(0).unsqueeze(0)

        attn = _masked_softmax(scores, mask)
        out = torch.matmul(attn, v)
        out = _merge_heads(out)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# LearnedSparseAttention
# ---------------------------------------------------------------------------

class LearnedSparseAttention(nn.Module):
    """Top-k sparse attention: keep only the k highest-scoring keys per query.

    The masking is differentiable through the softmax weights (the top-k
    selection itself is non-differentiable, but gradients flow through the
    kept attention weights).

    Causal constraint is always enforced: future tokens are never attended to,
    even if they would be in the top-k.
    """

    def __init__(self, d_model: int, n_heads: int, k: int = 4) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.k = k
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            (B, T, D)
        """
        B, T, D = x.shape
        device = x.device

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = _split_heads(q, self.n_heads)   # (B, H, T, d_h)
        k = _split_heads(k, self.n_heads)
        v = _split_heads(v, self.n_heads)

        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, T, T)

        # 1. Apply causal mask first (set future to -inf)
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
        scores = scores.masked_fill(~causal, float("-inf"))

        # 2. Top-k masking: for each query keep only top-k valid positions.
        #    Use kthvalue to find the threshold (differentiable path through
        #    the kept scores; the mask boundary is a hard threshold).
        #    We clamp k to the number of valid (causal) positions per row.
        #    "valid" for row i = i+1 positions (0..i).
        #    We build the mask row by row via kthvalue on the score tensor.

        # Number of valid positions per query (row i has i+1 causal slots)
        valid_counts = torch.arange(1, T + 1, device=device)          # (T,)
        k_eff = torch.clamp(valid_counts, max=self.k)                  # (T,)

        # Expand scores to (B*H, T, T) for easier row-wise operation
        BH = B * self.n_heads
        scores_2d = scores.view(BH, T, T)

        topk_mask = torch.zeros(BH, T, T, dtype=torch.bool, device=device)
        for i in range(T):
            n_valid = i + 1          # number of causal positions
            ki = min(self.k, n_valid)

            # scores for causal positions of row i: (BH, n_valid)
            row_scores = scores_2d[:, i, :n_valid]

            # kthvalue returns the ki-th smallest → we want top-ki largest
            # threshold = (n_valid - ki + 1)-th smallest = ki-th largest
            rank = n_valid - ki + 1  # 1-indexed rank of the threshold
            # kthvalue needs k >= 1
            threshold, _ = torch.kthvalue(row_scores, rank, dim=-1, keepdim=True)
            # Keep positions with score >= threshold
            topk_mask[:, i, :n_valid] = row_scores >= threshold

        topk_mask = topk_mask.view(B, self.n_heads, T, T)

        # 3. Apply top-k mask on top of already-causally-masked scores
        scores = scores.masked_fill(~topk_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        # Handle all-inf rows (shouldn't happen for row 0 with k>=1, but
        # safeguard against NaN from softmax of all -inf)
        attn = torch.nan_to_num(attn, nan=0.0)

        out = torch.matmul(attn, v)
        out = _merge_heads(out)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# SparseAttentionBlock
# ---------------------------------------------------------------------------

class SparseAttentionBlock(nn.Module):
    """Transformer block with configurable sparse attention.

    Architecture:
        Pre-LayerNorm → sparse attention → residual
        Pre-LayerNorm → FFN (2-layer, 4x hidden, GELU) → residual

    Supported patterns: "sliding", "strided", "bigbird", "learned".
    """

    _PATTERNS = ("sliding", "strided", "bigbird", "learned")

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        pattern: str = "sliding",
        **pattern_kwargs,
    ) -> None:
        super().__init__()
        if pattern not in self._PATTERNS:
            raise ValueError(
                f"Unknown pattern '{pattern}'. Choose from {self._PATTERNS}."
            )
        self.d_model = d_model
        self.n_heads = n_heads
        self.pattern = pattern

        # Build attention module
        if pattern == "sliding":
            self.attn = SlidingWindowAttention(d_model, n_heads, **pattern_kwargs)
        elif pattern == "strided":
            self.attn = StridedAttention(d_model, n_heads, **pattern_kwargs)
        elif pattern == "bigbird":
            self.attn = BigBirdAttention(d_model, n_heads, **pattern_kwargs)
        elif pattern == "learned":
            self.attn = LearnedSparseAttention(d_model, n_heads, **pattern_kwargs)

        # LayerNorms (pre-norm)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN: 2-layer, 4x hidden, GELU
        hidden = 4 * d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            (B, T, D)
        """
        # Attention sub-layer (pre-norm + residual)
        x = x + self.attn(self.norm1(x))
        # FFN sub-layer (pre-norm + residual)
        x = x + self.ffn(self.norm2(x))
        return x
