"""Contextual Position Encoding: ALiBi, KERPLE, T5 Relative, and Learned Absolute."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class PositionEncodingConfig:
    """Configuration for position encoding modules."""

    n_heads: int
    d_model: int
    max_seq_len: int = 2048
    alibi_slopes: Optional[List[float]] = None


class ALiBiPositionBias(nn.Module):
    """Attention with Linear Biases (ALiBi) position bias.

    Reference: Press et al. 2022 "Train Short, Test Long: Attention with Linear Biases
    Enables Input Length Extrapolation"
    """

    def __init__(self, n_heads: int, slopes: Optional[List[float]] = None) -> None:
        super().__init__()
        self.n_heads = n_heads
        computed = self.get_slopes(n_heads)
        if slopes is not None:
            slope_tensor = torch.tensor(slopes, dtype=torch.float32)
        else:
            slope_tensor = computed
        # Register as buffer (not a parameter — fixed)
        self.register_buffer("slopes", slope_tensor)

    def get_slopes(self, n_heads: int) -> Tensor:
        """Compute ALiBi slopes for each attention head.

        For n_heads=8: [1/2, 1/4, 1/8, ..., 1/256]
        General: slope_h = 2^(-8*h/n_heads) for h=1..n_heads, sorted descending.
        """
        slopes = [2 ** (-8.0 * h / n_heads) for h in range(1, n_heads + 1)]
        return torch.tensor(slopes, dtype=torch.float32)

    def forward(self, seq_len: int) -> Tensor:
        """Compute causal ALiBi bias matrix.

        Returns:
            Tensor of shape (n_heads, seq_len, seq_len).
            bias[h, i, j] = -slope_h * (i - j) for i >= j (causal)
                           = -inf                            for i < j  (masked future)
        """
        device = self.slopes.device
        # Positions: rows = query, cols = key
        i_idx = torch.arange(seq_len, device=device).unsqueeze(1)  # (T, 1)
        j_idx = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, T)

        # Relative distance: i - j; negative when attending to future
        distance = i_idx - j_idx  # (T, T)

        # Causal mask: future positions get -inf
        causal_mask = distance < 0  # (T, T)

        # Bias for valid (past/present) positions: -slope * (i - j) <= 0
        # slopes: (n_heads,) → (n_heads, 1, 1)
        slopes = self.slopes.view(self.n_heads, 1, 1)
        bias = -slopes * distance.float().unsqueeze(0)  # (n_heads, T, T)

        # Apply causal masking
        bias = bias.masked_fill(causal_mask.unsqueeze(0), float("-inf"))

        return bias


class T5RelativePositionBias(nn.Module):
    """T5-style relative position bias with learned bucket embeddings.

    Reference: Raffel et al. 2020 "Exploring the Limits of Transfer Learning with a
    Unified Text-to-Text Transformer"
    """

    def __init__(
        self,
        n_heads: int,
        n_buckets: int = 32,
        max_distance: int = 128,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_buckets = n_buckets
        self.max_distance = max_distance
        # Learnable embedding table: (n_buckets, n_heads)
        self.relative_attention_bias = nn.Embedding(n_buckets, n_heads)

    def _relative_position_bucket(self, relative_position: Tensor) -> Tensor:
        """Map relative positions to bucket indices in [0, n_buckets).

        First half of buckets: exact small distances (0 to n_buckets//2 - 1).
        Second half: log-spaced large distances.
        Handles both positive (future) and negative (past) directions.
        """
        n_buckets = self.n_buckets
        max_distance = self.max_distance

        # We'll use the absolute value and separate sign
        # Half buckets for each direction
        half_buckets = n_buckets // 2

        # Offset positive positions by half_buckets
        ret = torch.zeros_like(relative_position)
        is_positive = relative_position > 0
        abs_pos = torch.abs(relative_position)

        # Small distances get exact bucket (0 to half_buckets//2 - 1)
        exact_range = half_buckets // 2
        small = abs_pos < exact_range

        # Large distances: log-spaced into remaining buckets
        # Map [exact_range, max_distance] → [exact_range, half_buckets - 1]
        log_bucket = (
            exact_range
            + (
                torch.log(abs_pos.float().clamp(min=1) / exact_range)
                / math.log(max_distance / exact_range)
                * (half_buckets - exact_range)
            ).long()
        )
        log_bucket = log_bucket.clamp(exact_range, half_buckets - 1)

        ret = torch.where(small, abs_pos, log_bucket)
        ret = torch.where(is_positive, ret + half_buckets, ret)
        return ret.clamp(0, n_buckets - 1)

    def forward(self, seq_len: int) -> Tensor:
        """Compute T5 relative position bias.

        Returns:
            Tensor of shape (n_heads, seq_len, seq_len).
        """
        device = self.relative_attention_bias.weight.device
        q_pos = torch.arange(seq_len, dtype=torch.long, device=device)
        k_pos = torch.arange(seq_len, dtype=torch.long, device=device)

        # Relative positions: (T, T) where [i, j] = i - j
        relative_position = q_pos.unsqueeze(1) - k_pos.unsqueeze(0)  # (T, T)

        buckets = self._relative_position_bucket(relative_position)  # (T, T)

        # Lookup: (T, T, n_heads) → permute to (n_heads, T, T)
        bias = self.relative_attention_bias(buckets)  # (T, T, n_heads)
        bias = bias.permute(2, 0, 1).contiguous()  # (n_heads, T, T)
        return bias


class LearnedAbsolutePositionEncoding(nn.Module):
    """Learned absolute position encoding with bilinear interpolation support."""

    def __init__(self, max_seq_len: int, d_model: int) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.weight = nn.Parameter(torch.empty(max_seq_len, d_model))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, seq_len: int) -> Tensor:
        """Return position encodings for the first seq_len positions.

        Returns:
            Tensor of shape (1, seq_len, d_model).
        """
        return self.weight[:seq_len].unsqueeze(0)

    def interpolate(self, new_max_len: int) -> None:
        """Bilinear-interpolate weight to a new max sequence length in-place."""
        old_weight = self.weight.data  # (old_len, d_model)

        # Reshape for interpolate: (1, 1, old_len, d_model) → treat as 2D image
        # Use linear interpolation along the sequence dimension
        old_len = old_weight.shape[0]
        # (1, d_model, old_len) for 1D interpolation
        x = old_weight.T.unsqueeze(0)  # (1, d_model, old_len)
        x_interp = F.interpolate(x, size=new_max_len, mode="linear", align_corners=False)
        new_weight = x_interp.squeeze(0).T  # (new_max_len, d_model)

        # Replace parameter data
        self.weight = nn.Parameter(new_weight)
        self.max_seq_len = new_max_len


class KERPLEBias(nn.Module):
    """Kernel-based Relative Position Encoding (KERPLE).

    Uses a Gaussian-kernel-inspired bias: bias[h, i, j] = -r_h * (i-j)^2
    where r_h > 0 is a learnable per-head parameter.

    Reference: Chen et al. 2022 "KERPLE: Kernelized Relative Positional Embedding
    for Length Extrapolation"
    """

    def __init__(self, n_heads: int, init_r: float = 1.0) -> None:
        super().__init__()
        self.n_heads = n_heads
        # Learnable r per head; use softplus to ensure r > 0
        self.r_log = nn.Parameter(
            torch.full((n_heads,), math.log(math.expm1(init_r)))
        )

    @property
    def r(self) -> Tensor:
        """Return positive r values via softplus."""
        return F.softplus(self.r_log)

    def forward(self, seq_len: int) -> Tensor:
        """Compute KERPLE bias matrix.

        Returns:
            Tensor of shape (n_heads, seq_len, seq_len).
            bias[h, i, j] = -r_h * (i - j)^2
        """
        device = self.r_log.device
        i_idx = torch.arange(seq_len, device=device, dtype=torch.float32)
        j_idx = torch.arange(seq_len, device=device, dtype=torch.float32)

        # Squared distance: (T, T)
        dist_sq = (i_idx.unsqueeze(1) - j_idx.unsqueeze(0)) ** 2  # (T, T)

        # r: (n_heads,) → (n_heads, 1, 1)
        r = self.r.view(self.n_heads, 1, 1)

        # bias: (n_heads, T, T)
        bias = -r * dist_sq.unsqueeze(0)
        return bias
