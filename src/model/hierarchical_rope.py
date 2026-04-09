"""Hierarchical Rotary Position Embeddings (HiRoPE).

Extends standard RoPE with multi-scale position encoding that captures
both local and global structure by averaging RoPE rotations across
multiple frequency scales.
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class HierarchicalRoPEConfig:
    head_dim: int = 32
    max_seq_len: int = 512
    n_scales: int = 3               # number of hierarchical levels
    scale_factors: list[float] = field(default_factory=lambda: [1.0, 4.0, 16.0])
    base: float = 10000.0
    rope_fraction: float = 1.0      # fraction of head_dim for positional encoding


def compute_rope_frequencies(
    head_dim: int,
    base: float = 10000.0,
    scale: float = 1.0,
) -> Tensor:
    """Compute RoPE frequency tensor for one scale.

    freqs = 1 / (base * scale)^(2i/head_dim) for i in 0..head_dim//2
    Returns (head_dim//2,) frequencies.
    """
    half = head_dim // 2
    i = torch.arange(half, dtype=torch.float32)
    # freqs[i] = 1 / (base * scale)^(2i / head_dim)
    freqs = 1.0 / ((base * scale) ** (2.0 * i / head_dim))
    return freqs


def apply_rope_single_scale(
    x: Tensor,          # (B, H, T, D)
    freqs: Tensor,      # (head_dim//2,)
    positions: Tensor,  # (T,)
) -> Tensor:
    """Apply RoPE rotation at a single scale.

    cos_sin = cos(positions * freqs), sin(positions * freqs) — shapes (T, head_dim//2)
    Rotate x by splitting into pairs and applying 2D rotation.
    Returns (B, H, T, D).
    """
    # angles: (T, head_dim//2)
    angles = positions.float().unsqueeze(1) * freqs.unsqueeze(0)  # (T, half)
    cos = angles.cos()  # (T, half)
    sin = angles.sin()  # (T, half)

    # expand for broadcasting: (1, 1, T, half)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # split x into even and odd indices
    x_even = x[..., 0::2]  # (B, H, T, half)
    x_odd  = x[..., 1::2]  # (B, H, T, half)

    # 2D rotation
    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd  = x_even * sin + x_odd * cos

    # interleave back
    out = torch.stack([x_rot_even, x_rot_odd], dim=-1)  # (B, H, T, half, 2)
    out = out.flatten(-2)  # (B, H, T, D)
    return out


def apply_hierarchical_rope(
    x: Tensor,              # (B, H, T, D)
    cfg: HierarchicalRoPEConfig,
    positions: Tensor | None = None,   # (T,) if None use 0..T-1
) -> Tensor:
    """Apply hierarchical RoPE: average of RoPE at each scale.

    For each scale in cfg.scale_factors: apply_rope_single_scale
    Return average of all scaled outputs.
    """
    T = x.shape[2]
    if positions is None:
        positions = torch.arange(T, dtype=torch.float32, device=x.device)

    outputs = []
    for scale in cfg.scale_factors:
        freqs = compute_rope_frequencies(cfg.head_dim, cfg.base, scale).to(x.device)
        out = apply_rope_single_scale(x, freqs, positions)
        outputs.append(out)

    # average across scales
    return torch.stack(outputs, dim=0).mean(dim=0)


def compute_position_bias(
    seq_len: int,
    n_scales: int,
    scale_factors: list[float],
) -> Tensor:
    """Compute additive position bias for attention.

    For each scale, compute log(1 + |i - j|) / scale for all (i, j) pairs.
    Average across scales. Returns (T, T) bias tensor.
    """
    i_idx = torch.arange(seq_len).unsqueeze(1).float()   # (T, 1)
    j_idx = torch.arange(seq_len).unsqueeze(0).float()   # (1, T)
    dist = (i_idx - j_idx).abs()                          # (T, T)

    bias_sum = torch.zeros(seq_len, seq_len)
    for scale in scale_factors:
        bias_sum = bias_sum + torch.log1p(dist) / scale

    return bias_sum / n_scales


class HierarchicalRoPEAttention(nn.Module):
    """Multi-head attention with hierarchical RoPE."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        cfg: HierarchicalRoPEConfig,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = cfg.head_dim
        self.cfg = cfg

        inner_dim = n_heads * cfg.head_dim
        self.q_proj = nn.Linear(d_model, inner_dim, bias=False)
        self.k_proj = nn.Linear(d_model, inner_dim, bias=False)
        self.v_proj = nn.Linear(d_model, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, d_model, bias=False)

    def forward(self, x: Tensor, positions: Tensor | None = None) -> Tensor:
        """x: (B, T, D) -> (B, T, D).

        Apply hierarchical RoPE to Q and K, then standard attention.
        """
        B, T, D = x.shape
        H = self.n_heads
        head_dim = self.head_dim

        q = self.q_proj(x).view(B, T, H, head_dim).transpose(1, 2)  # (B, H, T, head_dim)
        k = self.k_proj(x).view(B, T, H, head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, head_dim).transpose(1, 2)

        # apply hierarchical RoPE to Q and K
        q = apply_hierarchical_rope(q, self.cfg, positions)
        k = apply_hierarchical_rope(k, self.cfg, positions)

        # scaled dot-product attention
        scale = math.sqrt(head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, T, T)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B, H, T, head_dim)

        # reshape and project
        out = out.transpose(1, 2).contiguous().view(B, T, H * head_dim)
        return self.out_proj(out)


def interpolate_rope_frequencies(
    freqs: Tensor,      # (head_dim//2,)
    target_scale: float,
    current_scale: float = 1.0,
) -> Tensor:
    """Scale frequencies for interpolation-based context extension.

    Returns freqs * (current_scale / target_scale).
    """
    return freqs * (current_scale / target_scale)


class RoPEScaleScheduler:
    """Schedule scale_factors during training for curriculum position learning."""

    def __init__(
        self,
        init_scales: list[float],
        target_scales: list[float],
        warmup_steps: int,
    ) -> None:
        self.init_scales = list(init_scales)
        self.target_scales = list(target_scales)
        self.warmup_steps = warmup_steps
        self._current_step = 0

    def get_scales(self, step: int) -> list[float]:
        """Linear interpolation from init to target scales over warmup_steps."""
        if self.warmup_steps <= 0:
            return list(self.target_scales)
        t = min(step / self.warmup_steps, 1.0)
        return [
            init + t * (target - init)
            for init, target in zip(self.init_scales, self.target_scales)
        ]

    def update(self, step: int) -> None:
        self._current_step = step
