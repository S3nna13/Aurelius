from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class LongRoPEConfig:
    dim: int = 128
    base: float = 10000.0
    original_max: int = 4096
    extended_max: int = 131072
    short_factor: Optional[list] = None
    long_factor: Optional[list] = None


class LongRoPEEmbedding(nn.Module):
    """LongRoPE: non-uniform per-dimension RoPE scaling with short/long rescue factors.

    Reference: arXiv 2402.13753
    """

    def __init__(self, config: LongRoPEConfig) -> None:
        super().__init__()
        self.config = config
        half = config.dim // 2

        if config.short_factor is not None:
            short_t = torch.tensor(config.short_factor, dtype=torch.float32)
            if short_t.shape[0] != half:
                raise ValueError(f"short_factor length {short_t.shape[0]} != dim//2 {half}")
        else:
            short_t = torch.linspace(1.0, 2.0, half)

        if config.long_factor is not None:
            long_t = torch.tensor(config.long_factor, dtype=torch.float32)
            if long_t.shape[0] != half:
                raise ValueError(f"long_factor length {long_t.shape[0]} != dim//2 {half}")
        else:
            long_t = torch.linspace(1.0, 2.0, half)

        self.register_buffer("short_factor", short_t, persistent=False)
        self.register_buffer("long_factor", long_t, persistent=False)

        exponents = torch.arange(0, half, dtype=torch.float32) * 2.0 / config.dim
        base_inv_freq = 1.0 / (config.base ** exponents)
        self.register_buffer("base_inv_freq", base_inv_freq, persistent=False)

    def _get_factors(self, seq_len: int) -> torch.Tensor:
        if seq_len <= self.config.original_max:
            return self.short_factor
        return self.long_factor

    def forward(
        self,
        seq_len: int,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        factors = self._get_factors(seq_len)
        base_inv_freq = self.base_inv_freq
        if device is not None:
            factors = factors.to(device)
            base_inv_freq = base_inv_freq.to(device)

        inv_freq = base_inv_freq / factors
        t = torch.arange(seq_len, device=inv_freq.device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        return cos, sin

    @staticmethod
    def apply_rotary(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        half = q.shape[-1] // 2

        def rotate(x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x[..., :half], x[..., half:]
            return torch.cat((-x2, x1), dim=-1)

        cos_full = torch.cat([cos, cos], dim=-1).unsqueeze(0).unsqueeze(0)
        sin_full = torch.cat([sin, sin], dim=-1).unsqueeze(0).unsqueeze(0)
        q_rot = q * cos_full + rotate(q) * sin_full
        k_rot = k * cos_full + rotate(k) * sin_full
        return q_rot, k_rot
