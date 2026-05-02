from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class RoPEConfig:
    dim: int = 128
    base: float = 10000.0
    max_position: int = 4096
    scaling_type: str = "none"
    scaling_factor: float = 1.0
    original_max_position: int = 4096


class RotaryEmbedding(nn.Module):
    """Standard RoPE with pluggable scaling strategies."""

    def __init__(self, config: RoPEConfig) -> None:
        super().__init__()
        self.config = config
        inv_freq = self._compute_inv_freq()
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _compute_inv_freq(self) -> torch.Tensor:
        cfg = self.config
        half = cfg.dim // 2
        exponents = torch.arange(0, half, dtype=torch.float32) * 2.0 / cfg.dim

        scaling = cfg.scaling_type

        if scaling == "none":
            effective_base = cfg.base
            inv_freq = 1.0 / (effective_base**exponents)

        elif scaling == "linear":
            inv_freq = 1.0 / (cfg.base**exponents) / cfg.scaling_factor

        elif scaling == "ntk":
            adjusted_base = cfg.base * (cfg.scaling_factor ** (cfg.dim / (cfg.dim - 2)))
            inv_freq = 1.0 / (adjusted_base**exponents)

        elif scaling == "yarn":
            alpha = cfg.original_max_position
            beta = cfg.max_position
            inv_freq_base = 1.0 / (cfg.base**exponents)
            wavelengths = 2.0 * math.pi / inv_freq_base
            low_mask = wavelengths < (2.0 * math.pi * alpha / cfg.scaling_factor)
            high_mask = wavelengths > (2.0 * math.pi * beta / cfg.scaling_factor)
            mid_mask = ~low_mask & ~high_mask
            scale = torch.ones(half)
            scale[low_mask] = 1.0
            scale[high_mask] = cfg.scaling_factor
            if mid_mask.any():
                w = (wavelengths[mid_mask] / (2.0 * math.pi) * cfg.scaling_factor - alpha) / (
                    beta - alpha
                )
                scale[mid_mask] = 1.0 - w + w * cfg.scaling_factor
            inv_freq = inv_freq_base / scale

        else:
            raise ValueError(f"Unknown scaling_type: {scaling!r}")

        return inv_freq

    def forward(
        self,
        seq_len: int,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq
        if device is not None:
            inv_freq = inv_freq.to(device)
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
