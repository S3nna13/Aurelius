"""Dynamic NTK-aware RoPE scaling for extended context (LocalLLaMA community, 2023)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class NTKRoPEConfig:
    base: float = 10000.0
    dim: int = 64
    max_seq_len: int = 2048
    alpha: float = 1.0
    scaling_factor: float = 1.0


class DynamicNTKRoPE(nn.Module):
    """Dynamic NTK-aware RoPE scaling.

    When the input sequence exceeds max_seq_len the base frequency is
    rescaled using the NTK correction so that the model can represent
    longer contexts without fine-tuning.
    """

    def __init__(self, config: NTKRoPEConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else NTKRoPEConfig()

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _compute_freqs(self, seq_len: int) -> Tensor:
        """Return positional frequencies of shape (seq_len, dim//2)."""
        cfg = self.config
        base = cfg.base

        if seq_len > cfg.max_seq_len:
            # NTK correction: scale base by alpha^(dim / (dim - 2))
            exponent = cfg.dim / (cfg.dim - 2)
            base = base * (cfg.alpha**exponent)

        half_dim = cfg.dim // 2
        # inv_freq[i] = 1 / base^(2i/dim)
        i = torch.arange(half_dim, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (2.0 * i / cfg.dim))  # (half_dim,)

        # positions: (seq_len,)
        t = torch.arange(seq_len, dtype=torch.float32)

        # outer product -> (seq_len, half_dim)
        freqs = torch.outer(t, inv_freq)
        return freqs

    @staticmethod
    def rotate_half(x: Tensor) -> Tensor:
        """Rotate the last dimension by half: cat([-x[..., half:], x[..., :half]], dim=-1)."""
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: Tensor, seq_len: int) -> Tensor:
        """Apply rotary embeddings to x.

        Args:
            x: (batch, heads, seq_len, head_dim)
            seq_len: sequence length (may differ from x.shape[2] during generation)

        Returns:
            Tensor of same shape as x with rotary embeddings applied.
        """
        freqs = self._compute_freqs(seq_len)  # (seq_len, dim//2)

        # head_dim = x.shape[-1]; we use the first dim//2 dimensions for rotation
        self.config.dim // 2
        head_dim = x.shape[-1]

        # Build cos/sin of shape (1, 1, seq_len, head_dim) by repeating freqs twice
        # to cover full head_dim (standard RoPE interleaving via cat)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)

        # Trim or pad to match head_dim
        if emb.shape[-1] < head_dim:
            pad = torch.zeros(seq_len, head_dim - emb.shape[-1])
            emb = torch.cat([emb, pad], dim=-1)
        else:
            emb = emb[..., :head_dim]

        cos = emb.cos()[None, None, :, :]  # (1, 1, seq_len, head_dim)
        sin = emb.sin()[None, None, :, :]

        # Trim x's seq dimension in case x has more tokens than seq_len
        x_rot = x[..., :seq_len, :]
        return x_rot * cos + self.rotate_half(x_rot) * sin

    def max_supported_len(self, alpha: float) -> int:
        """Approximate maximum supported sequence length for a given alpha.

        Approximation: max_seq_len * alpha^(dim/2 / (dim-2))
        """
        cfg = self.config
        exponent = (cfg.dim / 2) / (cfg.dim - 2)
        return int(cfg.max_seq_len * (alpha**exponent))
