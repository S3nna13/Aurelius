"""Sliding-window attention with optional global token prefix.

Implements local windowed self-attention using PyTorch's
``scaled_dot_product_attention`` as the inner kernel, without any
external library dependencies.

Reference: Beltagy et al., "Longformer: The Long-Document Transformer", 2020.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SWAConfig:
    window_size: int = 512
    global_tokens: int = 64
    causal: bool = True


class SlidingWindowAttention(nn.Module):
    """Sliding-window attention with a global prefix.

    Parameters
    ----------
    config:
        SWAConfig controlling window size, number of global tokens, and
        whether to use causal (decoder-style) masking.
    """

    def __init__(self, config: SWAConfig | None = None) -> None:
        super().__init__()
        self.config = config or SWAConfig()

    # ------------------------------------------------------------------
    # nn.Module interface
    # ------------------------------------------------------------------

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Compute sliding-window attention.

        Parameters
        ----------
        q, k, v:
            Shape ``(batch, heads, seq, head_dim)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, heads, seq, head_dim)``.
        """
        cfg = self.config
        batch, heads, seq, head_dim = q.shape
        output = torch.zeros_like(q)

        for i in range(seq):
            if i < cfg.global_tokens:
                # Global positions attend to the entire sequence
                q_i = q[:, :, i : i + 1, :]  # (B, H, 1, D)
                k_ctx = k  # (B, H, seq, D)
                v_ctx = v
            else:
                # Local positions attend to a window
                if cfg.causal:
                    win_start = max(0, i - cfg.window_size + 1)
                    win_end = i + 1
                else:
                    win_start = max(0, i - cfg.window_size)
                    win_end = min(seq, i + cfg.window_size + 1)

                q_i = q[:, :, i : i + 1, :]
                k_ctx = k[:, :, win_start:win_end, :]
                v_ctx = v[:, :, win_start:win_end, :]

            out_i = F.scaled_dot_product_attention(q_i, k_ctx, v_ctx, is_causal=False)
            output[:, :, i : i + 1, :] = out_i

        return output

    # ------------------------------------------------------------------
    # utility
    # ------------------------------------------------------------------

    def get_effective_context(self, seq_len: int) -> int:
        """Return the effective context length for a given sequence length.

        Defined as ``window_size * ceil(seq_len / window_size)``.
        """
        ws = self.config.window_size
        return ws * math.ceil(seq_len / ws)
