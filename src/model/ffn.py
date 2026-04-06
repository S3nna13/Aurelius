"""SwiGLU Feed-Forward Network.

Reference: Shazeer, 2020 — "GLU Variants Improve Transformer".
SwiGLU uses Swish(x * W_gate) * (x * W_up), then projects down.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AureliusConfig


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network.

    The gating mechanism splits the up-projection into a gate and value path:
        output = (Swish(x @ W_gate) * (x @ W_up)) @ W_down

    All linear layers are bias-free.
    """

    def __init__(self, config: AureliusConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))
