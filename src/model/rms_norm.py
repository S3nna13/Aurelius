"""RMSNorm — Root Mean Square Layer Normalization.

Reference: Zhang & Sennrich, 2019 — "Root Mean Square Layer Normalization".
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Simpler and faster than LayerNorm: normalises by the RMS of activations
    without subtracting the mean, then scales by a learnable weight vector.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # float32 for numerical stability during norm computation
        return x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps).to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x) * self.weight
