"""Manifold-constrained hyper-connections (mHC)."""

from __future__ import annotations

import torch
import torch.nn as nn


def _sinkhorn_knopp(matrix: torch.Tensor, steps: int = 20, eps: float = 1e-6) -> torch.Tensor:
    """Project a square matrix onto the doubly stochastic simplex."""
    if matrix.dim() != 2:
        raise ValueError("sinkhorn_knopp expects a 2D matrix")
    if matrix.size(0) != matrix.size(1):
        raise ValueError("sinkhorn_knopp expects a square matrix")

    # Keep the transform numerically stable and strictly positive.
    x = matrix.float()
    x = x - x.max()
    x = torch.exp(x)

    for _ in range(max(1, steps)):
        x = x / x.sum(dim=-1, keepdim=True).clamp_min(eps)
        x = x / x.sum(dim=-2, keepdim=True).clamp_min(eps)

    return x.to(dtype=matrix.dtype)


class ManifoldConstrainedHyperConnection(nn.Module):
    """Lightweight residual adapter used by the mHC tests."""

    def __init__(self, d_model: int, n_hc: int = 4, sinkhorn_steps: int = 20) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_hc = max(1, int(n_hc))
        self.sinkhorn_steps = max(1, int(sinkhorn_steps))

        self.assignment = nn.Parameter(torch.randn(self.n_hc, self.n_hc) * 0.02)
        self.adapters = nn.ModuleList(
            [nn.Linear(d_model, d_model, bias=False) for _ in range(self.n_hc)]
        )

    def forward(self, x: torch.Tensor, sublayer: nn.Module, *args, **kwargs):
        sublayer_result = sublayer(x, *args, **kwargs)
        if isinstance(sublayer_result, tuple):
            sublayer_out, *extra = sublayer_result
        else:
            sublayer_out = sublayer_result
            extra = []

        weights = _sinkhorn_knopp(self.assignment, steps=self.sinkhorn_steps)
        mix = torch.diagonal(weights).mean().to(dtype=sublayer_out.dtype, device=sublayer_out.device)

        adapted = torch.zeros_like(sublayer_out)
        for adapter in self.adapters:
            adapted = adapted + adapter(sublayer_out)
        adapted = adapted / float(self.n_hc)

        out = x + mix * adapted

        if extra:
            return (out, *extra)
        return out


class MHCLayer(nn.Module):
    """Wrap an attention module and FFN with mHC adapters."""

    def __init__(
        self,
        attn: nn.Module,
        ffn: nn.Module,
        d_model: int,
        n_hc: int = 4,
        sinkhorn_steps: int = 20,
    ) -> None:
        super().__init__()
        self.attn = attn
        self.ffn = ffn
        self.mhc_attn = ManifoldConstrainedHyperConnection(d_model, n_hc, sinkhorn_steps)
        self.mhc_ffn = ManifoldConstrainedHyperConnection(d_model, n_hc, sinkhorn_steps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None, torch.Tensor]:
        attn_result = self.mhc_attn(x, self.attn, freqs_cis, mask, past_kv)
        if isinstance(attn_result, tuple):
            x = attn_result[0]
            kv = attn_result[1] if len(attn_result) > 1 else None
        else:
            kv = None

        ffn_result = self.mhc_ffn(x, self.ffn)
        if isinstance(ffn_result, tuple):
            x = ffn_result[0]
            aux_loss = ffn_result[1] if len(ffn_result) > 1 else x.new_zeros(())
        else:
            x = ffn_result
            aux_loss = x.new_zeros(())

        if not isinstance(aux_loss, torch.Tensor):
            aux_loss = x.new_tensor(aux_loss)

        return x, kv, aux_loss


__all__ = [
    "MHCLayer",
    "ManifoldConstrainedHyperConnection",
    "_sinkhorn_knopp",
]
