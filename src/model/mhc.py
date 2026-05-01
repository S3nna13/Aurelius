"""Manifold-Constrained Hyper-Connections (mHC)."""

from __future__ import annotations

import torch
import torch.nn as nn


def _sinkhorn_knopp(matrix: torch.Tensor, steps: int = 20, eps: float = 1e-6) -> torch.Tensor:
    """Project a nonnegative matrix to an approximate doubly stochastic matrix."""
    if matrix.ndim != 2:
        raise ValueError("sinkhorn_knopp expects a 2D matrix")

    result = matrix.float().abs() + eps
    for _ in range(max(1, steps)):
        result = result / result.sum(dim=-1, keepdim=True).clamp_min(eps)
        result = result / result.sum(dim=-2, keepdim=True).clamp_min(eps)
    return result


class ManifoldConstrainedHyperConnection(nn.Module):
    """Wrap a sublayer with a residual path and a lightweight routing gate."""

    def __init__(self, d_model: int, n_hc: int = 4, sinkhorn_steps: int = 20) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_hc = n_hc
        self.sinkhorn_steps = sinkhorn_steps
        self.norm = nn.LayerNorm(d_model)
        self.route = nn.Linear(d_model, n_hc, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, sublayer: nn.Module, *args, **kwargs):
        residual = x
        normed = self.norm(x)
        routing = torch.softmax(self.route(normed), dim=-1)
        candidate = sublayer(normed, *args, **kwargs)

        extras: tuple = ()
        if isinstance(candidate, tuple):
            candidate, *rest = candidate
            extras = tuple(rest)

        updated = residual + self.proj(candidate) * (1.0 + routing[..., :1])
        if extras:
            return (updated, *extras)
        return updated


class MHCLayer(nn.Module):
    """Apply mHC to attention and FFN sublayers."""

    def __init__(self, attn: nn.Module, ffn: nn.Module, d_model: int) -> None:
        super().__init__()
        self.attn = attn
        self.ffn = ffn
        self.attn_mhc = ManifoldConstrainedHyperConnection(d_model)
        self.ffn_mhc = ManifoldConstrainedHyperConnection(d_model)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None, torch.Tensor]:
        attn_result = self.attn_mhc(x, self.attn, freqs_cis, mask, past_kv)
        if isinstance(attn_result, tuple):
            x, kv = attn_result
        else:
            kv = None
            x = attn_result

        ffn_result = self.ffn_mhc(x, self.ffn)
        if isinstance(ffn_result, tuple):
            x, aux_loss = ffn_result
        else:
            aux_loss = x.new_zeros(())
            x = ffn_result

        return x, kv, aux_loss
