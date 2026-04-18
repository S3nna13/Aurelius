"""ST-MoE router regularizers for collapse prevention.

This module follows the notation used in ST-MoE (Zoph et al., 2022):
`f_i` for token fraction per expert, `P_i` for mean router probability per
expert, `z(x)` for the log-partition term, `L_aux` for load balancing, and
`L_z` for the router z-loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass(frozen=True)
class RouterCollapseOutput:
    """Router outputs and ST-MoE regularization terms."""

    x: Tensor
    p: Tensor
    indices: Tensor
    f_i: Tensor
    P_i: Tensor
    z: Tensor
    L_aux: Tensor
    L_z: Tensor
    loss: Tensor


def _token_mask(token_mask: Tensor | None, shape: torch.Size, device: torch.device) -> Tensor:
    if token_mask is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    if token_mask.shape != shape:
        raise ValueError(
            f"token_mask must have shape {tuple(shape)}, got {tuple(token_mask.shape)}"
        )
    return token_mask.to(device=device, dtype=torch.bool)


def _masked_mean(value: Tensor, token_mask: Tensor) -> Tensor:
    weight = token_mask.to(dtype=value.dtype)
    denom = weight.sum()
    if denom.item() == 0:
        return value.new_zeros(())
    return (value * weight).sum() / denom


def router_z_loss(x: Tensor, token_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Return `z(x)` and the ST-MoE router z-loss `L_z`."""
    if x.dim() < 2:
        raise ValueError("x must include an expert dimension")
    mask = _token_mask(token_mask, x.shape[:-1], x.device)
    z = torch.logsumexp(x, dim=-1)
    L_z = _masked_mean(z.square(), mask)
    return z, L_z


def load_balancing_loss(
    p: Tensor,
    indices: Tensor,
    n_experts: int,
    token_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return `f_i`, `P_i`, and the ST-MoE load-balancing loss `L_aux`."""
    if p.dim() < 2:
        raise ValueError("p must include an expert dimension")
    if p.shape[-1] != n_experts:
        raise ValueError(f"expected {n_experts} experts, got {p.shape[-1]}")
    if indices.shape[:-1] != p.shape[:-1]:
        raise ValueError("indices must match p on batch dimensions")

    mask = _token_mask(token_mask, p.shape[:-1], p.device)
    p_flat = p.reshape(-1, n_experts)
    mask_flat = mask.reshape(-1)
    indices_flat = indices.reshape(-1, indices.size(-1))

    if not mask_flat.any():
        zero = p.new_zeros(n_experts)
        return zero, zero, p.new_zeros(())

    p_valid = p_flat[mask_flat]
    indices_valid = indices_flat[mask_flat]

    P_i = p_valid.mean(dim=0)

    counts = p.new_zeros(n_experts)
    counts.scatter_add_(0, indices_valid.reshape(-1), p.new_ones(indices_valid.numel()))
    f_i = counts / indices_valid.numel()

    L_aux = float(n_experts) * torch.sum(f_i * P_i)
    return f_i, P_i, L_aux


class RouterCollapsePrevention(nn.Module):
    """Router projection with ST-MoE collapse-prevention losses."""

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        k: int = 1,
        lambda_aux: float = 1.0,
        lambda_z: float = 1e-3,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if n_experts <= 0:
            raise ValueError(f"n_experts must be positive, got {n_experts}")
        if k <= 0 or k > n_experts:
            raise ValueError(
                f"k must satisfy 1 <= k <= n_experts, got k={k}, n_experts={n_experts}"
            )
        if lambda_aux < 0 or lambda_z < 0:
            raise ValueError("lambda_aux and lambda_z must be non-negative")

        self.d_model = d_model
        self.n_experts = n_experts
        self.k = k
        self.lambda_aux = lambda_aux
        self.lambda_z = lambda_z
        self.W_g = nn.Linear(d_model, n_experts, bias=False)
        nn.init.normal_(self.W_g.weight, std=0.02)

    def forward(self, h: Tensor, token_mask: Tensor | None = None) -> RouterCollapseOutput:
        if h.dim() != 3:
            raise ValueError(f"h must have shape (batch, seq_len, d_model), got {tuple(h.shape)}")
        if h.size(-1) != self.d_model:
            raise ValueError(f"expected hidden size {self.d_model}, got {h.size(-1)}")

        x = self.W_g(h)
        p = torch.softmax(x, dim=-1)
        _, indices = torch.topk(p, k=self.k, dim=-1)

        f_i, P_i, L_aux = load_balancing_loss(
            p=p,
            indices=indices,
            n_experts=self.n_experts,
            token_mask=token_mask,
        )
        z, L_z = router_z_loss(x, token_mask=token_mask)
        loss = self.lambda_aux * L_aux + self.lambda_z * L_z

        return RouterCollapseOutput(
            x=x,
            p=p,
            indices=indices,
            f_i=f_i,
            P_i=P_i,
            z=z,
            L_aux=L_aux,
            L_z=L_z,
            loss=loss,
        )
