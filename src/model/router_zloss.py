"""Switch Transformer router z-loss.

This module follows the notation from Switch Transformers (Fedus et al., 2021):
`x` for router logits, `z(x)` for the tokenwise log-partition term, and `L_z`
for the mean squared z-loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass(frozen=True)
class RouterZLossOutput:
    """Router logits and z-loss terms."""

    x: Tensor
    z: Tensor
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
    """Return `z(x)` and `L_z = mean(z(x)^2)` over unmasked tokens."""
    if x.dim() < 2:
        raise ValueError("x must include an expert dimension")
    mask = _token_mask(token_mask, x.shape[:-1], x.device)
    z = torch.logsumexp(x, dim=-1)
    L_z = _masked_mean(z.square(), mask)
    return z, L_z


class RouterZLoss(nn.Module):
    """Project hidden states to router logits and compute Switch z-loss."""

    def __init__(self, d_model: int, n_experts: int, lambda_z: float = 1e-3) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if n_experts <= 0:
            raise ValueError(f"n_experts must be positive, got {n_experts}")
        if lambda_z < 0:
            raise ValueError(f"lambda_z must be non-negative, got {lambda_z}")

        self.d_model = d_model
        self.n_experts = n_experts
        self.lambda_z = lambda_z
        self.W_router = nn.Linear(d_model, n_experts, bias=False)
        nn.init.normal_(self.W_router.weight, std=0.02)

    def forward(self, h: Tensor, token_mask: Tensor | None = None) -> RouterZLossOutput:
        if h.dim() != 3:
            raise ValueError(f"h must have shape (batch, seq_len, d_model), got {tuple(h.shape)}")
        if h.size(-1) != self.d_model:
            raise ValueError(f"expected hidden size {self.d_model}, got {h.size(-1)}")

        x = self.W_router(h)
        z, L_z = router_z_loss(x, token_mask=token_mask)
        loss = self.lambda_z * L_z
        return RouterZLossOutput(x=x, z=z, L_z=L_z, loss=loss)
