"""Best-of-N reranking with verifier-ranker scoring.

Given a prompt ``x`` and sampled candidates ``Y = {y_i}_{i=1}^N``, the module
computes verifier-ranker scores ``r_i = r_phi(x, y_i)`` and selects
``y_star = y_{n_star}``, where ``n_star = argmax_i r_i``.

This matches the notation used in Best-of-N Sampling Improves Large Language
Model Generation with Verifier-Ranker (arXiv:2305.14765).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BestOfNRerankingConfig:
    """Configuration for the verifier-ranker ``r_phi(x, y)``."""

    vocab_size: int
    d_model: int = 64
    d_hidden: int = 128
    pad_token_id: int = 0
    eps: float = 1e-8


def _resolve_mask(
    z: torch.Tensor,
    z_mask: torch.Tensor | None,
    pad_token_id: int,
) -> torch.Tensor:
    if z_mask is None:
        return z.ne(pad_token_id)
    if z_mask.shape != z.shape:
        raise ValueError("mask shape must match token tensor shape")
    return z_mask.to(dtype=torch.bool)


def _masked_mean(h: torch.Tensor, h_mask: torch.Tensor, eps: float) -> torch.Tensor:
    w = h_mask.to(dtype=h.dtype).unsqueeze(-1)
    denom = w.sum(dim=-2).clamp_min(eps)
    return (h * w).sum(dim=-2) / denom


def select_best_of_n(Y: torch.Tensor, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Select ``y_star`` from candidates ``Y`` using verifier scores ``r``.

    Args:
        Y: Candidate completions with shape ``(B, N, T_y)``.
        r: Verifier-ranker scores with shape ``(B, N)``.

    Returns:
        ``(y_star, n_star)`` where ``y_star`` has shape ``(B, T_y)`` and
        ``n_star`` has shape ``(B,)``.
    """
    if Y.ndim != 3:
        raise ValueError("Y must have shape (B, N, T_y)")
    if r.ndim != 2:
        raise ValueError("r must have shape (B, N)")
    if Y.shape[:2] != r.shape:
        raise ValueError("Y and r must agree on batch and candidate dimensions")

    B = Y.size(0)
    n_star = r.argmax(dim=1)
    y_star = Y[torch.arange(B, device=Y.device), n_star]
    return y_star, n_star


def best_of_n_reranking_loss(r: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Train ``r`` to rank the highest-utility candidate first.

    ``u`` denotes target utilities over the same ``N`` candidates and induces the
    target index ``argmax_i u_i``.
    """
    if r.shape != u.shape:
        raise ValueError("r and u must have the same shape")
    n_star = u.argmax(dim=1)
    return F.cross_entropy(r, n_star)


class BestOfNReranker(nn.Module):
    """Trainable verifier-ranker that scores sampled candidates ``y_i``."""

    def __init__(self, config: BestOfNRerankingConfig) -> None:
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.proj = nn.Sequential(
            nn.Linear(4 * config.d_model, config.d_hidden),
            nn.GELU(),
            nn.Linear(config.d_hidden, 1),
        )

    def score(
        self,
        x: torch.Tensor,
        Y: torch.Tensor,
        x_mask: torch.Tensor | None = None,
        Y_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute verifier-ranker scores ``r`` for all candidates in ``Y``."""
        if x.ndim != 2:
            raise ValueError("x must have shape (B, T_x)")
        if Y.ndim != 3:
            raise ValueError("Y must have shape (B, N, T_y)")
        if x.size(0) != Y.size(0):
            raise ValueError("x and Y must have the same batch size")

        B, N, T_y = Y.shape
        x_mask = _resolve_mask(x, x_mask, self.config.pad_token_id)
        Y_mask = _resolve_mask(Y, Y_mask, self.config.pad_token_id)

        h_x = _masked_mean(self.embed(x), x_mask, self.config.eps)

        Y_flat = Y.reshape(B * N, T_y)
        Y_mask_flat = Y_mask.reshape(B * N, T_y)
        h_y = _masked_mean(self.embed(Y_flat), Y_mask_flat, self.config.eps)

        h_x = h_x.unsqueeze(1).expand(B, N, h_x.size(-1)).reshape(B * N, h_x.size(-1))
        phi_xy = torch.cat([h_x, h_y, h_x * h_y, h_y - h_x], dim=-1)
        r = self.proj(phi_xy).reshape(B, N)
        return r

    def forward(
        self,
        x: torch.Tensor,
        Y: torch.Tensor,
        x_mask: torch.Tensor | None = None,
        Y_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(r, y_star, n_star)`` for prompt ``x`` and candidates ``Y``."""
        r = self.score(x=x, Y=Y, x_mask=x_mask, Y_mask=Y_mask)
        y_star, n_star = select_best_of_n(Y=Y, r=r)
        return r, y_star, n_star
