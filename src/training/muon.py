"""Muon optimizer — Momentum Orthogonalized by Newton-schulz.

Reference: Liu et al., "Muon is Scalable for LLM Training", arXiv:2502.16982.
GitHub: https://github.com/MoonshotAI/Moonlight

Key idea: apply Newton-Schulz matrix orthogonalization to the gradient update
for weight matrices, giving better loss-per-FLOP than AdamW on transformer
linear layers.

Usage:
    Apply Muon to all 2D weight matrices in transformer layers.
    Keep AdamW for embeddings, norms, biases, and 1D params.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch.optim import Optimizer


def apply_qk_clip(param: torch.Tensor, threshold: float) -> None:
    """Rescale param in-place if its Frobenius norm exceeds threshold.

    Uses Frobenius norm as a cheap proxy for spectral norm.
    Rescales by threshold/norm so the post-clip norm equals threshold exactly.
    """
    with torch.no_grad():
        norm = param.float().norm()
        if norm < 1e-30:
            return
        if norm > threshold:
            param.mul_(threshold / norm)


def _newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate the orthogonal factor of G via Newton-Schulz iterations.

    Computes an approximation to G @ (G^T @ G)^{-1/2}, which is the
    orthogonal polar factor of G.

    Args:
        G: 2D gradient tensor of shape (m, n) with m >= n preferred.
        steps: Number of Newton-Schulz iterations (5 is usually enough).

    Returns:
        Orthogonalized matrix of the same shape as G.
    """
    assert G.ndim >= 2
    # Ensure m >= n by transposing if needed, then transpose back
    transposed = G.shape[0] < G.shape[1]
    if transposed:
        G = G.T

    # Normalize to unit spectral norm for numerical stability
    norm = G.norm()
    if norm < 1e-30:
        return G.T if transposed else G
    X = G / norm

    # Newton-Schulz coefficients (a, b, c) for the iteration X <- a*X + b*X@X.T@X
    # These are the stable quintic coefficients from the Muon paper
    a, b, c = (3.4445, -4.7750, 2.0315)

    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X

    if transposed:
        X = X.T

    return X


class Muon(Optimizer):
    """Muon: Momentum Orthogonalized by Newton-schulz optimizer.

    Designed for 2D+ weight matrices in transformer layers.
    Do NOT use for embeddings, normalization weights, or 1D tensors.

    Args:
        params: Iterable of parameters (must all be 2D or higher).
        lr: Learning rate (default: 0.02 — higher than AdamW's typical 3e-4).
        momentum: Momentum coefficient (default: 0.95).
        weight_decay: L2 weight decay (default: 0.0).
        ns_steps: Newton-Schulz iteration steps (default: 5).
        qk_clip: Frobenius norm threshold for QK-clip (default: None = disabled).
        qk_clip_alpha: Rescaling balance between Q and K (default: 0.5 = equal split).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        qk_clip: float | None = None,
        qk_clip_alpha: float = 0.5,
    ) -> None:
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)
        self.qk_clip = qk_clip
        self.qk_clip_alpha = qk_clip_alpha
        self._qk_params: set[int] = set()

    def mark_qk_params(self, params: Iterable[torch.Tensor]) -> None:
        """Mark parameters as Q or K projections for QK-clip rescaling."""
        for p in params:
            self._qk_params.add(id(p))

    @torch.no_grad()
    def step(self, closure=None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.ndim < 2:
                    raise ValueError(
                        "Muon requires 2D+ parameters. Use AdamW for 1D params (norms, biases)."
                    )

                g = p.grad.float()

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                # Orthogonalize the momentum buffer
                g_orth = _newton_schulz(buf.view(buf.shape[0], -1), steps=ns_steps)
                g_orth = g_orth.view_as(p)

                # Scale update to unit RMS so lr is interpretable across layer sizes
                rms = g_orth.pow(2).mean().sqrt().clamp(min=1e-8)
                g_orth = g_orth / rms

                # Weight decay (applied before parameter update)
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                p.add_(g_orth.to(p.dtype), alpha=-lr)

                # QK-clip: rescale Q/K projections if Frobenius norm exceeds threshold
                if self.qk_clip is not None and id(p) in self._qk_params:
                    apply_qk_clip(p, self.qk_clip)

        return loss
