"""Muon optimizer with hybrid Newton-Schulz orthogonalization.

Upgraded version: uses two-stage Newton-Schulz iterations (8 fast + 2
stabilizing), Nesterov momentum, and RMS rescaling for stable convergence.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer


def _newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    if G.ndim < 2:
        raise ValueError("Muon requires 2D+ tensors.")
    transposed = G.shape[0] < G.shape[1]
    if transposed:
        G = G.T

    norm = G.norm()
    if norm < 1e-30:
        return G.T if transposed else G
    X = G / norm

    a, b, c = (3.4445, -4.7750, 2.0315)

    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X

    if transposed:
        X = X.T
    return X


def _hybrid_newton_schulz(G: torch.Tensor) -> torch.Tensor:
    """Two-stage Newton-Schulz: 8 rapid convergence + 2 stabilizing iterations."""
    if G.ndim < 2:
        raise ValueError("Muon requires 2D+ tensors.")
    transposed = G.shape[0] < G.shape[1]
    if transposed:
        G = G.T

    norm = G.norm()
    if norm < 1e-30:
        return G.T if transposed else G
    X = G / norm

    a1, b1, c1 = (3.4445, -4.7750, 2.0315)
    for _ in range(8):
        A = X @ X.T
        X = a1 * X + (b1 * A + c1 * A @ A) @ X

    a2, b2, c2 = (2.0, -1.5, 0.5)
    for _ in range(2):
        A = X @ X.T
        X = a2 * X + (b2 * A + c2 * A @ A) @ X

    if transposed:
        X = X.T
    return X


class Muon(Optimizer):
    """Muon optimizer with hybrid Newton-Schulz orthogonalization.

    Designed for 2D+ weight matrices (linear projections, embeddings, etc).
    For 1D parameters (norms, biases) use AdamW instead.

    Args:
        params: Iterable of parameters.
        lr: Learning rate.
        momentum: Momentum coefficient (default 0.95).
        weight_decay: Weight decay factor (default 0.1).
        ns_steps: Newton-Schulz steps (default 5; ignored if hybrid=True).
        hybrid: Use two-stage 8+2 Newton-Schulz (default True).
        nesterov: Apply Nesterov momentum (default True).
        update_rms: Target RMS for update rescaling (default 0.18).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        ns_steps: int = 5,
        hybrid: bool = True,
        nesterov: bool = True,
        update_rms: float = 0.18,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            hybrid=hybrid,
            nesterov=nesterov,
            update_rms=update_rms,
        )
        super().__init__(params, defaults)

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
            hybrid = group["hybrid"]
            nesterov = group["nesterov"]
            update_rms = group["update_rms"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.ndim < 2:
                    raise ValueError(
                        "Muon requires 2D+ parameters. Use AdamW for 1D params."
                    )

                g = p.grad.float()

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                if nesterov:
                    g_nesterov = momentum * buf + g
                else:
                    g_nesterov = buf

                if hybrid:
                    g_orth = _hybrid_newton_schulz(g_nesterov.view(g_nesterov.shape[0], -1))
                else:
                    g_orth = _newton_schulz(
                        g_nesterov.view(g_nesterov.shape[0], -1), steps=ns_steps
                    )
                g_orth = g_orth.view_as(p)

                rms = g_orth.pow(2).mean().sqrt().clamp(min=1e-8)
                g_orth = g_orth / rms * update_rms

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                p.add_(g_orth.to(p.dtype), alpha=-lr)

        return loss
