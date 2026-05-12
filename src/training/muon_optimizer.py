"""MUON optimizer — Momentum Orthogonalized by Newton-Schulz iterations.

Reference: Kosson et al. 2025 (arXiv:2502.16982). Apply to all hidden-layer
parameters. Embedding and LM head use AdamW (they don't benefit from
orthogonalization due to their asymmetric structure).
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.optim import Optimizer


def _zeropower_via_newtonschulz5(G: Tensor, steps: int = 5) -> Tensor:
    """Newton-Schulz iteration for approximate matrix square root inverse."""
    if G.ndim < 2:
        raise ValueError(f"Expected 2D+ tensor, got {G.ndim}D")
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G / (G.norm() + 1e-7)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + b * A @ X + c * A @ A @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(Optimizer):
    """MUON optimizer for hidden-layer weight matrices.

    Use AdamW for embeddings, LM head, biases, and scalar/vector params.
    Use Muon for all other weight matrices (attention, FFN).
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 ns_steps: int = 5, nesterov: bool = True):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]
            nesterov = group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if nesterov else buf
                if g.ndim >= 2:
                    g = _zeropower_via_newtonschulz5(g, steps=ns_steps)
                    g *= max(g.size(0), g.size(1)) ** 0.5
                p.add_(g, alpha=-lr)
        return loss


def build_muon_optimizer(model, muon_lr: float = 0.02, adam_lr: float = 3e-4,
                         weight_decay: float = 0.1) -> tuple:
    """Split model params into Muon (matrices) and AdamW (rest) groups."""
    muon_params, adam_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "embed" in name or "norm" in name or "bias" in name:
            adam_params.append(p)
        else:
            muon_params.append(p)

    muon_opt = Muon(muon_params, lr=muon_lr)
    adam_opt = torch.optim.AdamW(adam_params, lr=adam_lr,
                                  betas=(0.9, 0.95), weight_decay=weight_decay)
    return muon_opt, adam_opt


__all__ = ["Muon", "build_muon_optimizer", "_zeropower_via_newtonschulz5"]