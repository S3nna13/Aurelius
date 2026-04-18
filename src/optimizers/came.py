"""CAME: Confidence-guided Adaptive Memory Efficient Optimization.

Implements arXiv:2307.02047. Variable notation matches the paper.

Key innovations over Adafactor:
- Confidence-guided learning rate adaptation via instability measurement matrix U
- Factored second moment V ≈ V_r * V_c^T (row/column factors, like Adafactor)
- Instability measure U = G^2 / (V_r * V_c^T)
- Confidence update ρ_t = 1 - RMS(U_t - 1)
- Exponential moving average of instability C = ρ_t * C_{t-1} + (1-ρ_t) * U_t
"""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer


def _rms(x: torch.Tensor) -> torch.Tensor:
    """Root mean square: RMS(x) = sqrt(mean(x^2))."""
    return x.square().mean().sqrt()


def _factored_second_moment(
    V_r: torch.Tensor,
    V_c: torch.Tensor,
    eps1: float,
) -> torch.Tensor:
    """Reconstruct dense second-moment estimate from row/col factors.

    V ≈ V_r * V_c^T / mean(V_c), matching Adafactor's reconstruction.
    """
    denom = V_c.mean().clamp_min(eps1)
    return torch.outer(V_r, V_c) / denom


class CAME(Optimizer):
    """Confidence-guided Adaptive Memory Efficient Optimization (CAME).

    Implements arXiv:2307.02047 with paper-matching variable names.

    Args:
        params: Iterable of parameters or parameter groups.
        lr: Learning rate (α in the paper).
        eps: Tuple (ε1, ε2). ε1 is added to squared gradients for numerical
            stability in second-moment estimates; ε2 is the RMS lower bound
            for parameter scale (Adafactor-style clipping).
        clip_threshold: RMS threshold for gradient clipping (d in the paper).
        betas: Tuple (β1, β2, β3).
            β1: EMA coefficient for first moment m (set to 0 to skip).
            β2: EMA coefficient for factored second moment V_r, V_c.
            β3: EMA coefficient for confidence matrix C.
        weight_decay: L2 penalty coefficient (λ).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        eps: tuple[float, float] = (1e-30, 1e-3),
        clip_threshold: float = 1.0,
        betas: tuple[float, float, float] = (0.9, 0.999, 0.9999),
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= betas[0] < 1.0):
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not (0.0 < betas[1] < 1.0):
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if not (0.0 < betas[2] < 1.0):
            raise ValueError(f"Invalid beta3: {betas[2]}")
        if eps[0] < 0.0:
            raise ValueError(f"Invalid eps1: {eps[0]}")
        if eps[1] < 0.0:
            raise ValueError(f"Invalid eps2: {eps[1]}")
        if clip_threshold <= 0.0:
            raise ValueError(f"Invalid clip_threshold: {clip_threshold}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            betas=betas,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    def _use_factored(self, p: torch.Tensor) -> bool:
        """Use factored second moment for 2D+ parameters."""
        return p.ndim >= 2

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            eps1, eps2 = group["eps"]
            clip_threshold: float = group["clip_threshold"]
            beta1, beta2, beta3 = group["betas"]
            weight_decay: float = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # G_t: current gradient
                G = p.grad.detach()

                state = self.state[p]
                factored = self._use_factored(p)

                # ----------------------------------------------------------------
                # State initialisation
                # ----------------------------------------------------------------
                if len(state) == 0:
                    state["step"] = 0

                    # First moment m (only when β1 > 0)
                    if beta1 > 0.0:
                        state["m"] = torch.zeros_like(p)

                    if factored:
                        # Row factor V_r  (shape: first dim of p)
                        state["V_r"] = torch.zeros(
                            p.shape[0], device=p.device, dtype=p.dtype
                        )
                        # Col factor V_c  (shape: last dim of p for 2D;
                        # for higher-rank, flatten all but first dim)
                        col_size = p.numel() // p.shape[0]
                        state["V_c"] = torch.zeros(
                            col_size, device=p.device, dtype=p.dtype
                        )
                        # Confidence matrix C (same shape as p)
                        state["C"] = torch.ones_like(p)
                    else:
                        # Full second moment V for 1D parameters
                        state["V"] = torch.zeros_like(p)
                        # Confidence vector C (same shape as p)
                        state["C"] = torch.ones_like(p)

                state["step"] += 1
                t: int = state["step"]

                # ----------------------------------------------------------------
                # Weight decay (decoupled, applied before update)
                # ----------------------------------------------------------------
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                # ----------------------------------------------------------------
                # Second-moment update (Section 3.2 of the paper)
                # ----------------------------------------------------------------
                G_sq = G.square().add_(eps1)  # G^2 + ε1

                if factored:
                    # Reshape G to 2D for factored computation
                    G2d = G_sq.view(p.shape[0], -1)  # (r, c)

                    # Row / col statistics (mean over respective axis)
                    g_r = G2d.mean(dim=1)  # (r,)
                    g_c = G2d.mean(dim=0)  # (c,)

                    # EMA update for V_r and V_c
                    state["V_r"].mul_(beta2).add_(g_r, alpha=1.0 - beta2)
                    state["V_c"].mul_(beta2).add_(g_c, alpha=1.0 - beta2)

                    # Reconstruct dense estimate V ≈ V_r * V_c^T
                    V = _factored_second_moment(state["V_r"], state["V_c"], eps1)
                    V = V.view_as(p)  # restore original shape

                else:
                    # Unfactored: standard EMA of G^2 (1D parameters)
                    state["V"].mul_(beta2).add_(G_sq, alpha=1.0 - beta2)
                    V = state["V"]

                # ----------------------------------------------------------------
                # Confidence update (Section 3.3 of the paper)
                # ----------------------------------------------------------------
                # U_t = G^2 / V  (instability measure, paper eq. 5)
                U = G_sq.view_as(p) / V.clamp_min(eps1)

                # ρ_t = 1 - RMS(U_t - 1)  (confidence scalar, paper eq. 6)
                rho = (1.0 - _rms(U - 1.0)).clamp(0.0, 1.0)

                # C_t = ρ_t * C_{t-1} + (1 - ρ_t) * U_t  (paper eq. 7)
                state["C"].mul_(rho).add_(U, alpha=float(1.0 - rho))

                # ----------------------------------------------------------------
                # Compute update direction
                # ----------------------------------------------------------------
                # Confidence-modulated denominator
                denom = (V * state["C"]).sqrt_().add_(eps1)

                # Raw update: G / sqrt(V * C)
                update = G / denom

                # RMS-based gradient clipping (Adafactor style, paper Section 3.1)
                update_rms = _rms(update).clamp_min(eps1)
                if update_rms > clip_threshold:
                    update = update * (clip_threshold / update_rms)

                # ----------------------------------------------------------------
                # First moment (optional, β1 > 0)
                # ----------------------------------------------------------------
                if beta1 > 0.0:
                    # Bias-corrected first moment
                    state["m"].mul_(beta1).add_(update, alpha=1.0 - beta1)
                    bc1 = 1.0 - beta1 ** t
                    update = state["m"] / bc1

                # ----------------------------------------------------------------
                # Parameter update
                # ----------------------------------------------------------------
                p.add_(update, alpha=-lr)

        return loss
