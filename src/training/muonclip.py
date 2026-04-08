"""MuonClip optimizer — Muon with per-parameter gradient clipping for stable MoE training.

Reference: Kimi-K2.5 technical report (Moonshot AI, 2025).

Key idea: large MoE models exhibit gradient spikes that destabilize training.
MuonClip adds a percentile-based gradient clipping step BEFORE the Newton-Schulz
orthogonalization, suppressing outlier gradient values without global norm clipping.

For matrix parameters (2D): apply clipping → Newton-Schulz → SGD with Nesterov momentum.
For vector/scalar parameters (1D or 0D): apply clipping → Adam-style update.
"""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer


class MuonClip(Optimizer):
    """Muon optimizer with per-parameter gradient clipping (MuonClip).

    For matrix parameters: applies Newton-Schulz orthogonalization on gradients.
    For vector/scalar parameters: falls back to AdamW-style update.

    Args:
        params: Parameter groups.
        lr: Learning rate (default 0.02).
        momentum: Nesterov momentum coefficient (default 0.95).
        nesterov: Use Nesterov momentum (default True).
        ns_steps: Newton-Schulz iteration steps (default 5).
        clip_percentile: Percentile for gradient norm clipping (default 99.0).
        weight_decay: L2 regularization coefficient (default 0.0).
        betas: Adam betas for 1D param fallback (default (0.9, 0.999)).
        eps: Adam epsilon for numerical stability (default 1e-8).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        clip_percentile: float = 99.0,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            clip_percentile=clip_percentile,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
        super().__init__(params, defaults)

    def newton_schulz_orthogonalize(self, G: torch.Tensor, steps: int = 5) -> torch.Tensor:
        """Approximate orthogonalization via Newton-Schulz iterations.

        Normalizes the input, runs the cubic Newton-Schulz recurrence for
        `steps` iterations, then rescales the result by the original norm so
        that the update magnitude is preserved.

        Iteration: X_{k+1} = 1.5 * X_k - 0.5 * X_k @ X_k.T @ X_k

        Args:
            G: (m, n) gradient matrix.
            steps: Number of Newton-Schulz iterations.

        Returns:
            Approximately orthogonal matrix of the same shape as G.
        """
        assert G.ndim >= 2, "newton_schulz_orthogonalize requires a 2D+ tensor"

        # Ensure tall (m >= n) for numerical stability; transpose back at the end
        transposed = G.shape[0] < G.shape[1]
        if transposed:
            G = G.T

        norm = G.norm()
        if norm < 1e-30:
            return G.T if transposed else G

        X = G / (norm + 1e-30)

        for _ in range(steps):
            X = 1.5 * X - 0.5 * X @ X.T @ X

        # Rescale by original norm so the update has the same scale as the gradient
        X = X * norm

        if transposed:
            X = X.T

        return X

    def _clip_gradient(self, grad: torch.Tensor, clip_percentile: float) -> torch.Tensor:
        """Clip gradient by percentile of its absolute values.

        Computes threshold = percentile(|grad|, clip_percentile) and clips all
        values whose absolute value exceeds the threshold.

        Args:
            grad: Gradient tensor (any shape).
            clip_percentile: Percentile in [0, 100] used as the clipping threshold.

        Returns:
            Clipped gradient tensor of the same shape and dtype.
        """
        abs_grad = grad.abs().float()
        # torch.quantile expects a value in [0, 1]
        q = clip_percentile / 100.0
        threshold = torch.quantile(abs_grad.flatten(), q)
        # Clamp values to [-threshold, threshold]
        return grad.clamp(-threshold.to(grad.dtype), threshold.to(grad.dtype))

    @torch.no_grad()
    def step(self, closure=None) -> torch.Tensor | None:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            clip_percentile = group["clip_percentile"]
            wd = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # --- Step 1: per-parameter gradient clipping ---
                g = self._clip_gradient(p.grad.float(), clip_percentile)

                state = self.state[p]

                if p.ndim >= 2:
                    # ---- Matrix branch: Muon update ----
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)

                    buf = state["momentum_buffer"]

                    # Nesterov momentum: update buffer, look ahead
                    if nesterov:
                        buf.mul_(momentum).add_(g)
                        g_update = g.add(buf, alpha=momentum)
                    else:
                        buf.mul_(momentum).add_(g)
                        g_update = buf

                    # Orthogonalize the update
                    g_2d = g_update.view(g_update.shape[0], -1)
                    g_orth = self.newton_schulz_orthogonalize(g_2d, steps=ns_steps)
                    g_orth = g_orth.view_as(p)

                    # Scale to unit RMS so lr is interpretable across layer sizes
                    rms = g_orth.pow(2).mean().sqrt().clamp(min=1e-8)
                    g_orth = g_orth / rms

                    # Weight decay
                    if wd != 0.0:
                        p.mul_(1.0 - lr * wd)

                    p.add_(g_orth.to(p.dtype), alpha=-lr)

                else:
                    # ---- Vector / scalar branch: Adam-style update ----
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(g)
                        state["exp_avg_sq"] = torch.zeros_like(g)

                    state["step"] += 1
                    t = state["step"]
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                    # Bias correction
                    bias_corr1 = 1.0 - beta1 ** t
                    bias_corr2 = 1.0 - beta2 ** t
                    step_size = lr * math.sqrt(bias_corr2) / bias_corr1

                    denom = exp_avg_sq.sqrt().add_(eps)

                    # Weight decay
                    if wd != 0.0:
                        p.mul_(1.0 - lr * wd)

                    p.addcdiv_(exp_avg.to(p.dtype), denom.to(p.dtype), value=-step_size)

        return loss
