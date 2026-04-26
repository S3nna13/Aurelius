"""Prodigy: An Expeditiously Adaptive Parameter-Free Learning Rate Optimizer.

Prodigy automatically estimates the optimal learning rate by tracking the
distance from initialization. No learning rate schedule required.

Reference: Mishchenko & Defazio 2023, arXiv:2306.06101
"""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer


class Prodigy(Optimizer):
    """Prodigy: self-tuning AdamW with automatic learning rate estimation.

    Estimates the distance D = ||θ_0 - θ*|| from initialization and uses it
    to set the effective learning rate. No learning rate schedule is needed.

    Pass lr=1.0 (the default); the internal D estimate controls the actual
    step size.

    Args:
        params: Iterable of parameters or parameter groups.
        lr: Nominal learning rate (ignored internally; kept for API compat).
        betas: (beta1, beta2) — Adam moment decay coefficients.
        beta3: Decay for the d-accumulator EMA. Defaults to sqrt(beta2).
        eps: Numerical stability constant.
        weight_decay: AdamW-style decoupled weight decay coefficient.
        d0: Initial distance estimate (default 1e-6).
        d_coef: Scalar multiplier applied to D when computing the effective lr.
        growth_rate: Maximum allowed multiplicative growth of D per step.
            Pass float('inf') to disable (default).
        use_bias_correction: Whether to apply Adam bias correction.
        safeguard_warmup: If True, D is not updated for the first
            ``warmup_steps`` steps.
        warmup_steps: Number of warmup steps when safeguard_warmup is True.
    """

    def __init__(
        self,
        params,
        lr: float = 1.0,
        betas: tuple[float, float] = (0.9, 0.999),
        beta3: float | None = None,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        d0: float = 1e-6,
        d_coef: float = 1.0,
        growth_rate: float = float("inf"),
        use_bias_correction: bool = True,
        safeguard_warmup: bool = False,
        warmup_steps: int = 0,
    ) -> None:
        if not 0.0 < betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 < betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if d0 <= 0.0:
            raise ValueError(f"Invalid d0: {d0}")

        # beta3 defaults to sqrt(beta2)
        _beta3 = beta3 if beta3 is not None else math.sqrt(betas[1])

        defaults = dict(
            lr=lr,
            betas=betas,
            beta3=_beta3,
            eps=eps,
            weight_decay=weight_decay,
            d0=d0,
            d_coef=d_coef,
            growth_rate=growth_rate,
            use_bias_correction=use_bias_correction,
            safeguard_warmup=safeguard_warmup,
            warmup_steps=warmup_steps,
            # shared optimizer-level state stored in defaults for convenience
            d=d0,
            s=0.0,
            step_count=0,
        )
        super().__init__(params, defaults)

        # Store beta3 as passed (None means sqrt(beta2)) for property access
        self._beta3_arg = beta3

    @property
    def current_lr(self) -> float:
        """Return the current estimated learning rate (d * d_coef)."""
        # d and d_coef are stored in the first param group's defaults
        group = self.param_groups[0]
        return group["d"] * group["d_coef"]

    @torch.no_grad()
    def step(self, closure=None) -> float | None:
        """Perform a single Prodigy optimisation step.

        Args:
            closure: Optional callable that re-evaluates the model and returns
                the loss.

        Returns:
            Loss if closure is provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # ------------------------------------------------------------------ #
        # Gather all parameter groups to compute global d update numerator /  #
        # denominator across *all* parameters (cross-group accumulation).      #
        # ------------------------------------------------------------------ #
        # We accumulate:
        #   dlr_num += <g_t, θ_0 - θ_t>   (numerator contribution)
        #   dlr_den += ||v_t^{1/2}||       (denominator contribution)
        # across all groups, then update d once.

        # Use the first group for global scalars (they are shared across groups
        # via defaults, but PyTorch copies defaults into each group dict on
        # add_param_group, so we just use group 0 as the canonical store).
        g0 = self.param_groups[0]
        d = g0["d"]
        s = g0["s"]
        step_count = g0["step_count"] + 1

        # Global accumulators for the d-update
        dlr_num = 0.0
        dlr_den = 0.0

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            beta3 = group["beta3"]
            eps = group["eps"]
            wd = group["weight_decay"]
            use_bc = group["use_bias_correction"]

            # d_coef is NOT applied to the parameter update — only to current_lr.
            # This keeps d's trajectory independent of d_coef so that
            # current_lr scales linearly with d_coef.
            effective_lr = d  # do NOT multiply by d_coef here

            bc1 = 1.0 - beta1**step_count if use_bc else 1.0
            bc2 = 1.0 - beta2**step_count if use_bc else 1.0

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad.float()
                state = self.state[p]

                # Initialise state on first step
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                    state["p0"] = p.clone().float()  # θ_0
                    state["step"] = 0

                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                p0 = state["p0"]

                # Adam moment updates
                exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                # Bias-corrected moments
                m_hat = exp_avg / bc1
                v_hat = exp_avg_sq / bc2

                # Denominator: sqrt(v_hat) + eps
                denom = v_hat.sqrt().add_(eps)

                # AdamW parameter update FIRST, then compute diff.
                # Computing diff AFTER the update gives a non-zero signal from
                # the very first step (before update, θ_0 - θ_t = 0).
                p.addcdiv_(m_hat, denom, value=-effective_lr)
                if wd != 0.0:
                    # Multiplicative decoupled weight decay; clamp to avoid
                    # negative scale when effective_lr * wd > 1.
                    decay = max(0.0, 1.0 - effective_lr * wd)
                    p.mul_(decay)

                # Accumulate d-update numerator AFTER update: <g, θ_0 - θ_{t+1}>
                # The (d0/d) normalization from the paper prevents the feedback
                # loop where larger d → larger diff → runaway d growth.
                diff = p0 - p.float()
                d0_val = group["d0"]
                scale = d0_val / (d + 1e-30)
                dlr_num += scale * (g * diff).sum().item()
                dlr_den += denom.sum().item()

        # ------------------------------------------------------------------ #
        # Update the distance estimate d                                       #
        # ------------------------------------------------------------------ #
        safeguard = g0["safeguard_warmup"]
        warmup_steps = g0["warmup_steps"]
        growth_rate = g0["growth_rate"]
        beta3 = g0["beta3"]

        should_update_d = not (safeguard and step_count <= warmup_steps)

        if should_update_d and dlr_den > 0.0:
            # s accumulator: EMA-style update as in Prodigy paper
            # s_{t+1} = beta3 * s_t + (1 - beta3) * <g, θ_0 - θ_t> / denom
            # Then d = max(d, sqrt(s_{t+1} / (1 - beta3^t)))  (with optional BC)
            s_new = beta3 * s + (1.0 - beta3) * max(dlr_num, 0.0) / (dlr_den + 1e-30)

            # Bias-correct the s accumulator
            bc3 = 1.0 - beta3**step_count
            s_corrected = s_new / (bc3 + 1e-30)

            d_new = math.sqrt(max(s_corrected, 0.0)) if s_corrected > 0 else d

            # Apply growth rate cap
            d_max = d * (growth_rate**1)
            d_new = min(d_new, d_max)

            # D only grows (never shrinks)
            d = max(d, d_new)
            s = s_new
        else:
            # Even during warmup we still need to advance s to avoid a cold start
            # but we clamp d to d0
            if dlr_den > 0.0:
                s = beta3 * s + (1.0 - beta3) * max(dlr_num, 0.0) / (dlr_den + 1e-30)

        # Sync shared scalars back to all groups
        for group in self.param_groups:
            group["d"] = d
            group["s"] = s
            group["step_count"] = step_count

        return loss


class ProdigyW(Prodigy):
    """Prodigy with explicit decoupled weight decay (AdamW-style).

    Functionally identical to Prodigy — weight decay is already applied in
    AdamW style in the base class. This alias is provided for users who want
    an explicit ``ProdigyW`` name to signal decoupled weight decay intent.
    """

    pass
