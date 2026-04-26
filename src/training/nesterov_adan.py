"""NesterovAdan — Adan: Adaptive Nesterov Momentum Algorithm.

Clean, standalone implementation matching the paper's variable notation
exactly. Suitable as a drop-in optimizer with proper Nesterov momentum
fused with adaptive gradient updates.

Reference:
    Xie et al., "Adan: Adaptive Nesterov Momentum Algorithm for Faster
    Optimizing Deep Models", arXiv:2208.06677.

Paper notation used throughout:
    g_t      — gradient at step t
    m_t      — 1st moment (EMA of gradients),          β1
    v_t      — 2nd moment (EMA of gradient diffs),     β2
    n_t      — variance estimate (EMA of Nesterov²),   β3
    λ        — weight-decay coefficient
    α        — learning rate

Algorithm (paper, Algorithm 1):
    m_t = (1 - β1) * g_t  +  β1 * m_{t-1}
    v_t = (1 - β2) * (g_t - g_{t-1})  +  β2 * v_{t-1}
    n_t = (1 - β3) * [g_t + (1 - β2) * (g_t - g_{t-1})]²  +  β3 * n_{t-1}

    θ_t = θ_{t-1} / (1 + λ·α)
          - α / (1 + λ·α) · (m̂_t + (1 - β2) · v̂_t) / (√n̂_t + ε)

where m̂_t, v̂_t, n̂_t are bias-corrected estimates and on the first step
g_{t-1} is set to g_1 so that the gradient difference is zero.

When no_prox=True the weight decay is applied additively (AdamW-style)
instead of using the proximal formulation above.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from torch.optim import Optimizer


class NesterovAdan(Optimizer):
    """Adan optimizer with Nesterov momentum (paper-accurate, standalone).

    Maintains three moment buffers per parameter:

    * ``exp_avg``      (m_t) — EMA of gradients (β1)
    * ``exp_avg_diff`` (v_t) — EMA of gradient differences (β2)
    * ``exp_avg_sq``   (n_t) — EMA of squared Nesterov estimates (β3)

    An additional ``prev_grad`` buffer stores g_{t-1} for the gradient
    difference computation used in v_t and n_t.

    Args:
        params:       Iterable of parameters or param groups.
        lr (α):       Learning rate. Default: ``1e-3``.
        betas:        ``(β1, β2, β3)`` decay coefficients. Default:
                      ``(0.98, 0.92, 0.99)``.
        eps (ε):      Denominator stability term. Default: ``1e-8``.
        weight_decay (λ):
                      Decoupled weight-decay coefficient. Default: ``0.02``.
        no_prox:      If ``True``, apply weight decay additively (AdamW
                      style, ``θ ← θ - α·λ·θ``) instead of the proximal
                      formulation from the paper. Default: ``False``.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: tuple[float, float, float] = (0.98, 0.92, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.02,
        no_prox: bool = False,
    ) -> None:
        if not lr >= 0.0:
            raise ValueError(f"Invalid learning rate: {lr!r}")
        if not eps >= 0.0:
            raise ValueError(f"Invalid epsilon: {eps!r}")
        if len(betas) != 3 or not all(0.0 <= b < 1.0 for b in betas):
            raise ValueError(f"Invalid betas (each must be in [0, 1)): {betas!r}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay!r}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            no_prox=no_prox,
        )
        super().__init__(params, defaults)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_state(self, p: torch.Tensor) -> None:
        """Initialise per-parameter state buffers on the first step."""
        state = self.state[p]
        state["step"] = 0
        # m_t — 1st moment (EMA of g_t)
        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        # v_t — 2nd moment (EMA of gradient differences)
        state["exp_avg_diff"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        # n_t — variance estimate (EMA of Nesterov²)
        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        # g_{t-1}: set to g_1 on first step (difference = 0 on step 1)
        state["prev_grad"] = None  # filled at end of first step

    # ------------------------------------------------------------------
    # Optimizer step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> float | None:
        """Perform a single Adan optimisation step.

        Args:
            closure: Optional closure that re-evaluates the model and
                     returns the loss.

        Returns:
            The loss returned by *closure*, or ``None``.
        """
        loss: float | None = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # ---- hyper-parameters for this group ----------------------
            α: float = group["lr"]
            β1, β2, β3 = group["betas"]
            ε: float = group["eps"]
            λ: float = group["weight_decay"]
            no_prox: bool = group["no_prox"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # g_t: current gradient (detached, flat view is fine)
                g_t: torch.Tensor = p.grad.detach()

                state = self.state[p]

                # ---- lazy state initialisation ------------------------
                if len(state) == 0:
                    self._init_state(p)

                t: int = state["step"] + 1
                state["step"] = t

                m_t: torch.Tensor = state["exp_avg"]  # β1 moment
                v_t: torch.Tensor = state["exp_avg_diff"]  # β2 moment
                n_t: torch.Tensor = state["exp_avg_sq"]  # β3 moment

                # g_{t-1}: on the very first step treat it as g_1 → diff = 0
                if state["prev_grad"] is None:
                    g_prev = g_t.clone()
                else:
                    g_prev = state["prev_grad"]

                # ---- gradient difference ------------------------------
                # Δg_t = g_t - g_{t-1}
                Δg_t = g_t - g_prev  # zero on step 1 by construction

                # ---- Nesterov composite gradient  ---------------------
                # k_t = g_t + (1 - β2) * Δg_t   (used for n_t)
                k_t = g_t + (1.0 - β2) * Δg_t

                # ---- moment updates (paper Algorithm 1) ---------------
                # m_t = (1 - β1) * g_t  +  β1 * m_{t-1}
                m_t.mul_(β1).add_(g_t, alpha=1.0 - β1)

                # v_t = (1 - β2) * Δg_t  +  β2 * v_{t-1}
                v_t.mul_(β2).add_(Δg_t, alpha=1.0 - β2)

                # n_t = (1 - β3) * k_t²  +  β3 * n_{t-1}
                n_t.mul_(β3).addcmul_(k_t, k_t, value=1.0 - β3)

                # ---- bias corrections ---------------------------------
                bc1: float = 1.0 - β1**t  # 1 - β1^t
                bc2: float = 1.0 - β2**t  # 1 - β2^t
                bc3: float = 1.0 - β3**t  # 1 - β3^t

                # m̂_t = m_t / bc1
                m_hat = m_t / bc1
                # v̂_t = v_t / bc2
                v_hat = v_t / bc2
                # n̂_t = n_t / bc3
                n_hat = n_t / bc3

                # ---- adaptive denominator ----------------------------
                # denom = √n̂_t + ε
                denom = n_hat.sqrt().add_(ε)

                # ---- Nesterov update direction ------------------------
                # update = m̂_t + (1 - β2) * v̂_t
                update = m_hat.add(v_hat, alpha=1.0 - β2)

                # ---- parameter update with weight decay ---------------
                if no_prox:
                    # AdamW-style: θ ← θ - α·λ·θ - α·update/denom
                    # Equivalent to adding wd*p to the gradient direction.
                    if λ != 0.0:
                        p.mul_(1.0 - α * λ)
                    p.addcdiv_(update, denom, value=-α)
                else:
                    # Proximal (paper default):
                    # θ_t = θ_{t-1} / (1 + λ·α)
                    #        - α / (1 + λ·α) · update / denom
                    prox_denom = 1.0 + λ * α
                    p.div_(prox_denom)
                    p.addcdiv_(update, denom, value=-(α / prox_denom))

                # ---- store g_t for next step --------------------------
                state["prev_grad"] = g_t.clone()

        return loss

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_lr(self) -> list[float]:
        """Return the current learning rate for each param group."""
        return [g["lr"] for g in self.param_groups]

    def zero_moments(self) -> None:
        """Reset all moment buffers and step counters (warm-restart support)."""
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if not state:
                    continue
                state["step"] = 0
                state["exp_avg"].zero_()
                state["exp_avg_diff"].zero_()
                state["exp_avg_sq"].zero_()
                state["prev_grad"] = None
