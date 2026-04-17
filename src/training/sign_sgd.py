"""SignSGD and 1-bit Adam optimizers with gradient compression.

Implements two compressed-communication optimizers:

1. **SignSGD / MSignSGD** (Bernstein et al., 2018, arXiv:1802.04434)
   Update rule (plain):
       θ_{t+1} = θ_t - lr * sign(g_t)
   Momentum variant (MSignSGD):
       m_t = β * m_{t-1} + (1 - β) * g_t
       θ_{t+1} = θ_t - lr * sign(m_t)
   When momentum=0 the momentum buffer is omitted and plain SignSGD is used.

2. **1-bit Adam** (Tang et al., 2021, arXiv:2102.02888, Microsoft)
   Warm-up phase (step < T_warmup): standard Adam with full m_t and v_t.
   Compression phase (step ≥ T_warmup): v_t is frozen; m_t is 1-bit compressed
   with error feedback:
       e_{t-1}         = residual carried from last step
       m_t_corrected   = m_t + e_{t-1}
       α               = ||m_t_corrected||_1 / d       (d = number of elements)
       q_t             = α * sign(m_t_corrected)
       e_t             = m_t_corrected - q_t
       θ_{t+1}         = θ_t - lr * q_t / sqrt(v_frozen + ε)
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
from torch.optim import Optimizer


# ---------------------------------------------------------------------------
# SignSGD / MSignSGD
# ---------------------------------------------------------------------------


class SignSGD(Optimizer):
    """Compressed SGD using the sign of (optionally momentum-smoothed) gradients.

    When ``momentum > 0`` this implements MSignSGD (Algorithm 2, §3 of
    arXiv:1802.04434).  When ``momentum == 0`` it degenerates to plain SignSGD
    (Algorithm 1) with no state stored.

    Args:
        params: Iterable of parameters or param groups.
        lr: Learning rate η (default: 0.01).
        momentum: EMA coefficient β ∈ [0, 1) for the momentum buffer
            (default: 0.9).  Set to 0 for plain SignSGD with no state.
        weight_decay: L2 penalty coefficient λ (default: 0.0).  The gradient
            is augmented as  g̃_t = g_t + λ * θ_t  before the sign is taken.
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        """Perform a single SignSGD / MSignSGD step.

        Args:
            closure: Optional callable that re-evaluates the model and returns
                the loss.

        Returns:
            Loss returned by ``closure``, or ``None``.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            beta: float = group["momentum"]
            weight_decay: float = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g_t = p.grad
                if weight_decay != 0.0:
                    # g̃_t = g_t + λ θ_t
                    g_t = g_t.add(p, alpha=weight_decay)

                if beta == 0.0:
                    # Plain SignSGD — no state needed.
                    p.add_(g_t.sign(), alpha=-lr)
                else:
                    # MSignSGD — maintain EMA momentum buffer m_t.
                    state = self.state[p]
                    if len(state) == 0:
                        # m_0 = g_1  (initialise with first gradient)
                        state["m"] = g_t.clone()
                    else:
                        # m_t = β * m_{t-1} + (1 - β) * g_t
                        state["m"].mul_(beta).add_(g_t, alpha=1.0 - beta)

                    p.add_(state["m"].sign(), alpha=-lr)

        return loss


# ---------------------------------------------------------------------------
# 1-bit Adam
# ---------------------------------------------------------------------------


class OneBitAdam(Optimizer):
    """1-bit Adam with warm-up and error-feedback compression.

    Implements Algorithm 1 from Tang et al. 2021 (arXiv:2102.02888).

    Phases:
        **Warm-up** (``step_count < warmup_steps``):
            Standard Adam with bias-corrected m_t and v_t accumulation.
        **Compression** (``step_count ≥ warmup_steps``):
            v_t is frozen at its warm-up value.  m_t is 1-bit quantised with
            error feedback, yielding a compressed update q_t of the form
            ``α * sign(m_t_corrected)`` where
            ``α = ||m_t_corrected||_1 / d``.

    Args:
        params: Iterable of parameters or param groups.
        lr: Learning rate (default: 1e-3).
        betas: ``(β_1, β_2)`` EMA coefficients for m and v (default:
            (0.9, 0.999)).
        eps: Numerical stability term ε added to sqrt(v) (default: 1e-8).
        warmup_steps: Number of warm-up steps T_warmup before v is frozen and
            compression begins (default: 100).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        warmup_steps: int = 100,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= betas[0] < 1.0):
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not (0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if warmup_steps < 0:
            raise ValueError(f"Invalid warmup_steps: {warmup_steps}")

        defaults = dict(lr=lr, betas=betas, eps=eps, warmup_steps=warmup_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        """Perform a single 1-bit Adam step.

        Args:
            closure: Optional callable that re-evaluates the model and returns
                the loss.

        Returns:
            Loss returned by ``closure``, or ``None``.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            beta1, beta2 = group["betas"]
            eps: float = group["eps"]
            warmup_steps: int = group["warmup_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g_t = p.grad
                state = self.state[p]

                # ----------------------------------------------------------
                # Initialise state on first step.
                # ----------------------------------------------------------
                if len(state) == 0:
                    state["step"] = 0
                    # m_0 = 0  (first moment)
                    state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # v_0 = 0  (second moment)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # e_0 = 0  (error-feedback residual, used in compression phase)
                    state["e"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # v_frozen is None until the warm-up ends
                    state["v_frozen"] = None

                state["step"] += 1
                t: int = state["step"]

                m = state["m"]
                v = state["v"]

                # ----------------------------------------------------------
                # Update first moment: m_t = β_1 * m_{t-1} + (1 - β_1) * g_t
                # (done in both phases)
                # ----------------------------------------------------------
                m.mul_(beta1).add_(g_t, alpha=1.0 - beta1)

                if t <= warmup_steps:
                    # -------------------------------------------------------
                    # Warm-up phase — standard Adam
                    # -------------------------------------------------------
                    # v_t = β_2 * v_{t-1} + (1 - β_2) * g_t²
                    v.mul_(beta2).addcmul_(g_t, g_t, value=1.0 - beta2)

                    # Bias correction
                    bc1 = 1.0 - beta1 ** t
                    bc2 = 1.0 - beta2 ** t
                    m_hat = m / bc1
                    v_hat = v / bc2

                    # θ_{t+1} = θ_t - lr * m̂_t / (sqrt(v̂_t) + ε)
                    denom = v_hat.sqrt().add_(eps)
                    p.addcdiv_(m_hat, denom, value=-lr)

                    # Freeze v at the last warm-up step so compression can use it.
                    if t == warmup_steps:
                        state["v_frozen"] = v.clone()

                else:
                    # -------------------------------------------------------
                    # Compression phase — 1-bit Adam
                    # v is frozen; m_t is 1-bit quantised with error feedback.
                    # -------------------------------------------------------
                    if state["v_frozen"] is None:
                        # warmup_steps == 0: freeze immediately on first step.
                        state["v_frozen"] = v.clone()

                    v_frozen = state["v_frozen"]
                    e_prev = state["e"]

                    # Step 1 & 2: m_t_corrected = m_t + e_{t-1}
                    m_corrected = m.add(e_prev)

                    # Step 3: scale  α = ||m_t_corrected||_1 / d
                    d = m_corrected.numel()
                    alpha = m_corrected.abs().sum() / d  # scalar

                    # Step 3: q_t = α * sign(m_t_corrected)
                    q_t = m_corrected.sign().mul_(alpha)

                    # Step 4: e_t = m_t_corrected - q_t  (new residual)
                    state["e"] = m_corrected - q_t

                    # Step 5: θ_{t+1} = θ_t - lr * q_t / sqrt(v_frozen + ε)
                    denom = v_frozen.sqrt().add_(eps)
                    p.addcdiv_(q_t, denom, value=-lr)

        return loss
