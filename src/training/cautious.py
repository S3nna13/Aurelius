"""Cautious Optimizers: improving training with one line of code.

Applies a sign-agreement mask to the optimizer update: only update in
directions where the current gradient and the momentum-based update agree
in sign.  This prevents catastrophic parameter moves when the gradient has
reversed since the momentum was accumulated.

Reference: Zhu et al. 2024, arXiv:2411.16085 (Algorithm 1)

Variable notation follows the paper:
    g_t   — gradient at step t
    m_t   — first-moment (momentum) estimate
    v_t   — second-moment (variance) estimate
    m̂_t  — bias-corrected first moment
    v̂_t  — bias-corrected second moment
    d_t   — update direction (m̂_t / (sqrt(v̂_t) + ε) for Adam)
    mask_t — sign-agreement mask
    θ_t   — parameter after update
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.optim import Optimizer

# ---------------------------------------------------------------------------
# Shared masking logic (Algorithm 1, paper §3)
# ---------------------------------------------------------------------------


class CautiousMask:
    """Stateless helper that applies the cautious sign-agreement mask.

    Given update direction d_t and current gradient g_t:
        mask_t = (d_t * g_t > 0).float()
        mask_t = mask_t / (mask_t.mean() + ε)   # scale-preserving normalisation
        cautious_update = mask_t * d_t
    """

    _EPS: float = 1e-8  # ε for normalisation denominator

    @staticmethod
    def apply(d_t: Tensor, g_t: Tensor) -> Tensor:
        """Return the cautiously-masked update.

        Args:
            d_t: Update direction tensor (same shape as g_t).
            g_t: Current raw gradient tensor.

        Returns:
            Masked update of the same shape.  Elements where d_t and g_t
            disagree in sign are zeroed out; remaining elements are
            rescaled so that the mean absolute magnitude is preserved.
        """
        mask_t = (d_t * g_t > 0).to(dtype=d_t.dtype)
        mean_mask = mask_t.mean()
        mask_t = mask_t / (mean_mask + CautiousMask._EPS)
        return mask_t * d_t


# ---------------------------------------------------------------------------
# CautiousAdam — AdamW + cautious mask
# ---------------------------------------------------------------------------


class CautiousAdam(Optimizer):
    """AdamW with cautious update mask (C-AdamW).

    Identical to AdamW except that before applying the parameter update the
    sign-agreement mask is applied to d_t = m̂_t / (sqrt(v̂_t) + ε).

    Args:
        params:       Iterable of parameters or param groups.
        lr:           Learning rate (α).  Default: 1e-3.
        betas:        (β₁, β₂) for first- and second-moment EMA.
                      Default: (0.9, 0.999).
        eps:          ε added to denominator for numerical stability.
                      Default: 1e-8.
        weight_decay: Decoupled weight-decay coefficient (λ).  Default: 0.01.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if not lr >= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if not eps > 0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not weight_decay >= 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one CautiousAdam optimisation step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g_t = p.grad
                if g_t.is_sparse:
                    raise RuntimeError("CautiousAdam does not support sparse gradients")

                state = self.state[p]

                # Initialise state on first step
                if len(state) == 0:
                    state["t"] = 0
                    state["m_t"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["v_t"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                m_t = state["m_t"]
                v_t = state["v_t"]
                state["t"] += 1
                t = state["t"]

                # Decoupled weight decay (AdamW style)
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # Moment updates
                m_t.mul_(beta1).add_(g_t, alpha=1.0 - beta1)  # m_t = β₁·m_{t-1} + (1−β₁)·g_t
                v_t.mul_(beta2).addcmul_(
                    g_t, g_t, value=1.0 - beta2
                )  # v_t = β₂·v_{t-1} + (1−β₂)·g_t²

                # Bias correction
                bc1 = 1.0 - beta1**t
                bc2 = 1.0 - beta2**t
                m_hat = m_t / bc1  # m̂_t
                v_hat = v_t / bc2  # v̂_t

                # Adam update direction  d_t = m̂_t / (sqrt(v̂_t) + ε)
                d_t = m_hat / (v_hat.sqrt().add_(eps))

                # *** Cautious mask (the "one line") ***
                masked_d_t = CautiousMask.apply(d_t, g_t)

                # θ_t = θ_{t-1} - lr · mask_t · d_t
                p.add_(masked_d_t, alpha=-lr)

        return loss


# ---------------------------------------------------------------------------
# CautiousSGD — SGD with momentum + cautious mask
# ---------------------------------------------------------------------------


class CautiousSGD(Optimizer):
    """SGD with momentum and cautious update mask (C-SGD).

    The momentum buffer plays the role of d_t; the sign-agreement mask
    is applied against the current gradient g_t before the parameter update.

    Args:
        params:       Iterable of parameters or param groups.
        lr:           Learning rate.  Default: 0.01.
        momentum:     Momentum factor (μ).  Default: 0.9.
        weight_decay: L2 penalty coefficient.  Default: 0.0.
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ) -> None:
        if not lr >= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if not weight_decay >= 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one CautiousSGD optimisation step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g_t = p.grad

                # Optional L2 weight decay added to gradient
                if wd != 0.0:
                    g_t = g_t.add(p, alpha=wd)

                state = self.state[p]

                if len(state) == 0:
                    # First step: initialise momentum buffer with current gradient
                    state["buf"] = g_t.clone()
                    d_t = state["buf"]
                else:
                    buf = state["buf"]
                    buf.mul_(mu).add_(g_t, alpha=1.0 - mu)  # buf = μ·buf + (1−μ)·g_t
                    d_t = buf

                # *** Cautious mask ***
                masked_d_t = CautiousMask.apply(d_t, g_t)

                # θ_t = θ_{t-1} - lr · mask_t · d_t
                p.add_(masked_d_t, alpha=-lr)

        return loss


# ---------------------------------------------------------------------------
# make_cautious factory
# ---------------------------------------------------------------------------


def make_cautious(optimizer_class: type) -> type:
    """Return a new optimizer class that wraps *optimizer_class* with the
    cautious sign-agreement mask.

    The wrapped class intercepts each parameter update, extracts the update
    direction from the gradient attribute, and applies CautiousMask.apply
    before calling the parent step logic.

    Note: This factory works for optimizers that apply updates via
    ``p.add_(update)`` or equivalent in a for-loop over param groups and
    only uses ``p.grad`` as input.  For complex optimizers with fused kernels
    you should subclass directly (as CautiousAdam does).

    Args:
        optimizer_class: A subclass of ``torch.optim.Optimizer``.

    Returns:
        A new class named ``Cautious{optimizer_class.__name__}``.
    """

    class CautiousWrapper(optimizer_class):  # type: ignore[valid-type]
        """Auto-wrapped cautious version of {base}."""

        @torch.no_grad()
        def step(self, closure=None):
            # Snapshot params before the base step so we can compute d_t
            snapshots: dict[int, Tensor] = {}
            grads: dict[int, Tensor] = {}
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        snapshots[id(p)] = p.data.clone()
                        grads[id(p)] = p.grad.clone()

            loss = super().step(closure)

            # Compute update direction, apply cautious mask, reapply
            for group in self.param_groups:
                group.get("lr", 1.0)
                for p in group["params"]:
                    pid = id(p)
                    if pid not in snapshots:
                        continue
                    # d_t extracted from the raw update applied by the base optimizer
                    raw_update = p.data - snapshots[pid]  # θ_new − θ_old  (= −lr·d_t)
                    if raw_update.abs().max() == 0:
                        continue
                    g_t = grads[pid]
                    # sign-agreement mask on d_t direction (raw_update is −lr·d_t)
                    mask_t = (raw_update * g_t < 0).to(
                        dtype=p.dtype
                    )  # agree → update·g < 0 (opposite sign)
                    mean_mask = mask_t.mean()
                    mask_t = mask_t / (mean_mask + 1e-8)
                    # Revert to pre-step and apply masked update
                    p.data.copy_(snapshots[pid] + mask_t * raw_update)

            return loss

    CautiousWrapper.__name__ = f"Cautious{optimizer_class.__name__}"
    CautiousWrapper.__qualname__ = f"Cautious{optimizer_class.__name__}"
    CautiousWrapper.__doc__ = (CautiousWrapper.__doc__ or "").replace(
        "{base}", optimizer_class.__name__
    )
    return CautiousWrapper
