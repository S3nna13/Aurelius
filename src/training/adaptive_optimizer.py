"""Adaptive optimizers: Lion, SignSGD, and a simplified SOAP preconditioner."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer


@dataclass
class OptimizerConfig:
    """Configuration for adaptive optimizers."""

    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 0.0
    clip_threshold: float = 1.0
    precondition_freq: int = 10


class LionOptimizer(Optimizer):
    """Lion optimizer (Evolved Sign Momentum, Chen et al. 2023).

    Uses the sign of an EMA-interpolated gradient as the update direction.
    More memory-efficient than Adam: only one momentum buffer per parameter.

    Reference: arXiv:2302.06675

    Args:
        params: Model parameters or param groups.
        lr: Learning rate.
        betas: (beta1, beta2) for update interpolation and momentum EMA.
        weight_decay: L2 regularization coefficient.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> float | None:
        """Perform one Lion optimization step.

        For each parameter:
          update = sign(beta1 * m + (1-beta1) * grad)
          param  -= lr * (update + weight_decay * param)
          m       = beta2 * m + (1-beta2) * grad
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                m = state["m"]

                # Compute signed update: sign(beta1 * m + (1-beta1) * g)
                update = (beta1 * m + (1.0 - beta1) * g).sign_()

                # Apply: param -= lr * (update + wd * param)
                p.add_(update + wd * p, alpha=-lr)

                # Update momentum: m = beta2 * m + (1-beta2) * g
                m.mul_(beta2).add_(g, alpha=1.0 - beta2)

        return loss


class SignSGD(Optimizer):
    """SignSGD with momentum (approximation to 1-bit Adam).

    Accumulates momentum from gradients, then applies the sign of that
    momentum as the parameter update.

    Args:
        params: Model parameters or param groups.
        lr: Learning rate.
        momentum: Momentum decay coefficient.
        weight_decay: L2 regularization coefficient.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> float | None:
        """Perform one SignSGD step.

        For each parameter:
          m     = momentum * m + grad
          param -= lr * sign(m) + weight_decay * param
        """
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

                g = p.grad
                state = self.state[p]

                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                m = state["m"]

                # Accumulate momentum
                m.mul_(mu).add_(g)

                # Update: param -= lr * sign(m) + wd * param
                p.add_(m.sign() + wd * p, alpha=-lr)

        return loss


def compute_gradient_norm(params: list[nn.Parameter]) -> float:
    """Compute global gradient L2 norm across all parameters.

    Args:
        params: List of parameters with `.grad` attributes.

    Returns:
        Global L2 norm as a Python float.
    """
    total_sq = 0.0
    for p in params:
        if p.grad is not None:
            total_sq += p.grad.detach().float().norm(2).item() ** 2
    return math.sqrt(total_sq)


def clip_gradients_by_norm(
    params: list[nn.Parameter], max_norm: float = 1.0
) -> float:
    """Clip gradients in-place so the global L2 norm does not exceed max_norm.

    Args:
        params: List of parameters whose gradients will be clipped.
        max_norm: Maximum allowed global gradient norm.

    Returns:
        The gradient norm *before* clipping.
    """
    total_norm = compute_gradient_norm(params)
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for p in params:
            if p.grad is not None:
                p.grad.detach().mul_(scale)
    return total_norm


class GradientPreconditioner:
    """Simplified diagonal preconditioner (approximation of Shampoo).

    Maintains a running squared-gradient accumulator per parameter and
    uses it to precondition gradients with ``grad / sqrt(v2 + eps)``.

    Args:
        params: List of model parameters to track.
        epsilon: Numerical stability constant.
    """

    def __init__(self, params: list[nn.Parameter], epsilon: float = 1e-8) -> None:
        self._params = list(params)
        self._epsilon = epsilon
        # Running squared gradient per parameter (keyed by param id)
        self._v2: dict[int, Tensor] = {}

    def update(self, step: int, freq: int = 10) -> None:
        """Update the diagonal preconditioner.

        Every ``freq`` steps the squared-gradient accumulator is refreshed
        with the current gradient.

        Args:
            step: Current training step (1-indexed).
            freq: Frequency at which to update.
        """
        if step % freq != 0:
            return
        for p in self._params:
            if p.grad is None:
                continue
            pid = id(p)
            g2 = p.grad.detach().float().pow(2)
            if pid not in self._v2:
                self._v2[pid] = g2
            else:
                self._v2[pid] = self._v2[pid] * 0.9 + g2 * 0.1

    def precondition(self, param: nn.Parameter) -> Tensor | None:
        """Return a preconditioned gradient estimate for *param*.

        Args:
            param: The parameter whose gradient should be preconditioned.

        Returns:
            ``grad / sqrt(v2 + eps)`` if squared-gradient stats exist,
            otherwise ``None``.
        """
        pid = id(param)
        if pid not in self._v2 or param.grad is None:
            return None
        v2 = self._v2[pid].to(param.grad.device)
        return param.grad.float() / (v2 + self._epsilon).sqrt()


class SOAPOptimizer(Optimizer):
    """Simplified SOAP: Adam with a periodic diagonal preconditioner.

    Combines standard Adam first/second moment updates with a diagonal
    preconditioning step that is refreshed every ``precondition_freq`` steps.

    Args:
        params: Model parameters or param groups.
        config: :class:`OptimizerConfig` with hyperparameters.
    """

    def __init__(self, params, config: OptimizerConfig) -> None:
        defaults = dict(
            lr=config.lr,
            betas=config.betas,
            weight_decay=config.weight_decay,
            clip_threshold=config.clip_threshold,
            precondition_freq=config.precondition_freq,
        )
        super().__init__(params, defaults)
        # Collect all parameters for the preconditioner
        all_params: list[nn.Parameter] = []
        for group in self.param_groups:
            all_params.extend(group["params"])
        self._preconditioner = GradientPreconditioner(all_params)
        self._step_count = 0

    @torch.no_grad()
    def step(self, closure=None) -> float | None:
        """Perform one SOAP step.

        For each parameter:
          - Compute Adam first moment m1 and second moment v.
          - Every precondition_freq steps, refresh the diagonal preconditioner.
          - Apply: param -= lr * (m1 / (sqrt(v) + eps))
          - Apply weight decay.

        Returns:
            Loss evaluated by closure if provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1
        t = self._step_count
        eps = 1e-8

        # Update preconditioner (runs gradient-based update at freq intervals)
        for group in self.param_groups:
            freq = group["precondition_freq"]
            break
        self._preconditioner.update(t, freq=freq)

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            freq = group["precondition_freq"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.float()
                state = self.state[p]

                # Initialise state
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)

                m1 = state["exp_avg"]
                v = state["exp_avg_sq"]

                # Adam moment updates
                m1.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Bias correction
                bc1 = 1.0 - beta1 ** t
                bc2 = 1.0 - beta2 ** t
                m1_hat = m1 / bc1

                # Use preconditioned v if available, otherwise standard Adam v
                precond_grad = self._preconditioner.precondition(p)
                if precond_grad is not None:
                    # Blend preconditioned second moment estimate
                    v_eff = (v / bc2 + precond_grad.pow(2)) * 0.5
                else:
                    v_eff = v / bc2

                update = m1_hat / (v_eff.sqrt().add_(eps))

                # Apply update
                p.add_(update.to(p.dtype), alpha=-lr)

                # Weight decay
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

        return loss


def benchmark_optimizer_convergence(
    model: nn.Module,
    optimizer,
    n_steps: int = 50,
) -> dict[str, float]:
    """Benchmark an optimizer on a simple toy task.

    Runs the optimizer for *n_steps* minimizing the sum of the model's output
    on a fixed random input.

    Args:
        model: Any ``nn.Module`` (used as the toy loss landscape).
        optimizer: A PyTorch-compatible optimizer already attached to
            *model*'s parameters.
        n_steps: Number of gradient steps to take.

    Returns:
        Dictionary with keys:
          - ``"initial_loss"``: loss before any update.
          - ``"final_loss"``: loss after *n_steps* updates.
          - ``"convergence_ratio"``: ``initial_loss / max(final_loss, 1e-8)``.
    """
    model.train()
    # Fixed random input
    with torch.no_grad():
        x = torch.randn(8, next(model.parameters()).shape[-1])

    initial_loss: float | None = None
    final_loss: float = 0.0

    for step in range(n_steps):
        optimizer.zero_grad()
        out = model(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()

        val = loss.item()
        if step == 0:
            initial_loss = val
        final_loss = val

    if initial_loss is None:
        initial_loss = final_loss

    convergence_ratio = abs(initial_loss) / max(abs(final_loss), 1e-8)

    return {
        "initial_loss": float(initial_loss),
        "final_loss": float(final_loss),
        "convergence_ratio": float(convergence_ratio),
    }
