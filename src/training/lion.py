"""Lion optimizer: Sign of momentum for memory-efficient optimization.

Uses sign of EMA-interpolated gradient rather than gradient magnitude.
Only one momentum buffer vs Adam's two — more memory efficient.

Reference: Chen et al. 2023, arXiv:2302.06675
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer


class Lion(Optimizer):
    """Lion optimizer.

    Args:
        params: Model parameters
        lr: Learning rate (typically 1e-4 to 3e-4, ~3-10x smaller than Adam lr)
        betas: (beta1, beta2) — (0.9, 0.99) recommended
        weight_decay: L2 regularization coefficient
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
        """Perform one Lion optimization step."""
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

                # Get/initialize momentum buffer
                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                m = state["m"]

                # Compute update direction: sign of β1*m + (1-β1)*g
                update = (beta1 * m + (1 - beta1) * g).sign_()

                # Apply weight decay + sign update
                p.add_(update + wd * p, alpha=-lr)

                # Update momentum: β2*m + (1-β2)*g
                m.mul_(beta2).add_(g, alpha=1 - beta2)

        return loss


class LionW(Lion):
    """Lion with decoupled weight decay (LionW variant).

    Weight decay is applied directly to the parameter (decoupled), not through
    the gradient. This is the standard recommended variant with a slightly
    higher default learning rate.

    Args:
        params: Model parameters
        lr: Learning rate (default 3e-4)
        betas: (beta1, beta2) momentum coefficients (default (0.9, 0.99))
        weight_decay: Decoupled weight decay (default 1e-2)
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 1e-2,
    ) -> None:
        super().__init__(params, lr=lr, betas=betas, weight_decay=weight_decay)


class Lion8bit(Lion):
    """Memory-efficient Lion with 8-bit momentum quantization.

    Quantizes the momentum buffer to int8 to halve optimizer memory compared
    to standard Lion (which already uses half the memory of Adam).

    Quantization: scale = max(|m|) / 127; m_q = round(m / scale).to(int8)
    Dequantization: m = m_q.float() * scale

    Args:
        params: Model parameters
        lr: Learning rate (default 1e-4)
        betas: (beta1, beta2) momentum coefficients (default (0.9, 0.99))
        weight_decay: Decoupled weight decay (default 0.0)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr=lr, betas=betas, weight_decay=weight_decay)

    def _quantize_momentum(self, m: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Quantize momentum tensor to int8.

        Returns:
            (quantized_int8, scale) where scale = max(|m|) / 127
        """
        abs_max = m.abs().max().item()
        if abs_max == 0.0:
            scale = 1.0
        else:
            scale = abs_max / 127.0
        m_q = (m / scale).round().clamp(-127, 127).to(torch.int8)
        return m_q, scale

    def _dequantize_momentum(self, m_q: torch.Tensor, scale: float) -> torch.Tensor:
        """Dequantize int8 momentum back to float32.

        Args:
            m_q: Quantized int8 momentum tensor
            scale: Scale factor used during quantization

        Returns:
            Float32 momentum tensor
        """
        return m_q.float() * scale

    @torch.no_grad()
    def step(self, closure=None) -> float | None:
        """Perform one Lion8bit optimization step with quantized momentum."""
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

                # Initialize or dequantize momentum
                if "exp_avg_q" not in state:
                    # First step: start from zero float momentum
                    m = torch.zeros_like(p)
                else:
                    m = self._dequantize_momentum(state["exp_avg_q"], state["exp_avg_scale"])

                # Compute update direction: sign of β1*m + (1-β1)*g
                update = (beta1 * m + (1 - beta1) * g).sign_()

                # Apply weight decay then sign update
                p.add_(update + wd * p, alpha=-lr)

                # Update momentum: β2*m + (1-β2)*g
                m.mul_(beta2).add_(g, alpha=1 - beta2)

                # Quantize and store updated momentum
                m_q, scale = self._quantize_momentum(m)
                state["exp_avg_q"] = m_q
                state["exp_avg_scale"] = scale

        return loss


def compare_lion_adam_memory(n_params: int) -> dict:
    """Compute optimizer memory comparison between Adam, Lion, and Lion8bit.

    Adam stores 2 float32 buffers (exp_avg + exp_avg_sq) per parameter.
    Lion stores 1 float32 buffer (exp_avg / momentum) per parameter.
    Lion8bit stores 1 int8 buffer (1 byte) plus a per-tensor float scale.

    Args:
        n_params: Total number of scalar parameters

    Returns:
        dict with keys:
            'adam_buffers_mb': float  — Adam: 2 buffers * 4 bytes each
            'lion_buffers_mb': float  — Lion: 1 buffer * 4 bytes
            'lion8bit_buffers_mb': float — Lion8bit: 1 buffer * 1 byte + scales
            'lion_vs_adam_ratio': float — lion / adam (~0.5)
    """
    bytes_per_float32 = 4
    bytes_per_int8 = 1

    adam_bytes = n_params * 2 * bytes_per_float32
    lion_bytes = n_params * 1 * bytes_per_float32
    # Lion8bit: int8 per element + one float32 scale per tensor (approximated
    # as negligible relative to parameter count, so we just count the int8 buf)
    lion8bit_bytes = n_params * bytes_per_int8

    mb = 1024 * 1024

    return {
        "adam_buffers_mb": adam_bytes / mb,
        "lion_buffers_mb": lion_bytes / mb,
        "lion8bit_buffers_mb": lion8bit_bytes / mb,
        "lion_vs_adam_ratio": lion_bytes / adam_bytes,
    }
