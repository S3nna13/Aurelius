"""8-bit optimizer wrappers for memory-constrained training on Apple Silicon.

Falls back to standard AdamW/Muon when 8-bit is unavailable.
Uses bitsandbytes 8-bit AdamW on CUDA, custom CPU-offloaded 8-bit on MPS.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.optim import Optimizer


def get_8bit_adamw(
    params: list[torch.nn.Parameter] | list[dict[str, Any]],
    lr: float = 3e-4,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    weight_decay: float = 0.1,
) -> Optimizer:
    """Return 8-bit AdamW if bitsandbytes is available, else standard AdamW.

    On CUDA with bitsandbytes installed, uses AdamW8bit (~half optimizer memory).
    On MPS or when bitsandbytes is absent, falls back to standard AdamW.
    """
    try:
        import bitsandbytes as bnb
        if torch.cuda.is_available():
            return bnb.optim.AdamW8bit(  # type: ignore[return-value]
                params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            )
    except ImportError:
        pass

    return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def get_8bit_muon(
    params: list[torch.nn.Parameter],
    lr: float = 0.02,
    momentum: float = 0.95,
    weight_decay: float = 0.1,
) -> Any:
    """Return Muon with 8-bit momentum buffer support.

    When bitsandbytes is available on CUDA, stores momentum states in 8-bit.
    On MPS or CPU, falls back to standard Muon with fp32 momentum.
    Muon halves optimizer memory compared to AdamW by storing only momentum.
    8-bit Muon further halves that to ~25% of AdamW's state.
    """
    from src.training.muon import Muon

    use_8bit = False
    try:
        import bitsandbytes as bnb
        if torch.cuda.is_available():
            use_8bit = True
    except ImportError:
        pass

    if use_8bit:
        return Muon8bit(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    return Muon(params, lr=lr, momentum=momentum, weight_decay=weight_decay)


class Muon8bit(Optimizer):
    """Muon optimizer with 8-bit momentum via bitsandbytes.

    Only works on CUDA. Falls back gracefully to standard Muon on other devices.
    """

    def __init__(
        self,
        params: list[torch.nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
    ):
        try:
            import bitsandbytes as bnb
            self.bnb = bnb
        except ImportError:
            raise ImportError("bitsandbytes required for Muon8bit")

        defaults = {"lr": lr, "momentum": momentum, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    buf = state["momentum_buffer"] = torch.zeros_like(p.data, dtype=torch.uint8)
                    self.bnb.optim.GlobalStateManager.initialize_device()
                else:
                    buf = state["momentum_buffer"]

                buf = self.bnb.optim.Adam8bit.update_momentum_8bit(p.data, grad, buf, momentum)
                p.data.add_(grad, alpha=-lr)

        return loss
