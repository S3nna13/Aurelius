"""8-bit optimizer wrappers for memory-constrained training on Apple Silicon.

The helpers keep the historic API used by the trainer/tests while only using
bitsandbytes when CUDA is available and the package exposes the expected
optimizers.
"""

from __future__ import annotations

import torch
from torch.optim import AdamW, Optimizer

# ── Feature detection ───────────────────────────────────────────────


def _bitsandbytes_available() -> bool:
    try:
        import bitsandbytes
    except ImportError:
        return False

    return hasattr(bitsandbytes.optim, "AdamW8bit")


def _can_use_8bit() -> bool:
    return torch.cuda.is_available() and _bitsandbytes_available()


# ── Public helper functions ─────────────────────────────────────────


def get_8bit_adamw(params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
    if _can_use_8bit():
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
    return AdamW(
        params,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )


def get_8bit_muon(params, lr=0.02, momentum=0.95, weight_decay=0.1, ns_steps=5):
    if _can_use_8bit():
        return Muon8bit(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
        )

    from src.training.muon import Muon

    return Muon(
        params,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        ns_steps=ns_steps,
    )


# ── Wrapper class used by trainer.py ────────────────────────────────


class AdamW8bitWrapper(Optimizer):
    """AdamW wrapper that falls back to fp32 when bitsandbytes is unavailable."""

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        use_8bit: bool = True,
    ) -> None:
        params = list(params)
        self._use_8bit = use_8bit and _can_use_8bit()
        if self._use_8bit:
            import bitsandbytes as bnb

            self._opt = bnb.optim.AdamW8bit(
                params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
            )
        else:
            self._opt = AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        # Keep the wrapper's public state in sync with the real optimizer so
        # callers like OptimizerOffloader can inspect it.
        self.param_groups = self._opt.param_groups
        self.state = self._opt.state

    @property
    def is_8bit(self) -> bool:
        return self._use_8bit

    def step(self, closure=None):
        return self._opt.step(closure)

    def zero_grad(self, set_to_none: bool = True):
        self._opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self._opt.state_dict()

    def load_state_dict(self, state_dict):
        self._opt.load_state_dict(state_dict)

    def __repr__(self) -> str:
        return f"AdamW8bitWrapper(8bit={self._use_8bit})"


class Muon8bit(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        momentum_8bit: bool = True,
    ):
        if not _bitsandbytes_available():
            raise ImportError("bitsandbytes required for Muon8bit")
        params = list(params)
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)
        self._momentum_8bit = momentum_8bit

    @torch.no_grad()
    def step(self, closure=None):
        from src.training.muon import _newton_schulz

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.ndim < 2:
                    raise ValueError("Muon requires 2D+ parameters.")

                g = p.grad.float()
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                if buf.dtype != torch.float32:
                    buf = buf.float()
                    state["momentum_buffer"] = buf

                buf.mul_(momentum).add_(g)

                g_orth = _newton_schulz(buf.view(buf.shape[0], -1), steps=ns_steps)
                g_orth = g_orth.view_as(p)

                rms = g_orth.pow(2).mean().sqrt().clamp(min=1e-8)
                g_orth = g_orth / rms

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                p.add_(g_orth.to(p.dtype), alpha=-lr)

                if self._momentum_8bit and hasattr(torch, "float8_e4m3fn"):
                    state["momentum_buffer"] = state["momentum_buffer"].to(torch.float8_e4m3fn)

        return loss
