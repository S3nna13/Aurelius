"""GrokFast: Accelerated grokking by amplifying slow gradient components.

Modifies gradients in-place before the optimizer step by amplifying
the low-frequency (slow) component captured by an EMA or SMA filter.

Reference: Liu et al. 2024, "GrokFast: Accelerated Grokking by Amplifying Slow Gradients"
arXiv:2405.20233

Usage (EMA variant):
    grokfast = GrokFastEMA(model, alpha=0.98, lamb=2.0)

    # In training loop, AFTER loss.backward() but BEFORE optimizer.step():
    grokfast.step()   # update() then amplify()
    optimizer.step()

Usage (wrapped optimizer):
    gf_opt = GrokFastOptimizer(optimizer, model, alpha=0.98, lamb=2.0)
    gf_opt.step()     # applies grokfast then advances optimizer
"""
from __future__ import annotations

from collections import deque
from typing import Any

import torch
import torch.nn as nn


class GrokFastEMA:
    """GrokFast with Exponential Moving Average filter.

    Maintains a per-parameter EMA of gradients and amplifies the slow
    (low-frequency) component.  Call ``step()`` after ``loss.backward()``
    and before ``optimizer.step()``.

    Modified gradient = g + lamb * EMA(g)
    where EMA_t = alpha * EMA_{t-1} + (1 - alpha) * g_t
    """

    def __init__(
        self,
        model: nn.Module,
        alpha: float = 0.98,
        lamb: float = 2.0,
    ) -> None:
        self.model = model
        self.alpha = alpha
        self.lamb = lamb
        # EMA state keyed by parameter id (avoids name collisions in shared params)
        self._ema: dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self) -> None:
        """Update the per-parameter EMA with the current gradients.

        For each parameter with a non-None ``.grad``:
            EMA_t = alpha * EMA_{t-1} + (1 - alpha) * grad
        """
        for param in self.model.parameters():
            if param.grad is None:
                continue
            pid = id(param)
            g = param.grad.data
            if pid not in self._ema:
                self._ema[pid] = torch.zeros_like(g)
            self._ema[pid] = self.alpha * self._ema[pid] + (1.0 - self.alpha) * g

    def amplify(self) -> None:
        """Add ``lamb * EMA`` to the current gradient in-place.

        Must be called *after* ``update()`` so that the EMA has been refreshed.
        """
        for param in self.model.parameters():
            if param.grad is None:
                continue
            pid = id(param)
            if pid not in self._ema:
                continue
            param.grad.data.add_(self.lamb * self._ema[pid])

    def step(self) -> None:
        """Convenience method: calls ``update()`` then ``amplify()``."""
        self.update()
        self.amplify()

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Return a serialisable snapshot of the EMA state.

        The EMA tensors are stored as a list parallel to the parameter order
        returned by ``model.parameters()``, keeping only entries for params
        that appear in the internal map.
        """
        param_list = list(self.model.parameters())
        ema_entries: list[dict[str, Any]] = []
        for idx, param in enumerate(param_list):
            pid = id(param)
            if pid in self._ema:
                ema_entries.append({"param_idx": idx, "ema": self._ema[pid].clone()})
        return {
            "alpha": self.alpha,
            "lamb": self.lamb,
            "ema_entries": ema_entries,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore EMA state from a ``state_dict`` produced by ``state_dict()``."""
        self.alpha = state_dict["alpha"]
        self.lamb = state_dict["lamb"]
        param_list = list(self.model.parameters())
        self._ema = {}
        for entry in state_dict["ema_entries"]:
            idx = entry["param_idx"]
            if idx < len(param_list):
                pid = id(param_list[idx])
                self._ema[pid] = entry["ema"].clone()


class GrokFastSMA:
    """GrokFast with Simple Moving Average filter over the last K gradients.

    Uses more memory than EMA but can give a smoother slow component.

    Modified gradient = g + lamb * mean(last K gradients)
    """

    def __init__(
        self,
        model: nn.Module,
        window: int = 5,
        lamb: float = 2.0,
    ) -> None:
        self.model = model
        self.window = window
        self.lamb = lamb
        # Per-parameter deque of recent gradient tensors
        self._buffers: dict[int, deque] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self) -> None:
        """Push current ``param.grad.data`` into the rolling window."""
        for param in self.model.parameters():
            if param.grad is None:
                continue
            pid = id(param)
            if pid not in self._buffers:
                self._buffers[pid] = deque(maxlen=self.window)
            self._buffers[pid].append(param.grad.data.clone())

    def amplify(self) -> None:
        """Add ``lamb * mean(window)`` to the current gradient in-place."""
        for param in self.model.parameters():
            if param.grad is None:
                continue
            pid = id(param)
            if pid not in self._buffers or len(self._buffers[pid]) == 0:
                continue
            sma = torch.stack(list(self._buffers[pid])).mean(dim=0)
            param.grad.data.add_(self.lamb * sma)

    def step(self) -> None:
        """Convenience method: calls ``update()`` then ``amplify()``."""
        self.update()
        self.amplify()


class GrokFastOptimizer:
    """Wraps any ``torch.optim.Optimizer`` with GrokFast EMA gradient amplification.

    Call ``zero_grad()`` as normal, then after ``loss.backward()`` call
    ``step()`` — it applies GrokFast and then advances the inner optimizer.

    Example::

        opt = GrokFastOptimizer(torch.optim.AdamW(model.parameters()), model)
        loss.backward()
        opt.step()
        opt.zero_grad()
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        alpha: float = 0.98,
        lamb: float = 2.0,
    ) -> None:
        self.optimizer = optimizer
        self.grokfast = GrokFastEMA(model, alpha=alpha, lamb=lamb)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Apply GrokFast amplification, then advance the inner optimizer."""
        self.grokfast.step()
        self.optimizer.step()

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Delegate ``zero_grad`` to the inner optimizer."""
        self.optimizer.zero_grad(set_to_none=set_to_none)

    # ------------------------------------------------------------------
    # Passthrough properties
    # ------------------------------------------------------------------

    @property
    def param_groups(self):
        """Expose inner optimizer's ``param_groups`` directly."""
        return self.optimizer.param_groups
