"""GrokFast: Accelerated grokking by amplifying slow gradient components.

Modifies gradients in-place before the optimizer step by amplifying
the low-frequency (slow) component identified via EMA.

Reference: Lee et al. 2024, "GrokFast: Accelerated Grokking by Amplifying Slow Gradients"
arXiv:2405.20233

Usage:
    grokfast = GrokFastEMA(model, alpha=0.98, lamb=2.0)

    # In training loop, AFTER loss.backward() but BEFORE optimizer.step():
    grokfast.apply(model)
    optimizer.step()
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GrokFastConfig:
    alpha: float = 0.98     # EMA decay factor (close to 1 = slow)
    lamb: float = 2.0       # amplification factor for slow component
    window_size: int = 100  # for GrokFastMA variant


class GrokFastEMA:
    """GrokFast with Exponential Moving Average filter.

    Maintains EMA of gradients and amplifies the slow (EMA) component.

    Modified gradient = g + lamb * EMA(g)
    where EMA(g) = alpha * EMA_prev + (1-alpha) * g
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: GrokFastConfig | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg or GrokFastConfig()
        # EMA state: param_name → EMA tensor (same shape as param)
        self._ema: dict[str, torch.Tensor] = {}

    def apply(self, model: nn.Module | None = None) -> None:
        """Amplify gradients in-place for all parameters with .grad.

        Must be called after loss.backward() and before optimizer.step().
        """
        model = model or self.model
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            g = param.grad.data

            # Initialize EMA on first call
            if name not in self._ema:
                self._ema[name] = torch.zeros_like(g)

            # Update EMA: α * prev + (1-α) * g
            self._ema[name] = self.cfg.alpha * self._ema[name] + (1 - self.cfg.alpha) * g

            # Amplify: g += λ * EMA(g)
            param.grad.data = g + self.cfg.lamb * self._ema[name]

    def reset(self) -> None:
        """Reset EMA state (call when starting a new task)."""
        self._ema.clear()


class GrokFastMA:
    """GrokFast with moving average over a fixed window.

    More memory than EMA but potentially smoother slow component.
    Modified gradient = g + lamb * MA(g)
    where MA(g) = mean of last window_size gradients
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: GrokFastConfig | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg or GrokFastConfig()
        self._windows: dict[str, deque] = {}

    def apply(self, model: nn.Module | None = None) -> None:
        """Apply moving average gradient amplification in-place."""
        model = model or self.model
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            g = param.grad.data.clone()

            if name not in self._windows:
                self._windows[name] = deque(maxlen=self.cfg.window_size)

            self._windows[name].append(g)
            ma = torch.stack(list(self._windows[name])).mean(dim=0)
            param.grad.data = param.grad.data + self.cfg.lamb * ma

    def reset(self) -> None:
        self._windows.clear()


def apply_grokfast(
    model: nn.Module,
    ema_state: dict[str, torch.Tensor],
    alpha: float = 0.98,
    lamb: float = 2.0,
) -> dict[str, torch.Tensor]:
    """Functional interface for GrokFast EMA.

    Updates EMA state and modifies gradients in-place.
    Returns updated ema_state dict.
    """
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        g = param.grad.data
        if name not in ema_state:
            ema_state[name] = torch.zeros_like(g)
        ema_state[name] = alpha * ema_state[name] + (1 - alpha) * g
        param.grad.data = g + lamb * ema_state[name]
    return ema_state
