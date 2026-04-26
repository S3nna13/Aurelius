"""
Double-sided importance sampling policy gradient loss.
Drop-in training utility for token-level policy optimisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class DDISConfig:
    eps_low: float = 0.2
    eps_high: float = 0.2
    clip_outside: bool = True


class DoubleSidedISLoss(nn.Module):
    def __init__(self, config: DDISConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        log_pi_theta: Tensor,
        log_pi_old: Tensor,
        advantages: Tensor,
    ) -> Tensor:
        """Compute the double-sided IS policy gradient loss.

        Args:
            log_pi_theta: (N,) log probabilities under current policy
            log_pi_old:   (N,) log probabilities under behaviour policy
            advantages:   (N,) per-token advantage estimates

        Returns:
            Scalar loss tensor (negative mean advantage for gradient ascent).
        """
        cfg = self.config
        ratio = torch.exp(log_pi_theta - log_pi_old)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - cfg.eps_low, 1.0 + cfg.eps_high) * advantages

        if cfg.clip_outside:
            clipped_low = ratio < (1.0 - cfg.eps_low)
            clipped_high = ratio > (1.0 + cfg.eps_high)
            outside_mask = clipped_low | clipped_high
            valid = ~outside_mask
            loss_terms = torch.where(valid, torch.min(surr1, surr2), torch.zeros_like(surr1))
            return -loss_terms.mean()

        return -torch.min(surr1, surr2).mean()

    def ratio_stats(
        self,
        log_pi_theta: Tensor,
        log_pi_old: Tensor,
    ) -> dict[str, float]:
        """Compute ratio diagnostics.

        Returns:
            dict with keys 'mean_ratio', 'frac_clipped_low', 'frac_clipped_high'
        """
        cfg = self.config
        with torch.no_grad():
            ratio = torch.exp(log_pi_theta - log_pi_old)
            n = ratio.numel()
            mean_ratio = ratio.mean().item()
            frac_clipped_low = (ratio < (1.0 - cfg.eps_low)).float().sum().item() / n
            frac_clipped_high = (ratio > (1.0 + cfg.eps_high)).float().sum().item() / n
        return {
            "mean_ratio": mean_ratio,
            "frac_clipped_low": frac_clipped_low,
            "frac_clipped_high": frac_clipped_high,
        }
