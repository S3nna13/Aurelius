from __future__ import annotations

import torch
from torch import Tensor

from src.alignment.praxis.config import PRAXISConfig


class MultiTokenAlignmentHorizon:
    """Extends per-token advantages with discounted future credit.

    Ā_t = Σ_{k=0}^{K} γ^k · ā_{t+k}

    This gives each token credit for how aligned the next K tokens are,
    matching the multi-token prediction horizon used during pretraining.
    """

    def __init__(self, config: PRAXISConfig) -> None:
        self.gamma = config.gamma_mtah
        self.k = config.k_mtah

    def extend(self, advantages: Tensor) -> Tensor:
        """Apply temporal credit extension.

        Args:
            advantages: (B, T) — per-token advantage estimates.

        Returns:
            (B, T) — advantages with forward-looking credit added.
        """
        if self.k == 0:
            return advantages

        B, T = advantages.shape
        result = advantages.clone()

        for step in range(1, self.k + 1):
            if step >= T:
                break
            future = torch.zeros_like(advantages)
            future[:, : T - step] = advantages[:, step:]
            result = result + (self.gamma**step) * future

        return result
