"""Toggle token-efficient RL — Kimi K2.5 §3.4 (arXiv:2602.02276).

Alternates between budget-limited (phase 0) and standard (phase 1) optimization.
Achieves 25-30% token reduction with negligible performance loss.

Phase 0 (budget-limited):
    reward = mean_reward  IF accuracy >= lambda_threshold AND tokens_used <= token_budget
    reward = 0            OTHERWISE

Phase 1 (standard scaling):
    reward = mean_reward  (unrestricted)
"""

from dataclasses import dataclass

import torch


@dataclass
class ToggleReward:
    """Toggle reward module for token-efficient RL.

    Args:
        lambda_threshold: Minimum accuracy required for reward in phase 0. Default 0.8.
        token_budget: Maximum tokens allowed for reward in phase 0. Default 2048.
    """

    lambda_threshold: float = 0.8
    token_budget: int = 2048

    def __call__(
        self,
        mean_reward: torch.Tensor,
        accuracy: float,
        tokens_used: int,
        phase: int,
    ) -> torch.Tensor:
        """Compute the Toggle reward for the given phase.

        Args:
            mean_reward: Mean reward tensor of any shape.
            accuracy: Scalar accuracy of the model output (e.g., pass@1).
            tokens_used: Number of tokens used to produce the output.
            phase: Optimization phase. 0 = budget-limited, 1 = standard scaling.

        Returns:
            Reward tensor of same shape as mean_reward.
        """
        if phase == 1:
            return mean_reward
        # Phase 0: budget-limited — only reward when concise and accurate enough
        if accuracy >= self.lambda_threshold and tokens_used <= self.token_budget:
            return mean_reward
        return torch.zeros_like(mean_reward)
