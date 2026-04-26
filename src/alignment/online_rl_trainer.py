"""Online RL trainer with importance-sampling correction (PPO-style).

Implements GAE-lambda advantage estimation (Schulman 2015) and a clipped
surrogate objective for policy optimisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from src.alignment import ALIGNMENT_REGISTRY


@dataclass
class OnlineRLConfig:
    """Hyperparameters for the online RL training loop."""

    lr: float = 1e-5
    kl_coef: float = 0.1
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0


class GAEComputation:
    """Generalised Advantage Estimation (Schulman et al., 2015)."""

    @staticmethod
    def compute_gae(
        rewards: list[float],
        values: list[float],
        dones: list[bool],
        config: OnlineRLConfig,
    ) -> tuple[list[float], list[float]]:
        """Compute GAE advantages and discounted returns.

        The value at the terminal step is bootstrapped to 0.

        Args:
            rewards: Per-step rewards ``r[t]``, length T.
            values: Per-step value estimates ``V[t]``, length T.
            dones: Per-step done flags, length T.  ``True`` means the episode
                ended *after* step t, so ``V[t+1]`` should be treated as 0.
            config: RL hyper-parameters (gamma, gae_lambda).

        Returns:
            ``(advantages, returns)`` — both lists of length T.
        """
        T = len(rewards)
        advantages = [0.0] * T
        returns = [0.0] * T

        gae = 0.0
        next_value = 0.0  # bootstrap past the final step

        for t in reversed(range(T)):
            mask = 0.0 if dones[t] else 1.0
            next_val = values[t + 1] if t + 1 < T else next_value
            delta = rewards[t] + config.gamma * next_val * mask - values[t]
            gae = delta + config.gamma * config.gae_lambda * mask * gae
            advantages[t] = gae
            returns[t] = gae + values[t]

        return advantages, returns


class OnlineRLTrainer:
    """PPO-style online RL trainer with importance-sampling correction.

    This class is stateless with respect to model parameters — callers are
    responsible for owning the model and optimiser.  ``train_step`` accepts a
    pre-computed batch dict and returns scalar metrics.
    """

    def __init__(self, config: OnlineRLConfig | None = None) -> None:
        self.config = config or OnlineRLConfig()

    # ------------------------------------------------------------------
    # Loss components
    # ------------------------------------------------------------------

    def compute_policy_loss(
        self,
        logprobs: Tensor,
        old_logprobs: Tensor,
        advantages: Tensor,
        clip_eps: float,
    ) -> Tensor:
        """Clipped surrogate policy loss (PPO).

        Args:
            logprobs: Log-probabilities under the current policy, shape (B,).
            old_logprobs: Log-probabilities under the behaviour policy, (B,).
            advantages: Advantage estimates, shape (B,).
            clip_eps: Clipping radius epsilon.

        Returns:
            Scalar policy loss (negated, so minimising this maximises reward).
        """
        ratio = torch.exp(logprobs - old_logprobs)
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        surrogate = torch.min(ratio * advantages, clipped_ratio * advantages)
        return -surrogate.mean()

    def compute_value_loss(
        self,
        values: Tensor,
        returns: Tensor,
    ) -> Tensor:
        """Mean-squared-error value function loss.

        Args:
            values: Predicted value estimates, shape (B,).
            returns: Target returns, shape (B,).

        Returns:
            Scalar MSE loss.
        """
        return F.mse_loss(values, returns)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self, batch: dict) -> dict:
        """Compute combined PPO loss from a pre-collected batch.

        The caller is responsible for calling ``.backward()`` and
        ``.step()`` on the optimiser — this method only computes and
        returns the loss tensors and diagnostic scalars.

        Expected batch keys
        -------------------
        logprobs : Tensor (B,)      — current policy log-probs
        old_logprobs : Tensor (B,)  — behaviour policy log-probs
        advantages : Tensor (B,)    — GAE advantages
        returns : Tensor (B,)       — discounted returns
        values : Tensor (B,)        — current value estimates
        entropy : Tensor scalar     — mean policy entropy

        Returns
        -------
        dict with keys: policy_loss, value_loss, entropy, total_loss, kl
        """
        logprobs: Tensor = batch["logprobs"]
        old_logprobs: Tensor = batch["old_logprobs"]
        advantages: Tensor = batch["advantages"]
        returns: Tensor = batch["returns"]
        values: Tensor = batch["values"]
        entropy: Tensor = batch["entropy"]

        cfg = self.config

        policy_loss = self.compute_policy_loss(logprobs, old_logprobs, advantages, cfg.clip_eps)
        value_loss = self.compute_value_loss(values, returns)

        # KL divergence approximation: E[old_logp - logp]
        with torch.no_grad():
            kl = (old_logprobs - logprobs).mean()

        total_loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "total_loss": total_loss,
            "kl": kl,
        }


ALIGNMENT_REGISTRY["online_rl"] = OnlineRLTrainer
