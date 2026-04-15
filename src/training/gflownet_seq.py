"""GFlowNet training for sequence generation.

Implements Trajectory Balance (TB) and Detailed Balance (DB) objectives for
training language models to sample sequences proportional to a reward function,
promoting diverse generation rather than mode collapse.

References:
    - Bengio et al., "Flow Network based Generative Models for Non-Iterative
      Diverse Candidate Generation", NeurIPS 2021.
    - Malkin et al., "Trajectory Balance: Improved Credit Assignment in GFlowNets",
      NeurIPS 2022.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GFlowNetConfig:
    """Configuration for GFlowNet sequence training."""

    reward_temperature: float = 1.0
    flow_lr: float = 1e-3
    n_trajectories: int = 8
    max_seq_len: int = 64
    epsilon: float = 0.01


class FlowModel(nn.Module):
    """MLP that predicts log-flow for a state (hidden representation).

    Takes a d_model-dimensional hidden state and produces a scalar log-flow
    estimate per sequence in the batch.
    """

    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Predict log-flow from hidden state.

        Args:
            h: (B, d_model) hidden representation of the current state.

        Returns:
            (B,) scalar log-flow per sequence.
        """
        return self.net(h).squeeze(-1)


def compute_forward_log_prob(
    model: nn.Module,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute sum of log-probs for each token in sequence using model logits.

    For each position t in [0, T-1), we compute log p(token_{t+1} | tokens_{0..t})
    and sum them across the sequence.

    Args:
        model: AureliusTransformer (returns (loss, logits, pkv) tuple).
        input_ids: (B, T) token ids.

    Returns:
        (B,) tensor of summed log-probs per sequence.
    """
    _loss, logits, _pkv = model(input_ids)
    # logits: (B, T, V)
    # Shift: predict token t+1 from position t
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # (B, T-1, V)
    targets = input_ids[:, 1:]  # (B, T-1)
    # Gather the log-prob of the actual next token
    token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
    return token_log_probs.sum(dim=-1)  # (B,)


def trajectory_balance_loss(
    log_Z: torch.Tensor,
    forward_log_probs: torch.Tensor,
    log_reward: torch.Tensor,
) -> torch.Tensor:
    """Compute Trajectory Balance (TB) loss.

    TB objective: (log_Z + forward_log_probs - log_reward)^2

    Args:
        log_Z: Learnable log partition function, scalar or (B,).
        forward_log_probs: Sum of log forward probabilities per trajectory, (B,).
        log_reward: Log reward for each trajectory, (B,).

    Returns:
        Scalar mean TB loss.
    """
    residual = log_Z + forward_log_probs - log_reward
    return (residual ** 2).mean()


def detailed_balance_loss(
    log_flow_s: torch.Tensor,
    log_pf: torch.Tensor,
    log_flow_sp: torch.Tensor,
    log_pb: torch.Tensor,
) -> torch.Tensor:
    """Compute Detailed Balance (DB) loss.

    DB objective: (log F(s) + log P_F(s'|s) - log F(s') - log P_B(s|s'))^2

    Args:
        log_flow_s: Log flow at state s, (B,) or (B, T).
        log_pf: Log forward transition probability, (B,) or (B, T).
        log_flow_sp: Log flow at successor state s', (B,) or (B, T).
        log_pb: Log backward transition probability, (B,) or (B, T).

    Returns:
        Scalar mean DB loss.
    """
    residual = log_flow_s + log_pf - log_flow_sp - log_pb
    return (residual ** 2).mean()


class GFlowNetTrainer:
    """Trainer for GFlowNet sequence generation.

    Combines an autoregressive language model with a flow model to learn
    sampling proportional to a reward function using Trajectory Balance.
    """

    def __init__(
        self,
        model: nn.Module,
        flow_model: FlowModel,
        config: GFlowNetConfig,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.model = model
        self.flow_model = flow_model
        self.config = config
        self.optimizer = optimizer
        # Learnable log partition function
        self.log_Z = nn.Parameter(torch.zeros(1))
        # Add log_Z to optimizer
        self.optimizer.add_param_group({"params": [self.log_Z]})

    def _sample_trajectories(self) -> torch.Tensor:
        """Sample trajectories using the model with epsilon-greedy exploration.

        Returns:
            (n_trajectories, max_seq_len) tensor of token ids.
        """
        device = next(self.model.parameters()).device
        B = self.config.n_trajectories
        # Start with BOS token (0)
        sequences = torch.zeros(B, 1, dtype=torch.long, device=device)

        with torch.no_grad():
            for _ in range(self.config.max_seq_len - 1):
                _loss, logits, _pkv = self.model(sequences)
                # Get logits for the last position
                next_logits = logits[:, -1, :]  # (B, V)
                probs = F.softmax(next_logits, dim=-1)
                vocab_size = probs.shape[-1]

                # Epsilon-greedy: mix with uniform
                eps = self.config.epsilon
                mixed_probs = (1 - eps) * probs + eps / vocab_size

                # Sample next token
                next_tokens = torch.multinomial(mixed_probs, 1)  # (B, 1)
                sequences = torch.cat([sequences, next_tokens], dim=1)

        return sequences

    def train_step(
        self,
        reward_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> dict[str, float]:
        """Execute one GFlowNet training step.

        1. Sample trajectories from the model (with epsilon-greedy exploration).
        2. Compute forward log-probs under the model.
        3. Compute rewards and TB loss.
        4. Update model, flow model, and log_Z.

        Args:
            reward_fn: Callable that takes (B, T) token ids and returns (B,) rewards.

        Returns:
            Dict with 'loss', 'mean_reward', 'log_Z'.
        """
        self.model.train()
        self.flow_model.train()

        # 1. Sample trajectories
        sequences = self._sample_trajectories()  # (B, T)

        # 2. Compute forward log-probs (with gradients)
        forward_lp = compute_forward_log_prob(self.model, sequences)  # (B,)

        # 3. Compute rewards
        with torch.no_grad():
            rewards = reward_fn(sequences)  # (B,)
            # Temperature-scaled log rewards
            log_reward = torch.log(rewards.clamp(min=1e-8)) / self.config.reward_temperature

        # 4. TB loss
        loss = trajectory_balance_loss(self.log_Z, forward_lp, log_reward)

        # 5. Backprop and update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "mean_reward": rewards.mean().item(),
            "log_Z": self.log_Z.item(),
        }
