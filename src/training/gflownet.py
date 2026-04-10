"""GFlowNet training for diverse sequence generation.

GFlowNets learn a policy that samples objects proportional to a reward function,
enabling diverse exploration of high-reward sequences rather than mode collapse.

Trajectory Balance (TB) objective:
    Z * P_F(tau) = R(x) * P_B(tau)
    => loss = (log Z + log P_F - log P_B - log R)^2
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GFlowNetConfig:
    """Configuration for GFlowNet training."""

    n_trajectories: int = 8       # number of trajectories to sample per step
    max_seq_len: int = 16         # max sequence length for generation
    temperature: float = 1.0      # sampling temperature
    lambda_tb: float = 1.0        # trajectory balance loss weight
    epsilon: float = 0.05         # exploration noise (epsilon-greedy)


@dataclass
class Trajectory:
    """A single GFlowNet trajectory."""

    states: list[list[int]]       # sequence of token sequences (including initial)
    actions: list[int]            # tokens added at each step
    log_pf: float = 0.0           # log forward probability
    log_pb: float = 0.0           # log backward probability
    reward: float = 0.0           # scalar reward for the final state


def compute_trajectory_balance_loss(
    log_z: torch.Tensor,
    log_pf: torch.Tensor,
    log_pb: torch.Tensor,
    log_reward: torch.Tensor,
) -> torch.Tensor:
    """Compute Trajectory Balance loss.

    TB objective: (log Z + log P_F - log P_B - log R)^2

    Args:
        log_z: Learnable log partition function — scalar or (N,).
        log_pf: Log forward probabilities — scalar or (N,).
        log_pb: Log backward probabilities — scalar or (N,).
        log_reward: Log rewards — scalar or (N,).

    Returns:
        Mean TB loss over all trajectories.
    """
    residual = log_z + log_pf - log_pb - log_reward
    return (residual ** 2).mean()


def sample_trajectory(
    model: nn.Module,
    prompt_ids: list[int],
    config: GFlowNetConfig,
    reward_fn: Callable[[list[int]], float],
) -> Trajectory:
    """Sample a single trajectory from the model.

    Args:
        model: AureliusTransformer (or compatible module).
        prompt_ids: Initial token sequence (prompt).
        config: GFlowNetConfig.
        reward_fn: Callable mapping a token list to a scalar reward.

    Returns:
        Filled Trajectory dataclass.
    """
    current_state = list(prompt_ids)
    states = [list(current_state)]
    actions: list[int] = []
    log_pf = 0.0

    model.train(False)
    with torch.no_grad():
        while len(current_state) < config.max_seq_len:
            input_ids = torch.tensor([current_state], dtype=torch.long)
            _, logits, _ = model(input_ids)
            # logits: (1, seq_len, vocab_size) — use last position
            last_logits = logits[0, -1, :]  # (vocab_size,)

            # Apply temperature
            scaled_logits = last_logits / config.temperature
            log_probs = F.log_softmax(scaled_logits, dim=-1)
            probs = log_probs.exp()

            # Epsilon-greedy exploration: with probability epsilon sample uniformly
            if config.epsilon > 0.0 and torch.rand(1).item() < config.epsilon:
                action = int(torch.randint(0, probs.shape[-1], (1,)).item())
            else:
                action = int(torch.multinomial(probs, num_samples=1).item())

            log_pf += log_probs[action].item()

            current_state = current_state + [action]
            actions.append(action)
            states.append(list(current_state))

    reward = reward_fn(current_state)

    return Trajectory(
        states=states,
        actions=actions,
        log_pf=log_pf,
        log_pb=0.0,  # will be filled by compute_backward_prob
        reward=reward,
    )


def compute_backward_prob(trajectory: Trajectory) -> float:
    """Compute simplified uniform backward policy log probability.

    Assumes a uniform backward policy over the vocabulary at each step:
        log P_B = -|actions| * log(vocab_size)

    The vocab_size is inferred from the number of unique tokens seen across
    all states in the trajectory.

    Args:
        trajectory: A completed Trajectory.

    Returns:
        Log backward probability (negative float).
    """
    n_actions = len(trajectory.actions)
    if n_actions == 0:
        return 0.0

    # Infer vocab_size from all tokens seen in the trajectory
    all_tokens = {tok for state in trajectory.states for tok in state}
    vocab_size = max(len(all_tokens), n_actions + 1)

    log_pb = -n_actions * math.log(vocab_size)
    return log_pb


class GFlowNetTrainer:
    """GFlowNet trainer using the Trajectory Balance objective.

    Args:
        model: AureliusTransformer (or compatible module).
        optimizer: PyTorch optimizer for the model parameters.
        reward_fn: Callable mapping a token list to a scalar reward.
        config: GFlowNetConfig.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        reward_fn: Callable[[list[int]], float],
        config: GFlowNetConfig | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.reward_fn = reward_fn
        self.config = config or GFlowNetConfig()

        # Learnable log partition function
        self.log_z = nn.Parameter(torch.tensor(0.0))

    def train_step(self, prompt_ids: list[int]) -> dict:
        """Run one GFlowNet training step.

        Samples n_trajectories, computes TB loss, and updates model + log_z.

        Args:
            prompt_ids: Starting token sequence.

        Returns:
            dict with keys: loss, mean_reward, log_z, n_trajectories.
        """
        self.model.train()

        trajectories: list[Trajectory] = []
        for _ in range(self.config.n_trajectories):
            traj = sample_trajectory(self.model, prompt_ids, self.config, self.reward_fn)
            traj.log_pb = compute_backward_prob(traj)
            trajectories.append(traj)

        # Build tensors for TB loss
        log_pf_vals = torch.tensor([t.log_pf for t in trajectories], dtype=torch.float32)
        log_pb_vals = torch.tensor([t.log_pb for t in trajectories], dtype=torch.float32)
        # Clamp rewards to avoid log(0)
        rewards = torch.tensor([max(t.reward, 1e-8) for t in trajectories], dtype=torch.float32)
        log_reward_vals = rewards.log()

        # log_z is broadcast across N trajectories
        log_z_expanded = self.log_z.expand(self.config.n_trajectories)

        loss = self.config.lambda_tb * compute_trajectory_balance_loss(
            log_z_expanded, log_pf_vals, log_pb_vals, log_reward_vals
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        mean_reward = float(rewards.mean().item())

        return {
            "loss": loss.item(),
            "mean_reward": mean_reward,
            "log_z": self.log_z.item(),
            "n_trajectories": len(trajectories),
        }

    def generate_diverse(self, prompt_ids: list[int], n: int) -> list[list[int]]:
        """Sample n diverse sequences from the GFlowNet policy.

        Args:
            prompt_ids: Starting token sequence.
            n: Number of sequences to generate.

        Returns:
            List of final token sequences (deduplicated where possible).
        """
        seen: set[tuple[int, ...]] = set()
        results: list[list[int]] = []

        # Try up to 3x to get n unique sequences
        max_attempts = n * 3
        attempts = 0
        while len(results) < n and attempts < max_attempts:
            traj = sample_trajectory(self.model, prompt_ids, self.config, self.reward_fn)
            final_state = traj.states[-1]
            key = tuple(final_state)
            if key not in seen:
                seen.add(key)
                results.append(final_state)
            attempts += 1

        # If we couldn't get n unique, pad with duplicates
        while len(results) < n:
            traj = sample_trajectory(self.model, prompt_ids, self.config, self.reward_fn)
            results.append(traj.states[-1])

        return results[:n]
