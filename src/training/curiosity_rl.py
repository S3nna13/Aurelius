"""
Sparse Reward Shaping with Curiosity-Driven Intrinsic Motivation for LLM RLHF training.

Implements:
  - RandomNetworkDistillation (RND)
  - InverseDynamicsModel
  - ForwardDynamicsModel
  - ICMModule (Intrinsic Curiosity Module)
  - RewardShaper
  - TokenLevelCuriosity
  - CuriosityConfig
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Helper: small MLP builder
# ---------------------------------------------------------------------------


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, n_hidden: int = 1) -> nn.Sequential:
    """Build a simple MLP with ReLU activations."""
    layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# RandomNetworkDistillation
# ---------------------------------------------------------------------------


class RandomNetworkDistillation:
    """
    Random Network Distillation (RND) for intrinsic curiosity.

    A fixed random target network maps observations to an encoding.
    A predictor network is trained to match the target. Novel states
    produce a large prediction error, which serves as the intrinsic reward.
    """

    def __init__(self, d_obs: int, d_encoding: int = 64) -> None:
        hidden = max(d_encoding * 2, 32)
        # Fixed random target — never updated
        self.target_net = _mlp(d_obs, hidden, d_encoding)
        for p in self.target_net.parameters():
            p.requires_grad_(False)

        # Trainable predictor
        self.predictor_net = _mlp(d_obs, hidden, d_encoding)
        self._optimizer = torch.optim.Adam(self.predictor_net.parameters(), lr=1e-3)

    def intrinsic_reward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute intrinsic reward as squared prediction error.

        Args:
            obs: [B, d_obs]

        Returns:
            rewards: [B]  (non-negative)
        """
        with torch.no_grad():
            target_enc = self.target_net(obs)
        predictor_enc = self.predictor_net(obs)
        # Detach target so gradient only flows through predictor
        error = predictor_enc - target_enc.detach()
        reward = (error**2).sum(dim=-1)  # [B]
        return reward

    def update_predictor(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Update predictor network via MSE against frozen target.

        Args:
            obs: [B, d_obs]

        Returns:
            loss: scalar tensor
        """
        with torch.no_grad():
            target_enc = self.target_net(obs)
        predictor_enc = self.predictor_net(obs)
        loss = F.mse_loss(predictor_enc, target_enc)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return loss.detach()


# ---------------------------------------------------------------------------
# InverseDynamicsModel
# ---------------------------------------------------------------------------


class InverseDynamicsModel(nn.Module):
    """
    Inverse Dynamics Model: given (obs, next_obs) predict the action taken.

    Concatenates obs and next_obs, feeds through an MLP.
    """

    def __init__(self, d_obs: int, n_actions: int) -> None:
        super().__init__()
        hidden = max(d_obs * 2, 64)
        self.net = _mlp(d_obs * 2, hidden, n_actions, n_hidden=1)

    def forward(self, obs: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs:      [B, d_obs]
            next_obs: [B, d_obs]

        Returns:
            action_logits: [B, n_actions]
        """
        x = torch.cat([obs, next_obs], dim=-1)  # [B, 2*d_obs]
        return self.net(x)


# ---------------------------------------------------------------------------
# ForwardDynamicsModel
# ---------------------------------------------------------------------------


class ForwardDynamicsModel(nn.Module):
    """
    Forward Dynamics Model: given (obs, action) predict next_obs.

    The action is embedded, concatenated with obs, then fed through an MLP.
    """

    def __init__(self, d_obs: int, n_actions: int, d_action_embed: int = 16) -> None:
        super().__init__()
        self.action_embed = nn.Embedding(n_actions, d_action_embed)
        hidden = max((d_obs + d_action_embed) * 2, 64)
        self.net = _mlp(d_obs + d_action_embed, hidden, d_obs, n_hidden=1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs:    [B, d_obs]
            action: [B]  (long integer indices)

        Returns:
            next_obs_pred: [B, d_obs]
        """
        a_emb = self.action_embed(action)  # [B, d_action_embed]
        x = torch.cat([obs, a_emb], dim=-1)  # [B, d_obs + d_action_embed]
        return self.net(x)


# ---------------------------------------------------------------------------
# ICMModule
# ---------------------------------------------------------------------------


class ICMModule(nn.Module):
    """
    Intrinsic Curiosity Module (ICM).

    Combines a feature encoder φ, an InverseDynamicsModel and a
    ForwardDynamicsModel.  Intrinsic rewards are the forward prediction
    error in feature space.
    """

    def __init__(
        self,
        d_obs: int,
        n_actions: int,
        beta: float = 0.2,
        eta: float = 0.01,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.eta = eta

        # Feature encoder φ (shared 2-layer MLP)
        d_feat = max(d_obs, 32)
        self.feature_encoder = nn.Sequential(
            nn.Linear(d_obs, d_feat),
            nn.ReLU(),
            nn.Linear(d_feat, d_feat),
            nn.ReLU(),
        )

        self.inverse_model = InverseDynamicsModel(d_feat, n_actions)
        self.forward_model = ForwardDynamicsModel(d_feat, n_actions)

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation to feature space."""
        return self.feature_encoder(obs)

    def intrinsic_reward(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Intrinsic reward = forward prediction error in feature space.

        Args:
            obs:      [B, d_obs]
            next_obs: [B, d_obs]
            action:   [B]

        Returns:
            rewards: [B]  (non-negative)
        """
        phi_obs = self._encode(obs)
        phi_next = self._encode(next_obs)
        phi_next_pred = self.forward_model(phi_obs, action)
        error = phi_next_pred - phi_next.detach()
        reward = (error**2).sum(dim=-1)  # [B]
        return reward

    def total_loss(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        policy_loss: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combined ICM loss:
            (1 - beta) * policy_loss + beta * inverse_loss + eta * forward_loss

        Args:
            obs:         [B, d_obs]
            next_obs:    [B, d_obs]
            action:      [B]  (long)
            policy_loss: scalar tensor

        Returns:
            total_loss: scalar tensor
        """
        phi_obs = self._encode(obs)
        phi_next = self._encode(next_obs)

        # Inverse loss: predict action from (φ_obs, φ_next)
        action_logits = self.inverse_model(phi_obs, phi_next)
        inverse_loss = F.cross_entropy(action_logits, action)

        # Forward loss: predict φ_next from (φ_obs, action)
        phi_next_pred = self.forward_model(phi_obs, action)
        forward_loss = F.mse_loss(phi_next_pred, phi_next.detach())

        total = (1.0 - self.beta) * policy_loss + self.beta * inverse_loss + self.eta * forward_loss
        return total


# ---------------------------------------------------------------------------
# RewardShaper
# ---------------------------------------------------------------------------


class RewardShaper:
    """
    Shapes rewards by combining extrinsic and intrinsic signals,
    with optional running-statistics normalization.
    """

    def __init__(
        self,
        intrinsic_weight: float = 0.1,
        extrinsic_weight: float = 1.0,
    ) -> None:
        self.intrinsic_weight = intrinsic_weight
        self.extrinsic_weight = extrinsic_weight

        # Running statistics for reward normalization
        self._running_mean: float = 0.0
        self._running_var: float = 1.0
        self._count: int = 0

    def shape(
        self,
        extrinsic_rewards: torch.Tensor,
        intrinsic_rewards: torch.Tensor,
    ) -> torch.Tensor:
        """
        Weighted sum of extrinsic and intrinsic rewards.

        Args:
            extrinsic_rewards: [B]
            intrinsic_rewards: [B]

        Returns:
            shaped_rewards: [B]
        """
        return self.extrinsic_weight * extrinsic_rewards + self.intrinsic_weight * intrinsic_rewards

    def normalize_rewards(
        self,
        rewards: torch.Tensor,
        gamma: float = 0.99,
    ) -> torch.Tensor:
        """
        Subtract running mean, divide by running std (reward normalization).

        Uses Welford-style online update for the running statistics.

        Args:
            rewards: [B]
            gamma:   discount factor (used as exponential decay weight)

        Returns:
            normalized_rewards: [B]
        """
        batch_mean = rewards.mean().item()
        batch_var = rewards.var(unbiased=False).item()
        batch_size = rewards.shape[0]

        # Welford-style online update weighted by gamma
        if self._count == 0:
            self._running_mean = batch_mean
            self._running_var = max(batch_var, 1e-8)
            self._count = batch_size
        else:
            alpha = 1.0 - gamma  # decay weight for old statistics
            self._running_mean = (1.0 - alpha) * self._running_mean + alpha * batch_mean
            self._running_var = (1.0 - alpha) * self._running_var + alpha * batch_var
            self._running_var = max(self._running_var, 1e-8)
            self._count += batch_size

        mean = torch.tensor(self._running_mean, dtype=rewards.dtype, device=rewards.device)
        std = torch.tensor(math.sqrt(self._running_var), dtype=rewards.dtype, device=rewards.device)
        return (rewards - mean) / (std + 1e-8)


# ---------------------------------------------------------------------------
# TokenLevelCuriosity
# ---------------------------------------------------------------------------


class TokenLevelCuriosity:
    """
    Token-level curiosity signals for LLM RLHF.

    Provides:
      - token_surprise: negative log-prob of actual tokens (per-token perplexity)
      - sequence_novelty: fraction of trigrams not seen in history
    """

    def __init__(self, vocab_size: int, d_model: int) -> None:
        self.vocab_size = vocab_size
        self.d_model = d_model
        self._ngram_history: set[tuple[int, ...]] = set()

    def token_surprise(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Negative log probability of actual tokens — token-level perplexity as surprise.

        Args:
            logits:    [B, T, V]
            input_ids: [B, T]

        Returns:
            surprise: [B, T]
        """
        B, T, V = logits.shape
        log_probs = F.log_softmax(logits, dim=-1)  # [B, T, V]
        # Gather log-prob for actual token at each position
        token_log_probs = log_probs.gather(
            -1,
            input_ids.unsqueeze(-1),  # [B, T, 1]
        ).squeeze(-1)  # [B, T]
        return -token_log_probs  # negative log-prob = surprise

    def sequence_novelty(
        self,
        input_ids: torch.Tensor,
        history: set[tuple[int, ...]],
    ) -> torch.Tensor:
        """
        Fraction of trigrams (n=3) not seen in history.

        Args:
            input_ids: [B, T]
            history:   set of seen trigrams (tuples of ints)

        Returns:
            novelty: [B]  values in [0, 1]
        """
        B, T = input_ids.shape
        novelty_scores = []

        ids_list = input_ids.tolist()
        for b in range(B):
            seq = ids_list[b]
            if T < 3:
                # Fewer than 3 tokens — no trigrams possible; treat as fully novel
                novelty_scores.append(1.0)
                continue
            trigrams = [tuple(seq[i : i + 3]) for i in range(T - 2)]
            n_new = sum(1 for tg in trigrams if tg not in history)
            novelty_scores.append(n_new / len(trigrams))

        return torch.tensor(novelty_scores, dtype=torch.float32)

    def update_history(self, input_ids: torch.Tensor) -> None:
        """
        Add all trigrams from input_ids into the internal history.

        Args:
            input_ids: [B, T]
        """
        B, T = input_ids.shape
        ids_list = input_ids.tolist()
        for b in range(B):
            seq = ids_list[b]
            for i in range(T - 2):
                self._ngram_history.add(tuple(seq[i : i + 3]))


# ---------------------------------------------------------------------------
# CuriosityConfig
# ---------------------------------------------------------------------------


@dataclass
class CuriosityConfig:
    """Default hyper-parameters for the curiosity-RL module."""

    d_obs: int = 32
    d_encoding: int = 16
    n_actions: int = 16
    beta: float = 0.2
    eta: float = 0.01
    intrinsic_weight: float = 0.1
    vocab_size: int = 64
    d_model: int = 32
