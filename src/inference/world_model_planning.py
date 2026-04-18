"""
World Models for LLM Planning.

Implements a latent state model that predicts future states and rewards
for model-based planning. All components use pure native PyTorch.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class LatentEncoder(nn.Module):
    """Encodes a token sequence into a fixed-size latent vector.

    Architecture:
        Embedding(vocab_size, d_model)
        → TransformerEncoderLayer (single layer, nhead=max(1, d_model//8))
        → mean-pool over T
        → Linear(d_model, d_latent)
    """

    def __init__(self, d_model: int, d_latent: int, vocab_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        nhead = max(1, d_model // 8)
        # Ensure d_model divisible by nhead
        while d_model % nhead != 0 and nhead > 1:
            nhead -= 1
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.proj = nn.Linear(d_model, d_latent)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [B, T] long tensor of token ids
        Returns:
            z: [B, d_latent]
        """
        x = self.embedding(input_ids)          # [B, T, d_model]
        x = self.transformer(x)                # [B, T, d_model]
        x = x.mean(dim=1)                      # [B, d_model]
        z = self.proj(x)                       # [B, d_latent]
        return z


class LatentTransitionModel(nn.Module):
    """Predicts next latent state given current latent state and action.

    Architecture:
        action_embed: Embedding(n_actions, n_actions)
        MLP: [d_latent + n_actions] → 4*d_latent → d_latent
    """

    def __init__(self, d_latent: int, n_actions: int) -> None:
        super().__init__()
        self.action_embed = nn.Embedding(n_actions, n_actions)
        in_dim = d_latent + n_actions
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 4 * d_latent),
            nn.ReLU(),
            nn.Linear(4 * d_latent, d_latent),
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z:      [B, d_latent]
            action: [B] long tensor of action indices
        Returns:
            z_next: [B, d_latent]
        """
        a_emb = self.action_embed(action)      # [B, n_actions]
        x = torch.cat([z, a_emb], dim=-1)     # [B, d_latent + n_actions]
        z_next = self.mlp(x)                   # [B, d_latent]
        return z_next


class RewardPredictor(nn.Module):
    """Predicts a scalar reward from a latent state."""

    def __init__(self, d_latent: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_latent, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, d_latent]
        Returns:
            reward: [B]
        """
        return self.linear(z).squeeze(-1)      # [B]


class LatentDecoder(nn.Module):
    """Decodes a latent vector back to token logits.

    Architecture:
        MLP: d_latent → d_model → Linear(d_model, vocab_size * seq_len)
        reshape → [B, seq_len, vocab_size]
    """

    def __init__(
        self,
        d_latent: int,
        d_model: int,
        vocab_size: int,
        seq_len: int,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.mlp = nn.Sequential(
            nn.Linear(d_latent, d_model),
            nn.ReLU(),
            nn.Linear(d_model, vocab_size * seq_len),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, d_latent]
        Returns:
            logits: [B, seq_len, vocab_size]
        """
        B = z.shape[0]
        out = self.mlp(z)                      # [B, vocab_size * seq_len]
        logits = out.view(B, self.seq_len, self.vocab_size)
        return logits


# ---------------------------------------------------------------------------
# World Model
# ---------------------------------------------------------------------------

class WorldModel(nn.Module):
    """Combines encoder, transition model, reward predictor and decoder."""

    def __init__(
        self,
        d_model: int,
        d_latent: int,
        vocab_size: int,
        n_actions: int,
        seq_len: int,
    ) -> None:
        super().__init__()
        self.encoder = LatentEncoder(d_model, d_latent, vocab_size)
        self.transition = LatentTransitionModel(d_latent, n_actions)
        self.reward_pred = RewardPredictor(d_latent)
        self.decoder = LatentDecoder(d_latent, d_model, vocab_size, seq_len)

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids [B, T] → z [B, d_latent]"""
        return self.encoder(input_ids)

    def predict_next(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z:      [B, d_latent]
            action: [B]
        Returns:
            z_next: [B, d_latent]
            reward: [B]
        """
        z_next = self.transition(z, action)
        reward = self.reward_pred(z_next)
        return z_next, reward

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z [B, d_latent] → logits [B, seq_len, vocab_size]"""
        return self.decoder(z)

    def imagination_rollout(
        self,
        z0: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """H-step imagination rollout.

        Args:
            z0:      [B, d_latent]  initial latent state
            actions: [B, H]         action sequence for each batch element
        Returns:
            z_traj:  [B, H+1, d_latent]  latent trajectory (includes z0)
            rewards: [B, H]              reward at each step
        """
        B, H = actions.shape
        z_traj = [z0]
        rewards_list: List[torch.Tensor] = []

        z = z0
        for h in range(H):
            a = actions[:, h]                  # [B]
            z_next, r = self.predict_next(z, a)
            z_traj.append(z_next)
            rewards_list.append(r.unsqueeze(1))  # [B, 1]
            z = z_next

        z_traj_tensor = torch.stack(z_traj, dim=1)          # [B, H+1, d_latent]
        rewards_tensor = torch.cat(rewards_list, dim=1)     # [B, H]
        return z_traj_tensor, rewards_tensor


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class WorldModelTrainer:
    """Training utilities for the WorldModel."""

    def __init__(self, model: WorldModel, lr: float = 1e-3) -> None:
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def reconstruction_loss(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encode tokens → decode → cross-entropy against original tokens.

        Args:
            input_ids: [B, T]
        Returns:
            loss: scalar
        """
        z = self.model.encode(input_ids)       # [B, d_latent]
        logits = self.model.decode(z)          # [B, seq_len, vocab_size]

        # Align T and seq_len by truncating / padding the target
        T = input_ids.shape[1]
        seq_len = logits.shape[1]
        min_len = min(T, seq_len)

        # [B, min_len, vocab] → [B*min_len, vocab]
        logits_flat = logits[:, :min_len, :].reshape(-1, logits.shape[-1])
        targets_flat = input_ids[:, :min_len].reshape(-1)

        loss = F.cross_entropy(logits_flat, targets_flat)
        return loss

    def transition_loss(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        z_next_true: torch.Tensor,
    ) -> torch.Tensor:
        """MSE between predicted and true next latent state.

        Args:
            z:           [B, d_latent]
            action:      [B]
            z_next_true: [B, d_latent]
        Returns:
            loss: scalar
        """
        z_next_pred = self.model.transition(z, action)
        return F.mse_loss(z_next_pred, z_next_true)

    def reward_loss(
        self,
        z: torch.Tensor,
        true_reward: torch.Tensor,
    ) -> torch.Tensor:
        """MSE between predicted and true reward.

        Args:
            z:           [B, d_latent]
            true_reward: [B]
        Returns:
            loss: scalar
        """
        pred_reward = self.model.reward_pred(z)
        return F.mse_loss(pred_reward, true_reward)

    def train_step(
        self,
        input_ids_t: torch.Tensor,
        input_ids_next: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
    ) -> dict:
        """Single optimisation step combining all losses.

        Args:
            input_ids_t:    [B, T]  current token sequences
            input_ids_next: [B, T]  next token sequences
            actions:        [B]     actions taken
            rewards:        [B]     observed rewards
        Returns:
            dict with keys 'reconstruction_loss', 'transition_loss',
                           'reward_loss', 'total_loss'
        """
        self.optimizer.zero_grad()

        z_t = self.model.encode(input_ids_t)
        z_next_true = self.model.encode(input_ids_next)

        rec_loss = self.reconstruction_loss(input_ids_t)
        trans_loss = self.transition_loss(z_t, actions, z_next_true)
        rew_loss = self.reward_loss(z_t, rewards)

        total_loss = rec_loss + trans_loss + rew_loss
        total_loss.backward()
        self.optimizer.step()

        return {
            "reconstruction_loss": rec_loss.item(),
            "transition_loss": trans_loss.item(),
            "reward_loss": rew_loss.item(),
            "total_loss": total_loss.item(),
        }


# ---------------------------------------------------------------------------
# Planning Agent
# ---------------------------------------------------------------------------

class PlanningAgent:
    """Model-based planning agent using the WorldModel for look-ahead."""

    def __init__(
        self,
        world_model: WorldModel,
        horizon: int = 5,
        n_rollouts: int = 8,
    ) -> None:
        self.world_model = world_model
        self.horizon = horizon
        self.n_rollouts = n_rollouts
        # Derive n_actions from transition embedding table
        self.n_actions: int = world_model.transition.action_embed.num_embeddings

    @torch.no_grad()
    def random_shooting(self, z0: torch.Tensor) -> torch.Tensor:
        """Sample n_rollouts random action sequences, return the best.

        For each batch element the sequence with the highest cumulative
        reward is selected.

        Args:
            z0: [B, d_latent]
        Returns:
            best_actions: [B, horizon]
        """
        B, d_latent = z0.shape
        device = z0.device

        # actions_all: [n_rollouts, B, horizon]
        actions_all = torch.randint(
            0, self.n_actions, (self.n_rollouts, B, self.horizon), device=device
        )

        # Evaluate each candidate sequence
        # total_rewards: [n_rollouts, B]
        total_rewards = torch.zeros(self.n_rollouts, B, device=device)

        for k in range(self.n_rollouts):
            _, rewards = self.world_model.imagination_rollout(
                z0, actions_all[k]
            )  # rewards: [B, horizon]
            total_rewards[k] = rewards.sum(dim=1)

        # best_idx: [B]
        best_idx = total_rewards.argmax(dim=0)  # [B]

        # Gather best actions: [B, horizon]
        best_actions = actions_all[best_idx, torch.arange(B, device=device), :]
        return best_actions

    @torch.no_grad()
    def greedy_rollout(self, z0: torch.Tensor) -> torch.Tensor:
        """At each step pick the action maximising immediate reward.

        Args:
            z0: [B, d_latent]
        Returns:
            actions: [B, horizon]
        """
        B, _ = z0.shape
        device = z0.device
        z = z0
        chosen_actions: List[torch.Tensor] = []

        for _ in range(self.horizon):
            # Evaluate all actions for every batch element
            # Expand z: [B * n_actions, d_latent]
            z_exp = z.unsqueeze(1).expand(B, self.n_actions, -1).reshape(
                B * self.n_actions, -1
            )
            a_exp = torch.arange(self.n_actions, device=device).unsqueeze(0).expand(
                B, self.n_actions
            ).reshape(B * self.n_actions)

            z_next_exp, reward_exp = self.world_model.predict_next(z_exp, a_exp)
            # reward_exp: [B * n_actions]
            reward_matrix = reward_exp.view(B, self.n_actions)  # [B, n_actions]
            best_a = reward_matrix.argmax(dim=1)                # [B]

            chosen_actions.append(best_a.unsqueeze(1))         # [B, 1]

            # Advance z using chosen action
            z, _ = self.world_model.predict_next(z, best_a)

        return torch.cat(chosen_actions, dim=1)                 # [B, horizon]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class WorldModelConfig:
    """Default hyperparameters for the world model pipeline."""
    d_model: int = 32
    d_latent: int = 16
    vocab_size: int = 64
    n_actions: int = 8
    seq_len: int = 8
    horizon: int = 4
    n_rollouts: int = 8
    lr: float = 1e-4
