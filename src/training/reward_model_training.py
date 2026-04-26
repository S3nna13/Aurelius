"""
Reward model training pipeline from human preference data.
Implements Bradley-Terry model, listwise ranking, reward normalization, and evaluation.
Pure PyTorch only -- no transformers, einops, trl, xformers, flash_attn, etc.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# RewardModel
# ---------------------------------------------------------------------------


class RewardModel(nn.Module):
    """Scalar reward head on top of a language model backbone.

    Args:
        backbone: callable (input_ids: Tensor[B, T]) -> hidden_states (B, T, D)
        d_model: hidden dimension of the backbone output
    """

    def __init__(self, backbone: nn.Module, d_model: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.reward_head = nn.Linear(d_model, 1)

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Args:
            input_ids: (B, T) long tensor

        Returns:
            rewards: (B,) scalar reward per sequence -- last-token value
        """
        hidden_states = self.backbone(input_ids)  # (B, T, D)
        last_hidden = hidden_states[:, -1, :]  # (B, D)
        rewards = self.reward_head(last_hidden).squeeze(-1)  # (B,)
        return rewards


# ---------------------------------------------------------------------------
# BradleyTerryLoss
# ---------------------------------------------------------------------------


class BradleyTerryLoss(nn.Module):
    """Pairwise preference loss based on the Bradley-Terry model.

    Args:
        margin: optional margin added to the reward difference (default 0.0)
    """

    def __init__(self, margin: float = 0.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        chosen_rewards: Tensor,
        rejected_rewards: Tensor,
    ) -> tuple:
        """
        Args:
            chosen_rewards:  (B,) scalar rewards for chosen responses
            rejected_rewards: (B,) scalar rewards for rejected responses

        Returns:
            loss:     scalar -- negative log-likelihood under Bradley-Terry
            accuracy: scalar in [0, 1] -- fraction where chosen > rejected
        """
        diff = chosen_rewards - rejected_rewards - self.margin
        loss = -F.logsigmoid(diff).mean()
        accuracy = ((chosen_rewards - rejected_rewards) > 0).float().mean()
        return loss, accuracy


# ---------------------------------------------------------------------------
# ListwiseRankingLoss
# ---------------------------------------------------------------------------


class ListwiseRankingLoss(nn.Module):
    """Listwise ranking loss for K candidate responses per prompt."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, rewards: Tensor, preference_labels: Tensor) -> Tensor:
        """
        Args:
            rewards:           (B, K) predicted reward scores
            preference_labels: (B, K) ground-truth quality scores (higher = better)

        Returns:
            loss: scalar -- cross-entropy between target distribution and predicted ranking
        """
        target_dist = F.softmax(preference_labels, dim=1)  # (B, K)
        log_pred = F.log_softmax(rewards, dim=1)  # (B, K)
        per_sample = -(target_dist * log_pred).sum(dim=1)  # (B,)
        return per_sample.mean()


# ---------------------------------------------------------------------------
# RewardNormalizer
# ---------------------------------------------------------------------------


class RewardNormalizer:
    """Normalize rewards for training stability using exponential moving averages.

    Args:
        momentum: EMA decay factor (default 0.99)
    """

    def __init__(self, momentum: float = 0.99) -> None:
        self.momentum = momentum
        self._mean: float = 0.0
        self._var: float = 1.0
        self._n: int = 0

    def update(self, rewards: Tensor) -> None:
        """Update running mean and variance estimates with new rewards."""
        batch_mean = rewards.mean().item()
        batch_var = rewards.var(unbiased=False).item() if rewards.numel() > 1 else 0.0
        self._n += rewards.numel()
        if self._n == rewards.numel():
            # First update: initialise directly
            self._mean = batch_mean
            self._var = max(batch_var, 1e-8)
        else:
            self._mean = self.momentum * self._mean + (1.0 - self.momentum) * batch_mean
            self._var = self.momentum * self._var + (1.0 - self.momentum) * batch_var
            self._var = max(self._var, 1e-8)

    @property
    def std(self) -> float:
        return math.sqrt(self._var)

    def normalize(self, rewards: Tensor) -> Tensor:
        """Normalize rewards to approximately zero mean and unit variance."""
        return (rewards - self._mean) / (self.std + 1e-8)

    def denormalize(self, normalized: Tensor) -> Tensor:
        """Invert normalization to recover original reward scale."""
        return normalized * (self.std + 1e-8) + self._mean

    def stats(self) -> dict:
        """Return current running statistics."""
        return {
            "mean": self._mean,
            "std": self.std,
            "n_samples": self._n,
        }


# ---------------------------------------------------------------------------
# PreferenceDataset
# ---------------------------------------------------------------------------


class PreferenceDataset:
    """Dataset of preference pairs for reward model training.

    Args:
        chosen_ids:    list of (T,) long tensors for chosen responses
        rejected_ids:  list of (T,) long tensors for rejected responses
        margin_labels: optional confidence/strength of preference per pair
    """

    def __init__(
        self,
        chosen_ids: list,
        rejected_ids: list,
        margin_labels: list | None = None,
    ) -> None:
        if len(chosen_ids) != len(rejected_ids):
            raise ValueError(
                f"chosen_ids and rejected_ids must have the same length, "
                f"got {len(chosen_ids)} and {len(rejected_ids)}"
            )
        self._chosen = chosen_ids
        self._rejected = rejected_ids
        self._margins = margin_labels if margin_labels is not None else [1.0] * len(chosen_ids)

    def __len__(self) -> int:
        return len(self._chosen)

    def __getitem__(self, idx: int) -> tuple:
        return self._chosen[idx], self._rejected[idx], self._margins[idx]

    @staticmethod
    def collate_fn(batch: list) -> tuple:
        """Collate a list of (chosen, rejected, margin) tuples into batched tensors."""
        chosen_list, rejected_list, margins = zip(*batch)

        def pad_sequences(seqs: tuple) -> Tensor:
            max_len = max(s.size(0) for s in seqs)
            padded = []
            for s in seqs:
                pad_len = max_len - s.size(0)
                if pad_len > 0:
                    s = F.pad(s, (0, pad_len))
                padded.append(s)
            return torch.stack(padded, dim=0)

        chosen_batch = pad_sequences(chosen_list)
        rejected_batch = pad_sequences(rejected_list)
        margins_batch = torch.tensor(margins, dtype=torch.float32)
        return chosen_batch, rejected_batch, margins_batch

    def split(self, ratio: float = 0.8) -> tuple:
        """Split into train and validation datasets.

        Args:
            ratio: fraction of data used for training

        Returns:
            (train_dataset, val_dataset)
        """
        n = len(self)
        n_train = max(1, round(n * ratio))
        if n >= 2 and n_train >= n:
            n_train = n - 1

        train_ds = PreferenceDataset(
            self._chosen[:n_train],
            self._rejected[:n_train],
            self._margins[:n_train],
        )
        val_ds = PreferenceDataset(
            self._chosen[n_train:],
            self._rejected[n_train:],
            self._margins[n_train:],
        )
        return train_ds, val_ds


# ---------------------------------------------------------------------------
# RewardModelTrainer
# ---------------------------------------------------------------------------


class RewardModelTrainer:
    """Full training pipeline for a reward model.

    Args:
        reward_model: the RewardModel to train
        optimizer:    PyTorch optimizer
        bt_loss:      BradleyTerryLoss instance
        normalizer:   RewardNormalizer instance
    """

    def __init__(
        self,
        reward_model: RewardModel,
        optimizer: torch.optim.Optimizer,
        bt_loss: BradleyTerryLoss,
        normalizer: RewardNormalizer,
    ) -> None:
        self.reward_model = reward_model
        self.optimizer = optimizer
        self.bt_loss = bt_loss
        self.normalizer = normalizer

    def train_step(
        self,
        chosen_ids: Tensor,
        rejected_ids: Tensor,
    ) -> dict:
        """Perform one gradient update step.

        Args:
            chosen_ids:   (B, T) token ids for chosen responses
            rejected_ids: (B, T) token ids for rejected responses

        Returns:
            dict with keys: loss, accuracy, chosen_mean, rejected_mean, reward_margin
        """
        self.reward_model.train()
        self.optimizer.zero_grad()

        chosen_rewards = self.reward_model(chosen_ids)
        rejected_rewards = self.reward_model(rejected_ids)

        loss, accuracy = self.bt_loss(chosen_rewards, rejected_rewards)
        loss.backward()
        self.optimizer.step()

        all_rewards = torch.cat([chosen_rewards.detach(), rejected_rewards.detach()])
        self.normalizer.update(all_rewards)

        with torch.no_grad():
            chosen_mean = chosen_rewards.mean().item()
            rejected_mean = rejected_rewards.mean().item()
            reward_margin = (chosen_rewards - rejected_rewards).mean().item()

        return {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "chosen_mean": chosen_mean,
            "rejected_mean": rejected_mean,
            "reward_margin": reward_margin,
        }

    def score_pairs(
        self,
        chosen_ids: Tensor,
        rejected_ids: Tensor,
    ) -> dict:
        """Assess the reward model on a preference batch without updating weights.

        Args:
            chosen_ids:   (B, T) token ids for chosen responses
            rejected_ids: (B, T) token ids for rejected responses

        Returns:
            dict with keys: accuracy, mean_margin, consistency_rate
        """
        self.reward_model.eval()
        with torch.no_grad():
            chosen_rewards = self.reward_model(chosen_ids)
            rejected_rewards = self.reward_model(rejected_ids)

            diff = chosen_rewards - rejected_rewards
            accuracy = (diff > 0).float().mean().item()
            mean_margin = diff.mean().item()
            consistency_rate = accuracy

        return {
            "accuracy": accuracy,
            "mean_margin": mean_margin,
            "consistency_rate": consistency_rate,
        }

    # Public alias used by tests
    def evaluate(self, chosen_ids: Tensor, rejected_ids: Tensor) -> dict:
        """Alias for score_pairs -- assess reward model on preference batch."""
        return self.score_pairs(chosen_ids, rejected_ids)

    def pairwise_agreement(self, rewards_list: list) -> float:
        """Compute fraction of pairs whose ordering is consistent (descending).

        Args:
            rewards_list: list of scalar Tensors (one per response, best first)

        Returns:
            fraction of pairs in consistent order in [0, 1]
        """
        n = len(rewards_list)
        if n < 2:
            return 1.0

        def to_scalar(r):
            if isinstance(r, Tensor):
                return r.float().item()
            return float(r)

        vals = [to_scalar(r) for r in rewards_list]
        total = 0
        consistent = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += 1
                if vals[i] > vals[j]:
                    consistent += 1

        return consistent / total if total > 0 else 1.0
