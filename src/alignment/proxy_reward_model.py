"""Proxy Reward Models — lightweight learned value estimators for RLHF / alignment."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _SelfAttention(nn.Module):
    """Single-head self-attention (kept minimal for proxy/lightweight use)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self._scale = math.sqrt(d_model)

    def forward(self, x: Tensor) -> Tensor:  # x: [B, T, d_model]
        B, T, _ = x.shape
        qkv = self.qkv(x)  # [B, T, 3*d]
        q, k, v = qkv.chunk(3, dim=-1)
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / self._scale, dim=-1)  # [B,T,T]
        return self.out_proj(torch.bmm(attn, v))  # [B, T, d_model]


class _FFN(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * 4)
        self.fc2 = nn.Linear(d_model * 4, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class _EncoderBlock(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.attn = _SelfAttention(d_model)
        self.ffn = _FFN(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# RewardBackbone
# ---------------------------------------------------------------------------


class RewardBackbone(nn.Module):
    """Simple transformer encoder: embedding + n blocks + pooling.

    Args:
        d_model:    Model dimension.
        vocab_size: Vocabulary size (token embedding table).
        n_layers:   Number of encoder blocks.
        pooling:    One of "last" (final token), "mean" (average over T),
                    "cls" (prepend a learned CLS token, return its state).
    """

    VALID_POOLING = {"last", "mean", "cls"}

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        n_layers: int = 2,
        pooling: str = "last",
    ) -> None:
        super().__init__()
        if pooling not in self.VALID_POOLING:
            raise ValueError(f"pooling must be one of {self.VALID_POOLING}, got {pooling!r}")
        self.pooling = pooling
        self.d_model = d_model

        self.embed = nn.Embedding(vocab_size, d_model)
        # Learned positional embeddings — support up to 2048 positions
        self.pos_embed = nn.Embedding(2048, d_model)

        if pooling == "cls":
            # A single learned CLS token prepended at position 0
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList([_EncoderBlock(d_model) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Args:
            input_ids: [B, T]
        Returns:
            pooled:    [B, d_model]
        """
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)  # [1, T]
        x = self.embed(input_ids) + self.pos_embed(positions)  # [B, T, d_model]

        if self.pooling == "cls":
            # Prepend CLS token; shift positional offset by 1 (pos 0 is CLS)
            cls = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
            # Re-embed positions 0..T (now length T+1)
            positions_full = torch.arange(T + 1, device=input_ids.device).unsqueeze(0)
            x_tok = self.embed(input_ids) + self.pos_embed(positions_full[:, 1:])
            cls_pos = self.pos_embed(positions_full[:, :1])  # [1,1,d]
            x = torch.cat([cls + cls_pos, x_tok], dim=1)  # [B, T+1, d_model]

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)  # [B, T(+1), d_model]

        if self.pooling == "last":
            return x[:, -1, :]  # [B, d_model]
        elif self.pooling == "mean":
            return x.mean(dim=1)  # [B, d_model]
        else:  # cls
            return x[:, 0, :]  # [B, d_model]  — CLS position


# ---------------------------------------------------------------------------
# RewardHead
# ---------------------------------------------------------------------------


class RewardHead(nn.Module):
    """MLP head: d_model -> d_model//2 -> n_outputs.

    Args:
        d_model:   Input dimension (from backbone).
        n_outputs: Number of reward outputs (default 1 for scalar reward).
    """

    def __init__(self, d_model: int, n_outputs: int = 1) -> None:
        super().__init__()
        hidden = max(d_model // 2, 1)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_outputs),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, d_model]
        Returns:
            [B, n_outputs]
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# ProxyRewardModel
# ---------------------------------------------------------------------------


class ProxyRewardModel(nn.Module):
    """Combines a RewardBackbone and RewardHead into a full proxy reward model.

    Args:
        backbone: A RewardBackbone instance.
        head:     A RewardHead instance (must have n_outputs=1 for squeeze).
    """

    def __init__(self, backbone: RewardBackbone, head: RewardHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Args:
            input_ids: [B, T]
        Returns:
            rewards: [B]  (last dim squeezed)
        """
        pooled = self.backbone(input_ids)  # [B, d_model]
        return self.head(pooled).squeeze(-1)  # [B]

    @torch.no_grad()
    def score_batch(self, input_ids: Tensor) -> Tensor:
        """Inference-only scoring; returns detached rewards [B]."""
        return self.forward(input_ids).detach()


# ---------------------------------------------------------------------------
# EnsembleRewardModel
# ---------------------------------------------------------------------------


class EnsembleRewardModel(nn.Module):
    """Ensemble of ProxyRewardModels for uncertainty estimation.

    Args:
        models: List of ProxyRewardModel instances.
    """

    def __init__(self, models: list[ProxyRewardModel]) -> None:
        super().__init__()
        if len(models) == 0:
            raise ValueError("models list must be non-empty")
        self.models = nn.ModuleList(models)

    def forward(self, input_ids: Tensor):
        """
        Args:
            input_ids: [B, T]
        Returns:
            mean_reward: [B]
            std_reward:  [B]  (>= 0)
        """
        rewards = torch.stack([m(input_ids) for m in self.models], dim=0)  # [N, B]
        mean_reward = rewards.mean(dim=0)  # [B]
        std_reward = rewards.std(dim=0, unbiased=False)  # [B]
        # std with a single model degenerates to 0, which is correct
        return mean_reward, std_reward

    def uncertainty_score(self, input_ids: Tensor) -> Tensor:
        """Normalised uncertainty: std / (|mean| + 1).  Always non-negative.

        Args:
            input_ids: [B, T]
        Returns:
            [B]
        """
        mean_reward, std_reward = self.forward(input_ids)
        return std_reward / (mean_reward.abs() + 1.0)


# ---------------------------------------------------------------------------
# RewardModelTrainer
# ---------------------------------------------------------------------------


class RewardModelTrainer:
    """Training utilities for a ProxyRewardModel.

    Args:
        model:  The ProxyRewardModel to train.
        lr:     Learning rate for Adam optimiser.
        margin: Preference margin — rewards for chosen must exceed rejected by
                at least this amount (Bradley-Terry with margin).
    """

    def __init__(
        self,
        model: ProxyRewardModel,
        lr: float,
        margin: float = 0.1,
    ) -> None:
        self.model = model
        self.margin = margin
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def preference_loss(self, chosen_ids: Tensor, rejected_ids: Tensor) -> Tensor:
        """Bradley-Terry preference loss with margin.

        Loss = -log σ(r_chosen - r_rejected - margin)

        Args:
            chosen_ids:   [B, T]
            rejected_ids: [B, T]
        Returns:
            scalar loss
        """
        r_chosen = self.model(chosen_ids)  # [B]
        r_rejected = self.model(rejected_ids)  # [B]
        # Apply margin: we want r_chosen - r_rejected > margin
        logits = r_chosen - r_rejected - self.margin
        loss = -F.logsigmoid(logits).mean()
        return loss

    def regression_loss(self, input_ids: Tensor, targets: Tensor) -> Tensor:
        """MSE regression loss against scalar reward targets.

        Args:
            input_ids: [B, T]
            targets:   [B]
        Returns:
            scalar loss
        """
        preds = self.model(input_ids)  # [B]
        return F.mse_loss(preds, targets)

    def train_step(self, chosen_ids: Tensor, rejected_ids: Tensor) -> Tensor:
        """One gradient-descent step using preference loss.

        Args:
            chosen_ids:   [B, T]
            rejected_ids: [B, T]
        Returns:
            loss scalar (detached float tensor)
        """
        self.optimizer.zero_grad()
        loss = self.preference_loss(chosen_ids, rejected_ids)
        loss.backward()
        self.optimizer.step()
        return loss.detach()

    @torch.no_grad()
    def eval_accuracy(self, chosen_ids: Tensor, rejected_ids: Tensor) -> float:
        """Fraction of pairs where r_chosen > r_rejected.

        Args:
            chosen_ids:   [B, T]
            rejected_ids: [B, T]
        Returns:
            accuracy in [0.0, 1.0]
        """
        r_chosen = self.model(chosen_ids)
        r_rejected = self.model(rejected_ids)
        correct = (r_chosen > r_rejected).float().mean()
        return correct.item()


# ---------------------------------------------------------------------------
# RewardNormalizer
# ---------------------------------------------------------------------------


class RewardNormalizer:
    """Running mean/std normalizer for reward signals.

    Uses exponential moving average to track statistics and normalises each
    batch of rewards to approximately zero-mean, unit-variance.

    Args:
        momentum: EMA coefficient for running statistics (closer to 1 = slower
                  adaptation).  Default 0.99.
    """

    def __init__(self, momentum: float = 0.99) -> None:
        self.momentum = momentum
        self.running_mean: float = 0.0
        self.running_var: float = 1.0
        self._initialised: bool = False

    def update(self, rewards: Tensor) -> Tensor:
        """Update running statistics and return normalised rewards.

        Args:
            rewards: [B]
        Returns:
            normalised rewards: [B]  (approx zero-mean, unit-std)
        """
        batch_mean = rewards.mean().item()
        batch_var = rewards.var(unbiased=False).item()

        if not self._initialised:
            self.running_mean = batch_mean
            self.running_var = max(batch_var, 1e-8)
            self._initialised = True
        else:
            m = self.momentum
            self.running_mean = m * self.running_mean + (1.0 - m) * batch_mean
            self.running_var = m * self.running_var + (1.0 - m) * batch_var
            self.running_var = max(self.running_var, 1e-8)

        std = math.sqrt(self.running_var)
        return (rewards - self.running_mean) / std

    def reset(self) -> None:
        """Reset all running statistics to initial state."""
        self.running_mean = 0.0
        self.running_var = 1.0
        self._initialised = False


# ---------------------------------------------------------------------------
# ProxyRewardConfig
# ---------------------------------------------------------------------------


@dataclass
class ProxyRewardConfig:
    """Configuration dataclass for building a full proxy reward pipeline."""

    d_model: int = 32
    vocab_size: int = 64
    n_layers: int = 2
    pooling: str = "last"
    lr: float = 1e-4
    margin: float = 0.1
    n_ensemble: int = 3
    momentum: float = 0.99
