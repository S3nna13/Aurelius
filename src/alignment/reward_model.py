"""Reward model for GRPO alignment (Bradley-Terry ranking, arXiv:1952.00608).

Wraps an AureliusTransformer backbone with a scalar projection head.
Trained on (chosen, rejected) pairs to score response quality.
Score is extracted from the last token's hidden state.
"""
from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class RewardModelConfig:
    learning_rate: float = 1e-5
    num_epochs: int = 1
    batch_size: int = 4
    max_seq_len: int = 512
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    output_dir: str = "checkpoints/reward_model"


class RewardModel(nn.Module):
    """Transformer backbone with a scalar reward head.

    Args:
        backbone: AureliusTransformer (or any model with embed + layers + norm + lm_head).
                  The backbone is used for its intermediate representations only;
                  lm_head is NOT used.
        freeze_backbone: If True, only the reward head is trainable.
    """

    def __init__(self, backbone: nn.Module, freeze_backbone: bool = False) -> None:
        super().__init__()
        self.backbone = backbone

        # Freeze backbone if requested
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        # Scalar head: project from d_model to 1
        d_model = backbone.config.d_model
        self.reward_head = nn.Linear(d_model, 1, bias=True)
        nn.init.normal_(self.reward_head.weight, std=0.02)
        nn.init.zeros_(self.reward_head.bias)

    def _get_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Extract last-token hidden states from the backbone.

        Runs the backbone forward WITHOUT computing loss (no labels),
        then extracts the hidden state at the last token position.

        Returns: (batch, d_model)
        """
        B, S = input_ids.shape

        x = self.backbone.embed(input_ids)
        freqs_cis = self.backbone.freqs_cis[:S]

        for layer in self.backbone.layers:
            x, _ = layer(x, freqs_cis, mask=None, past_kv=None)

        x = self.backbone.norm(x)  # (B, S, d_model)
        return x[:, -1, :]  # last token: (B, d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute reward scores.

        Args:
            input_ids: (B, seq_len)

        Returns:
            (B,) scalar reward scores.
        """
        hidden = self._get_hidden_states(input_ids)  # (B, d_model)
        return self.reward_head(hidden).squeeze(-1)  # (B,)

    def score(self, input_ids: torch.Tensor) -> float:
        """Score a single sequence. Returns a Python float. Useful as reward_fn."""
        self.eval()
        with torch.no_grad():
            scores = self.forward(input_ids)
        return scores[0].item()


def bradley_terry_loss(
    reward_chosen: torch.Tensor,
    reward_rejected: torch.Tensor,
) -> torch.Tensor:
    """Bradley-Terry ranking loss for preference pairs.

    Minimizing this loss encourages r_chosen > r_rejected.

    Args:
        reward_chosen: (B,) -- scalar rewards for preferred responses.
        reward_rejected: (B,) -- scalar rewards for dispreferred responses.

    Returns:
        Scalar loss.
    """
    return -F.logsigmoid(reward_chosen - reward_rejected).mean()


def build_reward_fn(reward_model: RewardModel, tokenizer, device: str = "cpu"):
    """Build a reward_fn compatible with GRPOTrainer.

    Returns a callable: (prompt: str, response: str) -> float
    """
    def reward_fn(prompt: str, response: str) -> float:
        # Concatenate prompt + response as the full sequence
        text = prompt + response
        ids = tokenizer.encode(text)
        if not ids:
            return 0.0
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        # Truncate to max_seq_len
        if input_ids.shape[1] > reward_model.backbone.config.max_seq_len:
            input_ids = input_ids[:, -reward_model.backbone.config.max_seq_len:]
        return reward_model.score(input_ids)

    return reward_fn


class RewardModelTrainer:
    """Train a RewardModel on (chosen, rejected) token pairs.

    Args:
        model: RewardModel instance.
        cfg: Training configuration.
    """

    def __init__(self, model: RewardModel, cfg: RewardModelConfig | None = None) -> None:
        self.model = model
        self.cfg = cfg or RewardModelConfig()
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.cfg.learning_rate,
        )

    def train_step(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
    ) -> float:
        """One training step on a batch of preference pairs.

        Args:
            chosen_ids: (B, seq_len) -- preferred response token ids.
            rejected_ids: (B, seq_len) -- dispreferred response token ids.

        Returns:
            Loss value as float.
        """
        self.model.train()
        self.optimizer.zero_grad()

        r_chosen = self.model(chosen_ids)
        r_rejected = self.model(rejected_ids)

        loss = bradley_terry_loss(r_chosen, r_rejected)
        loss.backward()

        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()

        return loss.item()
