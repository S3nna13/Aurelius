"""Multi-Token Reward Model: assigns a reward to every token position.

Distinct from process_reward.py (step-level scoring) and sequence_reward.py
(single scalar per sequence). This module provides token-level credit
assignment for RL training, enabling token-level PPO or weighted DPO.

Key classes:
    MultiTokenRMConfig  -- dataclass holding all hyperparameters
    TokenRewardHead     -- linear head mapping hidden states to per-token rewards
    MultiTokenRewardModel -- full model: backbone + head + aggregation
    MultiTokenRMLoss    -- joint token-level + sequence-level MSE loss
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MultiTokenRMConfig:
    """Configuration for the Multi-Token Reward Model."""

    d_model: int = 64
    dropout: float = 0.0
    discount: float = 0.99
    # How to aggregate per-token rewards into a single sequence reward.
    # "mean"         -- simple mean over valid (non-masked) token positions
    # "last"         -- reward of the final valid token
    # "weighted_sum" -- learned per-token weights (softmax) weighted sum
    aggregate: str = "mean"
    token_weight: float = 0.5
    seq_weight: float = 0.5


# ---------------------------------------------------------------------------
# TokenRewardHead
# ---------------------------------------------------------------------------

class TokenRewardHead(nn.Module):
    """Linear projection from hidden states to per-token scalar rewards.

    Args:
        d_model: Hidden dimension of the backbone.
        dropout: Dropout probability applied before the linear layer.
    """

    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, 1, bias=True)
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Map hidden states to per-token rewards.

        Args:
            hidden_states: (batch, seq, d_model)

        Returns:
            (batch, seq) per-token reward tensor.
        """
        x = self.dropout(hidden_states)          # (B, T, d_model)
        rewards = self.linear(x).squeeze(-1)     # (B, T)
        return rewards


# ---------------------------------------------------------------------------
# MultiTokenRewardModel
# ---------------------------------------------------------------------------

class MultiTokenRewardModel(nn.Module):
    """Full multi-token reward model: backbone + TokenRewardHead.

    The backbone is provided as a callable factory ``backbone_fn`` so the
    model is not tied to any specific architecture.  The factory is called
    once during ``__init__`` and the resulting ``nn.Module`` is stored as
    ``self.backbone``.

    Per-token rewards are computed by extracting the final hidden states from
    the backbone via a forward hook on ``backbone.norm`` (AureliusTransformer
    convention).  If the backbone does not expose a ``.norm`` attribute the
    hidden states can be provided directly by subclassing and overriding
    ``_get_hidden_states``.

    Args:
        backbone_fn: Callable returning an ``nn.Module`` backbone.
        config:      ``MultiTokenRMConfig`` instance.
    """

    def __init__(
        self,
        backbone_fn: Callable[[], nn.Module],
        config: MultiTokenRMConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.backbone: nn.Module = backbone_fn()
        self.reward_head = TokenRewardHead(config.d_model, config.dropout)

        # Learned per-token weighting for "weighted_sum" aggregation.
        if config.aggregate == "weighted_sum":
            self.token_weight_linear = nn.Linear(config.d_model, 1, bias=False)
        else:
            self.token_weight_linear = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run backbone and return final hidden states (B, T, d_model).

        Uses a forward hook on ``self.backbone.norm``.
        """
        captured: list[torch.Tensor] = []

        hook = self.backbone.norm.register_forward_hook(
            lambda m, i, o: captured.append(o)
        )
        try:
            self.backbone(input_ids)
        finally:
            hook.remove()

        return captured[0]  # (B, T, d_model)

    def _aggregate(
        self,
        token_rewards: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Aggregate (B, T) token rewards to (B,) sequence reward.

        Args:
            token_rewards:  (B, T)
            hidden_states:  (B, T, d_model) — used only for "weighted_sum"
            attention_mask: (B, T) optional; 1 = valid token, 0 = padding

        Returns:
            (B,) sequence-level reward.
        """
        method = self.config.aggregate

        if method == "mean":
            if attention_mask is None:
                return token_rewards.mean(dim=1)
            mask = attention_mask.float()
            counts = mask.sum(dim=1).clamp(min=1.0)
            return (token_rewards * mask).sum(dim=1) / counts

        elif method == "last":
            if attention_mask is None:
                return token_rewards[:, -1]
            lengths = attention_mask.long().sum(dim=1)
            last_idx = (lengths - 1).clamp(min=0)  # (B,)
            B = token_rewards.size(0)
            return token_rewards[torch.arange(B, device=token_rewards.device), last_idx]

        elif method == "weighted_sum":
            # Softmax over valid positions, then weighted sum.
            assert self.token_weight_linear is not None
            raw_w = self.token_weight_linear(hidden_states).squeeze(-1)  # (B, T)
            if attention_mask is not None:
                mask_bool = attention_mask.bool()
                raw_w = raw_w.masked_fill(~mask_bool, float("-inf"))
            weights = torch.softmax(raw_w, dim=-1)           # (B, T)
            return (weights * token_rewards).sum(dim=1)      # (B,)

        else:
            raise ValueError(
                f"Unknown aggregate method: {method!r}. "
                "Choose 'mean', 'last', or 'weighted_sum'."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-token and sequence-level rewards.

        Args:
            input_ids:      (batch, seq) integer token ids.
            attention_mask: (batch, seq) float or bool; 1 = valid, 0 = padding.

        Returns:
            token_rewards:    (batch, seq)
            sequence_reward:  (batch,)
        """
        hidden_states = self._get_hidden_states(input_ids)   # (B, T, d_model)

        # Zero out masked positions in hidden states before scoring so that
        # masked tokens do not receive meaningful rewards.
        if attention_mask is not None:
            mask_expanded = attention_mask.float().unsqueeze(-1)  # (B, T, 1)
            hidden_states = hidden_states * mask_expanded

        token_rewards = self.reward_head(hidden_states)       # (B, T)

        # Zero masked token rewards explicitly (mask was applied to hidden
        # states, but ensure output zeros too).
        if attention_mask is not None:
            token_rewards = token_rewards * attention_mask.float()

        sequence_reward = self._aggregate(
            token_rewards, hidden_states, attention_mask
        )  # (B,)

        return token_rewards, sequence_reward

    def get_process_rewards(
        self,
        input_ids: torch.Tensor,
        step_boundaries: List[int],
    ) -> torch.Tensor:
        """Aggregate token rewards between step boundaries.

        Given ``step_boundaries = [b0, b1, b2, ...]``, the steps are:
            step 0: tokens [0,    b0)
            step 1: tokens [b0,   b1)
            ...
            step n: tokens [b_{n-1}, end)

        Args:
            input_ids:       (1, seq) single sequence.  Batch dim must be 1.
            step_boundaries: List of integer token positions that mark the
                             *end* (exclusive) of each step.  The final step
                             runs to the end of the sequence automatically.

        Returns:
            (n_steps,) mean token reward for each step.
        """
        with torch.no_grad():
            token_rewards, _ = self.forward(input_ids)  # (1, T)

        rewards_1d = token_rewards[0]  # (T,)
        T = rewards_1d.size(0)

        boundaries = sorted(set(b for b in step_boundaries if 0 < b <= T))
        starts = [0] + boundaries
        ends = boundaries + [T]

        step_rewards = []
        for s, e in zip(starts, ends):
            if e > s:
                step_rewards.append(rewards_1d[s:e].mean())
            else:
                step_rewards.append(torch.tensor(0.0, device=rewards_1d.device))

        return torch.stack(step_rewards)  # (n_steps,)

    def compute_advantage(
        self,
        token_rewards: torch.Tensor,
        discount: float = 0.99,
    ) -> torch.Tensor:
        """Compute discounted advantages from per-token rewards (GAE-like).

        Uses a simple discounted-return formulation with zero value baseline:

            G_t = r_t + discount * G_{t+1}
            A_t = G_t  (no value function; equivalent to REINFORCE)

        Because G accumulates future rewards, the *last* token has
        A_{T-1} = r_{T-1} (no future discount penalty), making it the
        position with the highest advantage when all rewards are equal and
        positive — consistent with the spec requirement.

        Args:
            token_rewards: (batch, seq) per-token rewards.
            discount:      Discount factor gamma.

        Returns:
            (batch, seq) advantage estimates.
        """
        B, T = token_rewards.shape
        advantages = torch.zeros_like(token_rewards)
        running = torch.zeros(B, device=token_rewards.device, dtype=token_rewards.dtype)

        for t in reversed(range(T)):
            running = token_rewards[:, t] + discount * running
            advantages[:, t] = running

        return advantages


# ---------------------------------------------------------------------------
# MultiTokenRMLoss
# ---------------------------------------------------------------------------

class MultiTokenRMLoss:
    """Joint token-level + sequence-level MSE loss for reward model training.

    Args:
        token_weight: Weight applied to the per-token MSE loss component.
        seq_weight:   Weight applied to the sequence-level MSE loss component.
    """

    def __init__(
        self,
        token_weight: float = 0.5,
        seq_weight: float = 0.5,
    ) -> None:
        self.token_weight = token_weight
        self.seq_weight = seq_weight

    def forward(
        self,
        token_rewards: torch.Tensor,
        seq_rewards: torch.Tensor,
        token_labels: Optional[torch.Tensor],
        seq_labels: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, dict]:
        """Compute the combined reward-model loss.

        Args:
            token_rewards: (batch, seq) predicted per-token rewards.
            seq_rewards:   (batch,) predicted sequence-level rewards.
            token_labels:  (batch, seq) ground-truth per-token rewards, or None.
            seq_labels:    (batch,) ground-truth sequence-level rewards, or None.

        Returns:
            total_loss: Scalar loss tensor (differentiable).
            metrics:    Dict containing "total_loss", and conditionally
                        "token_loss" and "seq_loss".
        """
        if token_labels is None and seq_labels is None:
            raise ValueError(
                "At least one of token_labels or seq_labels must be provided."
            )

        metrics: dict = {}
        total_loss = torch.zeros(
            1,
            device=token_rewards.device,
            dtype=token_rewards.dtype,
        ).squeeze()

        if token_labels is not None:
            token_loss = F.mse_loss(token_rewards, token_labels)
            total_loss = total_loss + self.token_weight * token_loss
            metrics["token_loss"] = token_loss.item()

        if seq_labels is not None:
            seq_loss = F.mse_loss(seq_rewards, seq_labels)
            total_loss = total_loss + self.seq_weight * seq_loss
            metrics["seq_loss"] = seq_loss.item()

        metrics["total_loss"] = total_loss.item()
        return total_loss, metrics
