"""Offline Reward Model Training — Bradley-Terry pairwise preference model.

Implements the reward model training objective from InstructGPT (Ziegler et al.
2019, Ouyang et al. 2022).  A scalar reward head is trained on top of frozen
base-model hidden states using the Bradley-Terry pairwise preference model:

    P(y_w > y_l | x) = sigmoid(r(x, y_w) - r(x, y_l))

    L_BT = -E[log sigmoid(r_w - r_l - margin)]
         = -E[F.logsigmoid(r_w - r_l - margin)]

The reward head is a single linear projection from the last *valid* token
hidden state (located via attention_mask) to a scalar.

Optional features:
    center_rewards  --  subtract the batch mean of all rewards before computing
                        the loss for numerical stability (does not change the
                        relative ordering of rewards).
    margin          --  require r_w - r_l > margin before the loss becomes
                        zero, encouraging a minimum preference gap.

Pure native PyTorch only; no external dependencies beyond stdlib + torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# RewardModelConfig
# ---------------------------------------------------------------------------

@dataclass
class RewardModelConfig:
    """Configuration for the Bradley-Terry reward model.

    Attributes:
        d_model:        Hidden dimension of the base language model.
        dropout:        Dropout probability applied before the linear projection.
                        Set to 0.0 to disable.
        center_rewards: If True, subtract the batch mean of all rewards before
                        computing the Bradley-Terry loss.  This stabilises
                        training without changing the sign of r_w - r_l.
        margin:         Minimum required reward gap.  The loss objective becomes
                        -logsigmoid(r_w - r_l - margin), so the model must learn
                        a gap of at least *margin* to achieve zero loss.
    """

    d_model: int = 2048
    dropout: float = 0.0
    center_rewards: bool = True
    margin: float = 0.0


# ---------------------------------------------------------------------------
# RewardBatch
# ---------------------------------------------------------------------------

@dataclass
class RewardBatch:
    """Input batch for a single reward-model training step.

    The hidden states are the *final-layer* outputs of the (frozen) base model
    for the full chosen / rejected response sequences.  The linear reward head
    extracts the last valid token from each sequence using the masks.

    Shapes:
        chosen_hidden   -- [B, T_w, d_model]
        rejected_hidden -- [B, T_l, d_model]
        chosen_mask     -- [B, T_w]  binary (1 = valid, 0 = padding)
        rejected_mask   -- [B, T_l]  binary
    """

    chosen_hidden: Tensor
    rejected_hidden: Tensor
    chosen_mask: Tensor
    rejected_mask: Tensor


# ---------------------------------------------------------------------------
# RewardHead
# ---------------------------------------------------------------------------

class RewardHead(nn.Module):
    """Scalar reward regression head.

    Extracts the hidden state at the last *valid* token position (as indicated
    by the attention mask) and projects it to a scalar reward via a single
    linear layer.

    Args:
        config: :class:`RewardModelConfig` instance.
    """

    def __init__(self, config: RewardModelConfig) -> None:
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(p=config.dropout) if config.dropout > 0.0 else nn.Identity()
        self.linear = nn.Linear(config.d_model, 1, bias=True)

    def _last_token_index(self, mask: Tensor) -> Tensor:
        """Return the index of the last valid token for each sequence.

        Scans the mask from left to right and returns the index of the
        rightmost 1.  If a row is all-zeros (degenerate case), falls back to
        the final position (T - 1).

        Args:
            mask: Binary mask, shape ``[B, T]``.

        Returns:
            Integer index tensor, shape ``[B]``.
        """
        T = mask.size(1)
        # Build a position tensor [0, 1, …, T-1] and zero out padding.
        positions = torch.arange(T, device=mask.device, dtype=torch.long)  # [T]
        # For each row, masked positions contribute their index; padding = -1.
        masked_positions = torch.where(
            mask.bool(),
            positions.unsqueeze(0),          # [1, T] → broadcast to [B, T]
            torch.full_like(positions, -1).unsqueeze(0),
        )  # [B, T]
        last_idx = masked_positions.max(dim=1).values  # [B]
        # For all-zero rows (last_idx == -1), use position T-1.
        last_idx = torch.where(last_idx < 0, torch.full_like(last_idx, T - 1), last_idx)
        return last_idx  # [B]

    def forward(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """Compute per-sequence scalar rewards.

        Args:
            hidden_states:  Last-layer hidden states, shape ``[B, T, d_model]``.
            attention_mask: Binary mask (1 = valid), shape ``[B, T]``.

        Returns:
            Scalar reward per sequence, shape ``[B]``.
        """
        B = hidden_states.size(0)
        last_idx = self._last_token_index(attention_mask)  # [B]

        # Gather hidden state at the last valid position for each sequence.
        # Expand last_idx to [B, 1, d_model] for gather.
        idx = last_idx.view(B, 1, 1).expand(B, 1, hidden_states.size(-1))
        last_hidden = hidden_states.gather(dim=1, index=idx).squeeze(1)  # [B, d_model]

        last_hidden = self.dropout(last_hidden)
        reward = self.linear(last_hidden).squeeze(-1)  # [B]
        return reward


# ---------------------------------------------------------------------------
# RewardModelTrainer
# ---------------------------------------------------------------------------

class RewardModelTrainer:
    """Stateless Bradley-Terry reward-model loss computation.

    All public methods are pure functions of their inputs and carry no mutable
    state beyond ``config``.

    Args:
        config: :class:`RewardModelConfig` instance.
    """

    def __init__(self, config: RewardModelConfig | None = None) -> None:
        self.config: RewardModelConfig = config if config is not None else RewardModelConfig()

    # ------------------------------------------------------------------
    # Core loss
    # ------------------------------------------------------------------

    def bradley_terry_loss(
        self,
        reward_chosen: Tensor,    # [B]
        reward_rejected: Tensor,  # [B]
    ) -> Tensor:
        """Compute the Bradley-Terry pairwise preference loss.

        L = -logsigmoid(r_w - r_l - margin).mean()

        If ``config.center_rewards`` is True the batch mean of all rewards is
        subtracted before computing the gap.  Centering does not affect the
        relative ordering; it simply anchors the reward scale around zero,
        improving numerical stability in early training.

        Args:
            reward_chosen:   Scalar rewards for the preferred responses, ``[B]``.
            reward_rejected: Scalar rewards for the dispreferred responses, ``[B]``.

        Returns:
            Scalar loss tensor.
        """
        if self.config.center_rewards:
            all_rewards = torch.cat([reward_chosen, reward_rejected], dim=0)
            mean_reward = all_rewards.mean()
            reward_chosen = reward_chosen - mean_reward
            reward_rejected = reward_rejected - mean_reward

        gap = reward_chosen - reward_rejected - self.config.margin
        loss = -F.logsigmoid(gap).mean()
        return loss

    # ------------------------------------------------------------------
    # Accuracy
    # ------------------------------------------------------------------

    def accuracy(self, reward_chosen: Tensor, reward_rejected: Tensor) -> float:
        """Fraction of pairs where the chosen reward exceeds the rejected reward.

        Args:
            reward_chosen:   ``[B]``
            reward_rejected: ``[B]``

        Returns:
            Float in ``[0.0, 1.0]``.
        """
        with torch.no_grad():
            return (reward_chosen > reward_rejected).float().mean().item()

    # ------------------------------------------------------------------
    # total_loss
    # ------------------------------------------------------------------

    def total_loss(self, batch: RewardBatch, head: RewardHead) -> dict[str, Tensor]:
        """Forward pass through the reward head and compute the BT loss.

        Args:
            batch: :class:`RewardBatch` carrying hidden states and masks.
            head:  :class:`RewardHead` module.

        Returns:
            Dictionary with scalar tensors:

            * ``"loss"``                 -- Bradley-Terry loss (differentiable).
            * ``"reward_chosen_mean"``   -- mean scalar reward for chosen responses.
            * ``"reward_rejected_mean"`` -- mean scalar reward for rejected responses.
            * ``"reward_gap"``           -- mean of (r_w - r_l) across the batch.
        """
        r_w = head(batch.chosen_hidden, batch.chosen_mask)    # [B]
        r_l = head(batch.rejected_hidden, batch.rejected_mask)  # [B]

        loss = self.bradley_terry_loss(r_w, r_l)

        return {
            "loss": loss,
            "reward_chosen_mean": r_w.mean(),
            "reward_rejected_mean": r_l.mean(),
            "reward_gap": (r_w - r_l).mean(),
        }

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def statistics(self, batch: RewardBatch, head: RewardHead) -> dict[str, float]:
        """Compute diagnostic statistics without building a computation graph.

        Args:
            batch: :class:`RewardBatch`.
            head:  :class:`RewardHead` module.

        Returns:
            Dictionary of plain Python floats:

            * ``"accuracy"``              -- fraction where r_w > r_l
            * ``"reward_chosen_mean"``    -- mean chosen reward
            * ``"reward_rejected_mean"``  -- mean rejected reward
            * ``"reward_gap_mean"``       -- mean (r_w - r_l)
        """
        with torch.no_grad():
            r_w = head(batch.chosen_hidden, batch.chosen_mask)
            r_l = head(batch.rejected_hidden, batch.rejected_mask)

            return {
                "accuracy": (r_w > r_l).float().mean().item(),
                "reward_chosen_mean": r_w.mean().item(),
                "reward_rejected_mean": r_l.mean().item(),
                "reward_gap_mean": (r_w - r_l).mean().item(),
            }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from src.training import TRAINING_REGISTRY  # noqa: E402

TRAINING_REGISTRY["reward_modeling"] = RewardModelTrainer
