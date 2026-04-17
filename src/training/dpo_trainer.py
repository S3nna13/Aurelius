"""Direct Preference Optimization (DPO) trainer.

Implements DPO from Rafailov et al. 2023 -- reference-free preference learning.
Pure native PyTorch only; no external dependencies beyond stdlib + torch.

Loss formula:
    L_DPO = -log sigmoid(beta * (log pi_theta(y_w|x) - log pi_ref(y_w|x))
                         - beta * (log pi_theta(y_l|x) - log pi_ref(y_l|x)))
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# DPOLoss
# ---------------------------------------------------------------------------

class DPOLoss(nn.Module):
    """Core DPO loss computation.

    Args:
        beta: KL regularisation strength (default 0.1).
    """

    def __init__(self, beta: float = 0.1) -> None:
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen_logps: Tensor,
        policy_rejected_logps: Tensor,
        ref_chosen_logps: Tensor,
        ref_rejected_logps: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute DPO loss and per-sample rewards.

        Args:
            policy_chosen_logps:   shape (batch,) -- log pi_theta(y_w | x)
            policy_rejected_logps: shape (batch,) -- log pi_theta(y_l | x)
            ref_chosen_logps:      shape (batch,) -- log pi_ref(y_w | x)
            ref_rejected_logps:    shape (batch,) -- log pi_ref(y_l | x)

        Returns:
            loss:            scalar mean DPO loss
            chosen_reward:   detached per-sample reward for chosen  (batch,)
            rejected_reward: detached per-sample reward for rejected (batch,)
        """
        chosen_reward = self.beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_reward = self.beta * (policy_rejected_logps - ref_rejected_logps)

        logits_diff = chosen_reward - rejected_reward   # (batch,)
        loss = -F.logsigmoid(logits_diff).mean()

        return loss, chosen_reward.detach(), rejected_reward.detach()


# ---------------------------------------------------------------------------
# SequenceLogProbs
# ---------------------------------------------------------------------------

class SequenceLogProbs:
    """Compute per-sequence summed log-probabilities from next-token logits."""

    @staticmethod
    def compute(
        logits: Tensor,
        labels: Tensor,
        ignore_index: int = -100,
    ) -> Tensor:
        """Return summed log-probs per sequence, ignoring masked positions.

        Args:
            logits: shape (batch, seq_len, vocab_size)
            labels: shape (batch, seq_len); positions with ignore_index are skipped

        Returns:
            log_probs: shape (batch,)
        """
        # Shift for next-token prediction: logits[t] predicts labels[t+1]
        shift_logits = logits[:, :-1, :].contiguous()   # (B, T-1, V)
        shift_labels = labels[:, 1:].contiguous()        # (B, T-1)

        log_probs_all = F.log_softmax(shift_logits, dim=-1)  # (B, T-1, V)

        # Replace ignore positions with 0 for safe gather, mask out after
        gather_labels = shift_labels.clone()
        gather_labels[gather_labels == ignore_index] = 0
        token_logps = log_probs_all.gather(
            dim=-1, index=gather_labels.unsqueeze(-1)
        ).squeeze(-1)                                    # (B, T-1)

        mask = (shift_labels != ignore_index).float()    # (B, T-1)
        seq_logps = (token_logps * mask).sum(dim=-1)     # (B,)
        return seq_logps


# ---------------------------------------------------------------------------
# ReferenceModelManager
# ---------------------------------------------------------------------------

class ReferenceModelManager:
    """Manages a frozen deep-copied reference model.

    Args:
        model: the policy model to copy and freeze.
    """

    def __init__(self, model: nn.Module) -> None:
        self._ref_model: nn.Module = copy.deepcopy(model)
        for param in self._ref_model.parameters():
            param.requires_grad_(False)
        self._ref_model.eval()

    def compute_logps(self, input_ids: Tensor, labels: Tensor) -> Tensor:
        """Compute reference log-probs under no_grad.

        Args:
            input_ids: shape (batch, seq_len)
            labels:    shape (batch, seq_len)

        Returns:
            log_probs: shape (batch,)
        """
        with torch.no_grad():
            logits = self._ref_model(input_ids)
            return SequenceLogProbs.compute(logits, labels)

    def is_frozen(self) -> bool:
        """Return True iff every reference parameter has requires_grad=False."""
        return all(not p.requires_grad for p in self._ref_model.parameters())

    @property
    def model(self) -> nn.Module:
        return self._ref_model


# ---------------------------------------------------------------------------
# DPOTrainer
# ---------------------------------------------------------------------------

class DPOTrainer:
    """Training loop for DPO fine-tuning.

    Args:
        policy_model:       the trainable policy nn.Module
        ref_model_manager:  ReferenceModelManager wrapping the frozen ref
        dpo_loss:           DPOLoss instance
        optimizer:          torch Optimizer for policy_model parameters
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model_manager: ReferenceModelManager,
        dpo_loss: DPOLoss,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.policy_model = policy_model
        self.ref_model_manager = ref_model_manager
        self.dpo_loss = dpo_loss
        self.optimizer = optimizer

    def train_step(
        self,
        chosen_ids: Tensor,
        chosen_labels: Tensor,
        rejected_ids: Tensor,
        rejected_labels: Tensor,
    ) -> dict[str, Any]:
        """Run one DPO training step.

        Args:
            chosen_ids:      (batch, seq_len) -- preferred sequence tokens
            chosen_labels:   (batch, seq_len) -- labels for preferred seq
            rejected_ids:    (batch, seq_len) -- dis-preferred sequence tokens
            rejected_labels: (batch, seq_len) -- labels for dis-preferred seq

        Returns:
            dict with keys: loss, chosen_reward, rejected_reward, reward_margin
        """
        self.policy_model.train()
        self.optimizer.zero_grad()

        # Policy log-probs (with grad)
        policy_chosen_logits = self.policy_model(chosen_ids)
        policy_chosen_logps = SequenceLogProbs.compute(
            policy_chosen_logits, chosen_labels
        )

        policy_rejected_logits = self.policy_model(rejected_ids)
        policy_rejected_logps = SequenceLogProbs.compute(
            policy_rejected_logits, rejected_labels
        )

        # Reference log-probs (no_grad inside manager)
        ref_chosen_logps = self.ref_model_manager.compute_logps(
            chosen_ids, chosen_labels
        )
        ref_rejected_logps = self.ref_model_manager.compute_logps(
            rejected_ids, rejected_labels
        )

        loss, chosen_reward, rejected_reward = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
        )

        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=1.0)
        self.optimizer.step()

        reward_margin = chosen_reward - rejected_reward

        return {
            "loss": loss.detach(),
            "chosen_reward": chosen_reward,
            "rejected_reward": rejected_reward,
            "reward_margin": reward_margin,
        }


# ---------------------------------------------------------------------------
# PreferenceDataset
# ---------------------------------------------------------------------------

class PreferenceDataset(Dataset):
    """Simple dataset of (chosen, rejected) token-ID pairs.

    Args:
        chosen_ids_list:   list of 1-D tensors, each shape (seq_len,)
        rejected_ids_list: list of 1-D tensors, each shape (seq_len,)
    """

    def __init__(
        self,
        chosen_ids_list: list[Tensor],
        rejected_ids_list: list[Tensor],
    ) -> None:
        if len(chosen_ids_list) != len(rejected_ids_list):
            raise ValueError(
                "chosen_ids_list and rejected_ids_list must have the same length"
            )
        self._chosen = chosen_ids_list
        self._rejected = rejected_ids_list

    def __len__(self) -> int:
        return len(self._chosen)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self._chosen[idx], self._rejected[idx]

    @staticmethod
    def collate_fn(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        """Stack chosen and rejected tensors into batches.

        Assumes all sequences within a batch share the same length.

        Args:
            batch: list of (chosen_ids, rejected_ids) tuples

        Returns:
            (chosen_batch, rejected_batch) -- each shape (batch, seq_len)
        """
        chosen_list, rejected_list = zip(*batch)
        return (
            torch.stack(list(chosen_list), dim=0),
            torch.stack(list(rejected_list), dim=0),
        )
