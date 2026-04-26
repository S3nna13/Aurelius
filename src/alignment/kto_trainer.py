"""Aurelius — KTO Trainer: Kahneman-Tversky Optimization.

Ethayarajh et al. (2024) "KTO: Model Alignment as Prospect Theoretic
Optimization" — https://arxiv.org/abs/2402.01306 (Apache-2.0)

KTO aligns language models using binary desirability labels on individual
responses rather than paired (chosen, rejected) comparisons. Inspired by
Kahneman-Tversky prospect theory, where losses loom larger than gains.

For desirable outputs:
    L_d = w_d * (1 - sigmoid(β · (log_ratio - KL)))

For undesirable outputs:
    L_u = w_u * sigmoid(β · (log_ratio - KL))

KL proxy (non-negative):
    KL = max(0, mean(policy_logprobs) - mean(ref_logprobs))

Total loss = mean(L_d over desirable) + mean(L_u over undesirable)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class KTOConfig:
    """Configuration for KTOTrainer.

    Attributes:
        learning_rate:      Optimiser learning rate.
        beta:               Temperature / KL coefficient.
        desirable_weight:   Weight applied to the desirable loss component (w_d).
        undesirable_weight: Weight applied to the undesirable loss component (w_u).
        batch_size:         Training batch size.
        max_seq_len:        Maximum sequence length in tokens.
    """

    learning_rate: float = 1e-6
    beta: float = 0.1
    desirable_weight: float = 1.0
    undesirable_weight: float = 1.0
    batch_size: int = 4
    max_seq_len: int = 512


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


@dataclass
class KTOBatch:
    """A single KTO training batch.

    Attributes:
        input_ids:  Token IDs, shape B×L.
        labels:     Target token IDs, shape B×L. Use -100 to mask positions.
        mask:       Attention mask, shape B×L or None.
        desirable:  Boolean tensor of shape (B,); True = desirable completion.
    """

    input_ids: Tensor
    labels: Tensor
    mask: Tensor | None
    desirable: Tensor  # (B,) bool


# ---------------------------------------------------------------------------
# Loss Module
# ---------------------------------------------------------------------------


class KTOLoss(nn.Module):
    """KTO objective as a differentiable nn.Module.

    Applies prospect-theoretic weighting to desirable and undesirable samples:

    KL proxy (batch-level):
        KL = max(0, mean(policy_logprobs) - mean(ref_logprobs))

    For desirable samples (desirable == True):
        loss_i = w_d * (1 - sigmoid(β · (log_ratio_i - KL)))

    For undesirable samples (desirable == False):
        loss_i = w_u * sigmoid(β · (log_ratio_i - KL))

    Final loss = mean(desirable losses) + mean(undesirable losses)
    (each group's mean is 0 if no samples in that group).

    Args:
        beta:               Temperature coefficient.
        desirable_weight:   Weight w_d for desirable samples.
        undesirable_weight: Weight w_u for undesirable samples.
    """

    def __init__(
        self,
        beta: float = 0.1,
        desirable_weight: float = 1.0,
        undesirable_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.register_buffer("beta", torch.tensor(beta, dtype=torch.float32))
        self.register_buffer(
            "desirable_weight", torch.tensor(desirable_weight, dtype=torch.float32)
        )
        self.register_buffer(
            "undesirable_weight", torch.tensor(undesirable_weight, dtype=torch.float32)
        )

    def forward(
        self,
        policy_logprobs: Tensor,
        ref_logprobs: Tensor,
        desirable: Tensor,
    ) -> Tensor:
        """Compute the KTO loss.

        Args:
            policy_logprobs: Per-sequence log-probs under policy. Shape (B,).
            ref_logprobs:    Per-sequence log-probs under reference. Shape (B,).
            desirable:       Boolean mask (B,); True = desirable.

        Returns:
            Scalar loss tensor.
        """
        log_ratio = policy_logprobs - ref_logprobs  # (B,)

        # KL proxy: clamp to non-negative
        kl = (policy_logprobs.mean() - ref_logprobs.mean()).clamp(min=0.0)

        des_mask = desirable.bool()
        unds_mask = ~des_mask

        zero = log_ratio.new_tensor(0.0)

        if des_mask.any():
            lr_des = log_ratio[des_mask]
            loss_des = (
                self.desirable_weight * (1.0 - torch.sigmoid(self.beta * (lr_des - kl)))
            ).mean()
        else:
            loss_des = zero

        if unds_mask.any():
            lr_unds = log_ratio[unds_mask]
            loss_unds = (self.undesirable_weight * torch.sigmoid(self.beta * (lr_unds - kl))).mean()
        else:
            loss_unds = zero

        return loss_des + loss_unds


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class KTOTrainer:
    """KTO trainer: binary desirability preference optimization.

    Unlike DPO, this trainer does not require paired examples. Each response
    carries a binary desirability label and the loss is computed with a shared
    KL proxy that grounds the implicit reward in the reference distribution.

    Args:
        policy:    The trainable policy model (nn.Module).
        ref_model: Frozen reference model (nn.Module).
        cfg:       KTOConfig with hyperparameters.
        optimizer: Optimiser for the policy model parameters.
    """

    def __init__(
        self,
        policy: nn.Module,
        ref_model: nn.Module,
        cfg: KTOConfig,
        optimizer: optim.Optimizer,
    ) -> None:
        self.policy = policy
        self.ref_model = ref_model
        self.cfg = cfg
        self.optimizer = optimizer
        self._loss_fn = KTOLoss(
            beta=cfg.beta,
            desirable_weight=cfg.desirable_weight,
            undesirable_weight=cfg.undesirable_weight,
        )

        # Freeze reference model
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Core primitive
    # ------------------------------------------------------------------

    def compute_logprobs(
        self,
        model: nn.Module,
        input_ids: Tensor,
        labels: Tensor,
        mask: Tensor | None,
    ) -> Tensor:
        """Compute per-sequence mean log-probabilities.

        Performs a forward pass through *model*, extracts per-token log-probs
        for label positions, masks padding, and returns the mean over valid
        tokens for each sequence.

        Args:
            model:     A causal LM; forward(input_ids) → logits (B, T, V).
            input_ids: Token IDs, shape (B, T).
            labels:    Target token IDs, shape (B, T). -100 positions are masked.
            mask:      Optional binary mask (B, T); 1 = valid, 0 = padding.
                       If None, non-(-100) positions in labels are used.

        Returns:
            Per-sequence mean log-prob, shape (B,).
        """
        logits = model(input_ids)  # (B, T, V)
        log_probs = F.log_softmax(logits, dim=-1)  # (B, T, V)

        labels_clamped = labels.clamp(min=0)  # (B, T)
        token_lp = log_probs.gather(dim=2, index=labels_clamped.unsqueeze(-1)).squeeze(-1)  # (B, T)

        if mask is not None:
            valid = mask.float()
        else:
            valid = (labels != -100).float()  # (B, T)

        valid_count = valid.sum(dim=-1).clamp(min=1.0)  # (B,)
        return (token_lp * valid).sum(dim=-1) / valid_count  # (B,)

    # ------------------------------------------------------------------
    # Training interface
    # ------------------------------------------------------------------

    def train_step(self, batch: KTOBatch) -> dict[str, float]:
        """Perform one KTO gradient update.

        Args:
            batch: KTOBatch with token IDs, labels, mask, and desirability flags.

        Returns:
            Dict with keys:
            - ``"loss"``             — scalar training loss.
            - ``"kto_desirable"``    — mean per-sequence loss on desirable samples.
            - ``"kto_undesirable"``  — mean per-sequence loss on undesirable samples.
            - ``"kl_proxy"``         — estimated KL proxy value.
        """
        self.policy.train()

        # Policy log-probs (differentiable)
        policy_lp = self.compute_logprobs(self.policy, batch.input_ids, batch.labels, batch.mask)

        # Reference log-probs (no gradient)
        with torch.no_grad():
            ref_lp = self.compute_logprobs(
                self.ref_model, batch.input_ids, batch.labels, batch.mask
            )

        loss = self._loss_fn(policy_lp, ref_lp, batch.desirable)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Compute per-group metrics (no grad)
        with torch.no_grad():
            log_ratio = policy_lp - ref_lp
            kl_proxy = (policy_lp.mean() - ref_lp.mean()).clamp(min=0.0).item()

            des_mask = batch.desirable.bool()
            unds_mask = ~des_mask

            if des_mask.any():
                lr_des = log_ratio[des_mask]
                kto_des = (
                    (
                        self.cfg.desirable_weight
                        * (
                            1.0
                            - torch.sigmoid(self._loss_fn.beta * (lr_des - torch.tensor(kl_proxy)))
                        )
                    )
                    .mean()
                    .item()
                )
            else:
                kto_des = 0.0

            if unds_mask.any():
                lr_unds = log_ratio[unds_mask]
                kto_unds = (
                    (
                        self.cfg.undesirable_weight
                        * torch.sigmoid(self._loss_fn.beta * (lr_unds - torch.tensor(kl_proxy)))
                    )
                    .mean()
                    .item()
                )
            else:
                kto_unds = 0.0

        return {
            "loss": loss.item(),
            "kto_desirable": kto_des,
            "kto_undesirable": kto_unds,
            "kl_proxy": kl_proxy,
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from src.alignment import ALIGNMENT_REGISTRY  # noqa: E402

ALIGNMENT_REGISTRY["kto"] = KTOTrainer
