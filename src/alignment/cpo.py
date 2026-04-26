"""Contrastive Preference Optimization (CPO) — Xu et al., 2024.

Reference-model-free preference optimization that combines a standard SFT term
with a contrastive margin loss. Avoids the "likelihood displacement" problem in
DPO without requiring a reference model at inference time.

Loss formula:
    L_CPO = -E[log p(y_w|x)]                                         # SFT term
           + -E[log σ(β · (log p(y_w|x) - log p(y_l|x) - δ))]      # contrastive term

where:
    y_w = chosen sequence
    y_l = rejected sequence
    β   = temperature scaling
    δ   = margin between chosen and rejected

References:
    - CPO: Contrastive Preference Optimization: Pushing the Boundaries of LLM
      Performance in Machine Translation (Xu et al., 2024)
      https://arxiv.org/abs/2401.08417
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CPOConfig:
    """Configuration for Contrastive Preference Optimization.

    Attributes:
        beta: Temperature scaling applied to the contrastive margin. Default: 0.1.
        delta: Minimum margin between chosen and rejected log-probs. Default: 0.5.
        sft_weight: Weight multiplier for the SFT (NLL) component. Default: 1.0.
        cpo_weight: Weight multiplier for the contrastive component. Default: 1.0.
        label_smoothing: Label smoothing coefficient in [0, 1). Default: 0.0.
    """

    beta: float = 0.1
    delta: float = 0.5
    sft_weight: float = 1.0
    cpo_weight: float = 1.0
    label_smoothing: float = 0.0


# ---------------------------------------------------------------------------
# Core CPO loss function
# ---------------------------------------------------------------------------


def cpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    config: CPOConfig,
) -> tuple[torch.Tensor, dict]:
    """Compute the CPO loss from pre-computed mean log-probabilities.

    Args:
        policy_chosen_logps: Shape (B,) — mean log-probs per chosen sequence
            (already length-normalized, i.e. mean over non-padding tokens).
        policy_rejected_logps: Shape (B,) — mean log-probs per rejected sequence
            (already length-normalized).
        config: CPOConfig with beta, delta, sft_weight, cpo_weight, label_smoothing.

    Returns:
        (total_loss, metrics) where total_loss is a scalar tensor and metrics is a
        dict with keys 'sft_loss', 'cpo_loss', 'margin'.
    """
    # ---- SFT term: maximise likelihood of chosen responses ----
    # -E[log p(y_w|x)]
    sft_loss = -policy_chosen_logps.mean()

    # ---- Contrastive margin ----
    # β · (log p(y_w|x) - log p(y_l|x) - δ)
    margin = config.beta * (policy_chosen_logps - policy_rejected_logps - config.delta)

    # ---- Contrastive term: -log σ(margin) ----
    if config.label_smoothing > 0.0:
        ls = config.label_smoothing
        contrastive_loss = (-F.logsigmoid(margin) * (1.0 - ls) - F.logsigmoid(-margin) * ls).mean()
    else:
        contrastive_loss = -F.logsigmoid(margin).mean()

    total_loss = config.sft_weight * sft_loss + config.cpo_weight * contrastive_loss

    metrics = {
        "sft_loss": sft_loss.detach().item(),
        "cpo_loss": contrastive_loss.detach().item(),
        "margin": (policy_chosen_logps - policy_rejected_logps).mean().detach().item(),
    }

    return total_loss, metrics


# ---------------------------------------------------------------------------
# CPOTrainer
# ---------------------------------------------------------------------------


class CPOTrainer:
    """Trainer for Contrastive Preference Optimization (CPO).

    No reference model is required. The trainer uses the policy model alone,
    computing separate forward passes for chosen and rejected sequences and
    deriving the CPO loss from their mean log-probabilities.

    Args:
        model: Policy language model. Forward call: logits = model(input_ids),
               where logits has shape (B, T, vocab_size).
        config: CPOConfig with hyperparameters.
        optimizer: Any PyTorch optimizer bound to model.parameters().
    """

    def __init__(
        self,
        model: nn.Module,
        config: CPOConfig,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.model = model
        self.config = config
        self.optimizer = optimizer

    def compute_logps(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mean log-probabilities per sequence over non-padding tokens.

        Uses a next-token prediction style: the log-prob of token t+1 is derived
        from the logits at position t, then masked so only label positions with
        valid (non-padding) tokens contribute.

        Args:
            model: Language model whose forward returns logits of shape (B, T, V).
            input_ids: Shape (B, T) — full token id sequences.
            attention_mask: Shape (B, T) — 1 for real tokens, 0 for padding.
            labels: Shape (B, T) — target token ids; padding positions should be
                    indicated by attention_mask == 0 (the mask drives exclusion).

        Returns:
            Shape (B,) — mean log-prob per sequence (over non-padding response tokens).
        """
        logits = model(input_ids)  # (B, T, V)

        # Next-token log-probs: predict token at position t+1 from logits at t
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # (B, T-1, V)

        # Gather log-prob of the actual next token
        target_ids = labels[:, 1:]  # (B, T-1)
        token_lp = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

        # Mask: only count positions where the target token is valid (not padding)
        mask = attention_mask[:, 1:].float()  # (B, T-1)

        # Mean over non-padding tokens (length normalization)
        token_counts = mask.sum(dim=-1).clamp(min=1.0)  # (B,) — avoid division by zero
        mean_logps = (token_lp * mask).sum(dim=-1) / token_counts  # (B,)

        return mean_logps

    def train_step(self, batch: dict) -> dict:
        """Perform a single CPO training step.

        Args:
            batch: Dict with keys:
                - chosen_input_ids:   (B, T) chosen token ids
                - chosen_labels:      (B, T) chosen labels (same as ids for LM)
                - rejected_input_ids: (B, T) rejected token ids
                - rejected_labels:    (B, T) rejected labels

                Optionally:
                - chosen_attention_mask:   (B, T) — defaults to all-ones if absent
                - rejected_attention_mask: (B, T) — defaults to all-ones if absent

        Returns:
            Dict with keys: 'loss', 'sft_loss', 'cpo_loss', 'margin'.
        """
        self.model.train()

        chosen_ids = batch["chosen_input_ids"]
        chosen_labels = batch["chosen_labels"]
        rejected_ids = batch["rejected_input_ids"]
        rejected_labels = batch["rejected_labels"]

        # Build default attention masks (all real tokens) if not provided
        chosen_mask = batch.get(
            "chosen_attention_mask",
            torch.ones_like(chosen_ids),
        )
        rejected_mask = batch.get(
            "rejected_attention_mask",
            torch.ones_like(rejected_ids),
        )

        # Compute mean log-probs for chosen and rejected
        policy_chosen_logps = self.compute_logps(self.model, chosen_ids, chosen_mask, chosen_labels)
        policy_rejected_logps = self.compute_logps(
            self.model, rejected_ids, rejected_mask, rejected_labels
        )

        total_loss, metrics = cpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            self.config,
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "sft_loss": metrics["sft_loss"],
            "cpo_loss": metrics["cpo_loss"],
            "margin": metrics["margin"],
        }
