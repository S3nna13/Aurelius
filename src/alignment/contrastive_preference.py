"""Contrastive Preference Optimization (CPO variants) for Aurelius.

Extends DPO by applying contrastive learning principles:
1. Multi-negative ranking: push chosen against multiple rejected simultaneously.
2. Hard negative mining: select rejected responses closest in reward to chosen.
3. Temperature-scaled InfoNCE loss for sharper preference signals.

Loss variants:
    - 'infonce'  : InfoNCE-style contrastive loss with temperature scaling
    - 'ranking'  : Multi-negative ranking loss (NLL over softmax across negatives)
    - 'triplet'  : Triplet margin loss

References:
    - InfoNCE: Oord et al., 2018 (Representation Learning with CPC)
    - DPO: Rafailov et al., 2023
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ContrastivePrefConfig:
    """Configuration for Contrastive Preference Optimization.

    Attributes:
        temperature: InfoNCE temperature scaling. Lower values produce sharper
            distributions. Default: 0.07.
        n_negatives: Number of rejected responses to contrast against per chosen.
            Default: 4.
        beta: KL-penalty coefficient (analogous to DPO beta). Default: 0.1.
        hard_negative_ratio: Fraction of negatives selected via hard mining
            (closest in reward to chosen). Default: 0.5.
        loss_type: One of 'infonce', 'ranking', 'triplet'. Default: 'infonce'.
    """

    temperature: float = 0.07
    n_negatives: int = 4
    beta: float = 0.1
    hard_negative_ratio: float = 0.5
    loss_type: str = "infonce"


# ---------------------------------------------------------------------------
# Standalone loss functions
# ---------------------------------------------------------------------------

def triplet_preference_loss(
    anchor_logps: torch.Tensor,
    positive_logps: torch.Tensor,
    negative_logps: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Triplet loss for preference learning.

    Penalises configurations where the anchor is not separated from the
    negative more than from the positive by at least `margin`.

    Loss = max(0, margin - (anchor - negative) + (anchor - positive))

    When the margin is satisfied the loss is exactly 0.

    Args:
        anchor_logps:   Shape (B,) -- log-probs for the anchor sequence.
        positive_logps: Shape (B,) -- log-probs for the positive (chosen) sequence.
        negative_logps: Shape (B,) -- log-probs for the negative (rejected) sequence.
        margin:         Minimum desired separation. Default: 1.0.

    Returns:
        Scalar mean triplet loss.
    """
    loss = torch.clamp(
        margin - (anchor_logps - negative_logps) + (anchor_logps - positive_logps),
        min=0.0,
    )
    return loss.mean()


def multi_negative_ranking_loss(
    chosen_logps: torch.Tensor,
    rejected_logps_stack: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Multi-negative ranking loss (InfoNCE over a pool of negatives).

    Treats the chosen response as the single positive and all rejected
    responses as negatives. Computes the NLL of the chosen logp being the
    maximum when all logps are scaled by temperature and passed through softmax.

    Args:
        chosen_logps:         Shape (B,) -- policy log-probs for chosen sequences.
        rejected_logps_stack: Shape (B, n_negatives) -- policy log-probs for
                              rejected sequences.
        temperature:          Temperature for the softmax. Default: 0.07.

    Returns:
        Scalar mean ranking loss.
    """
    # Concatenate: (B, 1 + n_negatives) with chosen first
    all_logps = torch.cat(
        [chosen_logps.unsqueeze(1), rejected_logps_stack], dim=1
    )

    scaled = all_logps / temperature

    # NLL of chosen (index 0) being the "correct" class
    log_softmax_vals = F.log_softmax(scaled, dim=1)
    loss = -log_softmax_vals[:, 0]

    return loss.mean()


# ---------------------------------------------------------------------------
# ContrastivePrefOptimizer
# ---------------------------------------------------------------------------

class ContrastivePrefOptimizer:
    """Contrastive Preference Optimizer.

    Extends DPO with multi-negative ranking, hard negative mining, and an
    InfoNCE-style temperature-scaled contrastive loss.

    Args:
        policy:              Policy language model (trainable).
                             Forward: model(input_ids) -> logits (B, T, V).
        ref_policy:          Frozen reference language model. Same forward signature.
        temperature:         InfoNCE temperature. Default: 0.07.
        n_negatives:         Number of negatives to contrast against. Default: 4.
        beta:                KL regularisation strength. Default: 0.1.
        hard_negative_ratio: Fraction of negatives selected via hard mining.
                             Default: 0.5.
    """

    IGNORE_INDEX: int = -100

    def __init__(
        self,
        policy: nn.Module,
        ref_policy: nn.Module,
        temperature: float = 0.07,
        n_negatives: int = 4,
        beta: float = 0.1,
        hard_negative_ratio: float = 0.5,
    ) -> None:
        self.policy = policy
        self.ref_policy = ref_policy
        self.temperature = temperature
        self.n_negatives = n_negatives
        self.beta = beta
        self.hard_negative_ratio = hard_negative_ratio

        # Freeze reference policy
        for p in self.ref_policy.parameters():
            p.requires_grad_(False)
        self.ref_policy.eval()

    # ------------------------------------------------------------------
    # Log-prob computation
    # ------------------------------------------------------------------

    def compute_sequence_logps(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute sum of log-probs over non-ignored label positions.

        Uses next-token prediction: logits at position t predict the token at
        t+1. Positions where labels == IGNORE_INDEX (-100) do not contribute.

        Args:
            model:     Language model. Forward: model(input_ids) -> logits (B, T, V).
            input_ids: Shape (B, T) -- token ids.
            labels:    Shape (B, T) -- target ids; IGNORE_INDEX marks positions
                       that should not contribute to the loss.

        Returns:
            Shape (B,) -- sum of log-probs over non-ignored positions.
        """
        logits = model(input_ids)  # (B, T, V)

        # Shift: logits[t] predicts labels[t+1]
        shift_logits = logits[:, :-1, :]  # (B, T-1, V)
        shift_labels = labels[:, 1:]      # (B, T-1)

        log_probs = F.log_softmax(shift_logits, dim=-1)  # (B, T-1, V)

        # Build mask for valid (non-ignored) positions
        valid_mask = (shift_labels != self.IGNORE_INDEX).float()  # (B, T-1)
        safe_labels = shift_labels.clamp(min=0)  # prevent negative index in gather

        token_lp = log_probs.gather(
            2, safe_labels.unsqueeze(-1)
        ).squeeze(-1)  # (B, T-1)

        return (token_lp * valid_mask).sum(dim=-1)  # (B,)

    # ------------------------------------------------------------------
    # InfoNCE preference loss
    # ------------------------------------------------------------------

    def info_nce_preference_loss(
        self,
        chosen_logps: torch.Tensor,
        rejected_logps_list: List[torch.Tensor],
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, dict]:
        """InfoNCE-style contrastive preference loss.

        Computes implicit rewards (log-ratios scaled by beta) for chosen and
        each rejected response, then applies temperature-scaled InfoNCE so the
        chosen reward is maximised relative to all rejected rewards at once.

        Loss (per sample) =
            -log( exp(chosen_ratio / T) /
                  (exp(chosen_ratio / T) + sum_i exp(rejected_ratio_i / T)) )

        Args:
            chosen_logps:            Shape (B,) -- policy log-probs for chosen.
            rejected_logps_list:     List of N tensors, each (B,) -- policy
                                     log-probs for rejected responses.
            ref_chosen_logps:        Shape (B,) -- reference log-probs for chosen.
            ref_rejected_logps_list: List of N tensors, each (B,) -- reference
                                     log-probs for rejected responses.

        Returns:
            (loss, metrics) where loss is a scalar tensor and metrics contains
            'accuracy', 'mean_chosen_reward', 'mean_rejected_reward'.
        """
        T = self.temperature

        # Implicit rewards: beta * (policy_logp - ref_logp)
        chosen_ratio = self.beta * (chosen_logps - ref_chosen_logps)  # (B,)

        rejected_ratios = [
            self.beta * (rej - ref_rej)
            for rej, ref_rej in zip(rejected_logps_list, ref_rejected_logps_list)
        ]  # list of (B,)

        # Temperature scaling
        chosen_scaled = chosen_ratio / T  # (B,)
        rejected_scaled = torch.stack(
            [r / T for r in rejected_ratios], dim=1
        )  # (B, n_negatives)

        # InfoNCE via log-sum-exp:
        #   loss = logsumexp([chosen, rej_0, ..., rej_N], dim=1) - chosen
        all_scaled = torch.cat(
            [chosen_scaled.unsqueeze(1), rejected_scaled], dim=1
        )  # (B, 1 + n_negatives)

        log_denominator = torch.logsumexp(all_scaled, dim=1)  # (B,)
        loss_per_sample = log_denominator - chosen_scaled      # (B,)
        loss = loss_per_sample.mean()

        # ---- Metrics ----
        with torch.no_grad():
            rejected_stack = torch.stack(rejected_ratios, dim=1)  # (B, n_negatives)
            max_rejected = rejected_stack.max(dim=1).values        # (B,)
            accuracy = (chosen_ratio > max_rejected).float().mean().item()
            mean_chosen_reward = chosen_ratio.mean().item()
            mean_rejected_reward = rejected_stack.mean().item()

        metrics = {
            "accuracy": accuracy,
            "mean_chosen_reward": mean_chosen_reward,
            "mean_rejected_reward": mean_rejected_reward,
        }

        return loss, metrics

    # ------------------------------------------------------------------
    # Hard negative selection
    # ------------------------------------------------------------------

    def select_hard_negatives(
        self,
        rejected_logps_list: List[torch.Tensor],
        ref_rejected_logps_list: List[torch.Tensor],
        chosen_logps: torch.Tensor,
        n_hard: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Select the hardest rejected examples (closest in reward to chosen).

        Computes the implicit reward for each rejected candidate and selects the
        `n_hard` candidates whose batch-mean reward is closest to the chosen
        batch-mean reward (smallest absolute difference).

        Args:
            rejected_logps_list:      List of K tensors, each (B,) -- policy
                                      log-probs for rejected responses.
            ref_rejected_logps_list:  List of K tensors, each (B,) -- reference
                                      log-probs for rejected responses.
            chosen_logps:             Shape (B,) -- policy log-probs for chosen
                                      (used as reward anchor).
            n_hard:                   Number of hard negatives to return (<= K).

        Returns:
            (selected_rejected_logps, selected_ref_logps) -- each a list of
            n_hard tensors of shape (B,).
        """
        if n_hard >= len(rejected_logps_list):
            return list(rejected_logps_list), list(ref_rejected_logps_list)

        chosen_reward_mean = chosen_logps.detach().mean()

        distances = []
        for rej_logps, ref_rej_logps in zip(rejected_logps_list, ref_rejected_logps_list):
            rej_reward = self.beta * (rej_logps.detach() - ref_rej_logps.detach())
            dist = (chosen_reward_mean - rej_reward.mean()).abs().item()
            distances.append(dist)

        # Ascending sort: closest (smallest distance) first
        sorted_indices = sorted(range(len(distances)), key=lambda i: distances[i])
        hard_indices = sorted_indices[:n_hard]

        selected_rejected = [rejected_logps_list[i] for i in hard_indices]
        selected_ref = [ref_rejected_logps_list[i] for i in hard_indices]

        return selected_rejected, selected_ref

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(
        self,
        chosen_ids: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_ids_list: List[torch.Tensor],
        rejected_labels_list: List[torch.Tensor],
    ) -> dict:
        """Single contrastive preference optimisation step.

        Runs forward passes for policy and reference, applies hard negative
        selection, computes the InfoNCE loss, runs backward, and returns
        metrics.

        Note: Does NOT call optimizer.zero_grad() or optimizer.step(). The
        caller manages the optimizer to support gradient accumulation.

        Args:
            chosen_ids:           Shape (B, T) -- token ids for chosen sequences.
            chosen_labels:        Shape (B, T) -- label ids; IGNORE_INDEX masks
                                  prompt tokens.
            rejected_ids_list:    List of tensors, each (B, T) -- rejected token ids.
            rejected_labels_list: List of tensors, each (B, T) -- rejected label ids.

        Returns:
            Dict with keys: 'loss', 'accuracy', 'mean_chosen_reward',
            'mean_rejected_reward'.
        """
        self.policy.train()

        # Policy log-probs
        chosen_logps = self.compute_sequence_logps(
            self.policy, chosen_ids, chosen_labels
        )
        rejected_logps_computed = [
            self.compute_sequence_logps(self.policy, rej_ids, rej_labels)
            for rej_ids, rej_labels in zip(rejected_ids_list, rejected_labels_list)
        ]

        # Reference log-probs (no grad)
        with torch.no_grad():
            ref_chosen_logps = self.compute_sequence_logps(
                self.ref_policy, chosen_ids, chosen_labels
            )
            ref_rejected_logps_computed = [
                self.compute_sequence_logps(self.ref_policy, rej_ids, rej_labels)
                for rej_ids, rej_labels in zip(rejected_ids_list, rejected_labels_list)
            ]

        # Hard negative selection
        n_hard = max(1, round(self.n_negatives * self.hard_negative_ratio))
        n_hard = min(n_hard, len(rejected_logps_computed))

        if len(rejected_logps_computed) > n_hard:
            selected_rejected, selected_ref = self.select_hard_negatives(
                rejected_logps_computed,
                ref_rejected_logps_computed,
                chosen_logps,
                n_hard=n_hard,
            )
        else:
            selected_rejected = rejected_logps_computed
            selected_ref = ref_rejected_logps_computed

        # Loss
        loss, metrics = self.info_nce_preference_loss(
            chosen_logps,
            selected_rejected,
            ref_chosen_logps,
            selected_ref,
        )

        loss.backward()

        return {
            "loss": loss.item(),
            "accuracy": metrics["accuracy"],
            "mean_chosen_reward": metrics["mean_chosen_reward"],
            "mean_rejected_reward": metrics["mean_rejected_reward"],
        }
