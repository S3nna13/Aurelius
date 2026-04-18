"""
src/security/membership_inference.py

Membership inference attacks: threshold-based and shadow model approaches
for determining whether a sample was part of the training set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MIAConfig:
    """Configuration for membership inference attacks."""

    loss_threshold: float = 2.0
    confidence_threshold: float = 0.5
    method: str = "threshold"  # "threshold" or "entropy"


def compute_sample_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Compute cross-entropy loss on a single sample.

    Args:
        model: Language model returning (loss, logits, ...) when called with
            (input_ids, labels=labels).
        input_ids: (1, S) or (B, S) token id tensor.
        labels: (1, S) or (B, S) target token id tensor.

    Returns:
        Scalar loss as a Python float.
    """
    with torch.no_grad():
        out = model(input_ids, labels=labels)
        loss: torch.Tensor = out[0]
    return float(loss.item())


def compute_sample_entropy(
    model: nn.Module,
    input_ids: torch.Tensor,
) -> float:
    """Compute entropy of the output distribution at the last token position.

    Args:
        model: Language model returning (loss_or_None, logits, ...) when called
            with (input_ids).
        input_ids: (1, S) or (B, S) token id tensor.

    Returns:
        Entropy value as a Python float (>= 0).
    """
    with torch.no_grad():
        out = model(input_ids)
        # out may be (None, logits, ...) or just logits depending on model
        if isinstance(out, (tuple, list)):
            logits: torch.Tensor = out[1]
        else:
            logits = out
    # Use last token position: (B, vocab_size)
    last_logits = logits[:, -1, :]
    probs = F.softmax(last_logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
    return float(entropy.mean().item())


class ThresholdMIA:
    """Threshold-based membership inference attack using loss values."""

    def __init__(self, config: MIAConfig) -> None:
        self.config = config

    def predict_member(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> bool:
        """Predict whether a sample is a training member.

        A sample with low loss is likely a member (the model has memorized it).

        Args:
            model: Language model.
            input_ids: (1, S) token id tensor.
            labels: (1, S) target token id tensor.

        Returns:
            True if the sample is predicted to be a member.
        """
        loss = compute_sample_loss(model, input_ids, labels)
        return loss < self.config.loss_threshold

    def predict_batch(
        self,
        model: nn.Module,
        input_ids_list: List[torch.Tensor],
        labels_list: List[torch.Tensor],
    ) -> List[bool]:
        """Predict membership for a list of samples.

        Args:
            model: Language model.
            input_ids_list: List of (1, S) token id tensors.
            labels_list: List of (1, S) target token id tensors.

        Returns:
            List of bool predictions, one per sample.
        """
        return [
            self.predict_member(model, ids, lbls)
            for ids, lbls in zip(input_ids_list, labels_list)
        ]

    def attack_accuracy(
        self,
        model: nn.Module,
        member_samples: List[Tuple[torch.Tensor, torch.Tensor]],
        nonmember_samples: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> float:
        """Compute attack accuracy assuming members should be predicted True.

        Args:
            model: Language model.
            member_samples: List of (input_ids, labels) tuples for training members.
            nonmember_samples: List of (input_ids, labels) tuples for non-members.

        Returns:
            Accuracy in [0, 1].
        """
        total = len(member_samples) + len(nonmember_samples)
        if total == 0:
            return 0.0

        correct = 0
        for ids, lbls in member_samples:
            if self.predict_member(model, ids, lbls):
                correct += 1
        for ids, lbls in nonmember_samples:
            if not self.predict_member(model, ids, lbls):
                correct += 1

        return correct / total


class EntropyMIA:
    """Entropy-based membership inference attack.

    Members tend to have lower entropy (more confident predictions).
    """

    def __init__(self, config: MIAConfig) -> None:
        self.config = config

    def predict_member(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
    ) -> bool:
        """Predict whether a sample is a training member based on output entropy.

        Args:
            model: Language model.
            input_ids: (1, S) token id tensor.

        Returns:
            True if the sample is predicted to be a member.
        """
        entropy = compute_sample_entropy(model, input_ids)
        return entropy < self.config.confidence_threshold
