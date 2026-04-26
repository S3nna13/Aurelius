"""Weak-to-Strong Generalization.

Burns et al. 2023 -- https://arxiv.org/abs/2312.09390

A weak model's predictions are used as training signal for a stronger model.
The strong model can generalize beyond its supervisor's capability because it
leverages its own internal representations more effectively.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# WeakSupervisor
# ---------------------------------------------------------------------------


class WeakSupervisor:
    """Wraps a small frozen model to produce soft pseudo-labels.

    Args:
        model: A callable nn.Module that accepts (B, ...) input and
            returns a logits tensor of shape (B, num_classes).
        num_classes: Number of output classes.
    """

    def __init__(self, model: nn.Module, num_classes: int) -> None:
        self.model = model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._num_classes = num_classes

    @property
    def num_classes(self) -> int:
        """Number of output classes."""
        return self._num_classes

    def get_soft_labels(self, x: Tensor) -> Tensor:
        """Return soft label probabilities for a batch of inputs.

        Args:
            x: Input tensor of shape (B, ...).

        Returns:
            Soft label tensor of shape (B, num_classes) whose rows sum to 1.
        """
        with torch.no_grad():
            logits = self.model(x)  # (B, num_classes)
        return F.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# WeakToStrongDataset
# ---------------------------------------------------------------------------


class WeakToStrongDataset(Dataset):
    """Dataset that pre-computes soft labels from a WeakSupervisor.

    Args:
        inputs: Raw input tensor of shape (N, ...).
        supervisor: A WeakSupervisor used to label the inputs.
    """

    def __init__(self, inputs: Tensor, supervisor: WeakSupervisor) -> None:
        self.inputs = inputs
        self.soft_labels = supervisor.get_soft_labels(inputs)  # (N, num_classes)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.soft_labels[idx]


# ---------------------------------------------------------------------------
# WeakToStrongLoss
# ---------------------------------------------------------------------------


class WeakToStrongLoss(nn.Module):
    """KL-divergence loss between strong model outputs and weak soft labels.

    Args:
        confidence_weighting: If True, weight each sample's loss by the
            maximum probability in the weak model's soft label distribution.
        confidence_threshold: Samples whose peak soft-label probability falls
            below this threshold are zeroed out when confidence_weighting
            is enabled.
    """

    def __init__(
        self,
        confidence_weighting: bool = False,
        confidence_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.confidence_weighting = confidence_weighting
        self.confidence_threshold = confidence_threshold

    def forward(self, student_logits: Tensor, soft_labels: Tensor) -> Tensor:
        """Compute the (optionally weighted) KL-divergence loss.

        Args:
            student_logits: Strong model's raw logits, shape (B, num_classes).
            soft_labels: Weak model's softmax probabilities, shape (B, num_classes).

        Returns:
            Scalar mean loss.
        """
        # Per-sample KL divergence: KL(soft_labels || student_probs)
        # F.kl_div expects log-probabilities as input and probabilities as target.
        per_sample = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            soft_labels,
            reduction="none",
        ).sum(dim=-1)  # (B,)

        if self.confidence_weighting:
            max_probs = soft_labels.max(dim=-1).values  # (B,)
            weights = max_probs * (max_probs >= self.confidence_threshold).float()
            total_weight = weights.sum()
            if total_weight == 0:
                return per_sample.mean() * 0.0
            return (per_sample * weights).sum() / total_weight

        return per_sample.mean()


# ---------------------------------------------------------------------------
# WeakToStrongTrainer
# ---------------------------------------------------------------------------


class WeakToStrongTrainer:
    """Orchestrates one training step of the strong model on weak pseudo-labels.

    Args:
        strong_model: The student model to be trained.
        optimizer: PyTorch optimiser bound to strong_model's parameters.
        loss_fn: A WeakToStrongLoss instance.
    """

    def __init__(
        self,
        strong_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: WeakToStrongLoss,
    ) -> None:
        self.strong_model = strong_model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_step(self, x: Tensor, soft_labels: Tensor) -> dict[str, float]:
        """Execute one gradient update step.

        Args:
            x: Input batch, shape (B, ...).
            soft_labels: Weak model's soft labels, shape (B, num_classes).

        Returns:
            Dictionary with keys:
              'loss': scalar training loss.
              'mean_confidence': mean of soft_labels.max(dim=-1).values.
              'frac_above_threshold': fraction of samples whose peak
                soft-label probability exceeds loss_fn.confidence_threshold.
        """
        self.strong_model.train()
        self.optimizer.zero_grad()

        student_logits = self.strong_model(x)  # (B, num_classes)
        loss = self.loss_fn(student_logits, soft_labels)
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            max_probs = soft_labels.max(dim=-1).values  # (B,)
            mean_confidence = max_probs.mean().item()
            frac_above = (max_probs > self.loss_fn.confidence_threshold).float().mean().item()

        return {
            "loss": loss.item(),
            "mean_confidence": mean_confidence,
            "frac_above_threshold": frac_above,
        }
