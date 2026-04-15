"""Knowledge distillation: train a small student model to mimic a larger teacher
by matching output distributions (soft targets).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class KDConfig:
    """Configuration for knowledge distillation training."""
    temperature: float = 4.0          # softmax temperature for soft labels
    alpha: float = 0.5                # weight on KD loss: total = alpha*kd + (1-alpha)*ce
    kd_loss_type: str = "forward_kl"  # "forward_kl" | "reverse_kl" | "mse"


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def soft_cross_entropy(
    student_logits: Tensor,
    teacher_logits: Tensor,
    temperature: float,
) -> Tensor:
    """KL divergence with temperature softening: KL(teacher_probs || student_probs).

    Both distributions are computed via softmax at the given temperature.
    Returns a scalar tensor.  Scaled by T^2 so gradients are comparable
    across temperatures (standard Hinton et al. scaling).

    Args:
        student_logits: (B, T, V) or (N, V) raw logits from student
        teacher_logits: same shape, raw logits from teacher
        temperature:    softening temperature > 0

    Returns:
        scalar loss tensor
    """
    student_log_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)

    # Flatten batch/time dims so kl_div batchmean is well-defined
    V = student_logits.shape[-1]
    loss = F.kl_div(
        student_log_soft.reshape(-1, V),
        teacher_soft.reshape(-1, V),
        reduction="batchmean",
    ) * (temperature ** 2)
    return loss


def forward_kl_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    temperature: float,
) -> Tensor:
    """KL(teacher || student) — same as soft_cross_entropy (teacher → student direction).

    Args:
        student_logits: raw logits from student
        teacher_logits: raw logits from teacher
        temperature:    softening temperature > 0

    Returns:
        scalar loss tensor
    """
    return soft_cross_entropy(student_logits, teacher_logits, temperature)


def reverse_kl_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    temperature: float,
) -> Tensor:
    """KL(student || teacher) — reverse direction.

    Args:
        student_logits: raw logits from student
        teacher_logits: raw logits from teacher
        temperature:    softening temperature > 0

    Returns:
        scalar loss tensor
    """
    teacher_log_soft = F.log_softmax(teacher_logits / temperature, dim=-1)
    student_soft = F.softmax(student_logits / temperature, dim=-1)

    V = student_logits.shape[-1]
    loss = F.kl_div(
        teacher_log_soft.reshape(-1, V),
        student_soft.reshape(-1, V),
        reduction="batchmean",
    ) * (temperature ** 2)
    return loss


def mse_distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
) -> Tensor:
    """MSE between raw student and teacher logits.

    Args:
        student_logits: raw logits from student (any shape)
        teacher_logits: raw logits from teacher (same shape)

    Returns:
        scalar loss tensor
    """
    return F.mse_loss(student_logits, teacher_logits)


def combined_kd_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    labels: Tensor,
    config: KDConfig,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Combine knowledge-distillation loss with standard cross-entropy.

    total_loss = alpha * kd_loss + (1 - alpha) * ce_loss

    Args:
        student_logits: (B, V) or (B, T, V) student raw logits
        teacher_logits: same shape, teacher raw logits (detached internally)
        labels:         (B,) or (B, T) ground-truth token ids for CE loss
        config:         KDConfig instance

    Returns:
        (total_loss, kd_loss, ce_loss) — all scalar tensors
    """
    # Detach teacher so gradients only flow through student
    teacher_logits = teacher_logits.detach()

    # Cross-entropy loss
    # Support both 2-D (B, V) and 3-D (B, T, V) logits
    if student_logits.dim() == 3:
        B, T, V = student_logits.shape
        ce_loss = F.cross_entropy(
            student_logits.reshape(-1, V),
            labels.reshape(-1),
        )
    else:
        ce_loss = F.cross_entropy(student_logits, labels)

    # Knowledge-distillation loss
    if config.kd_loss_type == "forward_kl":
        kd_loss = forward_kl_loss(student_logits, teacher_logits, config.temperature)
    elif config.kd_loss_type == "reverse_kl":
        kd_loss = reverse_kl_loss(student_logits, teacher_logits, config.temperature)
    elif config.kd_loss_type == "mse":
        kd_loss = mse_distillation_loss(student_logits, teacher_logits)
    else:
        raise ValueError(
            f"Unknown kd_loss_type: {config.kd_loss_type!r}. "
            "Choose from 'forward_kl', 'reverse_kl', 'mse'."
        )

    total_loss = config.alpha * kd_loss + (1.0 - config.alpha) * ce_loss
    return total_loss, kd_loss, ce_loss


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class KnowledgeDistillationTrainer:
    """Train a student model to mimic a teacher via knowledge distillation.

    Both student_fn and teacher_fn are callables:
        (token_ids: Tensor[B, T]) -> logits: Tensor[B, T, V]
    """

    def __init__(
        self,
        student_fn: Callable[[Tensor], Tensor],
        teacher_fn: Callable[[Tensor], Tensor],
        optimizer: torch.optim.Optimizer,
        config: KDConfig,
    ) -> None:
        self.student_fn = student_fn
        self.teacher_fn = teacher_fn
        self.optimizer = optimizer
        self.config = config

    def train_step(self, token_ids: Tensor, labels: Tensor) -> Dict[str, float]:
        """Run one training step.

        1. Teacher forward (no grad).
        2. Student forward.
        3. Compute combined KD + CE loss.
        4. Backward through student only.
        5. Optimizer step.

        Args:
            token_ids: (B, T) integer token indices
            labels:    (B, T) or (B,) target token ids

        Returns:
            dict with keys: "total_loss", "kd_loss", "ce_loss"
        """
        self.optimizer.zero_grad()

        with torch.no_grad():
            teacher_logits = self.teacher_fn(token_ids)

        student_logits = self.student_fn(token_ids)

        total_loss, kd_loss, ce_loss = combined_kd_loss(
            student_logits, teacher_logits, labels, self.config
        )

        total_loss.backward()
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "kd_loss": kd_loss.item(),
            "ce_loss": ce_loss.item(),
        }

    def evaluate(self, token_ids: Tensor, labels: Tensor) -> Dict[str, float]:
        """Compute losses without gradient tracking.

        Args:
            token_ids: (B, T) integer token indices
            labels:    (B, T) or (B,) target token ids

        Returns:
            dict with keys: "total_loss", "kd_loss", "ce_loss"
        """
        with torch.no_grad():
            teacher_logits = self.teacher_fn(token_ids)
            student_logits = self.student_fn(token_ids)

            total_loss, kd_loss, ce_loss = combined_kd_loss(
                student_logits, teacher_logits, labels, self.config
            )

        return {
            "total_loss": total_loss.item(),
            "kd_loss": kd_loss.item(),
            "ce_loss": ce_loss.item(),
        }


# ---------------------------------------------------------------------------
# Layer-wise (hidden-state) distillation
# ---------------------------------------------------------------------------

def layer_wise_distillation_loss(
    student_hidden: Tensor,
    teacher_hidden: Tensor,
) -> Tensor:
    """MSE between student and teacher hidden states.

    Shapes must match.  If they don't, raises ValueError (caller is
    responsible for projecting to a common dimension first).

    Args:
        student_hidden: (B, T, D_s) hidden states from student layer
        teacher_hidden: (B, T, D_t) hidden states from teacher layer

    Returns:
        scalar MSE loss tensor
    """
    if student_hidden.shape != teacher_hidden.shape:
        raise ValueError(
            f"Shape mismatch in layer_wise_distillation_loss: "
            f"student {student_hidden.shape} vs teacher {teacher_hidden.shape}. "
            "Project to a common dimension before calling this function."
        )
    return F.mse_loss(student_hidden, teacher_hidden.detach())
