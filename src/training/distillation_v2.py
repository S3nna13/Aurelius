"""Knowledge Distillation v2 for LLMs.

Trains a small student model to mimic a large teacher using:
- Soft targets (KL / MSE / cosine KD loss) with temperature scaling
- Hard CE loss against ground-truth labels
- Optional intermediate feature matching via projection layers
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DistillConfig:
    """Configuration for knowledge distillation."""
    temperature: float = 4.0          # softening temperature T
    alpha: float = 0.5                # weight for KD loss (1-alpha for CE)
    feature_loss_weight: float = 0.0  # 0 = feature matching disabled
    kd_loss_type: str = "kl"          # one of "kl", "mse", "cosine"

    def __post_init__(self) -> None:
        if self.kd_loss_type not in ("kl", "mse", "cosine"):
            raise ValueError(
                f"kd_loss_type must be 'kl', 'mse', or 'cosine', got '{self.kd_loss_type}'"
            )
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def kl_divergence_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """KL(teacher_soft || student_soft) with temperature scaling.

    Args:
        student_logits: (B, T, V) or (B, V) logits from the student.
        teacher_logits: (B, T, V) or (B, V) logits from the teacher.
        temperature:    Softening temperature.  Higher => softer distribution.

    Returns:
        Scalar tensor: T^2 * KL(teacher_soft || student_soft).
    """
    # Flatten to (N, V) for generality
    if student_logits.dim() == 3:
        B, T, V = student_logits.shape
        s = student_logits.reshape(-1, V)
        t = teacher_logits.reshape(-1, V)
    else:
        s = student_logits
        t = teacher_logits

    student_log_soft = F.log_softmax(s / temperature, dim=-1)
    teacher_soft = F.softmax(t / temperature, dim=-1)

    # F.kl_div expects (log_input, target); KL(teacher || student)
    kl = F.kl_div(student_log_soft, teacher_soft, reduction="batchmean")
    # Clamp to zero: tiny floating-point negatives can occur due to log-softmax precision
    return (temperature ** 2) * kl.clamp(min=0.0)


def mse_logit_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> torch.Tensor:
    """MSE between student and teacher logit distributions.

    Args:
        student_logits: (B, T, V) or (B, V).
        teacher_logits: (B, T, V) or (B, V).

    Returns:
        Scalar tensor: mean squared error.
    """
    return F.mse_loss(student_logits, teacher_logits)


def cosine_embedding_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
) -> torch.Tensor:
    """1 - cosine_similarity averaged over batch / tokens.

    Args:
        student_features: (B, T, D) or (B, D).
        teacher_features: (B, T, D) or (B, D).

    Returns:
        Scalar tensor in [0, 2].
    """
    if student_features.dim() == 3:
        B, T, D = student_features.shape
        s = student_features.reshape(-1, D)
        t = teacher_features.reshape(-1, D)
    else:
        s = student_features
        t = teacher_features

    cos_sim = F.cosine_similarity(s, t, dim=-1)  # (N,)
    return (1.0 - cos_sim).mean()


# ---------------------------------------------------------------------------
# Combined distillation loss
# ---------------------------------------------------------------------------

def compute_combined_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    config: DistillConfig,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute alpha * KD_loss + (1-alpha) * CE_loss.

    Args:
        student_logits: (B, T, V) or (B, V).
        teacher_logits: (B, T, V) or (B, V) (teacher; no_grad expected outside).
        labels:         (B, T) integer labels; use -100 for positions to ignore.
        config:         DistillConfig instance.

    Returns:
        (total_loss, metrics_dict) where metrics_dict contains:
            "kd_loss", "ce_loss", "total".
    """
    # --- KD loss ---
    if config.kd_loss_type == "kl":
        kd_loss = kl_divergence_loss(student_logits, teacher_logits, config.temperature)
    elif config.kd_loss_type == "mse":
        kd_loss = mse_logit_loss(student_logits, teacher_logits)
    else:  # cosine
        kd_loss = cosine_embedding_loss(student_logits, teacher_logits)

    # --- CE loss ---
    # Support (B, T, V) and (B, V)
    if student_logits.dim() == 3:
        B, T, V = student_logits.shape
        ce_loss = F.cross_entropy(
            student_logits.reshape(-1, V),
            labels.reshape(-1),
            ignore_index=-100,
        )
    else:
        ce_loss = F.cross_entropy(student_logits, labels, ignore_index=-100)
    # When all labels are masked (ignore_index=-100), CE returns nan.
    # Replace nan with 0 so the overall loss is well-defined.
    if ce_loss.isnan():
        ce_loss = torch.zeros_like(ce_loss)

    total_loss = config.alpha * kd_loss + (1.0 - config.alpha) * ce_loss

    metrics: Dict[str, torch.Tensor] = {
        "kd_loss": kd_loss,
        "ce_loss": ce_loss,
        "total": total_loss,
    }
    return total_loss, metrics


# ---------------------------------------------------------------------------
# Feature projector
# ---------------------------------------------------------------------------

class FeatureProjector(nn.Module):
    """Linear projection from student_dim to teacher_dim (no bias).

    Used to align intermediate feature dimensions before computing
    feature-level distillation loss.
    """

    def __init__(self, student_dim: int, teacher_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(student_dim, teacher_dim, bias=False)

    def forward(self, student_features: torch.Tensor) -> torch.Tensor:
        """Project student features to teacher's feature dimension.

        Args:
            student_features: (..., student_dim) tensor.

        Returns:
            (..., teacher_dim) tensor.
        """
        return self.proj(student_features)


# ---------------------------------------------------------------------------
# Distillation trainer
# ---------------------------------------------------------------------------

class DistillationTrainer:
    """Wraps student + teacher to compute distillation loss.

    Args:
        student:    Student nn.Module to be trained.
        teacher:    Teacher nn.Module (will be frozen automatically).
        config:     DistillConfig.
        projector:  Optional FeatureProjector for intermediate feature matching.
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        config: DistillConfig,
        projector: Optional[FeatureProjector] = None,
    ) -> None:
        self.student = student
        self.teacher = teacher
        self.config = config
        self.projector = projector
        self.freeze_teacher()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run both models and compute combined distillation loss.

        Teacher is always run under torch.no_grad().  Student is run
        normally so gradients can flow.

        Args:
            input_ids: (B, T) integer token ids.
            labels:    (B, T) integer labels with -100 for ignored positions.

        Returns:
            (loss, metrics_dict) — loss is a differentiable scalar.
        """
        with torch.no_grad():
            teacher_out = self.teacher(input_ids)

        student_out = self.student(input_ids)

        # Support models that return a tuple (loss, logits, ...) or just logits
        teacher_logits = _extract_logits(teacher_out)
        student_logits = _extract_logits(student_out)

        loss, metrics = compute_combined_distillation_loss(
            student_logits, teacher_logits, labels, self.config
        )
        return loss, metrics

    def freeze_teacher(self) -> None:
        """Set all teacher parameters to requires_grad=False."""
        for param in self.teacher.parameters():
            param.requires_grad = False

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Return list of trainable student parameters only."""
        params: List[nn.Parameter] = [
            p for p in self.student.parameters() if p.requires_grad
        ]
        if self.projector is not None:
            params += [p for p in self.projector.parameters() if p.requires_grad]
        return params


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_logits(model_output) -> torch.Tensor:
    """Extract logit tensor from various model output formats."""
    if isinstance(model_output, torch.Tensor):
        return model_output
    if isinstance(model_output, (tuple, list)):
        # Common convention: (loss, logits, ...) or (logits, ...)
        for item in model_output:
            if isinstance(item, torch.Tensor) and item.dim() >= 2:
                return item
    raise ValueError(
        f"Cannot extract logits from model output of type {type(model_output)}"
    )
