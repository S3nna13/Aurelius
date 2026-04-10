"""Knowledge distillation: train student to match teacher's soft predictions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class DistillationConfig:
    temperature: float = 4.0      # softmax temperature for soft labels
    alpha: float = 0.5            # weight: alpha*KD_loss + (1-alpha)*CE_loss
    kd_loss_type: str = "kl"      # "kl" | "mse" | "ce"


def soft_labels(logits: Tensor, temperature: float) -> Tensor:
    """Apply temperature scaling and softmax.
    logits: (B, T, V) -> soft probs: (B, T, V)"""
    return F.softmax(logits / temperature, dim=-1)


def kl_distillation_loss(
    student_logits: Tensor,    # (B, T, V)
    teacher_logits: Tensor,    # (B, T, V)
    temperature: float,
) -> Tensor:
    """KL divergence distillation loss.
    KL(teacher_soft || student_soft) * temperature^2
    Returns scalar."""
    student_log_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)

    B, T, V = student_logits.shape
    loss = F.kl_div(
        student_log_soft.view(-1, V),
        teacher_soft.view(-1, V),
        reduction="batchmean",
    ) * (temperature ** 2)
    return loss


def mse_distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
) -> Tensor:
    """MSE between student and teacher logits. Returns scalar."""
    return F.mse_loss(student_logits, teacher_logits)


def distillation_loss(
    student_logits: Tensor,    # (B, T, V)
    teacher_logits: Tensor,    # (B, T, V)
    labels: Tensor,            # (B, T) - ground truth token ids
    config: DistillationConfig,
) -> tuple[Tensor, dict]:
    """Combined distillation + cross-entropy loss.

    loss = alpha * kd_loss + (1-alpha) * ce_loss

    Returns (total_loss, metrics) where metrics has:
        'kd_loss': float
        'ce_loss': float
        'total_loss': float
    """
    B, T, V = student_logits.shape

    # Cross-entropy loss (standard language model loss, shifted)
    ce_loss = F.cross_entropy(
        student_logits[:, :-1].contiguous().view(-1, V),
        labels[:, 1:].contiguous().view(-1),
    )

    # Knowledge distillation loss
    if config.kd_loss_type == "kl":
        kd_loss = kl_distillation_loss(
            student_logits[:, :-1],
            teacher_logits[:, :-1].detach(),
            config.temperature,
        )
    elif config.kd_loss_type == "mse":
        kd_loss = mse_distillation_loss(
            student_logits[:, :-1],
            teacher_logits[:, :-1].detach(),
        )
    elif config.kd_loss_type == "ce":
        # Cross-entropy against teacher's soft labels
        teacher_soft = soft_labels(teacher_logits[:, :-1].detach(), config.temperature)
        student_log_soft = F.log_softmax(student_logits[:, :-1] / config.temperature, dim=-1)
        kd_loss = -(teacher_soft * student_log_soft).sum(dim=-1).mean() * (config.temperature ** 2)
    else:
        raise ValueError(f"Unknown kd_loss_type: {config.kd_loss_type!r}")

    total_loss = config.alpha * kd_loss + (1 - config.alpha) * ce_loss

    metrics = {
        "kd_loss": kd_loss.item(),
        "ce_loss": ce_loss.item(),
        "total_loss": total_loss.item(),
    }

    return total_loss, metrics


class FeatureDistillationLoss(nn.Module):
    """Intermediate layer feature matching (hidden state distillation)."""

    def __init__(self, student_dim: int, teacher_dim: int) -> None:
        super().__init__()
        # Optional projection if dims differ
        if student_dim != teacher_dim:
            self.proj = nn.Linear(student_dim, teacher_dim, bias=False)
        else:
            self.proj = None

    def forward(self, student_hidden: Tensor, teacher_hidden: Tensor) -> Tensor:
        """MSE loss between student and teacher hidden states.
        Both shape (B, T, d). Returns scalar."""
        if self.proj is not None:
            student_hidden = self.proj(student_hidden)
        return F.mse_loss(student_hidden, teacher_hidden.detach())


class DistillationTrainer:
    """Train student model to mimic teacher."""

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        config: DistillationConfig,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.student = student
        self.teacher = teacher
        self.config = config
        self.optimizer = optimizer

        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

    def train_step(self, input_ids: Tensor) -> dict:
        """
        1. Teacher forward (no grad)
        2. Student forward
        3. Compute distillation_loss
        4. Backward + step
        Returns dict with: kd_loss, ce_loss, total_loss
        """
        self.student.train()
        self.optimizer.zero_grad()

        # Teacher forward (no grad)
        with torch.no_grad():
            _, teacher_logits, _ = self.teacher(input_ids)

        # Student forward
        _, student_logits, _ = self.student(input_ids)

        # Labels are the input_ids themselves (causal LM)
        labels = input_ids

        total_loss, metrics = distillation_loss(
            student_logits, teacher_logits, labels, self.config
        )

        total_loss.backward()
        self.optimizer.step()

        return metrics


class LayerWiseDistillationTrainer(DistillationTrainer):
    """Distillation with intermediate layer matching."""

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        config: DistillationConfig,
        optimizer: torch.optim.Optimizer,
        feature_loss_weight: float = 0.1,
    ) -> None:
        super().__init__(student, teacher, config, optimizer)
        self.feature_loss_weight = feature_loss_weight

        # Determine hidden dimensions from model config if available
        student_dim = self._get_model_dim(student)
        teacher_dim = self._get_model_dim(teacher)

        self.feature_loss_fn = FeatureDistillationLoss(student_dim, teacher_dim)

        # Storage for captured hidden states
        self._student_hiddens: List[Tensor] = []
        self._teacher_hiddens: List[Tensor] = []

        # Register forward hooks on the final norm layers
        self._register_hooks()

    def _get_model_dim(self, model: nn.Module) -> int:
        """Extract d_model from the model's config if possible."""
        if hasattr(model, "config") and hasattr(model.config, "d_model"):
            return model.config.d_model
        # Fallback: infer from norm layer weight shape
        for name, module in model.named_modules():
            if hasattr(module, "weight") and module.weight is not None and module.weight.ndim == 1:
                return module.weight.shape[0]
        return 64  # default fallback

    def _register_hooks(self) -> None:
        """Register hooks on the final norm layers to capture hidden states."""
        def _student_hook(module, inp, output):
            hidden = output if isinstance(output, Tensor) else output[0]
            self._student_hiddens.append(hidden)

        def _teacher_hook(module, inp, output):
            hidden = output if isinstance(output, Tensor) else output[0]
            self._teacher_hiddens.append(hidden)

        student_norm = self._find_final_norm(self.student)
        teacher_norm = self._find_final_norm(self.teacher)

        if student_norm is not None:
            student_norm.register_forward_hook(_student_hook)
        if teacher_norm is not None:
            teacher_norm.register_forward_hook(_teacher_hook)

    def _find_final_norm(self, model: nn.Module):
        """Find the final normalization layer in the model."""
        # For AureliusTransformer, it's model.norm
        if hasattr(model, "norm"):
            return model.norm
        # Fallback: find last norm-like module
        last_norm = None
        for name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm):
                last_norm = module
        return last_norm

    def train_step(self, input_ids: Tensor) -> dict:
        """Include intermediate hidden state matching in addition to logit distillation.
        Returns same keys as DistillationTrainer plus 'feature_loss'."""
        self.student.train()
        self.optimizer.zero_grad()

        # Clear captured hiddens
        self._student_hiddens.clear()
        self._teacher_hiddens.clear()

        # Teacher forward (no grad)
        with torch.no_grad():
            _, teacher_logits, _ = self.teacher(input_ids)

        # Student forward
        _, student_logits, _ = self.student(input_ids)

        # Labels are the input_ids themselves (causal LM)
        labels = input_ids

        total_loss, metrics = distillation_loss(
            student_logits, teacher_logits, labels, self.config
        )

        # Feature (hidden state) distillation loss
        feature_loss = torch.tensor(0.0, device=input_ids.device)
        if self._student_hiddens and self._teacher_hiddens:
            s_hidden = self._student_hiddens[-1]
            t_hidden = self._teacher_hiddens[-1]
            feature_loss = self.feature_loss_fn(s_hidden, t_hidden)

        total_with_feature = total_loss + self.feature_loss_weight * feature_loss
        total_with_feature.backward()
        self.optimizer.step()

        metrics["feature_loss"] = feature_loss.item()
        metrics["total_loss"] = total_with_feature.item()

        return metrics
