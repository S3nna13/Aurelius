"""Knowledge Distillation for LLMs.

Implements forward KL, reverse KL, Jensen-Shannon Divergence (JSD), and MSE
distillation losses with a combined task + distillation loss module and a
lightweight training wrapper.

All computation uses pure native PyTorch -- no external libraries beyond torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation.

    Attributes:
        temperature: Softmax temperature used to soften logit distributions.
            Higher values produce softer (more uniform) probability distributions.
        alpha: Weight applied to the task (cross-entropy) loss.
            The distillation loss receives weight ``1 - alpha``.
        loss_type: Which distillation objective to use.
            One of 'forward_kl', 'reverse_kl', 'jsd', 'mse'.
    """

    temperature: float = 4.0
    alpha: float = 0.5
    loss_type: str = "forward_kl"


# ---------------------------------------------------------------------------
# Soft-target losses
# ---------------------------------------------------------------------------

class SoftTargetLoss(nn.Module):
    """Distillation losses computed on temperature-softened distributions.

    Args:
        temperature: Positive scalar controlling distribution sharpness.
    """

    def __init__(self, temperature: float) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = temperature

    def soft_probs(self, logits: Tensor) -> Tensor:
        """Convert logits to softmax probabilities at the configured temperature.

        Args:
            logits: Float tensor of shape (B, T, V).

        Returns:
            Probability tensor of shape (B, T, V) where each (B, T) slice sums to 1.
        """
        return F.softmax(logits / self.temperature, dim=-1)

    def forward_kl(self, student_logits: Tensor, teacher_logits: Tensor) -> Tensor:
        """KL(teacher || student) -- the standard knowledge distillation loss.

        Args:
            student_logits: Shape (B, T, V).
            teacher_logits: Shape (B, T, V).

        Returns:
            Scalar loss tensor.
        """
        teacher_probs = self.soft_probs(teacher_logits)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

    def reverse_kl(self, student_logits: Tensor, teacher_logits: Tensor) -> Tensor:
        """KL(student || teacher).

        Args:
            student_logits: Shape (B, T, V).
            teacher_logits: Shape (B, T, V).

        Returns:
            Scalar loss tensor.
        """
        student_probs = self.soft_probs(student_logits)
        teacher_log_probs = F.log_softmax(teacher_logits / self.temperature, dim=-1)
        return F.kl_div(teacher_log_probs, student_probs, reduction="batchmean")

    def jsd(self, student_logits: Tensor, teacher_logits: Tensor) -> Tensor:
        """Jensen-Shannon Divergence: 0.5*(KL(t||m) + KL(s||m)) where m = 0.5*(t+s).

        Args:
            student_logits: Shape (B, T, V).
            teacher_logits: Shape (B, T, V).

        Returns:
            Scalar loss tensor.
        """
        teacher_probs = self.soft_probs(teacher_logits)
        student_probs = self.soft_probs(student_logits)
        mixture = 0.5 * (teacher_probs + student_probs)
        log_mixture = torch.log(mixture.clamp(min=1e-10))

        kl_t = F.kl_div(log_mixture, teacher_probs, reduction="batchmean")
        kl_s = F.kl_div(log_mixture, student_probs, reduction="batchmean")
        return 0.5 * (kl_t + kl_s)

    def mse_loss(self, student_logits: Tensor, teacher_logits: Tensor) -> Tensor:
        """Mean squared error between soft probability distributions.

        Args:
            student_logits: Shape (B, T, V).
            teacher_logits: Shape (B, T, V).

        Returns:
            Scalar loss tensor (non-negative).
        """
        teacher_probs = self.soft_probs(teacher_logits)
        student_probs = self.soft_probs(student_logits)
        return F.mse_loss(student_probs, teacher_probs)

    def forward(self, student_logits: Tensor, teacher_logits: Tensor) -> Tensor:
        """Alias for forward_kl."""
        return self.forward_kl(student_logits, teacher_logits)


# ---------------------------------------------------------------------------
# Combined task + distillation loss
# ---------------------------------------------------------------------------

class DistillationLoss(nn.Module):
    """Combined task loss + distillation loss.

    total = alpha * task_loss + (1 - alpha) * distill_loss

    Args:
        config: DistillationConfig instance controlling temperature,
            alpha weighting, and which distillation objective to use.
    """

    def __init__(self, config: DistillationConfig) -> None:
        super().__init__()
        self.config = config
        self._soft = SoftTargetLoss(temperature=config.temperature)

    def task_loss(self, student_logits: Tensor, labels: torch.LongTensor) -> Tensor:
        """Standard causal language-model NLL loss.

        Positions with label -100 are masked out (ignored).

        Args:
            student_logits: Shape (B, T, V).
            labels: Shape (B, T) with token ids; use -100 to ignore.

        Returns:
            Scalar mean cross-entropy over non-masked positions.
        """
        B, T, V = student_logits.shape
        logits_2d = student_logits.reshape(B * T, V)
        labels_1d = labels.reshape(B * T)
        return F.cross_entropy(logits_2d, labels_1d, ignore_index=-100)

    def distill_loss(self, student_logits: Tensor, teacher_logits: Tensor) -> Tensor:
        """Compute the configured distillation objective.

        Args:
            student_logits: Shape (B, T, V).
            teacher_logits: Shape (B, T, V).

        Returns:
            Scalar distillation loss.

        Raises:
            ValueError: If config.loss_type is not recognised.
        """
        loss_type = self.config.loss_type
        if loss_type == "forward_kl":
            return self._soft.forward_kl(student_logits, teacher_logits)
        elif loss_type == "reverse_kl":
            return self._soft.reverse_kl(student_logits, teacher_logits)
        elif loss_type == "jsd":
            return self._soft.jsd(student_logits, teacher_logits)
        elif loss_type == "mse":
            return self._soft.mse_loss(student_logits, teacher_logits)
        else:
            raise ValueError(
                f"Unknown loss_type '{loss_type}'. "
                "Expected one of: 'forward_kl', 'reverse_kl', 'jsd', 'mse'."
            )

    def forward(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        labels: torch.LongTensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute total weighted loss and return a metrics dictionary.

        Args:
            student_logits: Shape (B, T, V).
            teacher_logits: Shape (B, T, V).
            labels: Shape (B, T); positions set to -100 are ignored in task loss.

        Returns:
            A tuple (total_loss, metrics) where metrics contains keys
            'task_loss', 'distill_loss', and 'total_loss'.
        """
        alpha = self.config.alpha
        t_loss = self.task_loss(student_logits, labels)
        d_loss = self.distill_loss(student_logits, teacher_logits)
        total = alpha * t_loss + (1.0 - alpha) * d_loss
        metrics: Dict[str, Tensor] = {
            "task_loss": t_loss,
            "distill_loss": d_loss,
            "total_loss": total,
        }
        return total, metrics


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DistillationTrainer:
    """Minimal training wrapper for one student-teacher distillation step.

    The teacher is automatically frozen on construction via freeze_teacher().

    Args:
        student_model: Trainable model; forward(input_ids) must return either
            logits directly or a tuple whose first element is logits.
        teacher_model: Reference model kept frozen; same forward contract.
        optimizer: PyTorch optimiser attached to student parameters.
        loss_fn: DistillationLoss instance.
    """

    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: DistillationLoss,
    ) -> None:
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.freeze_teacher()

    def freeze_teacher(self) -> None:
        """Disable gradients for all teacher parameters."""
        for param in self.teacher_model.parameters():
            param.requires_grad_(False)

    @staticmethod
    def _extract_logits(output) -> Tensor:
        """Return logits from a model output that may be a tensor or a tuple."""
        if isinstance(output, Tensor):
            return output
        return output[0]

    def train_step(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
    ) -> Dict[str, float]:
        """Run one forward-backward-update step.

        Args:
            input_ids: Shape (B, T) token ids fed to both models.
            labels: Shape (B, T) targets; use -100 for masked positions.

        Returns:
            Dictionary with scalar float values for 'task_loss',
            'distill_loss', and 'total_loss'.
        """
        self.student_model.train()
        self.teacher_model.eval()

        with torch.no_grad():
            teacher_output = self.teacher_model(input_ids)
            teacher_logits = self._extract_logits(teacher_output)

        student_output = self.student_model(input_ids)
        student_logits = self._extract_logits(student_output)

        total_loss, metrics = self.loss_fn(student_logits, teacher_logits, labels)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {k: v.item() for k, v in metrics.items()}
