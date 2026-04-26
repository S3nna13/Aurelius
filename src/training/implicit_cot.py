"""Implicit Chain-of-Thought Training (Deng et al., 2023).

Trains models to perform multi-hop reasoning without producing explicit
reasoning tokens in the output. A teacher model's hidden states at reasoning
steps are distilled into the student model's hidden states via a combined
task loss + hidden-state MSE distillation loss.

Reference:
    Deng et al. (2023) "Implicit Chain of Thought Reasoning via Knowledge Distillation"
    https://arxiv.org/abs/2311.01460
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ImplicitCoTConfig:
    """Configuration for Implicit Chain-of-Thought training."""

    n_reasoning_steps: int = 4
    """Number of implicit reasoning steps the model should internalise."""

    hidden_dim: int = 512
    """Hidden dimensionality used by the projector (student side)."""

    distill_weight: float = 0.5
    """Weight for the distillation loss component.
    total = (1 - distill_weight) * task_loss + distill_weight * distill_loss
    """

    temperature: float = 2.0
    """Softmax temperature for distillation (reserved for logit-based variants)."""


# ---------------------------------------------------------------------------
# Reasoning State Projector
# ---------------------------------------------------------------------------


class ReasoningStateProjector(nn.Module):
    """Projects student hidden states into the teacher hidden-state space.

    Used when student_dim != teacher_dim so that the MSE distillation loss
    can be computed in a common space.

    Args:
        student_dim: Dimensionality of the student hidden states.
        teacher_dim: Dimensionality of the teacher hidden states.
    """

    def __init__(self, student_dim: int, teacher_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(student_dim, teacher_dim, bias=False)

    def forward(self, student_hidden: Tensor) -> Tensor:
        """Project student hidden states.

        Args:
            student_hidden: ``(B, T, student_dim)``

        Returns:
            Tensor of shape ``(B, T, teacher_dim)``
        """
        return self.proj(student_hidden)


# ---------------------------------------------------------------------------
# Implicit CoT Loss
# ---------------------------------------------------------------------------


class ImplicitCoTLoss(nn.Module):
    """Combined task loss + hidden-state distillation loss.

    Args:
        config: :class:`ImplicitCoTConfig` instance controlling weights and
                temperature.
    """

    def __init__(self, config: ImplicitCoTConfig) -> None:
        super().__init__()
        self.config = config

    # ------------------------------------------------------------------
    # Individual loss components
    # ------------------------------------------------------------------

    def task_loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Standard causal language-modelling cross-entropy loss.

        Shifts logits and labels by one position (next-token prediction).
        Positions with label ``-100`` are ignored.

        Args:
            logits: ``(B, T, V)`` — raw model output.
            labels: ``(B, T)`` — ground-truth token ids; ``-100`` for masked.

        Returns:
            Scalar loss tensor.
        """
        # Shift: predict token t+1 from token t
        shift_logits = logits[:, :-1, :].contiguous()  # (B, T-1, V)
        shift_labels = labels[:, 1:].contiguous()  # (B, T-1)

        B, T, V = shift_logits.shape
        flat_labels = shift_labels.view(B * T)

        # Guard against the all-masked edge case where no active tokens remain;
        # PyTorch cross_entropy returns NaN in that case, so we short-circuit.
        active = (flat_labels != -100).any()
        if not active:
            return shift_logits.new_zeros(())

        loss = F.cross_entropy(
            shift_logits.view(B * T, V),
            flat_labels,
            ignore_index=-100,
        )
        return loss

    def distillation_loss(self, student_hidden: Tensor, teacher_hidden: Tensor) -> Tensor:
        """MSE distillation loss between student and teacher hidden states.

        If the hidden dimensions differ, the student states are linearly
        interpolated along the feature axis to match the teacher dimension
        before computing the MSE.

        Args:
            student_hidden: ``(B, T, d_s)``
            teacher_hidden: ``(B, T, d_t)``

        Returns:
            Scalar MSE loss tensor.
        """
        if student_hidden.shape != teacher_hidden.shape:
            # Align student feature dimension to teacher via interpolation.
            # Treat (B*T) as the "batch" and d as the "length" for
            # 1-D interpolation: shape needed is (N, C, L) → (B*T, 1, d_s)
            B, T, d_s = student_hidden.shape
            d_t = teacher_hidden.shape[-1]
            s = student_hidden.reshape(B * T, 1, d_s).float()
            s = F.interpolate(s, size=d_t, mode="linear", align_corners=False)
            student_hidden = s.reshape(B, T, d_t).to(teacher_hidden.dtype)

        return F.mse_loss(student_hidden, teacher_hidden.detach())

    # ------------------------------------------------------------------
    # Combined forward
    # ------------------------------------------------------------------

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        student_hidden: Tensor,
        teacher_hidden: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute combined implicit CoT loss.

        Args:
            logits: ``(B, T, V)`` — student model logits.
            labels: ``(B, T)`` — target token ids.
            student_hidden: ``(B, T, d_s)`` — student last hidden states.
            teacher_hidden: ``(B, T, d_t)`` — teacher last hidden states.

        Returns:
            Tuple of ``(total_loss, metrics_dict)`` where ``metrics_dict``
            has keys ``"task_loss"``, ``"distill_loss"``, and ``"total_loss"``.
        """
        t_loss = self.task_loss(logits, labels)
        d_loss = self.distillation_loss(student_hidden, teacher_hidden)

        w = self.config.distill_weight
        total = (1.0 - w) * t_loss + w * d_loss

        metrics: dict[str, Tensor] = {
            "task_loss": t_loss,
            "distill_loss": d_loss,
            "total_loss": total,
        }
        return total, metrics


# ---------------------------------------------------------------------------
# Implicit CoT Trainer
# ---------------------------------------------------------------------------


class ImplicitCoTTrainer:
    """Orchestrates Implicit Chain-of-Thought training.

    Handles freezing the teacher, extracting hidden states from both
    student and teacher, computing the combined loss, and updating student
    weights.

    Args:
        student_model: The model being trained.  Must accept ``input_ids``
            and return either ``(logits, hidden)`` or just ``logits``.
        teacher_model: The frozen reference model with the same interface.
        optimizer: A :class:`torch.optim.Optimizer` over student parameters.
        loss_fn: An :class:`ImplicitCoTLoss` instance.
    """

    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: ImplicitCoTLoss,
    ) -> None:
        self.student = student_model
        self.teacher = teacher_model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def freeze_teacher(self) -> None:
        """Freeze all parameters of the teacher model."""
        for param in self.teacher.parameters():
            param.requires_grad_(False)
        self.teacher.eval()

    def extract_hidden_states(self, model: nn.Module, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Run ``model`` on ``input_ids`` and return ``(logits, last_hidden)``.

        Handles two return conventions:
        - ``(logits, hidden)`` — the model returns both directly.
        - ``logits`` only — hidden states are approximated as the logit
          tensor itself (zero-cost fallback; prefer models that expose
          hidden states).

        Args:
            model: The language model to run.
            input_ids: ``(B, T)`` integer token ids.

        Returns:
            ``(logits, hidden)`` — shapes ``(B, T, V)`` and ``(B, T, d)``.
        """
        output = model(input_ids)

        if isinstance(output, (tuple, list)) and len(output) >= 2:
            logits, hidden = output[0], output[1]
        else:
            # Only logits returned — treat logits themselves as a proxy for
            # hidden states (useful for testing with minimal stubs).
            logits = output
            hidden = logits  # (B, T, V) used as pseudo-hidden

        return logits, hidden

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self, input_ids: Tensor, labels: Tensor) -> dict[str, float]:
        """Perform one forward + backward + optimiser update.

        Args:
            input_ids: ``(B, T)`` integer token ids.
            labels: ``(B, T)`` target token ids (``-100`` for ignored positions).

        Returns:
            Dict with float values for ``"task_loss"``, ``"distill_loss"``,
            and ``"total_loss"``.
        """
        self.student.train()

        # Student forward (with gradients)
        student_logits, student_hidden = self.extract_hidden_states(self.student, input_ids)

        # Teacher forward (no gradients)
        with torch.no_grad():
            _, teacher_hidden = self.extract_hidden_states(self.teacher, input_ids)

        total_loss, metrics = self.loss_fn(student_logits, labels, student_hidden, teacher_hidden)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {k: v.item() for k, v in metrics.items()}
