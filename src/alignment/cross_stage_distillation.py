"""On-policy cross-stage distillation for RL training.

Compatibility layer for both the newer `.loss()` tests and the legacy
training-side `compute_kl_loss()` / `set_reference()` usage.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

__all__ = ["CrossStageDistillation", "stage_transition"]


class CrossStageDistillation:
    """Cross-stage KL regularizer with a compact compatibility surface."""

    def __init__(
        self,
        alpha: float = 0.1,
        beta: float | None = None,
        adaptive: bool = False,
    ) -> None:
        if beta is not None:
            alpha = beta
        self.alpha = float(alpha)
        self.beta = self.alpha
        self.adaptive = adaptive
        self.ref_model: nn.Module | None = None
        self._frozen = False

    def set_reference(self, model: nn.Module) -> None:
        """Freeze a reference model for legacy training-side KL computation."""
        self.ref_model = model
        self._frozen = True
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()

    def _extract_logits(self, output: Any) -> torch.Tensor:
        if isinstance(output, (tuple, list)):
            return output[0]
        if hasattr(output, "logits"):
            return output.logits
        return output

    def _kl_from_logits(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        teacher_logits = teacher_logits.detach()
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        kl_per_token = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="none",
        ).sum(dim=-1)

        if attention_mask is not None:
            mask = attention_mask.to(dtype=kl_per_token.dtype)
            if mask.shape != kl_per_token.shape:
                mask = mask.expand_as(kl_per_token)
            kl_per_token = kl_per_token * mask

        return kl_per_token.mean()

    def compute_kl_loss(
        self,
        student_logits: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the raw KL term against the frozen reference model.

        If no reference model has been registered, returns a zero tensor on
        the correct device so callers can safely include it in totals.
        """
        if self.ref_model is None:
            return torch.zeros((), device=student_logits.device, dtype=student_logits.dtype)

        with torch.no_grad():
            ref_output = self.ref_model(input_ids)
        ref_logits = self._extract_logits(ref_output)
        return self._kl_from_logits(student_logits, ref_logits, attention_mask=attention_mask)

    def loss(
        self,
        rl_loss: torch.Tensor,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return RL loss plus the weighted KL regularizer."""
        kl_loss = self._kl_from_logits(
            student_logits,
            teacher_logits,
            attention_mask=attention_mask,
        )
        if self.adaptive:
            scale = min(1.0, kl_loss.detach().item() / 0.1) if kl_loss.detach().item() > 0 else 1.0
        else:
            scale = 1.0
        return rl_loss + self.alpha * scale * kl_loss


def stage_transition(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    next_stage_lr: float = 5e-7,
) -> None:
    """Prepare for the next RL stage transition."""
    for group in optimizer.param_groups:
        group["lr"] = next_stage_lr
    logger.info("Stage transition: LR set to %s", next_stage_lr)
