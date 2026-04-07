"""Forward KL knowledge distillation loss.

Combines a soft-label KL divergence term (teacher → student) with a hard-label
cross-entropy term, weighted by *alpha*:

    L = alpha * KL(teacher || student) + (1 - alpha) * CE(labels, student)

Temperature *T* softens both distributions; the KL term is rescaled by T² to
preserve gradient magnitude (Hinton et al., 2015).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """Forward KL distillation from teacher logits.

    Loss = alpha * KL(teacher || student) + (1 - alpha) * CE(labels, student)
    Temperature T softens both distributions.

    Args:
        temperature: Softmax temperature applied to both student and teacher
            logits before computing the KL divergence.  Values > 1 produce
            softer distributions.
        alpha: Blending weight for the KL term.  ``alpha=1.0`` uses only the
            distillation loss; ``alpha=0.0`` uses only the hard-label CE loss.
    """

    def __init__(self, temperature: float = 2.0, alpha: float = 0.5) -> None:
        super().__init__()
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = temperature
        self.alpha = alpha

    def forward(
        self,
        student_logits: torch.Tensor,   # (batch, seq_len, vocab_size)
        teacher_logits: torch.Tensor,   # (batch, seq_len, vocab_size)
        labels: torch.Tensor,           # (batch, seq_len), -100 = ignore
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the blended distillation loss.

        Args:
            student_logits: Raw logits from the student model.
            teacher_logits: Raw logits from the teacher model (no gradient
                needed, but gradients are not explicitly detached here to keep
                the API flexible).
            labels: Integer token ids; positions with value ``-100`` are
                excluded from the CE loss.

        Returns:
            A 3-tuple ``(total_loss, kl_loss, ce_loss)`` where each element is
            a scalar tensor.
        """
        T = self.temperature

        # ------------------------------------------------------------------
        # KL divergence loss (soft targets)
        # ------------------------------------------------------------------
        # F.kl_div expects log-probabilities for input and probabilities for
        # target:  KL(target || input) = sum(target * (log target - input))
        # Using batchmean reduction then scaling by T² restores gradient scale.
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits.detach() / T, dim=-1)

        # Flatten (batch * seq_len, vocab) for batchmean
        batch_seq = student_log_probs.shape[0] * student_log_probs.shape[1]
        kl_loss: torch.Tensor = F.kl_div(
            student_log_probs.view(batch_seq, -1),
            teacher_probs.view(batch_seq, -1),
            reduction="batchmean",
        ) * (T ** 2)

        # ------------------------------------------------------------------
        # Cross-entropy loss (hard labels, ignore -100)
        # ------------------------------------------------------------------
        # student_logits: (B, S, V) → CE expects (B, V, S)
        ce_loss: torch.Tensor = F.cross_entropy(
            student_logits.permute(0, 2, 1),
            labels,
            ignore_index=-100,
        )

        # ------------------------------------------------------------------
        # Blend
        # ------------------------------------------------------------------
        total_loss = self.alpha * kl_loss + (1.0 - self.alpha) * ce_loss
        return total_loss, kl_loss, ce_loss
