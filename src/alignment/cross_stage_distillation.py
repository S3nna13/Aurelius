"""On-Policy Cross-Stage Distillation — GLM-5 §5.4 (arXiv:2602.15763).

Prevents catastrophic forgetting across sequential RL curriculum stages.
After each sequential RL stage (reasoning → agentic → general RL), the final
checkpoint of the preceding stage serves as a teacher for on-policy KL
regularization applied to the current student policy.

Loss formula:
    L_CSD = L_RL(θ) + α · KL(π_θ ∥ π_teacher)

where:
    KL(π_θ ∥ π_teacher) = E[ log(π_θ / π_teacher) ]
                         = token-level KL averaged over B, T

Teacher logits are always detached — no gradient flows through the teacher.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class CrossStageDistillation:
    """On-policy cross-stage KL regularizer from GLM-5 §5.4.

    Args:
        alpha: Weight for the KL divergence regularisation term. Set to 0.0
               to disable distillation (pure RL loss pass-through).
    """

    alpha: float = 0.1

    def loss(
        self,
        rl_loss: torch.Tensor,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute L_CSD = L_RL + alpha * KL(pi_theta || pi_teacher).

        Args:
            rl_loss: Scalar RL loss for the current stage, e.g. GRPO / PPO
                     loss. Must already be reduced to a scalar.
            student_logits: Raw (un-normalised) logits from the student policy,
                            shape [B, T, V].
            teacher_logits: Raw logits from the preceding-stage teacher
                            checkpoint, shape [B, T, V]. These are detached
                            internally — callers do NOT need to detach first.
            attention_mask: Optional boolean / float mask of shape [B, T].
                            Tokens where mask == 0 (padding) are excluded from
                            the KL average. When None, all tokens contribute.

        Returns:
            Scalar tensor: L_RL + alpha * KL_mean.
        """
        if self.alpha == 0.0:
            return rl_loss

        # Teacher logits detached — no gradient through the teacher branch.
        p_teacher = F.softmax(teacher_logits.detach(), dim=-1)  # [B, T, V]
        log_p_student = F.log_softmax(student_logits, dim=-1)  # [B, T, V]

        # Per-token KL: KL(pi_theta || pi_teacher) at each position.
        # kl_div expects (log_input, target) and returns element-wise values.
        kl = F.kl_div(log_p_student, p_teacher, reduction="none").sum(-1)  # [B, T]

        if attention_mask is not None:
            # Cast mask to same dtype for multiplication.
            mask = attention_mask.to(kl.dtype)
            kl = kl * mask
            kl = kl.sum() / (mask.sum() + 1e-8)
        else:
            kl = kl.mean()

        return rl_loss + self.alpha * kl
