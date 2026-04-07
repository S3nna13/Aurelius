"""Knowledge Distillation for LLMs (Hinton et al., 2015 - arXiv:1503.02531).

Trains a student model to match both:
1. The ground-truth labels (hard target loss)
2. The teacher's softened output distribution (soft target loss)

Combined loss: alpha * CE(student, labels) + (1-alpha) * T^2 * KL(student_soft || teacher_soft)

The T^2 factor compensates for the reduced magnitude of gradients when
using softened distributions at temperature T.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class DistillationConfig:
    temperature: float = 4.0       # softening temperature T
    alpha: float = 0.5             # weight for hard target loss (1-alpha for soft)
    learning_rate: float = 1e-4
    num_steps: int = 1000
    batch_size: int = 4
    seq_len: int = 512
    grad_clip: float = 1.0
    log_interval: int = 100
    save_interval: int = 500
    output_dir: str = "checkpoints/distilled"


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute combined hard + soft distillation loss.

    Args:
        student_logits: (B, seq_len, vocab_size) --- student model output.
        teacher_logits: (B, seq_len, vocab_size) --- teacher model output (no grad).
        labels: (B, seq_len) --- ground truth token ids.
        temperature: Softening temperature. Higher = softer distribution.
        alpha: Weight for hard target loss.

    Returns:
        (total_loss, hard_loss, soft_loss) --- all scalar tensors.
    """
    B, S, V = student_logits.shape

    # Hard loss: student vs ground truth (standard cross-entropy, shifted)
    hard_loss = F.cross_entropy(
        student_logits[:, :-1].contiguous().view(-1, V),
        labels[:, 1:].contiguous().view(-1),
    )

    # Soft loss: KL divergence between softened distributions
    student_soft = F.log_softmax(student_logits[:, :-1] / temperature, dim=-1)  # (B, S-1, V)
    teacher_soft = F.softmax(teacher_logits[:, :-1].detach() / temperature, dim=-1)  # (B, S-1, V)

    # KL(student || teacher): F.kl_div(log_input=student, target=teacher)
    soft_loss = F.kl_div(
        student_soft.view(-1, V),
        teacher_soft.view(-1, V),
        reduction="batchmean",
    ) * (temperature ** 2)  # T^2 compensation

    total_loss = alpha * hard_loss + (1 - alpha) * soft_loss
    return total_loss, hard_loss, soft_loss


class DistillationTrainer:
    """Trains a student model to match a teacher via knowledge distillation.

    Args:
        teacher: Frozen teacher model (larger, better quality).
        student: Student model to train (smaller, faster).
        cfg: Distillation configuration.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        cfg: DistillationConfig | None = None,
    ) -> None:
        self.teacher = teacher
        self.student = student
        self.cfg = cfg or DistillationConfig()

        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

        self.optimizer = torch.optim.AdamW(
            [p for p in student.parameters() if p.requires_grad],
            lr=cfg.learning_rate if cfg else 1e-4,
        )

    def step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, float]:
        """One distillation training step.

        Args:
            input_ids: (B, seq_len) input token ids.
            labels: (B, seq_len) target token ids (often same as input_ids for LM).

        Returns:
            Dict with total_loss, hard_loss, soft_loss.
        """
        self.student.train()
        self.optimizer.zero_grad()

        # Teacher forward (no grad)
        with torch.no_grad():
            _, teacher_logits, _ = self.teacher(input_ids)

        # Student forward
        _, student_logits, _ = self.student(input_ids)

        # Compute distillation loss
        total_loss, hard_loss, soft_loss = distillation_loss(
            student_logits, teacher_logits, labels,
            temperature=self.cfg.temperature,
            alpha=self.cfg.alpha,
        )

        total_loss.backward()
        nn.utils.clip_grad_norm_(self.student.parameters(), self.cfg.grad_clip)
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "hard_loss": hard_loss.item(),
            "soft_loss": soft_loss.item(),
        }

    def train(self, dataloader) -> list[dict[str, float]]:
        """Run full distillation training on a DataLoader.

        DataLoader should yield {"input_ids": Tensor, "labels": Tensor} dicts
        or (input_ids, labels) tuples.

        Returns:
            List of per-step metric dicts.
        """
        from src.training.checkpoint import save_checkpoint

        history = []
        step = 0

        for batch in dataloader:
            if step >= self.cfg.num_steps:
                break

            if isinstance(batch, dict):
                input_ids = batch["input_ids"]
                labels = batch.get("labels", batch["input_ids"])
            else:
                input_ids, labels = batch[0], batch[1]

            metrics = self.step(input_ids, labels)
            history.append(metrics)
            step += 1

            if step % self.cfg.log_interval == 0:
                logger.info(
                    "step=%d  loss=%.4f  hard=%.4f  soft=%.4f  ppl=%.2f",
                    step,
                    metrics["total_loss"],
                    metrics["hard_loss"],
                    metrics["soft_loss"],
                    math.exp(min(metrics["hard_loss"], 20)),
                )

            if step % self.cfg.save_interval == 0:
                save_checkpoint(
                    self.student,
                    self.optimizer,
                    step=step,
                    epoch=0,
                    train_loss=metrics["total_loss"],
                    output_dir=self.cfg.output_dir,
                )

        return history
