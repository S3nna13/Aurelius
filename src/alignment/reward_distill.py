"""Reward model distillation: student learns from a frozen teacher reward model.

Supports:
1. Regression distillation -- student MSE-matches teacher scalar rewards.
2. Ranking distillation -- student preserves teacher pairwise preference order.
3. Temperature-scaled soft target distillation via KL divergence.
4. Contrastive distillation -- cosine embedding loss on reward differences.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class DistillConfig:
    """Configuration for reward distillation training."""
    temperature: float = 2.0          # temperature for soft targets
    alpha: float = 0.5                # weight: alpha * distill + (1-alpha) * hard label
    ranking_margin: float = 0.1       # margin for ranking hinge loss
    contrastive_margin: float = 1.0   # margin for contrastive loss
    use_regression: bool = True
    use_ranking: bool = True
    use_contrastive: bool = False


def rank_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    """Manual Spearman rank correlation between two 1D tensors.

    No scipy. Uses argsort twice to convert values to ranks, then computes
    Pearson correlation on the resulting rank vectors.

    Args:
        x: 1D tensor of values.
        y: 1D tensor of values, same length as x.

    Returns:
        Float in [-1, 1].
    """
    n = x.numel()
    if n < 2:
        return 1.0

    def _ranks(t: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(t)
        ranks = torch.zeros_like(order, dtype=torch.float32)
        ranks[order] = torch.arange(n, dtype=torch.float32)
        return ranks

    rx = _ranks(x.float())
    ry = _ranks(y.float())

    rx_c = rx - rx.mean()
    ry_c = ry - ry.mean()

    num = (rx_c * ry_c).sum()
    denom = (rx_c.pow(2).sum() * ry_c.pow(2).sum()).sqrt()
    if denom.item() == 0.0:
        return 1.0
    return (num / denom).item()


def knowledge_distillation_reward(
    teacher_rewards: torch.Tensor,  # (B,)
    student_rewards: torch.Tensor,  # (B,)
    temperature: float = 2.0,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Combined distillation loss.

    alpha * KL(teacher_soft || student_soft) + (1-alpha) * MSE.
    Converts scalar rewards to 2-class soft labels via sigmoid-based logits.

    Args:
        teacher_rewards: (B,) teacher scalar rewards.
        student_rewards: (B,) student scalar rewards.
        temperature: Softmax temperature applied to 2-class logits.
        alpha: Weight between KL term and MSE term.

    Returns:
        Scalar loss tensor.
    """
    def _to_logits(rewards: torch.Tensor) -> torch.Tensor:
        # Build (B, 2) logits: [rejected_score, chosen_score] = [-r, r]
        return torch.stack([-rewards, rewards], dim=1)

    t_logits = _to_logits(teacher_rewards) / temperature
    s_logits = _to_logits(student_rewards) / temperature

    t_soft = F.softmax(t_logits, dim=1)
    s_log_soft = F.log_softmax(s_logits, dim=1)

    kl = F.kl_div(s_log_soft, t_soft, reduction="batchmean")
    mse = F.mse_loss(student_rewards, teacher_rewards)
    return alpha * kl + (1.0 - alpha) * mse


class RewardDistillTrainer:
    """Distill a large teacher reward model into a smaller student reward model.

    The teacher is always kept frozen (no_grad). Supports regression, ranking,
    soft-target (KL), and contrastive distillation objectives.

    Args:
        teacher: Frozen reward model. forward(hidden) -> (B,) scalar rewards.
        student: Trainable reward model with the same interface.
        config: Distillation hyper-parameters.
        lr: Learning rate for the Adam optimizer.
        device: Torch device string.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        config: Optional[DistillConfig] = None,
        lr: float = 1e-4,
        device: str = "cpu",
    ) -> None:
        self.config = config if config is not None else DistillConfig()
        self.device = device
        self.teacher = teacher.to(device)
        self.student = student.to(device)

        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Individual loss components
    # ------------------------------------------------------------------

    def regression_loss(
        self,
        student_rewards: torch.Tensor,  # (B,)
        teacher_rewards: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        """MSE between student and teacher reward predictions."""
        return F.mse_loss(student_rewards, teacher_rewards)

    def ranking_loss(
        self,
        student_chosen: torch.Tensor,    # (B,)
        student_rejected: torch.Tensor,  # (B,)
        teacher_chosen: torch.Tensor,    # (B,)
        teacher_rejected: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        """Hinge loss weighted by teacher confidence.

        loss = mean(teacher_weight * max(0, margin - (s_chosen - s_rejected)))

        Teacher weight = softplus(teacher_chosen - teacher_rejected) so that
        pairs where the teacher is more confident contribute more.
        """
        margin = self.config.ranking_margin
        s_diff = student_chosen - student_rejected
        hinge = F.relu(margin - s_diff)

        t_diff = teacher_chosen - teacher_rejected
        weight = F.softplus(t_diff).detach()

        return (weight * hinge).mean()

    def soft_target_loss(
        self,
        student_logits: torch.Tensor,  # (B, 2)
        teacher_logits: torch.Tensor,  # (B, 2)
    ) -> torch.Tensor:
        """KL(softmax(teacher/T) || softmax(student/T)) per sample, mean-reduced."""
        T = self.config.temperature
        t_soft = F.softmax(teacher_logits / T, dim=1)
        s_log_soft = F.log_softmax(student_logits / T, dim=1)
        return F.kl_div(s_log_soft, t_soft, reduction="batchmean")

    def contrastive_distill_loss(
        self,
        student_rewards: torch.Tensor,  # (B,)
        teacher_rewards: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        """Push student rewards to have same sign of difference as teacher rewards.

        Uses cosine embedding loss on reward difference vectors built from
        consecutive sample pairs.
        """
        if student_rewards.numel() < 2:
            return torch.tensor(0.0, device=student_rewards.device, requires_grad=True)

        s_diff = student_rewards[:-1] - student_rewards[1:]
        t_diff = teacher_rewards[:-1] - teacher_rewards[1:]

        target = torch.sign(t_diff).detach()
        # Replace ties (sign==0) with +1 (no constraint needed)
        target = torch.where(target == 0, torch.ones_like(target), target).long()

        s_vec = s_diff.unsqueeze(1)
        t_vec = t_diff.unsqueeze(1)
        return F.cosine_embedding_loss(
            s_vec, t_vec, target, margin=self.config.contrastive_margin
        )

    # ------------------------------------------------------------------
    # Combined loss
    # ------------------------------------------------------------------

    def compute_total_loss(
        self,
        hidden_chosen: torch.Tensor,
        hidden_rejected: torch.Tensor,
        hard_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined distillation loss.

        Runs teacher (no_grad) and student on both chosen and rejected hidden
        states, then combines enabled loss terms.

        Returns:
            Tuple of (total_loss tensor, metrics dict).
        """
        hidden_chosen = hidden_chosen.to(self.device)
        hidden_rejected = hidden_rejected.to(self.device)

        with torch.no_grad():
            t_chosen = self.teacher(hidden_chosen).squeeze(-1)
            t_rejected = self.teacher(hidden_rejected).squeeze(-1)

        s_chosen = self.student(hidden_chosen).squeeze(-1)
        s_rejected = self.student(hidden_rejected).squeeze(-1)

        total = torch.tensor(0.0, device=self.device)
        metrics: Dict[str, float] = {}

        if self.config.use_regression:
            r_loss = self.regression_loss(
                torch.cat([s_chosen, s_rejected]),
                torch.cat([t_chosen, t_rejected]),
            )
            total = total + r_loss
            metrics["regression_loss"] = r_loss.item()

        if self.config.use_ranking:
            rank_l = self.ranking_loss(s_chosen, s_rejected, t_chosen, t_rejected)
            total = total + rank_l
            metrics["ranking_loss"] = rank_l.item()

        s_logits = torch.stack([s_rejected, s_chosen], dim=1)
        t_logits = torch.stack([t_rejected, t_chosen], dim=1)
        soft_l = self.soft_target_loss(s_logits, t_logits)
        total = total + self.config.alpha * soft_l
        metrics["soft_target_loss"] = soft_l.item()

        if self.config.use_contrastive:
            c_loss = self.contrastive_distill_loss(
                torch.cat([s_chosen, s_rejected]),
                torch.cat([t_chosen, t_rejected]),
            )
            total = total + c_loss
            metrics["contrastive_loss"] = c_loss.item()

        if hard_labels is not None:
            hard_labels = hard_labels.float().to(self.device)
            s_prefs = torch.sigmoid(s_chosen - s_rejected)
            bce = F.binary_cross_entropy(s_prefs, hard_labels)
            total = total + (1.0 - self.config.alpha) * bce
            metrics["hard_label_loss"] = bce.item()

        metrics["total_loss"] = total.item()
        return total, metrics

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(
        self,
        hidden_chosen: torch.Tensor,
        hidden_rejected: torch.Tensor,
        hard_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Zero grad, compute loss, backward, optimizer step.

        Returns:
            Metrics dict including 'loss' key.
        """
        self.student.train()
        self.optimizer.zero_grad()

        total, metrics = self.compute_total_loss(hidden_chosen, hidden_rejected, hard_labels)
        total.backward()
        self.optimizer.step()

        metrics["loss"] = metrics["total_loss"]
        return metrics

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_agreement(
        self,
        hidden_states: torch.Tensor,
        teacher_rewards: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute agreement metrics between student and teacher rankings.

        Metrics:
            rank_correlation: Spearman-like correlation (manual rank computation).
            top_k_agreement: fraction of teacher top-K items in student top-K.
            sign_agreement: fraction where sign(student_r) == sign(teacher_r).

        Args:
            hidden_states: (N, hidden_dim) input hidden states.
            teacher_rewards: (N,) teacher scalar rewards.

        Returns:
            Dict with float values for each metric.
        """
        self.student.eval()
        hidden_states = hidden_states.to(self.device)
        teacher_rewards = teacher_rewards.to(self.device)

        with torch.no_grad():
            student_rewards = self.student(hidden_states).squeeze(-1)

        N = student_rewards.numel()

        rho = rank_correlation(student_rewards.cpu(), teacher_rewards.cpu())

        k = max(1, N // 2)
        teacher_topk = torch.topk(teacher_rewards, k).indices
        student_topk = torch.topk(student_rewards, k).indices
        teacher_set = set(teacher_topk.tolist())
        student_set = set(student_topk.tolist())
        top_k_agr = len(teacher_set & student_set) / k

        if N == 0:
            sign_agr = 1.0
        else:
            s_sign = torch.sign(student_rewards)
            t_sign = torch.sign(teacher_rewards)
            sign_agr = (s_sign == t_sign).float().mean().item()

        return {
            "rank_correlation": float(rho),
            "top_k_agreement": float(top_k_agr),
            "sign_agreement": float(sign_agr),
        }
