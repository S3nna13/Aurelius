"""Knowledge Distillation v3 for LLMs.

Implements a comprehensive suite of knowledge distillation methods:
  - Response-based: SoftTargetLoss (logit matching with temperature scaling)
  - Feature-based: FeatDistilLoss (hidden state matching with projection)
  - Relation-based: AttentionTransferLoss (attention map matching)
  - Patient KD: PKDLoss (Sun et al. 2019)
  - Orchestrator: DistilTrainer
  - Evaluation: CompressionBenchmark

Pure native PyTorch only -- stdlib + torch, no external dependencies.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# SoftTargetLoss
# ---------------------------------------------------------------------------


class SoftTargetLoss(nn.Module):
    """Classic Hinton (2015) knowledge distillation loss with temperature scaling.

    Combines a soft KD loss (KL divergence on soft targets) with a hard-label
    cross-entropy loss, weighted by alpha.

    Args:
        temperature: Softening temperature T. Higher values produce softer
            probability distributions. Default: 4.0.
        alpha: Weight for the distillation (KD) loss. The CE loss receives
            weight (1 - alpha). Default: 0.7.
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.7) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.temperature = temperature
        self.alpha = alpha

    def forward(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        hard_labels: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute combined distillation + CE loss.

        Args:
            student_logits: (B, V) or (B, T, V) student output logits.
            teacher_logits: Same shape as student_logits, teacher outputs.
            hard_labels: (B,) or (B, T) integer class labels.

        Returns:
            (total_loss, kd_loss, ce_loss) -- all scalar tensors.
        """
        T = self.temperature
        alpha = self.alpha

        # Flatten sequence dimension if present so KL operates on (N, V)
        if student_logits.dim() == 3:
            B, S, V = student_logits.shape
            s_logits = student_logits.reshape(B * S, V)
            t_logits = teacher_logits.reshape(B * S, V)
            flat_labels = hard_labels.reshape(B * S)
        else:
            s_logits = student_logits
            t_logits = teacher_logits
            flat_labels = hard_labels

        # Soft targets with temperature
        student_soft = F.log_softmax(s_logits / T, dim=-1)
        teacher_soft = F.softmax(t_logits / T, dim=-1)

        # KL divergence: KL(teacher || student)
        # F.kl_div expects log-probabilities as input and probabilities as target
        # reduction='batchmean' divides by batch size (N)
        kd_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (T**2)

        # Hard-label cross-entropy
        ce_loss = F.cross_entropy(s_logits, flat_labels)

        total_loss = alpha * kd_loss + (1.0 - alpha) * ce_loss
        return total_loss, kd_loss, ce_loss


# ---------------------------------------------------------------------------
# FeatDistilLoss
# ---------------------------------------------------------------------------


class FeatDistilLoss(nn.Module):
    """Feature-level distillation on hidden states.

    Projects student hidden states into the teacher's dimensional space,
    normalises both to unit norm, then computes MSE. Teacher features are
    treated as fixed targets (no_grad).

    Args:
        student_dim: Dimensionality of student hidden states.
        teacher_dim: Dimensionality of teacher hidden states.
    """

    def __init__(self, student_dim: int, teacher_dim: int) -> None:
        super().__init__()
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        # Linear projection to align student dimension with teacher
        self.projector = nn.Linear(student_dim, teacher_dim, bias=False)

    def forward(self, student_hidden: Tensor, teacher_hidden: Tensor) -> Tensor:
        """Compute normalised-MSE between projected student and teacher features.

        Args:
            student_hidden: (B, T, student_dim) student hidden states.
            teacher_hidden: (B, T, teacher_dim) teacher hidden states.

        Returns:
            Scalar MSE loss after L2-normalisation of both representations.
        """
        # Project student to teacher dimension
        projected = self.projector(student_hidden)  # (B, T, teacher_dim)

        # No gradients flow through teacher
        teacher_detached = teacher_hidden.detach()
        teacher_normed = F.normalize(teacher_detached, p=2, dim=-1)
        student_normed = F.normalize(projected, p=2, dim=-1)

        return F.mse_loss(student_normed, teacher_normed)

    def layer_mapping(self, n_student_layers: int, n_teacher_layers: int) -> list[tuple[int, int]]:
        """Compute uniform spacing layer mapping from student to teacher layers.

        Each student layer i maps to teacher layer round(i * n_teacher / n_student).

        Args:
            n_student_layers: Number of student layers.
            n_teacher_layers: Number of teacher layers.

        Returns:
            List of (student_layer_idx, teacher_layer_idx) pairs of length
            n_student_layers.
        """
        mapping = []
        for i in range(n_student_layers):
            teacher_idx = round(i * n_teacher_layers / n_student_layers)
            # Clamp to valid range
            teacher_idx = min(teacher_idx, n_teacher_layers - 1)
            mapping.append((i, teacher_idx))
        return mapping


# ---------------------------------------------------------------------------
# AttentionTransferLoss
# ---------------------------------------------------------------------------


class AttentionTransferLoss(nn.Module):
    """Match attention patterns between teacher and student models.

    Normalises each attention map row-wise (over keys), then computes MSE.
    If teacher and student have different numbers of heads, teacher heads are
    averaged down to match the student head count.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, student_attn: Tensor, teacher_attn: Tensor) -> Tensor:
        """Compute attention transfer loss.

        Args:
            student_attn: (B, H_s, T, T) student attention weights.
            teacher_attn: (B, H_t, T, T) teacher attention weights.

        Returns:
            Scalar MSE loss between row-normalised attention maps.
        """
        B_s, H_s, T_s, _ = student_attn.shape
        B_t, H_t, T_t, _ = teacher_attn.shape

        # If head counts differ, average teacher heads to match student count
        if H_t != H_s:
            if H_t % H_s == 0:
                factor = H_t // H_s
                teacher_attn = teacher_attn.reshape(B_t, H_s, factor, T_t, T_t).mean(dim=2)
            else:
                # Interpolate over head dimension: treat (B*T*T) as batch, heads as spatial
                teacher_attn = teacher_attn.permute(0, 2, 3, 1)  # (B, T, T, H_t)
                teacher_attn = teacher_attn.reshape(B_t * T_t * T_t, 1, H_t).float()
                teacher_attn = F.interpolate(
                    teacher_attn, size=H_s, mode="linear", align_corners=False
                )
                teacher_attn = teacher_attn.reshape(B_t, T_t, T_t, H_s).permute(0, 3, 1, 2)

        # Row-wise normalisation: each row (query) sums to 1
        student_norm = student_attn / (student_attn.sum(dim=-1, keepdim=True) + 1e-9)
        teacher_norm = teacher_attn / (teacher_attn.sum(dim=-1, keepdim=True) + 1e-9)

        return F.mse_loss(student_norm, teacher_norm)


# ---------------------------------------------------------------------------
# PKDLoss
# ---------------------------------------------------------------------------


class PKDLoss(nn.Module):
    """Patient Knowledge Distillation (Sun et al. 2019).

    Maps specified student layers to teacher layers, mean-pools over the
    sequence dimension, normalises to unit vectors, and computes MSE.

    Args:
        student_layers: List of student layer indices to match.
        teacher_layers: List of teacher layer indices paired with student_layers.
            Must have the same length as student_layers.
    """

    def __init__(self, student_layers: list[int], teacher_layers: list[int]) -> None:
        super().__init__()
        if len(student_layers) != len(teacher_layers):
            raise ValueError(
                "student_layers and teacher_layers must have the same length, "
                f"got {len(student_layers)} and {len(teacher_layers)}"
            )
        self.student_layers = student_layers
        self.teacher_layers = teacher_layers

    def forward(
        self,
        student_hiddens: list[Tensor],
        teacher_hiddens: list[Tensor],
    ) -> Tensor:
        """Compute PKD loss over paired layer representations.

        Args:
            student_hiddens: List of (B, T, D_s) student hidden states, one per layer.
            teacher_hiddens: List of (B, T, D_t) teacher hidden states, one per layer.

        Returns:
            Scalar mean PKD loss across all paired layers.
        """
        # Accumulate into a zero tensor on the correct device/dtype
        total_loss = student_hiddens[0].new_zeros(())

        for s_idx, t_idx in zip(self.student_layers, self.teacher_layers):
            s_h = student_hiddens[s_idx]  # (B, T, D_s)
            t_h = teacher_hiddens[t_idx]  # (B, T, D_t)

            # Mean pool over sequence dimension: (B, D)
            s_pooled = s_h.mean(dim=1)
            t_pooled = t_h.mean(dim=1)

            # Normalise to unit vectors
            s_norm = F.normalize(s_pooled, p=2, dim=-1)
            t_norm = F.normalize(t_pooled, p=2, dim=-1)

            total_loss = total_loss + F.mse_loss(s_norm, t_norm)

        n_pairs = len(self.student_layers)
        if n_pairs > 0:
            total_loss = total_loss / n_pairs

        return total_loss


# ---------------------------------------------------------------------------
# DistilTrainer
# ---------------------------------------------------------------------------


class DistilTrainer:
    """Orchestrate multi-objective knowledge distillation training.

    Combines soft-target (logit), feature, and attention distillation losses
    into a single training step. The teacher model is frozen at construction.

    Args:
        student_model: Model to train. Must accept input_ids and return
            (logits, hidden_states_list, attn_maps_list).
        teacher_model: Fixed reference model with the same interface.
        optimizer: PyTorch optimizer for the student model parameters.
        soft_target_loss: SoftTargetLoss instance.
        feat_loss: FeatDistilLoss instance.
        attn_loss: AttentionTransferLoss instance.
    """

    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        soft_target_loss: SoftTargetLoss,
        feat_loss: FeatDistilLoss,
        attn_loss: AttentionTransferLoss,
    ) -> None:
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.optimizer = optimizer
        self.soft_target_loss = soft_target_loss
        self.feat_loss = feat_loss
        self.attn_loss = attn_loss

        # Freeze all teacher parameters
        for param in self.teacher_model.parameters():
            param.requires_grad_(False)
        self.teacher_model.eval()

    def train_step(self, input_ids: Tensor, labels: Tensor) -> dict:
        """Execute one distillation training step.

        Args:
            input_ids: (B, T) integer token ids.
            labels: (B, T) or (B,) integer target labels for CE loss.

        Returns:
            Dictionary with keys: total_loss, kd_loss, feat_loss, attn_loss, ce_loss.
            All values are Python floats (detached).
        """
        self.student_model.train()
        self.optimizer.zero_grad()

        # Student forward pass
        s_logits, s_hiddens, s_attns = self.student_model(input_ids)

        # Teacher forward pass (no gradients)
        with torch.no_grad():
            t_logits, t_hiddens, t_attns = self.teacher_model(input_ids)

        # 1. Soft-target (logit) distillation + CE
        total_loss, kd_loss_val, ce_loss_val = self.soft_target_loss(s_logits, t_logits, labels)

        # 2. Feature distillation (first paired layers)
        feat_loss_scalar = torch.zeros(1, device=total_loss.device)
        if s_hiddens and t_hiddens:
            feat_loss_scalar = self.feat_loss(s_hiddens[0], t_hiddens[0])
            total_loss = total_loss + feat_loss_scalar

        # 3. Attention transfer (first paired layers)
        attn_loss_scalar = torch.zeros(1, device=total_loss.device)
        if s_attns and t_attns:
            attn_loss_scalar = self.attn_loss(s_attns[0], t_attns[0])
            total_loss = total_loss + attn_loss_scalar

        total_loss.backward()
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "kd_loss": kd_loss_val.item(),
            "feat_loss": feat_loss_scalar.item(),
            "attn_loss": attn_loss_scalar.item(),
            "ce_loss": ce_loss_val.item(),
        }


# ---------------------------------------------------------------------------
# CompressionBenchmark
# ---------------------------------------------------------------------------


class CompressionBenchmark:
    """Measure student vs. teacher quality metrics for compression analysis."""

    def __init__(self) -> None:
        pass

    def parameter_ratio(self, student: nn.Module, teacher: nn.Module) -> float:
        """Compute ratio of student to teacher parameter counts.

        Args:
            student: Student model.
            teacher: Teacher model.

        Returns:
            student_params / teacher_params as a float.
        """
        student_params = sum(p.numel() for p in student.parameters())
        teacher_params = sum(p.numel() for p in teacher.parameters())
        if teacher_params == 0:
            raise ValueError("Teacher model has no parameters.")
        return student_params / teacher_params

    def perplexity_gap(self, student_logprobs: Tensor, teacher_logprobs: Tensor) -> float:
        """Compute the perplexity ratio: exp(mean(teacher_lp - student_lp)).

        A value >= 1.0 indicates the student has equal or higher perplexity
        (worse) than the teacher.

        Args:
            student_logprobs: (N,) per-token log-probabilities from student.
            teacher_logprobs: (N,) per-token log-probabilities from teacher.

        Returns:
            Ratio of student perplexity to teacher perplexity as a Python float.
        """
        # exp(mean(teacher_lp - student_lp))
        # = exp(-mean(teacher_lp)) / exp(-mean(student_lp))
        # = ppl_student / ppl_teacher
        gap = torch.exp((teacher_logprobs - student_logprobs).mean())
        return gap.item()

    def layer_similarity(
        self, student_hiddens: list[Tensor], teacher_hiddens: list[Tensor]
    ) -> list[float]:
        """Compute cosine similarity between paired student and teacher layers.

        Layers are paired by index up to min(len(student_hiddens), len(teacher_hiddens)).
        Each hidden state is mean-pooled over the sequence dimension before
        computing the batch-averaged cosine similarity.

        Args:
            student_hiddens: List of (B, T, D_s) student hidden state tensors.
            teacher_hiddens: List of (B, T, D_t) teacher hidden state tensors.

        Returns:
            List of cosine similarity values (floats) in [-1, 1], one per
            paired layer.
        """
        similarities = []
        n_pairs = min(len(student_hiddens), len(teacher_hiddens))
        for i in range(n_pairs):
            s_h = student_hiddens[i]  # (B, T, D_s)
            t_h = teacher_hiddens[i]  # (B, T, D_t)

            # Mean pool over sequence: (B, D)
            s_pooled = s_h.mean(dim=1)
            t_pooled = t_h.mean(dim=1)

            # Normalise both to unit vectors
            s_norm = F.normalize(s_pooled, p=2, dim=-1)
            t_norm = F.normalize(t_pooled, p=2, dim=-1)

            # Cosine similarity per sample (element-wise multiply + sum), then batch mean
            cos_sim = (s_norm * t_norm).sum(dim=-1).mean()
            similarities.append(cos_sim.item())

        return similarities
