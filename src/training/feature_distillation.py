"""Feature-level knowledge distillation: hidden state alignment, attention transfer, and FitNets hints."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class FeatureDistillConfig:
    """Configuration for feature-level knowledge distillation."""

    alpha: float = 0.5
    temperature: float = 4.0
    align_layers: list[tuple[int, int]] | None = None
    attention_transfer: bool = False
    hint_layer_student: int = 1
    hint_layer_teacher: int = 2


def hidden_state_mse_loss(
    student_hidden: Tensor,
    teacher_hidden: Tensor,
    projector: nn.Linear | None = None,
) -> Tensor:
    """Compute MSE loss between student and teacher hidden states.

    Args:
        student_hidden: (B, T, D_s) student hidden states.
        teacher_hidden: (B, T, D_t) teacher hidden states.
        projector: Optional linear projection from D_s to D_t.

    Returns:
        Scalar MSE loss.
    """
    if projector is not None:
        student_hidden = projector(student_hidden)
    return F.mse_loss(student_hidden, teacher_hidden)


def attention_transfer_loss(
    student_attn: Tensor,
    teacher_attn: Tensor,
) -> Tensor:
    """Attention Transfer loss (Zagoruyko & Komodakis, 2017).

    Normalizes each attention map by its L2 norm and computes MSE.

    Args:
        student_attn: (B, H, T, T) student attention weights.
        teacher_attn: (B, H, T, T) teacher attention weights.

    Returns:
        Scalar MSE loss between L2-normalized attention maps.
    """
    B = student_attn.shape[0]
    student_flat = student_attn.reshape(B, -1)
    teacher_flat = teacher_attn.reshape(B, -1)

    student_norm = F.normalize(student_flat, p=2, dim=1)
    teacher_norm = F.normalize(teacher_flat, p=2, dim=1)

    return F.mse_loss(student_norm, teacher_norm)


def kd_logit_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    temperature: float,
) -> Tensor:
    """KL divergence of temperature-softened logit distributions.

    Computes KL(teacher || student) with teacher as target, scaled by T^2.

    Args:
        student_logits: (B, T, V) or (N, V) student raw logits.
        teacher_logits: (B, T, V) or (N, V) teacher raw logits.
        temperature: Softening temperature.

    Returns:
        Scalar KL divergence loss scaled by T^2.
    """
    if student_logits.dim() == 3:
        B, S, V = student_logits.shape
        student_logits = student_logits.reshape(-1, V)
        teacher_logits = teacher_logits.reshape(-1, V)

    T = temperature
    student_log_soft = F.log_softmax(student_logits / T, dim=-1)
    teacher_soft = F.softmax(teacher_logits.detach() / T, dim=-1)

    kl = F.kl_div(student_log_soft, teacher_soft, reduction="batchmean")
    return kl * (T ** 2)


class FeatureProjector(nn.Module):
    """Linear projection from student hidden dim to teacher hidden dim.

    Used for FitNets-style hint learning to bridge dimension mismatch.
    """

    def __init__(self, student_dim: int, teacher_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(student_dim, teacher_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Project x from student_dim to teacher_dim.

        Args:
            x: (B, T, student_dim)

        Returns:
            (B, T, teacher_dim)
        """
        return self.proj(x)


def extract_hidden_states(
    model: nn.Module,
    input_ids: Tensor,
    layer_indices: list[int],
) -> dict[int, Tensor]:
    """Extract hidden states from specified transformer layers via forward hooks.

    Args:
        model: AureliusTransformer with a .layers ModuleList.
        input_ids: (B, T) token indices.
        layer_indices: Layer indices to capture.

    Returns:
        Dict mapping layer_idx to (B, T, D) hidden state tensor.
    """
    hidden_states: dict[int, Tensor] = {}
    hooks = []

    def make_hook(idx: int):
        def hook(module, input, output):
            # TransformerBlock returns (hidden, kv_cache) tuple
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            hidden_states[idx] = h
        return hook

    for idx in layer_indices:
        h = model.layers[idx].register_forward_hook(make_hook(idx))
        hooks.append(h)

    try:
        _, _, _ = model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    return hidden_states


class FeatureDistillTrainer:
    """Trains a student model using feature-level knowledge distillation.

    Args:
        student: Student model with .layers ModuleList.
        teacher: Teacher model (will be frozen).
        config: FeatureDistillConfig.
        optimizer: Optimizer for student parameters.
        projectors: Optional dict mapping student layer index to FeatureProjector.
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        config: FeatureDistillConfig,
        optimizer: torch.optim.Optimizer,
        projectors: dict[int, FeatureProjector] | None = None,
    ) -> None:
        self.student = student
        self.teacher = teacher
        self.config = config
        self.optimizer = optimizer
        self.projectors = projectors or {}

        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.train(False)

    def train_step(self, input_ids: Tensor, labels: Tensor) -> dict:
        """One feature distillation training step.

        Args:
            input_ids: (B, S) input token ids.
            labels: (B, S) target token ids for task loss.

        Returns:
            Dict with "loss", "task_loss", "feature_loss" (all floats).
        """
        self.student.train()
        self.optimizer.zero_grad()

        # Determine layer indices to capture
        student_layer_indices: list[int] = []
        teacher_layer_indices: list[int] = []

        if self.config.align_layers:
            for s_idx, t_idx in self.config.align_layers:
                student_layer_indices.append(s_idx)
                teacher_layer_indices.append(t_idx)

        # Forward student with hidden state capture
        if student_layer_indices:
            student_hidden = extract_hidden_states(
                self.student, input_ids, student_layer_indices
            )
        else:
            student_hidden = {}

        # Plain student forward for logits
        _, student_logits, _ = self.student(input_ids)

        # Teacher forward (no grad)
        with torch.no_grad():
            if teacher_layer_indices:
                teacher_hidden = extract_hidden_states(
                    self.teacher, input_ids, teacher_layer_indices
                )
            else:
                teacher_hidden = {}
            _, teacher_logits, _ = self.teacher(input_ids)

        # Task loss: CE on student logits (next-token prediction)
        B, S, V = student_logits.shape
        task_loss = F.cross_entropy(
            student_logits[:, :-1].contiguous().view(-1, V),
            labels[:, 1:].contiguous().view(-1),
        )

        # Feature alignment loss
        feature_loss = torch.tensor(0.0, device=input_ids.device)
        n_pairs = 0

        if self.config.align_layers:
            for s_idx, t_idx in self.config.align_layers:
                s_h = student_hidden.get(s_idx)
                t_h = teacher_hidden.get(t_idx)

                if s_h is not None and t_h is not None:
                    projector = self.projectors.get(s_idx)
                    if isinstance(projector, FeatureProjector):
                        proj_linear = projector.proj
                    else:
                        proj_linear = projector  # None or nn.Linear

                    pair_loss = hidden_state_mse_loss(s_h, t_h.detach(), projector=proj_linear)
                    feature_loss = feature_loss + pair_loss
                    n_pairs += 1

        if n_pairs > 0:
            feature_loss = feature_loss / n_pairs

        # Combined loss
        alpha = self.config.alpha
        total_loss = (1 - alpha) * task_loss + alpha * feature_loss

        total_loss.backward()
        self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "task_loss": task_loss.item(),
            "feature_loss": feature_loss.item(),
        }
