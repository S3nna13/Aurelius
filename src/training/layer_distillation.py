"""Layer-wise knowledge distillation: match intermediate hidden states, attention maps, and feature distributions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.hooks import RemovableHandle


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LayerDistillConfig:
    """Configuration for layer-wise knowledge distillation."""
    teacher_layers: list[int] = field(default_factory=list)   # empty = all
    student_layers: list[int] = field(default_factory=list)   # empty = all
    hidden_loss_weight: float = 1.0
    attention_loss_weight: float = 0.5
    feature_loss_weight: float = 0.5
    temperature: float = 4.0
    loss_type: str = "mse"       # "mse" | "cosine" | "kl"
    use_projector: bool = True   # linear projector to match dims


# ---------------------------------------------------------------------------
# Linear Projector
# ---------------------------------------------------------------------------

class LinearProjector(nn.Module):
    """Projects student hidden dim to teacher hidden dim."""

    def __init__(self, student_dim: int, teacher_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(student_dim, teacher_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def hidden_state_loss(
    student_hidden: Tensor,
    teacher_hidden: Tensor,
    loss_type: str = "mse",
) -> Tensor:
    """Compute loss between hidden state tensors of shape (B, T, D).

    Args:
        student_hidden: (B, T, D) student hidden states.
        teacher_hidden: (B, T, D) teacher hidden states.
        loss_type: One of "mse", "cosine", or "kl".

    Returns:
        Scalar loss tensor.
    """
    t_h = teacher_hidden.detach()

    if loss_type == "mse":
        return F.mse_loss(student_hidden, t_h)

    elif loss_type == "cosine":
        return (1.0 - F.cosine_similarity(student_hidden, t_h, dim=-1).mean())

    elif loss_type == "kl":
        s_log = F.log_softmax(student_hidden, dim=-1)
        t_soft = F.softmax(t_h, dim=-1)
        return F.kl_div(s_log, t_soft, reduction="batchmean")

    else:
        raise ValueError(f"Unknown loss_type: {loss_type!r}. Choose 'mse', 'cosine', or 'kl'.")


def attention_map_loss(student_attn: Tensor, teacher_attn: Tensor) -> Tensor:
    """KL divergence between attention distributions of shape (B, H, T, T).

    Args:
        student_attn: (B, H, T, T) student attention weights.
        teacher_attn: (B, H, T, T) teacher attention weights.

    Returns:
        Scalar loss tensor.
    """
    # Normalize rows with softmax
    s_soft = F.log_softmax(student_attn, dim=-1)
    t_soft = F.softmax(teacher_attn.detach(), dim=-1)

    # KL divergence: batchmean over (B * H * T) "batches" of T-dim distributions
    B, H, T, _ = student_attn.shape
    s_flat = s_soft.reshape(B * H * T, -1)
    t_flat = t_soft.reshape(B * H * T, -1)

    return F.kl_div(s_flat, t_flat, reduction="batchmean")


def feature_distribution_loss(
    student_feat: Tensor,
    teacher_feat: Tensor,
    temperature: float = 4.0,
) -> Tensor:
    """Temperature-scaled KL divergence between feature distributions.

    Accepts tensors of shape (B, T, D) or (B, D).

    Args:
        student_feat: Student features.
        teacher_feat: Teacher features.
        temperature: Softening temperature.

    Returns:
        Scalar loss tensor.
    """
    t_f = teacher_feat.detach()

    if student_feat.dim() == 3:
        B, T, D = student_feat.shape
        s_flat = student_feat.reshape(B * T, D)
        t_flat = t_f.reshape(B * T, D)
    else:
        s_flat = student_feat
        t_flat = t_f

    s_log = F.log_softmax(s_flat / temperature, dim=-1)
    t_soft = F.softmax(t_flat / temperature, dim=-1)

    return F.kl_div(s_log, t_soft, reduction="batchmean") * (temperature ** 2)


# ---------------------------------------------------------------------------
# Activation Hook
# ---------------------------------------------------------------------------

class ActivationHook:
    """Forward hook to capture layer activations."""

    def __init__(self) -> None:
        self.activations: list[Tensor] = []

    def hook(self, module: nn.Module, input: tuple, output) -> None:
        """Append output (or output[0] if tuple) to activations."""
        if isinstance(output, (tuple, list)):
            self.activations.append(output[0])
        else:
            self.activations.append(output)

    def clear(self) -> None:
        """Empty the activations list."""
        self.activations = []

    def register(self, layer: nn.Module) -> RemovableHandle:
        """Register the hook on a layer and return the handle."""
        return layer.register_forward_hook(self.hook)


# ---------------------------------------------------------------------------
# Layer Distillation Loss
# ---------------------------------------------------------------------------

class LayerDistillationLoss(nn.Module):
    """Computes full layer distillation loss over matched hidden states.

    Args:
        config: LayerDistillConfig.
        student_dim: Student model hidden dimension.
        teacher_dim: Teacher model hidden dimension.
    """

    def __init__(
        self,
        config: LayerDistillConfig,
        student_dim: int,
        teacher_dim: int,
    ) -> None:
        super().__init__()
        self.config = config

        if config.use_projector:
            self.projector: Optional[LinearProjector] = LinearProjector(student_dim, teacher_dim)
        else:
            self.projector = None

    def forward(
        self,
        student_hiddens: list[Tensor],
        teacher_hiddens: list[Tensor],
    ) -> tuple[Tensor, dict]:
        """Compute weighted hidden state distillation loss.

        Args:
            student_hiddens: Per-layer student hidden states, each (B, T, D_s).
            teacher_hiddens: Per-layer teacher hidden states, each (B, T, D_t).

        Returns:
            (total_loss, {"hidden_loss": float, "n_pairs": int})
        """
        pairs = list(zip(student_hiddens, teacher_hiddens))
        n_pairs = len(pairs)

        device = student_hiddens[0].device
        dtype = student_hiddens[0].dtype
        total_hidden = torch.zeros(1, device=device, dtype=dtype)

        for s_h, t_h in pairs:
            if self.projector is not None:
                s_h = self.projector(s_h)
            total_hidden = total_hidden + hidden_state_loss(s_h, t_h, self.config.loss_type)

        if n_pairs > 0:
            total_hidden = total_hidden / n_pairs

        total_loss = self.config.hidden_loss_weight * total_hidden

        return total_loss.squeeze(), {
            "hidden_loss": total_hidden.item(),
            "n_pairs": n_pairs,
        }


# ---------------------------------------------------------------------------
# Layer Distillation Trainer
# ---------------------------------------------------------------------------

class LayerDistillTrainer:
    """Trains a student model to match the teacher at layer level.

    Args:
        student: Student model (nn.Module). Forward returns (loss, logits, past_kv).
        teacher: Teacher model (nn.Module). Frozen during training.
        config: LayerDistillConfig.
        optimizer: Optimizer for student parameters.
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        config: LayerDistillConfig,
        optimizer: "torch.optim.Optimizer",
    ) -> None:
        self.student = student
        self.teacher = teacher
        self.config = config
        self.optimizer = optimizer

        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

        student_dim = self._infer_dim(student)
        teacher_dim = self._infer_dim(teacher)
        self.distill_loss = LayerDistillationLoss(config, student_dim, teacher_dim)

    @staticmethod
    def _infer_dim(model: nn.Module) -> int:
        if hasattr(model, "config") and hasattr(model.config, "d_model"):
            return model.config.d_model
        return 64

    def _get_layers(self, model: nn.Module) -> list[nn.Module]:
        """Return the list of transformer layers to hook."""
        if hasattr(model, "layers"):
            all_layers = list(model.layers)
        elif hasattr(model, "blocks"):
            all_layers = list(model.blocks)
        else:
            return []

        indices = (
            self.config.student_layers
            if model is self.student
            else self.config.teacher_layers
        )
        if not indices:
            return all_layers
        return [all_layers[i] for i in indices if i < len(all_layers)]

    def _forward_with_hooks(
        self, model: nn.Module, input_ids: Tensor
    ) -> tuple[Optional[Tensor], Tensor, list[Tensor]]:
        """Run forward pass while capturing hidden states via hooks.

        Returns:
            (loss, logits, hiddens)
        """
        hook_obj = ActivationHook()
        handles = []
        for layer in self._get_layers(model):
            handles.append(hook_obj.register(layer))

        try:
            loss, logits, _ = model(input_ids, labels=input_ids)
        finally:
            for h in handles:
                h.remove()

        return loss, logits, list(hook_obj.activations)

    def train_step(self, input_ids: Tensor) -> dict[str, float]:
        """Perform one layer distillation training step.

        Args:
            input_ids: (B, T) token ids.

        Returns:
            {"loss": float, "distill_loss": float, "ce_loss": float}
        """
        self.student.train()
        self.optimizer.zero_grad()

        # Teacher forward (no grad)
        with torch.no_grad():
            _, _, teacher_hiddens = self._forward_with_hooks(self.teacher, input_ids)

        # Student forward
        ce_loss, _logits, student_hiddens = self._forward_with_hooks(self.student, input_ids)

        # Distillation loss
        distill_loss_val, _info = self.distill_loss(student_hiddens, teacher_hiddens)

        # Total loss
        ce = ce_loss if ce_loss is not None else torch.zeros(1, device=input_ids.device)
        total = ce + distill_loss_val

        total.backward()
        self.optimizer.step()

        return {
            "loss": total.item(),
            "distill_loss": distill_loss_val.item(),
            "ce_loss": ce.item(),
        }
