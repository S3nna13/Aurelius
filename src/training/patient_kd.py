"""Patient Knowledge Distillation (Sun et al., 2019) and soft-target distillation variants."""

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
class PKDConfig:
    """Configuration for Patient Knowledge Distillation."""

    n_student_layers: int = 2
    n_teacher_layers: int = 4
    patience_strategy: str = "last"  # "last" | "skip" | "every_other"
    beta: float = 500.0  # weight for hidden state distillation loss
    temperature: float = 4.0  # for soft label distillation
    normalize_hidden: bool = True  # L2-normalize hidden states before MSE


# ---------------------------------------------------------------------------
# Layer mapping
# ---------------------------------------------------------------------------


def get_patient_layers(n_teacher: int, n_student: int, strategy: str) -> list[int]:
    """Return which teacher layer indices map to each student layer.

    Args:
        n_teacher: Number of teacher layers.
        n_student: Number of student layers.
        strategy: One of "last", "skip", "every_other".

    Returns:
        List of length n_student with teacher layer indices.
    """
    if strategy == "last":
        return [n_teacher - n_student + i for i in range(n_student)]
    elif strategy == "skip":
        return [int(i * n_teacher / n_student) for i in range(n_student)]
    elif strategy == "every_other":
        indices = [2 * i for i in range(n_student)]
        return [min(idx, n_teacher - 1) for idx in indices]
    else:
        raise ValueError(
            f"Unknown patience_strategy: {strategy!r}. Choose 'last', 'skip', or 'every_other'."
        )


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def pkd_hidden_loss(
    student_hiddens: list[Tensor],
    teacher_hiddens: list[Tensor],
    layer_mapping: list[int],
    normalize: bool = True,
) -> Tensor:
    """MSE loss between corresponding student and teacher hidden states.

    Args:
        student_hiddens: List of student hidden states, each shape (B, T, D_s).
        teacher_hiddens: List of teacher hidden states, each shape (B, T, D_t).
        layer_mapping: teacher layer index for each student layer index.
        normalize: If True, L2-normalize along the D dimension before MSE.

    Returns:
        Scalar mean MSE loss.
    """
    device = student_hiddens[0].device
    dtype = student_hiddens[0].dtype
    total_loss = torch.zeros(1, device=device, dtype=dtype)
    n = len(student_hiddens)

    for i, t_idx in enumerate(layer_mapping):
        s_h = student_hiddens[i]
        t_h = teacher_hiddens[t_idx]

        B, T, D_s = s_h.shape
        s_flat = s_h.reshape(B * T, D_s)
        t_flat = t_h.reshape(B * T, -1)

        if normalize:
            s_flat = F.normalize(s_flat, p=2, dim=-1)
            t_flat = F.normalize(t_flat, p=2, dim=-1)

        total_loss = total_loss + F.mse_loss(s_flat, t_flat)

    return total_loss / n


def attention_transfer_loss(
    student_attns: list[Tensor],
    teacher_attns: list[Tensor],
    layer_mapping: list[int],
) -> Tensor:
    """Attention matrix transfer loss (Zagoruyko & Komodakis style).

    Args:
        student_attns: List of student attention maps, each (B, n_heads_s, T, T).
        teacher_attns: List of teacher attention maps, each (B, n_heads_t, T, T).
        layer_mapping: teacher layer index for each student layer.

    Returns:
        Scalar MSE loss between L2-normalized attention maps.
    """
    device = student_attns[0].device
    dtype = student_attns[0].dtype
    total_loss = torch.zeros(1, device=device, dtype=dtype)
    n = len(student_attns)

    for i, t_idx in enumerate(layer_mapping):
        s_a = student_attns[i]
        t_a = teacher_attns[t_idx]

        s_map = s_a.mean(dim=1).reshape(s_a.shape[0], -1)
        t_map = t_a.mean(dim=1).reshape(t_a.shape[0], -1)

        s_map = F.normalize(s_map, p=2, dim=-1)
        t_map = F.normalize(t_map, p=2, dim=-1)

        total_loss = total_loss + F.mse_loss(s_map, t_map)

    return total_loss / n


# ---------------------------------------------------------------------------
# PKD Loss Module
# ---------------------------------------------------------------------------


class PKDLoss(nn.Module):
    """Combined Patient KD loss: soft KL + beta * hidden MSE.

    Args:
        config: PKDConfig.
        student_d_model: Student hidden dimension.
        teacher_d_model: Teacher hidden dimension.
    """

    def __init__(self, config: PKDConfig, student_d_model: int, teacher_d_model: int) -> None:
        super().__init__()
        self.config = config
        self.layer_mapping: list[int] = get_patient_layers(
            config.n_teacher_layers,
            config.n_student_layers,
            config.patience_strategy,
        )

        if student_d_model != teacher_d_model:
            self.projections: nn.ModuleList | None = nn.ModuleList(
                [
                    nn.Linear(student_d_model, teacher_d_model, bias=False)
                    for _ in range(config.n_student_layers)
                ]
            )
        else:
            self.projections = None

    def forward(
        self,
        student_hiddens: list[Tensor],
        teacher_hiddens: list[Tensor],
        student_logits: Tensor,
        teacher_logits: Tensor,
        labels: Tensor,
    ) -> tuple[Tensor, dict]:
        """Compute total PKD loss.

        Args:
            student_hiddens: Per-student-layer hidden states (B, T, D_s).
            teacher_hiddens: Per-teacher-layer hidden states (B, T, D_t).
            student_logits: (B, T, V).
            teacher_logits: (B, T, V).
            labels: (B, T).

        Returns:
            (total_loss, metrics_dict) where metrics_dict has soft_kl, hidden_loss, total.
        """
        T = self.config.temperature
        beta = self.config.beta

        if self.projections is not None:
            projected = [proj(h) for proj, h in zip(self.projections, student_hiddens)]
        else:
            projected = student_hiddens

        soft_kl = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction="batchmean",
        ) * (T**2)

        hidden_loss = pkd_hidden_loss(
            projected,
            teacher_hiddens,
            self.layer_mapping,
            normalize=self.config.normalize_hidden,
        )

        total = soft_kl + beta * hidden_loss

        return total, {
            "soft_kl": soft_kl.item(),
            "hidden_loss": hidden_loss.item(),
            "total": total.item(),
        }


# ---------------------------------------------------------------------------
# Patient KD Trainer
# ---------------------------------------------------------------------------


class PatientKDTrainer:
    """Trains a student model using Patient Knowledge Distillation.

    Args:
        student: Student model (nn.Module). Forward returns (loss, logits, pkv).
        teacher: Teacher model (nn.Module). Frozen during training.
        optimizer: Optimizer for student parameters.
        config: PKDConfig.
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: PKDConfig,
    ) -> None:
        self.student = student
        self.teacher = teacher
        self.optimizer = optimizer
        self.config = config

        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

        student_d = self._infer_d_model(student)
        teacher_d = self._infer_d_model(teacher)
        self.loss_fn = PKDLoss(config, student_d, teacher_d)

    @staticmethod
    def _infer_d_model(model: nn.Module) -> int:
        if hasattr(model, "config") and hasattr(model.config, "d_model"):
            return model.config.d_model
        for m in model.modules():
            if isinstance(m, nn.Embedding):
                return m.embedding_dim
        raise RuntimeError("Cannot infer d_model from model")

    def extract_hiddens(self, model: nn.Module, input_ids: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Run forward with hooks to collect per-layer hidden states.

        Args:
            model: AureliusTransformer with a ``layers`` ModuleList.
            input_ids: (B, T).

        Returns:
            (logits, [hidden_0, ...]) each hidden_i is (B, T, D).
        """
        hiddens: list[Tensor] = []
        hooks = []

        def make_hook(storage: list):
            def hook(module, inp, out):
                # TransformerBlock returns (hidden_tensor, kv_tuple)
                h = out[0] if isinstance(out, (tuple, list)) else out
                storage.append(h.detach())

            return hook

        for layer in model.layers:
            hooks.append(layer.register_forward_hook(make_hook(hiddens)))

        try:
            _, logits, _ = model(input_ids)
        finally:
            for h in hooks:
                h.remove()

        return logits, hiddens

    def train_step(self, input_ids: Tensor, labels: Tensor) -> dict[str, float]:
        """One Patient KD training step.

        Args:
            input_ids: (B, T).
            labels: (B, T).

        Returns:
            {"loss": float, "soft_kl": float, "hidden_loss": float}
        """
        self.student.train()
        self.optimizer.zero_grad()

        with torch.no_grad():
            teacher_logits, teacher_hiddens = self.extract_hiddens(self.teacher, input_ids)

        student_logits, student_hiddens = self.extract_hiddens(self.student, input_ids)

        total_loss, metrics = self.loss_fn(
            student_hiddens,
            teacher_hiddens,
            student_logits,
            teacher_logits,
            labels,
        )

        total_loss.backward()
        self.optimizer.step()

        return {
            "loss": metrics["total"],
            "soft_kl": metrics["soft_kl"],
            "hidden_loss": metrics["hidden_loss"],
        }
