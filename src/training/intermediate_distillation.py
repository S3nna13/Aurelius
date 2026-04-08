"""Intermediate Layer Knowledge Distillation (TinyBERT / DistilBERT style).

Extends basic output-logit KD with richer layer-level transfer:

1. Attention transfer:    L_attn   = MSE(student_attn_maps, teacher_attn_maps)
2. Hidden state transfer: L_hidden = MSE(student_hidden @ W_align, teacher_hidden)
3. Prediction transfer:   L_pred   = T^2 * KL(student_soft || teacher_soft)

References:
    Jiao et al. (2019) "TinyBERT: Distilling BERT for Natural Language Understanding"
    arXiv:1909.10351
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class IntermDistillConfig:
    temperature: float = 4.0        # softmax temperature for output KD
    alpha_pred: float = 0.5         # weight for prediction loss
    alpha_hidden: float = 0.33      # weight for hidden state loss
    alpha_attn: float = 0.17        # weight for attention loss
    layer_mapping: str = "uniform"  # "uniform" | "last_n" | "every_other"


# ---------------------------------------------------------------------------
# Layer mapper
# ---------------------------------------------------------------------------

class LayerMapper:
    """Maps student layers to teacher layers for knowledge transfer.

    Methods:
        "uniform":     student layer i -> teacher layer round(i * n_teacher / n_student)
        "last_n":      student layers -> last n_student teacher layers
        "every_other": student layer i -> teacher layer 2*i

    Args:
        n_student: Number of student layers.
        n_teacher: Number of teacher layers.
        method: Mapping strategy ("uniform", "last_n", or "every_other").
    """

    def __init__(self, n_student: int, n_teacher: int, method: str = "uniform") -> None:
        self.n_student = n_student
        self.n_teacher = n_teacher
        self.method = method

    def get_mapping(self) -> list[tuple[int, int]]:
        """Return list of (student_layer_idx, teacher_layer_idx) pairs."""
        if self.method == "uniform":
            pairs = []
            for i in range(self.n_student):
                t_idx = round(i * self.n_teacher / self.n_student)
                t_idx = min(t_idx, self.n_teacher - 1)
                pairs.append((i, t_idx))
            return pairs

        if self.method == "last_n":
            offset = self.n_teacher - self.n_student
            return [(i, offset + i) for i in range(self.n_student)]

        if self.method == "every_other":
            pairs = []
            for i in range(self.n_student):
                t_idx = min(2 * i, self.n_teacher - 1)
                pairs.append((i, t_idx))
            return pairs

        raise ValueError(
            f"Unknown layer_mapping method: {self.method!r}. "
            "Expected 'uniform', 'last_n', or 'every_other'."
        )


# ---------------------------------------------------------------------------
# Alignment projection
# ---------------------------------------------------------------------------

class AlignmentProjection(nn.Module):
    """Learnable projection to align student hidden dim to teacher hidden dim.

    If d_student == d_teacher: identity (no extra parameters).
    Else: Linear(d_student, d_teacher, bias=False).

    Args:
        d_student: Student model hidden dimension.
        d_teacher: Teacher model hidden dimension.
    """

    def __init__(self, d_student: int, d_teacher: int) -> None:
        super().__init__()
        if d_student != d_teacher:
            self.proj = nn.Linear(d_student, d_teacher, bias=False)
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ---------------------------------------------------------------------------
# Main distillation loss module
# ---------------------------------------------------------------------------

class IntermediateDistillationLoss(nn.Module):
    """Compute multi-level distillation loss.

    Combines prediction (KL), hidden-state (MSE), and attention (MSE) losses.

    Args:
        student_d_model: Student hidden dimension.
        teacher_d_model: Teacher hidden dimension.
        n_student_layers: Number of student transformer layers.
        n_teacher_layers: Number of teacher transformer layers.
        config: Distillation hyper-parameters.
    """

    def __init__(
        self,
        student_d_model: int,
        teacher_d_model: int,
        n_student_layers: int,
        n_teacher_layers: int,
        config: IntermDistillConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or IntermDistillConfig()
        self.student_d_model = student_d_model
        self.teacher_d_model = teacher_d_model

        self.mapper = LayerMapper(
            n_student=n_student_layers,
            n_teacher=n_teacher_layers,
            method=self.config.layer_mapping,
        )

        # One AlignmentProjection per student layer — registered as nn.ModuleList
        # so their parameters participate in optimisation.
        self.hidden_proj = nn.ModuleList(
            [
                AlignmentProjection(student_d_model, teacher_d_model)
                for _ in range(n_student_layers)
            ]
        )

    # ------------------------------------------------------------------
    # Loss components
    # ------------------------------------------------------------------

    def hidden_loss(
        self,
        student_hiddens: list[torch.Tensor],  # list of (B, T, d_student) per student layer
        teacher_hiddens: list[torch.Tensor],  # list of (B, T, d_teacher) per teacher layer
    ) -> torch.Tensor:
        """MSE loss between projected student hiddens and teacher hiddens at mapped layers.

        Normalised by the number of mapped layer pairs (mean over layers).
        """
        mapping = self.mapper.get_mapping()
        device = student_hiddens[0].device
        total = torch.zeros(1, device=device).squeeze()
        for s_idx, t_idx in mapping:
            projected = self.hidden_proj[s_idx](student_hiddens[s_idx])
            total = total + F.mse_loss(projected, teacher_hiddens[t_idx].detach())
        return total / len(mapping)

    def attention_loss(
        self,
        student_attn_maps: list[torch.Tensor],  # list of (B, H_s, T, T) per student layer
        teacher_attn_maps: list[torch.Tensor],  # list of (B, H_t, T, T) per teacher layer
    ) -> torch.Tensor:
        """MSE between student and teacher attention maps at mapped layers.

        If head counts differ: average over heads first -> (B, T, T), then MSE.
        Normalised by the number of mapped layer pairs.
        """
        mapping = self.mapper.get_mapping()
        device = student_attn_maps[0].device
        total = torch.zeros(1, device=device).squeeze()
        for s_idx, t_idx in mapping:
            s_map = student_attn_maps[s_idx]   # (B, H_s, T, T)
            t_map = teacher_attn_maps[t_idx]   # (B, H_t, T, T)

            # Average over the head dimension
            s_avg = s_map.mean(dim=1)   # (B, T, T)
            t_avg = t_map.mean(dim=1)   # (B, T, T)

            total = total + F.mse_loss(s_avg, t_avg.detach())
        return total / len(mapping)

    def prediction_loss(
        self,
        student_logits: torch.Tensor,   # (B, T, V_student)
        teacher_logits: torch.Tensor,   # (B, T, V_teacher)
        temperature: float | None = None,
    ) -> torch.Tensor:
        """KL divergence: KL(student_soft || teacher_soft).

        Soft probs computed as softmax(logits / T).
        If V_student != V_teacher: only the min(V_student, V_teacher) vocab
        positions are used (truncation to shared vocabulary).
        Multiplied by T^2 (standard temperature-scaling correction).
        """
        T = temperature if temperature is not None else self.config.temperature

        V_s = student_logits.shape[-1]
        V_t = teacher_logits.shape[-1]
        V = min(V_s, V_t)

        s_logits = student_logits[..., :V]   # (B, S, V)
        t_logits = teacher_logits[..., :V]   # (B, S, V)

        B, S, _ = s_logits.shape
        s_flat = s_logits.reshape(B * S, V)
        t_flat = t_logits.reshape(B * S, V)

        log_student = F.log_softmax(s_flat / T, dim=-1)
        teacher_probs = F.softmax(t_flat.detach() / T, dim=-1)

        kl = F.kl_div(log_student, teacher_probs, reduction="batchmean")
        return kl * (T ** 2)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_hiddens: list[torch.Tensor] | None = None,
        teacher_hiddens: list[torch.Tensor] | None = None,
        student_attn_maps: list[torch.Tensor] | None = None,
        teacher_attn_maps: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute all available losses and return weighted sum.

        Returns:
            dict with keys:
              - 'total':       weighted sum of active losses
              - 'pred_loss':   KL-divergence prediction loss
              - 'hidden_loss': hidden-state MSE (zero tensor if not provided)
              - 'attn_loss':   attention MSE (zero tensor if not provided)
        """
        device = student_logits.device

        pred = self.prediction_loss(student_logits, teacher_logits)

        hidden: torch.Tensor = torch.zeros(1, device=device).squeeze()
        if student_hiddens is not None and teacher_hiddens is not None:
            hidden = self.hidden_loss(student_hiddens, teacher_hiddens)

        attn: torch.Tensor = torch.zeros(1, device=device).squeeze()
        if student_attn_maps is not None and teacher_attn_maps is not None:
            attn = self.attention_loss(student_attn_maps, teacher_attn_maps)

        total = (
            self.config.alpha_pred * pred
            + self.config.alpha_hidden * hidden
            + self.config.alpha_attn * attn
        )

        return {
            "total": total,
            "pred_loss": pred,
            "hidden_loss": hidden,
            "attn_loss": attn,
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class IntermDistillTrainer:
    """Trainer that runs teacher forward (no_grad) and student forward,
    captures intermediate activations via hooks, and minimises distillation loss.

    Args:
        student: Student model (nn.Module with a .layers ModuleList).
        teacher: Teacher model (frozen).
        optimizer: PyTorch optimizer for the student.
        config: IntermDistillConfig hyper-parameters.
        tokenizer_encode: Optional callable for encoding text (unused internally).
        max_seq_len: Maximum sequence length for input validation.
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: IntermDistillConfig | None = None,
        tokenizer_encode=None,
        max_seq_len: int = 512,
    ) -> None:
        self.student = student
        self.teacher = teacher
        self.optimizer = optimizer
        self.config = config or IntermDistillConfig()
        self.tokenizer_encode = tokenizer_encode
        self.max_seq_len = max_seq_len

        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.train(False)

        student_d_model = self._infer_d_model(student)
        teacher_d_model = self._infer_d_model(teacher)
        n_student_layers = len(student.layers)
        n_teacher_layers = len(teacher.layers)

        self.loss_fn = IntermediateDistillationLoss(
            student_d_model=student_d_model,
            teacher_d_model=teacher_d_model,
            n_student_layers=n_student_layers,
            n_teacher_layers=n_teacher_layers,
            config=self.config,
        )

        # Register alignment projection parameters with the optimiser
        proj_params = list(self.loss_fn.parameters())
        if proj_params and hasattr(optimizer, "add_param_group"):
            optimizer.add_param_group({"params": proj_params})

    @staticmethod
    def _infer_d_model(model: nn.Module) -> int:
        """Infer d_model from model.config or the embedding weight shape."""
        if hasattr(model, "config") and hasattr(model.config, "d_model"):
            return model.config.d_model
        if hasattr(model, "embed"):
            return model.embed.weight.shape[1]
        raise AttributeError("Cannot infer d_model from model.")

    # ------------------------------------------------------------------
    # Hook helpers
    # ------------------------------------------------------------------

    def _capture_hiddens(self, model: nn.Module) -> tuple[list, dict]:
        """Register forward hooks on model.layers to capture hidden states.

        The hook captures the *output* hidden state of each TransformerBlock.
        TransformerBlock returns (hidden_state, kv_cache), so index 0 is used.

        Returns:
            (hooks_list, captures) where captures is a dict with keys:
                'hiddens'   : list[Tensor | None] — one entry per layer
                'attn_maps' : list[None] — placeholder (attn weights not exposed)

        Call hook.remove() on each element of hooks_list after the forward pass.
        """
        n_layers = len(model.layers)
        captures: dict[str, list] = {
            "hiddens": [None] * n_layers,
            "attn_maps": [None] * n_layers,
        }
        hooks: list = []

        for i in range(n_layers):
            def make_hook(idx: int):
                def hook(module, inputs, outputs):
                    hidden = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                    captures["hiddens"][idx] = hidden
                return hook

            h = model.layers[i].register_forward_hook(make_hook(i))
            hooks.append(h)

        return hooks, captures

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self, input_ids: torch.Tensor) -> dict[str, float]:
        """One distillation training step.

        Steps:
            1. Teacher forward (no_grad) — capture intermediate hiddens.
            2. Student forward — capture intermediate hiddens.
            3. Compute IntermediateDistillationLoss.
            4. Backward + optimizer step.

        Args:
            input_ids: (B, seq_len) token ids.

        Returns:
            dict with keys: 'loss', 'pred_loss', 'hidden_loss', 'attn_loss'.
        """
        self.student.train()

        # --- Teacher forward (no gradient) ---
        teacher_hooks, teacher_caps = self._capture_hiddens(self.teacher)
        with torch.no_grad():
            _, teacher_logits, _ = self.teacher(input_ids)
        for h in teacher_hooks:
            h.remove()

        teacher_hiddens = [
            t.detach() for t in teacher_caps["hiddens"] if t is not None
        ]

        # --- Student forward ---
        student_hooks, student_caps = self._capture_hiddens(self.student)
        self.optimizer.zero_grad()
        _, student_logits, _ = self.student(input_ids)
        for h in student_hooks:
            h.remove()

        student_hiddens = [t for t in student_caps["hiddens"] if t is not None]

        # --- Compute loss ---
        loss_dict = self.loss_fn(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            student_hiddens=student_hiddens if student_hiddens else None,
            teacher_hiddens=teacher_hiddens if teacher_hiddens else None,
        )

        total_loss: torch.Tensor = loss_dict["total"]
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
        self.optimizer.step()

        def _item(v):
            return v.item() if isinstance(v, torch.Tensor) else float(v)

        return {
            "loss": _item(loss_dict["total"]),
            "pred_loss": _item(loss_dict["pred_loss"]),
            "hidden_loss": _item(loss_dict["hidden_loss"]),
            "attn_loss": _item(loss_dict["attn_loss"]),
        }
