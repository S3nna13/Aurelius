"""Latent Representation Alignment for knowledge distillation at the hidden-state level.

Trains a model's internal representations to match those of a reference/teacher model
(or a target distribution). Operates on hidden states directly -- unlike output-KD
(logit matching), this aligns the student's "concept space" to the teacher's.

Supports three alignment modes:
- 'linear': Learned linear projection from student_dim -> teacher_dim, then cosine loss
- 'cosine': Direct cosine distance (1 - cosine_similarity), same-dim only
- 'mse':    MSE(student, teacher), requires matching dimensions

Also provides:
- LayerwiseAlignmentTrainer for hook-based multi-layer alignment training
- centered_kernel_alignment (CKA) metric for comparing representation matrices
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
class ReprAlignmentConfig:
    """Configuration for latent representation alignment."""

    align_type: str = "linear"
    normalize: bool = True
    align_weight: float = 0.1
    layer_pairs: list = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.layer_pairs is None:
            self.layer_pairs = []


# ---------------------------------------------------------------------------
# RepresentationAligner
# ---------------------------------------------------------------------------


class RepresentationAligner(nn.Module):
    """Align student hidden states to teacher hidden states.

    Args:
        student_dim: Hidden dimension of the student model.
        teacher_dim: Hidden dimension of the teacher model.
        align_type: Alignment strategy -- 'linear', 'cosine', or 'mse'.
        normalize: Whether to L2-normalize representations before loss computation.
    """

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        align_type: str = "linear",
        normalize: bool = True,
    ) -> None:
        super().__init__()

        if align_type not in ("linear", "cosine", "mse"):
            raise ValueError(f"align_type must be 'linear', 'cosine', or 'mse'; got {align_type!r}")

        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        self.align_type = align_type
        self.normalize = normalize

        # Learned projection only for 'linear' mode
        self.projection: nn.Linear | None = None
        if align_type == "linear":
            self.projection = nn.Linear(student_dim, teacher_dim, bias=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        student_repr: torch.Tensor,
        teacher_repr: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute alignment loss and diagnostic metrics.

        Args:
            student_repr: (batch, seq, student_dim) student hidden states.
            teacher_repr: (batch, seq, teacher_dim) teacher hidden states (detached internally).

        Returns:
            (alignment_loss, metrics_dict) where metrics_dict contains:
                'cosine_sim':       mean cosine similarity after optional projection.
                'mse':              mean squared error after optional projection.
                'projection_norm':  Frobenius norm of the projection weight (0 if no projection).
        """
        teacher_repr = teacher_repr.detach()

        # Project student into teacher space (if linear mode)
        student_proj = self._project(student_repr)

        # Optionally L2-normalize for metric computation
        if self.normalize:
            s_n = F.normalize(student_proj, p=2, dim=-1)
            t_n = F.normalize(teacher_repr, p=2, dim=-1)
        else:
            s_n = student_proj
            t_n = teacher_repr

        # Flatten for metrics
        s_flat = s_n.reshape(-1, s_n.shape[-1])
        t_flat = t_n.reshape(-1, t_n.shape[-1])

        if s_flat.shape[-1] == t_flat.shape[-1]:
            cosine_sim = F.cosine_similarity(s_flat, t_flat, dim=-1).mean().item()
            mse_val = F.mse_loss(s_flat, t_flat).item()
        else:
            cosine_sim = 0.0
            mse_val = float("inf")

        proj_norm = self.projection.weight.norm().item() if self.projection is not None else 0.0

        metrics = {
            "cosine_sim": cosine_sim,
            "mse": mse_val,
            "projection_norm": proj_norm,
        }

        loss = self.align_loss(student_repr, teacher_repr)
        return loss, metrics

    def align_loss(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
    ) -> torch.Tensor:
        """Compute alignment loss.

        For 'linear':  project student -> teacher_dim, then cosine distance.
        For 'cosine':  1 - cosine_similarity (requires matching dims).
        For 'mse':     MSE(student, teacher) (requires matching dims).

        Args:
            student: (..., student_dim) student hidden states.
            teacher: (..., teacher_dim) teacher hidden states.

        Returns:
            Scalar alignment loss.
        """
        teacher = teacher.detach()
        student_proj = self._project(student)

        if self.align_type in ("linear", "cosine"):
            if self.normalize:
                s = F.normalize(student_proj, p=2, dim=-1)
                t = F.normalize(teacher, p=2, dim=-1)
            else:
                s = student_proj
                t = teacher
            cos_sim = F.cosine_similarity(
                s.reshape(-1, s.shape[-1]),
                t.reshape(-1, t.shape[-1]),
                dim=-1,
            )
            return (1.0 - cos_sim).mean()

        else:  # 'mse'
            return F.mse_loss(student_proj, teacher)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _project(self, student: torch.Tensor) -> torch.Tensor:
        """Apply linear projection if available, else return unchanged."""
        if self.projection is not None:
            return self.projection(student)
        return student


# ---------------------------------------------------------------------------
# LayerwiseAlignmentTrainer
# ---------------------------------------------------------------------------


class LayerwiseAlignmentTrainer:
    """Train a student model with layerwise representation alignment to a teacher.

    Uses forward hooks to extract hidden states from specified layers of both the
    student and teacher model, then computes alignment losses at each layer pair.

    Args:
        student_model: The model being trained (nn.Module with .layers ModuleList).
        teacher_model: The reference model (frozen, placed in eval mode).
        layer_pairs: List of (student_layer_idx, teacher_layer_idx) tuples.
        align_weight: Scalar weight applied to the total alignment loss.
    """

    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        layer_pairs: list[tuple[int, int]],
        align_weight: float = 0.1,
    ) -> None:
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.layer_pairs = layer_pairs
        self.align_weight = align_weight

        # Freeze teacher
        for p in self.teacher_model.parameters():
            p.requires_grad_(False)
        self.teacher_model.eval()

        # Determine model hidden dims
        s_dim = _get_model_dim(student_model)
        t_dim = _get_model_dim(teacher_model)

        # One aligner per layer pair
        self.aligners = nn.ModuleList(
            [
                RepresentationAligner(s_dim, t_dim, align_type="linear", normalize=True)
                for _ in layer_pairs
            ]
        )

        # Optimizer covers student + aligner projection weights
        self.optimizer = torch.optim.Adam(
            list(self.student_model.parameters()) + list(self.aligners.parameters()),
            lr=1e-4,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_layer_representations(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        layer_indices: list[int],
    ) -> list[torch.Tensor]:
        """Extract hidden states at specified layer indices via forward hooks.

        Compatible with AureliusTransformer whose TransformerBlock.forward returns
        (hidden_state, kv_cache).

        Args:
            model: Model with a .layers nn.ModuleList attribute.
            input_ids: (batch, seq) token ids.
            layer_indices: Which layer indices to capture (into model.layers).

        Returns:
            List of (batch, seq, d_model) tensors, one per requested index (in order).
        """
        captured: list[torch.Tensor | None] = [None] * len(layer_indices)
        handles = []

        def make_hook(pos: int):
            def hook(module: nn.Module, inp: tuple, output) -> None:
                if isinstance(output, (tuple, list)):
                    # AureliusTransformer: output = (hidden_state, kv_cache)
                    h = output[0]
                else:
                    h = output
                captured[pos] = h

            return hook

        for pos, idx in enumerate(layer_indices):
            h = model.layers[idx].register_forward_hook(make_hook(pos))
            handles.append(h)

        try:
            model(input_ids)
        finally:
            for h in handles:
                h.remove()

        return [t for t in captured if t is not None]

    def compute_alignment_loss(
        self,
        student_ids: torch.Tensor,
        teacher_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute the total alignment loss across all layer pairs.

        Args:
            student_ids: (batch, seq) input tokens for the student model.
            teacher_ids: (batch, seq) input tokens for the teacher model.

        Returns:
            (total_alignment_loss, metrics_dict) where metrics_dict contains:
                'layer_losses':     list of per-pair scalar loss values.
                'mean_cosine_sim':  mean cosine similarity across all pairs.
        """
        s_layer_indices = [p[0] for p in self.layer_pairs]
        t_layer_indices = [p[1] for p in self.layer_pairs]

        # Teacher representations -- no grad
        with torch.no_grad():
            teacher_reprs = self.extract_layer_representations(
                self.teacher_model, teacher_ids, t_layer_indices
            )

        # Student representations -- with grad
        student_reprs = self.extract_layer_representations(
            self.student_model, student_ids, s_layer_indices
        )

        device = student_ids.device
        torch.zeros(1, device=device, requires_grad=False)
        # We accumulate with a grad-tracked path
        acc_loss: torch.Tensor | None = None
        layer_losses: list[float] = []
        cosine_sims: list[float] = []

        for i, (s_repr, t_repr) in enumerate(zip(student_reprs, teacher_reprs)):
            loss_i, metrics_i = self.aligners[i](s_repr, t_repr)
            acc_loss = loss_i if acc_loss is None else acc_loss + loss_i
            layer_losses.append(loss_i.item())
            cosine_sims.append(metrics_i["cosine_sim"])

        if acc_loss is None:
            # No layer pairs -- return zero with grad
            acc_loss = torch.zeros(1, device=device).squeeze()

        n = max(len(self.layer_pairs), 1)
        acc_loss = acc_loss / n

        mean_cosine_sim = sum(cosine_sims) / len(cosine_sims) if cosine_sims else 0.0

        return acc_loss, {
            "layer_losses": layer_losses,
            "mean_cosine_sim": mean_cosine_sim,
        }

    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict:
        """Full training step combining task loss and alignment loss.

        Args:
            input_ids: (batch, seq) input token ids.
            labels: (batch, seq) target token ids for next-token prediction CE loss.

        Returns:
            Dict with keys: 'loss', 'task_loss', 'alignment_loss', 'cosine_sim'.
        """
        self.student_model.train()
        self.optimizer.zero_grad()

        # Task loss: forward with labels for next-token CE
        loss_out, logits, _ = self.student_model(input_ids, labels=labels)
        if loss_out is None:
            B, S, V = logits.shape
            loss_out = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, V),
                labels[:, 1:].contiguous().view(-1),
            )
        task_loss: torch.Tensor = loss_out

        # Alignment loss
        align_loss, align_metrics = self.compute_alignment_loss(input_ids, input_ids)

        total_loss = task_loss + self.align_weight * align_loss
        total_loss.backward()
        self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "task_loss": task_loss.item(),
            "alignment_loss": align_loss.item(),
            "cosine_sim": align_metrics["mean_cosine_sim"],
        }


# ---------------------------------------------------------------------------
# Centered Kernel Alignment
# ---------------------------------------------------------------------------


def centered_kernel_alignment(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Centered Kernel Alignment (CKA) similarity between two representation matrices.

    CKA is invariant to orthogonal transformations and isotropic scaling.
    Uses linear (dot-product) kernels for efficiency.

    Reference: Kornblith et al. "Similarity of Neural Network Representations Revisited"
    ICML 2019. https://arxiv.org/abs/1905.00414

    Args:
        X: (n, d1) representation matrix.
        Y: (n, d2) representation matrix.

    Returns:
        CKA similarity in [0, 1]. 1.0 for identical representations (up to
        orthogonal transform + isotropic scaling), 0.0 for maximally dissimilar.
    """
    # Center columns (subtract column means)
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # Linear kernel matrices: (n, n)
    K = X @ X.T
    L = Y @ Y.T

    hsic_xy = _hsic(K, L)
    hsic_xx = _hsic(K, K)
    hsic_yy = _hsic(L, L)

    denom = (hsic_xx * hsic_yy).sqrt()
    if denom.item() < 1e-10:
        return 0.0

    cka = (hsic_xy / denom).item()
    return float(max(0.0, min(1.0, cka)))


def _hsic(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Unbiased HSIC estimator via Frobenius inner product of centered kernel matrices.

    Args:
        K: (n, n) centered kernel matrix.
        L: (n, n) centered kernel matrix.

    Returns:
        Scalar HSIC estimate.
    """
    n = K.shape[0]
    return (K * L).sum() / ((n - 1) ** 2)


# ---------------------------------------------------------------------------
# Internal utility
# ---------------------------------------------------------------------------


def _get_model_dim(model: nn.Module) -> int:
    """Infer the hidden dimension from a model.

    Tries model.config.d_model first (AureliusTransformer), then falls back to
    scanning embedding or linear layers.
    """
    if hasattr(model, "config") and hasattr(model.config, "d_model"):
        return int(model.config.d_model)
    if hasattr(model, "embed") and hasattr(model.embed, "embedding_dim"):
        return int(model.embed.embedding_dim)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            return int(m.out_features)
    raise ValueError("Cannot infer hidden dimension from model.")
