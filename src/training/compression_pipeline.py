"""End-to-end model compression pipeline: prune + distill + quantize in sequence."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CompressionConfig:
    """Configuration for the end-to-end compression pipeline."""
    prune_fraction: float = 0.3         # fraction of weights to prune
    distill_temperature: float = 4.0   # KD softening temperature
    distill_alpha: float = 0.7         # blend: alpha * kd_loss + (1-alpha) * task_loss
    quantize_bits: int = 8             # fake quantization bit-width
    stages: list[str] = field(default_factory=lambda: ["prune", "distill", "quantize"])
    stage_epochs: dict[str, int] = field(
        default_factory=lambda: {"prune": 1, "distill": 2, "quantize": 1}
    )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def compute_model_sparsity(model: nn.Module) -> dict[str, float]:
    """Count zero parameters per layer and overall.

    Only examines weight tensors (parameters named with 'weight' in name).

    Args:
        model: Any nn.Module.

    Returns:
        {"overall_sparsity": float, "n_zero_params": int, "n_total_params": int}
    """
    n_zero = 0
    n_total = 0

    for name, param in model.named_parameters():
        if "weight" in name:
            n_total += param.numel()
            n_zero += int((param.data == 0).sum().item())

    overall_sparsity = n_zero / n_total if n_total > 0 else 0.0

    return {
        "overall_sparsity": overall_sparsity,
        "n_zero_params": n_zero,
        "n_total_params": n_total,
    }


def magnitude_prune(model: nn.Module, fraction: float) -> nn.Module:
    """Zero out the bottom fraction of weights by absolute magnitude globally.

    Only prunes weight matrices (parameters whose name contains 'weight').
    Biases and norm parameters are left untouched.

    Modification is done in-place.

    Args:
        model: Any nn.Module.
        fraction: Fraction in [0, 1) of weights to zero out.

    Returns:
        The model (modified in-place).
    """
    if fraction <= 0.0:
        return model

    # Collect all weight tensors
    weight_params = [
        (name, param)
        for name, param in model.named_parameters()
        if "weight" in name
    ]

    if not weight_params:
        return model

    # Gather all weights into a flat tensor to determine the global threshold
    all_weights = torch.cat([p.data.abs().flatten() for _, p in weight_params])
    n_prune = max(1, int(fraction * all_weights.numel()))
    threshold, _ = torch.kthvalue(all_weights, n_prune)

    # Zero out weights below threshold
    with torch.no_grad():
        for _, param in weight_params:
            mask = param.data.abs() <= threshold
            param.data[mask] = 0.0

    return model


def distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    labels: Tensor,
    temperature: float = 4.0,
    alpha: float = 0.7,
) -> Tensor:
    """Standard knowledge distillation loss.

    kd_loss   = KL(softmax(student/T) || softmax(teacher/T)) * T^2
    task_loss = cross_entropy(student_logits, labels)
    total     = alpha * kd_loss + (1 - alpha) * task_loss

    Args:
        student_logits: (B, S, V) or (N, V) raw student logits.
        teacher_logits: Same shape as student_logits.
        labels: (B, S) or (N,) ground-truth token ids.
        temperature: Softening temperature.
        alpha: Weight for KD loss (1-alpha for task loss).

    Returns:
        Scalar loss tensor.
    """
    # Flatten to 2-D for uniform handling
    if student_logits.dim() == 3:
        B, S, V = student_logits.shape
        s_flat = student_logits.contiguous().view(-1, V)
        t_flat = teacher_logits.contiguous().view(-1, V)
        lbl_flat = labels.contiguous().view(-1)
    else:
        s_flat = student_logits
        t_flat = teacher_logits
        lbl_flat = labels

    # Task loss: cross-entropy
    task_loss = F.cross_entropy(s_flat, lbl_flat)

    # KD loss: KL divergence between softened distributions
    s_log_soft = F.log_softmax(s_flat / temperature, dim=-1)
    t_soft = F.softmax(t_flat.detach() / temperature, dim=-1)
    kd_loss = F.kl_div(s_log_soft, t_soft, reduction="batchmean") * (temperature ** 2)

    return alpha * kd_loss + (1.0 - alpha) * task_loss


def fake_quantize_weights(model: nn.Module, bits: int) -> nn.Module:
    """Simple symmetric per-tensor fake quantization of all weight matrices.

    Quantizes and immediately dequantizes weights so their values are snapped
    to representable levels. Modification is in-place.

    Args:
        model: Any nn.Module.
        bits: Number of quantization bits (e.g., 8 for INT8).

    Returns:
        The model (modified in-place).
    """
    n_levels = 2 ** (bits - 1) - 1  # symmetric: range [-n_levels, n_levels]

    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" not in name:
                continue
            w = param.data
            w_max = w.abs().max()
            if w_max == 0:
                continue
            scale = w_max / n_levels
            # Quantize: round to nearest integer level
            w_q = torch.round(w / scale).clamp(-n_levels, n_levels)
            # Dequantize: scale back to float domain
            param.data.copy_(w_q * scale)

    return model


# ---------------------------------------------------------------------------
# Abstract Stage
# ---------------------------------------------------------------------------

class CompressionStage:
    """Base class for a single compression stage."""

    def __init__(self, name: str) -> None:
        self.name = name

    def apply(
        self,
        model: nn.Module,
        data: Tensor,
        config: CompressionConfig,
        teacher: nn.Module | None = None,
    ) -> tuple[nn.Module, dict[str, float]]:
        """Apply this compression stage.

        Args:
            model: The model to compress (modified in-place where possible).
            data: Calibration / training data tensor (B, S).
            config: CompressionConfig.
            teacher: Optional teacher model (required for distillation stages).

        Returns:
            (model, metrics_dict)
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete Stages
# ---------------------------------------------------------------------------

class PruneStage(CompressionStage):
    """Magnitude-based global weight pruning stage."""

    def __init__(self) -> None:
        super().__init__("prune")

    def apply(
        self,
        model: nn.Module,
        data: Tensor,
        config: CompressionConfig,
        teacher: nn.Module | None = None,
    ) -> tuple[nn.Module, dict[str, float]]:
        magnitude_prune(model, config.prune_fraction)
        sparsity_info = compute_model_sparsity(model)
        return model, {"sparsity": sparsity_info["overall_sparsity"]}


class DistillStage(CompressionStage):
    """Single-step knowledge distillation stage."""

    def __init__(self) -> None:
        super().__init__("distill")

    def apply(
        self,
        model: nn.Module,
        data: Tensor,
        config: CompressionConfig,
        teacher: nn.Module | None = None,
    ) -> tuple[nn.Module, dict[str, float]]:
        if teacher is None:
            raise ValueError("DistillStage requires a teacher model.")

        # One forward pass of KD training
        model.train()
        teacher.eval()

        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=1e-4
        )
        optimizer.zero_grad()

        input_ids = data
        labels = input_ids

        # Teacher forward (no grad)
        with torch.no_grad():
            t_out = teacher(input_ids)
            teacher_logits = t_out[1] if isinstance(t_out, tuple) else t_out

        # Student forward
        s_out = model(input_ids)
        student_logits = s_out[1] if isinstance(s_out, tuple) else s_out

        # Flatten for loss computation (use shifted labels for LM)
        B, S, V = student_logits.shape
        student_flat = student_logits[:, :-1].contiguous().view(-1, V)
        teacher_flat = teacher_logits[:, :-1].contiguous().view(-1, V)
        lbl_flat = labels[:, 1:].contiguous().view(-1)

        T = config.distill_temperature
        alpha = config.distill_alpha

        task_loss = F.cross_entropy(student_flat, lbl_flat)

        s_log_soft = F.log_softmax(student_flat / T, dim=-1)
        t_soft = F.softmax(teacher_flat.detach() / T, dim=-1)
        kd_loss = F.kl_div(s_log_soft, t_soft, reduction="batchmean") * (T ** 2)

        total_loss = alpha * kd_loss + (1.0 - alpha) * task_loss

        total_loss.backward()
        optimizer.step()

        return model, {
            "distill_loss": total_loss.item(),
            "kd_loss": kd_loss.item(),
            "task_loss": task_loss.item(),
        }


class QuantizeStage(CompressionStage):
    """Fake quantization stage with error measurement."""

    def __init__(self) -> None:
        super().__init__("quantize")

    def apply(
        self,
        model: nn.Module,
        data: Tensor,
        config: CompressionConfig,
        teacher: nn.Module | None = None,
    ) -> tuple[nn.Module, dict[str, float]]:
        # Measure output before quantization
        model.eval()
        with torch.no_grad():
            out_before = model(data)
            logits_before = out_before[1] if isinstance(out_before, tuple) else out_before

        fake_quantize_weights(model, config.quantize_bits)

        # Measure output after quantization
        with torch.no_grad():
            out_after = model(data)
            logits_after = out_after[1] if isinstance(out_after, tuple) else out_after

        quant_error = float(
            F.mse_loss(logits_after, logits_before).item()
        )

        return model, {"quant_error": quant_error}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

_STAGE_REGISTRY: dict[str, type[CompressionStage]] = {
    "prune": PruneStage,
    "distill": DistillStage,
    "quantize": QuantizeStage,
}


class CompressionPipeline:
    """End-to-end compression pipeline combining multiple stages.

    Args:
        student: The model to compress.
        teacher: Optional teacher model (required if "distill" is in stages).
        config: CompressionConfig controlling which stages to run and with what params.
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module | None,
        config: CompressionConfig,
    ) -> None:
        self.student = student
        self.teacher = teacher
        self.config = config
        self._results: dict[str, dict[str, float]] = {}

    def run(self, data: Tensor) -> dict[str, dict]:
        """Execute stages in config.stages order.

        Args:
            data: Input tensor of shape (B, S) token ids.

        Returns:
            {stage_name: metrics_dict} for each stage that was run.
        """
        results: dict[str, dict] = {}

        for stage_name in self.config.stages:
            stage_cls = _STAGE_REGISTRY.get(stage_name)
            if stage_cls is None:
                raise ValueError(f"Unknown compression stage: {stage_name!r}")

            stage = stage_cls()
            self.student, metrics = stage.apply(
                self.student, data, self.config, self.teacher
            )
            results[stage_name] = metrics

        self._results = results
        return results

    def get_compressed_model(self) -> nn.Module:
        """Return the compressed student model after running the pipeline."""
        return self.student
