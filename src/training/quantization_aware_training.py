"""
Quantization-Aware Training (QAT) with fake quantization and Straight-Through Estimator (STE).

Supports:
- int8 / int4 fake quantization for weights and activations
- Per-tensor and per-channel quantization
- Symmetric and asymmetric modes
- Drop-in replacement of nn.Linear / nn.Embedding
- Model conversion, calibration, and integer export
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core fake-quantize autograd function (STE)
# ---------------------------------------------------------------------------

class FakeQuantize(torch.autograd.Function):
    """Fake-quantize with Straight-Through Estimator in the backward pass."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        n_bits: int,
    ) -> torch.Tensor:
        q_min = 0
        q_max = 2 ** n_bits - 1
        # Quantize
        x_q = torch.clamp(torch.round(x / scale) + zero_point, q_min, q_max)
        # Dequantize (stays in float)
        x_hat = (x_q - zero_point) * scale
        return x_hat

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        # STE: pass gradient through unchanged; return None for scale, zp, n_bits
        return grad, None, None, None


# ---------------------------------------------------------------------------
# Per-tensor quantizer
# ---------------------------------------------------------------------------

class PerTensorQuantizer(nn.Module):
    """
    Quantizes a tensor using a single (scale, zero_point) pair.

    Maintains running min/max statistics for calibration.
    """

    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = True,
        per_channel: bool = False,
    ) -> None:
        super().__init__()
        self.n_bits = n_bits
        self.symmetric = symmetric
        self.per_channel = per_channel

        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zero_point", torch.tensor(0.0))
        self.register_buffer("running_min", torch.tensor(float("inf")))
        self.register_buffer("running_max", torch.tensor(float("-inf")))
        self._calibrated = False

    def compute_scale_zp(
        self,
        x_min: torch.Tensor,
        x_max: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scale and zero_point from observed [x_min, x_max]."""
        q_min = 0
        q_max = 2 ** self.n_bits - 1

        if self.symmetric:
            abs_max = torch.maximum(x_min.abs(), x_max.abs())
            abs_max = torch.clamp(abs_max, min=1e-8)
            half_q = q_max / 2.0
            scale = abs_max / half_q
            zero_point = torch.zeros_like(scale)
        else:
            x_range = (x_max - x_min).clamp(min=1e-8)
            scale = x_range / (q_max - q_min)
            zero_point = torch.round(-x_min / scale)
            zero_point = zero_point.clamp(q_min, q_max)

        return scale, zero_point

    def calibrate(self, x: torch.Tensor) -> None:
        """Update running min/max and recompute scale/zero_point."""
        x_min = x.detach().min()
        x_max = x.detach().max()
        self.running_min = torch.minimum(self.running_min, x_min)
        self.running_max = torch.maximum(self.running_max, x_max)
        scale, zp = self.compute_scale_zp(self.running_min, self.running_max)
        self.scale.copy_(scale)
        self.zero_point.copy_(zp)
        self._calibrated = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._calibrated:
            self.calibrate(x)

        scale = self.scale.to(x.device)
        zp = self.zero_point.to(x.device)

        scale_t = scale.expand_as(x) if scale.numel() == 1 else scale
        zp_t = zp.expand_as(x) if zp.numel() == 1 else zp

        return FakeQuantize.apply(x, scale_t, zp_t, self.n_bits)


# ---------------------------------------------------------------------------
# Per-channel quantizer
# ---------------------------------------------------------------------------

class PerChannelQuantizer(PerTensorQuantizer):
    """
    Quantizes each output channel (dim=0) with its own scale / zero_point.
    """

    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = True,
        num_channels: int = 1,
    ) -> None:
        super().__init__(n_bits=n_bits, symmetric=symmetric, per_channel=True)
        self.num_channels = num_channels
        # Override buffers with per-channel tensors
        self.register_buffer("scale", torch.ones(num_channels))
        self.register_buffer("zero_point", torch.zeros(num_channels))
        self.register_buffer("running_min", torch.full((num_channels,), float("inf")))
        self.register_buffer("running_max", torch.full((num_channels,), float("-inf")))

    def calibrate(self, x: torch.Tensor) -> None:
        """Per-channel calibration: reduce over all dims except dim=0."""
        c = x.shape[0]
        x_flat = x.detach().reshape(c, -1)
        x_min = x_flat.min(dim=1).values
        x_max = x_flat.max(dim=1).values

        if self.running_min.shape[0] != c:
            self.running_min = torch.full((c,), float("inf"), device=x.device)
            self.running_max = torch.full((c,), float("-inf"), device=x.device)
            self.num_channels = c

        self.running_min = torch.minimum(self.running_min.to(x.device), x_min)
        self.running_max = torch.maximum(self.running_max.to(x.device), x_max)
        scale, zp = self.compute_scale_zp(self.running_min, self.running_max)
        self.scale = scale
        self.zero_point = zp
        self._calibrated = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._calibrated:
            self.calibrate(x)

        c = x.shape[0]
        if self.scale.shape[0] != c:
            self.calibrate(x)

        scale = self.scale.to(x.device)
        zp = self.zero_point.to(x.device)

        extra_dims = x.dim() - 1
        view_shape = (c,) + (1,) * extra_dims
        scale_t = scale.view(view_shape).expand_as(x)
        zp_t = zp.view(view_shape).expand_as(x)

        return FakeQuantize.apply(x, scale_t, zp_t, self.n_bits)


# ---------------------------------------------------------------------------
# QAT-aware Linear layer
# ---------------------------------------------------------------------------

class QATLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that applies fake quantization to
    both activations and weights during the forward pass.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        n_bits_weight: int = 8,
        n_bits_act: int = 8,
        symmetric: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.weight_quantizer = PerChannelQuantizer(
            n_bits=n_bits_weight, symmetric=symmetric, num_channels=out_features
        )
        self.act_quantizer = PerTensorQuantizer(
            n_bits=n_bits_act, symmetric=symmetric
        )

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        n_bits_weight: int = 8,
        n_bits_act: int = 8,
        symmetric: bool = True,
    ) -> "QATLinear":
        has_bias = linear.bias is not None
        qat_linear = cls(
            linear.in_features,
            linear.out_features,
            bias=has_bias,
            n_bits_weight=n_bits_weight,
            n_bits_act=n_bits_act,
            symmetric=symmetric,
        )
        qat_linear.weight.data.copy_(linear.weight.data)
        if has_bias and linear.bias is not None:
            qat_linear.bias.data.copy_(linear.bias.data)
        return qat_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = self.act_quantizer(x)
        w_q = self.weight_quantizer(self.weight)
        return F.linear(x_q, w_q, self.bias)


# ---------------------------------------------------------------------------
# QAT-aware Embedding layer
# ---------------------------------------------------------------------------

class QATEmbedding(nn.Module):
    """
    Drop-in replacement for nn.Embedding that fake-quantizes the weight
    at lookup time.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        n_bits: int = 8,
        symmetric: bool = True,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.n_bits = n_bits

        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.normal_(self.weight)

        self.weight_quantizer = PerChannelQuantizer(
            n_bits=n_bits, symmetric=symmetric, num_channels=num_embeddings
        )

    @classmethod
    def from_embedding(
        cls,
        emb: nn.Embedding,
        n_bits: int = 8,
        symmetric: bool = True,
    ) -> "QATEmbedding":
        qat_emb = cls(
            emb.num_embeddings,
            emb.embedding_dim,
            n_bits=n_bits,
            symmetric=symmetric,
        )
        qat_emb.weight.data.copy_(emb.weight.data)
        return qat_emb

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        w_q = self.weight_quantizer(self.weight)
        return F.embedding(input_ids, w_q)


# ---------------------------------------------------------------------------
# QATConverter: replace standard modules with QAT versions
# ---------------------------------------------------------------------------

class QATConverter:
    """
    Converts a standard nn.Module by replacing nn.Linear -> QATLinear and
    nn.Embedding -> QATEmbedding. Returns a new model (does not modify in-place).
    """

    def __init__(
        self,
        n_bits_weight: int = 8,
        n_bits_act: int = 8,
        symmetric: bool = True,
    ) -> None:
        self.n_bits_weight = n_bits_weight
        self.n_bits_act = n_bits_act
        self.symmetric = symmetric

    def _replace_modules(self, module: nn.Module) -> None:
        """Recursively replace Linear/Embedding children in-place."""
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(
                    module,
                    name,
                    QATLinear.from_linear(
                        child,
                        n_bits_weight=self.n_bits_weight,
                        n_bits_act=self.n_bits_act,
                        symmetric=self.symmetric,
                    ),
                )
            elif isinstance(child, nn.Embedding):
                setattr(
                    module,
                    name,
                    QATEmbedding.from_embedding(
                        child,
                        n_bits=self.n_bits_weight,
                        symmetric=self.symmetric,
                    ),
                )
            else:
                self._replace_modules(child)

    def convert(self, model: nn.Module) -> nn.Module:
        """Return a converted copy of model (does not modify original)."""
        new_model = copy.deepcopy(model)
        self._replace_modules(new_model)
        return new_model

    def calibrate_model(
        self,
        model: nn.Module,
        calibration_data: torch.Tensor,
    ) -> None:
        """
        Run one forward pass over calibration_data to trigger auto-calibration
        of all quantizer instances.
        """
        model.eval()
        with torch.no_grad():
            model(calibration_data)

    def export_int_model(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Export quantized integer weights.

        Returns {param_name: int_tensor} for each QATLinear / QATEmbedding weight.
        """
        result: Dict[str, torch.Tensor] = {}
        for name, module in model.named_modules():
            if isinstance(module, QATLinear):
                prefix = f"{name}.weight" if name else "weight"
                scale = module.weight_quantizer.scale
                zp = module.weight_quantizer.zero_point
                c = module.weight.shape[0]
                extra_dims = module.weight.dim() - 1
                view_shape = (c,) + (1,) * extra_dims
                scale_t = scale.view(view_shape)
                zp_t = zp.view(view_shape)
                w_int = torch.clamp(
                    torch.round(module.weight.detach() / scale_t) + zp_t,
                    0,
                    2 ** module.weight_quantizer.n_bits - 1,
                ).to(torch.int8)
                result[prefix] = w_int

            elif isinstance(module, QATEmbedding):
                prefix = f"{name}.weight" if name else "weight"
                scale = module.weight_quantizer.scale
                zp = module.weight_quantizer.zero_point
                c = module.weight.shape[0]
                scale_t = scale.view(c, 1)
                zp_t = zp.view(c, 1)
                w_int = torch.clamp(
                    torch.round(module.weight.detach() / scale_t) + zp_t,
                    0,
                    2 ** module.weight_quantizer.n_bits - 1,
                ).to(torch.int8)
                result[prefix] = w_int

        return result


# ---------------------------------------------------------------------------
# QuantizationTrainer
# ---------------------------------------------------------------------------

class QuantizationTrainer:
    """
    Minimal trainer wrapper for QAT fine-tuning.

    Expects model to produce logits of shape [B, T, vocab].
    Uses cross-entropy loss.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        n_bits: int = 8,
    ) -> None:
        self.model = model
        self.n_bits = n_bits
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single training step.

        input_ids: [B, T]  (long)
        labels:    [B, T]  (long)

        Returns scalar loss tensor (detached).
        """
        self.model.train()
        self.optimizer.zero_grad()

        logits = self.model(input_ids)

        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits.reshape(B * T, V),
            labels.reshape(B * T),
            ignore_index=-1,
        )
        loss.backward()
        self.optimizer.step()
        return loss.detach()

    def quantization_error(
        self,
        model_fp: nn.Module,
        model_qat: nn.Module,
        inputs: torch.Tensor,
    ) -> float:
        """Mean absolute difference in logits between fp model and qat model."""
        model_fp.eval()
        model_qat.eval()
        with torch.no_grad():
            logits_fp = model_fp(inputs)
            logits_qat = model_qat(inputs)
        return (logits_fp - logits_qat).abs().mean().item()


# ---------------------------------------------------------------------------
# QATConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class QATConfig:
    """Configuration for quantization-aware training."""

    n_bits_weight: int = 8
    n_bits_act: int = 8
    symmetric: bool = True
    lr: float = 1e-4
    per_channel_weight: bool = True
