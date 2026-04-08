"""GPTQ: Accurate Post-Training Quantization (Frantar et al., 2022)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GPTQConfig:
    """Configuration for GPTQ quantization."""

    bits: int = 4
    """Quantization bits (4 or 8)."""

    group_size: int = 128
    """Group quantization size; -1 for per-tensor."""

    damp_percent: float = 0.01
    """Diagonal damping for Hessian numerical stability."""

    block_size: int = 128
    """Number of columns processed per GPTQ block."""

    actorder: bool = False
    """Activation-order: sort columns by Hessian diagonal (simplified)."""


# ---------------------------------------------------------------------------
# Core quantization helpers
# ---------------------------------------------------------------------------

def quantize_to_bits(x: torch.Tensor, bits: int, scale: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    """Quantize x using scale/zero, then dequantize back to float.

    Args:
        x:     Tensor to quantize.
        bits:  Number of quantization bits.
        scale: Per-group (or per-column) scale, broadcast-compatible with x.
        zero:  Per-group (or per-column) zero-point, same shape as scale.

    Returns:
        Dequantized tensor with same shape as x.
    """
    qmax = 2 ** bits - 1
    # Quantize
    q = torch.round(x / scale + zero).clamp(0, qmax)
    # Dequantize
    return (q - zero) * scale


def _group_params(W: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-group min-max scale and zero-point for weight matrix W (out, in)."""
    out_features, in_features = W.shape

    if group_size == -1 or group_size >= in_features:
        # Per-tensor (treat entire row as one group)
        w_min = W.min(dim=1, keepdim=True).values
        w_max = W.max(dim=1, keepdim=True).values
        scales = (w_max - w_min).clamp(min=1e-8)
        zeros = -w_min / scales
        return scales, zeros  # shapes: (out, 1)

    n_groups = math.ceil(in_features / group_size)
    # Pad W to multiple of group_size
    pad = n_groups * group_size - in_features
    if pad > 0:
        W_padded = F.pad(W, (0, pad))
    else:
        W_padded = W

    W_grouped = W_padded.reshape(out_features, n_groups, group_size)
    w_min = W_grouped.min(dim=2).values   # (out, n_groups)
    w_max = W_grouped.max(dim=2).values   # (out, n_groups)
    scales = (w_max - w_min).clamp(min=1e-8)
    zeros = -w_min / scales
    return scales, zeros  # (out, n_groups)


# ---------------------------------------------------------------------------
# GPTQ weight quantization
# ---------------------------------------------------------------------------

def quantize_weight_gptq(
    W: torch.Tensor,
    H: torch.Tensor,
    config: GPTQConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simplified GPTQ algorithm for a single weight matrix.

    Args:
        W:      Weight matrix of shape (out_features, in_features).
        H:      Hessian estimate of shape (in_features, in_features).
        config: GPTQConfig controlling bits, group_size, damping, etc.

    Returns:
        Tuple of (W_quantized, scales, zeros) where:
            W_quantized: float32 dequantized weights, same shape as W.
            scales:      (out_features, n_groups) per-group scales.
            zeros:       (out_features, n_groups) per-group zero-points.
    """
    out_features, in_features = W.shape
    W = W.clone().float()

    # --- Damping ---------------------------------------------------------
    diag_H = H.diag()
    damp = config.damp_percent * diag_H.mean().item()
    H64 = H.double() + damp * torch.eye(in_features, dtype=torch.float64, device=H.device)

    # --- Cholesky of H^{-1} (upper triangular) ---------------------------
    try:
        H_inv64 = torch.linalg.inv(H64)
        # Force symmetry
        H_inv64 = (H_inv64 + H_inv64.T) / 2.0
        # Add small regularisation to guarantee positive definiteness
        H_inv64 = H_inv64 + 1e-10 * torch.eye(in_features, dtype=torch.float64, device=H.device)
        Hinv64 = torch.linalg.cholesky(H_inv64, upper=True)
    except torch.linalg.LinAlgError:
        # Fallback: identity Cholesky (no error propagation)
        Hinv64 = torch.eye(in_features, dtype=torch.float64, device=H.device)

    Hinv = Hinv64.float()

    # --- Optional activation order (sort by diag descending) -------------
    if config.actorder:
        perm = torch.argsort(diag_H, descending=True)
        W = W[:, perm]
        Hinv = Hinv[perm][:, perm]
        inv_perm = torch.argsort(perm)
    else:
        inv_perm = None

    # --- Process blocks --------------------------------------------------
    W_q = W.clone()
    for block_start in range(0, in_features, config.block_size):
        block_end = min(block_start + config.block_size, in_features)
        for i in range(block_start, block_end):
            # Determine group membership for column i
            if config.group_size == -1 or config.group_size >= in_features:
                col_scale = (W[:, i].max() - W[:, i].min()).clamp(min=1e-8)
                col_zero = -W[:, i].min() / col_scale
                # Broadcast to (out_features,)
                sc = col_scale.expand(out_features)
                zr = col_zero.expand(out_features)
            else:
                g = i // config.group_size
                g_start = g * config.group_size
                g_end = min(g_start + config.group_size, in_features)
                w_group = W[:, g_start:g_end]
                sc = (w_group.max(dim=1).values - w_group.min(dim=1).values).clamp(min=1e-8)
                zr = -w_group.min(dim=1).values / sc

            # Clamp before quantization for numerical safety
            w_col = W_q[:, i].clamp(
                -1e4 * sc.abs().max().item(), 1e4 * sc.abs().max().item()
            )

            # Quantize column
            qmax = 2 ** config.bits - 1
            q_col = torch.round(w_col / sc + zr).clamp(0, qmax)
            w_quant_col = (q_col - zr) * sc
            W_q[:, i] = w_quant_col

            # Error and update
            h_ii = Hinv[i, i].item()
            if abs(h_ii) < 1e-8:
                continue
            err = (w_col - w_quant_col) / h_ii   # (out_features,)
            if i + 1 < in_features:
                W_q[:, i + 1:] -= err.unsqueeze(1) * Hinv[i, i + 1:].unsqueeze(0)

    # --- Undo activation order permutation if applied --------------------
    if inv_perm is not None:
        W_q = W_q[:, inv_perm]

    # --- Compute final scales/zeros from quantized weights ---------------
    scales, zeros = _group_params(W_q, config.group_size)

    return W_q, scales, zeros


# ---------------------------------------------------------------------------
# Hessian estimation
# ---------------------------------------------------------------------------

def compute_hessian(layer: nn.Linear, calibration_data: list[torch.Tensor]) -> torch.Tensor:
    """Estimate the GPTQ Hessian H ≈ 2 * X^T X / N from layer inputs.

    Args:
        layer:            nn.Linear layer to compute Hessian for.
        calibration_data: List of input tensors (each shape (B, *, in_features)).

    Returns:
        H: (in_features, in_features) symmetric positive semi-definite matrix.
    """
    in_features = layer.in_features
    H = torch.zeros(in_features, in_features, dtype=torch.float64, device=next(layer.parameters()).device)
    n_samples = 0

    inputs_collected: list[torch.Tensor] = []

    def _hook(module: nn.Module, inp: tuple, out: torch.Tensor) -> None:
        x = inp[0].detach().to(torch.float64)
        # Flatten all but last dimension → (N, in_features)
        x_flat = x.reshape(-1, in_features)
        inputs_collected.append(x_flat)

    handle = layer.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            for batch in calibration_data:
                layer(batch)
    finally:
        handle.remove()

    if inputs_collected:
        X = torch.cat(inputs_collected, dim=0)  # (N_total, in_features)
        n_samples = X.shape[0]
        H = 2.0 * X.T @ X / max(n_samples, 1)

    return H.float()


# ---------------------------------------------------------------------------
# Quantized linear layer
# ---------------------------------------------------------------------------

class GPTQLinear(nn.Module):
    """Quantized linear layer produced by GPTQ.

    Stores dequantized float weights for simplicity (no custom CUDA kernel).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
        group_size: int = 128,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size

        n_groups = math.ceil(in_features / group_size) if group_size > 0 and group_size != -1 else 1

        # Quantized weights stored as float (dequantized)
        self.weight_q = nn.Parameter(
            torch.zeros(out_features, in_features), requires_grad=False
        )
        self.scales = nn.Parameter(
            torch.ones(out_features, n_groups), requires_grad=False
        )
        self.zeros = nn.Parameter(
            torch.zeros(out_features, n_groups), requires_grad=False
        )
        if bias:
            self.bias: Optional[nn.Parameter] = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight_q)
        if self.bias is not None:
            out = out + self.bias
        return out

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        config: GPTQConfig,
        calibration_data: list[torch.Tensor],
    ) -> "GPTQLinear":
        """Create a GPTQLinear from an existing nn.Linear layer.

        Args:
            linear:           Source linear layer.
            config:           GPTQConfig.
            calibration_data: List of input tensors for Hessian estimation.

        Returns:
            GPTQLinear with quantized weights.
        """
        H = compute_hessian(linear, calibration_data)
        W = linear.weight.detach().float()
        W_q, scales, zeros = quantize_weight_gptq(W, H, config)

        has_bias = linear.bias is not None
        in_f, out_f = linear.in_features, linear.out_features
        q_linear = cls(
            in_features=in_f,
            out_features=out_f,
            bits=config.bits,
            group_size=config.group_size,
            bias=has_bias,
        )

        q_linear.weight_q = nn.Parameter(W_q, requires_grad=False)
        q_linear.scales = nn.Parameter(scales, requires_grad=False)
        q_linear.zeros = nn.Parameter(zeros, requires_grad=False)

        if has_bias:
            q_linear.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)

        return q_linear


# ---------------------------------------------------------------------------
# Model-level quantization
# ---------------------------------------------------------------------------

def apply_gptq_to_model(
    model: nn.Module,
    config: GPTQConfig,
    calibration_data: dict[str, list[torch.Tensor]],
) -> nn.Module:
    """Replace nn.Linear layers in model with GPTQLinear.

    Args:
        model:            The model to quantize in-place.
        config:           GPTQConfig.
        calibration_data: Dict mapping layer name → list of input tensors.

    Returns:
        Modified model (same object, modified in-place).
    """
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if name not in calibration_data:
            # No calibration data for this layer — skip or use random
            cal = [torch.randn(1, module.in_features)]
        else:
            cal = calibration_data[name]

        q_layer = GPTQLinear.from_linear(module, config, cal)

        # Navigate to parent and replace
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], q_layer)

    return model
