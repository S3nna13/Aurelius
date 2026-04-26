"""LoftQ: Low-Rank Fine-Tuning with Quantization (Liu et al. 2023).

Initializes LoRA adapters to minimize quantization error so that:
    W_quantized + A @ B ≈ W_original

Unlike standard LoRA which initializes A ~ N(0, σ) and B = 0, LoftQ solves
for A and B via alternating optimization:
  1. Quantize W: W_q = quantize(W)
  2. SVD of residual: W - W_q ≈ A @ B (low-rank approx of quantization error)
  3. Update W_q = quantize(W - A @ B)
  4. Repeat for N iterations

This bridges quantization (QAT) and LoRA initialization for better accuracy
when deploying quantized models with LoRA adapters.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoftQConfig:
    rank: int = 16  # LoRA rank
    n_bits: int = 4  # Quantization bits (4 or 8)
    n_iter: int = 5  # Alternating optimization iterations
    target_modules: list[str] | None = None  # None = all Linear layers


# ---------------------------------------------------------------------------
# NF4 quantization (approximate uniform 4-bit)
# ---------------------------------------------------------------------------


def quantize_nf4(
    weight: torch.Tensor,
    group_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """NF4 quantization (approximate uniform 4-bit, Dettmers et al. 2023 QLoRA).

    For simplicity, approximates NF4 with uniform 4-bit signed quantization:
      1. Divide weight into groups of group_size along the last dimension.
      2. Per group: scale = max(|group|) / 7 (fits in [-7, 7] for 4-bit signed).
      3. Quantize: q = round(w / scale).clamp(-8, 7).
      4. Store as int8 (with per-group scale factors).

    Args:
        weight: (out_features, in_features) float weight tensor.
        group_size: Number of elements per quantization group (along in dim).

    Returns:
        quantized_int8: Same shape as weight, dtype=torch.int8.
        scales: (n_groups_total,) float scale factors, one per group.
    """
    out_features, in_features = weight.shape

    # Pad in_features to be divisible by group_size if needed
    pad = (group_size - in_features % group_size) % group_size
    if pad > 0:
        weight_padded = F.pad(weight, (0, pad))
    else:
        weight_padded = weight

    n_groups_per_row = weight_padded.shape[1] // group_size
    # Reshape: (out, n_groups_per_row, group_size)
    w_grouped = weight_padded.reshape(out_features, n_groups_per_row, group_size)

    # Per-group scale: max(|w|) / 7
    abs_max = w_grouped.abs().amax(dim=-1)  # (out, n_groups_per_row)
    scales = (abs_max / 7.0).clamp(min=1e-8)  # (out, n_groups_per_row)

    # Quantize
    w_q = (w_grouped / scales.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int8)

    # Trim padding and reshape back
    if pad > 0:
        # w_q is (out, n_groups_per_row, group_size); trim last group's padding
        w_q_flat = w_q.reshape(out_features, -1)[:, :in_features]
    else:
        w_q_flat = w_q.reshape(out_features, in_features)

    # Flatten scales
    scales_flat = scales.reshape(-1)  # (out * n_groups_per_row,)

    return w_q_flat, scales_flat


def dequantize_nf4(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 64,
) -> torch.Tensor:
    """Dequantize NF4-quantized weight: w_approx = quantized.float() * scale.

    Args:
        quantized: (out_features, in_features) int8 tensor.
        scales: (n_groups_total,) float scale factors.
        group_size: Must match the group_size used during quantize_nf4.

    Returns:
        (out_features, in_features) float tensor.
    """
    out_features, in_features = quantized.shape

    pad = (group_size - in_features % group_size) % group_size
    if pad > 0:
        q_padded = F.pad(quantized.float(), (0, pad))
    else:
        q_padded = quantized.float()

    n_groups_per_row = q_padded.shape[1] // group_size
    q_grouped = q_padded.reshape(out_features, n_groups_per_row, group_size)

    # Reshape scales back
    scales_2d = scales.reshape(out_features, n_groups_per_row)

    # Dequantize
    w_deq = q_grouped * scales_2d.unsqueeze(-1)

    # Trim padding
    if pad > 0:
        w_deq = w_deq.reshape(out_features, -1)[:, :in_features]
    else:
        w_deq = w_deq.reshape(out_features, in_features)

    return w_deq


# ---------------------------------------------------------------------------
# INT8 quantization (per-channel)
# ---------------------------------------------------------------------------


def quantize_int8(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel INT8 quantization.

    scale = max(|weight|, dim=-1) / 127
    quantized = round(weight / scale.unsqueeze(-1)).clamp(-128, 127)

    Args:
        weight: (out_features, in_features) float weight tensor.

    Returns:
        quantized_int8: (out_features, in_features) int8 tensor.
        scales: (out_features,) float scale factors.
    """
    scales = weight.abs().amax(dim=-1).clamp(min=1e-8) / 127.0  # (out_features,)
    quantized = (weight / scales.unsqueeze(-1)).round().clamp(-128, 127).to(torch.int8)
    return quantized, scales


def dequantize_int8(
    quantized: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Dequantize INT8-quantized weight.

    Args:
        quantized: (out_features, in_features) int8 tensor.
        scales: (out_features,) float scale factors.

    Returns:
        (out_features, in_features) float tensor.
    """
    return quantized.float() * scales.unsqueeze(-1)


# ---------------------------------------------------------------------------
# LoftQ initialization algorithm
# ---------------------------------------------------------------------------


def loftq_init(
    weight: torch.Tensor,
    rank: int,
    n_bits: int = 4,
    n_iter: int = 5,
    group_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """LoftQ initialization via alternating quantization and SVD.

    Solves W_q + A @ B ≈ weight via iterative refinement:
      For i in range(n_iter):
        1. residual = weight - (A @ B)  [skip A@B on first iter]
        2. Quantize residual -> W_q
        3. W_q_deq = dequantize(W_q)
        4. svd_residual = weight - W_q_deq
        5. U, S, Vh = svd(svd_residual)
        6. A = U[:, :rank] @ diag(sqrt(S[:rank]))
           B = diag(sqrt(S[:rank])) @ Vh[:rank, :]

    Args:
        weight: (out_features, in_features) float weight tensor.
        rank: LoRA rank. Clamped to min(out_features, in_features).
        n_bits: Quantization bits (4 or 8).
        n_iter: Number of alternating optimization iterations.
        group_size: Group size for NF4 quantization.

    Returns:
        A: (out_features, rank) LoRA down-projection.
        B: (rank, in_features) LoRA up-projection.
        W_q_dequant: (out_features, in_features) final quantized-dequantized weight.

    Such that W_q_dequant + A @ B ≈ weight (original).
    """
    out_features, in_features = weight.shape
    # Clamp rank to valid SVD range
    effective_rank = min(rank, out_features, in_features)

    w = weight.float().detach()

    # Initialize A and B to zero
    A = torch.zeros(out_features, effective_rank, dtype=w.dtype, device=w.device)
    B = torch.zeros(effective_rank, in_features, dtype=w.dtype, device=w.device)
    W_q_deq = torch.zeros_like(w)

    for i in range(n_iter):
        # Step 1: Compute residual after subtracting current low-rank approximation
        if i == 0:
            residual = w
        else:
            residual = w - A @ B

        # Step 2 & 3: Quantize and dequantize the residual
        if n_bits == 4:
            q, scales = quantize_nf4(residual, group_size=group_size)
            W_q_deq = dequantize_nf4(q, scales, group_size=group_size)
        else:
            q, scales = quantize_int8(residual)
            W_q_deq = dequantize_int8(q, scales)

        # Step 4: SVD residual = weight - W_q_deq (quantization error to capture)
        svd_residual = w - W_q_deq

        # Step 5: Truncated SVD
        U, S, Vh = torch.linalg.svd(svd_residual, full_matrices=False)
        # U: (out, min(out,in)), S: (min(out,in),), Vh: (min(out,in), in)

        # Step 6: Update A and B using top-rank singular values/vectors
        sqrt_S = S[:effective_rank].sqrt()  # (effective_rank,)
        A = U[:, :effective_rank] * sqrt_S.unsqueeze(0)  # (out, rank)
        B = sqrt_S.unsqueeze(-1) * Vh[:effective_rank, :]  # (rank, in)

    return A, B, W_q_deq


# ---------------------------------------------------------------------------
# LoftQLinear module
# ---------------------------------------------------------------------------


class LoftQLinear(nn.Module):
    """Linear layer with LoftQ-initialized LoRA adapters.

    The quantized-dequantized weight W_q is frozen. LoRA adapters A and B
    are trainable. Forward pass computes:

        output = x @ (W_q + scaling * A @ B).T + bias

    Args:
        original_linear: nn.Linear to replace.
        rank: LoRA rank.
        n_bits: Quantization bits (4 or 8).
        n_iter: LoftQ alternating optimization iterations.
        scaling: LoRA scaling factor (default: 1.0 / rank).
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 16,
        n_bits: int = 4,
        n_iter: int = 5,
        scaling: float | None = None,
    ) -> None:
        super().__init__()

        weight = original_linear.weight.data.float()
        out_features, in_features = weight.shape

        # Clamp rank for effective use
        effective_rank = min(rank, out_features, in_features)
        self.rank = effective_rank
        self.scaling = scaling if scaling is not None else 1.0 / effective_rank

        # Run LoftQ initialization
        A_init, B_init, W_q_deq = loftq_init(
            weight, rank=effective_rank, n_bits=n_bits, n_iter=n_iter
        )

        # Store frozen quantized-dequantized weight
        self.W_q = nn.Parameter(
            W_q_deq.to(original_linear.weight.dtype),
            requires_grad=False,
        )

        # Trainable LoRA adapters
        self.A = nn.Parameter(A_init.to(original_linear.weight.dtype))
        self.B = nn.Parameter(B_init.to(original_linear.weight.dtype))

        # Handle bias
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute x @ (W_q + scaling * A @ B).T + bias."""
        effective_weight = self.W_q + self.scaling * (self.A @ self.B)
        return F.linear(x, effective_weight, self.bias)


# ---------------------------------------------------------------------------
# apply_loftq: replace Linear layers in model
# ---------------------------------------------------------------------------


def apply_loftq(
    model: nn.Module,
    config: LoftQConfig,
    target_modules: list[str] | None = None,
) -> tuple[nn.Module, dict]:
    """Replace target Linear layers in model with LoftQLinear.

    Modifies model in-place.

    Args:
        model: nn.Module to modify.
        config: LoftQConfig with rank, n_bits, n_iter, target_modules.
        target_modules: List of module name substrings to target. If None,
                        uses config.target_modules. If still None, targets all
                        nn.Linear layers.

    Returns:
        (model, stats) where stats is:
          {
            'n_replaced': int,    # number of layers replaced
            'total_params': int,  # total trainable parameter count in model
            'lora_params': int,   # number of params in LoRA adapters (A and B)
          }
    """
    targets = target_modules or config.target_modules  # may be None

    n_replaced = 0
    lora_params = 0

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        # Filter by target_modules if provided
        if targets is not None:
            if not any(t in name for t in targets):
                continue

        # Skip layers where weight is too small for the requested rank
        out_f, in_f = module.weight.shape
        if out_f < config.rank or in_f < config.rank:
            continue

        # Navigate to parent module
        parts = name.rsplit(".", 1)
        if len(parts) == 1:
            parent, attr = model, parts[0]
        else:
            parent = model
            for part in parts[0].split("."):
                parent = getattr(parent, part)
            attr = parts[1]

        loftq_linear = LoftQLinear(
            original_linear=module,
            rank=config.rank,
            n_bits=config.n_bits,
            n_iter=config.n_iter,
        )
        setattr(parent, attr, loftq_linear)

        lora_params += loftq_linear.A.numel() + loftq_linear.B.numel()
        n_replaced += 1

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    stats = {
        "n_replaced": n_replaced,
        "total_params": total_params,
        "lora_params": lora_params,
    }
    return model, stats
