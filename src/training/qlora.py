"""QLoRA: Efficient Fine-tuning of Quantized LLMs (Dettmers et al., arXiv:2305.14314).

Key contributions implemented here:
  1. NF4 (NormalFloat4) quantization — information-theoretically optimal 4-bit
     quantization for normally distributed weights (Section 2.1).
  2. Block-wise quantization — divides each weight matrix into blocks of size B,
     computes a per-block scale, then quantizes each block independently (Section 2.1).
  3. QLoRA fine-tuning — frozen NF4 base weight + trainable bfloat16 LoRA adapters
     A and B; dequantize on the fly at forward time (Section 2.3).

Double quantization (Section 2.2) is **not** implemented here; this is a
single-quantization baseline fully faithful to the algorithm.

Paper variable notation is used throughout:
  W       — original full-precision weight matrix
  W_nf4   — block-wise NF4-quantized weight (codes + scales)
  r       — LoRA rank
  alpha   — LoRA scaling hyperparameter (scaling = alpha / r)
  A, B    — LoRA adapter matrices (A: in→r, B: r→out)
  B_size  — quantization block size (paper uses 64)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# NF4 quantization levels (Section 2.1)
# ---------------------------------------------------------------------------
# 16 levels are the (2i - 1) / (2 * 2^4) quantiles of N(0, 1) for i = 1..16,
# then normalized to [-1, 1] by dividing by the absolute maximum value.
#
# Φ^{-1}(p) = sqrt(2) * erfinv(2p - 1)
#
# We precompute them once as a module-level constant.


def _compute_nf4_levels() -> torch.Tensor:
    """Compute the 16 NF4 quantization levels from N(0,1) quantiles.

    qi = Φ^{-1}((2i - 1) / (2 * 2^4))  for i = 1 .. 16
    Then normalize so max(|qi|) = 1.
    """
    k = 16  # 2^4 levels
    # Probabilities: (2i-1)/(2k) for i=1..16
    i = torch.arange(1, k + 1, dtype=torch.float64)
    p = (2.0 * i - 1.0) / (2.0 * k)  # shape (16,)
    # Φ^{-1}(p) = sqrt(2) * erfinv(2p - 1)
    levels = math.sqrt(2.0) * torch.erfinv(2.0 * p - 1.0)  # shape (16,)
    # Normalize to [-1, 1]
    levels = levels / levels.abs().max()
    return levels.to(torch.float32)


# Module-level constant: 16 sorted float32 NF4 levels in [-1, 1].
NF4_LEVELS: torch.Tensor = _compute_nf4_levels()


# ---------------------------------------------------------------------------
# NF4Quantizer
# ---------------------------------------------------------------------------


class NF4Quantizer:
    """Block-wise NF4 (NormalFloat4) quantizer.

    Attributes:
        NF4_LEVELS: Tensor of 16 NF4 quantization levels, sorted, in [-1, 1].
    """

    NF4_LEVELS: torch.Tensor = NF4_LEVELS

    # ------------------------------------------------------------------
    # quantize
    # ------------------------------------------------------------------

    @staticmethod
    def quantize(
        w: torch.Tensor,
        block_size: int = 64,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize weight tensor w using block-wise NF4.

        Algorithm per Section 2.1:
          1. Flatten w, divide into blocks of size B_size.
          2. Per block: scale s = max(|w_block|); normalize w_block / s → [-1, 1].
          3. For each normalized value, find nearest NF4 level; store 4-bit code.

        Args:
            w:          Any-shape float tensor (treated as 1-D blocks).
            block_size: Number of elements per quantization block (default 64).

        Returns:
            codes:  uint8 tensor, same shape as w, values in [0, 15].
            scales: float32 tensor of shape (n_blocks,), one scale per block.
        """
        levels = NF4_LEVELS.to(w.device)

        original_shape = w.shape
        w_flat = w.detach().float().reshape(-1)  # (N,)
        N = w_flat.numel()

        # Pad to multiple of block_size
        pad = (block_size - N % block_size) % block_size
        if pad > 0:
            w_padded = F.pad(w_flat, (0, pad))
        else:
            w_padded = w_flat

        n_blocks = w_padded.numel() // block_size
        w_blocks = w_padded.reshape(n_blocks, block_size)  # (n_blocks, B_size)

        # Per-block scale: max(|w_block|); clamp to avoid division by zero
        scales = w_blocks.abs().amax(dim=1).clamp(min=1e-8)  # (n_blocks,)

        # Normalize each block to [-1, 1]
        w_norm = w_blocks / scales.unsqueeze(1)  # (n_blocks, B_size)

        # For each element, find the nearest NF4 level (closest in L2)
        # w_norm: (n_blocks, B_size) → expand to (n_blocks, B_size, 1)
        # levels: (16,) → expand to (1, 1, 16)
        diffs = (w_norm.unsqueeze(2) - levels.reshape(1, 1, 16)).abs()
        codes_2d = diffs.argmin(dim=2).to(torch.uint8)  # (n_blocks, B_size)

        # Remove padding and restore original shape
        codes_flat = codes_2d.reshape(-1)[:N]
        codes = codes_flat.reshape(original_shape)

        return codes, scales

    # ------------------------------------------------------------------
    # dequantize
    # ------------------------------------------------------------------

    @staticmethod
    def dequantize(
        codes: torch.Tensor,
        scales: torch.Tensor,
        original_shape: torch.Size | tuple[int, ...],
        block_size: int = 64,
    ) -> torch.Tensor:
        """Dequantize NF4 codes back to float.

        Args:
            codes:          uint8 tensor with values in [0, 15].
            scales:         float32 tensor of shape (n_blocks,).
            original_shape: Target shape of the output tensor.
            block_size:     Must match the block_size used in quantize().

        Returns:
            Float32 tensor of shape original_shape.
        """
        levels = NF4_LEVELS.to(codes.device)

        N = codes.numel()
        pad = (block_size - N % block_size) % block_size

        codes_flat = codes.reshape(-1).long()
        if pad > 0:
            codes_padded = F.pad(codes_flat, (0, pad))
        else:
            codes_padded = codes_flat

        n_blocks = codes_padded.numel() // block_size
        codes_blocks = codes_padded.reshape(n_blocks, block_size)  # (n_blocks, B_size)

        # Map codes to NF4 level values
        w_norm = levels[codes_blocks]  # (n_blocks, B_size)

        # Rescale by per-block scale
        w_deq = w_norm * scales.unsqueeze(1)  # (n_blocks, B_size)

        # Remove padding and reshape
        w_flat = w_deq.reshape(-1)[:N]
        return w_flat.reshape(original_shape)


# ---------------------------------------------------------------------------
# QLoRALinear
# ---------------------------------------------------------------------------


class QLoRALinear(nn.Module):
    """Linear layer with QLoRA fine-tuning (Dettmers et al. 2023, Section 2.3).

    The base weight W is stored frozen in NF4 (4-bit, block-wise quantized).
    LoRA adapters A and B are trainable in float32.

    Forward pass (paper notation):
        W_fp = dequantize(W_nf4)          — on-the-fly dequantization
        y = x @ W_fp.T + x @ A.T @ B.T * scaling
          = F.linear(x, W_fp) + scaling * F.linear(F.linear(x, A), B)

    Adapter initialisation follows standard LoRA:
        A ~ N(0, 1),  B = 0
    so the initial output equals the original frozen weight output.

    Args:
        in_features:  Input dimensionality.
        out_features: Output dimensionality.
        r:            LoRA rank.
        lora_alpha:   LoRA scaling hyperparameter α (scaling = α / r).
        block_size:   NF4 quantization block size (default 64).
        bias:         Whether to include a bias term.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        lora_alpha: float = 1.0,
        block_size: int = 64,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.block_size = block_size
        # scaling = α / r  (LoRA paper notation)
        self.scaling: float = lora_alpha / r

        # W_nf4: frozen NF4 base weight stored as codes + scales buffers.
        # codes:  (out_features, in_features) uint8
        # scales: (n_blocks,) float32
        n_blocks = math.ceil(out_features * in_features / block_size)
        self.register_buffer(
            "W_codes",
            torch.zeros(out_features, in_features, dtype=torch.uint8),
        )
        self.register_buffer(
            "W_scales",
            torch.ones(n_blocks, dtype=torch.float32),
        )

        # LoRA adapters A (in → r) and B (r → out) — trainable float32.
        # A ~ N(0, 1),  B = 0  (standard LoRA init)
        self.A = nn.Parameter(torch.empty(r, in_features, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(out_features, r, dtype=torch.float32))
        nn.init.normal_(self.A)

        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
        else:
            self.bias = None

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute QLoRA forward pass.

        y = x @ W_fp.T + scaling * x @ A.T @ B.T + bias
        """
        # Dequantize frozen W on the fly
        W_fp = NF4Quantizer.dequantize(
            self.W_codes,
            self.W_scales,
            original_shape=(self.out_features, self.in_features),
            block_size=self.block_size,
        ).to(x.dtype)  # match input dtype

        # Base path: x @ W_fp.T
        base_out = F.linear(x, W_fp, self.bias)

        # LoRA path: x @ A.T @ B.T  (scaling applied)
        lora_out = F.linear(F.linear(x, self.A.to(x.dtype)), self.B.to(x.dtype))

        return base_out + self.scaling * lora_out

    # ------------------------------------------------------------------
    # from_linear  (classmethod constructor)
    # ------------------------------------------------------------------

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        r: int,
        lora_alpha: float = 1.0,
        block_size: int = 64,
    ) -> QLoRALinear:
        """Construct a QLoRALinear from an existing nn.Linear.

        The linear layer's weight W is quantized to NF4 and stored frozen.
        LoRA adapters A and B are initialised to standard LoRA defaults.

        Args:
            linear:     Source nn.Linear module.
            r:          LoRA rank.
            lora_alpha: LoRA alpha scaling hyperparameter.
            block_size: NF4 block size.

        Returns:
            QLoRALinear with W_nf4 frozen and A, B trainable.
        """
        out_features, in_features = linear.weight.shape
        has_bias = linear.bias is not None

        module = cls(
            in_features=in_features,
            out_features=out_features,
            r=r,
            lora_alpha=lora_alpha,
            block_size=block_size,
            bias=has_bias,
        )

        # Quantize original weight to NF4
        W = linear.weight.detach().float()
        codes, scales = NF4Quantizer.quantize(W, block_size=block_size)
        module.W_codes.copy_(codes)
        module.W_scales.copy_(scales)

        # Copy bias if present
        if has_bias:
            module.bias = nn.Parameter(linear.bias.detach().float().clone())

        return module

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, lora_alpha={self.lora_alpha}, block_size={self.block_size}"
        )
