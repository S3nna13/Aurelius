"""DeepSeek-V3-style FP8 block quantization for ~2x inference throughput.

Quantizes weight matrices to float8_e4m3fn format (or int8 emulation if
torch.float8_e4m3fn is unavailable) with a shared scale per 128-element block.

More accurate than flat INT8 because each block gets its own scale adapted
to its local dynamic range.

Reference: DeepSeek-V3 Technical Report — per-block FP8 weight quantization.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BLOCK_SIZE: int = 128   # elements per quantization block
FP8_MAX: float = 448.0  # float8_e4m3fn representable maximum

# Detect native FP8 support (added in PyTorch 2.1)
HAS_FP8: bool = hasattr(torch, "float8_e4m3fn")
STORAGE_DTYPE: torch.dtype = torch.float8_e4m3fn if HAS_FP8 else torch.int8  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Core quantize / dequantize
# ---------------------------------------------------------------------------

def fp8_block_quantize(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight matrix to FP8 with per-block scales.

    Args:
        weight: (out_features, in_features) float32 or bfloat16 tensor.

    Returns:
        quantized: Same shape as *weight*, stored as ``STORAGE_DTYPE``
                   (float8_e4m3fn if available, otherwise int8).
        scales:    (n_blocks,) float32 tensor — one scale per BLOCK_SIZE
                   elements of the flattened weight.

    Algorithm:
        1. Flatten weight to 1-D.
        2. Pad to a multiple of BLOCK_SIZE.
        3. Reshape to (n_blocks, BLOCK_SIZE).
        4. Per block: scale = max(|block|) / FP8_MAX; then
           quantized = clamp(block / scale, -FP8_MAX, FP8_MAX).
        5. Store quantized as STORAGE_DTYPE, scales as float32.
        6. Reshape quantized back to original shape (+ pad trimmed).
    """
    original_shape = weight.shape
    w = weight.detach().float().flatten()  # work in float32

    n_elements = w.numel()
    remainder = n_elements % BLOCK_SIZE
    if remainder != 0:
        pad_size = BLOCK_SIZE - remainder
        w = F.pad(w, (0, pad_size))

    n_blocks = w.numel() // BLOCK_SIZE
    w_blocks = w.reshape(n_blocks, BLOCK_SIZE)

    # Per-block scale: max(|x|) / FP8_MAX, clamped away from zero
    block_abs_max = w_blocks.abs().max(dim=-1).values.clamp(min=1e-12)
    scales = (block_abs_max / FP8_MAX).float()  # (n_blocks,)

    # Quantize: divide by scale then clamp to FP8 range
    w_scaled = w_blocks / scales.unsqueeze(-1)
    w_clamped = w_scaled.clamp(-FP8_MAX, FP8_MAX)

    if HAS_FP8:
        w_q = w_clamped.to(STORAGE_DTYPE)
    else:
        # Emulate with int8: round to nearest integer in [-128, 127]
        # Map [-FP8_MAX, FP8_MAX] → [-127, 127] (symmetric, avoids -128)
        w_q = (w_clamped * (127.0 / FP8_MAX)).round().clamp(-127, 127).to(torch.int8)

    # Trim padded elements and reshape back to original shape
    w_q_flat = w_q.flatten()[:n_elements]
    quantized = w_q_flat.reshape(original_shape)

    return quantized, scales


def fp8_block_dequantize(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    original_shape: tuple,
) -> torch.Tensor:
    """Dequantize FP8 (or int8) weights back to float32.

    Args:
        quantized:      Quantized weight tensor (any shape; will be flattened).
        scales:         (n_blocks,) per-block float32 scales.
        original_shape: Original weight shape to restore.

    Returns:
        float32 tensor of *original_shape*.
    """
    n_elements = math.prod(original_shape)
    n_blocks = scales.numel()

    # Flatten and pad to n_blocks * BLOCK_SIZE
    padded_size = n_blocks * BLOCK_SIZE
    w_flat = quantized.detach().flatten()[:n_elements].float()

    if padded_size > n_elements:
        w_flat = F.pad(w_flat, (0, padded_size - n_elements))

    w_blocks = w_flat.reshape(n_blocks, BLOCK_SIZE)

    if HAS_FP8:
        # Values are already in [-FP8_MAX, FP8_MAX] after cast to float
        w_dq = w_blocks * scales.unsqueeze(-1)
    else:
        # int8 was stored as round(val * 127 / FP8_MAX) → recover val
        w_dq = (w_blocks / (127.0 / FP8_MAX)) * scales.unsqueeze(-1)

    return w_dq.flatten()[:n_elements].reshape(original_shape).float()


# ---------------------------------------------------------------------------
# FP8Linear layer
# ---------------------------------------------------------------------------

class FP8Linear(nn.Module):
    """Linear layer with FP8 weight storage.

    Stores weights in FP8 for memory efficiency and dequantizes to float32
    for computation.

    Memory:
        FP8  = 8 bits/weight + 32 bits per 128 weights (scale) ≈ 8.25 bits/weight
        BF16 = 16 bits/weight → ~2× memory reduction.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._original_shape: tuple[int, int] = (out_features, in_features)

        # Initialise random float weight, quantize, store as buffers
        w = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        w_q, scales = fp8_block_quantize(w)
        self.register_buffer("weight_q", w_q)       # STORAGE_DTYPE
        self.register_buffer("fp8_scales", scales)  # float32, (n_blocks,)

        if bias:
            self.bias: nn.Parameter | None = nn.Parameter(
                torch.zeros(out_features)
            )
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "FP8Linear":
        """Convert an existing ``nn.Linear`` to ``FP8Linear``.

        Args:
            linear: Source linear layer.

        Returns:
            New FP8Linear with quantized copy of *linear*'s weights.
        """
        obj: FP8Linear = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.in_features = linear.in_features
        obj.out_features = linear.out_features
        obj._original_shape = (linear.out_features, linear.in_features)

        w = linear.weight.detach().float()
        w_q, scales = fp8_block_quantize(w)
        obj.register_buffer("weight_q", w_q)
        obj.register_buffer("fp8_scales", scales)

        if linear.bias is not None:
            obj.bias = nn.Parameter(linear.bias.detach().clone())
        else:
            obj.bias = None

        return obj

    def _dequantized_weight(self) -> torch.Tensor:
        return fp8_block_dequantize(
            self.weight_q, self.fp8_scales, self._original_shape
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._dequantized_weight()
        return F.linear(x, w, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, storage={'fp8' if HAS_FP8 else 'int8'}"
        )


# ---------------------------------------------------------------------------
# Model-level helpers
# ---------------------------------------------------------------------------

def quantize_model_fp8(
    model: nn.Module,
    skip_layers: list[str] | None = None,
) -> nn.Module:
    """Replace all ``nn.Linear`` layers with ``FP8Linear`` in-place.

    Args:
        model:       PyTorch model to quantize.
        skip_layers: List of module name substrings to skip
                     (e.g., ``['lm_head', 'embed']``).

    Returns:
        The model modified in-place.
    """
    if skip_layers is None:
        skip_layers = []

    def _should_skip(name: str) -> bool:
        return any(pattern in name for pattern in skip_layers)

    # Collect replacements first to avoid mutating dict while iterating
    replacements: list[tuple[nn.Module, str, FP8Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if _should_skip(name):
            continue
        fp8_layer = FP8Linear.from_linear(module)
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        replacements.append((parent, parts[-1], fp8_layer))

    for parent, attr, new_layer in replacements:
        setattr(parent, attr, new_layer)

    return model


def compute_fp8_memory_savings(model: nn.Module) -> dict:
    """Compute memory statistics comparing FP8 vs BF16 storage.

    Args:
        model: PyTorch model (may contain ``FP8Linear`` or ``nn.Linear`` layers).

    Returns:
        dict with keys:
            ``fp8_params``    — total parameter count in FP8Linear layers.
            ``total_params``  — total model parameter count.
            ``fp8_memory_mb`` — estimated MB used by FP8 weight storage
                                (1 byte/weight + 4 bytes per 128-weight block).
            ``bf16_memory_mb`` — MB those same weights would consume in BF16
                                 (2 bytes/weight).
            ``savings_factor`` — bf16_memory_mb / fp8_memory_mb.
    """
    total_params: int = sum(p.numel() for p in model.parameters())
    fp8_params: int = 0
    fp8_bytes: float = 0.0
    bf16_bytes: float = 0.0

    for module in model.modules():
        if not isinstance(module, FP8Linear):
            continue
        n = module.out_features * module.in_features
        fp8_params += n

        n_blocks = math.ceil(n / BLOCK_SIZE)
        # weight_q: 1 byte/element (fp8 or int8); fp8_scales: 4 bytes/block
        fp8_bytes += n * 1 + n_blocks * 4
        # BF16 would be 2 bytes/element
        bf16_bytes += n * 2

    fp8_memory_mb = fp8_bytes / (1024 ** 2)
    bf16_memory_mb = bf16_bytes / (1024 ** 2)
    savings_factor = (bf16_memory_mb / fp8_memory_mb) if fp8_memory_mb > 0 else 1.0

    return {
        "fp8_params": fp8_params,
        "total_params": total_params,
        "fp8_memory_mb": fp8_memory_mb,
        "bf16_memory_mb": bf16_memory_mb,
        "savings_factor": savings_factor,
    }
