"""MXFP4 microscaling quantization for Aurelius linear layers.

Inspired by gpt-oss-20b microscaling (MX) format:
- Weights are stored in 4-bit integer format.
- Groups of `block_size` (16 or 32) weights share a single FP8 block-level
  scaling factor (emulated as FP32 clamped to the FP8 E8M0 range).
- More accurate than flat INT4 due to fine-grained, per-block scale.

Reference: OCP MX Specification, Microsoft MX / Microscaling formats.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# FP8 E8M0 biased exponent range: 2^-126 … 2^127
_FP8_MIN = 2.0**-126
_FP8_MAX = 2.0**127


@dataclass
class MXFPConfig:
    block_size: int = 32  # weights per scaling block (16 or 32)
    bits: int = 4  # quantization bits for weights (4)
    scale_bits: int = 8  # bits for block scale (FP8 emulated as FP32 clamped)
    symmetric: bool = True  # symmetric quantization around zero


# ---------------------------------------------------------------------------
# Core quantize / dequantize
# ---------------------------------------------------------------------------


def mxfp4_quantize(
    weight: torch.Tensor,
    cfg: MXFPConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight tensor using MXFP4 microscaling.

    Args:
        weight: Float tensor of any shape whose last dimension is divisible by
                ``cfg.block_size``.
        cfg:    MXFP4 configuration.

    Returns:
        quantized_int: int8 tensor with same shape as *weight*, values in
                       [-(2^(bits-1)), 2^(bits-1)-1].
        block_scales:  float32 tensor of shape (*weight.shape[:-1],
                       weight.shape[-1] // block_size), one scale per block.
    """
    orig_shape = weight.shape
    n_blocks = orig_shape[-1] // cfg.block_size

    # Reshape to (..., n_blocks, block_size)
    w = weight.reshape(*orig_shape[:-1], n_blocks, cfg.block_size).float()

    # Per-block max abs → block scale
    block_abs_max = w.abs().max(dim=-1).values.clamp(min=1e-30)

    # Emulate FP8 E8M0: clamp to power-of-two representable range
    block_scales = block_abs_max.clamp(_FP8_MIN, _FP8_MAX)  # float32

    # Normalize each block
    w_norm = w / block_scales.unsqueeze(-1)  # values in [-1, 1]

    # 4-bit symmetric: levels in [-(2^(bits-1)), 2^(bits-1)-1]
    q_min = -(2 ** (cfg.bits - 1))  # -8 for bits=4
    q_max = (2 ** (cfg.bits - 1)) - 1  #  7 for bits=4

    w_q = (w_norm * q_max).round().clamp(q_min, q_max).to(torch.int8)

    # Restore original last-dim layout
    w_q = w_q.reshape(orig_shape)

    return w_q, block_scales.float()


def mxfp4_dequantize(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    cfg: MXFPConfig,
    original_shape: tuple,
) -> torch.Tensor:
    """Reconstruct approximate weights from quantized ints and block scales.

    Args:
        quantized:      int8 tensor with shape *original_shape*.
        scales:         float32 tensor of shape (*original_shape[:-1],
                        original_shape[-1] // block_size).
        cfg:            MXFP4 configuration.
        original_shape: Shape of the original weight tensor.

    Returns:
        float32 tensor of *original_shape*.
    """
    n_blocks = original_shape[-1] // cfg.block_size
    q_max = (2 ** (cfg.bits - 1)) - 1  # 7 for bits=4

    # Reshape to (..., n_blocks, block_size)
    q = quantized.reshape(*original_shape[:-1], n_blocks, cfg.block_size).float()

    # Scale back: q / q_max is the normalised value, then multiply by block scale
    w = (q / q_max) * scales.unsqueeze(-1)

    return w.reshape(original_shape).float()


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------


def mxfp4_quantization_error(
    weight: torch.Tensor,
    cfg: MXFPConfig,
) -> dict[str, float]:
    """Quantize then dequantize and compute error metrics.

    Returns:
        dict with keys: "mse", "max_error", "relative_error", "snr_db".
    """
    q, scales = mxfp4_quantize(weight, cfg)
    reconstructed = mxfp4_dequantize(q, scales, cfg, tuple(weight.shape))

    diff = weight.float() - reconstructed
    mse = diff.pow(2).mean().item()
    max_error = diff.abs().max().item()

    weight_norm = weight.float().pow(2).mean().item()
    relative_error = (mse / (weight_norm + 1e-30)) ** 0.5

    # SNR: signal power / noise power in dB
    signal_power = weight_norm
    noise_power = mse
    snr_db = 10.0 * math.log10((signal_power + 1e-30) / (noise_power + 1e-30))

    return {
        "mse": float(mse),
        "max_error": float(max_error),
        "relative_error": float(relative_error),
        "snr_db": float(snr_db),
    }


# ---------------------------------------------------------------------------
# MXFP4Linear layer
# ---------------------------------------------------------------------------


class MXFP4Linear(nn.Module):
    """Linear layer with MXFP4 quantized weights.

    Weights are stored quantized; dequantization happens on-the-fly during
    the forward pass.  Gradients flow through the dequantized weights
    (straight-through estimator for scales).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        cfg: MXFPConfig | None = None,
    ) -> None:
        super().__init__()
        if cfg is None:
            cfg = MXFPConfig()
        self.cfg = cfg
        self.in_features = in_features
        self.out_features = out_features

        # Initialise random float weight, quantize, store buffers
        w = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        q, scales = mxfp4_quantize(w, cfg)
        self.register_buffer("weight_q", q)  # int8
        self.register_buffer("block_scales", scales)  # float32
        self._orig_shape = (out_features, in_features)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        cfg: MXFPConfig | None = None,
    ) -> MXFP4Linear:
        """Convert an existing ``nn.Linear`` to MXFP4Linear."""
        if cfg is None:
            cfg = MXFPConfig()
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.cfg = cfg
        obj.in_features = linear.in_features
        obj.out_features = linear.out_features
        obj._orig_shape = (linear.out_features, linear.in_features)

        w = linear.weight.detach().float()
        q, scales = mxfp4_quantize(w, cfg)
        obj.register_buffer("weight_q", q)
        obj.register_buffer("block_scales", scales)

        if linear.bias is not None:
            obj.bias = nn.Parameter(linear.bias.detach().clone())
        else:
            obj.bias = None

        return obj

    def _dequantized_weight(self) -> torch.Tensor:
        return mxfp4_dequantize(self.weight_q, self.block_scales, self.cfg, self._orig_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._dequantized_weight()
        return F.linear(x, w, self.bias)

    def compression_ratio(self) -> float:
        """Memory reduction vs FP32: original_bytes / quantized_bytes.

        FP32 weight: out * in * 4 bytes.
        Quantized:   out * in * 1 byte (int8 storage for 4-bit values)
                   + n_blocks * 4 bytes (float32 scales).
        """
        n_params = self.out_features * self.in_features
        fp32_bytes = n_params * 4
        n_blocks = n_params // self.cfg.block_size
        quant_bytes = n_params * 1 + n_blocks * 4  # int8 + float32 scales
        return fp32_bytes / quant_bytes

    def effective_bits(self) -> float:
        """Bits per parameter including scale overhead.

        Formula: (bits * n + scale_bits * n / block_size) / n
                = bits + scale_bits / block_size
        """
        return self.cfg.bits + self.cfg.scale_bits / self.cfg.block_size


# ---------------------------------------------------------------------------
# Model-level helpers
# ---------------------------------------------------------------------------


def quantize_model(
    model: nn.Module,
    cfg: MXFPConfig | None = None,
    skip_layers: list[str] | None = None,
) -> nn.Module:
    """Replace all ``nn.Linear`` layers with ``MXFP4Linear`` in-place.

    Args:
        model:       The model to quantize.
        cfg:         MXFP4 configuration (default: MXFPConfig()).
        skip_layers: List of name substrings; layers whose qualified name
                     contains any of these strings will not be quantized.

    Returns:
        The model (modified in place).
    """
    if cfg is None:
        cfg = MXFPConfig()
    if skip_layers is None:
        skip_layers = []

    def _should_skip(name: str) -> bool:
        return any(pattern in name for pattern in skip_layers)

    # Collect replacements first to avoid mutating while iterating
    replacements: list[tuple[nn.Module, str, MXFP4Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if _should_skip(name):
            continue
        mxfp4_layer = MXFP4Linear.from_linear(module, cfg)
        # Find the parent module and attribute name
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        replacements.append((parent, parts[-1], mxfp4_layer))

    for parent, attr, new_layer in replacements:
        setattr(parent, attr, new_layer)

    return model


def estimate_quantization_impact(
    model: nn.Module,
    sample_input: torch.Tensor,
    cfg: MXFPConfig | None = None,
) -> dict[str, float]:
    """Estimate quantization impact across all quantizable Linear layers.

    For each ``nn.Linear`` in the model, compute quantization error metrics.

    Args:
        model:        The model (not mutated).
        sample_input: Unused for computation but accepted for API consistency.
        cfg:          MXFP4 configuration (default: MXFPConfig()).

    Returns:
        dict with keys:
          "mean_snr_db", "min_snr_db", "compression_ratio",
          "total_params", "quantized_params".
    """
    if cfg is None:
        cfg = MXFPConfig()

    snr_values: list[float] = []
    total_params = 0
    quantized_params = 0

    for name, module in model.named_modules():
        total_params += sum(p.numel() for p in module.parameters(recurse=False))

    # Re-count to avoid double counting: only leaf parameters
    # Actually iterate all modules but count only parameters of each module
    # (non-recursive) and track which are quantizable linears.
    total_params = sum(p.numel() for p in model.parameters())

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        w = module.weight.detach().float()
        metrics = mxfp4_quantization_error(w, cfg)
        snr_values.append(metrics["snr_db"])
        quantized_params += w.numel()

    if not snr_values:
        mean_snr = float("nan")
        min_snr = float("nan")
        compression = 1.0
    else:
        mean_snr = sum(snr_values) / len(snr_values)
        min_snr = min(snr_values)
        # compression ratio: use a representative layer calculation
        # bits + scale overhead vs 32-bit
        eff_bits = cfg.bits + cfg.scale_bits / cfg.block_size
        compression = 32.0 / eff_bits

    return {
        "mean_snr_db": float(mean_snr),
        "min_snr_db": float(min_snr),
        "compression_ratio": float(compression),
        "total_params": int(total_params),
        "quantized_params": int(quantized_params),
    }
