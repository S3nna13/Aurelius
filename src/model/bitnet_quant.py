"""BitNet b1.58 quantization utilities.

Reference: Ma et al., 2024 — "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits".

Ternary weights {-1, 0, +1} reduce memory and enable multiply-free matmuls.
Activations are quantized to int8 range per tensor with straight-through estimator (STE).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BitNetConfig:
    """Configuration for BitNet b1.58 quantization."""

    bits: int = 8  # bits for activation quantization
    use_ternary_weights: bool = True  # whether to ternarize weights
    eps: float = 1e-8  # numerical stability in scale computation


# ---------------------------------------------------------------------------
# Core quantization primitives
# ---------------------------------------------------------------------------


def ternarize_weight(W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Ternarize weight tensor to values in {-1, 0, +1}.

    Scale is computed as mean(|W|) + eps so that the rounded ratio clips
    nicely to the ternary alphabet.  Straight-through estimator (STE) is
    applied via the detach trick so that gradients flow through W unchanged
    during backward.

    Args:
        W: float weight tensor of any shape.

    Returns:
        ternary_W: tensor with values in {-1, 0, +1}, same shape as W.
        scale:     scalar tensor, mean absolute value of W (+ eps).
    """
    scale = W.abs().mean() + 1e-8
    # Forward: round then clamp to ternary alphabet
    W_norm = W / scale
    W_rounded = torch.clamp(torch.round(W_norm), -1.0, 1.0)
    # STE: gradients pass through as if identity
    ternary_W = W_rounded.detach() + (W - W.detach())
    # Return ternary values (no gradient needed for scale) and the scale
    return ternary_W, scale.detach()


def quantize_activation(x: torch.Tensor, bits: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor symmetric int quantization of activations.

    The quantized values are returned as floats (not integer dtype) for
    compatibility with standard PyTorch ops.  STE is applied via the
    detach trick.

    Args:
        x:    input activation tensor.
        bits: number of bits (default 8 → int8 range [-127, 127]).

    Returns:
        quantized_float: quantized values in float, clamped to int range.
        scale:           per-tensor scale (scalar).
    """
    q_max = 2 ** (bits - 1) - 1  # e.g. 127 for int8
    scale = x.abs().max() / q_max
    # Avoid division by zero
    scale = scale.clamp(min=1e-8)
    x_scaled = x / scale
    x_rounded = torch.clamp(torch.round(x_scaled), -q_max, q_max)
    # STE
    quantized_float = x_rounded.detach() + (x_scaled - x_scaled.detach())
    return quantized_float, scale.detach()


# ---------------------------------------------------------------------------
# BitLinear module
# ---------------------------------------------------------------------------


class BitLinear(nn.Module):
    """Drop-in replacement for nn.Linear using BitNet b1.58 quantization.

    Weights are stored as full-precision float parameters and ternarized
    only during the forward pass.  Activations are quantized to int8 range.
    Output is rescaled by both weight_scale and act_scale to recover the
    correct magnitude.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        config: BitNetConfig | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_ternary = config.use_ternary_weights if config is not None else True
        self.bits = config.bits if config is not None else 8

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize activations (STE)
        quantized_x, act_scale = quantize_activation(x, bits=self.bits)

        if self.use_ternary:
            # Ternarize weights (STE)
            ternary_W, weight_scale = ternarize_weight(self.weight)
            out = F.linear(quantized_x, ternary_W, self.bias)
            out = out * weight_scale * act_scale
        else:
            out = F.linear(quantized_x * act_scale, self.weight, self.bias)

        return out

    def extra_repr(self) -> str:
        return f"d_in={self.in_features}, d_out={self.out_features}, use_ternary={self.use_ternary}"


# ---------------------------------------------------------------------------
# Model-level utilities
# ---------------------------------------------------------------------------


def apply_bitnet(model: nn.Module, config: BitNetConfig) -> nn.Module:
    """Recursively replace nn.Linear layers with BitLinear in-place.

    Embedding layers (nn.Embedding) are left untouched.  The replacement
    preserves weight data and bias (if present).

    Args:
        model:  any nn.Module.
        config: BitNetConfig to pass to each BitLinear.

    Returns:
        The same model object with Linear layers replaced.
    """
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Embedding):
            # Never replace embeddings
            continue
        elif isinstance(module, nn.Linear):
            has_bias = module.bias is not None
            new_layer = BitLinear(
                module.in_features,
                module.out_features,
                bias=has_bias,
                config=config,
            )
            # Copy existing weight/bias so we don't lose trained values
            with torch.no_grad():
                new_layer.weight.copy_(module.weight)
                if has_bias and new_layer.bias is not None:
                    new_layer.bias.copy_(module.bias)
            setattr(model, name, new_layer)
        else:
            # Recurse into child modules
            apply_bitnet(module, config)
    return model


def compute_effective_bits(model: nn.Module) -> dict[str, float]:
    """Count BitLinear vs nn.Linear layers and estimate compression ratio.

    Compression ratio = bits saved / total original bits, where BitLinear
    weights are treated as ~1.585 bits (log2(3)) and original Linear weights
    are 32-bit floats.

    Returns:
        dict with keys: "n_bitlinear", "n_linear", "compression_ratio".
    """
    n_bitlinear = 0
    n_linear = 0
    bitlinear_params = 0
    linear_params = 0

    for module in model.modules():
        if isinstance(module, BitLinear):
            n_bitlinear += 1
            bitlinear_params += module.weight.numel()
        elif isinstance(module, nn.Linear):
            n_linear += 1
            linear_params += module.weight.numel()

    bits_per_ternary = 1.585  # log2(3)
    bits_per_float = 32.0

    total_params = bitlinear_params + linear_params
    if total_params == 0:
        compression_ratio = 0.0
    else:
        original_bits = total_params * bits_per_float
        compressed_bits = bitlinear_params * bits_per_ternary + linear_params * bits_per_float
        bits_saved = original_bits - compressed_bits
        compression_ratio = bits_saved / original_bits

    return {
        "n_bitlinear": float(n_bitlinear),
        "n_linear": float(n_linear),
        "compression_ratio": compression_ratio,
    }


# ---------------------------------------------------------------------------
# BitNet FFN
# ---------------------------------------------------------------------------


class BitNetFFN(nn.Module):
    """Two-layer feed-forward network using BitLinear layers.

    Architecture: fc1 (d_model → d_ff) → SiLU → fc2 (d_ff → d_model).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        config: BitNetConfig | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = BitNetConfig()
        self.fc1 = BitLinear(d_model, d_ff, bias=False, config=config)
        self.fc2 = BitLinear(d_ff, d_model, bias=False, config=config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(x)))
