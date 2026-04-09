"""Advanced quantization-aware training: LSQ learned step sizes, mixed-precision, and STE gradients."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class AdvancedQATConfig:
    """Configuration for advanced QAT with LSQ and mixed-precision support."""

    bits: int = 8
    per_channel: bool = True
    symmetric: bool = True
    use_lsq: bool = True  # learned step size quantization
    lsq_init_factor: float = 2.0  # LSQ step size init: 2 * mean(|w|) / sqrt(2^bits - 1)
    clip_val: float = 6.0  # activation clip value
    mixed_precision_layers: list[str] = field(
        default_factory=lambda: ["lm_head"]
    )  # keep in fp16
    warmup_steps: int = 1000  # steps before enabling quantization


def round_ste(x: Tensor) -> Tensor:
    """Straight-through estimator for rounding.

    Forward: round(x)
    Backward: identity (gradient passes through unchanged)
    """
    return x + (x.round() - x).detach()


def clamp_ste(x: Tensor, min_val: float, max_val: float) -> Tensor:
    """STE clamp: forward = clamp, backward = identity (gradient passes through)."""
    return x + (x.clamp(min_val, max_val) - x).detach()


class LSQQuantizer(nn.Module):
    """Learned Step-size Quantization (Esser et al. 2020).

    Learns the quantization step size jointly with model weights via
    straight-through gradient estimation.

    Args:
        bits: Number of quantization bits.
        per_channel: Whether to use per-channel step sizes.
        n_channels: Number of channels (used when per_channel=True).
    """

    def __init__(self, bits: int, per_channel: bool, n_channels: int = 1) -> None:
        super().__init__()
        self.bits = bits
        self.per_channel = per_channel
        self.n_levels = 2**bits - 1

        if per_channel:
            self.step_size = nn.Parameter(torch.ones(n_channels) * 0.1)
        else:
            self.step_size = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor) -> Tensor:
        """Quantize x using learned step size with STE gradients."""
        step = self.step_size

        if self.per_channel and step.dim() == 1:
            # Reshape step_size for broadcasting over weight: (n_channels, 1)
            if x.dim() >= 2:
                shape = [-1] + [1] * (x.dim() - 1)
                step = step.view(shape)

        # Clamp step size to be positive
        step = step.abs().clamp(min=1e-8)

        half = self.n_levels // 2
        x_scaled = x / step
        x_rounded = round_ste(x_scaled)
        x_clamped = clamp_ste(x_rounded, -half, half)
        return x_clamped * step

    def initialize_from_weights(self, weights: Tensor) -> None:
        """Initialize step size from weight statistics (LSQ init).

        Sets step_size = 2 * mean(|w|) / sqrt(n_levels)
        """
        with torch.no_grad():
            init_val = 2.0 * weights.abs().mean() / math.sqrt(self.n_levels)
            init_val = init_val.clamp(min=1e-8)
            if self.per_channel and self.step_size.dim() == 1:
                # Per-channel: compute per output channel
                if weights.dim() >= 2:
                    per_ch = (
                        2.0
                        * weights.abs().mean(dim=tuple(range(1, weights.dim())))
                        / math.sqrt(self.n_levels)
                    ).clamp(min=1e-8)
                    if per_ch.shape == self.step_size.shape:
                        self.step_size.copy_(per_ch)
                        return
            self.step_size.fill_(init_val.item())


class FakeQuantLinear(nn.Module):
    """Linear layer with fake quantization of weights and activations.

    During training with quantization enabled, weights and activations
    are fake-quantized using LSQ quantizers before the linear op.

    Args:
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        config: Advanced QAT configuration.
        bias: Whether to include a bias term.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: AdvancedQATConfig,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        n_channels = out_features if config.per_channel else 1
        self.weight_quantizer = LSQQuantizer(
            bits=config.bits,
            per_channel=config.per_channel,
            n_channels=n_channels,
        )
        self.act_quantizer = LSQQuantizer(
            bits=config.bits,
            per_channel=False,  # activations are quantized per-tensor
            n_channels=1,
        )

        self.enabled = False  # start disabled

    def enable_quantization(self) -> None:
        """Enable fake quantization in the forward pass."""
        self.enabled = True

    def disable_quantization(self) -> None:
        """Disable fake quantization (fall back to standard linear)."""
        self.enabled = False

    def forward(self, x: Tensor) -> Tensor:
        """Compute linear with optional fake quantization."""
        if self.enabled:
            w_q = self.weight_quantizer(self.weight)
            x_q = self.act_quantizer(x)
            return F.linear(x_q, w_q, self.bias)
        return F.linear(x, self.weight, self.bias)


class MixedPrecisionScheduler:
    """Control which layers are quantized during training.

    Enables quantization after a warmup period, skipping layers
    specified in mixed_precision_layers.

    Args:
        model: The model containing FakeQuantLinear layers.
        config: Advanced QAT configuration.
    """

    def __init__(self, model: nn.Module, config: AdvancedQATConfig) -> None:
        self.model = model
        self.config = config
        self._quantized_layers: list[str] = []

    def step(self, current_step: int) -> None:
        """Enable quantization after warmup_steps, skipping fp16 layers."""
        if current_step < self.config.warmup_steps:
            return

        for name, module in self.model.named_modules():
            if not isinstance(module, FakeQuantLinear):
                continue
            # Skip layers that should remain in mixed precision
            if any(mp_name in name for mp_name in self.config.mixed_precision_layers):
                continue
            if not module.enabled:
                module.enable_quantization()
                if name not in self._quantized_layers:
                    self._quantized_layers.append(name)

    def get_quantized_layers(self) -> list[str]:
        """Return names of currently quantized layers."""
        return list(self._quantized_layers)


def convert_to_fake_quant(model: nn.Module, config: AdvancedQATConfig) -> nn.Module:
    """Replace nn.Linear layers with FakeQuantLinear (except mixed_precision_layers).

    Initializes LSQ step sizes from existing weight values.

    Args:
        model: Model to convert.
        config: Advanced QAT configuration.

    Returns:
        Modified model with FakeQuantLinear layers.
    """
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        # Skip mixed precision layers
        if any(mp in name for mp in config.mixed_precision_layers):
            continue

        in_features = module.in_features
        out_features = module.out_features
        has_bias = module.bias is not None

        fq_linear = FakeQuantLinear(
            in_features=in_features,
            out_features=out_features,
            config=config,
            bias=has_bias,
        )

        # Copy existing weights and bias
        with torch.no_grad():
            fq_linear.weight.copy_(module.weight)
            if has_bias and module.bias is not None:
                fq_linear.bias.copy_(module.bias)  # type: ignore[union-attr]

        # Initialize LSQ step sizes from the actual weights
        fq_linear.weight_quantizer.initialize_from_weights(module.weight)

        # Navigate to parent module and replace
        parts = name.rsplit(".", 1)
        if len(parts) == 1:
            parent, attr = model, parts[0]
        else:
            parent = model
            for part in parts[0].split("."):
                parent = getattr(parent, part)
            attr = parts[1]

        setattr(parent, attr, fq_linear)

    return model
