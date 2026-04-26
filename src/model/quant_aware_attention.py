"""Quantization-Aware Attention with Straight-Through Estimator.

Simulates integer quantization during forward pass while maintaining
full-precision gradients via STE. Supports per-channel and per-tensor
quantization for weights, activations, and KV cache separately.

References:
    Jacob et al. 2018 (Quantization and Training of Neural Networks)
    Dettmers et al. 2022 (LLM.int8() / bitsandbytes)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# FakeQuantize
# ---------------------------------------------------------------------------


class FakeQuantize(nn.Module):
    """Simulates symmetric quantization with Straight-Through Estimator (STE).

    During the forward pass, weights/activations are mapped to a discrete
    quantization grid and then scaled back to float ("fake quantized").
    During backward, gradients pass through unchanged via STE.

    Args:
        bits: Quantization bit width (4, 8, or 16).
        per_channel: If True, compute scale independently for each channel
            along ``axis``.
        axis: Channel axis used when ``per_channel=True``.
    """

    def __init__(self, bits: int = 8, per_channel: bool = False, axis: int = 0) -> None:
        super().__init__()
        self.bits = bits
        self.per_channel = per_channel
        self.axis = axis

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Compute quantization scale (per-channel or per-tensor)."""
        q_max = 2 ** (self.bits - 1) - 1  # e.g. 127 for 8-bit

        if self.per_channel:
            # Reduce over all axes except the channel axis
            ndim = x.dim()
            reduce_dims = [i for i in range(ndim) if i != self.axis]
            if reduce_dims:
                abs_max = x.abs().amax(dim=reduce_dims, keepdim=True)
            else:
                abs_max = x.abs()
            # Reshape so it broadcasts correctly over x
            scale = abs_max / (q_max + 1e-8)
        else:
            scale = x.abs().max() / (q_max + 1e-8)

        # Avoid divide-by-zero
        scale = scale.clamp(min=1e-8)
        return scale

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fake-quantize ``x`` and return a tensor with the same dtype/shape.

        The quantization is:
            scale  = max(|x|) / (2^(bits-1) - 1)
            x_int  = round(x / scale)        [clipped to valid range]
            x_q    = x_int * scale            [dequantize back to float]

        STE is applied so that gradients pass through unchanged.
        """
        q_min = -(2 ** (self.bits - 1))
        q_max = 2 ** (self.bits - 1) - 1

        scale = self._compute_scale(x)

        # Quantize
        x_int = torch.clamp(torch.round(x / scale), q_min, q_max)
        x_q_raw = x_int * scale

        # STE: in forward use x_q, in backward gradient = 1
        x_q = x + (x_q_raw - x).detach()
        return x_q

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def effective_bits(self) -> int:
        """Return the configured bit width."""
        return self.bits

    def extra_repr(self) -> str:
        return f"bits={self.bits}, per_channel={self.per_channel}, axis={self.axis}"


# ---------------------------------------------------------------------------
# QuantConfig
# ---------------------------------------------------------------------------


@dataclass
class QuantConfig:
    """Configuration for quantization-aware attention.

    Args:
        weight_bits: Bit width for weight quantization.
        activation_bits: Bit width for activation quantization.
        kv_bits: Bit width for KV-cache quantization.
        quantize_weights: Enable weight quantization.
        quantize_activations: Enable activation quantization.
        quantize_kv: Enable KV-cache quantization.
    """

    weight_bits: int = 8
    activation_bits: int = 8
    kv_bits: int = 8
    quantize_weights: bool = True
    quantize_activations: bool = False
    quantize_kv: bool = False


# ---------------------------------------------------------------------------
# QuantAwareAttention
# ---------------------------------------------------------------------------


class QuantAwareAttention(nn.Module):
    """Standard multi-head attention with optional fake quantization.

    Quantization can be applied independently to:
    * projection weights (before matmul)
    * post-projection activations
    * K/V tensors before attention computation

    Args:
        d_model: Model (embedding) dimension.
        n_heads: Number of attention heads. Must evenly divide ``d_model``.
        config: :class:`QuantConfig` instance; defaults to ``QuantConfig()``.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        config: QuantConfig | None = None,
    ) -> None:
        super().__init__()

        if config is None:
            config = QuantConfig()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"  # noqa: S101

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head**-0.5
        self.config = config

        # Projection layers
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Optional quantizers
        self.weight_quantizer = (
            FakeQuantize(config.weight_bits, per_channel=True) if config.quantize_weights else None
        )
        self.act_quantizer = (
            FakeQuantize(config.activation_bits) if config.quantize_activations else None
        )
        self.kv_quantizer = FakeQuantize(config.kv_bits) if config.quantize_kv else None

    # ------------------------------------------------------------------
    # Quantization helpers
    # ------------------------------------------------------------------

    def _quantize_weight(self, w: torch.Tensor) -> torch.Tensor:
        """Apply weight quantizer if enabled, else identity."""
        if self.weight_quantizer is not None:
            return self.weight_quantizer(w)
        return w

    def _quantize_act(self, x: torch.Tensor) -> torch.Tensor:
        """Apply activation quantizer if enabled, else identity."""
        if self.act_quantizer is not None:
            return self.act_quantizer(x)
        return x

    def _quantize_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Apply KV quantizer if enabled, else identity."""
        if self.kv_quantizer is not None:
            return self.kv_quantizer(x)
        return x

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute quantization-aware multi-head attention.

        Args:
            x: Input tensor of shape ``(B, T, d_model)``.

        Returns:
            Output tensor of shape ``(B, T, d_model)``.
        """
        B, T, _ = x.shape

        # ---- Projections with quantized weights -------------------------
        q = x @ self._quantize_weight(self.q_proj.weight).T + self.q_proj.bias
        k = x @ self._quantize_weight(self.k_proj.weight).T + self.k_proj.bias
        v = x @ self._quantize_weight(self.v_proj.weight).T + self.v_proj.bias

        # ---- Activation quantization ------------------------------------
        q = self._quantize_act(q)
        k = self._quantize_act(k)
        v = self._quantize_act(v)

        # ---- Reshape to (B, n_heads, T, d_head) -------------------------
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # ---- KV quantization -------------------------------------------
        k = self._quantize_kv(k)
        v = self._quantize_kv(v)

        # ---- Scaled dot-product attention --------------------------------
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, T)
        attn = F.softmax(attn, dim=-1)
        out = attn @ v  # (B, H, T, d_head)

        # ---- Merge heads -------------------------------------------------
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        # ---- Output projection with quantized weight --------------------
        out = out @ self._quantize_weight(self.out_proj.weight).T + self.out_proj.bias

        return out


# ---------------------------------------------------------------------------
# QuantCalibrator
# ---------------------------------------------------------------------------


class QuantCalibrator:
    """Calibrates quantization scales on representative data.

    Iterates over all :class:`FakeQuantize` submodules in a model and
    records their bit widths.  Calibration is performed by running a
    forward pass (fake quantizers observe the statistics of the data
    during the forward pass).

    Args:
        model: An ``nn.Module`` that may contain :class:`FakeQuantize` layers.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        # Collect all FakeQuantize submodules with their names
        self._quantizers: dict[str, FakeQuantize] = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, FakeQuantize)
        }

    def calibrate(self, x: torch.Tensor) -> None:
        """Run a forward pass to activate all fake quantizers.

        In a real QAT pipeline this would accumulate running min/max
        statistics.  Here we simply trigger the forward pass so every
        :class:`FakeQuantize` layer sees the calibration data.

        Args:
            x: Representative input tensor (e.g. a calibration batch).
        """
        with torch.no_grad():
            self.model(x)

    def get_quantizer_stats(self) -> dict[str, int]:
        """Return a mapping of quantizer name to bit width.

        Returns:
            Dict mapping each FakeQuantize submodule's dotted name to its
            ``bits`` attribute.
        """
        return {name: q.bits for name, q in self._quantizers.items()}
