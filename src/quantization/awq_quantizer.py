"""AWQ (Activation-aware Weight Quantization) implementation for Aurelius.

Reference: Lin et al. (2023) AWQ: Activation-aware Weight Quantization for LLM
Compression and Acceleration (arXiv:2306.00978).

This module keeps backwards compatibility with the original Aurelius AWQ
helpers (``AWQScaleSearch`` + ``AWQQuantizer.quantize_layer`` /
``AWQQuantizer.dequantize``) while adding the richer API required for the
latest benchmark cycle:

* :class:`ActivationStats` - per-channel activation magnitudes
* :meth:`AWQQuantizer.collect_activation_stats`
* :meth:`AWQQuantizer.compute_scale_factor`
* :meth:`AWQQuantizer.quantize_with_scales`
* :meth:`AWQQuantizer.reconstruction_error`
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# ---------------------------------------------------------------------------
# Registries (populated at module bottom)
# ---------------------------------------------------------------------------
QUANTIZATION_REGISTRY: dict[str, object] = {}
AWQ_QUANTIZER_REGISTRY: dict[str, type] = {}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class AWQConfig:
    """Configuration for :class:`AWQQuantizer`.

    ``clip_ratio`` is retained for backwards compatibility with the earlier
    Aurelius AWQ scale-search routine; ``version`` identifies the AWQ kernel
    variant (``"gemm"`` or ``"gemv"``).
    """

    bits: int = 4
    group_size: int = 128
    zero_point: bool = True
    version: str = "gemm"
    clip_ratio: float = 0.9


@dataclass(frozen=True)
class ActivationStats:
    """Summary statistics of activations passing through a linear layer."""

    channel_scales: torch.Tensor
    max_activations: torch.Tensor


# ---------------------------------------------------------------------------
# Scale search (legacy API, retained for existing call sites)
# ---------------------------------------------------------------------------


class AWQScaleSearch:
    """Grid-search for the per-channel scale that best preserves weight range."""

    def __init__(self, config: AWQConfig | None = None, n_grid: int = 20) -> None:
        self.config = config if config is not None else AWQConfig()
        self.n_grid = n_grid

    def search_scales(
        self,
        weight: torch.Tensor,
        activations: torch.Tensor,
    ) -> torch.Tensor:
        """Return optimal per-channel scale vector (one per output channel)."""
        act_mag = activations.abs().mean(dim=0)  # (in_features,)

        max_mag = act_mag.max().clamp(min=1e-8)
        ratios = torch.linspace(0.5, 1.0, self.n_grid, dtype=weight.dtype)
        grid = ratios * max_mag  # (n_grid,)

        out_features = weight.shape[0]
        best_scales = torch.ones(out_features, dtype=weight.dtype)

        for oc in range(out_features):
            w_row = weight[oc]
            best_loss = float("inf")
            best_s = 1.0
            for s in grid.tolist():
                loss = (w_row * s).abs().max().item()
                if loss < best_loss:
                    best_loss = loss
                    best_s = s
            best_scales[oc] = best_s

        return best_scales


# ---------------------------------------------------------------------------
# Full quantizer
# ---------------------------------------------------------------------------


class AWQQuantizer:
    """AWQ quantizer: activation-aware scaling + asymmetric INT-N quantization."""

    def __init__(self, config: AWQConfig | None = None) -> None:
        self.config = config if config is not None else AWQConfig()
        self._scale_search = AWQScaleSearch(self.config)

    # ------------------------------------------------------------------
    # New activation-aware API
    # ------------------------------------------------------------------

    def collect_activation_stats(
        self,
        activations: torch.Tensor,
    ) -> ActivationStats:
        """Compute per-channel activation statistics.

        Args:
            activations: tensor of shape ``(..., channels)``. The final
                dimension is assumed to be the channel axis.

        Returns:
            :class:`ActivationStats` with ``channel_scales`` equal to the mean
            absolute activation per channel and ``max_activations`` equal to
            the per-channel maximum absolute value.
        """
        if activations.ndim < 1:
            raise ValueError("activations must have at least one dimension")

        flat = activations.reshape(-1, activations.shape[-1]).abs()
        channel_scales = flat.mean(dim=0)
        max_activations = flat.max(dim=0).values
        return ActivationStats(
            channel_scales=channel_scales,
            max_activations=max_activations,
        )

    def compute_scale_factor(
        self,
        weight: torch.Tensor,
        stats: ActivationStats,
    ) -> torch.Tensor:
        """Compute AWQ per-channel scale factor.

        ``scale = (act_scale^0.5 / |W|.max(dim=0)^0.5).clamp(min=1e-4)``.
        """
        act_scale = stats.channel_scales.to(weight.dtype)
        w_max = weight.abs().max(dim=0).values.to(weight.dtype).clamp(min=1e-8)
        return (act_scale.pow(0.5) / w_max.pow(0.5)).clamp(min=1e-4)

    def quantize_with_scales(
        self,
        weight: torch.Tensor,
        scale_factor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize ``weight`` after applying a per-channel ``scale_factor``.

        Returns ``(w_int, scale, zero)``.
        """
        if scale_factor.numel() != weight.shape[-1]:
            raise ValueError(
                "scale_factor must have one element per input channel "
                f"(weight in_features={weight.shape[-1]}, got {scale_factor.numel()})"
            )

        bits = self.config.bits
        n_levels = 1 << bits  # 2^bits

        # Apply AWQ per-channel scaling along the input-feature axis.
        w_scaled = weight * scale_factor.view(1, -1).to(weight.dtype)

        if self.config.zero_point:
            qmax = n_levels - 1
            w_min = w_scaled.min(dim=1, keepdim=True).values
            w_max = w_scaled.max(dim=1, keepdim=True).values
            scale = (w_max - w_min).clamp(min=1e-8) / qmax
            zero = (-w_min / scale).round().clamp(0, qmax)
            w_int = (w_scaled / scale + zero).round().clamp(0, qmax).to(torch.int32)
            return w_int, scale.squeeze(1), zero.squeeze(1)

        # Symmetric
        qmax = (n_levels // 2) - 1
        max_abs = w_scaled.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
        scale = max_abs / qmax
        zero = torch.zeros_like(scale)
        w_int = (w_scaled / scale).round().clamp(-(n_levels // 2), qmax).to(torch.int32)
        return w_int, scale.squeeze(1), zero.squeeze(1)

    def reconstruction_error(
        self,
        original: torch.Tensor,
        w_int: torch.Tensor,
        scale: torch.Tensor,
        zero: torch.Tensor,
    ) -> float:
        """Return MSE between ``original`` and the dequantized weights."""
        scale_b = scale.view(-1, 1).to(torch.float32)
        zero_b = zero.view(-1, 1).to(torch.float32)
        recon = (w_int.to(torch.float32) - zero_b) * scale_b
        return float(((recon - original.to(torch.float32)) ** 2).mean().item())

    # ------------------------------------------------------------------
    # Legacy API (kept for backwards compatibility)
    # ------------------------------------------------------------------

    def quantize_layer(
        self,
        weight: torch.Tensor,
        activations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Legacy: scale-search + INT-N quantization (output-channel groups)."""
        scales_search = self._scale_search.search_scales(weight, activations)

        bits = self.config.bits
        n_levels = 1 << bits
        out_features, _ = weight.shape

        q_weight = torch.zeros_like(weight, dtype=torch.int32)
        quant_scales = torch.zeros(out_features, dtype=weight.dtype)
        zeros = torch.zeros(out_features, dtype=weight.dtype)

        for oc in range(out_features):
            w = weight[oc]
            w_scaled = w * scales_search[oc]

            if self.config.zero_point:
                w_min = w_scaled.min()
                w_max = w_scaled.max()
                scale = (w_max - w_min).clamp(min=1e-8) / (n_levels - 1)
                zero = (-w_min / scale).round().clamp(0, n_levels - 1)
            else:
                w_max_abs = w_scaled.abs().max().clamp(min=1e-8)
                scale = w_max_abs / ((n_levels // 2) - 1)
                zero = torch.tensor(0.0, dtype=weight.dtype)

            q = (w_scaled / scale + zero).round().clamp(0, n_levels - 1).to(torch.int32)
            q_weight[oc] = q
            quant_scales[oc] = scale
            zeros[oc] = zero

        return q_weight, quant_scales, zeros

    def dequantize(
        self,
        q_weight: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
    ) -> torch.Tensor:
        """Legacy dequantization for :meth:`quantize_layer` outputs."""
        scales = scales.to(q_weight.dtype if q_weight.is_floating_point() else torch.float32)
        zeros = zeros.to(scales.dtype)
        q_float = q_weight.to(scales.dtype)
        s = scales.unsqueeze(1)
        z = zeros.unsqueeze(1)
        return (q_float - z) * s


# ---------------------------------------------------------------------------
# Registry registration
# ---------------------------------------------------------------------------
QUANTIZATION_REGISTRY["awq"] = AWQQuantizer
AWQ_QUANTIZER_REGISTRY["default"] = AWQQuantizer
