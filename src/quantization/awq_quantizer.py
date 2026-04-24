"""AWQ (Activation-aware Weight Quantization) implementation for Aurelius.

Reference: Lin et al. (2023) AWQ: Activation-aware Weight Quantization for LLM
Compression and Acceleration.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# ---------------------------------------------------------------------------
# Registry (populated at module bottom)
# ---------------------------------------------------------------------------
QUANTIZATION_REGISTRY: dict[str, object] = {}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AWQConfig:
    bits: int = 4
    group_size: int = 128
    zero_point: bool = True
    clip_ratio: float = 0.9


# ---------------------------------------------------------------------------
# Scale search
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
        """Return optimal per-channel scale vector.

        Args:
            weight:      (out_features, in_features) weight matrix.
            activations: (n_samples, in_features) calibration activations.

        Returns:
            scales: (out_features,) — one scale per output channel.
        """
        # Per-channel activation magnitude: mean absolute value across samples
        act_mag = activations.abs().mean(dim=0)  # (in_features,)

        # Build grid: 20 evenly-spaced ratios in [0.5, 1.0] of max act magnitude
        max_mag = act_mag.max().clamp(min=1e-8)
        ratios = torch.linspace(0.5, 1.0, self.n_grid, dtype=weight.dtype)
        grid = ratios * max_mag  # (n_grid,)

        out_features = weight.shape[0]
        best_scales = torch.ones(out_features, dtype=weight.dtype)

        for oc in range(out_features):
            w_row = weight[oc]  # (in_features,)
            best_loss = float("inf")
            best_s = 1.0
            for s in grid.tolist():
                # Objective: minimise clipping loss on scaled weight
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
    """AWQ-style quantizer: scale-search + asymmetric INT-N quantization."""

    def __init__(self, config: AWQConfig | None = None) -> None:
        self.config = config if config is not None else AWQConfig()
        self._scale_search = AWQScaleSearch(self.config)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def quantize_layer(
        self,
        weight: torch.Tensor,
        activations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize a weight matrix with AWQ-style per-channel scaling.

        Args:
            weight:      (out_features, in_features)
            activations: (n_samples, in_features)

        Returns:
            q_weight: INT quantized weights (same shape as weight)
            scales:   (out_features,) float32 dequant scales
            zeros:    (out_features,) float32 zero points (INT)
        """
        scales_search = self._scale_search.search_scales(weight, activations)

        bits = self.config.bits
        n_levels = 2 ** bits  # e.g. 16 for 4-bit
        out_features, in_features = weight.shape

        q_weight = torch.zeros_like(weight, dtype=torch.int32)
        quant_scales = torch.zeros(out_features, dtype=weight.dtype)
        zeros = torch.zeros(out_features, dtype=weight.dtype)

        for oc in range(out_features):
            w = weight[oc]  # (in_features,)
            # Apply AWQ scale
            w_scaled = w * scales_search[oc]

            if self.config.zero_point:
                w_min = w_scaled.min()
                w_max = w_scaled.max()
                scale = (w_max - w_min).clamp(min=1e-8) / (n_levels - 1)
                zero = (-w_min / scale).round().clamp(0, n_levels - 1)
            else:
                # Symmetric
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
        """Reconstruct float weights from quantized representation.

        Args:
            q_weight: (out_features, in_features) INT quantized
            scales:   (out_features,)
            zeros:    (out_features,)

        Returns:
            Reconstructed float tensor of same shape as q_weight.
        """
        scales = scales.to(q_weight.dtype if q_weight.is_floating_point() else torch.float32)
        zeros = zeros.to(scales.dtype)
        q_float = q_weight.to(scales.dtype)
        # Broadcast: (out_features, 1)
        s = scales.unsqueeze(1)
        z = zeros.unsqueeze(1)
        return (q_float - z) * s


# ---------------------------------------------------------------------------
# Registry registration
# ---------------------------------------------------------------------------
QUANTIZATION_REGISTRY["awq"] = AWQQuantizer
