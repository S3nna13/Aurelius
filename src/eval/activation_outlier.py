"""Activation outlier detection and analysis for transformer models.

LLMs often have "outlier dimensions" — specific channels with extremely large
activations (e.g., 100x larger than average). These affect quantization quality
and model behavior. This module provides IQR- and z-score-based detection,
per-channel statistics, quantization scale suggestions, and a hook-based
detector that can attach to any nn.Module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OutlierReport:
    """Per-layer outlier analysis report."""

    layer_name: str
    outlier_dims: List[int]          # channel indices with outlier activations
    outlier_scores: List[float]      # score per outlier dim
    mean_activation: float
    max_activation: float
    outlier_ratio: float             # fraction of dims that are outliers
    suggested_scale: float           # suggested per-channel scale for quantization


@dataclass
class ModelOutlierSummary:
    """Aggregate outlier summary across all monitored layers."""

    per_layer_reports: Dict[str, OutlierReport]
    total_outlier_dims: int
    most_problematic_layers: List[str]   # sorted by outlier_ratio descending
    global_outlier_ratio: float


# ---------------------------------------------------------------------------
# Detection utility functions
# ---------------------------------------------------------------------------

def detect_outliers_iqr(
    activations: torch.Tensor,
    threshold: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """IQR-based outlier detection.

    Args:
        activations: (N, d) or (d,) — per-channel stats (absolute max or mean).
        threshold: IQR multiplier; values > Q3 + threshold*IQR are outliers.

    Returns:
        outlier_mask: (d,) bool tensor — True for outlier channels.
        outlier_scores: (d,) float tensor — (val - Q3) / IQR for outliers, 0 elsewhere.
    """
    # Reduce to (d,) per-channel values
    if activations.ndim == 2:
        vals = activations.abs().max(dim=0).values.float()
    else:
        vals = activations.float()

    d = vals.shape[0]

    q1 = torch.quantile(vals, 0.25)
    q3 = torch.quantile(vals, 0.75)
    iqr = q3 - q1

    if iqr.item() < 1e-12:
        # Degenerate case — no spread, no outliers
        return torch.zeros(d, dtype=torch.bool), torch.zeros(d, dtype=torch.float32)

    upper_fence = q3 + threshold * iqr
    outlier_mask = vals > upper_fence

    scores = torch.where(
        outlier_mask,
        (vals - q3) / iqr,
        torch.zeros_like(vals),
    )

    return outlier_mask, scores


def detect_outliers_zscore(
    activations: torch.Tensor,
    threshold: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Z-score outlier detection.

    Args:
        activations: (d,) per-channel values.
        threshold: z-score threshold.

    Returns:
        outlier_mask: (d,) bool tensor.
        outlier_scores: (d,) float tensor — z-scores for outliers, 0 elsewhere.
    """
    vals = activations.float()
    mean = vals.mean()
    std = vals.std(unbiased=False)

    if std.item() < 1e-12:
        d = vals.shape[0]
        return torch.zeros(d, dtype=torch.bool), torch.zeros(d, dtype=torch.float32)

    z = (vals - mean) / std
    outlier_mask = z.abs() > threshold

    scores = torch.where(outlier_mask, z.abs(), torch.zeros_like(z))

    return outlier_mask, scores


def compute_per_channel_stats(
    activation_tensor: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compute per-channel statistics across batch/seq dimensions.

    Args:
        activation_tensor: (N, d) or (B, T, d).

    Returns:
        dict with keys 'mean', 'std', 'max', 'absmax', each of shape (d,).
    """
    t = activation_tensor.float()

    if t.ndim == 2:
        # (N, d) — reduce over N
        mean = t.mean(dim=0)
        std = t.std(dim=0, unbiased=False)
        max_ = t.max(dim=0).values
        absmax = t.abs().max(dim=0).values
    elif t.ndim == 3:
        # (B, T, d) — reduce over B and T
        B, T, d = t.shape
        flat = t.reshape(B * T, d)
        mean = flat.mean(dim=0)
        std = flat.std(dim=0, unbiased=False)
        max_ = flat.max(dim=0).values
        absmax = flat.abs().max(dim=0).values
    else:
        raise ValueError(
            f"activation_tensor must be 2D (N, d) or 3D (B, T, d), got {t.ndim}D"
        )

    return {"mean": mean, "std": std, "max": max_, "absmax": absmax}


def suggest_quantization_scale(
    activations: torch.Tensor,
    n_bits: int = 8,
    outlier_dims: Optional[torch.Tensor] = None,
    outlier_scale_factor: float = 4.0,
) -> torch.Tensor:
    """Suggest per-channel quantization scales.

    Outlier dims get a larger scale (upscaled) to better represent them.
    Non-outlier dims get standard absmax / (2^(n_bits-1) - 1) scale.

    Args:
        activations: (d,) per-channel absmax values.
        n_bits: quantization bit-width.
        outlier_dims: (d,) bool mask — True for outlier channels.
        outlier_scale_factor: multiplier applied to outlier channel scales.

    Returns:
        (d,) scale tensor.
    """
    vals = activations.float()
    max_int = float(2 ** (n_bits - 1) - 1)
    base_scale = vals / max_int

    if outlier_dims is None:
        return base_scale

    scale = torch.where(
        outlier_dims,
        base_scale * outlier_scale_factor,
        base_scale,
    )
    return scale


# ---------------------------------------------------------------------------
# ActivationOutlierDetector
# ---------------------------------------------------------------------------

class ActivationOutlierDetector:
    """Attach hooks to model layers, collect activations, detect outliers.

    Usage::

        detector = ActivationOutlierDetector(model)
        for batch in dataloader:
            detector.collect(batch["input_ids"])
        summary = detector.analyze()
        detector.remove_hooks()
    """

    def __init__(
        self,
        model: nn.Module,
        target_modules: Optional[List[str]] = None,
        method: str = "iqr",
        threshold: float = 3.0,
        n_bits: int = 8,
    ) -> None:
        """
        Args:
            model: The nn.Module to analyze.
            target_modules: Names of submodules to monitor. None = all Linear layers.
            method: Outlier detection method — "iqr" | "zscore".
            threshold: Detection threshold (IQR multiplier or z-score cutoff).
            n_bits: Quantization bit-width for scale suggestions.
        """
        if method not in ("iqr", "zscore"):
            raise ValueError(f"method must be 'iqr' or 'zscore', got '{method}'")

        self.model = model
        self.method = method
        self.threshold = threshold
        self.n_bits = n_bits

        # Determine which modules to hook
        self._target_names: List[str] = []
        if target_modules is not None:
            self._target_names = list(target_modules)
        else:
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    self._target_names.append(name)

        # Storage: layer_name -> list of per-channel absmax tensors (one per forward pass)
        self._absmax_store: Dict[str, List[torch.Tensor]] = {
            n: [] for n in self._target_names
        }
        self._hooks: List[Any] = []
        self._n_samples: int = 0

        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward hooks on target layers."""
        name_to_module: Dict[str, nn.Module] = dict(self.model.named_modules())

        for name in self._target_names:
            if name not in name_to_module:
                continue
            module = name_to_module[name]

            def make_hook(layer_name: str):
                def hook(module: nn.Module, inp: Any, out: Any) -> None:
                    if isinstance(out, torch.Tensor):
                        tensor = out.detach().float()
                    else:
                        # Some modules return tuples
                        tensor = out[0].detach().float()

                    # Compute per-channel absmax
                    if tensor.ndim == 2:
                        absmax = tensor.abs().max(dim=0).values
                    elif tensor.ndim == 3:
                        B, T, d = tensor.shape
                        absmax = tensor.reshape(B * T, d).abs().max(dim=0).values
                    else:
                        absmax = tensor.abs().reshape(-1, tensor.shape[-1]).max(dim=0).values

                    self._absmax_store[layer_name].append(absmax.cpu())

                return hook

            handle = module.register_forward_hook(make_hook(name))
            self._hooks.append(handle)

    def collect(self, input_ids: torch.Tensor) -> None:
        """Run one forward pass and accumulate per-layer activation stats."""
        with torch.no_grad():
            self.model(input_ids)
        self._n_samples += 1

    def analyze(self) -> ModelOutlierSummary:
        """Analyze collected activations and return a ModelOutlierSummary.

        Clears collected data after analysis.
        """
        per_layer_reports: Dict[str, OutlierReport] = {}

        for name in self._target_names:
            stored = self._absmax_store.get(name, [])
            if not stored:
                continue

            # Stack all collected absmax tensors and take channel-wise max
            stacked = torch.stack(stored, dim=0)   # (S, d)
            absmax = stacked.max(dim=0).values      # (d,)

            if self.method == "iqr":
                outlier_mask, scores = detect_outliers_iqr(absmax, self.threshold)
            else:
                outlier_mask, scores = detect_outliers_zscore(absmax, self.threshold)

            outlier_indices = outlier_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
            outlier_score_vals = scores[outlier_mask].tolist()

            n_dims = absmax.shape[0]
            n_outliers = int(outlier_mask.sum().item())
            outlier_ratio = n_outliers / n_dims if n_dims > 0 else 0.0

            mean_act = float(absmax.mean().item())
            max_act = float(absmax.max().item())

            # Suggested scale: mean of per-channel scales
            scale_tensor = suggest_quantization_scale(
                absmax, n_bits=self.n_bits, outlier_dims=outlier_mask
            )
            suggested_scale = float(scale_tensor.mean().item())

            per_layer_reports[name] = OutlierReport(
                layer_name=name,
                outlier_dims=outlier_indices,
                outlier_scores=outlier_score_vals,
                mean_activation=mean_act,
                max_activation=max_act,
                outlier_ratio=outlier_ratio,
                suggested_scale=suggested_scale,
            )

        # Aggregate
        total_outlier_dims = sum(len(r.outlier_dims) for r in per_layer_reports.values())
        sorted_layers = sorted(
            per_layer_reports.keys(),
            key=lambda n: per_layer_reports[n].outlier_ratio,
            reverse=True,
        )

        if per_layer_reports:
            global_outlier_ratio = float(
                sum(r.outlier_ratio for r in per_layer_reports.values())
                / len(per_layer_reports)
            )
        else:
            global_outlier_ratio = 0.0

        # Clear stored data
        for name in self._absmax_store:
            self._absmax_store[name] = []
        self._n_samples = 0

        return ModelOutlierSummary(
            per_layer_reports=per_layer_reports,
            total_outlier_dims=total_outlier_dims,
            most_problematic_layers=sorted_layers,
            global_outlier_ratio=global_outlier_ratio,
        )

    def remove_hooks(self) -> None:
        """Clean up all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    @property
    def n_samples_collected(self) -> int:
        """Number of forward passes collected."""
        return self._n_samples


# ---------------------------------------------------------------------------
# Visualization helper (no plotting — returns stats dict)
# ---------------------------------------------------------------------------

def visualize_outlier_distribution(
    activations: torch.Tensor,
    outlier_mask: torch.Tensor,
) -> Dict[str, Any]:
    """Return statistics for visualization (no actual plotting).

    Args:
        activations: (d,) per-channel values.
        outlier_mask: (d,) bool tensor.

    Returns:
        dict with keys:
            'normal_mean', 'normal_std',
            'outlier_values', 'outlier_indices',
            'histogram_bins', 'histogram_counts'.
    """
    vals = activations.float()
    mask = outlier_mask.bool()

    normal_vals = vals[~mask]
    outlier_vals = vals[mask]

    normal_mean = float(normal_vals.mean().item()) if normal_vals.numel() > 0 else 0.0
    normal_std = float(normal_vals.std(unbiased=False).item()) if normal_vals.numel() > 0 else 0.0

    outlier_indices = mask.nonzero(as_tuple=False).squeeze(-1).tolist()
    outlier_values = outlier_vals.tolist()

    # Simple histogram over all values (20 bins)
    n_bins = 20
    if vals.numel() > 0:
        min_val = float(vals.min().item())
        max_val = float(vals.max().item())
        if abs(max_val - min_val) < 1e-12:
            bin_edges = [min_val + i for i in range(n_bins + 1)]
        else:
            bin_width = (max_val - min_val) / n_bins
            bin_edges = [min_val + i * bin_width for i in range(n_bins + 1)]

        counts = []
        for i in range(n_bins):
            lo = bin_edges[i]
            hi = bin_edges[i + 1]
            if i < n_bins - 1:
                cnt = int(((vals >= lo) & (vals < hi)).sum().item())
            else:
                cnt = int(((vals >= lo) & (vals <= hi)).sum().item())
            counts.append(cnt)
    else:
        bin_edges = []
        counts = []

    return {
        "normal_mean": normal_mean,
        "normal_std": normal_std,
        "outlier_values": outlier_values,
        "outlier_indices": outlier_indices,
        "histogram_bins": bin_edges,
        "histogram_counts": counts,
    }
