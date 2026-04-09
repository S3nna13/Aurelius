"""Activation anomaly detection for debugging transformer models.

Detects unusual activation patterns that may indicate training issues such as
dead neurons, exploding activations, NaN/Inf values, and statistical outliers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration and report dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AnomalyConfig:
    """Configuration for activation anomaly detection."""

    z_score_threshold: float = 3.0       # z-score for outlier detection
    dead_neuron_threshold: float = 0.01  # fraction of activations near zero
    exploding_threshold: float = 100.0   # absolute value threshold
    n_calibration_samples: int = 10      # samples to compute baseline stats
    layers_to_monitor: list[int] = field(default_factory=lambda: [0, 1])


@dataclass
class AnomalyReport:
    """A single detected anomaly in layer activations."""

    layer_idx: int
    anomaly_type: str        # "dead" | "exploding" | "outlier" | "nan" | "inf"
    severity: float          # 0-1
    description: str
    affected_fraction: float


# ---------------------------------------------------------------------------
# Detection utility functions
# ---------------------------------------------------------------------------

def detect_nan_inf(activations: Tensor) -> tuple[bool, bool]:
    """Check for NaN and Inf in activations.

    Returns:
        (has_nan, has_inf) — True if any element is NaN / Inf respectively.
    """
    has_nan = bool(torch.isnan(activations).any().item())
    has_inf = bool(torch.isinf(activations).any().item())
    return has_nan, has_inf


def detect_exploding_activations(
    activations: Tensor,
    threshold: float = 100.0,
) -> float:
    """Return fraction of values exceeding *threshold* in absolute value."""
    total = activations.numel()
    if total == 0:
        return 0.0
    exceeding = (activations.abs() > threshold).sum().item()
    return float(exceeding) / total


def detect_dead_neurons(
    activations: Tensor,     # (B, T, D)
    threshold: float = 0.01,
) -> float:
    """Compute fraction of neurons (D dimension) that are almost always near zero.

    A neuron is "dead" if |mean_activation| < threshold averaged over B and T.

    Returns:
        Fraction of dead neurons (scalar in [0, 1]).
    """
    if activations.ndim != 3:
        # Gracefully handle non-3D inputs by flattening to (..., D)
        activations = activations.reshape(-1, activations.shape[-1])
        mean_per_neuron = activations.abs().mean(dim=0)
    else:
        # Mean over B and T, keeping D
        mean_per_neuron = activations.abs().mean(dim=(0, 1))

    n_neurons = mean_per_neuron.numel()
    if n_neurons == 0:
        return 0.0
    dead = (mean_per_neuron < threshold).sum().item()
    return float(dead) / n_neurons


def compute_activation_statistics(
    activations: Tensor,     # (B, T, D)
) -> dict[str, float]:
    """Compute summary statistics over all elements.

    Returns:
        dict with keys: "mean", "std", "min", "max", "kurtosis", "skewness"
        kurtosis = E[(x - mean)^4] / std^4 - 3  (excess kurtosis)
        skewness = E[(x - mean)^3] / std^3
    """
    flat = activations.float().reshape(-1)
    mean = flat.mean().item()
    std = flat.std(unbiased=False).item()
    min_val = flat.min().item()
    max_val = flat.max().item()

    if std < 1e-12:
        kurtosis = 0.0
        skewness = 0.0
    else:
        centered = flat - mean
        kurtosis = float(((centered ** 4).mean() / (std ** 4)) - 3.0)
        skewness = float((centered ** 3).mean() / (std ** 3))

    return {
        "mean": float(mean),
        "std": float(std),
        "min": float(min_val),
        "max": float(max_val),
        "kurtosis": kurtosis,
        "skewness": skewness,
    }


def detect_outlier_activations(
    activations: Tensor,
    baseline_mean: float,
    baseline_std: float,
    z_threshold: float = 3.0,
) -> float:
    """Fraction of activations more than z_threshold std deviations from baseline_mean."""
    if baseline_std < 1e-12:
        return 0.0
    z_scores = (activations.float() - baseline_mean).abs() / baseline_std
    total = activations.numel()
    if total == 0:
        return 0.0
    outliers = (z_scores > z_threshold).sum().item()
    return float(outliers) / total


def compute_layer_similarity(
    activations_a: Tensor,   # (B, T, D)
    activations_b: Tensor,   # (B, T, D)
) -> float:
    """Mean cosine similarity between corresponding (B*T) activation vectors."""
    # Flatten to (N, D)
    a = activations_a.float().reshape(-1, activations_a.shape[-1])
    b = activations_b.float().reshape(-1, activations_b.shape[-1])

    a_norm = a / (a.norm(dim=-1, keepdim=True) + 1e-12)
    b_norm = b / (b.norm(dim=-1, keepdim=True) + 1e-12)

    cos_sim = (a_norm * b_norm).sum(dim=-1)  # (N,)
    return float(cos_sim.mean().item())


# ---------------------------------------------------------------------------
# ActivationMonitor
# ---------------------------------------------------------------------------

class ActivationMonitor:
    """Monitor model activations during forward pass via hooks."""

    def __init__(self, model: nn.Module, cfg: AnomalyConfig) -> None:
        self.model = model
        self.cfg = cfg
        self._hooks: list[Any] = []
        self._activations: dict[int, list[Tensor]] = {}

    def start_monitoring(self) -> None:
        """Register forward hooks on cfg.layers_to_monitor."""
        self._hooks.clear()
        self._activations.clear()

        for layer_idx in self.cfg.layers_to_monitor:
            # Capture layer_idx in closure
            def make_hook(idx: int):
                def hook(module: nn.Module, inp: Any, out: Any) -> None:
                    # TransformerBlock returns (hidden_state, kv) — capture tensor
                    if isinstance(out, Tensor):
                        captured = out.detach()
                    else:
                        captured = out[0].detach()
                    self._activations.setdefault(idx, []).append(captured)
                return hook

            layer = self.model.layers[layer_idx]
            handle = layer.register_forward_hook(make_hook(layer_idx))
            self._hooks.append(handle)

    def stop_monitoring(self) -> None:
        """Remove all hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def get_activations(self) -> dict[int, list[Tensor]]:
        """Return {layer_idx: [captured tensors]} from hooks."""
        return self._activations

    def run_forward(self, input_ids: Tensor) -> None:
        """Run model forward pass while monitoring."""
        with torch.no_grad():
            self.model(input_ids)

    def clear(self) -> None:
        """Clear captured activations."""
        self._activations.clear()


# ---------------------------------------------------------------------------
# AnomalyDetector
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """Detect and report activation anomalies."""

    def __init__(self, cfg: AnomalyConfig) -> None:
        self.cfg = cfg
        # Per-layer baseline: {layer_idx: {"mean": float, "std": float}}
        self._baseline: dict[int, dict[str, float]] = {}

    def calibrate(self, monitor: ActivationMonitor, input_ids: Tensor) -> None:
        """Run n_calibration_samples forward passes to establish baseline statistics.

        Stores per-layer mean/std for later outlier detection.
        """
        monitor.start_monitoring()

        all_activations: dict[int, list[Tensor]] = {}

        for _ in range(self.cfg.n_calibration_samples):
            monitor.clear()
            monitor.run_forward(input_ids)
            for layer_idx, tensors in monitor.get_activations().items():
                all_activations.setdefault(layer_idx, []).extend(tensors)

        monitor.stop_monitoring()
        monitor.clear()

        # Compute baseline stats per layer
        self._baseline.clear()
        for layer_idx, tensors in all_activations.items():
            combined = torch.cat([t.float().reshape(-1) for t in tensors])
            self._baseline[layer_idx] = {
                "mean": float(combined.mean().item()),
                "std": float(combined.std(unbiased=False).item()),
            }

    def analyze(self, monitor: ActivationMonitor, input_ids: Tensor) -> list[AnomalyReport]:
        """Run forward pass, analyze activations, return list of detected anomalies.

        Checks: NaN/Inf, exploding, dead neurons, outliers.
        """
        monitor.start_monitoring()
        monitor.clear()
        monitor.run_forward(input_ids)
        activations_map = monitor.get_activations()
        monitor.stop_monitoring()

        reports: list[AnomalyReport] = []

        for layer_idx in self.cfg.layers_to_monitor:
            tensors = activations_map.get(layer_idx, [])
            if not tensors:
                continue

            # Concatenate along batch/time — take first captured tensor
            act = tensors[0]  # (B, T, D)

            # 1. NaN / Inf
            has_nan, has_inf = detect_nan_inf(act)
            if has_nan:
                reports.append(AnomalyReport(
                    layer_idx=layer_idx,
                    anomaly_type="nan",
                    severity=1.0,
                    description=f"Layer {layer_idx}: NaN values detected in activations.",
                    affected_fraction=float(torch.isnan(act).float().mean().item()),
                ))
            if has_inf:
                reports.append(AnomalyReport(
                    layer_idx=layer_idx,
                    anomaly_type="inf",
                    severity=1.0,
                    description=f"Layer {layer_idx}: Inf values detected in activations.",
                    affected_fraction=float(torch.isinf(act).float().mean().item()),
                ))

            # Skip further checks if NaN/Inf present (metrics would be unreliable)
            if has_nan or has_inf:
                continue

            # 2. Exploding activations
            exploding_frac = detect_exploding_activations(act, self.cfg.exploding_threshold)
            if exploding_frac > 0.0:
                severity = min(1.0, exploding_frac * 10.0)
                reports.append(AnomalyReport(
                    layer_idx=layer_idx,
                    anomaly_type="exploding",
                    severity=severity,
                    description=(
                        f"Layer {layer_idx}: {exploding_frac:.2%} of activations exceed "
                        f"threshold {self.cfg.exploding_threshold}."
                    ),
                    affected_fraction=exploding_frac,
                ))

            # 3. Dead neurons
            dead_frac = detect_dead_neurons(act, self.cfg.dead_neuron_threshold)
            if dead_frac > 0.05:  # report only if more than 5% dead
                severity = min(1.0, dead_frac)
                reports.append(AnomalyReport(
                    layer_idx=layer_idx,
                    anomaly_type="dead",
                    severity=severity,
                    description=f"Layer {layer_idx}: {dead_frac:.2%} of neurons are dead.",
                    affected_fraction=dead_frac,
                ))

            # 4. Outlier activations (only if calibrated)
            if layer_idx in self._baseline:
                bl = self._baseline[layer_idx]
                outlier_frac = detect_outlier_activations(
                    act,
                    bl["mean"],
                    bl["std"],
                    self.cfg.z_score_threshold,
                )
                if outlier_frac > 0.01:  # report only if more than 1% outliers
                    severity = min(1.0, outlier_frac * 5.0)
                    reports.append(AnomalyReport(
                        layer_idx=layer_idx,
                        anomaly_type="outlier",
                        severity=severity,
                        description=(
                            f"Layer {layer_idx}: {outlier_frac:.2%} of activations are "
                            f"statistical outliers (z>{self.cfg.z_score_threshold})."
                        ),
                        affected_fraction=outlier_frac,
                    ))

        monitor.clear()
        return reports

    def summarize(self, reports: list[AnomalyReport]) -> dict[str, int | float]:
        """Return summary statistics over a list of anomaly reports."""
        n_nan = sum(1 for r in reports if r.anomaly_type == "nan")
        n_exploding = sum(1 for r in reports if r.anomaly_type == "exploding")
        n_dead = sum(1 for r in reports if r.anomaly_type == "dead")
        mean_severity = (
            float(sum(r.severity for r in reports) / len(reports)) if reports else 0.0
        )
        return {
            "n_anomalies": len(reports),
            "n_nan": n_nan,
            "n_exploding": n_exploding,
            "n_dead": n_dead,
            "mean_severity": mean_severity,
        }
