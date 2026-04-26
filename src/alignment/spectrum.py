"""Spectrum layer selection for LoRA (arXiv:2406.06623).

Selects which weight matrices are the best targets for LoRA adaptation
by computing the signal-to-noise ratio of their singular value spectra.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn


class LayerSNR(NamedTuple):
    name: str
    snr: float
    module_type: str  # e.g. "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", etc.


def compute_snr(weight: torch.Tensor) -> float:
    """Compute the signal-to-noise ratio of a weight matrix's singular value spectrum.

    Uses the Marchenko-Pastur distribution to estimate the noise floor.
    Signal = sum of singular values above the noise threshold.
    Noise = sum of singular values at or below the noise threshold.
    SNR is normalized by the largest singular value for comparability across layers.

    Args:
        weight: 2-D weight tensor (out_features, in_features).

    Returns:
        SNR value. Higher = more learned signal = better LoRA target.
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected 2-D weight, got shape {weight.shape}")

    S = torch.linalg.svdvals(weight.float())  # descending order

    # Robust noise floor via IQR (Marchenko-Pastur inspired)
    q75 = torch.quantile(S, 0.75).item()
    q25 = torch.quantile(S, 0.25).item()
    sigma = (q75 - q25) / 1.3489

    # Marchenko-Pastur upper edge: sigma * (1 + sqrt(gamma))
    # gamma = n_rows / n_cols (aspect ratio)
    gamma = weight.shape[0] / weight.shape[1]
    epsilon = sigma * (1.0 + math.sqrt(gamma))

    signal = S[S > epsilon].sum().item()
    noise = S[S <= epsilon].sum().item()

    if noise == 0.0:
        return float("inf")

    # Normalize by top singular value for cross-layer comparability
    return (signal / noise) / S[0].item()


class SpectrumSelector:
    """Select the best LoRA target layers by SNR.

    Groups weight matrices by their module-type suffix (e.g., "q_proj") and
    selects the top `top_k_fraction` per group, so each module type is
    represented proportionally.

    Args:
        model: The transformer model to analyze.
        top_k_fraction: Fraction of layers per group to select (0 < f <= 1).
    """

    def __init__(self, model: nn.Module, top_k_fraction: float = 0.25) -> None:
        if not 0 < top_k_fraction <= 1.0:
            raise ValueError(f"top_k_fraction must be in (0, 1], got {top_k_fraction}")
        self.model = model
        self.top_k_fraction = top_k_fraction

    def _module_type(self, name: str) -> str:
        """Extract the leaf module-type suffix from a parameter name.

        e.g. 'layers.0.attn.q_proj.weight' -> 'q_proj'
        """
        # strip '.weight' and take the last component
        parts = name.replace(".weight", "").split(".")
        return parts[-1]

    def compute_all_snrs(self) -> list[LayerSNR]:
        """Compute SNR for every 2-D weight matrix in the model."""
        results: list[LayerSNR] = []
        for name, param in self.model.named_parameters():
            if param.ndim != 2 or not name.endswith(".weight"):
                continue
            snr = compute_snr(param.data)
            results.append(LayerSNR(name=name, snr=snr, module_type=self._module_type(name)))
        return results

    def select_layers(self) -> list[str]:
        """Return parameter names of the top layers to apply LoRA to.

        Groups by module type, then selects top `top_k_fraction` per group
        (at least 1 per group). Returns names sorted by SNR descending.
        """
        all_snrs = self.compute_all_snrs()

        # Group by module type
        groups: dict[str, list[LayerSNR]] = {}
        for entry in all_snrs:
            groups.setdefault(entry.module_type, []).append(entry)

        selected: list[LayerSNR] = []
        for group_entries in groups.values():
            # Sort descending by SNR
            sorted_entries = sorted(group_entries, key=lambda e: e.snr, reverse=True)
            k = max(1, math.ceil(len(sorted_entries) * self.top_k_fraction))
            selected.extend(sorted_entries[:k])

        # Return names sorted by SNR descending
        selected.sort(key=lambda e: e.snr, reverse=True)
        return [e.name for e in selected]
