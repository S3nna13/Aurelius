"""Fisher information-weighted model merging (advanced API).

Implements Fisher-Weighted Merging with:
- Diagonal Fisher Information Matrix estimation
- Fisher-weighted two-model merge
- RegMean (Gram-matrix approximation) merging
- Multi-model FisherMerger class with quality evaluation

Reference:
    Matena & Raffel (2022) -- "Merging Models with Fisher-Weighted Averaging"
    https://arxiv.org/abs/2111.09832

NOTE: This module is distinct from fisher_merge.py which provides a simpler
batch-dataloader-based API.  Here we operate on raw Tensor samples.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FisherMergeConfig:
    """Configuration for Fisher-weighted model merging.

    Attributes:
        n_samples: Number of data samples to use when estimating Fisher info.
        fisher_floor: Floor value added to denominators to avoid division by zero.
        normalize: If True, normalize Fisher weights before merging.
        merge_strategy: One of "fisher_weighted", "diagonal_fim", "kronecker_approx".
    """
    n_samples: int = 64
    fisher_floor: float = 1e-6
    normalize: bool = True
    merge_strategy: str = "fisher_weighted"


# ---------------------------------------------------------------------------
# Diagonal Fisher estimation
# ---------------------------------------------------------------------------

def compute_diagonal_fisher(
    model: nn.Module,
    data: list[Tensor],
    n_samples: int = 64,
) -> dict[str, Tensor]:
    """Estimate the diagonal of the Fisher Information Matrix via empirical Fisher.

    For each sample we do a forward pass, pick the predicted label (argmax of
    logits), compute cross-entropy against that label, and accumulate the
    squared gradients.  This is the standard empirical / "type-II" Fisher
    diagonal approximation.

    The model is expected to return ``(loss, logits, past_key_values)`` for a
    plain call ``model(input_ids)``.

    Args:
        model:     PyTorch module with the Aurelius forward signature.
        data:      List of input_ids tensors (each shape [B, T] or [T]).
        n_samples: Maximum number of samples to use (clips ``len(data)``).

    Returns:
        Dict mapping parameter name -> Fisher diagonal tensor, same shape as
        each parameter tensor.  Values are >= 0 (squared gradients).
    """
    model.eval()

    # Initialise accumulators -- only for leaf parameters that require grad
    fisher: dict[str, Tensor] = {
        name: torch.zeros_like(param, dtype=torch.float32)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    n_used = min(n_samples, len(data))

    for idx in range(n_used):
        sample = data[idx]
        # Ensure batch dimension exists
        if sample.dim() == 1:
            sample = sample.unsqueeze(0)

        model.zero_grad()

        # Forward -- model returns (loss, logits, pkv)
        with torch.enable_grad():
            _loss, logits, _pkv = model(sample)

        # Use argmax of logits as pseudo-label (empirical Fisher)
        # logits: [B, T, V]  -> labels: [B, T]
        B, T, V = logits.shape
        pseudo_labels = logits.detach().argmax(dim=-1)  # [B, T]

        # Cross-entropy with pseudo-labels
        loss = F.cross_entropy(
            logits.reshape(B * T, V),
            pseudo_labels.reshape(B * T),
        )

        loss.backward()

        # Accumulate squared gradients
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.detach().float() ** 2

    # Average over samples used
    if n_used > 0:
        for name in fisher:
            fisher[name] /= n_used

    model.zero_grad()
    return fisher


# ---------------------------------------------------------------------------
# Two-model Fisher-weighted merge
# ---------------------------------------------------------------------------

def fisher_merge_two(
    model_a: nn.Module,
    model_b: nn.Module,
    fisher_a: dict[str, Tensor],
    fisher_b: dict[str, Tensor],
    config: FisherMergeConfig,
) -> dict[str, Tensor]:
    """Fisher-weighted merge of two models.

    Element-wise formula::

        merged_param = (F_a * theta_a + F_b * theta_b) / (F_a + F_b + floor)

    Non-floating-point parameters (e.g. integer buffers) are copied from model_a
    unchanged.

    Args:
        model_a:  First model.
        model_b:  Second model.
        fisher_a: Diagonal Fisher dict for model_a.
        fisher_b: Diagonal Fisher dict for model_b.
        config:   Merge configuration.

    Returns:
        State dict of merged parameters.
    """
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    merged: dict[str, Tensor] = {}

    for name, param_a in state_a.items():
        if name not in state_b:
            merged[name] = param_a.clone()
            continue

        param_b = state_b[name]

        # Only Fisher-weight floating-point, real tensors
        if not param_a.dtype.is_floating_point or param_a.is_complex():
            merged[name] = param_a.clone()
            continue

        if name not in fisher_a or name not in fisher_b:
            # Fall back to arithmetic mean if Fisher info not available
            merged[name] = ((param_a.float() + param_b.float()) / 2.0).to(param_a.dtype)
            continue

        fa = fisher_a[name].float()
        fb = fisher_b[name].float()

        if config.normalize:
            max_a = fa.max()
            max_b = fb.max()
            if max_a > 0:
                fa = fa / max_a
            if max_b > 0:
                fb = fb / max_b

        numerator = fa * param_a.float() + fb * param_b.float()
        denominator = fa + fb + config.fisher_floor
        merged[name] = (numerator / denominator).to(param_a.dtype)

    return merged


# ---------------------------------------------------------------------------
# Apply state dict
# ---------------------------------------------------------------------------

def apply_state_dict(model: nn.Module, state_dict: dict[str, Tensor]) -> None:
    """Load a state dict into model in-place.

    Wraps ``model.load_state_dict`` with ``strict=False`` so that buffers not
    present in the state dict (e.g. RoPE frequency buffers not stored in the
    Fisher dict) are left unchanged.

    Args:
        model:      Model to update.
        state_dict: State dict to load.
    """
    model.load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# RegMean merging
# ---------------------------------------------------------------------------

def regmean_merge(
    model_a: nn.Module,
    model_b: nn.Module,
    data: list[Tensor],
) -> dict[str, Tensor]:
    """RegMean-inspired merge using activation magnitude weighting.

    RegMean (Jin et al., 2022) weights linear-layer parameters by the
    input activation Gram matrix.  Here we implement a simplified but
    faithful approximation: we compute the mean L2 magnitude of activations
    for each linear layer on the provided data, then use those magnitudes to
    weight the merge of that layer's parameters.

    For non-linear (or non-matched) parameters we fall back to an arithmetic
    mean.

    Args:
        model_a: First model.
        model_b: Second model (same architecture).
        data:    List of input_ids tensors used to compute activation magnitudes.

    Returns:
        State dict of merged parameters.
    """
    if not data:
        # No data -- return arithmetic mean
        state_a = model_a.state_dict()
        state_b = model_b.state_dict()
        merged = {}
        for name, pa in state_a.items():
            if name in state_b and pa.dtype.is_floating_point and not pa.is_complex():
                merged[name] = ((pa.float() + state_b[name].float()) / 2.0).to(pa.dtype)
            else:
                merged[name] = pa.clone()
        return merged

    # ------------------------------------------------------------------
    # Collect activation magnitudes from both models via forward hooks
    # ------------------------------------------------------------------
    def _collect_magnitudes(model: nn.Module, samples: list[Tensor]) -> dict[str, float]:
        magnitudes: dict[str, float] = {}
        hooks = []
        counts: dict[str, int] = {}

        def _make_hook(layer_name: str):
            def _hook(_module, inp, _out):
                if inp and isinstance(inp[0], Tensor):
                    mag = inp[0].detach().float().norm().item()
                    magnitudes[layer_name] = magnitudes.get(layer_name, 0.0) + mag
                    counts[layer_name] = counts.get(layer_name, 0) + 1
            return _hook

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(_make_hook(name)))

        model.eval()
        with torch.no_grad():
            for sample in samples:
                inp = sample if sample.dim() > 1 else sample.unsqueeze(0)
                try:
                    model(inp)
                except Exception:
                    pass

        for h in hooks:
            h.remove()

        # Average over samples
        for k in magnitudes:
            if counts.get(k, 1) > 0:
                magnitudes[k] /= counts[k]

        return magnitudes

    mag_a = _collect_magnitudes(model_a, data)
    mag_b = _collect_magnitudes(model_b, data)

    # Build a mapping from linear-module-name prefix to weight scalar
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    merged: dict[str, Tensor] = {}

    linear_prefixes: list[str] = [
        name for name, module in model_a.named_modules()
        if isinstance(module, nn.Linear)
    ]

    def _find_prefix(param_name: str):
        for prefix in linear_prefixes:
            if param_name.startswith(prefix + "."):
                return prefix
        return None

    for name, pa in state_a.items():
        if name not in state_b or not pa.dtype.is_floating_point or pa.is_complex():
            merged[name] = pa.clone()
            continue

        pb = state_b[name]
        prefix = _find_prefix(name)

        if prefix is not None and prefix in mag_a and prefix in mag_b:
            wa = mag_a[prefix]
            wb = mag_b[prefix]
            total = wa + wb
            if total > 0:
                alpha = wa / total
                beta = wb / total
            else:
                alpha = beta = 0.5
        else:
            alpha = beta = 0.5

        merged[name] = (alpha * pa.float() + beta * pb.float()).to(pa.dtype)

    return merged


# ---------------------------------------------------------------------------
# FisherMerger class
# ---------------------------------------------------------------------------

class FisherMerger:
    """High-level API for Fisher-weighted multi-model merging."""

    def __init__(self, config: FisherMergeConfig) -> None:
        self.config = config

    def compute_fisher(self, model: nn.Module, data: list[Tensor]) -> dict[str, Tensor]:
        """Compute diagonal Fisher for ``model`` using ``data``.

        Wrapper around :func:`compute_diagonal_fisher` that passes ``n_samples``
        from the configuration.

        Args:
            model: Model to analyse.
            data:  List of input tensors.

        Returns:
            Dict mapping parameter name -> Fisher diagonal tensor.
        """
        return compute_diagonal_fisher(model, data, n_samples=self.config.n_samples)

    def merge(
        self,
        models: list[nn.Module],
        datasets: list[list[Tensor]],
    ) -> dict[str, Tensor]:
        """Merge N models using Fisher-weighted averaging.

        For N > 2 we iteratively merge pairs left-to-right, carrying the
        merged weights into a temporary model for the next step.

        Args:
            models:   List of models to merge (same architecture).
            datasets: One dataset (list of Tensors) per model.

        Returns:
            State dict of the merged model.
        """
        if not models:
            raise ValueError("models list must not be empty")
        if len(models) != len(datasets):
            raise ValueError("models and datasets must have the same length")

        if len(models) == 1:
            return models[0].state_dict()

        # Compute Fisher for every model up-front
        fishers = [
            self.compute_fisher(model, data)
            for model, data in zip(models, datasets)
        ]

        # Iteratively merge pairs
        current_state = fisher_merge_two(
            models[0], models[1], fishers[0], fishers[1], self.config
        )

        for i in range(2, len(models)):
            tmp_model = copy.deepcopy(models[0])
            apply_state_dict(tmp_model, current_state)
            current_state = fisher_merge_two(
                tmp_model, models[i], fishers[0], fishers[i], self.config
            )

        return current_state

    def evaluate_merge_quality(
        self,
        merged_state: dict[str, Tensor],
        original_models: list[nn.Module],
        test_data: list[Tensor],
    ) -> dict:
        """Evaluate quality of a merged state dict vs the original models.

        Args:
            merged_state:    State dict produced by :meth:`merge`.
            original_models: List of the original (pre-merge) models.
            test_data:       List of input_ids tensors for evaluation.

        Returns:
            Dict with keys ``merged_loss``, ``avg_original_loss``, ``degradation``.
        """
        if not original_models:
            return {"merged_loss": 0.0, "avg_original_loss": 0.0, "degradation": 0.0}

        # Build merged model
        merged_model = copy.deepcopy(original_models[0])
        apply_state_dict(merged_model, merged_state)
        merged_model.eval()

        def _avg_loss(model: nn.Module, samples: list[Tensor]) -> float:
            if not samples:
                return 0.0
            total = 0.0
            count = 0
            model.eval()
            with torch.no_grad():
                for sample in samples:
                    inp = sample if sample.dim() > 1 else sample.unsqueeze(0)
                    try:
                        _loss, logits, _pkv = model(inp)
                        B, T, V = logits.shape
                        pseudo = logits.argmax(dim=-1)
                        loss = F.cross_entropy(
                            logits.reshape(B * T, V),
                            pseudo.reshape(B * T),
                        )
                        total += loss.item()
                        count += 1
                    except Exception:
                        pass
            return total / max(count, 1)

        merged_loss = _avg_loss(merged_model, test_data)
        orig_losses = [_avg_loss(m, test_data) for m in original_models]
        avg_original_loss = sum(orig_losses) / max(len(orig_losses), 1)
        degradation = merged_loss - avg_original_loss

        return {
            "merged_loss": float(merged_loss),
            "avg_original_loss": float(avg_original_loss),
            "degradation": float(degradation),
        }
