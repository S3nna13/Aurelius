"""Fisher information-weighted model merging.

Implements Fisher-Weighted Averaging as described in:
  Matena & Raffel (2022) -- "Merging Models with Fisher-Weighted Averaging"
  https://arxiv.org/abs/2111.09832

The key idea: instead of a simple arithmetic average of model weights,
we weight each model's parameters by the diagonal of its empirical
Fisher information matrix.  Parameters where a model has high curvature
(large Fisher value) are more "certain", so they should dominate the merge.

Fisher-weighted merge formula (element-wise):
    theta_merged = sum_i (F_i * theta_i) / sum_i F_i

where F_i is the diagonal Fisher for model i and theta_i its parameters.
"""

from __future__ import annotations

import copy
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fisher estimation
# ---------------------------------------------------------------------------


def compute_fisher(
    model: nn.Module,
    dataloader: Iterable,
    n_batches: int = 100,
    normalize: bool = True,
) -> dict[str, Tensor]:
    """Estimate the diagonal empirical Fisher information matrix.

    Computes the empirical Fisher diagonal via squared gradients of the
    log-likelihood (equivalently, the training loss):

        F_theta ~ (1/N) sum_i [grad_theta log p(y_i | x_i)]^2

    Args:
        model: The model to compute Fisher information for.
        dataloader: Iterable yielding (input_ids, labels) tuples.
        n_batches: Number of mini-batches to accumulate over.
        normalize: If True, normalize each parameter's Fisher values to [0, 1].

    Returns:
        Dict mapping parameter name -> Fisher diagonal tensor (same shape as param).
    """
    model.eval()

    # Initialise accumulators
    fisher: dict[str, Tensor] = {
        name: torch.zeros_like(param, dtype=torch.float32)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    n_seen = 0
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= n_batches:
            break

        input_ids, labels = batch

        # Zero gradients before this batch
        model.zero_grad()

        # Forward pass -- model returns (loss, logits, pkv)
        loss = model(input_ids, labels=labels)[0]

        # Backward pass to get gradients of the loss
        loss.backward()

        # Accumulate squared gradients (empirical Fisher diagonal)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.detach() ** 2

        n_seen += 1

    # Average over batches seen
    if n_seen > 0:
        for name in fisher:
            fisher[name] /= n_seen

    # Normalize each parameter's Fisher to [0, 1]
    if normalize:
        for name in fisher:
            max_val = fisher[name].max()
            if max_val > 0:
                fisher[name] = fisher[name] / max_val

    # Restore training state
    model.zero_grad()

    return fisher


# ---------------------------------------------------------------------------
# Fisher-weighted merge
# ---------------------------------------------------------------------------


def fisher_merge(
    models: list[nn.Module],
    fishers: list[dict[str, Tensor]],
    normalize: bool = True,
) -> dict[str, Tensor]:
    """Fisher-weighted average of model parameters.

    For each parameter, computes:
        theta_merged = sum_i (F_i * theta_i) / sum_i F_i

    If all Fisher values for a parameter are zero (no information), falls back
    to an arithmetic mean for that parameter.

    Args:
        models: List of models to merge.
        fishers: List of Fisher dicts (one per model), as returned by compute_fisher.
        normalize: If True, re-normalise Fisher weights before merging so that
                   all models contribute on an equal scale.

    Returns:
        State dict of merged parameters.
    """
    if not models:
        raise ValueError("models list must not be empty")
    if len(models) != len(fishers):
        raise ValueError("models and fishers must have the same length")

    # Collect state dicts
    states = [m.state_dict() for m in models]

    # All parameter names from the first model
    param_names = list(states[0].keys())

    merged_state: dict[str, Tensor] = {}

    for name in param_names:
        base_tensor = states[0][name]

        # Only Fisher-weight floating-point, non-complex parameters
        if not base_tensor.dtype.is_floating_point or base_tensor.is_complex():
            merged_state[name] = base_tensor.clone()
            continue

        # Check we have Fisher info for this parameter
        has_fisher = all(name in f for f in fishers)

        if not has_fisher:
            # Fall back to arithmetic mean
            tensors = [
                s[name].float() for s in states if name in s and s[name].shape == base_tensor.shape
            ]
            if tensors:
                merged_state[name] = torch.stack(tensors).mean(0).to(base_tensor.dtype)
            else:
                merged_state[name] = base_tensor.clone()
            continue

        # Gather Fisher weights; optionally re-normalise per model
        f_weights = []
        for f in fishers:
            fw = f[name].float().clone()
            if normalize:
                max_val = fw.max()
                if max_val > 0:
                    fw = fw / max_val
            f_weights.append(fw)

        # Weighted sum: sum (F_i * theta_i)
        weighted_sum = torch.zeros_like(base_tensor, dtype=torch.float32)
        weight_total = torch.zeros_like(base_tensor, dtype=torch.float32)

        for s, fw in zip(states, f_weights):
            if name in s and s[name].shape == base_tensor.shape:
                weighted_sum += fw * s[name].float()
                weight_total += fw

        # Where weight_total is zero, fall back to arithmetic mean
        zero_mask = weight_total == 0
        if zero_mask.any():
            arith_mean = torch.stack([s[name].float() for s in states if name in s]).mean(0)
            merged_val = torch.where(
                zero_mask,
                arith_mean,
                weighted_sum / weight_total.clamp(min=1e-12),
            )
        else:
            merged_val = weighted_sum / weight_total

        merged_state[name] = merged_val.to(base_tensor.dtype)

    return merged_state


# ---------------------------------------------------------------------------
# End-to-end convenience function
# ---------------------------------------------------------------------------


def fisher_merge_models(
    base_model: nn.Module,
    models: list[nn.Module],
    dataloaders: list[Iterable],
    n_batches: int = 50,
) -> nn.Module:
    """Compute Fisher information for each model, merge, and return merged model.

    Args:
        base_model: Defines architecture; merged weights will be loaded into a
                    deepcopy of this model.
        models: Fine-tuned models to merge.
        dataloaders: One dataloader per model (same length as models).
        n_batches: Batches to use for Fisher estimation per model.

    Returns:
        New nn.Module (deepcopy of base_model) with Fisher-merged weights.
    """
    if len(models) != len(dataloaders):
        raise ValueError("models and dataloaders must have the same length")

    logger.info("Computing Fisher information for %d models...", len(models))
    fishers = []
    for i, (model, dl) in enumerate(zip(models, dataloaders)):
        logger.info("  Model %d / %d", i + 1, len(models))
        f = compute_fisher(model, dl, n_batches=n_batches)
        fishers.append(f)

    logger.info("Merging models with Fisher-weighted averaging...")
    merged_state = fisher_merge(models, fishers)

    result = copy.deepcopy(base_model)
    result.load_state_dict(merged_state)
    return result


# ---------------------------------------------------------------------------
# Analysis / diagnostics
# ---------------------------------------------------------------------------


@dataclass
class FisherMergeResult:
    """Summary statistics for a Fisher-weighted merge."""

    n_models: int
    n_params: int
    mean_fisher_weight: float  # average Fisher value across all params/models
    merge_time_ms: float
    param_names: list[str] = field(default_factory=list)


def analyze_fisher_merge(
    models: list[nn.Module],
    fishers: list[dict[str, Tensor]],
) -> FisherMergeResult:
    """Compute statistics about a Fisher-weighted merge without merging.

    Args:
        models: Models that would be merged.
        fishers: Fisher dicts for each model.

    Returns:
        FisherMergeResult with summary statistics.
    """
    t0 = time.perf_counter()

    # Collect all parameter names (from named_parameters to get leaf params)
    param_names: list[str] = []
    seen: set[str] = set()
    for name, _ in models[0].named_parameters():
        if name not in seen:
            param_names.append(name)
            seen.add(name)

    # Compute mean Fisher value across all models and parameters
    total_fisher = 0.0
    total_elements = 0
    for f in fishers:
        for name, fval in f.items():
            total_fisher += fval.sum().item()
            total_elements += fval.numel()

    mean_fisher = total_fisher / max(total_elements, 1)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return FisherMergeResult(
        n_models=len(models),
        n_params=len(param_names),
        mean_fisher_weight=mean_fisher,
        merge_time_ms=elapsed_ms,
        param_names=param_names,
    )
