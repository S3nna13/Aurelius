"""Structured pruning for Aurelius models.

Reduces model size by removing low-importance FFN neurons based on
gradient-based importance scores (Taylor et al., 2019 - arXiv:1905.11946).

Importance score: I_i = sum_data (grad_i * weight_i)^2

High score = neuron is important for task performance.
Low score = neuron can be removed.

FFN structure (SwiGLU):
  gate_proj: (d_ff, d_model)  — gating path
  up_proj:   (d_ff, d_model)  — value path
  down_proj: (d_model, d_ff)  — output projection

Pruning removes rows from gate_proj and up_proj,
and corresponding columns from down_proj.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class PruningConfig:
    pruning_ratio: float = 0.2        # fraction of FFN neurons to remove
    n_calibration_steps: int = 100    # batches for importance scoring
    calibration_batch_size: int = 4


class PruningResult(NamedTuple):
    original_params: int
    pruned_params: int
    compression_ratio: float
    layers_pruned: list[str]


def compute_ffn_importance(
    model: nn.Module,
    dataloader,
    n_steps: int = 100,
) -> dict[str, torch.Tensor]:
    """Compute per-neuron importance scores for all FFN layers.

    Runs forward+backward on calibration data and accumulates
    (gradient * weight)^2 as the importance score per FFN neuron.

    Args:
        model: AureliusTransformer.
        dataloader: DataLoader yielding {"input_ids", "labels"} dicts.
        n_steps: Number of calibration batches.

    Returns:
        Dict mapping "layers.i.ffn" -> (d_ff,) importance tensor.
    """
    model.train()
    importances: dict[str, torch.Tensor] = {}

    # Initialize importance accumulators for gate_proj (the gating path)
    for name, module in model.named_modules():
        if name.endswith(".ffn") and hasattr(module, "gate_proj"):
            d_ff = module.gate_proj.out_features
            importances[name] = torch.zeros(d_ff, device=next(model.parameters()).device)

    if not importances:
        logger.warning("No FFN layers found with gate_proj attribute")
        return importances

    # Register hooks to capture gate_proj activations
    hooks = []
    activations: dict[str, torch.Tensor] = {}

    for name, module in model.named_modules():
        if name in importances:
            def make_hook(n):
                def hook(mod, inp, out):
                    activations[n] = out.detach()
                return hook
            hooks.append(module.gate_proj.register_forward_hook(make_hook(name)))

    n = 0
    for batch in dataloader:
        if n >= n_steps:
            break

        if isinstance(batch, dict):
            input_ids = batch["input_ids"]
            labels = batch.get("labels", batch["input_ids"])
        else:
            input_ids, labels = batch[0], batch[1]

        model.zero_grad()
        loss, _, _ = model(input_ids=input_ids, labels=labels)
        loss.backward()

        # Accumulate importance: (grad * weight)^2 summed over in_features
        for ffn_name, module in model.named_modules():
            if ffn_name in importances and hasattr(module, "gate_proj"):
                g = module.gate_proj.weight.grad  # (d_ff, d_model)
                w = module.gate_proj.weight.data   # (d_ff, d_model)
                if g is not None:
                    # Per-neuron importance = sum over in_features of (g*w)^2
                    importances[ffn_name] += (g * w).pow(2).sum(dim=1)

        n += 1

    # Clean up hooks
    for h in hooks:
        h.remove()

    # Average over batches
    importances = {k: v / max(1, n) for k, v in importances.items()}

    model.zero_grad()
    return importances


def prune_ffn_neurons(
    model: nn.Module,
    importances: dict[str, torch.Tensor],
    pruning_ratio: float = 0.2,
) -> PruningResult:
    """Remove low-importance neurons from FFN layers.

    For each FFN layer:
    1. Sort neurons by importance (ascending)
    2. Keep top (1 - pruning_ratio) fraction
    3. Reconstruct gate_proj, up_proj, down_proj with fewer neurons

    Args:
        model: AureliusTransformer.
        importances: From compute_ffn_importance().
        pruning_ratio: Fraction of neurons to remove.

    Returns:
        PruningResult with stats.
    """
    original_params = sum(p.numel() for p in model.parameters())
    layers_pruned = []

    for ffn_name, importance in importances.items():
        # Find the FFN module
        module = model
        for part in ffn_name.split("."):
            module = getattr(module, part)

        if not (hasattr(module, "gate_proj") and hasattr(module, "up_proj") and hasattr(module, "down_proj")):
            continue

        d_ff = importance.shape[0]
        n_keep = max(1, int(d_ff * (1 - pruning_ratio)))

        # Select neurons to keep (highest importance)
        _, keep_indices = torch.topk(importance, n_keep, largest=True, sorted=True)
        keep_indices, _ = torch.sort(keep_indices)  # sort for determinism

        # Reconstruct gate_proj: (d_ff, d_model) → (n_keep, d_model)
        old_gate = module.gate_proj.weight.data
        new_gate = nn.Linear(old_gate.shape[1], n_keep, bias=False)
        new_gate.weight.data = old_gate[keep_indices]
        module.gate_proj = new_gate

        # Reconstruct up_proj: (d_ff, d_model) → (n_keep, d_model)
        old_up = module.up_proj.weight.data
        new_up = nn.Linear(old_up.shape[1], n_keep, bias=False)
        new_up.weight.data = old_up[keep_indices]
        module.up_proj = new_up

        # Reconstruct down_proj: (d_model, d_ff) → (d_model, n_keep)
        old_down = module.down_proj.weight.data
        new_down = nn.Linear(n_keep, old_down.shape[0], bias=False)
        new_down.weight.data = old_down[:, keep_indices]
        module.down_proj = new_down

        layers_pruned.append(ffn_name)
        logger.info(
            "Pruned %s: %d → %d neurons (%.1f%% removed)",
            ffn_name, d_ff, n_keep, pruning_ratio * 100,
        )

    pruned_params = sum(p.numel() for p in model.parameters())
    compression = original_params / max(1, pruned_params)

    return PruningResult(
        original_params=original_params,
        pruned_params=pruned_params,
        compression_ratio=compression,
        layers_pruned=layers_pruned,
    )


def prune_model(
    model: nn.Module,
    dataloader,
    cfg: PruningConfig | None = None,
) -> PruningResult:
    """High-level function: compute importances and prune in one call.

    Args:
        model: AureliusTransformer.
        dataloader: Calibration data.
        cfg: Pruning configuration.

    Returns:
        PruningResult with compression stats.
    """
    if cfg is None:
        cfg = PruningConfig()

    logger.info("Computing FFN importance scores (%d calibration steps)...", cfg.n_calibration_steps)
    importances = compute_ffn_importance(model, dataloader, n_steps=cfg.n_calibration_steps)

    if not importances:
        logger.warning("No FFN layers found to prune")
        n = sum(p.numel() for p in model.parameters())
        return PruningResult(n, n, 1.0, [])

    logger.info("Pruning %d FFN layers at %.0f%% ratio...", len(importances), cfg.pruning_ratio * 100)
    return prune_ffn_neurons(model, importances, cfg.pruning_ratio)
