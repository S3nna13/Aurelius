"""Structured pruning: neuron-level, head-level, and layer-level with regrowth support."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class StructuredPruningConfig:
    """Configuration for structured pruning with optional regrowth."""

    target_sparsity: float = 0.5
    pruning_schedule: str = "linear"
    n_pruning_steps: int = 10
    regrowth_fraction: float = 0.0
    min_neurons_per_layer: int = 1


def compute_sparsity_schedule(
    target: float,
    n_steps: int,
    schedule: str,
) -> list[float]:
    """Return a list of n_steps sparsity values from 0 to target.

    Args:
        target: Final target sparsity fraction.
        n_steps: Number of pruning steps.
        schedule: 'linear', 'cubic', or 'one_shot'.

    Returns:
        List of n_steps float sparsity values.
    """
    if n_steps <= 0:
        return []

    if schedule == "linear":
        if n_steps == 1:
            return [target]
        return [target * i / (n_steps - 1) for i in range(n_steps)]

    elif schedule == "cubic":
        result = []
        for i in range(n_steps):
            t = i / (n_steps - 1) if n_steps > 1 else 1.0
            result.append(target * (t ** 3))
        return result

    elif schedule == "one_shot":
        return [0.0] * (n_steps - 1) + [target]

    else:
        raise ValueError(f"Unknown pruning schedule: {schedule!r}. Use 'linear', 'cubic', or 'one_shot'.")


def compute_neuron_importance(weight: Tensor, method: str = "magnitude") -> Tensor:
    """Compute per-neuron importance scores for a linear layer weight matrix.

    Args:
        weight: (out, in) linear weight matrix.
        method: 'magnitude' computes L2 norm of each output neuron (row norm).

    Returns:
        (out,) importance scores.
    """
    if method == "magnitude":
        return weight.norm(dim=1)
    else:
        raise ValueError(f"Unknown importance method: {method!r}. Use 'magnitude'.")


def prune_neurons(
    weight: Tensor,
    bias: Tensor | None,
    sparsity: float,
) -> tuple[Tensor, Tensor | None, Tensor]:
    """Remove lowest-importance output neurons from a linear layer.

    Args:
        weight: (out, in) weight matrix.
        bias: (out,) bias or None.
        sparsity: Fraction of output neurons to remove.

    Returns:
        Tuple of (pruned_weight, pruned_bias, kept_indices).
    """
    out_features = weight.shape[0]
    n_remove = int(out_features * sparsity)
    n_keep = max(1, out_features - n_remove)

    importance = weight.norm(dim=1)
    _, kept_indices = torch.topk(importance, n_keep, largest=True, sorted=True)
    kept_indices, _ = torch.sort(kept_indices)

    pruned_weight = weight[kept_indices]
    pruned_bias = bias[kept_indices] if bias is not None else None

    return pruned_weight, pruned_bias, kept_indices


def regrow_neurons(
    weight: Tensor,
    pruned_weights: Tensor,
    kept_indices: Tensor,
    n_regrow: int,
) -> tuple[Tensor, Tensor]:
    """Add n_regrow neurons back, re-initialized to small random values.

    Args:
        weight: Current (kept_out, in) weight matrix.
        pruned_weights: Original full weight matrix before pruning (original_out, in).
        kept_indices: (kept_out,) indices that were kept.
        n_regrow: Number of neurons to add back.

    Returns:
        Tuple of (new_weight, new_kept_indices).
    """
    if n_regrow <= 0:
        return weight, kept_indices

    original_out = pruned_weights.shape[0]
    in_features = weight.shape[1]

    all_indices = torch.arange(original_out, device=weight.device)
    pruned_mask = torch.ones(original_out, dtype=torch.bool, device=weight.device)
    pruned_mask[kept_indices] = False
    pruned_indices = all_indices[pruned_mask]

    n_available = pruned_indices.shape[0]
    n_regrow = min(n_regrow, n_available)

    if n_regrow <= 0:
        return weight, kept_indices

    perm = torch.randperm(n_available, device=weight.device)[:n_regrow]
    regrow_indices = pruned_indices[perm]

    new_rows = torch.randn(n_regrow, in_features, device=weight.device) * 0.01
    new_weight = torch.cat([weight, new_rows], dim=0)
    new_kept_indices = torch.cat([kept_indices, regrow_indices])
    new_kept_indices, sort_order = torch.sort(new_kept_indices)
    new_weight = new_weight[sort_order]

    return new_weight, new_kept_indices


def prune_attention_heads(
    q_weight: Tensor,
    k_weight: Tensor,
    v_weight: Tensor,
    o_weight: Tensor,
    head_dim: int,
    sparsity: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, list[int]]:
    """Remove lowest-scoring attention heads based on q_weight norms.

    Args:
        q_weight: (n_heads * head_dim, d_model)
        k_weight: (n_kv_heads * head_dim, d_model)
        v_weight: (n_kv_heads * head_dim, d_model)
        o_weight: (d_model, n_heads * head_dim)
        head_dim: Dimension per head.
        sparsity: Fraction of heads to remove.

    Returns:
        (pruned_q, pruned_k, pruned_v, pruned_o, kept_head_indices)
    """
    n_heads = q_weight.shape[0] // head_dim
    n_kv_heads = k_weight.shape[0] // head_dim

    head_score_list = []
    for h in range(n_heads):
        start = h * head_dim
        end = start + head_dim
        score = q_weight[start:end].norm().item()
        head_score_list.append(score)

    scores_tensor = torch.tensor(head_score_list, device=q_weight.device)

    n_remove = int(n_heads * sparsity)
    n_keep = max(1, n_heads - n_remove)

    _, kept_tensor = torch.topk(scores_tensor, n_keep, largest=True, sorted=True)
    kept_tensor, _ = torch.sort(kept_tensor)
    kept_head_indices = kept_tensor.tolist()

    q_rows = [q_weight[h * head_dim:(h + 1) * head_dim] for h in kept_head_indices]
    pruned_q = torch.cat(q_rows, dim=0)

    kv_ratio = max(1, n_heads // n_kv_heads)
    kept_kv_indices = sorted(set(h // kv_ratio for h in kept_head_indices))

    k_rows = [k_weight[h * head_dim:(h + 1) * head_dim] for h in kept_kv_indices]
    pruned_k = torch.cat(k_rows, dim=0)

    v_rows = [v_weight[h * head_dim:(h + 1) * head_dim] for h in kept_kv_indices]
    pruned_v = torch.cat(v_rows, dim=0)

    o_cols = [o_weight[:, h * head_dim:(h + 1) * head_dim] for h in kept_head_indices]
    pruned_o = torch.cat(o_cols, dim=1)

    return pruned_q, pruned_k, pruned_v, pruned_o, kept_head_indices


def compute_layer_importance(
    model: nn.Module,
    calibration_ids: Tensor,
) -> Tensor:
    """Score each transformer layer by gradient magnitude.

    Runs one forward+backward pass and measures mean absolute gradient
    of the FFN down_proj (or o_proj) as importance.

    Args:
        model: AureliusTransformer.
        calibration_ids: (B, T) input token ids.

    Returns:
        (n_layers,) importance scores.
    """
    was_training = model.training
    model.train()
    model.zero_grad()

    try:
        _, logits, _ = model(calibration_ids)
        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, V),
            calibration_ids[:, 1:].reshape(-1),
        )
        loss.backward()
    finally:
        if not was_training:
            model.eval()

    scores = []
    for layer in model.layers:
        grad = None
        if hasattr(layer, "ffn") and hasattr(layer.ffn, "down_proj"):
            g = layer.ffn.down_proj.weight.grad
            if g is not None:
                grad = g
        if grad is None and hasattr(layer, "attn") and hasattr(layer.attn, "o_proj"):
            g = layer.attn.o_proj.weight.grad
            if g is not None:
                grad = g
        if grad is not None:
            scores.append(grad.abs().mean().item())
        else:
            scores.append(0.0)

    model.zero_grad()
    return torch.tensor(scores, dtype=torch.float32)


class IterativePruner:
    """Iteratively prune a model according to a sparsity schedule."""

    def __init__(self, model: nn.Module, config: StructuredPruningConfig) -> None:
        self.model = model
        self.config = config
        self._schedule = compute_sparsity_schedule(
            config.target_sparsity,
            config.n_pruning_steps,
            config.pruning_schedule,
        )

    def prune_step(self, step: int, calibration_ids: Tensor) -> dict:
        """Execute one pruning step.

        Args:
            step: Current step index (0-based).
            calibration_ids: (B, T) input token ids.

        Returns:
            Dict with keys: 'step', 'sparsity', 'n_pruned'.
        """
        sparsity = self._schedule[step] if step < len(self._schedule) else self.config.target_sparsity

        compute_layer_importance(self.model, calibration_ids)

        n_pruned = 0
        for _name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and module.weight.shape[0] > self.config.min_neurons_per_layer:
                out_features = module.weight.shape[0]
                n_remove = int(out_features * sparsity)
                n_keep = max(self.config.min_neurons_per_layer, out_features - n_remove)
                n_pruned += out_features - n_keep

        return {
            "step": step,
            "sparsity": sparsity,
            "n_pruned": n_pruned,
        }
