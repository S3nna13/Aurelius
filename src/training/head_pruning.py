"""Attention head pruning for Aurelius models.

Implements structured head-level pruning based on Michel et al. (2019),
"Are Sixteen Heads Really Better than One?" Most attention heads can be pruned
with minimal accuracy loss. Structured pruning (removing entire heads) is
hardware-friendly unlike unstructured sparsity.

Importance metric: gradient × activation sensitivity (grad_sensitivity) or
L1 norm of output projection weights (l1_norm).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class HeadPruningConfig:
    target_sparsity: float = 0.5          # fraction of heads to prune (0.5 = half)
    importance_metric: str = "grad_sensitivity"  # "grad_sensitivity" | "l1_norm" | "random"
    n_calibration_batches: int = 10       # batches to estimate importance
    prune_layers: list[int] | None = None  # None = prune all layers


@dataclass
class HeadImportanceScore:
    layer_idx: int
    head_idx: int
    score: float  # higher = more important (keep)


def _get_attn_module(model: nn.Module, layer_idx: int):
    """Return the attention sub-module for a given layer index."""
    return model.layers[layer_idx].attn


def _get_out_proj(attn_module: nn.Module) -> nn.Linear:
    """Return the output projection from an attention module, trying several attribute names."""
    for attr in ("W_o", "out_proj", "o_proj"):
        if hasattr(attn_module, attr):
            return getattr(attn_module, attr)
    raise AttributeError(
        f"Cannot find output projection on {type(attn_module).__name__}. "
        "Expected one of: W_o, out_proj, o_proj"
    )


def _n_layers_heads(model: nn.Module) -> tuple[int, int]:
    """Return (n_layers, n_heads) from model config."""
    cfg = model.config
    return cfg.n_layers, cfg.n_heads


def compute_head_importance_l1(
    model: nn.Module,
) -> list[HeadImportanceScore]:
    """Compute head importance as L1 norm of the attention output projection weights.

    For each layer l and head h:
        importance[l, h] = ||W_o[h*head_dim:(h+1)*head_dim, :]||_1

    where W_o is the output projection (n_heads * head_dim -> d_model).
    Returns sorted list (highest score first).
    """
    cfg = model.config
    n_layers = cfg.n_layers
    n_heads = cfg.n_heads
    head_dim = cfg.head_dim

    scores: list[HeadImportanceScore] = []

    for layer_idx in range(n_layers):
        attn = _get_attn_module(model, layer_idx)
        out_proj = _get_out_proj(attn)
        # Weight shape: (d_model, n_heads * head_dim)  [Linear stores as (out, in)]
        weight = out_proj.weight  # (d_model, n_heads * head_dim)

        for head_idx in range(n_heads):
            start = head_idx * head_dim
            end = (head_idx + 1) * head_dim
            # Slice the input dimension corresponding to this head
            head_weight = weight[:, start:end]  # (d_model, head_dim)
            score = head_weight.abs().sum().item()
            scores.append(HeadImportanceScore(layer_idx=layer_idx, head_idx=head_idx, score=score))

    scores.sort(key=lambda s: s.score, reverse=True)
    return scores


def compute_head_importance_gradient(
    model: nn.Module,
    data_batches: list[torch.Tensor],  # list of input_ids tensors
    loss_fn: Callable | None = None,
) -> list[HeadImportanceScore]:
    """Compute head importance using gradient x activation sensitivity.

    For each head, compute: importance = mean(|gradient * activation|) over data.

    Uses hooks on attention output projections to collect gradient norms per head.
    If loss_fn is None, uses standard CE loss from model output.
    """
    cfg = model.config
    n_layers = cfg.n_layers
    n_heads = cfg.n_heads
    head_dim = cfg.head_dim

    # Accumulate importance: shape (n_layers, n_heads)
    device = next(model.parameters()).device
    importance_acc = torch.zeros(n_layers, n_heads, device=device)
    count = 0

    # Register hooks on each layer's attention output projection
    hooks = []
    activations: dict[int, torch.Tensor] = {}

    for layer_idx in range(n_layers):
        attn = _get_attn_module(model, layer_idx)
        out_proj = _get_out_proj(attn)

        def make_fwd_hook(idx: int):
            def hook(mod: nn.Module, inp: tuple, out: torch.Tensor) -> None:
                # inp[0] is the input to the linear: (batch, seq, n_heads * head_dim)
                activations[idx] = inp[0].detach()
            return hook

        hooks.append(out_proj.register_forward_hook(make_fwd_hook(layer_idx)))

    model.train()

    for batch in data_batches:
        if isinstance(batch, dict):
            input_ids = batch["input_ids"].to(device)
        else:
            input_ids = batch.to(device)

        model.zero_grad()

        if loss_fn is not None:
            loss = loss_fn(model, input_ids)
        else:
            result = model(input_ids=input_ids, labels=input_ids)
            if isinstance(result, tuple):
                loss = result[0]
            else:
                loss = result

        loss.backward()

        with torch.no_grad():
            for layer_idx in range(n_layers):
                attn = _get_attn_module(model, layer_idx)
                out_proj = _get_out_proj(attn)

                act = activations.get(layer_idx)
                if act is None:
                    continue

                # act: (batch, seq, n_heads * head_dim)
                grad = out_proj.weight.grad  # (d_model, n_heads * head_dim)
                if grad is None:
                    continue

                # Compute per-head sensitivity
                for head_idx in range(n_heads):
                    start = head_idx * head_dim
                    end = (head_idx + 1) * head_dim
                    # head activation: (batch, seq, head_dim)
                    head_act = act[..., start:end]
                    # head gradient slice from weight: (d_model, head_dim)
                    head_grad = grad[:, start:end]
                    # sensitivity = mean(|act|) * sum(|grad|)  (proxy for grad * act)
                    sensitivity = head_act.abs().mean().item() * head_grad.abs().sum().item()
                    importance_acc[layer_idx, head_idx] += sensitivity

        count += 1

    for h in hooks:
        h.remove()

    model.zero_grad()

    if count > 0:
        importance_acc /= count

    scores: list[HeadImportanceScore] = []
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            scores.append(HeadImportanceScore(
                layer_idx=layer_idx,
                head_idx=head_idx,
                score=importance_acc[layer_idx, head_idx].item(),
            ))

    scores.sort(key=lambda s: s.score, reverse=True)
    return scores


class HeadMask(nn.Module):
    """Learnable head mask for soft pruning.

    Multiplies each head's output by a binary (or soft) mask value.
    Mask values are initialized to 1.0 and can be trained or set to 0 to prune.

    Args:
        n_layers: Number of transformer layers.
        n_heads: Number of attention heads per layer.
    """

    def __init__(self, n_layers: int, n_heads: int) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        # mask: (n_layers, n_heads) float tensor, all ones
        self.mask = nn.Parameter(torch.ones(n_layers, n_heads))

    def get_mask(self, layer_idx: int) -> torch.Tensor:
        """Return (n_heads,) mask for this layer."""
        return self.mask[layer_idx]

    def prune_head(self, layer_idx: int, head_idx: int) -> None:
        """Set mask[layer_idx, head_idx] = 0 (hard prune)."""
        with torch.no_grad():
            self.mask[layer_idx, head_idx] = 0.0

    def active_heads(self) -> int:
        """Count non-zero mask entries."""
        return int((self.mask != 0).sum().item())


class StructuredHeadPruner:
    """End-to-end structured head pruning pipeline.

    Steps:
    1. Compute importance scores (l1 or gradient-based)
    2. Sort by importance, select heads to prune
    3. Apply HeadMask to zero out pruned heads
    4. Optionally fine-tune to recover accuracy

    Args:
        model: AureliusTransformer
        config: HeadPruningConfig
    """

    def __init__(self, model: nn.Module, config: HeadPruningConfig | None = None) -> None:
        self.model = model
        self.config = config or HeadPruningConfig()
        self._head_mask: HeadMask | None = None

    def _init_mask(self) -> HeadMask:
        n_layers, n_heads = _n_layers_heads(self.model)
        mask = HeadMask(n_layers=n_layers, n_heads=n_heads)
        self._head_mask = mask
        return mask

    def compute_importance(
        self,
        data_batches: list[torch.Tensor] | None = None,
    ) -> list[HeadImportanceScore]:
        """Compute scores using the configured metric."""
        metric = self.config.importance_metric

        if metric == "l1_norm":
            return compute_head_importance_l1(self.model)
        elif metric == "grad_sensitivity":
            if data_batches is None:
                raise ValueError("data_batches required for grad_sensitivity metric")
            return compute_head_importance_gradient(self.model, data_batches)
        elif metric == "random":
            n_layers, n_heads = _n_layers_heads(self.model)
            scores = [
                HeadImportanceScore(layer_idx=l, head_idx=h, score=torch.rand(1).item())
                for l in range(n_layers)
                for h in range(n_heads)
            ]
            scores.sort(key=lambda s: s.score, reverse=True)
            return scores
        else:
            raise ValueError(f"Unknown importance metric: {metric!r}")

    def prune(
        self,
        importance_scores: list[HeadImportanceScore] | None = None,
        data_batches: list[torch.Tensor] | None = None,
    ) -> dict:
        """Prune heads based on importance scores.

        Returns: {'n_pruned': int, 'n_active': int, 'pruned_heads': list[tuple[int,int]]}
        """
        if importance_scores is None:
            importance_scores = self.compute_importance(data_batches)

        n_layers, n_heads = _n_layers_heads(self.model)
        total_heads = n_layers * n_heads

        # Filter to configured layers only
        if self.config.prune_layers is not None:
            eligible = [s for s in importance_scores if s.layer_idx in self.config.prune_layers]
        else:
            eligible = list(importance_scores)

        n_to_prune = int(len(eligible) * self.config.target_sparsity)

        # Sort ascending by score — least important first
        eligible_sorted = sorted(eligible, key=lambda s: s.score)
        heads_to_prune = eligible_sorted[:n_to_prune]

        mask = self._init_mask()

        pruned_list: list[tuple[int, int]] = []
        for head_score in heads_to_prune:
            mask.prune_head(head_score.layer_idx, head_score.head_idx)
            pruned_list.append((head_score.layer_idx, head_score.head_idx))

        n_active = mask.active_heads()
        n_pruned = total_heads - n_active

        logger.info(
            "Pruned %d/%d heads (%.1f%% sparsity)",
            n_pruned, total_heads, 100.0 * n_pruned / max(1, total_heads),
        )

        return {
            "n_pruned": n_pruned,
            "n_active": n_active,
            "pruned_heads": pruned_list,
        }

    def pruning_stats(self) -> dict:
        """Return current pruning statistics.

        Returns:
            {'total_heads': int, 'pruned_heads': int, 'sparsity': float,
             'per_layer_active': list[int]}
        """
        n_layers, n_heads = _n_layers_heads(self.model)
        total_heads = n_layers * n_heads

        if self._head_mask is None:
            return {
                "total_heads": total_heads,
                "pruned_heads": 0,
                "sparsity": 0.0,
                "per_layer_active": [n_heads] * n_layers,
            }

        mask = self._head_mask
        per_layer_active = [
            int((mask.mask[layer_idx] != 0).sum().item())
            for layer_idx in range(n_layers)
        ]
        n_active = sum(per_layer_active)
        n_pruned = total_heads - n_active
        sparsity = n_pruned / max(1, total_heads)

        return {
            "total_heads": total_heads,
            "pruned_heads": n_pruned,
            "sparsity": sparsity,
            "per_layer_active": per_layer_active,
        }


def estimate_flop_reduction(
    n_layers: int,
    n_heads: int,
    head_dim: int,
    seq_len: int,
    n_pruned_heads: int,
) -> float:
    """Estimate FLOPs reduction ratio from head pruning.

    Attention FLOPs ∝ n_heads * (T^2 + T*head_dim)
    Reduction = n_pruned_heads / n_heads (approximate, ignoring W_o)

    Returns fraction of FLOPs saved: [0, 1]
    """
    if n_heads == 0:
        return 0.0
    reduction = n_pruned_heads / n_heads
    return float(max(0.0, min(1.0, reduction)))
