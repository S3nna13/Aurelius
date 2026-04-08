"""Model analysis utilities: parameter counting, FLOPs estimation, architecture visualization, and weight statistics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

@dataclass
class ModelStats:
    total_params: int
    trainable_params: int
    frozen_params: int
    embedding_params: int
    attention_params: int
    ffn_params: int
    norm_params: int
    other_params: int

    @property
    def param_breakdown_pct(self) -> dict[str, float]:
        """Return percentage breakdown of parameter categories."""
        total = self.total_params if self.total_params > 0 else 1
        return {
            "embedding": self.embedding_params / total * 100,
            "attention": self.attention_params / total * 100,
            "ffn": self.ffn_params / total * 100,
            "norm": self.norm_params / total * 100,
            "other": self.other_params / total * 100,
        }


def _categorize_name(name: str) -> str:
    """Return category string for a parameter name."""
    lower = name.lower()
    if any(kw in lower for kw in ("embed", "lm_head")):
        return "embedding"
    if any(kw in lower for kw in ("attn", "attention")):
        return "attention"
    if any(kw in lower for kw in ("ffn", "mlp", "feed_forward")):
        return "ffn"
    if any(kw in lower for kw in ("norm", "layer_norm")):
        return "norm"
    return "other"


def count_parameters(model: nn.Module) -> ModelStats:
    """
    Count and categorize all parameters.

    Categorization rules (by module name patterns):
    - embed/lm_head -> embedding_params
    - attn/attention -> attention_params
    - ffn/mlp/feed_forward -> ffn_params
    - norm/layer_norm -> norm_params
    - everything else -> other_params

    Only count unique parameters (use id() to avoid double-counting shared weights).
    """
    seen: set[int] = set()
    total = 0
    trainable = 0
    frozen = 0
    embedding = 0
    attention = 0
    ffn = 0
    norm = 0
    other = 0

    for name, param in model.named_parameters():
        pid = id(param)
        if pid in seen:
            continue
        seen.add(pid)

        n = param.numel()
        total += n

        if param.requires_grad:
            trainable += n
        else:
            frozen += n

        cat = _categorize_name(name)
        if cat == "embedding":
            embedding += n
        elif cat == "attention":
            attention += n
        elif cat == "ffn":
            ffn += n
        elif cat == "norm":
            norm += n
        else:
            other += n

    return ModelStats(
        total_params=total,
        trainable_params=trainable,
        frozen_params=frozen,
        embedding_params=embedding,
        attention_params=attention,
        ffn_params=ffn,
        norm_params=norm,
        other_params=other,
    )


# ---------------------------------------------------------------------------
# FLOPs estimation
# ---------------------------------------------------------------------------

def estimate_flops(
    model: nn.Module,
    seq_len: int,
    batch_size: int = 1,
) -> dict:
    """
    Estimate forward pass FLOPs.

    - Embedding: 0 FLOPs (lookup)
    - Self-attention per layer: 4 * B * T * d_model^2 + 2 * B * T^2 * d_model
    - FFN per layer: 2 * B * T * d_model * d_ff  (x3 for SwiGLU gate)
    - LM head: 2 * B * T * d_model * vocab_size
    """
    cfg = model.config
    B = batch_size
    T = seq_len
    d = cfg.d_model
    d_ff = cfg.d_ff
    n_layers = cfg.n_layers
    vocab_size = cfg.vocab_size

    attn_per_layer = 4 * B * T * d * d + 2 * B * T * T * d
    ffn_per_layer = 3 * 2 * B * T * d * d_ff  # SwiGLU has 3 matmuls

    attention_flops = attn_per_layer * n_layers
    ffn_flops = ffn_per_layer * n_layers
    lm_head_flops = 2 * B * T * d * vocab_size

    total_flops = attention_flops + ffn_flops + lm_head_flops

    return {
        "total_flops": total_flops,
        "attention_flops": attention_flops,
        "ffn_flops": ffn_flops,
        "lm_head_flops": lm_head_flops,
        "tflops": total_flops / 1e12,
    }


# ---------------------------------------------------------------------------
# Weight statistics
# ---------------------------------------------------------------------------

def weight_statistics(model: nn.Module) -> dict:
    """
    Compute statistics over all weight tensors.

    Returns:
        mean, std, l2_norm, max_abs, n_dead_neurons_pct, per_layer (first 50)
    """
    all_weights: list[torch.Tensor] = []
    per_layer: list[dict] = []

    for name, param in model.named_parameters():
        data = param.detach().float()
        all_weights.append(data.flatten())

        if len(per_layer) < 50:
            per_layer.append({
                "name": name,
                "shape": tuple(param.shape),
                "mean": float(data.mean()),
                "std": float(data.std()),
                "l2": float(data.norm(2)),
            })

    if not all_weights:
        return {
            "mean": 0.0,
            "std": 0.0,
            "l2_norm": 0.0,
            "max_abs": 0.0,
            "n_dead_neurons_pct": 0.0,
            "per_layer": per_layer,
        }

    flat = torch.cat(all_weights)
    n_zero = float((flat == 0).sum())
    n_total = float(flat.numel())

    return {
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "l2_norm": float(flat.norm(2)),
        "max_abs": float(flat.abs().max()),
        "n_dead_neurons_pct": n_zero / n_total * 100.0,
        "per_layer": per_layer,
    }


# ---------------------------------------------------------------------------
# Activation histograms
# ---------------------------------------------------------------------------

def activation_histogram(
    model: nn.Module,
    input_ids: torch.Tensor,
    layer_names: list[str] | None = None,
    n_bins: int = 50,
) -> dict[str, dict]:
    """
    Collect activation histograms from forward pass via hooks.

    Args:
        layer_names: if None, hook into all linear layers

    Returns: {layer_name: {'bins': Tensor(n_bins+1), 'counts': Tensor(n_bins)}}
    """
    # Determine which modules to hook
    target_modules: dict[str, nn.Module] = {}
    if layer_names is None:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                target_modules[name] = module
    else:
        named = dict(model.named_modules())
        for name in layer_names:
            if name in named:
                target_modules[name] = named[name]

    activations: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(layer_name: str):
        def hook(module, input, output):
            activations[layer_name] = output.detach().float().flatten()
        return hook

    for name, module in target_modules.items():
        handle = module.register_forward_hook(make_hook(name))
        handles.append(handle)

    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        for handle in handles:
            handle.remove()

    result: dict[str, dict] = {}
    for name, flat in activations.items():
        min_val = float(flat.min())
        max_val = float(flat.max())
        # Avoid degenerate range
        if min_val == max_val:
            max_val = min_val + 1e-6
        counts, bins = torch.histogram(flat, bins=n_bins, range=(min_val, max_val))
        result[name] = {"bins": bins, "counts": counts}

    return result


# ---------------------------------------------------------------------------
# Architecture summary
# ---------------------------------------------------------------------------

class ArchitectureSummary:
    """
    Generate human-readable architecture summary (like torch summary).

    Args:
        model: nn.Module
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def summary_string(self) -> str:
        """
        Return multi-line string with model overview.
        """
        stats = count_parameters(self.model)
        model_name = type(self.model).__name__
        total = stats.total_params
        trainable = stats.trainable_params
        pct = trainable / total * 100.0 if total > 0 else 0.0
        pct_breakdown = stats.param_breakdown_pct

        lines = [
            "=" * 48,
            f"Model: {model_name}",
            f"Total Parameters: {total:,}",
            f"Trainable: {trainable:,} ({pct:.1f}%)",
            "=" * 48,
            "Layer breakdown:",
            f"  Embedding: {stats.embedding_params:,} ({pct_breakdown['embedding']:.1f}%)",
            f"  Attention: {stats.attention_params:,} ({pct_breakdown['attention']:.1f}%)",
            f"  FFN: {stats.ffn_params:,} ({pct_breakdown['ffn']:.1f}%)",
            f"  Norm: {stats.norm_params:,} ({pct_breakdown['norm']:.1f}%)",
            "=" * 48,
        ]
        return "\n".join(lines)

    def layer_table(self) -> list[dict]:
        """Return list of {'name', 'type', 'params', 'output_shape_hint'} for all modules."""
        rows = []
        for name, module in self.model.named_modules():
            param_count = sum(p.numel() for p in module.parameters(recurse=False))
            rows.append({
                "name": name,
                "type": type(module).__name__,
                "params": param_count,
                "output_shape_hint": "",
            })
        return rows


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def compare_models(
    model_a: nn.Module,
    model_b: nn.Module,
    name_a: str = "Model A",
    name_b: str = "Model B",
) -> dict:
    """
    Compare two models.

    Returns:
        params_a, params_b, param_diff_pct, shared_architecture, weight_distance
    """
    stats_a = count_parameters(model_a)
    stats_b = count_parameters(model_b)

    params_a = stats_a.total_params
    params_b = stats_b.total_params

    if params_b > 0:
        param_diff_pct = abs(params_a - params_b) / params_b * 100.0
    else:
        param_diff_pct = 0.0 if params_a == 0 else float("inf")

    # Check shared architecture: same parameter names and shapes
    params_dict_a = dict(model_a.named_parameters())
    params_dict_b = dict(model_b.named_parameters())

    shared_architecture = set(params_dict_a.keys()) == set(params_dict_b.keys()) and all(
        params_dict_a[k].shape == params_dict_b[k].shape for k in params_dict_a
    )

    # Compute L2 distance between matching params
    distance_sq = 0.0
    for name in params_dict_a:
        if name in params_dict_b:
            a_data = params_dict_a[name].detach().float()
            b_data = params_dict_b[name].detach().float()
            if a_data.shape == b_data.shape:
                diff = (a_data - b_data).norm(2)
                distance_sq += float(diff ** 2)

    weight_distance = float(distance_sq ** 0.5)

    return {
        "params_a": params_a,
        "params_b": params_b,
        "param_diff_pct": param_diff_pct,
        "shared_architecture": shared_architecture,
        "weight_distance": weight_distance,
    }
