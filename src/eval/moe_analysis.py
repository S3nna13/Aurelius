"""Analysis utilities for Mixture of Experts models.

Provides expert utilization tracking, routing pattern visualization,
expert specialization metrics, and load imbalance detection.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core stats dataclass
# ---------------------------------------------------------------------------

@dataclass
class MoERoutingStats:
    """Statistics from one forward pass of a MoE model."""
    layer_idx: int
    n_experts: int
    n_tokens: int
    expert_load: torch.Tensor          # (n_experts,) fraction of tokens per expert
    routing_entropy: float             # entropy of routing distribution
    load_imbalance: float              # coefficient of variation = std/mean
    top_expert_fraction: float         # fraction of tokens to most-loaded expert


# ---------------------------------------------------------------------------
# Standalone math utilities
# ---------------------------------------------------------------------------

def routing_entropy(probs: torch.Tensor) -> float:
    """Shannon entropy of routing probabilities.

    Args:
        probs: (n_experts,) non-negative tensor that sums to 1.

    Returns:
        float in [0, log(n_experts)]
    """
    eps = 1e-10
    p = probs.float()
    h = -(p * torch.log(p + eps)).sum()
    return h.item()


def jensen_shannon_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """JSD between two probability distributions (log base 2, range [0, 1]).

    JSD(P||Q) = 0.5*KL(P||M) + 0.5*KL(Q||M) where M = (P+Q)/2

    Args:
        p: (n,) non-negative, sums to 1
        q: (n,) non-negative, sums to 1

    Returns:
        float in [0, 1]
    """
    eps = 1e-10
    p = p.float().clamp(min=eps)
    q = q.float().clamp(min=eps)
    # Re-normalise after clamp so they remain valid distributions
    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)

    kl_pm = (p * torch.log2(p / m)).sum()
    kl_qm = (q * torch.log2(q / m)).sum()

    jsd = 0.5 * kl_pm + 0.5 * kl_qm
    # Clamp to [0, 1] to guard against tiny numerical negatives
    return jsd.clamp(min=0.0, max=1.0).item()


# ---------------------------------------------------------------------------
# ExpertUtilizationTracker
# ---------------------------------------------------------------------------

class ExpertUtilizationTracker:
    """Hook-based tracker that monitors expert utilization across forward passes.

    Attaches to SparseMoEFFN or BalancedMoEFFN modules in the model.
    Accumulates routing statistics across multiple forward passes.

    Usage::

        tracker = ExpertUtilizationTracker(model)
        tracker.attach()
        for batch in data:
            model(batch)
        stats = tracker.get_stats()
        tracker.detach()
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._hooks: list = []
        self._routing_history: list[MoERoutingStats] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def attach(self) -> None:
        """Register pre-hooks on all MoE FFN layers in the model."""
        from src.model.moe import SparseMoEFFN
        from src.model.moe_balanced import BalancedMoEFFN

        layer_idx = 0
        for module in self.model.modules():
            if isinstance(module, (SparseMoEFFN, BalancedMoEFFN)):
                hook = module.register_forward_pre_hook(
                    self._make_hook(layer_idx, module.n_experts)
                )
                self._hooks.append(hook)
                layer_idx += 1

    def detach(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _make_hook(self, layer_idx: int, n_experts: int):
        """Return a forward pre-hook that captures routing statistics."""

        def hook(module, inputs):
            # inputs is a tuple; first element is the activation tensor x
            x = inputs[0]
            with torch.no_grad():
                x_flat = x.view(-1, x.shape[-1])   # (N, D)
                N = x_flat.shape[0]

                # Run router to get logits then soft probabilities
                router_logits = module.router(x_flat)        # (N, n_experts)

                # For BalancedMoEFFN the bias should also be added
                from src.model.moe_balanced import BalancedMoEFFN
                if isinstance(module, BalancedMoEFFN):
                    router_logits = router_logits + module.expert_bias

                router_probs = F.softmax(router_logits, dim=-1)  # (N, n_experts)

                # Top-k selection to build load distribution
                top_k = module.top_k
                _, top_k_indices = torch.topk(router_probs, top_k, dim=-1)  # (N, top_k)

                # expert_load: fraction of (token, slot) pairs per expert
                expert_counts = torch.zeros(n_experts, device=x.device, dtype=torch.float32)
                for k in range(top_k):
                    one_hot = F.one_hot(top_k_indices[:, k], num_classes=n_experts).float()
                    expert_counts += one_hot.sum(dim=0)
                expert_load = expert_counts / (N * top_k)  # (n_experts,)

                # Routing entropy from mean routing probabilities
                mean_probs = router_probs.mean(dim=0)   # (n_experts,)
                ent = routing_entropy(mean_probs)

                # Load imbalance: coefficient of variation
                mean_load = expert_load.mean()
                std_load = expert_load.std()
                imbalance = (std_load / (mean_load + 1e-10)).item()

                top_frac = expert_load.max().item()

                stat = MoERoutingStats(
                    layer_idx=layer_idx,
                    n_experts=n_experts,
                    n_tokens=N,
                    expert_load=expert_load.cpu(),
                    routing_entropy=ent,
                    load_imbalance=imbalance,
                    top_expert_fraction=top_frac,
                )
                self._routing_history.append(stat)

        return hook

    def get_stats(self) -> list[MoERoutingStats]:
        """Return accumulated routing stats."""
        return list(self._routing_history)

    def summary(self) -> dict:
        """Aggregate stats across all recorded passes.

        Returns::

            {
                'per_layer': dict[int, dict],  # layer_idx -> aggregate stats
                'overall_imbalance': float,    # mean load_imbalance across layers
                'dead_experts': list[tuple],   # (layer_idx, expert_idx) with load < 0.01
            }
        """
        if not self._routing_history:
            return {"per_layer": {}, "overall_imbalance": 0.0, "dead_experts": []}

        # Collect per-layer records
        per_layer_records: dict[int, list[MoERoutingStats]] = {}
        for s in self._routing_history:
            per_layer_records.setdefault(s.layer_idx, []).append(s)

        per_layer: dict[int, dict] = {}
        all_imbalances: list[float] = []
        dead_experts: list[tuple] = []

        for layer_idx, records in per_layer_records.items():
            loads = torch.stack([r.expert_load for r in records], dim=0)  # (T, n_experts)
            mean_load = loads.mean(dim=0)    # (n_experts,)
            n_experts_count = records[0].n_experts

            mean_imbalance = sum(r.load_imbalance for r in records) / len(records)
            mean_entropy = sum(r.routing_entropy for r in records) / len(records)

            per_layer[layer_idx] = {
                "mean_expert_load": mean_load,
                "mean_imbalance": mean_imbalance,
                "mean_entropy": mean_entropy,
                "n_passes": len(records),
            }

            all_imbalances.append(mean_imbalance)

            # Dead-expert detection: mean utilisation < 1%
            for expert_idx in range(n_experts_count):
                if mean_load[expert_idx].item() < 0.01:
                    dead_experts.append((layer_idx, expert_idx))

        overall_imbalance = sum(all_imbalances) / len(all_imbalances) if all_imbalances else 0.0

        return {
            "per_layer": per_layer,
            "overall_imbalance": overall_imbalance,
            "dead_experts": dead_experts,
        }


# ---------------------------------------------------------------------------
# Expert specialization
# ---------------------------------------------------------------------------

def compute_expert_specialization(
    model: nn.Module,
    dataset: list[dict],
    tokenizer_encode,
    max_seq_len: int = 128,
) -> dict:
    """Measure whether experts specialize by input category.

    For each category, run forward passes and collect which experts activate.
    Compute Jensen-Shannon divergence between expert usage distributions across
    categories. High JSD means experts are specialized; low JSD means generalist.

    Args:
        model: a MoE model with SparseMoEFFN or BalancedMoEFFN layers.
        dataset: list of dicts with 'text' and 'category' keys.
        tokenizer_encode: callable that converts text to a list of int token ids.
        max_seq_len: maximum sequence length to use.

    Returns:
        dict with keys 'per_category_expert_usage', 'jsd_matrix',
        'mean_jsd', 'specialization_score'.
    """
    tracker = ExpertUtilizationTracker(model)
    tracker.attach()

    # Group samples by category
    categories: dict[str, list[dict]] = {}
    for sample in dataset:
        cat = sample["category"]
        categories.setdefault(cat, []).append(sample)

    category_names = sorted(categories.keys())
    per_category_usage: dict[str, torch.Tensor] = {}

    model.eval()
    with torch.no_grad():
        for cat in category_names:
            tracker._routing_history.clear()

            for sample in categories[cat]:
                ids = tokenizer_encode(sample["text"])
                ids = ids[:max_seq_len]
                x = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
                try:
                    model(x)
                except Exception:
                    pass

            stats = tracker.get_stats()
            if stats:
                all_loads = torch.stack([s.expert_load for s in stats], dim=0)
                mean_load = all_loads.mean(dim=0)
                total = mean_load.sum()
                if total > 0:
                    mean_load = mean_load / total
                per_category_usage[cat] = mean_load
            else:
                per_category_usage[cat] = torch.ones(1)

    tracker.detach()

    # Build pairwise JSD matrix
    n_cats = len(category_names)
    jsd_matrix = torch.zeros(n_cats, n_cats)

    for i, cat_i in enumerate(category_names):
        for j, cat_j in enumerate(category_names):
            if i != j:
                jsd_val = jensen_shannon_divergence(
                    per_category_usage[cat_i],
                    per_category_usage[cat_j],
                )
                jsd_matrix[i, j] = jsd_val

    if n_cats > 1:
        mask = ~torch.eye(n_cats, dtype=torch.bool)
        mean_jsd = jsd_matrix[mask].mean().item()
    else:
        mean_jsd = 0.0

    return {
        "per_category_expert_usage": per_category_usage,
        "jsd_matrix": jsd_matrix,
        "mean_jsd": mean_jsd,
        "specialization_score": mean_jsd,
    }


# ---------------------------------------------------------------------------
# Collapse detection
# ---------------------------------------------------------------------------

def detect_expert_collapse(
    routing_stats: list[MoERoutingStats],
    collapse_threshold: float = 0.8,
) -> dict:
    """Detect if one expert is receiving more than collapse_threshold fraction of tokens.

    Args:
        routing_stats: list of MoERoutingStats from forward passes.
        collapse_threshold: fraction above which a layer is considered collapsed.

    Returns:
        dict with keys 'collapsed_layers', 'max_single_expert_load',
        'is_collapsed', 'collapse_score'.
    """
    if not routing_stats:
        return {
            "collapsed_layers": [],
            "max_single_expert_load": 0.0,
            "is_collapsed": False,
            "collapse_score": 0.0,
        }

    collapsed_layers: list[int] = []
    max_load = 0.0

    for stat in routing_stats:
        top_frac = stat.top_expert_fraction
        if top_frac > max_load:
            max_load = top_frac
        if top_frac > collapse_threshold:
            if stat.layer_idx not in collapsed_layers:
                collapsed_layers.append(stat.layer_idx)

    n_experts = routing_stats[0].n_experts
    # Collapse score: 0 when uniform, 1 when fully collapsed to one expert
    uniform_load = 1.0 / n_experts if n_experts > 0 else 1.0
    collapse_score = max(0.0, (max_load - uniform_load) / (1.0 - uniform_load + 1e-10))
    collapse_score = min(collapse_score, 1.0)

    return {
        "collapsed_layers": collapsed_layers,
        "max_single_expert_load": max_load,
        "is_collapsed": len(collapsed_layers) > 0,
        "collapse_score": collapse_score,
    }
