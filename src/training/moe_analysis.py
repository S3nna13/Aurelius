"""MoE routing analysis: expert utilization, specialization, and routing visualization tools."""
from __future__ import annotations

import torch
from torch import Tensor


def compute_routing_entropy(router_probs: Tensor) -> Tensor:
    """Compute per-token routing entropy.

    Args:
        router_probs: (B*T, n_experts) — softmax probabilities from router.

    Returns:
        Tensor of shape (B*T,) — per-token entropy in nats.
        Higher entropy means more uniform routing across experts.
    """
    # -sum(p * log(p + eps), dim=-1)
    return -(router_probs * torch.log(router_probs + 1e-10)).sum(dim=-1)


def compute_expert_specialization(
    router_probs: Tensor,
    token_labels: Tensor,
    n_classes: int,
) -> Tensor:
    """Compute expert specialization matrix over token class labels.

    For each expert e and class c, computes the mean routing weight of
    class-c tokens to expert e, then row-normalizes so each expert's row
    sums to 1.

    Args:
        router_probs:  (N, n_experts) — softmax routing probabilities.
        token_labels:  (N,) — integer class labels in [0, n_classes).
        n_classes:     Number of distinct token classes.

    Returns:
        Tensor of shape (n_experts, n_classes) — row-normalized specialization
        matrix. Row e gives expert e's distribution over classes.
    """
    n_experts = router_probs.shape[1]
    device = router_probs.device
    dtype = router_probs.dtype

    spec = torch.zeros(n_experts, n_classes, device=device, dtype=dtype)
    counts = torch.zeros(n_classes, device=device, dtype=dtype)

    for c in range(n_classes):
        mask = token_labels == c           # (N,) bool
        n_c = mask.sum().item()
        if n_c == 0:
            continue
        counts[c] = float(n_c)
        # mean routing weight from class-c tokens to each expert
        spec[:, c] = router_probs[mask].mean(dim=0)  # (n_experts,)

    # Row-normalize: each expert's row sums to 1
    row_sums = spec.sum(dim=-1, keepdim=True)          # (n_experts, 1)
    row_sums = row_sums.clamp(min=1e-10)
    spec = spec / row_sums

    return spec


def gini_coefficient(x: Tensor) -> float:
    """Classic Gini coefficient of a non-negative 1-D tensor.

    Gini = sum of absolute differences / (2 * n * mean)

    Args:
        x: 1-D tensor of non-negative values.

    Returns:
        Float in [0, 1]. 0 = perfectly equal distribution.
    """
    x = x.float().flatten()
    n = x.numel()
    if n == 0:
        return 0.0
    mean_val = x.mean().item()
    if mean_val == 0.0:
        return 0.0
    # sum of absolute pairwise differences
    diff_sum = (x.unsqueeze(0) - x.unsqueeze(1)).abs().sum().item()
    return diff_sum / (2.0 * n * n * mean_val)


def compute_load_imbalance(router_probs: Tensor) -> dict:
    """Compute expert load imbalance statistics.

    Args:
        router_probs: (N, n_experts) — softmax routing probabilities.

    Returns:
        dict with keys:
            "load_per_expert": Tensor (n_experts,) — mean weight per expert
            "cv":              float — coefficient of variation (std / mean)
            "max_load":        float
            "min_load":        float
            "gini":            float — Gini coefficient of load distribution
    """
    load_per_expert = router_probs.mean(dim=0)  # (n_experts,)

    mean_load = load_per_expert.mean().item()
    std_load = load_per_expert.std(unbiased=False).item()
    cv = std_load / max(mean_load, 1e-10)

    return {
        "load_per_expert": load_per_expert,
        "cv": cv,
        "max_load": load_per_expert.max().item(),
        "min_load": load_per_expert.min().item(),
        "gini": gini_coefficient(load_per_expert),
    }


def track_routing_over_time(router_prob_history: list[Tensor]) -> dict:
    """Track per-expert load across training steps.

    Args:
        router_prob_history: List of (N, n_experts) tensors, one per step.

    Returns:
        dict with keys:
            "load_trajectory": Tensor (n_steps, n_experts) — per-expert mean load
            "mean_cv":         float — mean CV across all steps
            "load_variance":   Tensor (n_experts,) — variance of each expert's load
    """
    step_loads = []
    cvs = []

    for probs in router_prob_history:
        stats = compute_load_imbalance(probs)
        step_loads.append(stats["load_per_expert"])
        cvs.append(stats["cv"])

    load_trajectory = torch.stack(step_loads, dim=0)   # (n_steps, n_experts)
    mean_cv = float(sum(cvs) / len(cvs)) if cvs else 0.0
    load_variance = load_trajectory.var(dim=0, unbiased=False)  # (n_experts,)

    return {
        "load_trajectory": load_trajectory,
        "mean_cv": mean_cv,
        "load_variance": load_variance,
    }


class MoEAnalyzer:
    """Stateful accumulator for MoE routing analysis over a training run.

    Args:
        n_experts: Total number of experts.
        top_k:     Number of experts activated per token (informational only).
    """

    def __init__(self, n_experts: int, top_k: int = 2) -> None:
        self.n_experts = n_experts
        self.top_k = top_k
        self._router_probs: list[Tensor] = []
        self._n_tokens_seen: int = 0

    def record_batch(self, router_probs: Tensor) -> None:
        """Accumulate router probabilities from one batch.

        Args:
            router_probs: (N, n_experts) — softmax routing probabilities.
        """
        self._router_probs.append(router_probs.detach().cpu())
        self._n_tokens_seen += router_probs.shape[0]

    def summary(self) -> dict:
        """Compute summary statistics over all recorded batches.

        Returns:
            dict with keys:
                "mean_entropy":    float — mean per-token routing entropy
                "load_imbalance":  dict from compute_load_imbalance
                "n_tokens_seen":   int — total tokens accumulated
        """
        all_probs = torch.cat(self._router_probs, dim=0)   # (N_total, n_experts)

        entropy = compute_routing_entropy(all_probs)         # (N_total,)
        mean_entropy = entropy.mean().item()

        load_imbalance = compute_load_imbalance(all_probs)

        return {
            "mean_entropy": mean_entropy,
            "load_imbalance": load_imbalance,
            "n_tokens_seen": self._n_tokens_seen,
        }

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._router_probs = []
        self._n_tokens_seen = 0
