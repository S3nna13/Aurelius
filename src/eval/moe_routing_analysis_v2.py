"""Mixture of Experts Routing Analysis — v2.

Diagnostic tools for understanding load balance, routing collapse,
and expert utilization in MoE layers.  All classes operate on plain
PyTorch tensors; no external dependencies beyond stdlib + torch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MoEAnalysisConfig:
    """Configuration knobs for MoE diagnostic thresholds."""
    n_experts: int = 8
    top_k: int = 2
    collapse_threshold: float = 0.01   # fraction below which an expert is "dead"
    entropy_threshold: float = 0.5     # nats; below this routing is "low-entropy"
    imbalance_threshold: float = 0.3   # std of utilization; above this is imbalanced


# ---------------------------------------------------------------------------
# RouterStats
# ---------------------------------------------------------------------------

class RouterStats:
    """Accumulates per-expert token counts and routing weight statistics
    across multiple batches, then exposes derived metrics.

    Parameters
    ----------
    n_experts : int
        Total number of experts in the MoE layer.
    top_k : int
        Number of experts selected per token.
    """

    def __init__(self, n_experts: int, top_k: int) -> None:
        self.n_experts = n_experts
        self.top_k = top_k
        self.reset()

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def update(
        self,
        routing_weights: torch.Tensor,  # [B*T, n_experts]  — full softmax/gate probs
        selected_experts: torch.Tensor, # [B*T, top_k]      — indices of chosen experts
    ) -> None:
        """Accumulate one batch of routing statistics.

        Parameters
        ----------
        routing_weights : Tensor [B*T, n_experts]
            Full probability distribution produced by the router (e.g. softmax).
        selected_experts : Tensor [B*T, top_k]
            Integer indices of the top-k selected experts for each token.
        """
        bt = routing_weights.shape[0]

        # --- token counts per expert ---
        # one-hot across top_k selections, then sum over tokens
        counts = torch.zeros(self.n_experts, dtype=torch.float32)
        for k in range(selected_experts.shape[1]):
            col = selected_experts[:, k]  # [B*T]
            for e in range(self.n_experts):
                counts[e] += (col == e).sum().item()

        self._expert_token_counts += counts
        self._total_tokens += bt

        # --- mean routing probability (used for load-balance loss) ---
        self._routing_weight_sum += routing_weights.detach().float().sum(dim=0)  # [n_experts]
        self._routing_weight_count += bt

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def expert_utilization(self) -> torch.Tensor:
        """Fraction of (token, slot) assignments going to each expert.

        The values sum to ``top_k``; divide by ``top_k`` to get a proper
        probability vector.  Equivalently, if perfectly balanced every
        expert should receive ``top_k / n_experts`` of all assignments.

        Returns
        -------
        Tensor [n_experts]
        """
        if self._total_tokens == 0:
            return torch.zeros(self.n_experts)
        total_assignments = float(self._total_tokens * self.top_k)
        return self._expert_token_counts / total_assignments

    def load_balance_loss(self) -> float:
        """Auxiliary load-balance loss: ``n_experts * sum(f_i * p_i)``.

        ``f_i`` is the fraction of tokens assigned to expert ``i``, and
        ``p_i`` is the mean gate probability for expert ``i``.  A perfect
        uniform distribution gives a value of ``1/n_experts * top_k``.

        Returns
        -------
        float  (≥ 0)
        """
        if self._total_tokens == 0 or self._routing_weight_count == 0:
            return 0.0
        f = self.expert_utilization()                              # [n_experts]
        p = self._routing_weight_sum / self._routing_weight_count  # [n_experts]
        loss = float(self.n_experts * (f * p).sum().item())
        return max(loss, 0.0)

    def routing_entropy(self) -> float:
        """Shannon entropy H(p̄) of the mean routing distribution (nats).

        A uniform distribution over ``n_experts`` gives ``ln(n_experts)``.

        Returns
        -------
        float  (≥ 0)
        """
        if self._routing_weight_count == 0:
            return 0.0
        p_mean = self._routing_weight_sum / self._routing_weight_count  # [n_experts]
        # clamp to avoid log(0)
        p_mean = p_mean.clamp(min=1e-9)
        p_mean = p_mean / p_mean.sum()  # re-normalise in case of floating-point drift
        entropy = float(-(p_mean * p_mean.log()).sum().item())
        return max(entropy, 0.0)

    def collapse_score(self) -> float:
        """Fraction of experts receiving < ``collapse_threshold * (top_k/n_experts)``
        of all token assignments (defaults to < 1 % of tokens).

        More precisely: fraction of experts whose utilization is below 1 %
        of the *total assignments* share (i.e., utilization < 0.01).

        Returns
        -------
        float in [0, 1]
        """
        util = self.expert_utilization()  # sums to 1 across n_experts (after normalising)
        # re-normalise to a proper probability vector for the threshold
        total = util.sum()
        if total <= 0:
            return 0.0
        prob = util / total  # sums to 1
        threshold = 0.01
        dead = (prob < threshold).sum().item()
        return float(dead) / self.n_experts

    def reset(self) -> None:
        """Clear all accumulated statistics."""
        self._expert_token_counts: torch.Tensor = torch.zeros(self.n_experts)
        self._routing_weight_sum: torch.Tensor = torch.zeros(self.n_experts)
        self._routing_weight_count: int = 0
        self._total_tokens: int = 0


# ---------------------------------------------------------------------------
# ExpertActivationTracker
# ---------------------------------------------------------------------------

class ExpertActivationTracker:
    """Tracks which experts are activated together and the tokens they process.

    Parameters
    ----------
    n_experts : int
        Total number of experts.
    """

    def __init__(self, n_experts: int) -> None:
        self.n_experts = n_experts
        self._co_act: torch.Tensor = torch.zeros(n_experts, n_experts, dtype=torch.float32)
        self._n_batches: int = 0

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def record_batch(self, expert_ids: torch.Tensor) -> None:
        """Record which experts were activated for each token.

        Parameters
        ----------
        expert_ids : Tensor [B*T, top_k]
            Integer expert indices selected per token.
        """
        bt = expert_ids.shape[0]
        top_k = expert_ids.shape[1]
        for token_idx in range(bt):
            ids = expert_ids[token_idx]  # [top_k]
            for i in range(top_k):
                for j in range(top_k):
                    ei = int(ids[i].item())
                    ej = int(ids[j].item())
                    self._co_act[ei, ej] += 1.0
        self._n_batches += bt

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def co_activation_matrix(self) -> torch.Tensor:
        """Normalised co-activation matrix.

        Entry ``[i, j]`` is the fraction of tokens where both expert ``i``
        and expert ``j`` appeared in the top-k selection.  The diagonal
        holds per-expert activation frequencies (and is always ≥ off-diag).

        Returns
        -------
        Tensor [n_experts, n_experts]
        """
        if self._n_batches == 0:
            return self._co_act.clone()
        return self._co_act / max(self._co_act.max().item(), 1.0)

    def expert_specialization(
        self,
        features: torch.Tensor,  # [N, d]
        expert_ids: torch.Tensor, # [N]  — single expert index per token
    ) -> torch.Tensor:
        """Compute the mean feature vector (centroid) for each expert.

        Parameters
        ----------
        features : Tensor [N, d]
        expert_ids : Tensor [N]   (integer, one expert per token)

        Returns
        -------
        Tensor [n_experts, d]
            Zero vector for experts that received no tokens.
        """
        n, d = features.shape
        centroids = torch.zeros(self.n_experts, d, dtype=features.dtype)
        counts = torch.zeros(self.n_experts, dtype=torch.float32)
        for e in range(self.n_experts):
            mask = expert_ids == e
            if mask.any():
                centroids[e] = features[mask].mean(dim=0)
                counts[e] = mask.sum().float()
        return centroids

    def top_tokens_per_expert(
        self,
        token_ids: torch.Tensor,  # [N]  — integer token ids
        expert_ids: torch.Tensor, # [N]  — single expert index per token
        k: int = 5,
    ) -> Dict[int, List[int]]:
        """Return the ``k`` most frequent token ids routed to each expert.

        Parameters
        ----------
        token_ids : Tensor [N]
        expert_ids : Tensor [N]
        k : int

        Returns
        -------
        dict mapping expert_index → list of up to k token_ids (most-frequent first)
        """
        result: Dict[int, List[int]] = {}
        for e in range(self.n_experts):
            mask = expert_ids == e
            if not mask.any():
                result[e] = []
                continue
            tids = token_ids[mask]
            # count occurrences
            counts: Dict[int, int] = {}
            for tid in tids.tolist():
                tid = int(tid)
                counts[tid] = counts.get(tid, 0) + 1
            sorted_tids = sorted(counts.keys(), key=lambda x: counts[x], reverse=True)
            result[e] = sorted_tids[:k]
        return result


# ---------------------------------------------------------------------------
# RoutingDiversityMetrics
# ---------------------------------------------------------------------------

class RoutingDiversityMetrics:
    """Stateless collection of diversity / consistency metrics for MoE routing."""

    @staticmethod
    def token_routing_entropy(routing_weights: torch.Tensor) -> torch.Tensor:
        """Per-token Shannon entropy of the routing distribution (nats).

        Parameters
        ----------
        routing_weights : Tensor [N, n_experts]
            Routing probabilities for each token.

        Returns
        -------
        Tensor [N]  — entropy per token (≥ 0)
        """
        p = routing_weights.float().clamp(min=1e-9)
        p = p / p.sum(dim=-1, keepdim=True)  # re-normalise
        entropy = -(p * p.log()).sum(dim=-1)
        return entropy.clamp(min=0.0)

    @staticmethod
    def expert_similarity(expert_weights: torch.Tensor) -> torch.Tensor:
        """Pairwise cosine similarity between expert FFN weight matrices.

        Parameters
        ----------
        expert_weights : Tensor [n_experts, d_ff]
            One row per expert (e.g. flattened weight matrix).

        Returns
        -------
        Tensor [n_experts, n_experts]  — values in [-1, 1]
        """
        normed = F.normalize(expert_weights.float(), p=2, dim=-1)
        sim = normed @ normed.T
        return sim

    @staticmethod
    def routing_consistency(
        weights_t1: torch.Tensor,  # [N, n_experts]
        weights_t2: torch.Tensor,  # [N, n_experts]
    ) -> float:
        """Mean cosine similarity between per-token routing vectors at two steps.

        Parameters
        ----------
        weights_t1, weights_t2 : Tensor [N, n_experts]

        Returns
        -------
        float in [0, 1]  (clipped from [-1,1] to [0,1] for interpretability)
        """
        n1 = F.normalize(weights_t1.float(), p=2, dim=-1)  # [N, n_experts]
        n2 = F.normalize(weights_t2.float(), p=2, dim=-1)
        cos_sim = (n1 * n2).sum(dim=-1)  # [N]
        mean_sim = float(cos_sim.mean().item())
        # clip to [0, 1]
        return max(0.0, min(1.0, mean_sim))


# ---------------------------------------------------------------------------
# MoEDiagnostics
# ---------------------------------------------------------------------------

class MoEDiagnostics:
    """High-level diagnostic wrapper combining RouterStats and ExpertActivationTracker.

    Parameters
    ----------
    router_stats : RouterStats
    tracker : ExpertActivationTracker
    """

    def __init__(
        self,
        router_stats: RouterStats,
        tracker: ExpertActivationTracker,
    ) -> None:
        self.router_stats = router_stats
        self.tracker = tracker

    def full_report(self) -> Dict[str, float]:
        """Compute all key metrics and return as a flat dict.

        Returns
        -------
        dict with keys:
            utilization_std, load_balance_loss, routing_entropy,
            collapse_score, mean_co_activation
        """
        util = self.router_stats.expert_utilization()
        util_std = float(util.std().item())

        co_act = self.tracker.co_activation_matrix()
        # mean of off-diagonal elements (co-activation between different experts)
        n = self.tracker.n_experts
        mask = ~torch.eye(n, dtype=torch.bool)
        if mask.any():
            mean_co = float(co_act[mask].mean().item())
        else:
            mean_co = 0.0

        return {
            "utilization_std": util_std,
            "load_balance_loss": self.router_stats.load_balance_loss(),
            "routing_entropy": self.router_stats.routing_entropy(),
            "collapse_score": self.router_stats.collapse_score(),
            "mean_co_activation": mean_co,
        }

    def detect_issues(self) -> List[str]:
        """Return a list of warning strings for detected routing problems.

        Possible warnings
        -----------------
        ``"LOAD_IMBALANCE"``   — utilization_std > 0.3
        ``"ROUTING_COLLAPSE"`` — collapse_score > 0.5
        ``"LOW_ENTROPY"``      — routing_entropy < 0.5

        Returns
        -------
        list[str]  — empty if no issues detected
        """
        issues: List[str] = []
        report = self.full_report()

        if report["utilization_std"] > 0.3:
            issues.append("LOAD_IMBALANCE")
        if report["collapse_score"] > 0.5:
            issues.append("ROUTING_COLLAPSE")
        if report["routing_entropy"] < 0.5:
            issues.append("LOW_ENTROPY")

        return issues
