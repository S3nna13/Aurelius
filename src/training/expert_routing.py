"""Expert routing analysis and load-balancing training utilities for MoE models.

Provides standalone functions and stateful classes for analyzing and regularizing
expert routing in Mixture-of-Experts training. These utilities operate on routing
tensors produced by the router (e.g. SparseMoEFFN or SwitchRouter) and are
deliberately decoupled from the router/layer implementations in src/model/moe.py
and src/model/moe_improvements.py.

Typical usage during a training step:

    router_probs, expert_mask = get_routing_tensors(...)   # from your MoE layer

    lb_loss   = compute_load_balance_loss(router_probs, expert_mask)
    ent_loss  = compute_router_entropy(router_probs)
    util_info = compute_expert_utilization(expert_mask)

    tracker.update(expert_mask)
    stats = tracker.get_stats()

    total_loss, loss_dict = routing_loss(task_loss, router_probs, expert_mask)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ExpertRoutingConfig:
    """Configuration for expert routing analysis and regularization.

    Attributes:
        n_experts:          Total number of experts in the MoE layer.
        n_active:           Number of top-k experts activated per token.
        load_balance_coeff: Scalar weight on the load-balance auxiliary loss.
        entropy_coeff:      Scalar weight on the router entropy regularizer.
                            Higher values encourage the router to spread
                            probability mass across more experts.
        track_history:      If True, ExpertLoadTracker stores history.
        history_window:     Number of tokens (not steps) kept in the rolling
                            window used by ExpertLoadTracker.
    """
    n_experts: int = 8
    n_active: int = 2
    load_balance_coeff: float = 0.01
    entropy_coeff: float = 0.001
    track_history: bool = True
    history_window: int = 1000


# ---------------------------------------------------------------------------
# Load-balance loss
# ---------------------------------------------------------------------------

def compute_load_balance_loss(router_probs: Tensor, expert_mask: Tensor) -> Tensor:
    """Switch Transformer load-balance loss over a (B, T, n_experts) batch.

    Computes the auxiliary loss that encourages even distribution of tokens
    across experts:

        L_lb = n_experts * sum_i( f_i * P_i )

    where
        f_i = fraction of tokens routed to expert i
              (computed from expert_mask, so it's a *discrete* dispatch count)
        P_i = mean router probability assigned to expert i across all tokens
              (computed from router_probs, so it's *differentiable*)

    The product f_i * P_i is minimised when both are equal (uniform routing),
    matching the formulation in Switch Transformer (Fedus et al. 2021).

    Args:
        router_probs: (B, T, n_experts) — softmax probabilities from the router.
        expert_mask:  (B, T, n_experts) — one-hot / top-k selection indicators
                      (float or bool; each position is 1 if that expert was
                      chosen for that token, 0 otherwise).

    Returns:
        Scalar tensor — the load-balance loss (no coefficient applied here;
        scale with ExpertRoutingConfig.load_balance_coeff externally or use
        RoutingAwareLoss which applies it automatically).
    """
    n_experts = router_probs.shape[-1]

    # f_i: fraction of total token-expert assignments going to expert i.
    # expert_mask may have multiple 1s per token (top-k), so normalise by
    # the total number of (token, expert) selection events.
    mask_float = expert_mask.float()                    # (B, T, E)
    total_selections = mask_float.sum().clamp(min=1.0)
    f_i = mask_float.sum(dim=(0, 1)) / total_selections  # (E,)

    # P_i: mean router probability per expert across all tokens.
    P_i = router_probs.mean(dim=(0, 1))                 # (E,)

    lb_loss = n_experts * (f_i * P_i).sum()
    return lb_loss


# ---------------------------------------------------------------------------
# Router entropy
# ---------------------------------------------------------------------------

def compute_router_entropy(router_probs: Tensor) -> Tensor:
    """Mean entropy of the router probability distribution over tokens.

    High entropy → the router spreads probability across many experts
    (diverse routing).  Low entropy → the router confidently picks one expert
    (collapsed routing, often undesirable).

        H = -sum_i( p_i * log(p_i + eps) )   per token
        returns mean(H) over all (B*T) tokens.

    Args:
        router_probs: (B, T, n_experts) — softmax probabilities from the router.

    Returns:
        Scalar tensor — mean entropy (nats).
    """
    entropy = -(router_probs * torch.log(router_probs + 1e-9)).sum(dim=-1)  # (B, T)
    return entropy.mean()


# ---------------------------------------------------------------------------
# Expert utilization
# ---------------------------------------------------------------------------

def compute_expert_utilization(expert_mask: Tensor) -> dict[str, Tensor]:
    """Per-expert utilization statistics from a selection mask.

    Args:
        expert_mask: (B, T, n_experts) — selection mask (float or bool).

    Returns:
        Dictionary with keys:
            "utilization"    — (n_experts,) float tensor: fraction of total
                               token-expert selections going to each expert.
            "max_util"       — scalar: highest per-expert utilization.
            "min_util"       — scalar: lowest per-expert utilization.
            "utilization_cv" — scalar: coefficient of variation (std / mean),
                               0 when load is perfectly balanced, higher when
                               routing is skewed.
    """
    mask_float = expert_mask.float()                    # (B, T, E)
    total_selections = mask_float.sum().clamp(min=1.0)
    utilization = mask_float.sum(dim=(0, 1)) / total_selections  # (E,)

    max_util = utilization.max()
    min_util = utilization.min()

    mean_util = utilization.mean()
    std_util = utilization.std(unbiased=False)
    # CV is undefined when mean is 0; guard with a small epsilon.
    utilization_cv = std_util / mean_util.clamp(min=1e-9)

    return {
        "utilization": utilization,
        "max_util": max_util,
        "min_util": min_util,
        "utilization_cv": utilization_cv,
    }


# ---------------------------------------------------------------------------
# Running load tracker
# ---------------------------------------------------------------------------

class ExpertLoadTracker:
    """Running statistics over a sliding window of expert assignments.

    Maintains a fixed-size deque of per-expert token-counts.  Each call to
    ``update`` pushes the latest batch's counts in and (if the buffer is full)
    pops the oldest counts out, giving an approximate sliding-window view of
    how tokens have been distributed across experts over the last
    ``config.history_window`` tokens.

    Args:
        config: ExpertRoutingConfig — uses n_experts, history_window, and
                track_history.
    """

    def __init__(self, config: ExpertRoutingConfig) -> None:
        self.config = config
        # Store per-expert raw counts (not fractions) so we can sum correctly.
        # Each element is a 1-D list/tensor of length n_experts.
        self._history: deque[list[float]] = deque()
        self._total_tokens: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, expert_mask: Tensor) -> None:
        """Accumulate token assignments from a new batch.

        Args:
            expert_mask: (B, T, n_experts) — selection mask (float or bool).
                         Each '1' entry counts as one token dispatched to that
                         expert.
        """
        if not self.config.track_history:
            return

        mask_float = expert_mask.float()
        # Per-expert counts for this batch (scalar per expert).
        counts = mask_float.sum(dim=(0, 1))  # (E,)
        batch_tokens = int(mask_float.sum().item())

        counts_list = counts.detach().cpu().tolist()
        self._history.append(counts_list)
        self._total_tokens += batch_tokens

        # Trim: drop oldest entries until window constraint is satisfied.
        # We approximate "window" in units of tokens; pop until we're ≤ window.
        while self._total_tokens > self.config.history_window and len(self._history) > 1:
            oldest = self._history.popleft()
            self._total_tokens -= int(sum(oldest))

    def get_stats(self) -> dict[str, float]:
        """Compute summary statistics from the current history window.

        Returns:
            Dictionary with keys:
                "mean_utilization" — float: average fraction of tokens per
                                     expert (should be ~1/n_experts when balanced).
                "cv"               — float: coefficient of variation of
                                     per-expert utilization (0 = perfect balance).
                "dead_experts"     — int (as float): number of experts that
                                     received strictly less than 1 % of tokens
                                     over the history window.
        """
        n_experts = self.config.n_experts

        if not self._history:
            return {
                "mean_utilization": 0.0,
                "cv": 0.0,
                "dead_experts": 0.0,
            }

        # Aggregate counts over the entire history window.
        agg = [0.0] * n_experts
        total = 0.0
        for counts in self._history:
            for i, c in enumerate(counts):
                agg[i] += c
            total += sum(counts)

        total = max(total, 1e-9)
        utilization = [c / total for c in agg]

        mean_util = sum(utilization) / n_experts
        variance = sum((u - mean_util) ** 2 for u in utilization) / n_experts
        std_util = variance ** 0.5
        cv = std_util / max(mean_util, 1e-9)

        dead_experts = sum(1 for u in utilization if u < 0.01)

        return {
            "mean_utilization": mean_util,
            "cv": cv,
            "dead_experts": float(dead_experts),
        }

    def reset(self) -> None:
        """Clear all accumulated history."""
        self._history.clear()
        self._total_tokens = 0


# ---------------------------------------------------------------------------
# Combined routing-aware loss
# ---------------------------------------------------------------------------

class RoutingAwareLoss:
    """Combine task loss with load-balance and entropy regularization.

    Wraps the three loss components into one callable that returns both the
    combined scalar and a breakdown dict for logging.

        total = task_loss
              + load_balance_coeff * compute_load_balance_loss(...)
              + entropy_coeff      * (-compute_router_entropy(...))

    Note: we *negate* the entropy because higher entropy is desirable.
    Minimizing ``-H`` encourages the router to be more uniform.

    Args:
        config: ExpertRoutingConfig.
    """

    def __init__(self, config: ExpertRoutingConfig) -> None:
        self.config = config

    def __call__(
        self,
        task_loss: Tensor,
        router_probs: Tensor,
        expert_mask: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute total loss and return component breakdown.

        Args:
            task_loss:    Scalar tensor — primary language-modelling (or other)
                          loss from the model.
            router_probs: (B, T, n_experts) — softmax router probabilities.
            expert_mask:  (B, T, n_experts) — top-k selection mask.

        Returns:
            (total_loss, info_dict) where info_dict has keys:
                "task_loss"         — the unmodified task_loss
                "load_balance_loss" — weighted load-balance contribution
                "entropy_loss"      — weighted entropy regularization contribution
                                      (negative entropy, so minimizing this
                                      maximizes diversity)
        """
        lb_raw = compute_load_balance_loss(router_probs, expert_mask)
        ent_raw = compute_router_entropy(router_probs)

        load_balance_loss = self.config.load_balance_coeff * lb_raw
        # Negative entropy: minimizing this term maximizes router entropy.
        entropy_loss = self.config.entropy_coeff * (-ent_raw)

        total_loss = task_loss + load_balance_loss + entropy_loss

        info = {
            "task_loss": task_loss,
            "load_balance_loss": load_balance_loss,
            "entropy_loss": entropy_loss,
        }
        return total_loss, info
