"""MCTS RL Trainer — AlphaZero-style MCTS policy/value training.

Implements the Monte Carlo Tree Search (MCTS) reinforcement learning training
objective used in AlphaZero-style LLM reasoning systems.  MCTS visit counts
and mean Q-values serve as training targets for a policy/value network:

    MCTS policy:    π_MCTS(a|s) = N(s,a)^(1/τ) / Σ N(s,a')^(1/τ)
    MCTS value:     v_MCTS(s)  = Q(s) = mean reward over MCTS rollouts

    Policy loss:    L_π = -Σ_a π_MCTS(a) · log π_θ(a|s)
    Value loss:     L_v = (v_θ(s) - v_MCTS(s))²
    Combined loss:  L   = L_π + λ · L_v

UCT exploration score (used during tree search):
    UCT(s,a) = Q(s,a) + c_puct · prior(a) · sqrt(N_parent) / (1 + N(s,a))

Pure native PyTorch only; no external dependencies beyond stdlib + torch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# MCTSRLConfig
# ---------------------------------------------------------------------------

@dataclass
class MCTSRLConfig:
    """Configuration for the MCTS RL Trainer.

    Attributes:
        temperature:   τ — temperature for visit-count normalisation.
                       Lower values sharpen the policy toward the highest-visited
                       action; higher values flatten it toward uniform.
        value_weight:  λ — weight on the value loss relative to the policy loss.
        n_simulations: Number of MCTS simulations to run per position.
        c_puct:        UCT exploration constant controlling the exploration /
                       exploitation trade-off during tree search.
        discount:      γ — future reward discount factor for value targets.
    """

    temperature: float = 1.0
    value_weight: float = 1.0
    n_simulations: int = 50
    c_puct: float = 1.4
    discount: float = 0.99


# ---------------------------------------------------------------------------
# MCTSNode
# ---------------------------------------------------------------------------

@dataclass
class MCTSNode:
    """A single node in the MCTS tree.

    Attributes:
        state_id:    Unique integer identifier for this state.
        visit_count: Number of times this node has been visited (N(s)).
        value_sum:   Accumulated value over all visits (sum of backed-up Q).
        prior:       Prior policy probability π_θ(a | parent state).
        children:    Mapping from action index to child state_id.
        parent_id:   state_id of the parent node, or None for the root.
    """

    state_id: int
    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 1.0
    children: dict[int, int] = field(default_factory=dict)
    parent_id: int | None = None

    @property
    def q_value(self) -> float:
        """Mean value estimate: value_sum / (visit_count + ε)."""
        return self.value_sum / (self.visit_count + 1e-8)


# ---------------------------------------------------------------------------
# MCTSStats
# ---------------------------------------------------------------------------

@dataclass
class MCTSStats:
    """Statistics produced by MCTS simulation for a single position.

    Attributes:
        state_id:     Identifier of the root state for this simulation.
        visit_counts: Integer visit counts per action, shape ``[A]``.
        action_values: Mean Q-value per action (backed-up from rollouts), shape ``[A]``.
        mcts_policy:  Temperature-normalised visit-count distribution, shape ``[A]``.
                      This is the target distribution for policy training.
    """

    state_id: int
    visit_counts: Tensor    # [A]
    action_values: Tensor   # [A]
    mcts_policy: Tensor     # [A]  temperature-normalised


# ---------------------------------------------------------------------------
# MCTSRLTrainer
# ---------------------------------------------------------------------------

class MCTSRLTrainer:
    """Stateless MCTS RL loss computation for policy/value networks.

    Accepts MCTS statistics produced externally (or synthetically) and
    computes cross-entropy policy loss + MSE value loss following the
    AlphaZero training objective.

    Args:
        config: :class:`MCTSRLConfig` instance.  If *None* defaults are used.
    """

    def __init__(self, config: MCTSRLConfig | None = None) -> None:
        self.config: MCTSRLConfig = config if config is not None else MCTSRLConfig()

    # ------------------------------------------------------------------
    # Visit-count normalisation
    # ------------------------------------------------------------------

    def normalize_visits(self, visit_counts: Tensor) -> Tensor:
        """Convert raw visit counts into a probability distribution.

        Applies temperature scaling:
            p(a) = N(a)^(1/τ) / Σ_a' N(a')^(1/τ)

        Args:
            visit_counts: Non-negative visit counts, shape ``[A]``.

        Returns:
            Probability distribution over actions, shape ``[A]``.
            Sums to 1.0; uses float dtype.
        """
        tau = self.config.temperature
        counts = visit_counts.float()
        if tau == 0.0:
            # Deterministic argmax: one-hot at the greedy action.
            idx = counts.argmax()
            policy = torch.zeros_like(counts)
            policy[idx] = 1.0
            return policy
        # Raise to the 1/τ power in log-space for numerical stability.
        log_counts = (counts + 1e-30).log() / tau
        return F.softmax(log_counts, dim=0)

    # ------------------------------------------------------------------
    # Policy loss
    # ------------------------------------------------------------------

    def policy_loss(self, log_policy: Tensor, mcts_policy: Tensor) -> Tensor:
        """Cross-entropy between the MCTS target distribution and the model log-policy.

        L_π = -mean_batch( sum_a π_MCTS(a) · log π_θ(a) )

        Args:
            log_policy:  Log-probabilities from the model, shape ``[B, A]``.
            mcts_policy: Target distribution from MCTS, shape ``[B, A]``.

        Returns:
            Scalar loss tensor.
        """
        # Element-wise product then sum over actions, negate, average over batch.
        return -(mcts_policy * log_policy).sum(dim=1).mean()

    # ------------------------------------------------------------------
    # Value loss
    # ------------------------------------------------------------------

    def value_loss(self, predicted_values: Tensor, mcts_values: Tensor) -> Tensor:
        """Mean-squared error between model value predictions and MCTS value targets.

        L_v = mean_batch( (v_θ(s) - v_MCTS(s))^2 )

        Args:
            predicted_values: Model value predictions, shape ``[B]``.
            mcts_values:      MCTS value targets, shape ``[B]``.

        Returns:
            Scalar loss tensor.
        """
        return F.mse_loss(predicted_values, mcts_values)

    # ------------------------------------------------------------------
    # Combined loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        log_policy: Tensor,
        predicted_values: Tensor,
        mcts_stats: list[MCTSStats],
    ) -> dict[str, Tensor]:
        """Compute the combined MCTS RL loss.

        Stacks per-position MCTS statistics into batched tensors and
        computes policy loss, value loss, total loss, and KL divergence
        from the MCTS target policy.

        Args:
            log_policy:       Model log-policy, shape ``[B, A]``.
            predicted_values: Model value predictions, shape ``[B]``.
            mcts_stats:       List of :class:`MCTSStats`, one per position (len B).

        Returns:
            Dictionary with scalar tensors:

            * ``"loss"``         -- λ·L_v + L_π (total combined loss).
            * ``"policy_loss"``  -- cross-entropy L_π.
            * ``"value_loss"``   -- MSE L_v.
            * ``"kl_from_mcts"`` -- KL(π_MCTS ‖ π_θ), for monitoring.
        """
        # Stack MCTS targets from the list of stats objects.
        mcts_policies = torch.stack([s.mcts_policy for s in mcts_stats], dim=0)   # [B, A]
        mcts_values   = torch.stack([s.action_values.mean() for s in mcts_stats]) # [B]

        l_pi = self.policy_loss(log_policy, mcts_policies)
        l_v  = self.value_loss(predicted_values, mcts_values)
        total = l_pi + self.config.value_weight * l_v

        # KL(π_MCTS ‖ π_θ) = Σ π_MCTS · (log π_MCTS - log π_θ)
        # Clamp mcts_policies to avoid log(0).
        log_mcts = (mcts_policies + 1e-30).log()
        kl = (mcts_policies * (log_mcts - log_policy)).sum(dim=1).mean()

        return {
            "loss": total,
            "policy_loss": l_pi,
            "value_loss": l_v,
            "kl_from_mcts": kl,
        }

    # ------------------------------------------------------------------
    # UCT score
    # ------------------------------------------------------------------

    def uct_score(self, node: MCTSNode, parent_visits: int) -> float:
        """Compute the UCT selection score for a child node.

        UCT(s,a) = Q(s,a) + c_puct · prior(a) · sqrt(N_parent) / (1 + N(s,a))

        Args:
            node:          Child :class:`MCTSNode` to score.
            parent_visits: Visit count of the parent node N(parent).

        Returns:
            UCT score as a plain Python float.
        """
        exploration = (
            self.config.c_puct
            * node.prior
            * math.sqrt(parent_visits)
            / (1 + node.visit_count)
        )
        return node.q_value + exploration

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def statistics(self, mcts_stats: list[MCTSStats]) -> dict[str, float]:
        """Compute diagnostic statistics from a batch of MCTS results.

        Args:
            mcts_stats: List of :class:`MCTSStats`.

        Returns:
            Dictionary of plain Python floats:

            * ``"mean_visit_count"`` -- average total visits per position.
            * ``"policy_entropy"``   -- average entropy of the MCTS policy.
            * ``"mean_q_value"``     -- average mean Q-value across positions.
        """
        with torch.no_grad():
            total_visits = []
            entropies = []
            mean_qs = []
            for s in mcts_stats:
                total_visits.append(s.visit_counts.float().sum().item())
                # Entropy: -Σ p log(p), guard against p=0.
                p = s.mcts_policy.float().clamp(min=1e-30)
                entropies.append(-(p * p.log()).sum().item())
                mean_qs.append(s.action_values.float().mean().item())

            return {
                "mean_visit_count": sum(total_visits) / max(len(total_visits), 1),
                "policy_entropy":   sum(entropies)    / max(len(entropies), 1),
                "mean_q_value":     sum(mean_qs)      / max(len(mean_qs), 1),
            }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from src.training import TRAINING_REGISTRY  # noqa: E402

TRAINING_REGISTRY["mcts_rl"] = MCTSRLTrainer
