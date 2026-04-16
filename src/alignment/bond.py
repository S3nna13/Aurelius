"""Aurelius 1.3B — BOND: Best-of-N Distillation (arXiv:2407.14622).

BOND trains a policy to imitate the best-of-N sampling distribution.
Given N responses scored by a reward model, the best response(s) are used
as targets. Soft BOND weights all N responses by their reward-softmax;
Hard BOND reduces to imitating only the argmax response.

The key insight: distill the implicit best-of-N policy (which achieves
high reward) back into the generative model, avoiding explicit RL.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BONDConfig:
    """Configuration for BOND alignment training."""

    n_samples: int = 8          # N candidates sampled per prompt
    temperature: float = 1.0    # Reward softmax temperature (soft BOND)
    kl_coef: float = 0.1        # KL-regularization coefficient vs. ref model
    hard_bond: bool = False      # If True use top-1 (hard) instead of soft weighting
    reward_scaling: bool = True  # Normalise rewards to zero-mean unit-var before weighting


# ---------------------------------------------------------------------------
# Stand-alone utility: bootstrap J*(n) estimate
# ---------------------------------------------------------------------------

def compute_j_star(
    rewards: torch.Tensor,
    n: int,
    n_bootstrap: int = 1000,
) -> float:
    """Bootstrap estimate of J*(n) = E[max(r_1, ..., r_n)].

    Args:
        rewards: 1-D tensor of reward samples drawn i.i.d. from the reward
                 distribution (any length >= n).
        n:       Best-of-N sample size.
        n_bootstrap: Number of bootstrap draws.

    Returns:
        Scalar float estimate of E[max_{i=1..n} r_i].
    """
    rewards = rewards.float().cpu()
    m = rewards.numel()
    if m < n:
        raise ValueError(f"Need at least n={n} reward samples, got {m}.")

    # Draw n_bootstrap groups of size n (with replacement) and take max of each.
    indices = torch.randint(0, m, (n_bootstrap, n))  # (n_bootstrap, n)
    samples = rewards[indices]                        # (n_bootstrap, n)
    maxima = samples.max(dim=1).values               # (n_bootstrap,)
    return maxima.mean().item()


# ---------------------------------------------------------------------------
# BONDTrainer
# ---------------------------------------------------------------------------

class BONDTrainer:
    """Trains a policy model using Best-of-N Distillation.

    Args:
        policy_model: The trainable policy (nn.Module).
        ref_model:    Frozen reference model for KL regularization.
        n_samples:    Number of candidates per prompt.
        temperature:  Softmax temperature applied to rewards (soft BOND).
        kl_coef:      Weight of KL penalty term.
        hard_bond:    If True, use hard (top-1) weighting instead of soft.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        n_samples: int = 8,
        temperature: float = 1.0,
        kl_coef: float = 0.1,
        hard_bond: bool = False,
    ) -> None:
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.n_samples = n_samples
        self.temperature = temperature
        self.kl_coef = kl_coef
        self.hard_bond = hard_bond

    # ------------------------------------------------------------------
    # Core weight computation
    # ------------------------------------------------------------------

    def compute_bond_weights(
        self,
        rewards: torch.Tensor,
        n: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Compute per-sample BOND weights.

        Args:
            rewards:     1-D tensor of shape (batch * n,).  The rewards are
                         ordered so that positions [i*n : (i+1)*n] belong to
                         the i-th prompt.
            n:           Group size (number of candidates per prompt).
            temperature: Softmax temperature; only used in soft BOND mode.
                         Ignored when self.hard_bond is True.

        Returns:
            weights: 1-D tensor of shape (batch * n,), summing to 1 within
                     each group of n consecutive elements.
        """
        rewards = rewards.float()
        total = rewards.numel()
        if total % n != 0:
            raise ValueError(
                f"rewards length {total} is not divisible by n={n}."
            )
        batch = total // n
        # Reshape to (batch, n) for group-wise operations.
        r = rewards.view(batch, n)

        if self.hard_bond:
            # One-hot: 1.0 at argmax, 0.0 elsewhere.
            argmax = r.argmax(dim=1, keepdim=True)          # (batch, 1)
            weights = torch.zeros_like(r)
            weights.scatter_(1, argmax, 1.0)
        else:
            # Soft: temperature-scaled softmax over rewards in each group.
            scaled = r / max(temperature, 1e-8)
            weights = F.softmax(scaled, dim=1)              # (batch, n)

        return weights.view(-1)  # (batch * n,)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_bond_loss(
        self,
        policy_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted NLL + KL regularization loss.

        L = -sum_i(w_i * log pi(y_i|x)) + kl_coef * KL(pi || pi_ref)

        where KL(pi || pi_ref) ≈ sum_i w_i * (log pi(y_i) - log pi_ref(y_i)).

        Args:
            policy_log_probs: (batch * n,) log-probabilities under the policy.
            ref_log_probs:    (batch * n,) log-probabilities under ref model.
            weights:          (batch * n,) BOND weights (sum to 1 per group).

        Returns:
            Scalar loss tensor.
        """
        policy_log_probs = policy_log_probs.float()
        ref_log_probs = ref_log_probs.float()
        weights = weights.float()

        # Weighted negative log-likelihood (distillation target).
        bond_loss = -(weights * policy_log_probs).sum()

        # KL divergence: E_w[log pi - log pi_ref].
        kl_loss = (weights * (policy_log_probs - ref_log_probs)).sum()

        total_loss = bond_loss + self.kl_coef * kl_loss
        return total_loss

    # ------------------------------------------------------------------
    # J*(n) approximation
    # ------------------------------------------------------------------

    def j_star_approximation(
        self,
        rewards: torch.Tensor,
        n: int,
    ) -> torch.Tensor:
        """Estimate J*(n) = E[max(r_1, ..., r_n)] from the current batch.

        The rewards tensor is assumed to contain groups of size n (one group
        per prompt), so we take the max within each group and average.

        Args:
            rewards: 1-D tensor of shape (batch * n,).
            n:       Candidates per prompt.

        Returns:
            Scalar tensor — the average within-group maximum reward.
        """
        rewards = rewards.float()
        batch = rewards.numel() // n
        r = rewards.view(batch, n)
        return r.max(dim=1).values.mean()

    # ------------------------------------------------------------------
    # Train step
    # ------------------------------------------------------------------

    def train_step(
        self,
        input_ids: torch.Tensor,
        ref_log_probs: torch.Tensor,
        policy_log_probs: torch.Tensor,
        rewards: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Single BOND training step.

        Args:
            input_ids:        (batch * n, seq_len) token ids (unused in the
                              loss directly, provided for potential future use).
            ref_log_probs:    (batch * n,) sequence log-probs under ref model.
            policy_log_probs: (batch * n,) sequence log-probs under policy.
            rewards:          (batch * n,) scalar rewards per candidate.

        Returns:
            Dict with keys:
              - "loss"        total loss (bond + kl)
              - "bond_loss"   unregularized weighted NLL component
              - "kl_loss"     KL divergence component
              - "j_star"      estimated J*(n) for this batch
              - "effective_n" effective sample size from soft weights
        """
        # Optionally scale rewards to zero-mean / unit-variance.
        r = rewards.float()

        # Compute BOND weights (using instance temperature).
        weights = self.compute_bond_weights(r, self.n_samples, self.temperature)

        # --- individual loss components ---
        policy_lp = policy_log_probs.float()
        ref_lp = ref_log_probs.float()

        bond_loss = -(weights * policy_lp).sum()
        kl_loss = (weights * (policy_lp - ref_lp)).sum()
        loss = bond_loss + self.kl_coef * kl_loss

        # J*(n) estimate.
        j_star = self.j_star_approximation(rewards, self.n_samples)

        # Effective N: exp(H(w)) where H is entropy of the weight distribution.
        # Measures how many candidates are effectively contributing.
        batch = weights.numel() // self.n_samples
        w_groups = weights.view(batch, self.n_samples)  # (batch, n)
        # Avoid log(0) by clamping.
        entropy = -(w_groups * (w_groups + 1e-10).log()).sum(dim=1).mean()
        effective_n = entropy.exp()

        return {
            "loss": loss,
            "bond_loss": bond_loss,
            "kl_loss": kl_loss,
            "j_star": j_star,
            "effective_n": effective_n,
        }
