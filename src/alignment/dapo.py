"""DAPO: Decoupled Clip and Dynamic Sampling Policy Optimization.

Implements the four key techniques from:
  "DAPO: An Open-Source LLM Reinforcement Learning System at Scale"
  arXiv:2503.14476 (Qwen Team, 2025)

Techniques:
  1. Clip-Higher / Decoupled Clipping (Section 3.1)
  2. Dynamic Sampling (Section 3.2)
  3. Token-level Policy Gradient (Section 3.3)
  4. Entropy Bonus (Section 3.4)
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 1. Clip-Higher: Decoupled Clipping Loss (Section 3.1)
# ---------------------------------------------------------------------------


class DAPOLoss(nn.Module):
    """Decoupled clip policy-gradient loss with optional entropy bonus.

    Standard PPO uses a symmetric clip: r ∈ [1-ε, 1+ε].
    DAPO decouples the clip bounds:
      - Positive advantage (A > 0): allow larger upward movement → clip at 1+ε_high
      - Negative advantage (A < 0): restrict downward movement → clip at 1-ε_low

    Formally (Section 3.1):
        r_t = π_θ(a_t|s_t) / π_old(a_t|s_t)
        clipped_r_t = clip(r_t, 1 - ε_low, 1 + ε_high)
        L_clip = -E_t[min(r_t * A_t, clipped_r_t * A_t)]

    Token-level normalization (Section 3.3):
        L_token = Σ_t L_clip_t / Σ_t 1

    Entropy bonus (Section 3.4):
        L = L_token - β_entropy * H(π)

    Args:
        eps_low: ε_low — lower clip bound offset for negative advantages.
        eps_high: ε_high — upper clip bound offset for positive advantages.
        beta_entropy: β_entropy — coefficient for entropy regularization bonus.
    """

    def __init__(
        self,
        eps_low: float = 0.1,
        eps_high: float = 0.2,
        beta_entropy: float = 0.001,
    ) -> None:
        super().__init__()
        if eps_low < 0:
            raise ValueError(f"eps_low must be >= 0, got {eps_low}")
        if eps_high < 0:
            raise ValueError(f"eps_high must be >= 0, got {eps_high}")
        if beta_entropy < 0:
            raise ValueError(f"beta_entropy must be >= 0, got {beta_entropy}")
        self.eps_low = eps_low
        self.eps_high = eps_high
        self.beta_entropy = beta_entropy

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        entropy: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute DAPO loss.

        Args:
            log_probs: Log probabilities under current policy π_θ.
                       Shape: (batch, seq_len) or (N,).
            old_log_probs: Log probabilities under old policy π_old (detached).
                           Same shape as log_probs.
            advantages: Advantage estimates A_t. Same shape as log_probs.
            entropy: Optional per-token entropy -Σ π log π. Same shape as log_probs.
                     If None and beta_entropy > 0, entropy term is skipped with zero
                     contribution (caller must supply entropy explicitly).

        Returns:
            (loss, metrics) where:
              loss: scalar tensor (differentiable w.r.t. log_probs).
              metrics: dict with keys 'clip_fraction', 'mean_ratio', 'entropy'.
        """
        # r_t = π_θ / π_old = exp(log π_θ - log π_old)
        log_ratio = log_probs - old_log_probs
        r = torch.exp(log_ratio)  # shape: (batch, seq_len) or (N,)

        # Decoupled clip: asymmetric bounds depending on sign of advantage
        # clip(r, 1 - ε_low, 1 + ε_high) for all tokens — same bounds for both
        # but advantage sign determines which side is binding.
        r_clipped = torch.clamp(r, 1.0 - self.eps_low, 1.0 + self.eps_high)

        # Surrogate objectives
        surr1 = r * advantages
        surr2 = r_clipped * advantages

        # Token-level clip loss: -min(surr1, surr2) per token
        token_loss = -torch.min(surr1, surr2)  # (batch, seq_len) or (N,)

        # Token-level normalization (Section 3.3): mean over tokens, not sequences
        loss = token_loss.mean()

        # Entropy bonus (Section 3.4): L = L_policy - β_entropy * H(π)
        # entropy values are positive (H ≥ 0), so subtracting increases entropy
        entropy_val = 0.0
        if entropy is not None:
            entropy_mean = entropy.mean()
            loss = loss - self.beta_entropy * entropy_mean
            entropy_val = entropy_mean.item()

        # Metrics
        # clip_fraction: fraction of tokens where clipping was active
        clipped_mask = (r_clipped != r).float()
        clip_fraction = clipped_mask.mean().item()
        mean_ratio = r.mean().item()

        metrics: dict[str, float] = {
            "clip_fraction": clip_fraction,
            "mean_ratio": mean_ratio,
            "entropy": entropy_val,
        }

        return loss, metrics


# ---------------------------------------------------------------------------
# 2. Dynamic Sampling / Filter (Section 3.2)
# ---------------------------------------------------------------------------


class DAPOFilter:
    """Dynamic sampling filter for DAPO.

    Filters out batches that provide no learning signal:
      - All-correct batches (all rewards = 1): no negative examples to learn from.
      - All-wrong batches (all rewards = 0): no positive examples to reinforce.

    Only batches with mixed rewards (0 < mean(rewards) < 1) are kept.

    Per Section 3.2, this is used during oversampling: generate G > K candidates,
    then filter groups, keeping only those with at least one correct and one
    incorrect response.
    """

    def should_keep(self, rewards: torch.Tensor) -> bool:
        """Return True if the batch has learning signal (mixed rewards).

        Args:
            rewards: 1-D tensor of scalar rewards in [0, 1].

        Returns:
            True if 0 < mean(rewards) < 1, i.e., batch has both
            correct and incorrect responses.
        """
        if rewards.numel() == 0:
            return False
        mean_r = rewards.float().mean().item()
        return 0.0 < mean_r < 1.0


# ---------------------------------------------------------------------------
# 3. DAPO Trainer (thin orchestration wrapper)
# ---------------------------------------------------------------------------


class DAPOTrainer:
    """Minimal DAPO training step orchestrator.

    Ties together DAPOLoss and DAPOFilter for a single training iteration.

    Args:
        eps_low: ε_low for DAPOLoss.
        eps_high: ε_high for DAPOLoss.
        beta_entropy: β_entropy for DAPOLoss.
    """

    def __init__(
        self,
        eps_low: float = 0.1,
        eps_high: float = 0.2,
        beta_entropy: float = 0.001,
    ) -> None:
        self.loss_fn = DAPOLoss(
            eps_low=eps_low,
            eps_high=eps_high,
            beta_entropy=beta_entropy,
        )
        self.filter = DAPOFilter()

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute DAPO loss for a batch.

        Expected batch keys:
          - log_probs: current policy log-probs, shape (batch, seq_len) or (N,)
          - old_log_probs: old policy log-probs, same shape
          - advantages: advantage estimates, same shape
          - rewards: optional 1-D reward tensor for dynamic sampling check
          - entropy: optional per-token entropy, same shape as log_probs

        Returns:
            Scalar loss tensor.

        Raises:
            ValueError: if required keys are missing from batch.
        """
        required = {"log_probs", "old_log_probs", "advantages"}
        missing = required - set(batch.keys())
        if missing:
            raise ValueError(f"batch is missing required keys: {missing}")

        log_probs = batch["log_probs"]
        old_log_probs = batch["old_log_probs"]
        advantages = batch["advantages"]
        entropy = batch.get("entropy", None)

        loss, _ = self.loss_fn(
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            entropy=entropy,
        )
        return loss
