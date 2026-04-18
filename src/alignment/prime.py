"""PRIME: Process Reward Model via Implicit Reward (arXiv:2502.01456).

Extracts implicit per-step process rewards from policy rollout log-probabilities
without explicit step annotations. The key insight is that the log-prob ratio
log(π_θ(a_t|s_t) / π_ref(a_t|s_t)) provides a dense reward signal at every
token, which PRIME aggregates into step-level rewards for RL training.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PRIMEConfig:
    """Configuration for PRIME process reward extraction."""
    beta: float = 0.05          # KL penalty weight
    credit_mode: str = "last"   # "last" | "mean" | "cumsum" — credit assignment
    normalize: bool = True      # normalize rewards to unit std across the sequence


# ---------------------------------------------------------------------------
# Core module
# ---------------------------------------------------------------------------

class PRIMEReward(nn.Module):
    """Extract dense implicit process rewards from policy vs. reference log-probs.

    PRIME (Process Reward Model via Implicit Reward) avoids costly human step
    annotations by deriving per-token rewards from the ratio of the policy's
    log-probabilities to a reference model's. These token rewards are then
    aggregated to step/sequence level and combined with sparse outcome rewards.

    Args:
        config: PRIMEConfig instance controlling beta, credit_mode, normalize.
    """

    def __init__(self, config: Optional[PRIMEConfig] = None) -> None:
        super().__init__()
        self.config = config if config is not None else PRIMEConfig()

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def compute_implicit_rewards(
        self,
        log_probs: torch.Tensor,      # (B, T) policy log-probs per token
        ref_log_probs: torch.Tensor,  # (B, T) reference log-probs
        mask: torch.Tensor,           # (B, T) valid token mask (1=valid, 0=pad)
    ) -> torch.Tensor:
        """Per-token implicit reward = log(π_θ / π_ref).

        The reward at each token position is simply the log-probability ratio
        between the current policy and the reference policy. Positive values
        indicate positions where the policy is more confident than the reference;
        negative values indicate the opposite.

        Args:
            log_probs:     (B, T) — log-probabilities under the current policy.
            ref_log_probs: (B, T) — log-probabilities under the reference policy.
            mask:          (B, T) — binary mask; 0 at padding positions.

        Returns:
            token_rewards: (B, T) — implicit reward at each valid token position;
                           zero at masked (padding) positions.
        """
        token_rewards = log_probs - ref_log_probs          # (B, T) log-ratio
        token_rewards = token_rewards * mask.float()        # zero out padding
        return token_rewards

    def aggregate_step_rewards(
        self,
        token_rewards: torch.Tensor,  # (B, T)
        mask: torch.Tensor,           # (B, T)
    ) -> torch.Tensor:
        """Aggregate per-token rewards to a single step/sequence-level scalar.

        Three credit assignment modes:
          - "last":   reward of the final valid token in each sequence.
          - "mean":   mean over all valid token positions.
          - "cumsum": sum of token rewards over valid positions (credit to all steps).

        Args:
            token_rewards: (B, T) — per-token implicit rewards (padding already zeroed).
            mask:          (B, T) — binary validity mask.

        Returns:
            step_rewards: (B,) — one scalar reward per batch element.
        """
        mode = self.config.credit_mode
        mask_f = mask.float()  # (B, T)

        if mode == "last":
            # Index of the last valid position in each row
            # Sum along T; subtract 1 to get 0-based index; clamp to [0, T-1]
            last_idx = mask_f.sum(dim=1).long() - 1          # (B,)
            last_idx = last_idx.clamp(min=0)
            step_rewards = token_rewards.gather(
                1, last_idx.unsqueeze(1)
            ).squeeze(1)                                      # (B,)

        elif mode == "mean":
            valid_counts = mask_f.sum(dim=1).clamp(min=1.0)  # (B,)
            step_rewards = token_rewards.sum(dim=1) / valid_counts  # (B,)

        elif mode == "cumsum":
            step_rewards = token_rewards.sum(dim=1)           # (B,)

        else:
            raise ValueError(
                f"Unknown credit_mode '{mode}'. "
                "Choose from: 'last', 'mean', 'cumsum'."
            )

        return step_rewards

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        log_probs: torch.Tensor,        # (B, T) policy log-probs
        ref_log_probs: torch.Tensor,    # (B, T) reference log-probs
        outcome_rewards: torch.Tensor,  # (B,)  sparse outcome reward
        mask: torch.Tensor,             # (B, T) valid token mask
    ) -> tuple[torch.Tensor, dict]:
        """Compute full PRIME reward: implicit token rewards + outcome.

        The final per-token dense reward broadcast back adds the scalar outcome
        reward for each sequence to every valid position, giving a dense signal
        that combines implicit process credit with sparse outcome supervision.

        Args:
            log_probs:       (B, T) — current policy log-probs.
            ref_log_probs:   (B, T) — reference policy log-probs.
            outcome_rewards: (B,)   — sparse outcome reward (e.g. task score).
            mask:            (B, T) — binary validity mask.

        Returns:
            dense_rewards: (B, T) — dense per-token reward (masked, outcome added).
            metrics:       dict   — diagnostic scalars for logging.
        """
        cfg = self.config

        # Step 1: per-token implicit rewards
        token_rewards = self.compute_implicit_rewards(log_probs, ref_log_probs, mask)

        # Step 2: optionally normalize token rewards (per-batch, masked positions)
        if cfg.normalize:
            mask_f = mask.float()
            valid_count = mask_f.sum().clamp(min=1.0)
            mean = (token_rewards * mask_f).sum() / valid_count
            # Variance over valid tokens
            var = ((token_rewards - mean) ** 2 * mask_f).sum() / valid_count
            std = (var + 1e-8).sqrt()
            token_rewards = (token_rewards - mean * mask_f) / std
            token_rewards = token_rewards * mask_f  # re-zero padding

        # Step 3: aggregate to step level and add outcome reward
        step_rewards = self.aggregate_step_rewards(token_rewards, mask)  # (B,)
        combined_step = step_rewards + outcome_rewards                    # (B,)

        # Step 4: broadcast outcome back to token level (add to each valid position)
        dense_rewards = token_rewards + outcome_rewards.unsqueeze(1) * mask.float()

        # Step 5: compute KL penalty (mean log-ratio over valid tokens)
        mask_f = mask.float()
        valid_count = mask_f.sum().clamp(min=1.0)
        kl_penalty = ((log_probs - ref_log_probs) * mask_f).sum() / valid_count

        metrics = {
            "implicit_reward_mean": step_rewards.mean().item(),
            "implicit_reward_std": step_rewards.std().item() if step_rewards.numel() > 1 else 0.0,
            "outcome_reward_mean": outcome_rewards.mean().item(),
            "combined_reward_mean": combined_step.mean().item(),
            "kl_penalty": kl_penalty.item(),
        }

        return dense_rewards, metrics
