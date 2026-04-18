"""Dr. GRPO: Decomposed Reward for Group Relative Policy Optimization.

Implements the corrected advantage estimation from arXiv:2503.20783
("Dr. GRPO: Decomposing GRPO's Implicit Assumptions for Enhanced Reasoning").

Key corrections over standard GRPO (arXiv:2402.03300):
  1. Removes question-level bias via global mean normalization instead of
     within-group mean — so difficulty does not contaminate the advantage.
  2. Unbiased std estimation (ddof=1 within group).
  3. Advantage clipping for numerical stability.
  4. Token-level loss averaging — prevents longer completions from
     dominating the gradient update.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Advantage computation
# ---------------------------------------------------------------------------

@dataclass
class DrGRPOAdvantageConfig:
    """Configuration for Dr. GRPO advantage estimation."""
    clip_value: float = 10.0   # Symmetric clipping range for advantages
    eps: float = 1e-8          # Numerical floor added to std denominator
    use_global_mean: bool = True  # Use global (cross-question) mean normalization


class DrGRPOAdvantage:
    """Compute Dr. GRPO corrected advantages.

    Standard GRPO normalizes each group independently:
        Â_i = (r_i - mean_group) / std_group

    Dr. GRPO replaces the within-group mean with the global mean across all
    questions in the batch, removing the implicit question-difficulty bias:
        Â_i = (r_i - mean_global) / std_group

    When all rewards in a group are equal, std_group == 0 and the advantages
    are set to zero (no gradient signal), never NaN.

    Args:
        clip_value: Symmetric clipping bound applied after normalization.
        eps: Small constant added to std to prevent division by zero.
        use_global_mean: If True, normalise by the global mean (Dr. GRPO).
            If False, normalise by the within-group mean (standard GRPO).
    """

    def __init__(
        self,
        clip_value: float = 10.0,
        eps: float = 1e-8,
        use_global_mean: bool = True,
    ) -> None:
        if clip_value <= 0:
            raise ValueError(f"clip_value must be positive, got {clip_value}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.clip_value = clip_value
        self.eps = eps
        self.use_global_mean = use_global_mean

    def compute(
        self,
        rewards: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute corrected advantages.

        Args:
            rewards: (B, G) — reward per completion per question in the batch.
            lengths: (B, G) — token lengths, unused here (reserved for future
                length-normalisation of reward before advantage computation).

        Returns:
            advantages: (B, G) — normalised and clipped advantages.
        """
        if rewards.dim() != 2:
            raise ValueError(
                f"rewards must be 2-D (B, G), got shape {tuple(rewards.shape)}"
            )

        B, G = rewards.shape  # noqa: N806

        # Global mean: scalar across all (question, completion) pairs.
        if self.use_global_mean:
            mean = rewards.mean()  # scalar
        else:
            # Within-group mean — standard GRPO behaviour.
            mean = rewards.mean(dim=1, keepdim=True)  # (B, 1)

        # Unbiased std within each group (ddof=1).  When G == 1 the std is
        # undefined; fall back to zero, producing zero advantage.
        if G > 1:
            std = rewards.std(dim=1, keepdim=True, unbiased=True)  # (B, 1)
        else:
            std = torch.zeros(B, 1, dtype=rewards.dtype, device=rewards.device)

        # Centre and scale; use eps to guard against std == 0.
        advantages = (rewards - mean) / (std + self.eps)  # (B, G)

        # Where std == 0 the group had uniform rewards → zero advantages.
        zero_std_mask = std < self.eps  # (B, 1)
        advantages = torch.where(
            zero_std_mask.expand_as(advantages),
            torch.zeros_like(advantages),
            advantages,
        )

        # Clip to prevent instability from outlier advantage values.
        advantages = advantages.clamp(-self.clip_value, self.clip_value)

        return advantages


# ---------------------------------------------------------------------------
# Loss module
# ---------------------------------------------------------------------------

class DrGRPOLoss(nn.Module):
    """PPO-clip policy gradient loss with Dr. GRPO advantage corrections.

    Operates over token-level log-probabilities, with token-level averaging
    of the loss so that completion length does not bias the gradient.

    Loss formula per completion i in question b:
        obj_i  = sum_t { min(r_t * Â_i, clip(r_t, 1-ε, 1+ε) * Â_i) } / |o_i|
        L      = - E_{b,i}[ obj_i ]

    where r_t = exp(log π_θ(a_t|s_t) - log π_ref(a_t|s_t)) is the per-token
    importance ratio and |o_i| is the number of valid tokens in completion i.

    Args:
        clip_eps: PPO clipping epsilon ε ∈ (0, 1).
        advantage_clip: Symmetric clipping range for advantages.
        token_level: If True, divide each completion's objective by its token
            count (Dr. GRPO). If False, use standard sum-over-tokens.
    """

    def __init__(
        self,
        clip_eps: float = 0.2,
        advantage_clip: float = 10.0,
        token_level: bool = True,
    ) -> None:
        super().__init__()
        if not 0 < clip_eps < 1:
            raise ValueError(f"clip_eps must be in (0, 1), got {clip_eps}")
        self.clip_eps = clip_eps
        self.advantage_clip = advantage_clip
        self.token_level = token_level
        self._advantage_fn = DrGRPOAdvantage(
            clip_value=advantage_clip, use_global_mean=True
        )

    def forward(
        self,
        log_probs: torch.Tensor,      # (B, G, T) current policy
        ref_log_probs: torch.Tensor,  # (B, G, T) reference policy
        rewards: torch.Tensor,        # (B, G) per-completion rewards
        mask: torch.Tensor,           # (B, G, T) bool — True for valid tokens
    ) -> tuple[torch.Tensor, dict]:
        """Compute the Dr. GRPO loss.

        Args:
            log_probs: (B, G, T) — current-policy per-token log-probabilities.
            ref_log_probs: (B, G, T) — reference-policy per-token log-probs
                (treated as detached; gradients do not flow through them).
            rewards: (B, G) — scalar reward assigned to each completion.
            mask: (B, G, T) — boolean mask; True where a token is valid.

        Returns:
            loss: scalar — negative clipped objective (to minimise).
            metrics: dict with keys
                'mean_advantage' (float),
                'std_advantage'  (float),
                'clip_fraction'  (float) — fraction of ratios that were clipped.
        """
        B, G, T = log_probs.shape  # noqa: N806

        if ref_log_probs.shape != (B, G, T):
            raise ValueError("ref_log_probs shape must match log_probs")
        if rewards.shape != (B, G):
            raise ValueError(f"rewards must be (B, G)={B,G}, got {tuple(rewards.shape)}")
        if mask.shape != (B, G, T):
            raise ValueError("mask shape must match log_probs")

        # --- Advantages -------------------------------------------------------
        advantages = self._advantage_fn.compute(rewards)  # (B, G)
        # Expand to token level for broadcasting: (B, G, 1)
        adv_expanded = advantages.unsqueeze(-1)

        # --- Per-token importance ratios --------------------------------------
        # ref_log_probs treated as constant w.r.t. gradients.
        log_ratio = log_probs - ref_log_probs.detach()  # (B, G, T)
        # Clamp log_ratio before exp() to prevent overflow (e.g. e^100 → Inf).
        # Ratios outside ~[e^{-20}, e^{20}] are well outside the PPO clip anyway.
        log_ratio_clamped = log_ratio.clamp(-20.0, 20.0)
        ratio = log_ratio_clamped.exp()                  # (B, G, T)

        # --- PPO-clip objective -----------------------------------------------
        clipped_ratio = ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps)
        obj_unclipped = ratio * adv_expanded           # (B, G, T)
        obj_clipped   = clipped_ratio * adv_expanded   # (B, G, T)
        per_token_obj = torch.min(obj_unclipped, obj_clipped)  # (B, G, T)

        # Zero out padding tokens.
        float_mask = mask.float()
        per_token_obj = per_token_obj * float_mask      # (B, G, T)

        # --- Token-level vs completion-level averaging -----------------------
        token_counts = float_mask.sum(dim=-1).clamp(min=1.0)  # (B, G)

        if self.token_level:
            # Divide each completion's sum by its token count, then average
            # over (B, G) — the Dr. GRPO approach.
            per_completion_obj = per_token_obj.sum(dim=-1) / token_counts  # (B, G)
        else:
            # Standard: sum over tokens then average over completions.
            per_completion_obj = per_token_obj.sum(dim=-1)  # (B, G)

        loss = -per_completion_obj.mean()

        # --- Clip fraction metric --------------------------------------------
        with torch.no_grad():
            # Use the same clamped ratio for consistency.
            was_clipped = (ratio < 1.0 - self.clip_eps) | (ratio > 1.0 + self.clip_eps)
            # Only count valid token positions.
            valid_tokens = float_mask.bool()
            n_valid = valid_tokens.float().sum()
            if n_valid > 0:
                clip_frac = (was_clipped & valid_tokens).float().sum() / n_valid
            else:
                clip_frac = torch.zeros(1, device=loss.device)

        metrics = {
            "mean_advantage": advantages.mean().item(),
            "std_advantage":  advantages.std().item() if advantages.numel() > 1 else 0.0,
            "clip_fraction":  clip_frac.item(),
        }

        return loss, metrics
