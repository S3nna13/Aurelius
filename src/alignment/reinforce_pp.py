"""REINFORCE++ — Global-batch advantage normalization without a critic.

Two variants:
  - Standard: normalize across entire training batch (general RLHF)
  - Baseline: group-sampling with leave-one-out baseline (complex reasoning)

Reference: arXiv:2501.03262
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def reinforce_pp_loss(
    log_probs: Tensor,
    targets: Tensor,
    rewards: Tensor,
    ref_log_probs: Tensor | None = None,
    kl_coef: float = 0.01,
    entropy_coef: float = 0.001,
    variant: str = "standard",
    group_ids: Tensor | None = None,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """
    Args:
        log_probs:     (B, S, V) full token log-probabilities from policy
        targets:       (B, S) target token ids
        rewards:       (B,) scalar rewards per sequence
        ref_log_probs: (B, S, V) reference policy log-probs for KL penalty
        kl_coef:       KL regularization coefficient
        entropy_coef:  entropy bonus coefficient
        variant:       "standard" (global normalization) or "baseline" (leave-one-out)
        group_ids:     (B,) group ids required by the leave-one-out baseline variant
        attention_mask: optional (B, S) mask; padding tokens are excluded
    Returns:
        scalar loss
    """
    if variant not in {"standard", "baseline"}:
        raise ValueError(f"Unsupported REINFORCE++ variant: {variant}")

    mask = None
    token_counts = None
    if attention_mask is not None:
        mask = attention_mask.to(device=log_probs.device, dtype=log_probs.dtype)
        token_counts = mask.sum(dim=-1).clamp(min=1.0)

    # Compute advantages
    if variant == "baseline":
        if group_ids is None:
            raise ValueError("group_ids is required for the baseline variant")
        group_ids = group_ids.to(device=rewards.device)
        baselines = torch.empty_like(rewards)
        for group_id in torch.unique(group_ids):
            idx = group_ids == group_id
            group_rewards = rewards[idx]
            if group_rewards.numel() == 1:
                baselines[idx] = group_rewards.mean()
            else:
                baselines[idx] = (group_rewards.sum() - group_rewards) / (group_rewards.numel() - 1)
        advantages = rewards - baselines
        advantages = advantages / advantages.std(unbiased=False).clamp(min=1e-8)
    else:
        # Global normalization across full batch
        advantages = (rewards - rewards.mean()) / (
            rewards.std(unbiased=False).clamp(min=1e-8)
        )

    # REINFORCE gradient: -log_prob * advantage (summed over tokens, averaged over batch)
    target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    if mask is not None:
        target_log_probs = target_log_probs * mask
    seq_log_probs = target_log_probs.sum(dim=-1)  # (B,)
    pg_loss = -(seq_log_probs * advantages.detach()).mean()

    probs = log_probs.exp()

    # KL penalty: KL(policy || reference) = sum p * (log_p - log_q)
    kl_loss = torch.tensor(0.0, device=log_probs.device)
    if ref_log_probs is not None:
        kl_per_token = (probs * (log_probs - ref_log_probs)).sum(dim=-1)
        if mask is not None:
            kl_per_token = kl_per_token * mask
        kl_per_seq = kl_per_token.sum(dim=-1)
        if token_counts is not None:
            kl_per_seq = kl_per_seq / token_counts
        kl = kl_per_seq.mean()
        kl_loss = kl_coef * kl

    # Entropy bonus: H = -sum_over_vocab p * log p
    entropy_per_token = -(probs * log_probs).sum(dim=-1)
    if mask is not None:
        entropy_per_token = entropy_per_token * mask
    entropy_per_seq = entropy_per_token.sum(dim=-1)
    if token_counts is not None:
        entropy_per_seq = entropy_per_seq / token_counts
    entropy = entropy_per_seq.mean()
    entropy_loss = -entropy_coef * entropy

    return pg_loss + kl_loss + entropy_loss


class ReinforcePPTrainer:
    """Minimal REINFORCE++ trainer wrapper."""

    def __init__(self, model, ref_model=None, kl_coef: float = 0.01,
                 entropy_coef: float = 0.001, variant: str = "standard"):
        self.model = model
        self.ref_model = ref_model
        self.kl_coef = kl_coef
        self.entropy_coef = entropy_coef
        self.variant = variant

    def compute_loss(
        self,
        input_ids: Tensor,
        rewards: Tensor,
        attention_mask: Tensor | None = None,
        group_ids: Tensor | None = None,
    ) -> Tensor:
        logits = self.model(input_ids).logits
        log_probs = F.log_softmax(logits[:, :-1], dim=-1)
        tgt = input_ids[:, 1:]
        token_mask = attention_mask[:, 1:] if attention_mask is not None else None

        ref_log_probs = None
        if self.ref_model is not None:
            with torch.no_grad():
                ref_logits = self.ref_model(input_ids).logits
                ref_log_probs = F.log_softmax(ref_logits[:, :-1], dim=-1)

        return reinforce_pp_loss(
            log_probs,
            tgt,
            rewards,
            ref_log_probs,
            self.kl_coef,
            self.entropy_coef,
            self.variant,
            group_ids=group_ids,
            attention_mask=token_mask,
        )


__all__ = ["reinforce_pp_loss", "ReinforcePPTrainer"]
