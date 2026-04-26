"""PPO value head: adds a scalar value estimator to AureliusTransformer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PPOConfig:
    clip_eps: float = 0.2  # PPO clipping parameter
    value_loss_coef: float = 0.5  # weight for value loss in total loss
    entropy_coef: float = 0.01  # weight for entropy bonus
    d_model: int = 1024


class ValueHead(nn.Module):
    """Scalar value estimator added on top of transformer hidden states.

    Wraps an AureliusTransformer and adds a value head that produces
    per-token value estimates.
    """

    def __init__(self, backbone: nn.Module, cfg: PPOConfig) -> None:
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg
        self.value_head = nn.Linear(cfg.d_model, 1, bias=True)
        self._hidden: list[torch.Tensor] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """
        Returns:
            (loss, logits, values)
            - loss: scalar loss from backbone if labels given, else None
            - logits: (B, S, V) policy logits
            - values: (B, S) per-token value estimates
        """
        # Hook into backbone.norm to get hidden states
        hidden_states: list[torch.Tensor] = []
        hook = self.backbone.norm.register_forward_hook(lambda m, i, o: hidden_states.append(o))
        try:
            loss, logits, _ = self.backbone(input_ids, labels=labels)
        finally:
            hook.remove()

        h = hidden_states[0]  # (B, S, d_model)
        values = self.value_head(h).squeeze(-1)  # (B, S)
        return loss, logits, values


def ppo_loss(
    logits_new: torch.Tensor,
    logits_old: torch.Tensor,
    values: torch.Tensor,
    value_targets: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    cfg: PPOConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute PPO loss with clipped policy gradient + value loss + entropy.

    Args:
        logits_new: (B, S, V) current policy logits
        logits_old: (B, S, V) old policy logits (from rollout)
        values: (B, S) current value estimates
        value_targets: (B, S) TD or Monte Carlo value targets
        advantages: (B, S) advantage estimates (normalized)
        response_mask: (B, S) bool, True = response tokens (not prompt)
        cfg: PPOConfig

    Returns:
        (total_loss, metrics_dict)
        metrics: {"policy_loss", "value_loss", "entropy", "ratio_mean"}
    """
    B, S, V = logits_new.shape

    log_probs_new = F.log_softmax(logits_new, dim=-1)  # (B, S, V)
    log_probs_old = F.log_softmax(logits_old, dim=-1)  # (B, S, V)

    # For ratio: need actual tokens. Use argmax of old as proxy.
    old_tokens = logits_old.argmax(dim=-1)  # (B, S)

    # Gather log probs at old_tokens positions
    lp_new = log_probs_new.gather(2, old_tokens.unsqueeze(-1)).squeeze(-1)  # (B, S)
    lp_old = log_probs_old.gather(2, old_tokens.unsqueeze(-1)).squeeze(-1)  # (B, S)

    # Ratio
    ratio = (lp_new - lp_old).exp()  # (B, S)

    # Clipped policy loss
    adv = advantages
    loss_unclipped = -ratio * adv
    loss_clipped = -ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv
    policy_loss_per_token = torch.max(loss_unclipped, loss_clipped)

    # Entropy bonus
    probs_new = log_probs_new.exp()
    entropy_per_token = -(probs_new * log_probs_new).sum(dim=-1)  # (B, S)

    # Value loss
    value_loss_per_token = (values - value_targets) ** 2

    # Apply response mask and mean
    mask = response_mask.float()
    n_valid = mask.sum() + 1e-8

    policy_loss = (policy_loss_per_token * mask).sum() / n_valid
    value_loss = (value_loss_per_token * mask).sum() / n_valid
    entropy = (entropy_per_token * mask).sum() / n_valid

    total_loss = policy_loss + cfg.value_loss_coef * value_loss - cfg.entropy_coef * entropy

    metrics = {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
        "ratio_mean": (ratio * mask).sum().item() / n_valid.item(),
    }
    return total_loss, metrics
