"""Token-level Direct Preference Optimization (Token-DPO).

Implements token-level credit assignment for DPO training, allowing the model
to learn which specific tokens in a response were good or bad, rather than
assigning a single scalar preference signal to the whole sequence.

References:
    Zeng et al. 2024, "Token-level Direct Preference Optimization"
    https://arxiv.org/abs/2404.11999
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TokenDPOConfig:
    """Configuration for Token-level DPO training."""

    beta: float = 0.1
    """KL penalty coefficient."""

    normalize_weights: bool = True
    """Whether to normalize token weights to sum to 1.0 per sequence."""

    weight_temperature: float = 1.0
    """Temperature for softmax weight computation. Lower = sharper focus."""

    min_weight: float = 0.0
    """Minimum token weight (clamp floor). Set > 0 to ensure all tokens contribute."""

    loss_type: str = "sigmoid"
    """Loss type: 'sigmoid' (standard DPO), 'hinge', or 'ipo'."""


# ---------------------------------------------------------------------------
# Standalone helper functions
# ---------------------------------------------------------------------------

def compute_per_token_advantages(
    chosen_logps: Tensor,
    rejected_logps: Tensor,
) -> Tensor:
    """Compute per-token advantage scores normalized to zero mean per sequence.

    The advantage of a token at position i is defined as the difference between
    the policy log-prob of that token in the chosen vs rejected response,
    providing a signal for how much each position distinguishes good from bad.

    Args:
        chosen_logps: (batch, seq) -- per-token log probs for chosen sequences.
        rejected_logps: (batch, seq) -- per-token log probs for rejected sequences.

    Returns:
        (batch, seq) -- per-token advantages, normalized to zero mean per sequence.
    """
    advantages = chosen_logps - rejected_logps  # (batch, seq)
    mean = advantages.mean(dim=-1, keepdim=True)  # (batch, 1)
    normalized = advantages - mean  # (batch, seq)
    return normalized


# ---------------------------------------------------------------------------
# Token-level DPO Trainer
# ---------------------------------------------------------------------------

class TokenDPOTrainer:
    """Trainer for Token-level Direct Preference Optimization.

    Decomposes the DPO objective into per-token contributions, allowing the
    model to learn fine-grained credit assignment over response tokens.

    Args:
        policy: Policy model. Forward call must accept (B, T) input_ids and
                return (B, T, V) logits or a tuple/list where one element
                is a 3-D (B, T, V) tensor.
        ref_policy: Frozen reference model with same interface as policy.
        beta: KL penalty coefficient.
        token_weight_fn: Optional callable(chosen_ids, rejected_ids) ->
                         (chosen_weights, rejected_weights) both (B, T).
                         If None, uses per-token advantage-based weights.
        normalize_weights: Whether to normalize token weights.
        config: Optional TokenDPOConfig; if provided, overrides beta and
                normalize_weights kwargs.
    """

    def __init__(
        self,
        policy: nn.Module,
        ref_policy: nn.Module,
        beta: float = 0.1,
        token_weight_fn: Optional[Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]] = None,
        normalize_weights: bool = True,
        config: Optional[TokenDPOConfig] = None,
    ) -> None:
        self.policy = policy
        self.ref_policy = ref_policy
        self.token_weight_fn = token_weight_fn

        if config is not None:
            self.config = config
        else:
            self.config = TokenDPOConfig(
                beta=beta,
                normalize_weights=normalize_weights,
            )

        # Freeze reference model
        for p in self.ref_policy.parameters():
            p.requires_grad_(False)
        self.ref_policy.eval()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_logits(self, model: nn.Module, input_ids: Tensor) -> Tensor:
        """Extract (B, T, V) logits from a model forward pass.

        Handles models returning plain logits OR tuple/list outputs where
        logits may be at index 0 or 1 (e.g. (loss, logits, ...) patterns).
        """
        output = model(input_ids)
        if isinstance(output, Tensor):
            return output
        if isinstance(output, (tuple, list)):
            # Prefer index 1 (loss, logits, ...) then fall back to index 0
            for idx in (1, 0):
                if idx < len(output) and isinstance(output[idx], Tensor):
                    candidate = output[idx]
                    if candidate.dim() == 3:
                        return candidate
            # Last resort: scan for any 3-D tensor
            for item in output:
                if isinstance(item, Tensor) and item.dim() == 3:
                    return item
        raise ValueError(
            f"Cannot extract logits from model output of type {type(output)}. "
            "Expected a 3-D (B, T, V) tensor or a tuple/list containing one."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_token_log_probs(
        self,
        model: nn.Module,
        input_ids: Tensor,
        labels: Tensor,
    ) -> Tensor:
        """Compute per-token log probabilities for non-ignored positions.

        Args:
            input_ids: (batch, seq) -- input token ids.
            labels: (batch, seq) -- target token ids; -100 marks ignored positions.

        Returns:
            (batch, seq) -- per-token log probs; 0.0 at ignored positions.
        """
        logits = self._get_logits(model, input_ids)  # (B, T, V)
        log_probs = F.log_softmax(logits, dim=-1)  # (B, T, V)

        # Guard against -100 indices before gather
        safe_labels = labels.clone()
        safe_labels[safe_labels == -100] = 0

        token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)  # (B, T)

        mask = (labels != -100).float()
        return token_log_probs * mask  # (B, T)

    def compute_tdpo_weights(
        self,
        advantage_scores: Tensor,
        temperature: float = 1.0,
    ) -> Tensor:
        """Compute softmax-normalized per-token weights from advantage scores.

        Args:
            advantage_scores: (batch, seq) -- higher = more important token.
            temperature: Softmax temperature. Lower = sharper distribution.

        Returns:
            (batch, seq) -- softmax-normalized weights summing to 1.0 per sequence.
        """
        scaled = advantage_scores / max(temperature, 1e-8)
        weights = F.softmax(scaled, dim=-1)  # (batch, seq)

        if self.config.min_weight > 0.0:
            weights = weights.clamp(min=self.config.min_weight)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        return weights

    def compute_token_dpo_loss(
        self,
        policy_chosen_logps: Tensor,
        policy_rejected_logps: Tensor,
        ref_chosen_logps: Tensor,
        ref_rejected_logps: Tensor,
        chosen_weights: Optional[Tensor] = None,
        rejected_weights: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict]:
        """Compute token-level DPO loss.

        When token weights are provided, each token's contribution to the
        sequence log-prob sum is scaled by its corresponding weight.

        Args:
            policy_chosen_logps: (batch, seq) per-token log probs for chosen under policy.
            policy_rejected_logps: (batch, seq) per-token log probs for rejected under policy.
            ref_chosen_logps: (batch, seq) per-token log probs for chosen under reference.
            ref_rejected_logps: (batch, seq) per-token log probs for rejected under reference.
            chosen_weights: Optional (batch, seq) token weights for chosen sequences.
                            If None, uniform weights are used (standard DPO).
            rejected_weights: Optional (batch, seq) token weights for rejected sequences.

        Returns:
            (loss, metrics_dict) where:
                - loss: scalar tensor
                - metrics_dict has keys: loss, chosen_rewards, rejected_rewards, accuracy
        """
        seq_len = policy_chosen_logps.shape[-1]

        # Default to uniform weights (equivalent to standard DPO when normalized)
        if chosen_weights is None:
            chosen_weights = torch.ones_like(policy_chosen_logps) / seq_len
        if rejected_weights is None:
            rejected_weights = torch.ones_like(policy_rejected_logps) / seq_len

        # Normalize per sequence if requested
        if self.config.normalize_weights:
            chosen_w_sum = chosen_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            rejected_w_sum = rejected_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            chosen_weights = chosen_weights / chosen_w_sum
            rejected_weights = rejected_weights / rejected_w_sum

        # Weighted sequence-level log probs
        pi_chosen = (policy_chosen_logps * chosen_weights).sum(dim=-1)      # (batch,)
        pi_rejected = (policy_rejected_logps * rejected_weights).sum(dim=-1)  # (batch,)
        ref_chosen = (ref_chosen_logps * chosen_weights).sum(dim=-1)          # (batch,)
        ref_rejected = (ref_rejected_logps * rejected_weights).sum(dim=-1)    # (batch,)

        # Implicit rewards
        chosen_rewards = self.config.beta * (pi_chosen - ref_chosen)       # (batch,)
        rejected_rewards = self.config.beta * (pi_rejected - ref_rejected)  # (batch,)
        reward_diff = chosen_rewards - rejected_rewards                      # (batch,)

        if self.config.loss_type == "sigmoid":
            loss = -F.logsigmoid(reward_diff).mean()
        elif self.config.loss_type == "hinge":
            loss = torch.clamp(1.0 - reward_diff, min=0.0).mean()
        elif self.config.loss_type == "ipo":
            loss = ((reward_diff - 1.0 / (2.0 * self.config.beta)) ** 2).mean()
        else:
            raise ValueError(
                f"Unknown loss_type: {self.config.loss_type!r}. "
                "Use 'sigmoid', 'hinge', or 'ipo'."
            )

        accuracy = (chosen_rewards.detach() > rejected_rewards.detach()).float().mean().item()

        metrics: Dict = {
            "loss": loss.item(),
            "chosen_rewards": chosen_rewards.detach().mean().item(),
            "rejected_rewards": rejected_rewards.detach().mean().item(),
            "accuracy": accuracy,
        }

        return loss, metrics

    def train_step(
        self,
        chosen_ids: Tensor,
        rejected_ids: Tensor,
        chosen_labels: Tensor,
        rejected_labels: Tensor,
    ) -> Dict:
        """Perform one token-level DPO training step.

        Args:
            chosen_ids: (batch, seq) -- input token ids for chosen sequences.
            rejected_ids: (batch, seq) -- input token ids for rejected sequences.
            chosen_labels: (batch, seq) -- target labels for chosen; -100 for masked positions.
            rejected_labels: (batch, seq) -- target labels for rejected; -100 for masked positions.

        Returns:
            Metrics dict with keys: loss, chosen_rewards, rejected_rewards, accuracy.
        """
        self.policy.train()

        policy_chosen_logps = self.compute_token_log_probs(
            self.policy, chosen_ids, chosen_labels
        )
        policy_rejected_logps = self.compute_token_log_probs(
            self.policy, rejected_ids, rejected_labels
        )

        with torch.no_grad():
            ref_chosen_logps = self.compute_token_log_probs(
                self.ref_policy, chosen_ids, chosen_labels
            )
            ref_rejected_logps = self.compute_token_log_probs(
                self.ref_policy, rejected_ids, rejected_labels
            )

        # Compute token weights
        chosen_weights: Optional[Tensor] = None
        rejected_weights: Optional[Tensor] = None

        if self.token_weight_fn is not None:
            chosen_weights, rejected_weights = self.token_weight_fn(chosen_ids, rejected_ids)
        else:
            with torch.no_grad():
                advantages = compute_per_token_advantages(
                    policy_chosen_logps.detach(),
                    policy_rejected_logps.detach(),
                )
                chosen_weights = self.compute_tdpo_weights(
                    advantages,
                    temperature=self.config.weight_temperature,
                )
                rejected_weights = chosen_weights

        loss, metrics = self.compute_token_dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            chosen_weights=chosen_weights,
            rejected_weights=rejected_weights,
        )

        # Store loss tensor for callers that need to call .backward()
        metrics["_loss_tensor"] = loss
        return metrics
