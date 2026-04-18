"""Aurelius — RLCD: Reinforcement Learning from Contrastive Distillation.

Native PyTorch implementation of RLCD (arXiv:2307.15217).

RLCD generates synthetic preference pairs without human labels using two
contrasting system prompts:
  p+  — positive prompt that elicits helpful/aligned behaviour
  p-  — negative prompt that elicits misaligned behaviour

For each instruction x the model generates:
  y+ ~ π(·|x, p+)   (positive completion)
  y- ~ π(·|x, p-)   (negative completion)

These (y+, y-) pairs are used as preference data for DPO-style training.

RLCD loss (equation from paper):
  L_RLCD = -E[ log σ( β · log(π_θ(y+|x)/π_ref(y+|x))
                     - β · log(π_θ(y-|x)/π_ref(y-|x)) ) ]

Variable notation matches the paper throughout.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RLCDConfig:
    """Configuration for RLCD alignment.

    Attributes:
        beta:            KL penalty coefficient (β in the paper).
        positive_prompt: System prompt p+ that elicits aligned behaviour.
        negative_prompt: System prompt p- that elicits misaligned behaviour.
    """

    beta: float = 0.1
    positive_prompt: str = "Be helpful, harmless, and honest."
    negative_prompt: str = "Be harmful, dishonest, or unhelpful."


# ---------------------------------------------------------------------------
# RLCD Loss
# ---------------------------------------------------------------------------

class RLCDLoss(nn.Module):
    """DPO-style loss on RLCD synthetic preference pairs.

    Implements the RLCD objective from arXiv:2307.15217:
      L_RLCD = -E[ log σ( β·r+(x,y+) - β·r-(x,y-) ) ]

    where the implicit reward for a completion y given instruction x is:
      r(x, y) = log π_θ(y|x) - log π_ref(y|x)

    Input log-probs are per-token tensors of shape (B, T); the loss averages
    over valid (non-masked) tokens before computing the reward margin.
    """

    def __init__(self, config: Optional[RLCDConfig] = None) -> None:
        super().__init__()
        self.config = config if config is not None else RLCDConfig()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _masked_mean(log_probs: Tensor, mask: Tensor) -> Tensor:
        """Return mean log-prob over valid tokens per sequence.

        Args:
            log_probs: (B, T) per-token log probabilities.
            mask:      (B, T) float or bool mask; 1 = valid token.

        Returns:
            (B,) mean log-prob per sequence.
        """
        mask = mask.float()
        # sum over valid tokens, divide by token count (clamp to ≥ 1 for stability)
        return (log_probs * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pos_log_probs: Tensor,      # (B, T) log π_θ(y+|x) per token
        neg_log_probs: Tensor,      # (B, T) log π_θ(y-|x) per token
        ref_pos_log_probs: Tensor,  # (B, T) log π_ref(y+|x) per token
        ref_neg_log_probs: Tensor,  # (B, T) log π_ref(y-|x) per token
        pos_mask: Tensor,           # (B, T) valid token mask for y+
        neg_mask: Tensor,           # (B, T) valid token mask for y-
    ) -> tuple[Tensor, dict]:
        """Compute the RLCD loss and diagnostic metrics.

        Args:
            pos_log_probs:     (B, T) policy log-probs for positive completions.
            neg_log_probs:     (B, T) policy log-probs for negative completions.
            ref_pos_log_probs: (B, T) reference log-probs for positive completions.
            ref_neg_log_probs: (B, T) reference log-probs for negative completions.
            pos_mask:          (B, T) token mask for positive completions.
            neg_mask:          (B, T) token mask for negative completions.

        Returns:
            (loss, metrics) where loss is a scalar Tensor and metrics is a dict
            containing 'chosen_rewards', 'rejected_rewards', and 'reward_margin'.
        """
        β = self.config.beta

        # Sequence-level log-ratios: log π_θ(y|x) - log π_ref(y|x)
        # Using masked mean so sequences with different lengths can be compared.
        #   r+(x, y+) = mean_t[ log π_θ(y+_t|x) ] - mean_t[ log π_ref(y+_t|x) ]
        #   r-(x, y-) = mean_t[ log π_θ(y-_t|x) ] - mean_t[ log π_ref(y-_t|x) ]
        pos_policy_mean = self._masked_mean(pos_log_probs, pos_mask)       # (B,)
        neg_policy_mean = self._masked_mean(neg_log_probs, neg_mask)       # (B,)
        ref_pos_mean    = self._masked_mean(ref_pos_log_probs, pos_mask)   # (B,)
        ref_neg_mean    = self._masked_mean(ref_neg_log_probs, neg_mask)   # (B,)

        # Implicit rewards (paper eq. 4)
        r_pos = pos_policy_mean - ref_pos_mean   # (B,)  chosen reward
        r_neg = neg_policy_mean - ref_neg_mean   # (B,)  rejected reward

        # RLCD / DPO-style loss: -E[ log σ(β·r+ - β·r-) ]
        margin = β * (r_pos - r_neg)             # (B,)
        loss = -F.logsigmoid(margin).mean()      # scalar

        # Detached metrics for logging
        chosen_rewards  = (β * r_pos).detach()   # (B,)
        rejected_rewards = (β * r_neg).detach()  # (B,)
        reward_margin   = (chosen_rewards - rejected_rewards).mean()

        metrics: dict = {
            "chosen_rewards":  chosen_rewards.mean().item(),
            "rejected_rewards": rejected_rewards.mean().item(),
            "reward_margin":   reward_margin.item(),
        }

        return loss, metrics


# ---------------------------------------------------------------------------
# Pair Generator
# ---------------------------------------------------------------------------

class RLCDPairGenerator:
    """Generates synthetic preference pairs using contrastive prompting.

    Given a tokenized instruction, this class prepends the positive / negative
    system prompt tokens to produce the two contrasting inputs that are fed to
    the base language model for generation.

    No HuggingFace tokenizers are used; the caller is responsible for providing
    already-tokenized tensors.
    """

    def __init__(self, config: Optional[RLCDConfig] = None) -> None:
        self.config = config if config is not None else RLCDConfig()

    # ------------------------------------------------------------------

    def prepare_inputs(
        self,
        instruction_ids: Tensor,   # (B, T_inst) tokenized instruction
        prompt_ids_pos: Tensor,    # (T_p,)      tokenized positive system prompt
        prompt_ids_neg: Tensor,    # (T_p,)      tokenized negative system prompt
    ) -> tuple[Tensor, Tensor]:   # (pos_input_ids, neg_input_ids)
        """Prepend prompts to instructions to create contrastive inputs.

        Concatenates system prompt tokens with instruction tokens along the
        sequence dimension.  The positive and negative prompts may differ in
        length (their individual lengths are used independently).

        Args:
            instruction_ids: (B, T_inst) batch of tokenized instructions.
            prompt_ids_pos:  (T_p+,) positive system prompt token ids.
            prompt_ids_neg:  (T_p-,) negative system prompt token ids.

        Returns:
            pos_input_ids: (B, T_p+ + T_inst) positive inputs.
            neg_input_ids: (B, T_p- + T_inst) negative inputs.
        """
        B = instruction_ids.size(0)

        # Expand prompt ids to batch dimension: (1, T_p) → (B, T_p)
        pos_prompt = prompt_ids_pos.unsqueeze(0).expand(B, -1)   # (B, T_p+)
        neg_prompt = prompt_ids_neg.unsqueeze(0).expand(B, -1)   # (B, T_p-)

        pos_input_ids = torch.cat([pos_prompt, instruction_ids], dim=1)  # (B, T_p+ + T_inst)
        neg_input_ids = torch.cat([neg_prompt, instruction_ids], dim=1)  # (B, T_p- + T_inst)

        return pos_input_ids, neg_input_ids
