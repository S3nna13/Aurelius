"""Aurelius — ODIN: Offset-based Direct alignment (arXiv:2402.07319).

ODIN disentangles the reward into helpfulness and length/format components to
mitigate reward hacking in RLHF.  The reward decomposes as:

    r_ODIN(x, y) = r_h(x, y) - λ · r_l(x, y)

where r_h is the helpfulness (implicit) reward and r_l is a length penalty.

For the DPO-style training objective ODIN applies length normalisation to each
implicit reward before computing the preference margin:

    r_i = [log π_θ(y_i|x) − log π_ref(y_i|x)] / |y_i|

    L_ODIN = −log σ(β · (r_chosen − r_rejected))

Pure PyTorch implementation following the same conventions as dpo.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ODINConfig", "ODINLoss"]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ODINConfig:
    """Hyper-parameters for the ODIN objective (§3, arXiv:2402.07319).

    Attributes:
        beta: KL-regularisation temperature (same role as DPO β).
        length_penalty: λ — weight applied to the length penalty term.
            Kept for forward-compatibility; the paper sets λ = 1 and absorbs
            it into the length-normalised log-ratio.
        normalize_length: When True the per-sequence implicit reward is
            divided by the number of valid (non-padding) tokens before the
            preference margin is computed.
    """

    beta: float = 0.1
    length_penalty: float = 1.0   # λ — weight for length penalty
    normalize_length: bool = True  # divide log-ratios by sequence length


# ---------------------------------------------------------------------------
# Core module
# ---------------------------------------------------------------------------


class ODINLoss(nn.Module):
    """ODIN preference-optimisation loss with disentangled length reward.

    The forward pass accepts per-token log-probabilities and binary masks
    (1 = valid token, 0 = padding).  It:

      1. Sums per-token log-probs under the policy and reference models
         separately, applying the mask to ignore padding.
      2. Computes the implicit reward for each response:
             r_i = Σ_t [log π_θ − log π_ref] · mask_i,t
      3. Optionally length-normalises each reward by the count of valid tokens
         (divided by `length_penalty` λ to allow re-weighting).
      4. Returns the ODIN loss and a metrics dict.
    """

    def __init__(self, config: ODINConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else ODINConfig()

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        chosen_log_probs: torch.Tensor,      # (B, T_w)  per-token log π_θ(y_w|x)
        rejected_log_probs: torch.Tensor,    # (B, T_l)  per-token log π_θ(y_l|x)
        ref_chosen_log_probs: torch.Tensor,  # (B, T_w)  per-token log π_ref(y_w|x)
        ref_rejected_log_probs: torch.Tensor,# (B, T_l)  per-token log π_ref(y_l|x)
        chosen_mask: torch.Tensor,           # (B, T_w)  valid-token mask
        rejected_mask: torch.Tensor,         # (B, T_l)  valid-token mask
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute ODIN loss.

        Args:
            chosen_log_probs:       (B, T_w) per-token log π_θ for chosen responses.
            rejected_log_probs:     (B, T_l) per-token log π_θ for rejected responses.
            ref_chosen_log_probs:   (B, T_w) per-token log π_ref for chosen responses.
            ref_rejected_log_probs: (B, T_l) per-token log π_ref for rejected responses.
            chosen_mask:            (B, T_w) float/bool mask, 1 for valid tokens.
            rejected_mask:          (B, T_l) float/bool mask, 1 for valid tokens.

        Returns:
            loss:    Scalar ODIN loss tensor (differentiable w.r.t. *_log_probs).
            metrics: Dict with keys chosen_reward, rejected_reward, reward_margin,
                     chosen_length, rejected_length.
        """
        cfg = self.config

        chosen_mask_f = chosen_mask.float()
        rejected_mask_f = rejected_mask.float()

        # Sequence lengths (number of valid tokens per example)
        chosen_len = chosen_mask_f.sum(dim=-1).clamp(min=1.0)    # (B,)
        rejected_len = rejected_mask_f.sum(dim=-1).clamp(min=1.0) # (B,)

        # Sum per-token log-probs over valid positions → (B,)
        pi_chosen_sum = (chosen_log_probs * chosen_mask_f).sum(dim=-1)
        pi_rejected_sum = (rejected_log_probs * rejected_mask_f).sum(dim=-1)
        ref_chosen_sum = (ref_chosen_log_probs * chosen_mask_f).sum(dim=-1)
        ref_rejected_sum = (ref_rejected_log_probs * rejected_mask_f).sum(dim=-1)

        # Implicit rewards: r_i = log π_θ(y_i|x) − log π_ref(y_i|x)
        r_chosen = pi_chosen_sum - ref_chosen_sum       # (B,)
        r_rejected = pi_rejected_sum - ref_rejected_sum # (B,)

        # Length normalisation: divide by |y_i| (and scale by λ)
        if cfg.normalize_length:
            r_chosen = r_chosen / (chosen_len * cfg.length_penalty)
            r_rejected = r_rejected / (rejected_len * cfg.length_penalty)

        # ODIN preference margin and loss (equation 4, arXiv:2402.07319)
        margin = cfg.beta * (r_chosen - r_rejected)   # (B,)
        loss = -F.logsigmoid(margin).mean()

        # Detach rewards for metrics to avoid holding onto the computation graph
        with torch.no_grad():
            metrics: Dict[str, float] = {
                "chosen_reward":   r_chosen.mean().item(),
                "rejected_reward": r_rejected.mean().item(),
                "reward_margin":   (r_chosen - r_rejected).mean().item(),
                "chosen_length":   chosen_len.mean().item(),
                "rejected_length": rejected_len.mean().item(),
            }

        return loss, metrics
