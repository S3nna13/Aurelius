from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor
from src.alignment.praxis.config import PRAXISConfig
from src.alignment.dapo import DAPOLoss


class PRAXISLoss:
    """PRAXIS combined loss: DAPO + KL penalty + entropy bonus + ESA, with constitutional gate.

    The constitutional gate zeroes the policy gradient for any sequence whose
    constitutional score is below tau_gate, preventing harmful content from
    receiving any alignment signal until it clears the safety threshold.
    """

    def __init__(self, config: PRAXISConfig) -> None:
        self.config = config
        self.dapo   = DAPOLoss(
            eps_low=config.eps_low,
            eps_high=config.eps_high,
            beta_entropy=config.lambda_ent,
        )

    def forward(
        self,
        log_probs: Tensor,        # (B, T)
        old_log_probs: Tensor,    # (B, T)
        advantages: Tensor,       # (B, T)
        fused_rewards: Tensor,    # (B,) from PrecisionFusion
        mask: Tensor,             # (B, T) bool
        ref_log_probs: Tensor | None = None,  # (B, T) for KL
        const_scores: Tensor | None = None,   # (B,) constitutional gate signal
        entropy: Tensor | None = None,        # (B, T) optional entropy term
        esa_loss: Tensor | None = None,       # scalar from ExpertSafetyAffinity
    ) -> tuple[Tensor, dict]:
        cfg = self.config

        # 1. Constitutional gate: zero loss for sequences below tau_gate
        if const_scores is not None:
            gate_mask = (const_scores >= cfg.tau_gate).float()  # (B,)
            if gate_mask.sum() == 0:
                zero = log_probs.new_zeros(())
                return zero, {"dapo_loss": 0.0, "kl_penalty": 0.0, "const_gate_ratio": 0.0}
            advantages = advantages * gate_mask.unsqueeze(1)

        # 2. DAPO policy gradient loss
        dapo_loss, dapo_metrics = self.dapo.forward(
            log_probs, old_log_probs, advantages, entropy=entropy
        )

        # 3. KL penalty against reference
        kl_penalty = log_probs.new_zeros(())
        if ref_log_probs is not None:
            mask_f     = mask.float()
            valid_cnt  = mask_f.sum().clamp(min=1.0)
            kl_penalty = ((log_probs - ref_log_probs) * mask_f).sum() / valid_cnt
            kl_penalty = cfg.beta_kl * kl_penalty

        # 4. ESA routing loss
        esa_term = esa_loss if esa_loss is not None else log_probs.new_zeros(())

        total_loss = dapo_loss + kl_penalty + esa_term

        metrics = {
            "dapo_loss": dapo_loss.item(),
            "kl_penalty": kl_penalty.item(),
            "esa_loss": esa_term.item() if hasattr(esa_term, "item") else float(esa_term),
            "total_loss": total_loss.item(),
        }
        if const_scores is not None:
            gate_mask = (const_scores >= cfg.tau_gate).float()
            metrics["const_gate_ratio"] = gate_mask.mean().item()
        metrics.update(dapo_metrics)

        return total_loss, metrics