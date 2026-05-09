from __future__ import annotations
import torch
from torch import Tensor
from src.alignment.praxis.config import PRAXISConfig


class RewardSignalBundle:
    """Computes 6-signal reward decomposition for PRAXIS.

    Signals:
        R_prime: PRIME implicit process reward (log-ratio based)
        R_const: Constitutional critique head aggregate score
        R_ccot:  CCoT chain-of-thought quality (from outcome rewards, scaled)
        R_odin:  Length-normalized reward (outcome / sqrt(len))
        R_hier:  Hierarchical reward model score
        R_src:   Steering-Reward Correspondence signal (scalar, broadcast)
    """

    def __init__(self, config: PRAXISConfig, prime, critique_head, hier_model, mc_reward) -> None:
        self.config = config
        self.prime        = prime
        self.critique     = critique_head
        self.hier         = hier_model
        self.mc_reward    = mc_reward

    def compute(
        self,
        hidden: Tensor,           # (B, T, D)
        log_probs: Tensor,        # (B, T)
        ref_log_probs: Tensor,    # (B, T)
        outcome_rewards: Tensor,  # (B,)
        mask: Tensor,             # (B, T) bool
    ) -> dict[str, tuple[Tensor, Tensor]]:
        B, T, D = hidden.shape
        eps = 1e-6

        # --- R_prime: PRIME dense reward, aggregate to (B,) mean over valid tokens ---
        dense_r, _ = self.prime(log_probs, ref_log_probs, outcome_rewards, mask.float())
        valid_len   = mask.float().sum(dim=1).clamp(min=1.0)
        r_prime     = (dense_r * mask.float()).sum(dim=1) / valid_len
        r_prime_std = torch.ones_like(r_prime) * (r_prime.std() + eps)

        # --- R_const: Constitutional critique score ---
        critique_scores = self.critique(hidden)           # (B, n_principles)
        r_const         = critique_scores.mean(dim=-1)   # (B,)
        r_const_std     = critique_scores.std(dim=-1).clamp(min=eps)

        # --- R_ccot: CCoT quality approximated as outcome * log(1 + valid_len/T) ---
        r_ccot     = outcome_rewards * torch.log1p(valid_len / T)
        r_ccot_std = torch.ones_like(r_ccot) * (r_ccot.std() + eps)

        # --- R_odin: Length-normalized outcome reward ---
        r_odin     = outcome_rewards / valid_len.sqrt()
        r_odin_std = torch.ones_like(r_odin) * (r_odin.std() + eps)

        # --- R_hier: Hierarchical reward model (expects (B, D) input) ---
        h_pooled    = hidden[:, -1, :]                             # (B, D)
        r_hier_mean, r_hier_std = self.mc_reward.predict_with_uncertainty(
            h_pooled, n_samples=self.config.mc_dropout_n
        )

        # --- R_src: Steering-Reward Correspondence (scalar, broadcast to B) ---
        hier_raw  = self.hier(h_pooled)                           # (B,)
        r_src     = hier_raw * 0.0                                # placeholder; SRC injected by trainer
        r_src_std = torch.ones_like(r_src) * eps

        return {
            "r_prime": (r_prime,     r_prime_std),
            "r_const": (r_const,     r_const_std),
            "r_ccot":  (r_ccot,      r_ccot_std),
            "r_odin":  (r_odin,      r_odin_std),
            "r_hier":  (r_hier_mean, r_hier_std.clamp(min=eps)),
            "r_src":   (r_src,       r_src_std),
        }