from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.alignment.constitutional_ai_v3 import CritiqueHead
from src.alignment.praxis.config import PRAXISConfig
from src.alignment.praxis.expert_safety_affinity import ExpertSafetyAffinity
from src.alignment.praxis.mtah import MultiTokenAlignmentHorizon
from src.alignment.praxis.praxis_loss import PRAXISLoss
from src.alignment.praxis.precision_fusion import PrecisionFusion
from src.alignment.praxis.reward_signals import RewardSignalBundle
from src.alignment.praxis.steering_reward import SteeringRewardCorrespondence
from src.alignment.prime import PRIMEConfig, PRIMEReward
from src.alignment.reward_uncertainty import MCDropoutReward
from src.alignment.thinking_tokens import ThinkingLossWeights
from src.alignment.warp import anchor_merge


class PRAXISTrainer:
    """PRAXIS unified alignment trainer.

    Orchestrates 6-signal reward decomposition with PrecisionFusion, DAPO
    policy gradient, constitutional gradient gating, ESA routing loss, SRC
    reward signal, MTAH temporal advantage extension, and periodic WARP merge.
    """

    def __init__(
        self,
        model: nn.Module,
        config: PRAXISConfig,
        ref_state_dict: dict | None = None,
        sft_state_dict: dict | None = None,
    ) -> None:
        self.model  = model
        self.config = config
        self.ref_state_dict = ref_state_dict or {}
        self.sft_state_dict = sft_state_dict or {}

        # Sub-components
        self.prime       = PRIMEReward(PRIMEConfig(beta=config.beta_kl))
        self.critique    = CritiqueHead(config.d_model, config.n_principles)
        self.mc_reward   = MCDropoutReward(config.d_model)
        self.hier        = None  # HierarchicalRewardModel requires explicit criteria list
        self.think_wts   = ThinkingLossWeights(config.think_weight, config.answer_weight)
        self.fusion      = PrecisionFusion(n_signals=6)
        self.src         = SteeringRewardCorrespondence(model, config)
        self.mtah        = MultiTokenAlignmentHorizon(config)
        self.loss_fn     = PRAXISLoss(config)

        # ESA uses only MoE layers (those with a router attribute)
        moe_layers = nn.ModuleList([
            layer for layer in model.layers
            if hasattr(layer, "ffn") and hasattr(getattr(layer, "ffn", None), "router")
        ])
        self.esa = ExpertSafetyAffinity(moe_layers, config)

        self.bundle = RewardSignalBundle(config, self.prime, self.critique,
                                        self.hier or (lambda x: torch.zeros(x.shape[0])),
                                        self.mc_reward)

    def _get_log_probs(self, model_out, input_ids: Tensor, mask: Tensor) -> Tensor:
        _, logits, _, _ = model_out
        log_probs = F.log_softmax(logits, dim=-1)
        target = input_ids.clamp(min=0)
        gathered = log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)
        return gathered * mask.float()

    def train_step(self, batch: dict, step: int) -> dict:
        cfg = self.config
        input_ids = batch["input_ids"]
        labels    = batch.get("labels", input_ids)
        mask      = batch.get("attention_mask", torch.ones_like(input_ids, dtype=torch.bool))

        self.model.train()

        # Forward pass with hidden states
        model_out = self.model(input_ids, mask=mask, labels=labels, return_hidden_states=True)
        _, logits, _, hidden = model_out                            # (B, T, D)

        # Current policy log-probs — must stay outside no_grad to keep computation graph
        log_probs_cur = self._get_log_probs(model_out, input_ids, mask)   # (B, T)

        # Reference log-probs: detached clone of current policy.
        # NOTE: using current policy as its own reference makes PRIME implicit reward identically
        # zero this step. Wire in a frozen SFT/reference model forward pass here to activate PRIME.
        with torch.no_grad():
            ref_log_probs = log_probs_cur.detach().clone()         # (B, T)

        # Outcome rewards: cross-entropy of correct tokens (negated = positive reward for low loss)
        B, T = input_ids.shape
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_labels = labels.reshape(-1).clamp(min=0)
        ce_per_token = F.cross_entropy(flat_logits, flat_labels, reduction="none")
        outcome_rewards = -ce_per_token.reshape(B, T).mean(dim=1)   # (B,)
        # Per-batch z-score normalization (GroupRewardNormalizer expects grouped rollouts)
        r_std = outcome_rewards.std(unbiased=False).clamp(min=1e-8)
        outcome_rewards = (outcome_rewards - outcome_rewards.mean()) / r_std

        # Compute 6 reward signals
        signals = self.bundle.compute(hidden, log_probs_cur, ref_log_probs, outcome_rewards, mask)

        # Inject SRC into r_src slot
        src_scalar = self.src.compute(input_ids)
        r_src_mean = torch.full((B,), src_scalar.item(), device=hidden.device)
        r_src_std  = torch.ones(B, device=hidden.device) * 1e-6
        signals["r_src"] = (r_src_mean, r_src_std)

        # PrecisionFusion
        means = [signals[k][0] for k in ["r_prime", "r_const", "r_ccot", "r_odin", "r_hier", "r_src"]]
        stds  = [signals[k][1] for k in ["r_prime", "r_const", "r_ccot", "r_odin", "r_hier", "r_src"]]
        fused_reward = self.fusion.fuse(means, stds)                 # (B,)

        # MTAH advantage extension — detach so advantages don't carry spurious grad from PRIME
        advantages_base = fused_reward.detach().unsqueeze(1).expand(-1, T)  # (B, T)
        advantages = self.mtah.extend(advantages_base)                      # (B, T)

        # Thinking token weights — computed per sample so <think> boundaries are respected
        think_weights = torch.stack(
            [self.think_wts.compute_weights(input_ids[i].tolist()) for i in range(B)],
            dim=0,
        ).to(hidden.device)                              # (B, T)
        advantages = advantages * think_weights

        # Constitutional scores for gating
        with torch.no_grad():
            const_scores = signals["r_const"][0]                     # (B,)

        # ESA routing loss
        esa_loss = self.esa.compute(hidden.detach(), const_scores)

        # Per-token entropy approximation: -log p(chosen) is MC estimate of H(π) per position
        entropy = -log_probs_cur * mask.float()   # (B, T)
        total_loss, metrics = self.loss_fn.forward(
            log_probs_cur, log_probs_cur.detach(), advantages, fused_reward, mask,
            ref_log_probs=ref_log_probs, const_scores=const_scores,
            entropy=entropy, esa_loss=esa_loss,
        )

        # WARP periodic anchor merge
        if step % cfg.warp_interval == 0 and self.sft_state_dict and hasattr(self.model, "state_dict"):
            current_sd = self.model.state_dict()
            merged_sd  = anchor_merge(self.sft_state_dict, current_sd, alpha=1 - cfg.warp_anchor_mu)
            self.model.load_state_dict(merged_sd, strict=False)

        metrics["fused_reward_mean"] = fused_reward.mean().item()
        metrics["src_reward"]        = src_scalar.item()
        return metrics