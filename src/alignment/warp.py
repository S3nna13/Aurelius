"""Aurelius -- WARP: Weight Averaging Rewarded Policies (Rame et al. 2024).

Pure PyTorch implementation of the WARP alignment technique, which merges
multiple RLHF-fine-tuned policies in weight space via spherical interpolation
(SLERP) and then applies an anchor merge back toward the SFT reference model
to prevent reward hacking.

Algorithm:
  Phase 1: SFT fine-tuning -> theta_sft  (reference / anchor model)
  Phase 2: K independent RLHF fine-tunings -> {theta_1, ..., theta_K}
  Phase 3: SLERP-merge {theta_1, ..., theta_K} -> theta_merged
  Phase 4: Anchor merge: theta_final = lerp(theta_sft, theta_merged, alpha)
           alpha=1.0 -> pure merged policy; alpha=0.0 -> pure SFT
"""

from __future__ import annotations

import copy
import logging
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core SLERP utilities
# ---------------------------------------------------------------------------


def slerp_two(t: float, v0: Tensor, v1: Tensor, eps: float = 1e-8) -> Tensor:
    """Spherical linear interpolation between two flat weight vectors.

    Interpolates along the great circle on the unit hypersphere defined by
    v0 and v1.  When the two vectors are nearly collinear (dot product ~ 1)
    the function falls back to plain linear interpolation to avoid numerical
    instability.

    Args:
        t:   Interpolation parameter in [0, 1].  t=0 -> v0, t=1 -> v1.
        v0:  First weight vector (any shape; treated as a flat vector internally).
        v1:  Second weight vector, same shape as v0.
        eps: Threshold below which sin(theta) is considered zero (collinear
             vectors), triggering the linear-interpolation fallback.

    Returns:
        Interpolated tensor with the same shape as v0 and v1.
    """
    orig_shape = v0.shape
    v0_flat = v0.reshape(-1).float()
    v1_flat = v1.reshape(-1).float()

    # Normalise
    v0_norm = v0_flat / (v0_flat.norm() + eps)
    v1_norm = v1_flat / (v1_flat.norm() + eps)

    # Angle between the two vectors
    dot = torch.clamp(torch.dot(v0_norm, v1_norm), -1.0, 1.0)
    theta = torch.acos(dot)  # angle in [0, pi]

    sin_theta = torch.sin(theta)

    if sin_theta.abs().item() < eps:
        # Nearly collinear -> fall back to linear interpolation
        result = (1.0 - t) * v0_flat + t * v1_flat
    else:
        coef0 = torch.sin((1.0 - t) * theta) / sin_theta
        coef1 = torch.sin(t * theta) / sin_theta
        result = coef0 * v0_flat + coef1 * v1_flat

    return result.reshape(orig_shape).to(v0.dtype)


# ---------------------------------------------------------------------------
# Multi-policy SLERP merge
# ---------------------------------------------------------------------------


def merge_policies_slerp(
    policy_state_dicts: List[Dict[str, Tensor]],
    weights: Optional[List[float]] = None,
) -> Dict[str, Tensor]:
    """SLERP-merge K policy state_dicts into one merged state_dict.

    The merge is performed iteratively: the first two policies are blended
    with SLERP, and successive policies are blended into the running result.
    Uniform weights are used when ``weights`` is None.

    Args:
        policy_state_dicts: List of K state dicts, all with the same keys and
                            tensor shapes.
        weights:            Optional list of K non-negative floats.  If None,
                            each policy receives weight 1/K.

    Returns:
        A single merged state dict with the same keys and shapes.
    """
    K = len(policy_state_dicts)
    if K == 0:
        raise ValueError("policy_state_dicts must be non-empty")
    if K == 1:
        return {k: v.clone() for k, v in policy_state_dicts[0].items()}

    if weights is None:
        weights = [1.0 / K] * K
    else:
        if len(weights) != K:
            raise ValueError(
                f"len(weights)={len(weights)} must equal len(policy_state_dicts)={K}"
            )
        total = sum(weights)
        weights = [w / total for w in weights]  # normalise

    # Iterative pairwise SLERP weighted merge.
    # We accumulate a running "merged" dict and blend the next policy into it.
    # At each step we compute the effective interpolation parameter t such that
    # the cumulative weight of the new policy is respected.

    merged = {k: v.clone().float() for k, v in policy_state_dicts[0].items()}
    cumulative_weight = weights[0]

    for i in range(1, K):
        w_i = weights[i]
        cumulative_weight += w_i
        # t = fraction of the running result that should come from policy i
        t = w_i / cumulative_weight if cumulative_weight > 0 else 0.0

        new_merged: Dict[str, Tensor] = {}
        for key in merged:
            v0 = merged[key]
            v1 = policy_state_dicts[i][key].float()
            new_merged[key] = slerp_two(t, v0, v1)

        merged = new_merged

    # Cast back to original dtype
    orig_dtype = next(iter(policy_state_dicts[0].values())).dtype
    return {k: v.to(orig_dtype) for k, v in merged.items()}


# ---------------------------------------------------------------------------
# Anchor merge (linear interpolation toward SFT)
# ---------------------------------------------------------------------------


def anchor_merge(
    sft_state_dict: Dict[str, Tensor],
    merged_state_dict: Dict[str, Tensor],
    alpha: float = 0.5,
) -> Dict[str, Tensor]:
    """Linear interpolation between SFT (anchor) and merged policy.

    theta_final = (1 - alpha) * theta_sft + alpha * theta_merged

    Args:
        sft_state_dict:    State dict of the SFT reference model.
        merged_state_dict: State dict of the SLERP-merged policy.
        alpha:             Blend coefficient in [0, 1].  alpha=0 -> pure SFT;
                           alpha=1 -> pure merged policy.

    Returns:
        Final merged state dict.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    result: Dict[str, Tensor] = {}
    for key in sft_state_dict:
        sft_param = sft_state_dict[key].float()
        merged_param = merged_state_dict[key].float()
        blended = (1.0 - alpha) * sft_param + alpha * merged_param
        result[key] = blended.to(sft_state_dict[key].dtype)
    return result


# ---------------------------------------------------------------------------
# WARPTrainer
# ---------------------------------------------------------------------------


class WARPTrainer:
    """Trainer that implements the full WARP alignment pipeline.

    Args:
        sft_model:  Reference model (frozen copy kept internally as anchor).
        reward_fn:  Callable that maps logits (B, T, V) to a per-sample scalar
                    reward tensor of shape (B,).
        alpha:      Anchor-merge coefficient; 0 = pure SFT, 1 = pure merged.
        n_policies: Number of independent RLHF policies to train.
        lr:         Learning rate for each policy optimiser.
        kl_coef:    Coefficient for the KL penalty term in the RLHF objective.
    """

    def __init__(
        self,
        sft_model: nn.Module,
        reward_fn: Callable[[Tensor], Tensor],
        alpha: float = 0.5,
        n_policies: int = 3,
        lr: float = 1e-5,
        kl_coef: float = 0.1,
    ) -> None:
        self.sft_model = sft_model
        self.reward_fn = reward_fn
        self.alpha = alpha
        self.n_policies = n_policies
        self.lr = lr
        self.kl_coef = kl_coef

        # Frozen SFT reference (used for KL computation and anchor merge)
        self._ref_model = copy.deepcopy(sft_model)
        for p in self._ref_model.parameters():
            p.requires_grad_(False)
        self._ref_model.eval()

    # ------------------------------------------------------------------
    # KL penalty
    # ------------------------------------------------------------------

    def get_kl_penalty(self, logits: Tensor, ref_logits: Tensor) -> Tensor:
        """KL divergence KL(policy || ref) computed from logits.

        Args:
            logits:     Policy logits, shape (B, T, V) or (B, V).
            ref_logits: Reference model logits, same shape as logits.

        Returns:
            Scalar mean KL divergence (non-negative).
        """
        log_p = F.log_softmax(logits, dim=-1)
        log_q = F.log_softmax(ref_logits, dim=-1)
        # KL(P || Q) = sum_x P(x) * (log P(x) - log Q(x))
        p = log_p.exp()
        kl = (p * (log_p - log_q)).sum(dim=-1)  # sum over vocab
        return kl.mean()  # mean over (B, T) or (B,)

    # ------------------------------------------------------------------
    # Single policy fine-tuning
    # ------------------------------------------------------------------

    def train_policy(
        self,
        input_ids: Tensor,
        n_steps: int = 10,
    ) -> Dict[str, Tensor]:
        """Fine-tune one policy with reward + KL penalty from SFT.

        Creates a fresh deep copy of the SFT model and optimises it for
        ``n_steps`` gradient updates using the combined objective:
            loss = -reward + kl_coef * KL(policy || sft)

        Args:
            input_ids: Token ids of shape (B, T).
            n_steps:   Number of gradient update steps.

        Returns:
            State dict of the trained policy.
        """
        # Fresh policy initialised from SFT
        policy = copy.deepcopy(self.sft_model)
        policy.train()

        optimizer = torch.optim.Adam(policy.parameters(), lr=self.lr)

        for _ in range(n_steps):
            optimizer.zero_grad()

            # Forward pass through policy
            logits = self._forward(policy, input_ids)

            # Forward pass through frozen reference (no grad)
            with torch.no_grad():
                ref_logits = self._forward(self._ref_model, input_ids)

            # Reward signal (maximise -> negate for gradient descent)
            reward = self.reward_fn(logits)  # (B,)
            reward_loss = -reward.mean()

            # KL penalty
            kl = self.get_kl_penalty(logits, ref_logits)

            loss = reward_loss + self.kl_coef * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

        policy.eval()
        return {k: v.detach().clone() for k, v in policy.state_dict().items()}

    # ------------------------------------------------------------------
    # Full WARP pipeline
    # ------------------------------------------------------------------

    def run(self, input_ids: Tensor, n_steps: int = 5) -> nn.Module:
        """Run the full WARP pipeline and return the final merged model.

        Steps:
          1. Train ``n_policies`` independent policies via RLHF.
          2. SLERP-merge all K policies into theta_merged.
          3. Anchor-merge with SFT: theta_final = lerp(theta_sft, theta_merged, alpha).

        Args:
            input_ids: Token ids passed to each policy during training.
            n_steps:   Gradient steps per policy.

        Returns:
            Final nn.Module with weights set to theta_final.
        """
        logger.info("WARP: training %d policies for %d steps each", self.n_policies, n_steps)

        # Phase 2: train K policies
        policy_state_dicts: List[Dict[str, Tensor]] = []
        for i in range(self.n_policies):
            logger.debug("WARP: training policy %d/%d", i + 1, self.n_policies)
            sd = self.train_policy(input_ids, n_steps=n_steps)
            policy_state_dicts.append(sd)

        # Phase 3: SLERP-merge
        logger.info("WARP: SLERP-merging %d policies", self.n_policies)
        merged_sd = merge_policies_slerp(policy_state_dicts)

        # Phase 4: Anchor merge toward SFT
        logger.info("WARP: anchor merge with alpha=%.3f", self.alpha)
        sft_sd = {k: v.detach().clone() for k, v in self.sft_model.state_dict().items()}
        final_sd = anchor_merge(sft_sd, merged_sd, alpha=self.alpha)

        # Build final model
        final_model = copy.deepcopy(self.sft_model)
        final_model.load_state_dict(final_sd)
        final_model.eval()

        logger.info("WARP: pipeline complete")
        return final_model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _forward(model: nn.Module, input_ids: Tensor) -> Tensor:
        """Run a forward pass and return logits, handling multiple output formats."""
        out = model(input_ids)
        if isinstance(out, Tensor):
            return out
        if isinstance(out, (tuple, list)):
            # Convention used by AureliusTransformer: (loss, logits, cache)
            # or (logits, ...). Find the first Tensor with ndim >= 2.
            for item in out:
                if isinstance(item, Tensor) and item.ndim >= 2:
                    return item
        raise ValueError(f"Cannot extract logits from model output: {type(out)}")
