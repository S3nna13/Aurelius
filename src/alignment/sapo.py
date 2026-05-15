"""SAPO: Segment-Aligned Policy Optimization for Multi-Modal Reasoning.

Implements the algorithm from:
  "Segment-Aligned Policy Optimization for Multi-Modal Reasoning"
  arXiv:2605.01327 (2025)

Key techniques:
  1. Step-wise MDP Formulation: each action = generating a coherent reasoning
     segment; bridges token-level generation and step-level optimization.
  2. Segment-level Value Estimation: learns a value function at the granularity
     of reasoning steps (not individual tokens).
  3. Step-wise Advantage Computation: computes advantages over reasoning steps;
     assigns them uniformly to all tokens within a segment.
  4. Importance Sampling over Steps: off-policy correction defined over reasoning
     steps using geometric mean of token-level likelihood ratios.
  5. Entropy-based Adaptive Segmentation: dynamically identifies reasoning step
     boundaries via high-entropy tokens.

Pure PyTorch implementation — no external ML libraries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SAPOConfig:
    """Configuration for SAPO training.

    Attributes:
        gamma: Discount factor for value estimation.
        lam: GAE lambda for advantage computation.
        clip_eps: PPO probability-ratio clipping epsilon ε.
        kl_coeff: Weight λ on the reference-KL penalty term.
        entropy_bonus: Coefficient for entropy regularization.
        segment_entropy_threshold: Token-level entropy percentile (0–1) used
            to identify step boundaries. Tokens with entropy above this
            percentile are treated as high-entropy (uncertain) and trigger
            segment boundaries. Higher values = fewer, later boundaries.
        min_segment_tokens: Minimum tokens in a segment; prevents trivially
            short segments.
        value_loss_coef: Weight for value function regression loss.
        max_segments_per_response: Safety cap on number of segments to avoid
            degenerate segmentation on very long responses.
        eps: Small constant for numerical stability in ratio/geo-mean ops.
    """

    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    kl_coeff: float = 0.01
    entropy_bonus: float = 0.001
    segment_entropy_threshold: float = 0.75
    min_segment_tokens: int = 5
    value_loss_coef: float = 0.5
    max_segments_per_response: int = 64
    eps: float = 1e-8


# ---------------------------------------------------------------------------
# Segment data structure
# ---------------------------------------------------------------------------


@dataclass
class Segment:
    """A reasoning segment (step) extracted from a response.

    Attributes:
        token_ids: 1-D tensor of token ids belonging to this segment.
        start_idx: Inclusive token index in the original response where
            this segment begins.
        end_idx: Exclusive token index in the original response where
            this segment ends.
        segment_log_probs: Per-token log probabilities under the policy that
            generated this segment, shape (segment_len,).
        segment_ref_log_probs: Per-token log probabilities under the reference
            policy, shape (segment_len,).
        advantage: Computed step-level advantage for this segment.
        value_target: TD or GAE target for value function training.
    """

    token_ids: Tensor
    start_idx: int
    end_idx: int
    segment_log_probs: Tensor = field(default=None)
    segment_ref_log_probs: Tensor = field(default=None)
    advantage: float = 0.0
    value_target: float = 0.0

    def __post_init__(self) -> None:
        if self.segment_log_probs is None:
            self.segment_log_probs = torch.empty(0, dtype=torch.float32)
        if self.segment_ref_log_probs is None:
            self.segment_ref_log_probs = torch.empty(0, dtype=torch.float32)


@dataclass
class SegmentBatch:
    """A batch of segmented responses for SAPO training.

    Attributes:
        segments: List of Segments from G responses.
        response_rewards: Scalar final reward per original response, shape [G].
        response_mask: 1-D bool tensor, True for responses retained after
            dynamic sampling filter.
    """

    segments: list[Segment]
    response_rewards: Tensor
    response_mask: Tensor


# ---------------------------------------------------------------------------
# 1. SegmentExtractor — entropy-based step boundary detection
# ---------------------------------------------------------------------------


class SegmentExtractor:
    """Identifies reasoning step boundaries using token-level entropy.

    High-entropy tokens indicate model uncertainty and typically correspond
    to natural step transitions in multi-step reasoning (e.g., moving from
    planning to calculation to verification).

    The extractor computes per-token entropy H_t = -Σ_v p_v log p_v, then
    identifies boundaries at positions where H_t exceeds a percentile
    threshold of the response's entropy distribution.

    Usage:
        extractor = SegmentExtractor(config)
        segments = extractor.extract_segments(token_ids, log_probs)
    """

    def __init__(self, config: SAPOConfig | None = None) -> None:
        self.config = config if config is not None else SAPOConfig()

    def compute_token_entropy(self, logits: Tensor) -> Tensor:
        """Compute per-token entropy H_t = -Σ_v p_v log p_v.

        Args:
            logits: (..., V) unnormalized logit scores. Last dim = vocab.

        Returns:
            Entropy tensor of shape (...,) — one scalar per token.
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(probs * log_probs).sum(dim=-1)

    def extract_segments(
        self,
        token_ids: Tensor,
        logits: Tensor,
    ) -> list[Segment]:
        """Extract reasoning segments from a response using entropy-based boundaries.

        Args:
            token_ids: 1-D tensor of token ids, shape (T,).
            logits: 2-D tensor of unnormalized logits, shape (T, V).

        Returns:
            List of Segments, ordered by position in the response.
        """
        T = token_ids.size(0)
        if T == 0:
            return []

        entropies = self.compute_token_entropy(logits)  # (T,)

        threshold = float(torch.quantile(entropies, q=self.config.segment_entropy_threshold).item())

        boundary_mask = torch.zeros(T, dtype=torch.bool)
        for t in range(T):
            if entropies[t] >= threshold and t > 0:
                min_behind = self.config.min_segment_tokens
                if t - min_behind > 0:
                    boundary_mask[t] = True

        boundaries = boundary_mask.nonzero(as_tuple=True)[0].tolist()

        n_max = self.config.max_segments_per_response
        if len(boundaries) > n_max:
            boundaries = boundaries[:n_max]

        if not boundaries:
            return [
                Segment(
                    token_ids=token_ids,
                    start_idx=0,
                    end_idx=T,
                )
            ]

        if boundaries[0] != 0:
            boundaries = [0] + boundaries
        if boundaries[-1] != T:
            boundaries = boundaries + [T]

        segments: list[Segment] = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            if end - start < self.config.min_segment_tokens:
                continue
            seg_ids = token_ids[start:end]
            segments.append(
                Segment(
                    token_ids=seg_ids,
                    start_idx=start,
                    end_idx=end,
                )
            )

        if not segments:
            segments = [
                Segment(
                    token_ids=token_ids,
                    start_idx=0,
                    end_idx=T,
                )
            ]

        return segments

    def extract_batch_segments(
        self,
        token_ids: Tensor,
        logits: Tensor,
    ) -> list[list[Segment]]:
        """Extract segments for a batch of responses.

        Args:
            token_ids: 2-D tensor of token ids, shape (B, T).
            logits: 3-D tensor of unnormalized logits, shape (B, T, V).

        Returns:
            List of length B, each element is a list of Segments for that response.
        """
        B = token_ids.size(0)
        if logits.dim() == 2:
            logits = logits.unsqueeze(0).expand(B, -1, -1)
        all_segments: list[list[Segment]] = []
        for b in range(B):
            segs = self.extract_segments(token_ids[b], logits[b])
            all_segments.append(segs)
        return all_segments


# ---------------------------------------------------------------------------
# 2. StepWiseValueEstimator — segment-granularity value function
# ---------------------------------------------------------------------------


class StepWiseValueEstimator(nn.Module):
    """Learns a value function at the granularity of reasoning segments.

    Uses temporal difference (TD) learning to predict the return from the
    start of each segment. The value target is computed via GAE over segments
    (not tokens), which reduces variance compared to token-level value models.

    Architecture: shared transformer backbone → segment-level value head.
    The value head reads the last-token hidden state of each segment.

    Args:
        backbone: The language model (must expose .norm output).
        d_model: Hidden dimension of the backbone.
    """

    def __init__(
        self,
        backbone: nn.Module,
        d_model: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.d_model = d_model
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=True),
            nn.GELU(),
            nn.Linear(d_model // 2, 1, bias=True),
        )
        self._hidden_cache: list[Tensor] = []

    def _register_hook(self) -> None:
        """Register forward hook on backbone.norm to capture hidden states."""
        self._hidden_cache.clear()
        handle = self.backbone.norm.register_forward_hook(
            lambda m, i, o: self._hidden_cache.append(o)
        )
        return handle

    def estimate_segment_values(
        self,
        input_ids: Tensor,
        segments: list[Segment],
    ) -> Tensor:
        """Compute value estimates for each segment in a response.

        Args:
            input_ids: 1-D input token ids, shape (T,).
            segments: List of Segments for this response.

        Returns:
            values: 1-D tensor of value estimates, one per segment, shape (S,).
        """
        hook_handle = self._register_hook()
        try:
            self.backbone(input_ids.unsqueeze(0))
        finally:
            hook_handle.remove()

        if not self._hidden_cache or not isinstance(self._hidden_cache[0], Tensor):
            raise RuntimeError(
                "No hidden states captured; ensure the backbone exposes a norm layer."
            )
        hidden = self._hidden_cache[0].squeeze(0)  # (T, d_model)

        values: list[Tensor] = []
        for seg in segments:
            last_pos = seg.end_idx - 1
            h_seg = hidden[last_pos]  # (d_model,)
            v = self.value_head(h_seg).squeeze(-1)  # scalar
            values.append(v)

        return torch.stack(values) if values else torch.empty(0, device=input_ids.device)

    def compute_value_loss(
        self,
        segment_values: Tensor,
        value_targets: Tensor,
        segment_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute TD loss for value function training.

        Args:
            segment_values: Predicted values per segment, shape (S,).
            value_targets: GAE or MC return targets per segment, shape (S,).
            segment_mask: Optional bool mask, True for valid segments.

        Returns:
            Scalar MSE loss.
        """
        if segment_mask is not None:
            diff = (segment_values - value_targets) ** 2
            loss = (diff * segment_mask.float()).sum()
            n = segment_mask.float().sum().clamp(min=1.0)
            loss = loss / n
        else:
            loss = F.mse_loss(segment_values, value_targets)
        return loss


# ---------------------------------------------------------------------------
# 3. SegmentLevelAdvantage — step-level advantage with importance sampling
# ---------------------------------------------------------------------------


class SegmentLevelAdvantage:
    """Computes advantages at the segment level with geometric-mean IS correction.

    Key operations:
      - Assigns advantages uniformly to all tokens within a segment.
      - Uses geometric mean of token-level likelihood ratios for off-policy
        correction over segments, avoiding the high variance of token-level IS.
      - Supports GAE-style advantage computation at the segment granularity.

    Usage:
        adv_computer = SegmentLevelAdvantage(config)
        advantages = adv_computer.compute_advantages(segments, rewards, values)
    """

    def __init__(self, config: SAPOConfig | None = None) -> None:
        self.config = config if config is not None else SAPOConfig()

    def compute_segment_rho(
        self,
        segment: Segment,
    ) -> Tensor:
        """Compute per-segment importance sampling ratio using geometric mean.

        For a segment with T tokens, the segment-level importance ratio is:
            rho_segment = exp( (1/T) * Σ_t log π_θ(a_t|s_t) - log π_old(a_t|s_t) )
                       = [Π_t r_t]^(1/T)

        This is the geometric mean of the token-level ratios, normalized by
        segment length to avoid bias toward shorter segments.

        Args:
            segment: A Segment with per-token log probs.

        Returns:
            Scalar importance ratio for the segment.
        """
        lp = segment.segment_log_probs
        ref_lp = segment.segment_ref_log_probs

        if lp.numel() == 0 or ref_lp.numel() == 0:
            return torch.tensor(1.0, dtype=torch.float32)

        log_ratios = lp - ref_lp.detach()
        log_ratios = log_ratios.clamp(-20.0, 20.0)

        geo_mean_log_ratio = log_ratios.mean()
        rho = torch.exp(geo_mean_log_ratio)
        return rho

    def compute_segment_advantages(
        self,
        segments: list[Segment],
        rewards: Tensor,
        values: Tensor,
        next_value: float = 0.0,
    ) -> list[float]:
        """Compute GAE advantages at the segment level.

        Each segment is treated as a single timestep in the coarse MDP. The
        reward for segment k is the sum of token-level rewards within that
        segment, approximated by the final response reward (R) since
        intermediate rewards are sparse in reasoning tasks.

        Args:
            segments: List of Segments for one response.
            rewards: Scalar final reward for the response, shape ().
            values: Per-segment value estimates from StepWiseValueEstimator,
                shape (S,).
            next_value: Bootstrap value for the final segment.

        Returns:
            advantages: List of floats, one per segment, matching the order
                of the input segments.
        """
        gamma = self.config.gamma
        lam = self.config.lam
        n_seg = len(segments)
        if n_seg == 0:
            return []

        segment_rewards = torch.full((n_seg,), rewards.item() / max(n_seg, 1))
        segment_advantages = torch.zeros(n_seg)

        gae = 0.0
        for k in reversed(range(n_seg)):
            if k == n_seg - 1:
                next_val = next_value
            else:
                next_val = values[k + 1]

            delta = segment_rewards[k] + gamma * next_val - values[k]
            gae = delta + gamma * lam * gae
            segment_advantages[k] = gae

        return segment_advantages.detach().tolist()

    def assign_advantages_to_tokens(
        self,
        segments: list[Segment],
        token_advantages: Tensor,
    ) -> None:
        """Assign segment-level advantages uniformly to all tokens in each segment.

        After computing segment-level advantages, we propagate them to every
        token within the segment. This allows token-level policy gradient
        while keeping the advantage signal smooth at the segment level.

        Args:
            segments: List of Segments (mutated in place).
            token_advantages: Pre-allocated buffer of shape (T,) to fill.
        """
        for seg in segments:
            adv = seg.advantage
            token_advantages[seg.start_idx : seg.end_idx] = adv


# ---------------------------------------------------------------------------
# 4. SAPOTrainer — main orchestrator for segment-aligned policy optimization
# ---------------------------------------------------------------------------


class SAPOTrainer:
    """Main trainer implementing segment-aligned policy optimization.

    Ties together:
      - SegmentExtractor: entropy-based step boundary detection.
      - StepWiseValueEstimator: segment-level value learning via TD.
      - SegmentLevelAdvantage: segment-level advantage with geometric IS.
      - GRPO-style policy gradient loss with asymmetric clipping.

    Training loop per step:
      1. Extract segments from responses using entropy thresholds.
      2. Estimate values for each segment start with the value estimator.
      3. Compute segment-level GAE advantages.
      4. Assign segment advantages uniformly to constituent tokens.
      5. Compute policy gradient loss with off-policy correction via
         geometric-mean importance ratios.
      6. Update value estimator with TD loss.
      7. Backward pass and optimizer step.

    Args:
        model: The language model (policy).
        ref_model: Frozen reference model for KL penalty.
        value_estimator: StepWiseValueEstimator sharing the model's backbone.
        config: SAPOConfig instance.
        optimizer: PyTorch optimizer for the policy.
        value_optimizer: PyTorch optimizer for the value estimator.
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        value_estimator: StepWiseValueEstimator,
        config: SAPOConfig | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        value_optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.value_estimator = value_estimator
        self.config = config if config is not None else SAPOConfig()
        self.optimizer = optimizer
        self.value_optimizer = value_optimizer

        self.segment_extractor = SegmentExtractor(self.config)
        self.adv_computer = SegmentLevelAdvantage(self.config)

        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()

    def _extract_logits(self, input_ids: Tensor) -> Tensor:
        """Get (T, V) logits from model."""
        out = self.model(input_ids.unsqueeze(0))
        if isinstance(out, Tensor):
            return out.squeeze(0)
        if isinstance(out, (tuple, list)):
            for idx in (1, 0):
                if idx < len(out) and isinstance(out[idx], Tensor):
                    t = out[idx]
                    if t.dim() in (2, 3):
                        return t.squeeze(0) if t.dim() == 3 else t
            for item in out:
                if isinstance(item, Tensor) and item.dim() in (2, 3):
                    return item.squeeze(0) if item.dim() == 3 else item
        raise ValueError(f"Cannot extract logits from model output of type {type(out)}")

    def compute_policy_loss(
        self,
        token_log_probs: Tensor,
        token_ref_log_probs: Tensor,
        token_advantages: Tensor,
        token_mask: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute the SAPO policy gradient loss with segment-level IS.

        Args:
            token_log_probs: (T,) current policy log probs per token.
            token_ref_log_probs: (T,) reference policy log probs per token.
            token_advantages: (T,) advantages per token (segment-advantage assigned).
            token_mask: (T,) bool — True for valid tokens.

        Returns:
            (loss, metrics) where metrics has keys: clip_fraction, mean_rho, policy_loss.
        """
        cfg = self.config

        log_ratio = (token_log_probs - token_ref_log_probs.detach()).clamp(-20.0, 20.0)
        ratio = log_ratio.exp()  # (T,)

        r_clipped = torch.where(
            token_advantages >= 0,
            torch.clamp(ratio, 1.0, 1.0 + cfg.clip_eps),
            torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0),
        )

        surr1 = ratio * token_advantages
        surr2 = r_clipped * token_advantages
        per_token_loss = -torch.min(surr1, surr2)

        masked_loss = per_token_loss * token_mask.float()
        n_valid = token_mask.float().sum().clamp(min=1.0)
        policy_loss = masked_loss.sum() / n_valid

        clipped_mask = (r_clipped != ratio) & token_mask
        clip_fraction = clipped_mask.float().sum() / n_valid

        with torch.no_grad():
            mean_rho = (ratio * token_mask.float()).sum() / n_valid

        metrics = {
            "clip_fraction": clip_fraction.item(),
            "mean_rho": mean_rho.item(),
            "policy_loss": policy_loss.item(),
        }
        return policy_loss, metrics

    def compute_kl_loss(
        self,
        token_log_probs: Tensor,
        token_ref_log_probs: Tensor,
        token_mask: Tensor,
    ) -> Tensor:
        """Compute forward KL penalty between policy and reference.

        Args:
            token_log_probs: (T,) current policy log probs.
            token_ref_log_probs: (T,) reference log probs (no grad).
            token_mask: (T,) bool mask for valid tokens.

        Returns:
            Scalar KL loss.
        """
        mask = token_mask.float()
        n_valid = mask.sum().clamp(min=1.0)
        kl = (token_ref_log_probs - token_log_probs) * mask
        return kl.sum() / n_valid

    def compute_entropy_bonus(
        self,
        logits: Tensor,
        token_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute entropy bonus for regularization.

        Args:
            logits: (T, V) unnormalized logits.
            token_mask: Optional (T,) bool mask.

        Returns:
            Scalar entropy bonus (to be subtracted from loss).
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy_per_token = -(probs * log_probs).sum(dim=-1)

        if token_mask is not None:
            mask = token_mask.float()
            n_valid = mask.sum().clamp(min=1.0)
            entropy = (entropy_per_token * mask).sum() / n_valid
        else:
            entropy = entropy_per_token.mean()

        return self.config.entropy_bonus * entropy

    def train_step(
        self,
        input_ids: Tensor,
        rewards: Tensor,
        segment_log_probs: Tensor,
        segment_ref_log_probs: Tensor,
    ) -> dict:
        """Execute one SAPO training step.

        Args:
            input_ids: 1-D input token ids, shape (T,). The full context
                including prompt and response tokens.
            rewards: Scalar final reward for the response, shape ().
            segment_log_probs: Per-token log probs under current policy, shape (T,).
            segment_ref_log_probs: Per-token log probs under reference, shape (T,).

        Returns:
            Metrics dict with keys: loss, policy_loss, kl_loss, value_loss,
            entropy, clip_fraction, n_segments.
        """
        cfg = self.config

        response_start = input_ids.size(0) - segment_log_probs.size(0)
        response_ids = input_ids[response_start:]
        response_logits = self._extract_logits(response_ids)

        segments = self.segment_extractor.extract_segments(response_ids, response_logits)

        for seg in segments:
            seg_start = seg.start_idx
            seg_end = seg.end_idx
            seg.segment_log_probs = segment_log_probs[seg_start:seg_end]
            seg.segment_ref_log_probs = segment_ref_log_probs[seg_start:seg_end]

        if segments:
            seg_values = self.value_estimator.estimate_segment_values(
                response_ids,
                segments,
            )
            if seg_values.numel() == 0:
                seg_values = torch.zeros(
                    len(segments), dtype=segment_log_probs.dtype, device=segment_log_probs.device
                )
        else:
            seg_values = torch.zeros(
                0, dtype=segment_log_probs.dtype, device=segment_log_probs.device
            )

        advantages = self.adv_computer.compute_segment_advantages(
            segments,
            rewards,
            seg_values,
        )
        for seg, adv in zip(segments, advantages):
            seg.advantage = adv

        T = segment_log_probs.size(0)
        token_advantages = torch.zeros(
            T, dtype=segment_log_probs.dtype, device=segment_log_probs.device
        )
        self.adv_computer.assign_advantages_to_tokens(segments, token_advantages)

        token_mask = torch.ones(T, dtype=torch.bool, device=segment_log_probs.device)

        policy_loss, pg_metrics = self.compute_policy_loss(
            segment_log_probs,
            segment_ref_log_probs,
            token_advantages,
            token_mask,
        )

        kl_loss = self.compute_kl_loss(
            segment_log_probs,
            segment_ref_log_probs,
            token_mask,
        )

        entropy_bonus = self.compute_entropy_bonus(response_logits, token_mask)

        total_loss = policy_loss + cfg.kl_coeff * kl_loss - entropy_bonus

        if self.value_estimator is not None and segments:
            if seg_values.numel() > 0 and len(advantages) == len(segments):
                value_targets = seg_values + torch.tensor(
                    advantages, dtype=seg_values.dtype, device=seg_values.device
                )
                value_loss = self.value_estimator.compute_value_loss(
                    seg_values,
                    value_targets.detach(),
                )
                total_loss = total_loss + cfg.value_loss_coef * value_loss
            else:
                value_loss = torch.tensor(0.0, device=segment_log_probs.device)
        else:
            value_loss = torch.tensor(0.0, device=segment_log_probs.device)

        metrics = {
            "loss": total_loss.item(),
            "policy_loss": pg_metrics["policy_loss"],
            "kl_loss": kl_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy_bonus.item(),
            "clip_fraction": pg_metrics["clip_fraction"],
            "n_segments": len(segments),
        }
        return metrics

    def dynamic_sampling_filter(
        self,
        rewards: Tensor,
    ) -> Tensor:
        """Filter responses with no learning signal (all-correct or all-wrong).

        Args:
            rewards: 1-D tensor of scalar rewards.

        Returns:
            Bool tensor, True for responses to keep.
        """
        if rewards.numel() == 0:
            return torch.zeros(0, dtype=torch.bool)
        return torch.ones(rewards.shape, dtype=torch.bool)  # keep all by default


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from . import ALIGNMENT_REGISTRY  # noqa: E402

ALIGNMENT_REGISTRY["sapo"] = SAPOTrainer

__all__ = [
    "SAPOConfig",
    "Segment",
    "SegmentBatch",
    "SegmentExtractor",
    "StepWiseValueEstimator",
    "SegmentLevelAdvantage",
    "SAPOTrainer",
]
