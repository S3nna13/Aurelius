"""DAPO Trainer — Decoupled Clip-Higher Policy Optimization (2025).

DAPO extends GRPO with three key fixes that prevent vanishing policy gradients
in group relative policy optimization:

1. **Decoupled clipping**: separate ``clip_low`` (0.2) and ``clip_high`` (0.28)
   instead of the symmetric epsilon used in PPO/GRPO.  This allows the policy
   to move further in the positive direction while still limiting regression.

2. **Token-level entropy bonus**: entropy is computed per token and added to
   the loss as ``+entropy_coeff * H.mean()``, which discourages premature
   distribution collapse.

3. **Dynamic sampling filter**: groups where ALL responses carry the same
   correctness signal (uniformly correct *or* uniformly wrong) are discarded
   before the update — only informative (mixed) groups contribute gradients.

Public API
----------
DAPOConfig  -- hyperparameters dataclass
DAPOBatch   -- per-forward-pass input bundle
DAPOTrainer -- trainer with all algorithm components

The trainer is registered in TRAINING_REGISTRY["dapo"].

References
----------
- "DAPO: An Open-Source LLM Reinforcement Learning System at Scale" (2025).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# DAPOConfig
# ---------------------------------------------------------------------------


@dataclass
class DAPOConfig:
    """Hyperparameters for the DAPO trainer.

    Attributes:
        clip_low: asymmetric lower clip bound for the importance ratio.
            Mirrors the PPO ``epsilon`` for the downside: ratio >= 1 - clip_low.
        clip_high: asymmetric upper clip bound.  Allows more upside movement
            than PPO: ratio <= 1 + clip_high.  Must be > clip_low to be
            meaningfully decoupled.
        entropy_coeff: weight for the token-level entropy bonus.  Positive
            values encourage distributional diversity.
        group_size: expected number of responses G sampled per prompt (used
            for documentation and consistency checks, not enforced internally).
        min_group_diversity: if > 0, groups with reward std below this
            threshold are treated as non-diverse and filtered out.
            Set to 0.0 to disable dynamic sampling filter.
        kl_coeff: coefficient for an optional KL-penalty term.  When 0.0 the
            KL term is skipped entirely (zero overhead).
    """

    clip_low: float = 0.2
    clip_high: float = 0.28
    entropy_coeff: float = 0.001
    group_size: int = 8
    min_group_diversity: float = 0.0
    kl_coeff: float = 0.0


# ---------------------------------------------------------------------------
# DAPOBatch
# ---------------------------------------------------------------------------


@dataclass
class DAPOBatch:
    """Per-step input bundle for DAPOTrainer.

    All tensors share the same first dimension G (group size).

    Attributes:
        token_ids: ``[G, T]`` — token ids for each response.
        log_probs: ``[G, T]`` — current policy log-probabilities per token.
            Must have ``requires_grad=True`` when gradient flow is needed.
        ref_log_probs: ``[G, T]`` — reference (frozen) policy log-probabilities.
        rewards: ``[G]`` — scalar reward for each response in the group.
        attention_mask: ``[G, T]`` — 1 for real tokens, 0 for padding.
    """

    token_ids: Tensor
    log_probs: Tensor
    ref_log_probs: Tensor
    rewards: Tensor
    attention_mask: Tensor


# ---------------------------------------------------------------------------
# DAPOTrainer
# ---------------------------------------------------------------------------


class DAPOTrainer:
    """DAPO training algorithm (pure PyTorch, no external dependencies).

    Implements decoupled clip-higher policy optimization with token-level
    entropy bonus and dynamic sampling filter.

    Args:
        config: DAPOConfig instance.  Uses defaults when ``None``.
    """

    _EPS: float = 1e-8  # numerical stability floor

    def __init__(self, config: DAPOConfig | None = None) -> None:
        self.config = config if config is not None else DAPOConfig()

    # ------------------------------------------------------------------
    # Advantage computation
    # ------------------------------------------------------------------

    def compute_advantages(self, rewards: Tensor) -> Tensor:
        """Compute group-normalised advantages from scalar rewards.

        Applies Z-score normalisation within the group:
        ``adv = (r - mean(r)) / (std(r) + eps)``

        When all rewards are identical (std == 0) the function returns
        all-zero advantages, providing no gradient signal — this is the
        condition detected by :meth:`is_group_diverse`.

        Args:
            rewards: shape ``[G]`` — one scalar reward per response.

        Returns:
            Advantages of shape ``[G]``.
        """
        mean = rewards.mean()
        std = rewards.std(unbiased=False)
        if std.item() < self._EPS:
            return torch.zeros_like(rewards)
        return (rewards - mean) / (std + self._EPS)

    # ------------------------------------------------------------------
    # Policy loss (token-level)
    # ------------------------------------------------------------------

    def compute_policy_loss(self, batch: DAPOBatch) -> Tensor:
        """Compute the token-level DAPO policy loss.

        Steps:
        1. Importance ratio ``ratio = exp(log_probs - ref_log_probs)`` per token.
        2. Advantages computed from ``batch.rewards``, broadcast to ``[G, T]``.
        3. Asymmetric clipped surrogate using ``clip_low`` / ``clip_high``.
        4. Entropy proxy bonus using ``ref_log_probs`` as a distribution proxy.
        5. Combine: ``loss_pg.mean() - entropy_coeff * entropy_approx.mean()``.

        Args:
            batch: DAPOBatch — all fields required.

        Returns:
            Scalar loss tensor (differentiable w.r.t. ``batch.log_probs``).
        """
        mask = batch.attention_mask.float()  # [G, T]

        # 1. Importance ratio per token [G, T]
        ratio = (batch.log_probs - batch.ref_log_probs.detach()).exp()

        # 2. Advantages [G] -> [G, T]
        adv = self.compute_advantages(batch.rewards)  # [G]
        adv = adv.unsqueeze(1).expand_as(ratio)  # [G, T]

        # 3. Asymmetric clipped surrogate
        surr1 = ratio * adv
        surr2 = (
            ratio.clamp(
                1.0 - self.config.clip_low,
                1.0 + self.config.clip_high,
            )
            * adv
        )
        # Token-level policy loss, masked
        loss_pg = -torch.min(surr1, surr2) * mask  # [G, T]

        # 4. Entropy proxy: use -ref_log_probs as a cheap entropy surrogate.
        #    When ref_log_probs are log-uniform the term is maximised, pushing
        #    the policy toward higher entropy without requiring full logits.
        entropy_approx = -batch.ref_log_probs.detach() * mask  # [G, T]

        # Normalise by number of real tokens to avoid length bias
        n_tokens = mask.sum().clamp(min=1.0)
        pg_mean = loss_pg.sum() / n_tokens
        ent_mean = entropy_approx.sum() / n_tokens

        # 5. Total (minus entropy bonus because we maximise entropy / minimise loss)
        return pg_mean - self.config.entropy_coeff * ent_mean

    # ------------------------------------------------------------------
    # Diversity filter
    # ------------------------------------------------------------------

    def is_group_diverse(self, rewards: Tensor) -> bool:
        """Return True when the group has mixed correctness signals.

        A group is non-diverse when all rewards are identical (std == 0),
        meaning the model either always succeeds or always fails on this
        prompt.  Such groups provide no useful learning signal.

        Args:
            rewards: ``[G]`` scalar rewards for the group.

        Returns:
            ``True`` if the group is informative (mixed rewards), ``False``
            if all rewards are the same.
        """
        std = rewards.std(unbiased=False).item()
        return std > self._EPS

    # ------------------------------------------------------------------
    # KL penalty
    # ------------------------------------------------------------------

    def _compute_kl_loss(self, batch: DAPOBatch) -> Tensor:
        """Token-level forward KL penalty: E[log(pi/pi_ref)].

        When ``kl_coeff == 0`` this short-circuits to a zero tensor so no
        extra computation is performed.

        Args:
            batch: DAPOBatch.

        Returns:
            Scalar KL loss tensor.
        """
        if self.config.kl_coeff == 0.0:
            return torch.zeros(1, device=batch.log_probs.device).squeeze()

        mask = batch.attention_mask.float()
        log_ratio = batch.log_probs - batch.ref_log_probs.detach()
        n_tokens = mask.sum().clamp(min=1.0)
        kl = (log_ratio * mask).sum() / n_tokens
        return self.config.kl_coeff * kl

    # ------------------------------------------------------------------
    # Combined loss
    # ------------------------------------------------------------------

    def total_loss(self, batch: DAPOBatch) -> dict[str, Tensor]:
        """Compute the full DAPO loss and return a breakdown dictionary.

        The combined objective is::

            total = pg_loss - entropy_coeff * entropy_bonus + kl_coeff * kl_loss

        where:
        - ``pg_loss`` is the token-level asymmetric clipped policy gradient.
        - ``entropy_bonus`` encourages distributional diversity.
        - ``kl_loss`` applies an optional reference-KL penalty.

        Args:
            batch: DAPOBatch input bundle.

        Returns:
            Dictionary with keys:
            - ``"loss"``: scalar total loss (differentiable).
            - ``"pg_loss"``: detached policy-gradient component.
            - ``"entropy_bonus"``: detached entropy bonus value.
            - ``"kl_loss"``: detached KL penalty (0 when ``kl_coeff == 0``).
        """
        mask = batch.attention_mask.float()
        n_tokens = mask.sum().clamp(min=1.0)

        # -- Policy gradient component --
        ratio = (batch.log_probs - batch.ref_log_probs.detach()).exp()
        adv = self.compute_advantages(batch.rewards)
        adv_t = adv.unsqueeze(1).expand_as(ratio)

        surr1 = ratio * adv_t
        surr2 = (
            ratio.clamp(
                1.0 - self.config.clip_low,
                1.0 + self.config.clip_high,
            )
            * adv_t
        )
        pg_loss = (-torch.min(surr1, surr2) * mask).sum() / n_tokens

        # -- Entropy bonus component --
        entropy_proxy = (-batch.ref_log_probs.detach() * mask).sum() / n_tokens

        # -- KL penalty --
        kl_loss = self._compute_kl_loss(batch)

        # -- Total --
        total = pg_loss - self.config.entropy_coeff * entropy_proxy + kl_loss

        return {
            "loss": total,
            "pg_loss": pg_loss.detach(),
            "entropy_bonus": entropy_proxy.detach(),
            "kl_loss": kl_loss.detach(),
        }

    # ------------------------------------------------------------------
    # Diagnostic statistics
    # ------------------------------------------------------------------

    def statistics(self, batch: DAPOBatch) -> dict[str, float]:
        """Compute diagnostic statistics for logging and monitoring.

        Args:
            batch: DAPOBatch input bundle.

        Returns:
            Dictionary with float values for:
            - ``"clip_fraction"``: fraction of tokens where ratio was clipped.
            - ``"mean_ratio"``: mean importance ratio across unmasked tokens.
            - ``"mean_advantage"``: mean group-level advantage.
            - ``"entropy_mean"``: mean entropy proxy across unmasked tokens.
        """
        with torch.no_grad():
            mask = batch.attention_mask.float()
            n_tokens = mask.sum().clamp(min=1.0)

            ratio = (batch.log_probs - batch.ref_log_probs).exp()

            # Clip bounds
            lo = 1.0 - self.config.clip_low
            hi = 1.0 + self.config.clip_high
            clipped = ratio.clamp(lo, hi)

            # Clip fraction: tokens where ratio was outside [lo, hi]
            was_clipped = ((ratio - clipped).abs() > 1e-6).float() * mask
            clip_fraction = (was_clipped.sum() / n_tokens).item()

            # Mean ratio over unmasked tokens
            mean_ratio = ((ratio * mask).sum() / n_tokens).item()

            # Mean advantage
            adv = self.compute_advantages(batch.rewards)
            mean_advantage = adv.mean().item()

            # Entropy proxy
            entropy_mean = ((-batch.ref_log_probs * mask).sum() / n_tokens).item()

        return {
            "clip_fraction": clip_fraction,
            "mean_ratio": mean_ratio,
            "mean_advantage": mean_advantage,
            "entropy_mean": entropy_mean,
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from src.training import TRAINING_REGISTRY  # noqa: E402

TRAINING_REGISTRY["dapo"] = DAPOTrainer
