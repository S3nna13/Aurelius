"""Aurelius -- Rejection Sampling Optimization (RSO) alignment.

Native PyTorch implementation of RSO as described in:
  "Statistical Rejection Sampling Improves Preference Optimization"
  Liu et al., arXiv:2309.06657 (2023)

Key insight (Section 3): DPO's optimality assumption requires data sampled
from the optimal policy pi*. RSO approximates this by using rejection sampling
to draw (y_w, y_l) pairs from a proxy of pi*, then applying the DPO loss.

Paper notation is preserved throughout:
  pi       -- policy being trained
  pi_ref   -- frozen reference (SFT) policy
  R        -- reward model
  beta     -- KL-regularisation temperature
  q        -- proposal distribution (typically SFT policy, i.e. pi_ref)
  Z        -- partition function (normalisation constant)
  y_w      -- winning (preferred) response
  y_l      -- losing (rejected) response
  w        -- importance weight for pre-collected pairs (Section 4)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RSOSampler
# ---------------------------------------------------------------------------

class RSOSampler:
    """Rejection-sampling wrapper (Algorithm 1 of Liu et al., 2023).

    Given a reward model R and a set of candidate responses drawn from a
    proposal distribution q (e.g. SFT policy), accepts each candidate with
    probability:

        A(y) = min(1, exp(R(x,y) - max_R))

    where max_R = max_i R(x, y_i) acts as the normalisation anchor Z(x)
    (unnormalised accept/reject step).  After acceptance, the caller can rank
    the accepted responses by R to construct (y_w, y_l) pairs.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = torch.Generator()
        if seed is not None:
            self._rng.manual_seed(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rejection_sample(
        self,
        reward_scores: torch.Tensor,
        log_probs: torch.Tensor,
        n_samples: int,
    ) -> torch.Tensor:
        """Select a subset of candidates via rejection sampling.

        Accept-reject step (Algorithm 1, line 2):
            accept candidate i with probability
                p_i = min(1, exp(reward_scores[i] - max(reward_scores)))

        The log_probs argument is kept for interface completeness (and future
        importance-corrected variants) but the pure rejection-sampling step
        only uses reward_scores for the accept probability.

        Args:
            reward_scores: Shape (N,) -- R(x, y_i) for each candidate.
            log_probs:     Shape (N,) -- log q(y_i|x) from the proposal.
            n_samples:     How many accepted indices to return (<= N).

        Returns:
            selected_indices: 1-D LongTensor of length <= n_samples holding
                              indices into the original candidate list, sorted
                              by descending reward (best first).

        Raises:
            ValueError: if n_samples < 1 or inputs are empty / mismatched.
        """
        if reward_scores.ndim != 1 or log_probs.ndim != 1:
            raise ValueError(
                "reward_scores and log_probs must be 1-D tensors; "
                f"got shapes {reward_scores.shape}, {log_probs.shape}"
            )
        N = reward_scores.shape[0]
        if N == 0:
            raise ValueError("reward_scores must be non-empty.")
        if log_probs.shape[0] != N:
            raise ValueError(
                f"reward_scores and log_probs must have equal length; "
                f"got {N} vs {log_probs.shape[0]}"
            )
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}.")

        # Clamp n_samples to available candidates
        n_samples = min(n_samples, N)

        # Compute unnormalised acceptance probabilities
        # p_i = exp(R_i - R_max)  in (0, 1]
        r = reward_scores.float()
        r_max = r.max()
        accept_probs = torch.exp(r - r_max)  # shape (N,)

        # Draw uniform samples and apply accept/reject
        u = torch.rand(N, generator=self._rng, dtype=torch.float32,
                       device=reward_scores.device)
        accepted_mask = u < accept_probs  # shape (N,)
        accepted_indices = accepted_mask.nonzero(as_tuple=False).squeeze(1)

        if accepted_indices.numel() == 0:
            # Guarantee at least the highest-reward sample is returned
            accepted_indices = torch.tensor([int(r.argmax())],
                                            device=reward_scores.device)

        # Sort by descending reward and take top-n_samples
        order = r[accepted_indices].argsort(descending=True)
        accepted_indices = accepted_indices[order]

        return accepted_indices[:n_samples]


# ---------------------------------------------------------------------------
# RSOLoss
# ---------------------------------------------------------------------------

class RSOLoss:
    """RSO loss function (Sections 3 and 4, Liu et al., 2023).

    The RSO loss is the DPO loss applied to RSO-sampled pairs:

        L_RSO = -E[log sigma(beta * ((log pi(y_w|x) - log pi_ref(y_w|x))
                                   - (log pi(y_l|x) - log pi_ref(y_l|x))))]

    When using pre-collected data (Section 4), each pair is re-weighted by
    how much better y_w is relative to y_l under the reward model:

        w(y_w, y_l) = exp(R(y_w) - R(y_l)) / Z

    with Z = mean(w) for numerical stability, giving:

        L_RSO_weighted = -E[w * log sigma(beta * diff)]

    Args:
        beta: KL regularisation temperature (beta in the paper). Default 0.1.
        use_importance_weights: If True apply per-pair importance weights
            (Section 4); if False reduces to standard DPO loss.
    """

    def __init__(
        self,
        beta: float = 0.1,
        use_importance_weights: bool = True,
    ) -> None:
        if beta <= 0:
            raise ValueError(f"beta must be positive; got {beta}")
        self.beta = beta
        self.use_importance_weights = use_importance_weights

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(
        self,
        log_probs_w: torch.Tensor,
        log_probs_l: torch.Tensor,
        ref_log_probs_w: torch.Tensor,
        ref_log_probs_l: torch.Tensor,
        reward_w: torch.Tensor,
        reward_l: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the RSO loss for a batch of preference pairs.

        Paper variable mapping:
            log_probs_w     <->  log pi(y_w | x)
            log_probs_l     <->  log pi(y_l | x)
            ref_log_probs_w <->  log pi_ref(y_w | x)
            ref_log_probs_l <->  log pi_ref(y_l | x)
            reward_w        <->  R(x, y_w)
            reward_l        <->  R(x, y_l)

        Args:
            log_probs_w:      Shape (B,) -- policy log-probs for winning responses.
            log_probs_l:      Shape (B,) -- policy log-probs for losing responses.
            ref_log_probs_w:  Shape (B,) -- reference log-probs for winning responses.
            ref_log_probs_l:  Shape (B,) -- reference log-probs for losing responses.
            reward_w:         Shape (B,) -- reward scores for winning responses.
            reward_l:         Shape (B,) -- reward scores for losing responses.

        Returns:
            (loss, metrics) where loss is a scalar tensor (requires_grad=True
            if inputs require grad) and metrics is a dict with:
                'reward_margin'           -- mean(R(y_w) - R(y_l))
                'importance_weight_mean'  -- mean of importance weights w
                'accuracy'               -- fraction where pi margin > 0
        """
        # --- validate shapes -----------------------------------------------
        B = log_probs_w.shape[0]
        shapes = {
            "log_probs_w": log_probs_w,
            "log_probs_l": log_probs_l,
            "ref_log_probs_w": ref_log_probs_w,
            "ref_log_probs_l": ref_log_probs_l,
            "reward_w": reward_w,
            "reward_l": reward_l,
        }
        for name, t in shapes.items():
            if t.ndim != 1 or t.shape[0] != B:
                raise ValueError(
                    f"{name} must have shape ({B},); got {t.shape}"
                )

        # --- DPO margin (beta * diff, paper eq. directly) ------------------
        # diff = (log pi(y_w) - log pi_ref(y_w)) - (log pi(y_l) - log pi_ref(y_l))
        diff = (log_probs_w - ref_log_probs_w) - (log_probs_l - ref_log_probs_l)
        margin = self.beta * diff  # shape (B,)

        # --- importance weights (Section 4) --------------------------------
        # w_i = exp(R(y_w,i) - R(y_l,i))
        # Normalise by mean to keep the scale of the loss comparable to DPO.
        reward_w_f = reward_w.float()
        reward_l_f = reward_l.float()
        raw_w = torch.exp(reward_w_f - reward_l_f)  # shape (B,)

        if self.use_importance_weights:
            # Z = mean(w) as a stable normaliser
            Z = raw_w.mean().clamp(min=1e-8)
            w = raw_w / Z  # shape (B,)
        else:
            w = torch.ones_like(raw_w)

        # --- loss ----------------------------------------------------------
        # L_RSO = -E[w * log sigma(beta * diff)]
        per_pair_loss = -F.logsigmoid(margin)  # shape (B,)
        loss = (w.detach() * per_pair_loss).mean()

        # --- metrics -------------------------------------------------------
        with torch.no_grad():
            reward_margin = (reward_w_f - reward_l_f).mean()
            iw_mean = raw_w.mean()
            # accuracy: fraction where the model correctly prefers y_w over y_l
            accuracy = (margin > 0).float().mean()

        metrics: Dict[str, float] = {
            "reward_margin": reward_margin.item(),
            "importance_weight_mean": iw_mean.item(),
            "accuracy": accuracy.item(),
        }

        return loss, metrics


# ---------------------------------------------------------------------------
# RSOConfig
# ---------------------------------------------------------------------------

@dataclass
class RSOConfig:
    """Configuration for RSOTrainer."""

    beta: float = 0.1
    use_importance_weights: bool = True
    seed: Optional[int] = 42


# ---------------------------------------------------------------------------
# RSOTrainer
# ---------------------------------------------------------------------------

class RSOTrainer:
    """Thin training wrapper around RSOLoss.

    Accepts pre-tokenised batches of preference pairs with reward scores and
    computes the RSO loss.  Gradient stepping is left to the caller.

    Expected batch keys:
        'log_probs_w'      -- Shape (B,) policy log-probs for winning responses.
        'log_probs_l'      -- Shape (B,) policy log-probs for losing responses.
        'ref_log_probs_w'  -- Shape (B,) reference log-probs for winning.
        'ref_log_probs_l'  -- Shape (B,) reference log-probs for losing.
        'reward_w'         -- Shape (B,) reward scores for winning responses.
        'reward_l'         -- Shape (B,) reward scores for losing responses.
    """

    _REQUIRED_KEYS = (
        "log_probs_w",
        "log_probs_l",
        "ref_log_probs_w",
        "ref_log_probs_l",
        "reward_w",
        "reward_l",
    )

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        config: Optional[RSOConfig] = None,
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.config = config or RSOConfig()
        self._loss_fn = RSOLoss(
            beta=self.config.beta,
            use_importance_weights=self.config.use_importance_weights,
        )
        self.sampler = RSOSampler(seed=self.config.seed)

        # Freeze reference model
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute the RSO loss for a pre-built batch.

        Args:
            batch: Dict with the required keys listed in the class docstring.

        Returns:
            Scalar loss tensor (with grad_fn if inputs have requires_grad).

        Raises:
            KeyError:   if a required key is missing from batch.
            ValueError: if tensors have incompatible shapes.
        """
        missing = [k for k in self._REQUIRED_KEYS if k not in batch]
        if missing:
            raise KeyError(
                f"Batch is missing required keys: {missing!r}. "
                f"Required: {self._REQUIRED_KEYS}"
            )

        loss, _ = self._loss_fn(
            log_probs_w=batch["log_probs_w"],
            log_probs_l=batch["log_probs_l"],
            ref_log_probs_w=batch["ref_log_probs_w"],
            ref_log_probs_l=batch["ref_log_probs_l"],
            reward_w=batch["reward_w"],
            reward_l=batch["reward_l"],
        )
        return loss
