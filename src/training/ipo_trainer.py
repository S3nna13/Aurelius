"""Identity Preference Optimization (IPO) trainer.

Implements IPO from Azar et al. 2024 -- "A General Theoretical Paradigm to
Understand Learning from Human Feedback" (arXiv:2306.00539).

DPO overfits by maximising the chosen/rejected log-prob gap without bound.
IPO replaces the log-sigmoid objective with a squared-loss regression toward a
regularisation target 1/(2*tau):

    h(x, y_w, y_l) = (log π_θ(y_w|x) - log π_ref(y_w|x))
                   - (log π_θ(y_l|x) - log π_ref(y_l|x))

    L_IPO = E[(h - 1/(2*tau))^2]

Unlike DPO, the gap is pulled toward a *finite* comfortable value rather than
driven to infinity, providing an implicit KL budget controlled by tau.

Pure native PyTorch only; no external dependencies beyond stdlib + torch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# IPOConfig
# ---------------------------------------------------------------------------

@dataclass
class IPOConfig:
    """Configuration for the IPO trainer.

    Attributes:
        tau:  Regularisation strength.  The optimisation target for the
              implicit reward gap is ``1 / (2 * tau)``.  Smaller tau →
              stronger regularisation (smaller target gap).
        eps:  Numerical epsilon used as a floor when summing mask weights to
              avoid division by zero for fully-masked sequences.
    """

    tau: float = 0.1
    eps: float = 1e-8


# ---------------------------------------------------------------------------
# IPOBatch
# ---------------------------------------------------------------------------

@dataclass
class IPOBatch:
    """Input batch for a single IPO loss computation.

    All log-prob tensors contain *per-token* log-probabilities under the
    respective policy.  Masks indicate which token positions are valid
    (1.0 = include, 0.0 = ignore).

    Shapes:
        chosen_log_probs      -- [B, T_w]  current policy, chosen response
        rejected_log_probs    -- [B, T_l]  current policy, rejected response
        chosen_ref_log_probs  -- [B, T_w]  reference policy, chosen response
        rejected_ref_log_probs-- [B, T_l]  reference policy, rejected response
        chosen_mask           -- [B, T_w]
        rejected_mask         -- [B, T_l]
    """

    chosen_log_probs: Tensor
    rejected_log_probs: Tensor
    chosen_ref_log_probs: Tensor
    rejected_ref_log_probs: Tensor
    chosen_mask: Tensor
    rejected_mask: Tensor


# ---------------------------------------------------------------------------
# IPOTrainer
# ---------------------------------------------------------------------------

class IPOTrainer:
    """Stateless IPO loss computation.

    All public methods are pure functions of their inputs and carry no mutable
    state beyond ``config``.  This makes the trainer easy to unit-test and
    trivial to wrap with any training loop.

    Args:
        config: :class:`IPOConfig` instance.
    """

    def __init__(self, config: IPOConfig | None = None) -> None:
        self.config: IPOConfig = config if config is not None else IPOConfig()

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def sequence_mean_log_prob(self, log_probs: Tensor, mask: Tensor) -> Tensor:
        """Compute the masked mean log-probability per sequence.

        Args:
            log_probs: Per-token log-probabilities, shape ``[B, T]``.
            mask:      Binary validity mask (1 = valid token), shape ``[B, T]``.

        Returns:
            Mean log-prob per sequence, shape ``[B]``.
            Fully-masked sequences receive 0.0 to avoid NaN.
        """
        # Clamp mask count to eps so we never divide by zero.
        denom = mask.sum(dim=-1).clamp(min=self.config.eps)   # [B]
        return (log_probs * mask).sum(dim=-1) / denom         # [B]

    def compute_h(self, batch: IPOBatch) -> Tensor:
        """Compute the implicit reward gap h for each (x, y_w, y_l) triple.

        h = (mean_lp_chosen  - mean_ref_lp_chosen)
          - (mean_lp_rejected - mean_ref_lp_rejected)

        Positive h means the current policy prefers y_w over y_l relative to
        the reference.

        Args:
            batch: :class:`IPOBatch` carrying policy and reference log-probs.

        Returns:
            Implicit reward gap per sample, shape ``[B]``.
        """
        mean_chosen = self.sequence_mean_log_prob(
            batch.chosen_log_probs, batch.chosen_mask
        )
        mean_ref_chosen = self.sequence_mean_log_prob(
            batch.chosen_ref_log_probs, batch.chosen_mask
        )
        mean_rejected = self.sequence_mean_log_prob(
            batch.rejected_log_probs, batch.rejected_mask
        )
        mean_ref_rejected = self.sequence_mean_log_prob(
            batch.rejected_ref_log_probs, batch.rejected_mask
        )

        h = (mean_chosen - mean_ref_chosen) - (mean_rejected - mean_ref_rejected)
        return h  # [B]

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def total_loss(self, batch: IPOBatch) -> dict[str, Tensor]:
        """Compute the IPO squared-loss objective and auxiliary statistics.

        The target gap is ``1 / (2 * tau)``.  The loss is the mean squared
        deviation of h from that target across the batch:

            L_IPO = mean((h - 1/(2*tau))^2)

        Args:
            batch: :class:`IPOBatch` carrying all required log-probs and masks.

        Returns:
            Dictionary with scalar tensors:

            * ``"loss"``            -- IPO loss (differentiable).
            * ``"h_mean"``          -- batch-mean of the implicit reward gap.
            * ``"target"``          -- regularisation target ``1/(2*tau)``.
            * ``"reward_accuracy"`` -- fraction of pairs where h > 0 (chosen
                                       preferred by current policy vs reference).
        """
        target = torch.tensor(
            1.0 / (2.0 * self.config.tau),
            dtype=batch.chosen_log_probs.dtype,
            device=batch.chosen_log_probs.device,
        )

        h = self.compute_h(batch)                              # [B]
        loss = ((h - target) ** 2).mean()                     # scalar

        h_mean = h.mean()
        reward_accuracy = (h > 0).float().mean()

        return {
            "loss": loss,
            "h_mean": h_mean,
            "target": target,
            "reward_accuracy": reward_accuracy,
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def statistics(self, batch: IPOBatch) -> dict[str, float]:
        """Compute diagnostic statistics without building a computation graph.

        Args:
            batch: :class:`IPOBatch`.

        Returns:
            Dictionary of plain Python floats:

            * ``"h_mean"``               -- mean implicit reward gap
            * ``"h_std"``                -- std of implicit reward gap
            * ``"reward_accuracy"``      -- fraction where h > 0
            * ``"chosen_logp_mean"``     -- mean of masked-mean chosen log-probs
            * ``"rejected_logp_mean"``   -- mean of masked-mean rejected log-probs
        """
        with torch.no_grad():
            h = self.compute_h(batch)

            chosen_logp = self.sequence_mean_log_prob(
                batch.chosen_log_probs, batch.chosen_mask
            )
            rejected_logp = self.sequence_mean_log_prob(
                batch.rejected_log_probs, batch.rejected_mask
            )

            return {
                "h_mean": h.mean().item(),
                "h_std": h.std().item(),
                "reward_accuracy": (h > 0).float().mean().item(),
                "chosen_logp_mean": chosen_logp.mean().item(),
                "rejected_logp_mean": rejected_logp.mean().item(),
            }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from src.training import TRAINING_REGISTRY  # noqa: E402

TRAINING_REGISTRY["ipo"] = IPOTrainer
