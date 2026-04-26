"""Aurelius -- Binary Classifier Optimization (BCO).

BCO (Jung et al., 2024) trains a binary classifier to distinguish preferred
from rejected responses, then uses the classifier's log-odds as a reward signal
for policy optimization.

Reference:
    "BCO: Binary Classifier Optimization for Offline Preference-Based
     Reinforcement Learning from Human Feedback"
    Jung et al., arXiv:2404.04656 (2024).

Paper notation used throughout:
    x          — prompt / context
    y_w        — preferred (chosen / "winner") response
    y_l        — dispreferred (rejected / "loser") response
    f_φ(x, y)  — binary classifier: P(y is preferred | x)
    β          — KL regularization coefficient
    π          — policy (current model)
    π_ref      — reference model (frozen)
    lp_w       — log π(y_w | x)
    lp_l       — log π(y_l | x)
    lp_ref_w   — log π_ref(y_w | x)
    lp_ref_l   — log π_ref(y_l | x)

**BCO-0 (Section 3.1) — offline, paired preference data:**

The implicit classifier is implemented via the log-ratio margin:

    diff = (lp_w - lp_ref_w) - (lp_l - lp_ref_l)

    f_φ(x, y_w) ≈ σ(diff / β)            [classifier score for winner]

    Reward: r(x, y) = log f_φ - log(1 - f_φ) = log-odds = diff / β

    Policy loss (KL-regularized RL, simplified):
        L_BCO = -2 · E[log σ(diff / β)]

    This equals 2× the standard DPO loss, sharing the same gradient direction
    but placing the probability mass differently from the binary-CE perspective.

**BCO+ (Section 3.3) — not implemented here:**
    BCO+ augments BCO-0 with an additional loss term on unpaired (prompt-only)
    data to stabilise training when the classifier is far from uniform. BCO+
    adds:
        L_unlab = BCE(f_φ(x, y_unlab), 0.5)  [pull classifier toward 0.5]
    Extend BCOLoss with an unpaired_loss() method to support BCO+.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# BCO Loss
# ---------------------------------------------------------------------------


class BCOLoss(nn.Module):
    """BCO-0 loss for offline paired preference data (Jung et al. 2024, §3.1).

    Trains an implicit binary classifier f_φ(x, y) ≈ σ(diff / β) where
        diff = (lp_w - lp_ref_w) - (lp_l - lp_ref_l)

    The BCO-0 policy loss is the sum of binary cross-entropy terms for both
    the preferred and dispreferred responses:

        L_BCO = BCE(σ(diff/β), 1) + BCE(1 - σ(diff/β), 1)
              = -log σ(diff/β) - log σ(diff/β)
              = -2 · log σ(diff/β)

    Args:
        beta: KL regularization coefficient β (default 0.1).
              If β ≤ 0, raises ValueError to prevent division by zero.
    """

    def __init__(self, beta: float = 0.1) -> None:
        super().__init__()
        if beta <= 0.0:
            raise ValueError(
                f"BCOLoss requires beta > 0 to avoid division by zero; got beta={beta}. "
                "The β parameter controls the log-odds scaling: diff/β. "
                "Use a small positive value (e.g., 1e-6) if you need near-zero β."
            )
        self.beta = beta

    def forward(
        self,
        lp_w: torch.Tensor,
        lp_l: torch.Tensor,
        lp_ref_w: torch.Tensor,
        lp_ref_l: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute BCO-0 loss from pre-computed log probabilities.

        Args:
            lp_w:      Shape (B,) — log π(y_w | x),     policy   log-probs for chosen.
            lp_l:      Shape (B,) — log π(y_l | x),     policy   log-probs for rejected.
            lp_ref_w:  Shape (B,) — log π_ref(y_w | x), reference log-probs for chosen.
            lp_ref_l:  Shape (B,) — log π_ref(y_l | x), reference log-probs for rejected.

        Returns:
            (loss, metrics) where loss is a scalar tensor and metrics is a dict with:
                'classifier_score' : mean σ(diff/β) ∈ (0, 1)
                'reward_margin'    : mean diff (raw log-ratio margin)
                'accuracy'         : fraction of examples where diff > 0 (correct ranking)
        """
        # Implicit reward margin: diff = Δ_w - Δ_l  (DPO-style log-ratio difference)
        diff = (lp_w - lp_ref_w) - (lp_l - lp_ref_l)  # (B,)

        # Implicit classifier score f_φ(x, y_w) = σ(diff / β)
        scaled = diff / self.beta  # (B,)
        classifier_score = torch.sigmoid(scaled)  # (B,) ∈ (0, 1)

        # BCO-0 loss = -2 · log σ(diff/β)  [sum of BCE for w and l]
        # Use F.logsigmoid for numerical stability (avoids log(0))
        loss = -2.0 * F.logsigmoid(scaled).mean()

        # ----------- metrics (detached) -----------
        with torch.no_grad():
            accuracy = (diff > 0.0).float().mean().item()
            reward_margin_val = diff.mean().item()
            classifier_score_val = classifier_score.mean().item()

        metrics: dict[str, float] = {
            "classifier_score": classifier_score_val,
            "reward_margin": reward_margin_val,
            "accuracy": accuracy,
        }

        return loss, metrics


# ---------------------------------------------------------------------------
# BCO Trainer
# ---------------------------------------------------------------------------


class BCOTrainer:
    """Trainer wrapper for BCO-0 preference optimization.

    Holds a BCOLoss instance and provides a compute_loss interface that
    accepts a batch dict with pre-computed log-probability tensors.

    Expected batch keys:
        'lp_w'      : (B,) policy log-probs for chosen responses
        'lp_l'      : (B,) policy log-probs for rejected responses
        'lp_ref_w'  : (B,) reference log-probs for chosen responses
        'lp_ref_l'  : (B,) reference log-probs for rejected responses

    Args:
        beta: KL regularization coefficient β (default 0.1).
    """

    def __init__(self, beta: float = 0.1) -> None:
        self.loss_fn = BCOLoss(beta=beta)

    @property
    def beta(self) -> float:
        return self.loss_fn.beta

    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute BCO-0 loss from a batch dict.

        Args:
            batch: Dict with keys 'lp_w', 'lp_l', 'lp_ref_w', 'lp_ref_l'.

        Returns:
            (loss, metrics) — same as BCOLoss.forward().

        Raises:
            KeyError: if any required key is missing from the batch.
        """
        required = ("lp_w", "lp_l", "lp_ref_w", "lp_ref_l")
        for key in required:
            if key not in batch:
                raise KeyError(
                    f"BCOTrainer.compute_loss: missing required batch key {key!r}. "
                    f"Expected keys: {required}"
                )

        return self.loss_fn(
            lp_w=batch["lp_w"],
            lp_l=batch["lp_l"],
            lp_ref_w=batch["lp_ref_w"],
            lp_ref_l=batch["lp_ref_l"],
        )
