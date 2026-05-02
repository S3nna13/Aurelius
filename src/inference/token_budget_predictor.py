"""Token Budget Predictor for adaptive computation.

Predicts the required token budget (response length) from a query's
hidden representation before generation starts, enabling early stopping
and compute-efficient inference.

Inspired by: Muennighoff et al. 2025 (token budget forcing)
"""

from __future__ import annotations

from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# BudgetCategory
# ---------------------------------------------------------------------------


class BudgetCategory(Enum):
    """Coarse-grained response-length category."""

    SHORT = 0  # < 50 tokens
    MEDIUM = 1  # 50–200 tokens
    LONG = 2  # > 200 tokens

    @classmethod
    def from_length(cls, length: int) -> BudgetCategory:
        """Map an integer token count to a BudgetCategory.

        Args:
            length: Response length in tokens.

        Returns:
            Corresponding :class:`BudgetCategory`.
        """
        if length < 50:
            return cls.SHORT
        elif length <= 200:
            return cls.MEDIUM
        else:
            return cls.LONG


# ---------------------------------------------------------------------------
# TokenBudgetPredictor
# ---------------------------------------------------------------------------


class TokenBudgetPredictor(nn.Module):
    """Predicts response length (token budget) from a query hidden state.

    Architecture:
        Regression head: Linear(d_model, hidden_size) → ReLU →
                         Linear(hidden_size, 1) → Softplus → clamp
        Classification head: Linear(d_model, 3)

    Args:
        d_model: Dimensionality of the query hidden state.
        max_budget: Maximum token count to clamp predictions to.
        hidden_size: Width of the regression head's hidden layer.
    """

    def __init__(
        self,
        d_model: int,
        max_budget: int = 512,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()
        self.max_budget = max_budget

        # Regression head
        self.reg_fc1 = nn.Linear(d_model, hidden_size)
        self.reg_fc2 = nn.Linear(hidden_size, 1)

        # Classification head (3 budget categories)
        self.cls_head = nn.Linear(d_model, 3)

    def predict_budget(self, query_hidden: Tensor) -> Tensor:
        """Predict a continuous token budget from query hidden states.

        Args:
            query_hidden: ``(B, d_model)`` — last hidden state of the prompt.

        Returns:
            ``(B,)`` float tensor of predicted token counts clamped to
            ``[1, max_budget]``.
        """
        x = F.relu(self.reg_fc1(query_hidden))  # (B, hidden_size)
        x = self.reg_fc2(x)  # (B, 1)
        x = F.softplus(x).squeeze(-1)  # (B,)
        return x.clamp(1.0, float(self.max_budget))

    def predict_category(self, query_hidden: Tensor) -> Tensor:
        """Predict a coarse BudgetCategory index from query hidden states.

        Args:
            query_hidden: ``(B, d_model)`` — last hidden state of the prompt.

        Returns:
            ``(B,)`` long tensor of category indices (0 = SHORT, 1 = MEDIUM,
            2 = LONG).
        """
        logits = self.cls_head(query_hidden)  # (B, 3)
        return logits.argmax(dim=-1)  # (B,) long

    def forward(self, query_hidden: Tensor) -> tuple[Tensor, Tensor]:
        """Convenience forward returning (budget, category).

        Args:
            query_hidden: ``(B, d_model)``.

        Returns:
            Tuple of ``(predicted_budget, predicted_category)``.
        """
        return self.predict_budget(query_hidden), self.predict_category(query_hidden)


# ---------------------------------------------------------------------------
# BudgetLoss
# ---------------------------------------------------------------------------


class BudgetLoss(nn.Module):
    """Combined regression + classification loss for training the predictor.

    Args:
        reg_weight: Weight applied to the regression (smooth-L1) loss.
        cls_weight: Weight applied to the classification (cross-entropy) loss.
    """

    def __init__(self, reg_weight: float = 1.0, cls_weight: float = 0.1) -> None:
        super().__init__()
        self.reg_weight = reg_weight
        self.cls_weight = cls_weight

    def forward(
        self,
        query_hidden: Tensor,
        predictor: TokenBudgetPredictor,
        target_lengths: Tensor,
    ) -> dict[str, Tensor]:
        """Compute the combined budget prediction loss.

        Args:
            query_hidden: ``(B, d_model)`` query hidden states.
            predictor: The :class:`TokenBudgetPredictor` to evaluate.
            target_lengths: ``(B,)`` int tensor of ground-truth response
                lengths.

        Returns:
            Dict with keys ``'total'``, ``'reg_loss'``, and ``'cls_loss'``.
        """
        # --- Regression ---
        predicted_budget = predictor.predict_budget(query_hidden)  # (B,)
        reg_loss = F.smooth_l1_loss(predicted_budget, target_lengths.float())

        # --- Classification ---
        logits = predictor.cls_head(query_hidden)  # (B, 3)
        cls_target = torch.tensor(
            [BudgetCategory.from_length(int(line.item())).value for line in target_lengths],
            dtype=torch.long,
            device=query_hidden.device,
        )
        cls_loss = F.cross_entropy(logits, cls_target)

        total = self.reg_weight * reg_loss + self.cls_weight * cls_loss

        return {
            "total": total,
            "reg_loss": reg_loss,
            "cls_loss": cls_loss,
        }


# ---------------------------------------------------------------------------
# AdaptiveStopCriteria
# ---------------------------------------------------------------------------


class AdaptiveStopCriteria:
    """Decides whether generation should stop based on a predicted budget.

    Args:
        predictor: A trained :class:`TokenBudgetPredictor`.
        slack_factor: Multiply the predicted budget by this value to obtain
            the hard-stop threshold (provides a small margin).
    """

    def __init__(
        self,
        predictor: TokenBudgetPredictor,
        slack_factor: float = 1.2,
    ) -> None:
        self.predictor = predictor
        self.slack_factor = slack_factor

    def _hard_stop(self, query_hidden: Tensor) -> float:
        """Compute the hard-stop threshold for a single query."""
        budget = self.predictor.predict_budget(query_hidden)[0].item()
        return self.slack_factor * budget

    def should_stop(self, query_hidden: Tensor, n_generated: int) -> bool:
        """Return True when the generator should halt.

        Args:
            query_hidden: ``(1, d_model)`` or ``(B, d_model)`` hidden state;
                only the first element is used.
            n_generated: Number of tokens generated so far.

        Returns:
            ``True`` if ``n_generated >= slack_factor * predicted_budget``.
        """
        return n_generated >= self._hard_stop(query_hidden)

    def remaining_budget(self, query_hidden: Tensor, n_generated: int) -> int:
        """Estimate how many tokens are still available.

        Args:
            query_hidden: ``(1, d_model)`` or ``(B, d_model)`` hidden state.
            n_generated: Number of tokens generated so far.

        Returns:
            Non-negative integer of remaining tokens.
        """
        return max(0, int(self._hard_stop(query_hidden)) - n_generated)


# ---------------------------------------------------------------------------
# BudgetCalibrator
# ---------------------------------------------------------------------------


class BudgetCalibrator:
    """Post-hoc linear calibration: ``calibrated = a * predicted + b``.

    Fits a least-squares correction on a held-out calibration set.

    Attributes:
        a: Learned scale factor (initialised to 1.0).
        b: Learned bias (initialised to 0.0).
    """

    def __init__(self) -> None:
        self.a: float = 1.0
        self.b: float = 0.0

    def fit(self, predicted: Tensor, actual: Tensor) -> None:
        """Fit the calibration parameters via closed-form least-squares.

        Solves ``min ||[predicted | 1] @ [a, b]^T - actual||^2``.

        Args:
            predicted: ``(N,)`` float tensor of raw model predictions.
            actual: ``(N,)`` float tensor of ground-truth lengths.
        """
        predicted = predicted.float().reshape(-1, 1)
        ones = torch.ones_like(predicted)
        A = torch.cat([predicted, ones], dim=1)  # (N, 2)

        result = torch.linalg.lstsq(A, actual.float().reshape(-1))
        coeffs = result.solution  # (2,)

        self.a = float(coeffs[0].item())
        self.b = float(coeffs[1].item())

    def calibrate(self, predicted: Tensor) -> Tensor:
        """Apply the learned linear correction.

        Args:
            predicted: ``(N,)`` float tensor of raw model predictions.

        Returns:
            ``(N,)`` float tensor of calibrated predictions.
        """
        return self.a * predicted.float() + self.b
