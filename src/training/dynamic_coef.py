"""Dynamic coefficient scheduling for multi-task / multi-objective training.

Implements three strategies:
  - GradNorm (Chen et al., 2018): normalises gradients across tasks.
  - UncertaintyWeighting (Kendall et al., 2018): homoscedastic uncertainty.
  - LossRatioScheduler: drives loss ratios toward user-specified targets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GradNorm Scheduler
# ---------------------------------------------------------------------------


class GradNormScheduler(nn.Module):
    """GradNorm: gradient normalisation for balanced multi-task training.

    Reference: Chen et al., "GradNorm: Gradient Normalization for Adaptive
    Loss Balancing in Deep Multitask Networks", ICML 2018.

    Args:
        n_tasks: Number of tasks.
        alpha:   Restoring-force strength (hyperparameter).
        lr:      Learning rate for the weight parameters.
    """

    def __init__(self, n_tasks: int, alpha: float = 1.5, lr: float = 0.01) -> None:
        super().__init__()
        if n_tasks < 1:
            raise ValueError("n_tasks must be >= 1")
        self.n_tasks = n_tasks
        self.alpha = alpha
        self.lr = lr
        # Learnable log-weights; softmax of these become the task weights.
        self.log_weights: nn.Parameter = nn.Parameter(torch.zeros(n_tasks))
        # Cached initial losses for computing loss ratios.
        self._initial_losses: Tensor | None = None

    # ------------------------------------------------------------------
    def update(
        self,
        task_losses: list[Tensor],
        shared_params: list[Tensor],
    ) -> Tensor:
        """Compute the GradNorm auxiliary loss and update internal state.

        This loss should be back-propagated so that ``self.log_weights``
        receives gradients and can be updated by an external optimiser OR
        you can call ``.backward()`` on the returned tensor directly and
        step a separate parameter-group optimiser.

        Args:
            task_losses:   Per-task scalar loss tensors (length n_tasks).
            shared_params: Parameters of the shared backbone (used to
                           compute per-task gradient norms).

        Returns:
            GradNorm loss (scalar tensor) for back-propagation.
        """
        if len(task_losses) != self.n_tasks:
            raise ValueError(f"Expected {self.n_tasks} task losses, got {len(task_losses)}")

        weights = self.get_weights()  # (n_tasks,)

        # Initialise reference losses on first call.
        with torch.no_grad():
            current_losses = torch.stack([line.detach() for line in task_losses])
        if self._initial_losses is None:
            self._initial_losses = current_losses.clone()

        # Compute per-task gradient norms w.r.t. the last shared parameter.
        # We use the last element of shared_params as the representative layer.
        ref_param = shared_params[-1]
        grad_norms: list[Tensor] = []
        for i, (w, loss) in enumerate(zip(weights, task_losses)):
            weighted_loss = w * loss
            grads = torch.autograd.grad(
                weighted_loss,
                ref_param,
                retain_graph=True,
                create_graph=True,
                allow_unused=True,
            )
            if grads[0] is None:
                # Fallback: use weight magnitude as a proxy.
                grad_norms.append(w.abs())
            else:
                grad_norms.append(grads[0].norm(2))

        grad_norms_t = torch.stack(grad_norms)  # (n_tasks,)

        # Mean gradient norm (detached) = target for every task.
        mean_norm = grad_norms_t.detach().mean()

        # Relative inverse training rates.
        loss_ratios = current_losses / (self._initial_losses + 1e-8)
        mean_ratio = loss_ratios.mean()
        relative_rates = loss_ratios / (mean_ratio + 1e-8)

        # Target norms.
        target_norms = (mean_norm * relative_rates.pow(self.alpha)).detach()

        # GradNorm loss: L1 distance between actual and target norms.
        gradnorm_loss = F.l1_loss(grad_norms_t, target_norms)

        return gradnorm_loss

    # ------------------------------------------------------------------
    def get_weights(self) -> Tensor:
        """Return softmax-normalised task weights summing to 1."""
        return F.softmax(self.log_weights, dim=0)


# ---------------------------------------------------------------------------
# Uncertainty Weighting
# ---------------------------------------------------------------------------


class UncertaintyWeighting(nn.Module):
    """Homoscedastic uncertainty weighting (Kendall et al., 2018).

    Each task has a learnable log-variance ``log(sigma_i^2)``.  The combined
    loss is:

        L = sum_i  [ L_i / (2 * sigma_i^2) + log(sigma_i) ]

    Args:
        n_tasks: Number of tasks.
    """

    def __init__(self, n_tasks: int) -> None:
        super().__init__()
        if n_tasks < 1:
            raise ValueError("n_tasks must be >= 1")
        self.n_tasks = n_tasks
        # log_var[i] = log(sigma_i^2)
        self.log_var: nn.Parameter = nn.Parameter(torch.zeros(n_tasks))

    # ------------------------------------------------------------------
    def compute_weighted_loss(
        self,
        task_losses: list[Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Compute uncertainty-weighted combined loss.

        Args:
            task_losses: Per-task scalar loss tensors (length n_tasks).

        Returns:
            (total_loss, per_task_weights) where per_task_weights[i] =
            1 / (2 * sigma_i^2).
        """
        if len(task_losses) != self.n_tasks:
            raise ValueError(f"Expected {self.n_tasks} task losses, got {len(task_losses)}")

        # sigma_i^2 = exp(log_var[i])
        # weight_i  = 1 / (2 * sigma_i^2) = 0.5 * exp(-log_var[i])
        precision = 0.5 * torch.exp(-self.log_var)  # (n_tasks,)

        losses_t = torch.stack(task_losses)  # (n_tasks,)
        # Regularisation term: log(sigma_i) = 0.5 * log_var[i]
        reg = 0.5 * self.log_var

        total_loss: Tensor = (precision * losses_t + reg).sum()
        return total_loss, precision

    # ------------------------------------------------------------------
    def get_log_variances(self) -> Tensor:
        """Return per-task log variances, shape (n_tasks,)."""
        return self.log_var


# ---------------------------------------------------------------------------
# Loss Ratio Scheduler
# ---------------------------------------------------------------------------


class LossRatioScheduler:
    """Adjusts per-task coefficients to drive loss ratios toward targets.

    At each update step the coefficients are nudged proportionally to the
    discrepancy between the current loss ratio and the desired ratio.

    Args:
        target_ratios:    Mapping of task name -> desired loss fraction.
        adjustment_rate:  Step size for coefficient updates.
    """

    def __init__(
        self,
        target_ratios: dict[str, float],
        adjustment_rate: float = 0.01,
    ) -> None:
        if not target_ratios:
            raise ValueError("target_ratios must not be empty")
        self.target_ratios = dict(target_ratios)
        self.adjustment_rate = adjustment_rate
        # Initialise all coefficients to 1.0.
        self._coefficients: dict[str, float] = {k: 1.0 for k in target_ratios}

    # ------------------------------------------------------------------
    def update(self, current_losses: dict[str, float]) -> dict[str, float]:
        """Adjust coefficients and return updated values.

        The update rule for task i is:
            coef_i += adjustment_rate * (target_ratio_i - current_ratio_i)
            coef_i  = max(coef_i, 1e-6)   # keep positive

        Args:
            current_losses: Mapping of task name -> current loss value.

        Returns:
            Updated coefficient dictionary.
        """
        total_loss = sum(current_losses.values()) + 1e-8
        for name in self._coefficients:
            if name not in current_losses:
                continue
            current_ratio = current_losses[name] / total_loss
            target_ratio = self.target_ratios.get(name, 0.0)
            delta = self.adjustment_rate * (target_ratio - current_ratio)
            self._coefficients[name] = max(self._coefficients[name] + delta, 1e-6)
        return dict(self._coefficients)

    # ------------------------------------------------------------------
    def get_coefficients(self) -> dict[str, float]:
        """Return current per-task coefficients."""
        return dict(self._coefficients)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DynamicCoefConfig:
    """Configuration for dynamic coefficient scheduling."""

    method: str = "uncertainty"  # "gradnorm" | "uncertainty" | "loss_ratio"
    n_tasks: int = 2
    alpha: float = 1.5  # GradNorm restoring-force strength
    lr: float = 0.01  # GradNorm weight-parameter learning rate
    adjustment_rate: float = 0.01  # LossRatioScheduler step size
    normalize: bool = True  # Whether to normalise coefficients


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_dynamic_coef_scheduler(
    config: DynamicCoefConfig,
    task_names: list[str],
) -> GradNormScheduler | UncertaintyWeighting | LossRatioScheduler:
    """Instantiate a dynamic coefficient scheduler from a config.

    Args:
        config:     Scheduler configuration.
        task_names: Ordered list of task names (length must match config.n_tasks
                    for 'loss_ratio'; used for documentation otherwise).

    Returns:
        The appropriate scheduler object.

    Raises:
        ValueError: If config.method is not recognised.
    """
    method = config.method.lower()
    if method == "gradnorm":
        return GradNormScheduler(
            n_tasks=config.n_tasks,
            alpha=config.alpha,
            lr=config.lr,
        )
    elif method == "uncertainty":
        return UncertaintyWeighting(n_tasks=config.n_tasks)
    elif method == "loss_ratio":
        if len(task_names) != config.n_tasks:
            logger.warning(
                "task_names length (%d) != config.n_tasks (%d); using task_names.",
                len(task_names),
                config.n_tasks,
            )
        # Equal target ratios by default.
        n = len(task_names)
        target_ratios = {name: 1.0 / n for name in task_names}
        return LossRatioScheduler(
            target_ratios=target_ratios,
            adjustment_rate=config.adjustment_rate,
        )
    else:
        raise ValueError(
            f"Unknown dynamic coefficient method: '{config.method}'. "
            "Choose from 'gradnorm', 'uncertainty', or 'loss_ratio'."
        )
