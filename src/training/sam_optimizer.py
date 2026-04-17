"""
SAM (Sharpness-Aware Minimization) optimizer and variants.

Implements:
  - SAMOptimizer  : Foret et al. 2021
  - ASAM          : Kwon et al. 2021 (adaptive per-weight scale)
  - LookSAM       : LookAhead + SAM combination
  - SharpnessAnalyzer : flatness_ratio and gradient_diversity metrics
  - SAMTrainingLoop   : convenience wrapper for two-closure SAM training

Pure stdlib + torch only.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# SAMOptimizer
# ---------------------------------------------------------------------------

class SAMOptimizer:
    """Sharpness-Aware Minimization (Foret et al. 2021).

    Usage::

        base_opt = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer = SAMOptimizer(base_opt, rho=0.05)

        # First closure: compute gradients at current params
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.first_step(zero_grad=True)

        # Second closure: compute gradients at perturbed params
        criterion(model(x), y).backward()
        optimizer.second_step(zero_grad=True)
    """

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        rho: float = 0.05,
        adaptive: bool = False,
    ) -> None:
        if rho <= 0:
            raise ValueError(f"rho must be positive, got {rho}")
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.adaptive = adaptive
        # Storage for original parameters; keyed by param id
        self._saved_params: dict[int, Tensor] = {}

    # ------------------------------------------------------------------
    # Properties delegating to base_optimizer
    # ------------------------------------------------------------------

    @property
    def state(self) -> dict:
        return self.base_optimizer.state

    @property
    def param_groups(self) -> list:
        return self.base_optimizer.param_groups

    # ------------------------------------------------------------------
    # SAM steps
    # ------------------------------------------------------------------

    def first_step(self, zero_grad: bool = False) -> None:
        """Perturb parameters toward the sharpest ascent direction.

        After this call, model parameters are at w + e_w.  The original
        parameters are saved in ``self._saved_params``.

        Args:
            zero_grad: If True, zero gradients after computing the
                perturbation (so they are fresh for the second forward).
        """
        grad_norm = self._grad_norm()
        if grad_norm == 0.0:
            # Nothing to perturb; avoid division by zero
            grad_norm = 1.0

        self._saved_params.clear()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self._saved_params[id(p)] = p.data.clone()
                if self.adaptive:
                    # Adaptive: scale perturbation by |w|
                    e_w = (torch.pow(p, 2) * p.grad) * scale
                else:
                    e_w = p.grad * scale
                p.data.add_(e_w)

        if zero_grad:
            self.base_optimizer.zero_grad()

    def second_step(self, zero_grad: bool = False) -> None:
        """Restore original parameters and apply base optimizer step.

        Must be called after :meth:`first_step` with gradients computed
        at the perturbed point.

        Args:
            zero_grad: If True, zero gradients after the optimizer step.
        """
        # Restore original params
        for group in self.param_groups:
            for p in group["params"]:
                if id(p) in self._saved_params:
                    p.data.copy_(self._saved_params[id(p)])

        self.base_optimizer.step()

        if zero_grad:
            self.base_optimizer.zero_grad()

        self._saved_params.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _grad_norm(self) -> float:
        """Compute the global L2 norm of current gradients."""
        total = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    total += p.grad.data.norm(2).item() ** 2
        return math.sqrt(total)

    def zero_grad(self) -> None:
        self.base_optimizer.zero_grad()

    def state_dict(self) -> dict:
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.base_optimizer.load_state_dict(state_dict)


# ---------------------------------------------------------------------------
# ASAM
# ---------------------------------------------------------------------------

class ASAM:
    """Adaptive SAM (Kwon et al. 2021).

    Perturbation is scaled by weight magnitude for each parameter:

        e_w = rho * |w| * grad / ||w * grad||

    This makes the perturbation invariant to weight scale.
    """

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        rho: float = 0.05,
        eta: float = 0.01,
    ) -> None:
        if rho <= 0:
            raise ValueError(f"rho must be positive, got {rho}")
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.eta = eta
        self._saved_params: dict[int, Tensor] = {}

    @property
    def state(self) -> dict:
        return self.base_optimizer.state

    @property
    def param_groups(self) -> list:
        return self.base_optimizer.param_groups

    def first_step(self, zero_grad: bool = False) -> None:
        """Compute adaptive perturbation and move params to w + e_w."""
        # Compute ||T(w) * grad|| where T(w) = |w| + eta
        t_norm_sq = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                t_w = p.data.abs() + self.eta
                t_norm_sq += (t_w * p.grad).norm(2).item() ** 2
        t_norm = math.sqrt(t_norm_sq) + 1e-12

        self._saved_params.clear()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                self._saved_params[id(p)] = p.data.clone()
                t_w = p.data.abs() + self.eta
                # e_w = rho * t_w^2 * grad / t_norm
                e_w = (self.rho / t_norm) * t_w * t_w * p.grad
                p.data.add_(e_w)

        if zero_grad:
            self.base_optimizer.zero_grad()

    def second_step(self, zero_grad: bool = False) -> None:
        """Restore original params and apply base optimizer step."""
        for group in self.param_groups:
            for p in group["params"]:
                if id(p) in self._saved_params:
                    p.data.copy_(self._saved_params[id(p)])

        self.base_optimizer.step()

        if zero_grad:
            self.base_optimizer.zero_grad()

        self._saved_params.clear()

    def zero_grad(self) -> None:
        self.base_optimizer.zero_grad()

    def state_dict(self) -> dict:
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.base_optimizer.load_state_dict(state_dict)


# ---------------------------------------------------------------------------
# LookSAM
# ---------------------------------------------------------------------------

class LookSAM:
    """LookAhead + SAM combination.

    Maintains a set of "slow" weights.  Every ``k`` SAM inner steps,
    the slow weights are blended toward the fast weights:

        slow_w = alpha * fast_w + (1 - alpha) * slow_w

    then the fast weights are reset to the slow weights.
    """

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        rho: float = 0.05,
        alpha: float = 0.5,
        k: int = 5,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.alpha = alpha
        self.k = k
        self._step_count = 0
        self._sam = SAMOptimizer(base_optimizer, rho=rho)

        # Initialize slow weights as copies of the current fast weights
        self._slow_weights: list[Tensor] = []
        for group in self.param_groups:
            for p in group["params"]:
                self._slow_weights.append(p.data.clone())

    @property
    def state(self) -> dict:
        return self.base_optimizer.state

    @property
    def param_groups(self) -> list:
        return self.base_optimizer.param_groups

    # ------------------------------------------------------------------

    def step(self, closure: Callable[[], Tensor]) -> Tensor:
        """Perform one LookSAM step.

        Args:
            closure: A closure that recomputes the model forward pass and
                returns the loss.  The closure is called twice internally:
                once for the SAM first step, and once for the SAM second
                step.

        Returns:
            The loss value from the first closure evaluation.
        """
        # --- SAM first step ---
        loss = closure()
        self._sam.first_step(zero_grad=True)

        # --- SAM second step ---
        closure()
        self._sam.second_step(zero_grad=True)

        self._step_count += 1

        # --- LookAhead slow weight update every k steps ---
        if self._step_count % self.k == 0:
            idx = 0
            for group in self.param_groups:
                for p in group["params"]:
                    slow = self._slow_weights[idx]
                    slow.mul_(1.0 - self.alpha).add_(p.data * self.alpha)
                    self._slow_weights[idx] = slow
                    idx += 1
            self.sync_slow_weights()

        return loss

    def sync_slow_weights(self) -> None:
        """Copy slow weights into fast (model) weights."""
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                p.data.copy_(self._slow_weights[idx])
                idx += 1

    def zero_grad(self) -> None:
        self.base_optimizer.zero_grad()

    def state_dict(self) -> dict:
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.base_optimizer.load_state_dict(state_dict)


# ---------------------------------------------------------------------------
# SharpnessAnalyzer
# ---------------------------------------------------------------------------

class SharpnessAnalyzer:
    """Measure loss-landscape sharpness around the current weights."""

    def __init__(self, model: nn.Module, loss_fn: Callable) -> None:
        self.model = model
        self.loss_fn = loss_fn

    # ------------------------------------------------------------------

    def flatness_ratio(
        self,
        input_ids: Tensor,
        labels: Tensor,
        rho: float = 0.05,
        n_directions: int = 10,
    ) -> float:
        """Average ratio of perturbed-point loss to original loss.

        flatness = mean(loss_perturbed) / loss_original

        Values > 1 indicate a sharp region; values near 1 indicate flat.

        Args:
            input_ids: Model input tensor.
            labels:    Target tensor.
            rho:       Perturbation ball radius.
            n_directions: Number of random unit-norm directions to sample.

        Returns:
            flatness_ratio as a Python float.
        """
        self.model.eval()
        with torch.no_grad():
            loss_orig = self.loss_fn(self.model(input_ids), labels).item()

        if loss_orig == 0.0:
            loss_orig = 1e-8  # avoid division by zero

        # Collect all parameters as a flat list for easy manipulation
        params = [p for p in self.model.parameters()]

        perturbed_losses: list[float] = []
        for _ in range(n_directions):
            # Generate a random perturbation with global norm = rho
            noise_parts: list[Tensor] = []
            for p in params:
                noise_parts.append(torch.randn_like(p))
            # Normalise to rho
            global_norm = math.sqrt(sum(n.norm().item() ** 2 for n in noise_parts))
            global_norm = max(global_norm, 1e-12)
            scale = rho / global_norm

            # Apply perturbation
            for p, n in zip(params, noise_parts):
                p.data.add_(n * scale)

            # Evaluate
            self.model.eval()
            with torch.no_grad():
                loss_p = self.loss_fn(self.model(input_ids), labels).item()
            perturbed_losses.append(loss_p)

            # Restore
            for p, n in zip(params, noise_parts):
                p.data.sub_(n * scale)

        return sum(perturbed_losses) / (len(perturbed_losses) * loss_orig)

    # ------------------------------------------------------------------

    def gradient_diversity(self, input_ids: Tensor, labels: Tensor) -> float:
        """Relative gradient norm: ||grad|| / sqrt(n_params).

        Measures how large the gradient is relative to parameter count.
        Always >= 0 and finite for well-behaved models.

        Args:
            input_ids: Model input tensor.
            labels:    Target tensor.

        Returns:
            gradient_diversity as a Python float.
        """
        self.model.zero_grad()
        loss = self.loss_fn(self.model(input_ids), labels)
        loss.backward()

        total_sq = 0.0
        n_params = 0
        for p in self.model.parameters():
            if p.grad is not None:
                total_sq += p.grad.data.norm(2).item() ** 2
                n_params += p.numel()

        self.model.zero_grad()

        if n_params == 0:
            return 0.0
        return math.sqrt(total_sq) / math.sqrt(n_params)


# ---------------------------------------------------------------------------
# SAMTrainingLoop
# ---------------------------------------------------------------------------

class SAMTrainingLoop:
    """Convenience wrapper that handles the two-closure SAM training pattern.

    Example::

        base_opt = torch.optim.SGD(model.parameters(), lr=0.01)
        sam = SAMOptimizer(base_opt, rho=0.05)
        loop = SAMTrainingLoop(model, sam, nn.CrossEntropyLoss())
        info = loop.train_step(input_ids, labels)
        # info = {"loss": float, "perturbation_norm": float}
    """

    def __init__(
        self,
        model: nn.Module,
        sam_optimizer: SAMOptimizer,
        loss_fn: Callable,
    ) -> None:
        self.model = model
        self.sam_optimizer = sam_optimizer
        self.loss_fn = loss_fn

    # ------------------------------------------------------------------

    def train_step(self, input_ids: Tensor, labels: Tensor) -> dict:
        """Perform one SAM training step.

        1. Forward + backward at current params -> first_step (perturb).
        2. Forward + backward at perturbed params -> second_step (restore + update).

        Returns:
            A dict with keys:
              - ``loss``: scalar loss at the original (first) point.
              - ``perturbation_norm``: L2 norm of the weight perturbation e_w.
        """
        # ---- First pass: gradients at current w ----
        self.sam_optimizer.zero_grad()
        self.model.train()
        output = self.model(input_ids)
        loss = self.loss_fn(output, labels)
        loss.backward()
        loss_val = loss.item()

        # Snapshot params before first_step so we can measure perturbation norm
        pre_params: dict[int, Tensor] = {}
        for group in self.sam_optimizer.param_groups:
            for p in group["params"]:
                pre_params[id(p)] = p.data.clone()

        self.sam_optimizer.first_step(zero_grad=True)

        # Measure perturbation norm ||e_w||
        pert_sq = 0.0
        for group in self.sam_optimizer.param_groups:
            for p in group["params"]:
                if id(p) in pre_params:
                    diff = p.data - pre_params[id(p)]
                    pert_sq += diff.norm(2).item() ** 2
        perturbation_norm = math.sqrt(pert_sq)

        # ---- Second pass: gradients at perturbed w ----
        self.model.train()
        output2 = self.model(input_ids)
        loss2 = self.loss_fn(output2, labels)
        loss2.backward()

        self.sam_optimizer.second_step(zero_grad=True)

        return {"loss": loss_val, "perturbation_norm": perturbation_norm}
